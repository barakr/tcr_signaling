"""Tests for mixed-affinity pMHC species (competitive-selectivity feature).

Background. KS is a static Boltzmann/MC energy model with no k_on/k_off -- the
only affinity-like lever is `u_assoc` (well depth, kT), historically a single
scalar shared by every pMHC in a run. This module adds a discrete species
mixture: a small `pmhc_species` table (fraction + u_assoc + optional
sigma_bind/h0_tcr/sigma_r per species), each pMHC assigned one species at
init weighted by `fraction`, with per-species bound-fraction as the new
output -- the actual competitive-selectivity readout (does a TCR
preferentially engage the higher-affinity species when several compete for
the same patch).

Only reachable via `--params <file>` (no CLI-flag precedent exists anywhere
in this codebase for a list-valued flag; --pmhc_deposition's numeric-alias
trick doesn't apply to a table of objects). `n_species <= 1` (the field
absent from the params file) must stay the exact legacy single-affinity
path -- see tests/reference_values.json protection (KS rule 3) and
TestLegacyPathUnaffected below.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from models.kinetic_segregation import ks_build

_PKG = Path(__file__).resolve().parents[1]
_BINARY = ks_build.find_binary() or (_PKG / ks_build.binary_name())

_COMMON = [
    "--time_sec",
    "0.3",
    "--rigidity_kT",
    "5",
    "--grid_size",
    "32",
    "--n_tcr",
    "60",
    "--n_cd45",
    "150",
    "--n_pmhc",
    "60",
    "--pmhc_mode",
    "inner_circle",
    "--pmhc_radius",
    "62.5",
    "--patch_size",
    "500",
    "--no-gpu",
]

_TWO_SPECIES = {
    "pmhc_species": [
        {"fraction": 0.5, "u_assoc": 40.0, "sigma_bind": 3.0},
        {"fraction": 0.5, "u_assoc": 5.0, "sigma_bind": 3.0},
    ]
}


def _write_params(tmp_path, payload, name="species.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _run(tmp_path, *, extra_args=(), params=None, label="run", expect_ok=True):
    if not _BINARY.exists():
        try:
            ks_build.build()
        except RuntimeError as exc:
            pytest.skip(f"Failed to build binary: {exc}")
    if not _BINARY.exists():
        pytest.skip("ks_gpu not available")
    cmd = [str(_BINARY), "--run-dir", str(tmp_path / label), *_COMMON, *extra_args]
    if params is not None:
        params_path = _write_params(tmp_path, params, name=f"{label}.json")
        cmd += ["--params", str(params_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if expect_ok:
        assert proc.returncode == 0, proc.stderr[-1500:]
        return json.loads(proc.stdout), proc.stderr
    return proc


class TestLegacyPathUnaffected:
    """n_species<=1 (no pmhc_species given) must be indistinguishable from the
    single-affinity path this codebase already pins in reference_values.json.
    """

    def test_no_species_key_omits_species_outputs(self, tmp_path):
        out, _ = _run(tmp_path, extra_args=["--u_assoc", "20"])
        assert "bound_fraction_by_species" not in out
        assert "n_pmhc_by_species" not in out
        assert "pmhc_species" not in out["inputs"]

    def test_same_seed_same_result_with_and_without_empty_species(self, tmp_path):
        """A run with u_assoc set directly must match one that never touches
        the species machinery at all -- same code path, not just same output
        shape."""
        a, _ = _run(tmp_path, extra_args=["--u_assoc", "20", "--seed", "7"], label="a")
        b, _ = _run(tmp_path, extra_args=["--u_assoc", "20", "--seed", "7"], label="b")
        assert a == b


class TestSpeciesAssignment:
    def test_fractions_produce_expected_split(self, tmp_path):
        payload = {
            "pmhc_species": [
                {"fraction": 0.75, "u_assoc": 20.0},
                {"fraction": 0.25, "u_assoc": 20.0},
            ]
        }
        out, _ = _run(tmp_path, params=payload)
        counts = out["n_pmhc_by_species"]
        assert len(counts) == 2
        total = sum(counts)
        assert total == 60  # --n_pmhc 60
        # n_pmhc=60 is not huge, so allow real binomial slack around 0.75/0.25.
        assert counts[0] / total == pytest.approx(0.75, abs=0.15)

    def test_unnormalized_fractions_are_normalized(self, tmp_path):
        """Fractions don't need to sum to 1 -- e.g. raw mixture ratios like
        2:1 should behave the same as 0.667:0.333."""
        raw, _ = _run(
            tmp_path,
            params={"pmhc_species": [
                {"fraction": 2.0, "u_assoc": 20.0},
                {"fraction": 1.0, "u_assoc": 20.0},
            ]},
            label="raw",
        )
        normalized, _ = _run(
            tmp_path,
            params={"pmhc_species": [
                {"fraction": 2.0 / 3.0, "u_assoc": 20.0},
                {"fraction": 1.0 / 3.0, "u_assoc": 20.0},
            ]},
            label="norm",
        )
        assert raw["n_pmhc_by_species"] == normalized["n_pmhc_by_species"]

    def test_species_table_echoed_in_inputs(self, tmp_path):
        out, _ = _run(tmp_path, params=_TWO_SPECIES)
        species = out["inputs"]["pmhc_species"]
        assert len(species) == 2
        assert {s["u_assoc"] for s in species} == {40.0, 5.0}


class TestValidation:
    def test_missing_required_field_errors(self, tmp_path):
        proc = _run(
            tmp_path,
            params={"pmhc_species": [{"fraction": 1.0}]},  # no u_assoc
            expect_ok=False,
        )
        assert proc.returncode != 0
        assert "fraction" in proc.stderr and "u_assoc" in proc.stderr

    def test_zero_total_fraction_errors(self, tmp_path):
        proc = _run(
            tmp_path,
            params={"pmhc_species": [
                {"fraction": 0.0, "u_assoc": 20.0},
                {"fraction": 0.0, "u_assoc": 5.0},
            ]},
            expect_ok=False,
        )
        assert proc.returncode != 0

    def test_empty_array_errors(self, tmp_path):
        proc = _run(tmp_path, params={"pmhc_species": []}, expect_ok=False)
        assert proc.returncode != 0

    def test_too_many_species_errors(self, tmp_path):
        proc = _run(
            tmp_path,
            params={"pmhc_species": [{"fraction": 1.0, "u_assoc": 20.0}] * 9},
            expect_ok=False,
        )
        assert proc.returncode != 0
        assert "MAX_PMHC_SPECIES" in proc.stderr


class TestBoundFractionBySpecies:
    def test_absent_without_monitor_binding(self, tmp_path):
        """Species mixture alone isn't enough -- bind_threshold must also be
        set (via --monitor-binding), same as the aggregate bound-fraction
        machinery it reuses."""
        out, _ = _run(tmp_path, params=_TWO_SPECIES)
        assert "bound_fraction_by_species" not in out

    def test_sums_to_the_aggregate_bound_fraction(self, tmp_path):
        """sim_bound_fraction_by_species() mirrors the same per-pMHC
        proximity loop (same threshold, same break-on-first-match order) as
        main.cpp's compute_bound_fraction() used for binding_timeseries --
        so summed over species it must exactly reproduce the final aggregate
        bound fraction, not just approximately."""
        out, _ = _run(
            tmp_path,
            params=_TWO_SPECIES,
            extra_args=["--monitor-binding", "5.0", "--monitor-interval", "1"],
        )
        by_species = out["bound_fraction_by_species"]
        aggregate_final = out["binding_timeseries"][-1]
        assert sum(by_species) == pytest.approx(aggregate_final, abs=1e-9)

    def test_higher_affinity_species_binds_more(self, tmp_path):
        """The actual discrimination signal this feature exists to measure:
        with a large affinity gap and equal mixture fractions, the
        high-u_assoc species should show a higher bound fraction."""
        out, _ = _run(
            tmp_path,
            params={
                "pmhc_species": [
                    {"fraction": 0.5, "u_assoc": 60.0, "sigma_bind": 3.0},
                    {"fraction": 0.5, "u_assoc": 2.0, "sigma_bind": 3.0},
                ]
            },
            extra_args=[
                "--monitor-binding",
                "5.0",
                "--monitor-interval",
                "1",
                "--time_sec",
                "1.0",
            ],
        )
        high, low = out["bound_fraction_by_species"]
        assert high > low, (
            f"expected the high-affinity species to bind more: high={high}, low={low}"
        )


class TestGpuGuard:
    def test_gpu_species_combination_is_refused_when_gpu_is_actually_active(self, tmp_path):
        """The GPU/Metal Phase-2 kernel has no per-species channel -- refuse
        rather than silently compute wrong physics. Only meaningful on a
        platform where the GPU backend actually initializes (Apple/Metal);
        on the stub backend (Linux/Windows) GPU is never active regardless
        of species, so there is nothing to guard against and this skips."""
        probe, _ = _run(tmp_path, extra_args=["--u_assoc", "20"], label="probe")
        if probe["diagnostics"]["backend"] != "metal":
            pytest.skip("GPU backend not active on this platform (stub fallback)")
        proc = _run(tmp_path, params=_TWO_SPECIES, label="guard", expect_ok=False)
        assert proc.returncode != 0
        assert "not implemented on the GPU" in proc.stderr
