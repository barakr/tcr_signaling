"""The compute backend must depend on the flags, and on nothing else.

This file exists because for a long time it depended on the caller's working
directory. `metal_engine.m` looked for its shader at `src/shaders.metallib`,
`shaders.metallib` and `models/kinetic_segregation/src/shaders.metallib` —
all resolved against the **cwd** — so on macOS:

    cd projects/tcr_signaling && ks_gpu …   → GPU → depletion_width_nm 312.67
    cd <parent repo root>     && ks_gpu …   → CPU → depletion_width_nm 283.91

Same binary, same arguments, ~7% apart, and nothing in the output said which had
run. The ModelSpec entrypoint is launched from the parent root, so every
framework-driven sweep on macOS silently took the CPU path while anyone running
the model by hand from the submodule root got the GPU path.

The tests below are deliberately platform-neutral. Off Apple there is no Metal
at all (CMake compiles `gpu_stub.c`, whose `gpu_engine_create()` returns NULL),
so the backend is `cpu` everywhere — but "the answer does not depend on where I
stand" is exactly as meaningful there, and `--require-gpu` is *more* meaningful,
because it must fail. Nothing here is gated on macOS.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from models.kinetic_segregation import ks_build

_PKG_DIR = Path(__file__).resolve().parents[1]
_SRC = _PKG_DIR / "src"
_SUBMODULE_ROOT = _PKG_DIR.parents[1]
_BINARY = ks_build.find_binary() or (_PKG_DIR / ks_build.binary_name())

# Small enough to run several times in a fast suite; large enough that the GPU
# and CPU paths give visibly different numbers, which is the whole point.
_ARGS = [
    "--time_sec", "0.05",
    "--rigidity_kT", "20",
    "--n_steps", "50",
    "--grid_size", "16",
    "--n_tcr", "5",
    "--n_cd45", "10",
]  # fmt: skip


def _ensure_binary():
    if _BINARY.exists():
        return
    try:
        ks_build.build()
    except RuntimeError as exc:
        pytest.skip(f"Failed to build binary: {exc}")


def _run(tmp_path, *, cwd, label, extra=(), binary=None):
    """Run the model from `cwd` and return (CompletedProcess, parsed json|None)."""
    _ensure_binary()
    run_dir = tmp_path / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(binary or _BINARY), *_ARGS, *extra, "--run-dir", str(run_dir)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = None
    if proc.returncode == 0:
        payload = json.loads(proc.stdout)
    return proc, payload


# ── The defect itself ───────────────────────────────────────────────────────


class TestBackendIsIndependentOfWorkingDirectory:
    def test_same_result_from_every_working_directory(self, tmp_path):
        """Identical args from four unrelated cwds must give identical physics.

        The four are chosen to span the old candidate list: the submodule root
        and the package directory used to *find* a shader (GPU), while a
        temporary directory and the filesystem root used to miss it (CPU).
        """
        cwds = {
            "submodule_root": _SUBMODULE_ROOT,
            "package_dir": _PKG_DIR,
            "tmp": tmp_path,
            "fs_root": Path(sys.executable).anchor or "/",
        }
        results = {}
        for name, cwd in cwds.items():
            _, payload = _run(tmp_path, cwd=cwd, label=f"cwd_{name}")
            results[name] = (
                payload["depletion_width_nm"],
                payload["diagnostics"]["backend"],
            )

        distinct = set(results.values())
        assert len(distinct) == 1, (
            "backend and/or result depend on the working directory — the shader "
            "lookup in src/metal_engine.m must resolve against the executable's "
            f"directory, never the cwd. Got: {results}"
        )

    def test_binary_away_from_its_shader_does_not_borrow_one_from_the_cwd(self, tmp_path):
        """A binary with no shader beside it must not pick one up from the cwd.

        The complement of the test above: copy the executable somewhere bare and
        run it from the submodule root, which is precisely where the old
        cwd-relative lookup *would* have found `models/kinetic_segregation/src/
        shaders.metal`. It must fall back to the CPU and say so.
        """
        bare = tmp_path / "bare"
        bare.mkdir()
        isolated = bare / _BINARY.name
        isolated.write_bytes(_BINARY.read_bytes())
        isolated.chmod(0o755)

        _, payload = _run(tmp_path, cwd=_SUBMODULE_ROOT, label="isolated", binary=isolated)
        assert payload["diagnostics"]["backend"] == "cpu"


# ── Provenance: a stored run must say what produced it ──────────────────────


class TestBackendIsRecorded:
    def test_backend_is_reported_in_json(self, tmp_path):
        """`__main__.py` discards stderr on a zero exit, so JSON is the only
        channel that reaches a stored run's provenance."""
        _, payload = _run(tmp_path, cwd=tmp_path, label="record")
        assert payload["diagnostics"]["backend"] in {"metal", "cpu"}

    def test_backend_is_reported_on_stderr(self, tmp_path):
        proc, _ = _run(tmp_path, cwd=tmp_path, label="stderr")
        assert "Backend: " in proc.stderr

    def test_no_gpu_reports_cpu(self, tmp_path):
        _, payload = _run(tmp_path, cwd=tmp_path, label="nogpu", extra=["--no-gpu"])
        assert payload["diagnostics"]["backend"] == "cpu"

    def test_recorded_backend_matches_the_numbers(self, tmp_path):
        """The field must track the physics, not just be printed alongside it.

        On macOS the two paths differ; off Apple `--no-gpu` and the default are
        the same CPU core, so the assertion is 'same label ⇒ same number', which
        holds on every platform and still catches a hardcoded label.
        """
        _, default = _run(tmp_path, cwd=tmp_path, label="match_default")
        _, forced_cpu = _run(tmp_path, cwd=tmp_path, label="match_cpu", extra=["--no-gpu"])
        same_backend = default["diagnostics"]["backend"] == forced_cpu["diagnostics"]["backend"]
        same_number = default["depletion_width_nm"] == forced_cpu["depletion_width_nm"]
        assert same_backend == same_number, (
            "diagnostics.backend disagrees with the result it labels: "
            f"default={default['diagnostics']['backend']}/"
            f"{default['depletion_width_nm']} vs "
            f"--no-gpu={forced_cpu['diagnostics']['backend']}/"
            f"{forced_cpu['depletion_width_nm']}"
        )


# ── --require-gpu: a downgrade you asked not to have is an error ────────────


class TestRequireGpu:
    def test_require_gpu_and_no_gpu_are_mutually_exclusive(self, tmp_path):
        proc, _ = _run(
            tmp_path,
            cwd=tmp_path,
            label="contradiction",
            extra=["--no-gpu", "--require-gpu"],
        )
        assert proc.returncode == 1
        assert "mutually exclusive" in proc.stderr

    def test_require_gpu_fails_when_the_backend_is_unavailable(self, tmp_path):
        """Copy the binary away from its shader, then insist on the GPU.

        On Apple this fails because no shader is beside the copy; off Apple it
        fails because `gpu_stub.c` never provides a GPU at all. Either way the
        flag must refuse rather than quietly hand back CPU numbers — which is
        the whole reason it exists.
        """
        bare = tmp_path / "bare_require"
        bare.mkdir()
        isolated = bare / _BINARY.name
        isolated.write_bytes(_BINARY.read_bytes())
        isolated.chmod(0o755)

        proc, _ = _run(
            tmp_path,
            cwd=_SUBMODULE_ROOT,
            label="require_fail",
            extra=["--require-gpu"],
            binary=isolated,
        )
        assert proc.returncode == 1
        assert "--require-gpu" in proc.stderr

    @pytest.mark.requires_metal
    def test_require_gpu_succeeds_with_metal_available(self, tmp_path):
        """The in-place binary has its shader in `src/`, so this must pass —
        from a cwd that has nothing to do with the repo."""
        proc, payload = _run(tmp_path, cwd=tmp_path, label="require_ok", extra=["--require-gpu"])
        assert proc.returncode == 0, proc.stderr[-500:]
        assert payload["diagnostics"]["backend"] == "metal"


# ── Static guard: the rule, asserted where no Metal build exists ────────────


class TestShaderLookupIsStaticallyExecutableRelative:
    """`metal_engine.m` is compiled only on Apple, so a regression here is
    invisible to the Linux and Windows CI jobs. This reads the source instead,
    and therefore runs everywhere — the same trick `test_portability.py` uses.
    """

    def test_no_cwd_relative_shader_candidates(self):
        """A shader filename may appear only as an argument to the helper.

        The rule is positional, not textual: `_ks_shader_candidates(execDir, name)`
        is the one place that turns a name into paths, and it anchors them to the
        executable. A literal used anywhere else is a path the process would
        resolve against its cwd — which is the bug. A literal containing a `/` is
        always wrong, even inside the helper call.
        """
        text = (_SRC / "metal_engine.m").read_text()
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("*", "/*", "//"))]
        offenders = []
        for ln in code:
            for literal in re.findall(r'@"([^"]*shaders\.metal(?:lib)?)"', ln):
                if "/" in literal or "_ks_shader_candidates" not in ln:
                    offenders.append(ln.strip())
        assert not offenders, (
            "cwd-relative shader candidates are back in metal_engine.m: "
            f"{offenders}. These make the compute backend — and therefore the "
            "physics — depend on the caller's working directory. Resolve every "
            "candidate against the executable's directory instead."
        )

    def test_candidates_are_built_from_the_executable_directory(self):
        text = (_SRC / "metal_engine.m").read_text()
        assert "_ks_shader_candidates" in text
        assert "executablePath" in text, (
            "the shader lookup no longer derives its search paths from the "
            "executable's own location"
        )
