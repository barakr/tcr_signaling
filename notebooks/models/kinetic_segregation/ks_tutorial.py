"""Shared helpers for the kinetic-segregation tutorial notebooks.

Deliberately depends on nothing beyond numpy and the standard library: the KS
notebooks teach the simulator itself, not the `bayesian-metamodeling` framework,
so they stay runnable in a bare checkout (and cheap to execute in CI, which
installs only numpy/scipy/matplotlib/pytest).

Nothing here re-implements physics. Energy curves are read out of the compiled C
through ctypes, because `src/ks_physics.h` is the single source of truth (KS rule
1 in CLAUDE.md) and a numpy re-typing of the Gaussian well would silently drift
from it the next time the model changes.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "COLOR_CD45",
    "COLOR_PMHC",
    "COLOR_TCR",
    "DEFAULTS",
    "ks_dir",
    "ensure_binary",
    "ensure_potentials",
    "run_ks",
    "load_frame",
    "load_frame_meta",
]

# Paul Tol colourblind-safe palette, mirrored from render_movie.py so every
# figure in the repo colours TCR/CD45/pMHC the same way.
COLOR_TCR = "#EE6677"
COLOR_CD45 = "#4477AA"
COLOR_PMHC = "#228833"

# Built-in defaults, transcribed from src/simulation.h. Quoted in the notebooks
# so a reader can see what they get without passing anything.
DEFAULTS: dict[str, Any] = {
    "patch_size": 2000.0,  # nm, PATCH_SIZE_NM
    "cd45_height": 50.0,  # nm, CD45_HEIGHT_NM
    "h0_tcr": 13.0,  # nm, H0_TCR_NM  -- the TCR well is centred HERE, not at 0
    "init_height": 70.0,  # nm, INIT_HEIGHT_NM
    "sigma_bind": 3.0,  # nm, SIGMA_BIND_NM
    "u_assoc": 20.0,  # kT, U_ASSOC_DEFAULT
    "D_mol": 1e4,  # nm^2/s, D_MOL_DEFAULT
    "D_h": 5e4,  # nm^2/s, D_H_DEFAULT
    "n_tcr": 125,
    "n_cd45": 500,
    "grid_size": 64,
    "step_mode": "brownian",
    "binding_mode": "gaussian",
    "pmhc_mode": "uniform",
    "seed": 42,
}

# Flags the C CLI spells with hyphens; everything else uses underscores.
_HYPHEN_FLAGS = {
    "run_dir",
    "no_gpu",
    "dump_frames",
    "dump_interval",
    "monitor_binding",
    "monitor_interval",
    "snapshot_interval",
    "grid_substeps",
}
_BOOL_FLAGS = {"no_gpu", "dump_frames"}

_LIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"


def ks_dir() -> Path:
    """Locate models/kinetic_segregation by searching upward from the cwd.

    The previous notebooks used `Path.cwd().parent.parent.parent`, which assumed
    the repo was checked out as a submodule and broke everywhere else.
    """
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        target = cand / "models" / "kinetic_segregation"
        if (target / "CMakeLists.txt").is_file():
            return target
        if (cand / "CMakeLists.txt").is_file() and cand.name == "kinetic_segregation":
            return cand
    raise RuntimeError(
        f"Could not find models/kinetic_segregation above {here}. "
        "Run this notebook from inside the tcr_signaling repo."
    )


def _build(target: str) -> None:
    proc = subprocess.run(
        ["make", target] if target else ["make"],
        cwd=str(ks_dir()),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"`make {target}` failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def ensure_binary() -> Path:
    """Return the path to `ks_gpu`, building it if absent."""
    binary = ks_dir() / "ks_gpu"
    if not binary.exists():
        _build("")
    if not binary.exists():
        raise RuntimeError(f"{binary} still missing after `make`.")
    return binary


def ensure_potentials() -> ctypes.CDLL:
    """Load the C potential functions, building the shared library if needed.

    Signatures mirror models/kinetic_segregation/tests/test_potentials.py so the
    notebooks and the test suite call the identical symbols.
    """
    lib_path = ks_dir() / "build" / f"libks_potentials{_LIB_EXT}"
    if not lib_path.exists():
        _build("testlib")
    if not lib_path.exists():
        raise RuntimeError(f"{lib_path} still missing after `make testlib`.")

    lib = ctypes.CDLL(str(lib_path))

    # tcr_pmhc_potential(h, h0_tcr, u_assoc, sigma_bind) -> double
    lib.tcr_pmhc_potential.restype = ctypes.c_double
    lib.tcr_pmhc_potential.argtypes = [ctypes.c_double] * 4

    # cd45_repulsion(h, cd45_height, k_rep) -> double
    lib.cd45_repulsion.restype = ctypes.c_double
    lib.cd45_repulsion.argtypes = [ctypes.c_double] * 3

    # bending_energy_delta(h*, n, kappa, dx, gi, gj, old_val, new_val) -> double
    lib.bending_energy_delta.restype = ctypes.c_double
    lib.bending_energy_delta.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    return lib


def _to_argv(params: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + (key.replace("_", "-") if key in _HYPHEN_FLAGS else key)
        if key in _BOOL_FLAGS:
            if value:
                argv.append(flag)
            continue
        argv += [flag, str(value)]
    return argv


def run_ks(run_dir: str | Path | None = None, *, gpu: bool = False, **params: Any) -> dict:
    """Run the KS simulator once and return its parsed JSON.

    Keyword arguments map straight onto CLI flags, so `rigidity_kT=20` becomes
    `--rigidity_kT 20`. Defaults to CPU (`--no-gpu`) for reproducibility across
    machines; pass `gpu=True` to exercise the Metal backend on Apple hardware.

    With `run_dir=None` the simulation runs in a temporary directory that is
    deleted afterwards — fine unless you asked for `dump_frames`, in which case
    pass an explicit `run_dir` so the frames survive.
    """
    binary = ensure_binary()
    params.setdefault("time_sec", 1.0)
    params.setdefault("rigidity_kT", 20.0)

    tmp: tempfile.TemporaryDirectory | None = None
    if run_dir is None:
        tmp = tempfile.TemporaryDirectory()
        run_dir = tmp.name
    try:
        argv = [str(binary), "--run-dir", str(run_dir)]
        if not gpu:
            argv.append("--no-gpu")
        argv += _to_argv(params)
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ks_gpu exited {proc.returncode}\ncmd: {' '.join(argv)}\n{proc.stderr[-2000:]}"
            )
        return json.loads(proc.stdout)
    finally:
        if tmp is not None:
            tmp.cleanup()


def load_frame_meta(run_dir: str | Path) -> dict:
    """Read frames/meta.json, which pins the binary dump contract (KS rule 8)."""
    return json.loads((Path(run_dir) / "frames" / "meta.json").read_text(encoding="utf-8"))


def load_frame(run_dir: str | Path, step: int, meta: dict | None = None):
    """Decode one dumped frame into (height_field, tcr_xy, cd45_xy).

    Same decoding as render_movie.load_frame; it takes the run directory and
    reads the shape from meta.json rather than making the caller pass sizes.
    """
    frames = Path(run_dir) / "frames"
    meta = meta or load_frame_meta(run_dir)
    n = int(meta["grid_size"])
    n_tcr, n_cd45 = int(meta["n_tcr"]), int(meta["n_cd45"])

    h = np.fromfile(frames / f"h_{step:05d}.bin", dtype=np.float32).reshape(n, n)
    mol = np.fromfile(frames / f"mol_{step:05d}.bin", dtype=np.float64)
    tcr = mol[: n_tcr * 2].reshape(n_tcr, 2)
    cd45 = mol[n_tcr * 2 :].reshape(n_cd45, 2)
    return h, tcr, cd45
