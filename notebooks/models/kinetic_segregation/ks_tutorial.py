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
import itertools
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
    "toolchain_hint",
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


def _ks_build():
    """Import the model's own ks_build module (path/build resolution).

    Imported lazily and by path rather than at module import time: `ks_dir()`
    searches upward from the notebook's working directory, which is only known
    once a cell runs. Reusing the model's resolver rather than duplicating it
    here is what keeps the notebooks working on Windows, where the binary is
    `ks_gpu.exe` and the ctypes library is `ks_potentials.dll`.
    """
    global _KS_BUILD
    if _KS_BUILD is None:
        target = str(ks_dir())
        if target not in sys.path:
            sys.path.insert(0, target)
        import ks_build  # noqa: PLC0415  (deliberately deferred; see docstring)

        _KS_BUILD = ks_build
    return _KS_BUILD


_KS_BUILD = None


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


def toolchain_hint() -> str:
    """Platform-correct instructions for installing a compiler and CMake.

    Notebooks print this when a build fails, so a Windows reader hitting a
    missing compiler gets the winget command rather than a CMake stack trace.
    """
    return _ks_build().toolchain_hint()


def ensure_binary() -> Path:
    """Return the path to `ks_gpu` (`ks_gpu.exe` on Windows), building if absent."""
    kb = _ks_build()
    binary = kb.find_binary(auto_build=True)
    if binary is None:
        raise RuntimeError(
            f"{kb.binary_name()} still missing after the build.\n\n{kb.toolchain_hint()}"
        )
    return binary


def ensure_potentials() -> ctypes.CDLL:
    """Load the C potential functions, building the shared library if needed.

    The ctypes prototypes live in the model's `ks_build.load_potentials`, so the
    notebooks and the test suite cannot drift into calling the same C symbol
    with two different signatures.
    """
    return _ks_build().load_potentials()


_SESSION_TMP: tempfile.TemporaryDirectory | None = None
_RUN_SEQ = itertools.count()


def _scratch_dir() -> Path:
    """A fresh subdirectory inside ONE session-scoped temporary directory.

    Deliberately not a `TemporaryDirectory` per simulation. KS_4 alone runs
    dozens of parameter points, and creating then recursively deleting a
    directory for each is pure overhead — negligible on Unix, but on Windows
    every create and delete goes through the filesystem filter drivers
    (Defender among them), which is a per-call cost that does not shrink with
    how small the simulation is.

    The whole tree is removed when the kernel exits. Each run writes one small
    JSON, so nothing accumulates that matters; frame dumps take an explicit
    `run_dir` precisely because they are the large case.
    """
    global _SESSION_TMP
    if _SESSION_TMP is None:
        _SESSION_TMP = tempfile.TemporaryDirectory(prefix="ks_tutorial_")
    path = Path(_SESSION_TMP.name) / f"run_{next(_RUN_SEQ):04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    With `run_dir=None` the simulation runs in a scratch directory that is
    cleaned up when the kernel exits — fine unless you asked for `dump_frames`,
    in which case pass an explicit `run_dir` so the frames are easy to find.
    """
    binary = ensure_binary()
    params.setdefault("time_sec", 1.0)
    params.setdefault("rigidity_kT", 20.0)

    if run_dir is None:
        run_dir = _scratch_dir()
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
