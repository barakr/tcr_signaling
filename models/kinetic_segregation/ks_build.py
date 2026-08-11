"""Locate and build the compiled KS artifacts, on every supported platform.

Everything that needs `ks_gpu` or the ctypes potentials library goes through
here: the test suite, the `python -m models.kinetic_segregation` wrapper, and
the tutorial notebook helper. One resolver instead of ten, because the three
things that vary by platform are exactly the three things easy to get wrong in
one file and not the others:

* **Executable suffix** — `ks_gpu` on Unix, `ks_gpu.exe` on Windows.
* **Shared-library naming** — `libks_potentials.dylib` / `.so`, but
  `ks_potentials.dll` on Windows (no `lib` prefix).
* **Config subdirectories** — the Visual Studio generator is multi-config and
  appends `Release/` or `Debug/` to its output paths. CMakeLists.txt pins the
  usual case, but a user running `cmake --build` by hand can still land there.

The build itself shells `cmake --build`, not `make`. `make` is a convenience
wrapper for Unix humans and is not present on a stock Windows box, while CMake
is required on every platform already.

Framework-free by design: standard library only, so the tutorial notebooks can
import it without `bayesian-metamodeling` installed.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "BUILD_HINT",
    "KS_DIR",
    "binary_name",
    "build",
    "find_binary",
    "find_potentials",
    "have_cmake",
    "have_compiler",
    "load_potentials",
    "toolchain_hint",
]

KS_DIR = Path(__file__).resolve().parent

_IS_WINDOWS = os.name == "nt"

# Config subdirectories a multi-config generator may introduce, in the order we
# prefer them. Release first: CMakeLists.txt defaults CMAKE_BUILD_TYPE to Release.
_CONFIGS = ("", "Release", "RelWithDebInfo", "Debug")


def binary_name() -> str:
    """`ks_gpu` or `ks_gpu.exe`."""
    return "ks_gpu.exe" if _IS_WINDOWS else "ks_gpu"


def _potentials_names() -> tuple[str, ...]:
    """Candidate filenames for the ctypes shared library.

    Windows drops the `lib` prefix and uses `.dll`; macOS uses `.dylib`. Both
    spellings are listed for Windows because MinGW-style toolchains keep the
    prefix even when targeting a `.dll`.
    """
    if _IS_WINDOWS:
        return ("ks_potentials.dll", "libks_potentials.dll")
    if sys.platform == "darwin":
        return ("libks_potentials.dylib",)
    return ("libks_potentials.so",)


def _candidates(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [root / cfg / name for name in names for cfg in _CONFIGS]


def have_cmake() -> bool:
    """Is CMake on PATH?"""
    return shutil.which(os.environ.get("CMAKE", "cmake")) is not None


def have_compiler() -> bool:
    """Is a C/C++ compiler available to CMake?

    On Unix this is just "is there a compiler on PATH". On Windows it is not:
    MSVC's `cl.exe` is only on PATH inside a Developer Command Prompt, and CMake
    finds it for itself through the registry. Testing `which("c++")` there
    reports "no compiler" on a perfectly working Visual Studio install — a false
    negative that tells a correctly-configured user to reinstall.

    So on Windows we ask `vswhere`, the tool Microsoft ships for exactly this,
    for an installation carrying the VC++ toolset. That component is precisely
    the "Desktop development with C++" workload, which is the piece people miss.
    """
    if _IS_WINDOWS:
        if shutil.which("cl"):
            return True
        program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles", "")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.is_file():
            return False
        try:
            proc = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return bool(proc.stdout.strip())
    return any(shutil.which(c) for c in ("c++", "clang++", "g++", "cc", "gcc"))


def toolchain_hint() -> str:
    """Platform-correct instructions for installing a compiler and CMake."""
    if _IS_WINDOWS:
        return (
            "Windows needs a C/C++ compiler and CMake:\n"
            "  1. Install 'Visual Studio 2022 Build Tools' and tick the\n"
            "     'Desktop development with C++' workload (this provides both\n"
            "     the MSVC compiler and CMake):\n"
            "       winget install Microsoft.VisualStudio.2022.BuildTools "
            '--override "--wait --quiet --add '
            'Microsoft.VisualStudio.Workload.VCTools --includeRecommended"\n'
            "  2. Reopen your terminal so the new tools are on PATH.\n"
            "  If CMake is still not found, `conda install -c conda-forge cmake` "
            "or `winget install Kitware.CMake` adds it."
        )
    if sys.platform == "darwin":
        return (
            "macOS needs the Command Line Tools and CMake:\n"
            "  xcode-select --install\n"
            "  conda install -c conda-forge cmake     # or: brew install cmake"
        )
    return (
        "Linux needs a C/C++ compiler and CMake:\n"
        "  sudo apt install build-essential cmake     # Debian/Ubuntu\n"
        "  conda install -c conda-forge cmake compilers   # without sudo"
    )


BUILD_HINT = "cd models/kinetic_segregation && cmake -S . -B build && cmake --build build"


def build(target: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Configure and build via CMake. Returns the completed `cmake --build` call.

    Raises RuntimeError with the compiler output on failure — a build error the
    caller can actually read beats a later "file not found".
    """
    cmake = os.environ.get("CMAKE", "cmake")
    configure = [cmake, "-S", str(KS_DIR), "-B", str(KS_DIR / "build")]
    # Single-config generators need the build type at configure time; it is
    # ignored (with a warning suppressed by --log-level) by multi-config ones.
    configure += ["-DCMAKE_BUILD_TYPE=Release"]

    proc = subprocess.run(configure, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "CMake configuration failed. Is a compiler installed?\n\n"
            f"{toolchain_hint()}\n\n--- cmake output ---\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    cmd = [cmake, "--build", str(KS_DIR / "build"), "--config", "Release"]
    if target:
        cmd += ["--target", target]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Build failed{f' (target {target})' if target else ''}.\n\n"
            f"--- compiler output ---\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    return proc


def find_binary(auto_build: bool = False) -> Path | None:
    """Return the path to `ks_gpu`, or None if it is not built.

    With `auto_build=True`, attempts a build first and propagates the compiler
    error on failure.
    """
    roots = (KS_DIR, KS_DIR / "build")
    for cand in [c for root in roots for c in _candidates(root, (binary_name(),))]:
        if cand.is_file():
            return cand
    if not auto_build:
        return None
    build()
    for cand in [c for root in roots for c in _candidates(root, (binary_name(),))]:
        if cand.is_file():
            return cand
    return None


def find_potentials(auto_build: bool = False) -> Path | None:
    """Return the path to the ctypes potentials library, or None."""
    names = _potentials_names()
    roots = (KS_DIR / "build", KS_DIR)
    for cand in [c for root in roots for c in _candidates(root, names)]:
        if cand.is_file():
            return cand
    if not auto_build:
        return None
    build("ks_potentials")
    for cand in [c for root in roots for c in _candidates(root, names)]:
        if cand.is_file():
            return cand
    return None


def load_potentials(path: Path | None = None) -> ctypes.CDLL:
    """Load the potentials library and declare the shared ctypes signatures.

    Declared once here so the test suite and the notebooks cannot drift into
    calling the same C symbol with two different prototypes.
    """
    if path is None:
        path = find_potentials(auto_build=True)
    if path is None:
        raise RuntimeError(f"potentials library not built. {BUILD_HINT}")

    lib = ctypes.CDLL(str(path))
    _dbl = ctypes.c_double
    _dblp = ctypes.POINTER(ctypes.c_double)
    _int = ctypes.c_int

    # Signatures mirror src/potentials.h, which is authoritative.

    # tcr_pmhc_potential(double h, double h0_tcr, double u_assoc, double sigma_bind)
    lib.tcr_pmhc_potential.restype = _dbl
    lib.tcr_pmhc_potential.argtypes = [_dbl] * 4

    # cd45_repulsion(double h, double cd45_height, double k_rep)
    lib.cd45_repulsion.restype = _dbl
    lib.cd45_repulsion.argtypes = [_dbl] * 3

    # bending_energy_delta(const double *h, int n, double kappa, double dx,
    #                      int gi, int gj, double old_val, double new_val)
    # NB: `h` must ALREADY contain new_val at [gi][gj] — see the header.
    lib.bending_energy_delta.restype = _dbl
    lib.bending_energy_delta.argtypes = [_dblp, _int, _dbl, _dbl, _int, _int, _dbl, _dbl]

    # mol_repulsion(const double *pos, int idx, const double *all_pos, int n_mol,
    #               double eps, double r_cut, double patch_size)
    lib.mol_repulsion.restype = _dbl
    lib.mol_repulsion.argtypes = [_dblp, _int, _dblp, _int, _dbl, _dbl, _dbl]

    return lib
