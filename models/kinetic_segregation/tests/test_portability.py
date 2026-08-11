"""Static guards on the things that make the KS model build on Windows.

These are cheap invariants rather than behavioural tests, and they exist because
each one encodes a defect that actually shipped:

* `M_PI` used in a file that reaches `<math.h>` directly compiles on macOS and
  Linux and fails only on MSVC, so a Unix-only developer cannot notice it.
* A test that hardcodes `"ks_gpu"` finds nothing on Windows (`ks_gpu.exe`) and
  *skips* — which reads as a pass. Same for a `make` that does not exist there.

The Windows CI job catches all of this too, but it catches it after a push and
only while the job exists; these run in every developer's `make fast`, name the
rule, and cost milliseconds.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from models.kinetic_segregation import ks_build

_PKG = Path(__file__).resolve().parents[1]
_SRC = _PKG / "src"
_TESTS = _PKG / "tests"


# ── C sources ───────────────────────────────────────────────────────────────


def _cpu_sources() -> list[Path]:
    """C sources compiled by the CPU build (excludes the Metal shader)."""
    return sorted(p for p in _SRC.glob("*.c"))


class TestMathPortability:
    def test_m_pi_users_include_ks_compat(self):
        """Any C file using M_PI must include ks_compat.h, not bare <math.h>.

        MSVC only declares M_PI when _USE_MATH_DEFINES precedes <math.h>.
        ks_compat.h does that; a direct `#include <math.h>` does not, and the
        resulting build failure is invisible on Unix.
        """
        offenders = []
        for path in _cpu_sources():
            text = path.read_text()
            if "M_PI" not in text:
                continue
            if '#include "ks_compat.h"' not in text:
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} use M_PI without including ks_compat.h — this builds on "
            "macOS/Linux and fails on MSVC. Replace the <math.h> include."
        )

    def test_ks_compat_defines_m_pi_without_the_system_header(self):
        """The fallback must be unconditional enough to survive a hostile CRT."""
        text = (_SRC / "ks_compat.h").read_text()
        assert "_USE_MATH_DEFINES" in text
        assert "#ifndef M_PI" in text, "needs a literal fallback, not just the define"

    def test_no_posix_only_clock_outside_ks_compat(self):
        """clock_gettime is POSIX; MSVC has no such function.

        It is legitimate inside ks_compat.h, which branches on _WIN32.
        """
        offenders = [p.name for p in _cpu_sources() if "clock_gettime" in p.read_text()]
        assert not offenders, (
            f"{offenders} call clock_gettime directly; use ks_clock_ms() from ks_compat.h"
        )


# ── Python side ─────────────────────────────────────────────────────────────


_NOTEBOOK_DIR = _PKG.parents[1] / "notebooks"


def _notebook_code(path: Path) -> str:
    """Concatenated source of a notebook's CODE cells only.

    Markdown is excluded deliberately: prose may legitimately mention `make` as
    the Unix shorthand. It is executable cells that must not depend on it.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"
    )


_REPO = _PKG.parents[1]

# Directories holding Python that invokes the compiled model. `experiments/` and
# `examples/` are in here because a Windows user follows those too — KS_5 sends
# readers straight to experiments/ks_behavior_sweep/run.py.
_SCRIPT_DIRS = (
    _PKG,
    _PKG / "benchmark",
    _REPO / "examples",
    _REPO / "experiments",
)


def _python_consumers() -> list[tuple[str, str]]:
    """(label, source) for everything that runs the model.

    Test modules, the notebook helper, the executable cells of the tutorial
    notebooks, and the analysis/generation scripts. Two gaps found the hard way:
    a `subprocess.run(["make"])` sat in KS_3's movie cell where a .py-only scan
    could not see it, and ten generation scripts hardcoded the bare binary name
    where a tests-only scan could not see them either.
    """
    items: list[tuple[str, str]] = []
    for p in _TESTS.glob("test_*.py"):
        if p.name != "test_portability.py":
            items.append((p.name, p.read_text()))
    helper = _NOTEBOOK_DIR / "models" / "kinetic_segregation" / "ks_tutorial.py"
    if helper.is_file():
        items.append((helper.name, helper.read_text()))
    for p in sorted(_NOTEBOOK_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        items.append((p.name, _notebook_code(p)))
    # This file is excluded from its own scan: it necessarily spells out every
    # pattern it forbids, in the regexes and in the failure messages.
    seen = {name for name, _ in items} | {"test_portability.py"}
    for directory in _SCRIPT_DIRS:
        if not directory.is_dir():
            continue
        for p in sorted(directory.rglob("*.py")):
            if p.name in seen or "build" in p.parts:
                continue
            items.append((p.name, p.read_text()))
            seen.add(p.name)
    return sorted(items)


class TestNoHardcodedPlatformAssumptions:
    def test_nothing_hardcodes_the_bare_binary_name(self):
        """`/ "ks_gpu"` misses ks_gpu.exe, and a missing binary *skips*.

        Go through ks_build.binary_name() / find_binary() instead.
        """
        pattern = re.compile(r"""/\s*["']ks_gpu["']""")
        offenders = [n for n, src in _python_consumers() if pattern.search(src)]
        assert not offenders, (
            f"{offenders} build a path with the literal 'ks_gpu'; on Windows the file "
            "is ks_gpu.exe and these tests would silently skip. Use ks_build.binary_name()."
        )

    def test_nothing_shells_out_to_make(self):
        """`make` is absent on a stock Windows box; CMake is required everywhere."""
        pattern = re.compile(r"""\[\s*["']make["']""")
        offenders = [n for n, src in _python_consumers() if pattern.search(src)]
        assert not offenders, (
            f"{offenders} invoke `make`, which does not exist on Windows. "
            "Use ks_build.build() (or ks_tutorial.ensure_binary()), which call cmake."
        )

    def test_nothing_hardcodes_a_shared_library_suffix(self):
        """`.dylib if darwin else .so` silently excludes Windows' .dll."""
        offenders = [n for n, src in _python_consumers() if '".dylib"' in src or "'.dylib'" in src]
        assert not offenders, (
            f"{offenders} pick a library suffix inline; use ks_build.find_potentials()."
        )

    def test_the_scan_actually_covers_what_it_claims(self):
        """Guard the guard: a glob that matches nothing passes every test above.

        This is the same failure mode the rest of this file exists to prevent —
        a check that quietly inspects nothing and reports success.
        """
        names = [n for n, _ in _python_consumers()]
        assert any(n.endswith(".ipynb") for n in names), "no notebooks scanned"
        assert sum(1 for n in names if n.startswith("KS_")) >= 5, (
            f"expected the 5 KS notebooks in the scan, saw {names}"
        )
        # The generation/analysis scripts, which were the second gap.
        assert "run_benchmark.py" in names, "benchmark/ not scanned"
        assert any(n.startswith("generate_") for n in names), "generate_* scripts not scanned"
        assert "ks_tutorial.py" in names, "notebook helper not scanned"


class TestWindowsEventLoop:
    def test_conftest_gives_windows_a_selector_event_loop(self):
        """pyzmq needs `add_reader`, which only the selector loop implements.

        Python defaults to the Proactor loop on Windows, so pyzmq compensates
        with an extra thread and emits a RuntimeWarning — which `filterwarnings
        = error` turns into a failure of every notebook test before a single
        cell runs, and which hung the run at teardown on top of that.

        Checked statically and on every platform on purpose: a test that skipped
        off Windows would be a skip with an unsanctioned reason (which the CI
        audit rejects) *and* would only verify the rule where it already holds.
        """
        text = (_REPO / "conftest.py").read_text()
        assert "WindowsSelectorEventLoopPolicy" in text, (
            "conftest.py must set the selector event loop policy on Windows; "
            "without it every notebook test fails there on a pyzmq RuntimeWarning"
        )


class TestKsBuildResolver:
    def test_binary_name_matches_this_platform(self):
        expected = "ks_gpu.exe" if os.name == "nt" else "ks_gpu"
        assert ks_build.binary_name() == expected

    def test_finds_the_built_binary(self):
        """The resolver must locate what the build just produced."""
        binary = ks_build.find_binary()
        if binary is None:
            pytest.skip("ks_gpu not built")
        assert binary.is_file()
        assert binary.name == ks_build.binary_name()

    def test_config_subdirectories_are_searched(self):
        """Multi-config generators (Visual Studio) add Release/ or Debug/.

        CMakeLists.txt pins the common case, but `cmake --build` run by hand can
        still land there, so the resolver must look.
        """
        assert "Release" in ks_build._CONFIGS
        assert "Debug" in ks_build._CONFIGS
        assert ks_build._CONFIGS[0] == "", "the pinned location must be preferred"

    def test_toolchain_hint_is_actionable_on_every_platform(self):
        """A reader who hits a missing compiler gets a command, not a stack trace."""
        hint = ks_build.toolchain_hint()
        assert hint.strip()
        # Whatever the platform, the hint names a way to obtain CMake.
        assert "cmake" in hint.lower()
