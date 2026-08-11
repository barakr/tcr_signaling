"""Root conftest for tcr_signaling submodule tests.

Ensures ``models`` package is importable when running pytest from
``projects/tcr_signaling/``, gives Windows an event loop pyzmq can use, and
gates the Metal-only tests off macOS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the submodule root so ``from models.<model>.model import ...`` works.
_submodule_root = str(Path(__file__).resolve().parent)
if _submodule_root not in sys.path:
    sys.path.insert(0, _submodule_root)


# ── Windows: give pyzmq a selector event loop ───────────────────────────────
#
# pyzmq needs `add_reader`, which only the selector event loop implements.
# Python has defaulted to the Proactor loop on Windows since 3.8, so pyzmq
# compensates by spawning an extra selector thread and emits a RuntimeWarning
# saying so. `pytest.ini` sets `filterwarnings = error`, which turned that
# benign compensation into a failure of every notebook test *before a single
# notebook cell ran* — and, in a single pytest process running all five, left
# zmq half-initialised so the run then hung at teardown until CI killed it.
#
# Setting the policy removes the condition rather than silencing the warning;
# it is what Jupyter itself does on Windows. No effect on macOS or Linux, where
# the attribute does not exist.
if sys.platform == "win32":  # pragma: no cover - platform-specific
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Why these skip rather than run: `CMakeLists.txt` compiles `src/gpu_stub.c`
# instead of `src/metal_engine.m` off Apple, and the stub's
# `gpu_engine_create()` returns NULL, which makes `simulation.c` fall back to
# the CPU path. So on Linux `--gpu` IS the CPU path: a CPU-vs-GPU comparison
# would pass while comparing a result to itself. That is a false green, which
# is worse than an honest skip — hence a narrow, reasoned gate rather than
# skipping the files wholesale. Everything in those files that does not touch
# the GPU path still runs on every platform.
_METAL_SKIP_REASON = (
    "requires Metal GPU backend (macOS only): off Apple, CMake builds gpu_stub.c "
    "whose gpu_engine_create() returns NULL, so --gpu silently falls back to CPU "
    "and a CPU-vs-GPU assertion would compare CPU against itself"
)


def pytest_collection_modifyitems(config, items):
    if sys.platform == "darwin":
        return
    skip_metal = pytest.mark.skip(reason=_METAL_SKIP_REASON)
    for item in items:
        if "requires_metal" in item.keywords:
            item.add_marker(skip_metal)
