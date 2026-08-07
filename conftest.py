"""Root conftest for tcr_signaling submodule tests.

Ensures ``models`` package is importable when running pytest from
``projects/tcr_signaling/``, and gates the Metal-only tests off macOS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the submodule root so ``from models.<model>.model import ...`` works.
_submodule_root = str(Path(__file__).resolve().parent)
if _submodule_root not in sys.path:
    sys.path.insert(0, _submodule_root)


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
