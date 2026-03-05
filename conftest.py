"""Root conftest for tcr_signaling submodule tests.

Ensures ``models`` package is importable when running pytest from
``projects/tcr_signaling/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the submodule root so ``from models.<model>.model import ...`` works.
_submodule_root = str(Path(__file__).resolve().parent)
if _submodule_root not in sys.path:
    sys.path.insert(0, _submodule_root)
