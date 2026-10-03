"""Shared test configuration.

Puts the repository root on sys.path so `import src...` resolves whether the
suite is run as `pytest`, `python -m pytest`, from a subdirectory, or by an
editor's test runner. Without it the suite passes from the root and fails
everywhere else, which is a confusing thing to debug.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
