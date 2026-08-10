"""Conftest for the regression suite (audit/pass-1-bugs.md findings).

Makes the package importable from a source checkout and forces a headless
matplotlib backend. No other magic.
"""

import matplotlib

matplotlib.use("Agg", force=True)

try:
    import wholistic_registration  # noqa: F401
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
