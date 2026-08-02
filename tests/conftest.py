"""Pytest bootstrap: put the repo root on sys.path so `from tracker.kalman import ...`
resolves regardless of the directory pytest is invoked from (safer than relying on
rootdir inference, especially on Windows)."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
