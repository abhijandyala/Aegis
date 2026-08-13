"""Railway / Railpack entrypoint. Same process as `python web/server.py`."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.server import main

if __name__ == "__main__":
    main()
