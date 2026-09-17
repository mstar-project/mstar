"""Enable ``python -m tools.vibesys ...`` from the mstar repo root."""

from __future__ import annotations

from tools.vibesys.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
