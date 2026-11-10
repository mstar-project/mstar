"""``MSTAR_MAX_BODY_MB``, the request body limit ``mstar-server`` enforces
(``DefaultBodyLimit``), read here for what it bounds by default on the Python
side: a WebSocket message (``ws_max_message_mb`` unset)."""
from __future__ import annotations

import os
import re

DEFAULT_MAX_BODY_MB = 128
def max_body_bytes() -> int:
    """``MSTAR_MAX_BODY_MB`` in bytes, parsed as mstar-server parses it:
    a value that is not a non-negative integer means the default."""
    raw = os.environ.get("MSTAR_MAX_BODY_MB", "").strip()
    mb = int(raw) if re.fullmatch(r"\+?[0-9]+", raw) else DEFAULT_MAX_BODY_MB
    return mb << 20

