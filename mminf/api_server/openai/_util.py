"""Small shared helpers for the OpenAI-compatible serving handlers."""

from __future__ import annotations

import json
import time
import uuid

SSE_DONE = "data: [DONE]\n\n"


def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
