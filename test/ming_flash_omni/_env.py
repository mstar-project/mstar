"""Shared server-URL resolution for the Ming-flash-omni request scripts.

Mirrors ``test/qwen3-omni/_env.py``: read HOST/PORT from the environment (or a
``.env`` next to this file / in the cwd) so every script in this directory talks
to the same server without repeating the plumbing.
"""

import os
from pathlib import Path

_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_file = Path(".env")
        if not env_file.exists():
            return  # fall back to existing env vars / defaults

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def get_server_url() -> str:
    load_env()
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    return f"http://{host}:{port}/generate"
