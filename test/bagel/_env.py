"""Load test/bagel/.env into os.environ so every script shares one config."""

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
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    return f"http://{host}:{port}/generate"
