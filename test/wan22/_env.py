"""Load .env into os.environ so every wan22 test script shares one config.

The wan22 benchmark works fine
with no .env (the defaults below), but this keeps the
launcher / request / benchmark scripts consistent with the other model dirs.
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


def get_host_port() -> tuple[str, int]:
    load_env()
    return os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8100"))


def get_server_url() -> str:
    host, port = get_host_port()
    return f"http://{host}:{port}"
