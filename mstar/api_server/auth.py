"""The deployment's optional API key, which ``mstar-server`` checks
(``rust/server/src/auth.rs``; see docs/serving.rst).

Off unless ``--api-key`` or ``MSTAR_API_KEY`` sets a key. The launcher hands
the key to mstar-server explicitly and keeps it out of every other child's
environment.
"""

from __future__ import annotations

import os
import sys

API_KEY_ENV = "MSTAR_API_KEY"

def resolve_api_key(flag: str | None = None) -> str | None:
    """The deployment's key: ``--api-key`` if given, else ``MSTAR_API_KEY``.
    An empty value means no key."""
    key = flag if flag is not None else os.environ.get(API_KEY_ENV)
    return key or None


def keep_api_key_from_children(api_key: str | None) -> None:
    """Only the HTTP frontend needs the key. Drop it from the environment and
    ``sys.argv`` (the spawn start method copies both into each child) before
    this process spawns the conductor, workers or media subprocesses."""
    os.environ.pop(API_KEY_ENV, None)
    if api_key is None:
        return
    redacted = []
    for arg in sys.argv:
        if arg == api_key:
            arg = "<redacted>"
        elif arg.partition("=")[2] == api_key:
            arg = arg.partition("=")[0] + "=<redacted>"
        redacted.append(arg)
    sys.argv[:] = redacted

