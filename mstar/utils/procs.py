"""Process-lifetime helpers for the server's process tree (api_server → conductor → workers).

Two failure modes leave GPU memory behind: a process killed with SIGTERM dies without running
``atexit``/``finally`` (so the conductor never terminates its workers), and a child whose parent
died for any reason keeps running (a worker blocked on its message socket holds its GPU memory
for the life of the node). ``graceful_sigterm`` turns SIGTERM into a normal interpreter exit;
``die_with_parent`` asks the kernel to signal the process when its parent thread exits.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1


def graceful_sigterm() -> None:
    """Make SIGTERM unwind the interpreter (``SystemExit``) instead of killing it outright, so
    ``atexit`` handlers, ``finally`` blocks and transport cleanup run."""

    def _graceful_exit(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)


def die_with_parent(sig: int = signal.SIGTERM) -> bool:
    """Deliver ``sig`` to this process when its parent (the thread that spawned it) exits.
    Linux only (``prctl(PR_SET_PDEATHSIG)``); returns whether it was installed. A parent that
    died before the call is detected through ``getppid() == 1`` and exits immediately."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0) != 0:
            return False
    except OSError:
        return False
    if os.getppid() == 1:  # orphaned in the window before prctl took effect
        os.kill(os.getpid(), sig)
    return True
