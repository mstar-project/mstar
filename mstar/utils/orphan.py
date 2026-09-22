"""Watchdog that leaves once the process that spawned this one is gone."""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time

logger = logging.getLogger(__name__)


def exit_when_orphaned(
    who: str,
    parent_name: str,
    stop_signal: int,
    parent=None,
    poll_s: float = 0.5,
    grace_s: float = 5.0,
) -> None:
    """Exit this process once ``parent`` dies, so nothing outlives its parent
    holding GPU memory or IPC handles.

    Each parent terminates its children on every exit path it controls; this
    covers the one it cannot (killed outright). ``stop_signal`` is the graceful
    stop for this process, but a main thread stuck in a C call cannot service
    it, so exit hard after ``grace_s``.
    """
    parent = mp.parent_process() if parent is None else parent
    if parent is None:
        return
    while parent.is_alive():
        time.sleep(poll_s)
    logger.error("%s: the %s process is gone, exiting", who, parent_name)
    os.kill(os.getpid(), stop_signal)
    time.sleep(grace_s)
    os._exit(1)


def watch_parent(
    who: str, parent_name: str, stop_signal: int = signal.SIGTERM
) -> None:
    """Start :func:`exit_when_orphaned` on a daemon thread."""
    threading.Thread(
        target=exit_when_orphaned,
        args=(who, parent_name, stop_signal),
        daemon=True,
        name="parent-watch",
    ).start()
