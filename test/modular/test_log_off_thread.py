"""The API server and conductor log on every request, and a write to a stalled
disk must hold only the thread that writes, not the loop that logged."""

from __future__ import annotations

import atexit
import logging
import threading
import time

from mstar.utils.logging_config import log_off_thread


class _Stalled(logging.Handler):
    """A handler whose writes wait until the test lets them through."""

    def __init__(self):
        super().__init__()
        self.go = threading.Event()
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record):
        self.go.wait(5)
        self.lines.append(self.format(record))


def test_a_stalled_write_holds_the_writer_not_the_caller():
    root = logging.getLogger()
    saved = root.handlers[:]
    stalled = _Stalled()
    root.handlers[:] = [stalled]
    drain = log_off_thread()
    try:
        log = logging.getLogger("mstar.test")
        t0 = time.perf_counter()
        log.warning("request %s done", "r1")
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("request %s failed", "r2")
        assert time.perf_counter() - t0 < 1
        assert stalled.lines == []

        stalled.go.set()
        drain()
        assert stalled.lines[0] == "WARNING mstar.test: request r1 done"
        assert stalled.lines[1].startswith("ERROR mstar.test: request r2 failed")
        assert "ValueError: boom" in stalled.lines[1]
        # the drain puts the handler back, so what's logged after it writes directly
        assert root.handlers == [stalled]
        log.warning("after")
        assert stalled.lines[2] == "WARNING mstar.test: after"
    finally:
        stalled.go.set()
        atexit.unregister(drain)
        root.handlers[:] = saved
