"""Lightweight, env-gated timing prints shared by the API server and data worker.

Enabled with ``MSTAR_TIMING=1`` (anything other than unset/``0``/``false``).
``perf_counter`` is process-wide monotonic, so timestamps stamped in the
API-server handler thread and read in the data-worker thread are directly
comparable — that's how queue-wait (polling) latency is separated from actual
work in the [API-TIMING]/[DW-TIMING] brackets.
"""
import os

TIMING_ENABLED = os.environ.get("MSTAR_TIMING", "") not in ("", "0", "false")


def make_tlog(prefix: str):
    """Return a ``tlog(msg)`` that prints ``[<prefix>] <msg>`` when enabled."""
    def _tlog(msg: str) -> None:
        if TIMING_ENABLED:
            print(f"[{prefix}] {msg}", flush=True)

    return _tlog
