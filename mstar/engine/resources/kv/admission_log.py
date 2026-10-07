"""A JSONL log of what a KV pool decided about admitting requests.

Off unless ``MSTAR_STEP_TELEMETRY_DIR`` is set, and then one bool check where
the pool would log. When on, rows are appended to
``<dir>/admission_<pool>_pid<pid>.jsonl`` through a buffer that is written out
when it fills and at exit, so a decision costs a ``json.dumps`` and not a write.

Events: ``reserve`` (a request was admitted), ``wait`` (it could not be: logged
when the reason changes, not every time it asks again), ``refuse`` (it can
never fit), ``grant_deferred`` (a page grant was held back as unsafe),
``release`` (an admitted request was removed; ``reserved_s`` is how long it
held its reservation), ``kv_release`` (a finished request gave its pages and its
reservation back before it was removed, when the conductor sends RELEASE_KV:
``held`` pages given back, of which ``freed`` went to the free list and the rest
stay cached by the index; ``claim`` and ``reserved_s`` as for ``release``, which
is then not logged for that request).
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time

ENV_DIR = "MSTAR_STEP_TELEMETRY_DIR"
_FLUSH_BYTES = 1 << 16

_writers: dict[str, "AdmissionLog"] = {}
_writers_lock = threading.Lock()


class AdmissionLog:
    def __init__(self, path: str, flush_bytes: int = _FLUSH_BYTES):
        self.path = path
        self._flush_bytes = flush_bytes
        self._lines: list[str] = []
        self._size = 0
        self._lock = threading.Lock()

    def write(self, event: str, pool: str, **fields) -> None:
        row = {
            "ts_wall": time.time(), "ts_mono": time.monotonic(), "pool": pool, "event": event,
        }
        row.update(fields)
        line = json.dumps(row, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            self._lines.append(line)
            self._size += len(line)
            if self._size >= self._flush_bytes:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._lines:
            return
        with open(self.path, "a") as out:
            out.writelines(self._lines)
        self._lines.clear()
        self._size = 0


def open_log(pool: str) -> AdmissionLog | None:
    """The log for ``pool`` in this process, or None when telemetry is off.

    One writer per file however many pools of the same name are built, so
    their rows are not interleaved mid-buffer.
    """
    directory = os.environ.get(ENV_DIR)
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", pool)
    path = os.path.join(directory, f"admission_{safe}_pid{os.getpid()}.jsonl")
    with _writers_lock:
        writer = _writers.get(path)
        if writer is None:
            writer = _writers[path] = AdmissionLog(path)
        return writer


def flush_all() -> None:
    with _writers_lock:
        writers = list(_writers.values())
    for writer in writers:
        writer.flush()


atexit.register(flush_all)
