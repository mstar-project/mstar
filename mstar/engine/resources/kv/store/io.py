"""The store threads: run every call to a KV store backend off the worker's main thread.

``exists``, ``get`` and ``put`` each run on a thread of their own, so a lookup never queues behind a read
and a read never queues behind a put. A call's deadline starts when its thread starts it, so waiting behind
another call doesn't count against the store. While a lookup or a read runs past its deadline the store
isn't asked again. Five failures in a row of one kind of call, or five calls run past their deadlines, skip
reading or writing for 30 s: puts the store refuses (a full store, say) stop writes, not reads.

The threads are daemons: a call to a store that has gone away can block without end, and it mustn't keep
the worker from exiting.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

from mstar.engine.resources.kv.store.base import GetResult, KVStoreBackend, store_key
from mstar.engine.resources.kv.store.config import KVStoreConfig
from mstar.utils.threads import Thread

logger = logging.getLogger(__name__)

_FAILURES_TO_SKIP = 5
_SKIP_SECONDS = 30.0
_LOG_INTERVAL = 5.0


class _Lane:
    """One daemon thread running calls in order, and when the call it is running passes its deadline."""

    def __init__(self, name: str, finished: Callable[[], None]):
        self.overdue_at: float | None = None
        self._finished = finished
        # calls submitted and calls the thread has taken, each counted on one thread only
        self._submitted = 0
        self._taken = 0
        self._calls: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, call: Callable[[], Any]) -> Future:
        future: Future = Future()
        self._submitted += 1
        self._calls.put((future, call))
        return future

    @property
    def waiting(self) -> int:
        """Calls submitted and not yet taken (a cancelled one counts until the thread passes it)."""
        return self._submitted - self._taken

    def close(self) -> None:
        self._calls.put(None)

    def _run(self) -> None:
        while (item := self._calls.get()) is not None:
            future, call = item
            self._taken += 1
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(call())
            except BaseException as e:  # carried by the future, as a pool's worker does
                future.set_exception(e)
            # again after completing: a check that cleared the first mark before the result was set sees this one
            self._finished()


class StoreIO:
    """The store threads of one KV cache. The caller polls the futures it gets back; a future that raises
    is a miss for a read and a skipped page for a write. ``clock`` is `time.monotonic` but in tests."""

    def __init__(
        self, backend: KVStoreBackend, config: KVStoreConfig, layout: str, page_bytes: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend = backend
        self.config = config
        self.layout = layout
        self.page_bytes = page_bytes
        self.clock = clock
        # Called on a store thread once a call has run, before its future completes, so whoever waits on the
        # future finds it already reported (`StoreTier` marks that it has work).
        self.on_finish: Callable[[], None] | None = None
        self._exists = _Lane("kv-store-exists", self._finished)
        self._gets = _Lane("kv-store-get", self._finished)
        self._puts = _Lane("kv-store-put", self._finished)
        self._lock = threading.Lock()
        # failures in a row, per call: a store whose lookups answer but whose reads keep failing is skipped too
        self._failures: dict[str, int] = {}
        # until when reads (lookups and gets) and writes (puts) are skipped
        self._skip_until = {"read": 0.0, "write": 0.0}
        self._logged_at: float | None = None
        self._unlogged = 0
        self.pages_refused = 0
        self.errors = 0

    def available(self) -> bool:
        """Whether to ask the store now: not while it is skipped, nor while a lookup or read runs past its
        deadline."""
        now = self.clock()
        if now < self._skip_until["read"]:
            return False
        return all(overdue is None or now <= overdue for overdue in (self._exists.overdue_at, self._gets.overdue_at))

    def _finished(self) -> None:
        if self.on_finish is not None:
            self.on_finish()

    def lookups_waiting(self) -> int:
        """Lookups submitted and not yet started."""
        return self._exists.waiting

    def writable(self) -> bool:
        return self.clock() >= self._skip_until["write"]

    def names(self, keys: Sequence[bytes]) -> list[str]:
        return [store_key(self.layout, key) for key in keys]

    def exists(self, keys: Sequence[bytes]) -> Future:
        """How many of ``keys``, from the first, the store holds."""
        names = self.names(keys)
        return self._submit(self._exists, "exists", self.config.timeout_ms / 1000,
                            lambda: _leading(self.backend.exists(names)))

    def get(self, keys: Sequence[bytes], pages: list[int]) -> Future:
        """Read ``keys`` into host ``pages``, which no copy still uses. The future gives a `GetResult` per
        page; a get with a failed page counts as a failed call, as one that raises does."""
        names = self.names(keys)
        return self._submit(
            self._gets, "get", self.config.get_timeout(len(pages) * self.page_bytes),
            lambda: self.backend.get(names, pages),
            failed=lambda results: sum(result is GetResult.FAILED for result in results),
        )

    def put(self, keys: Sequence[bytes], pages: list[int]) -> Future:
        """Store host ``pages`` under ``keys``, skipping keys the store already has."""
        names = self.names(keys)

        def run() -> list[bool]:
            held = self.backend.exists(names)
            missing = [i for i, found in enumerate(held) if not found]
            if missing:
                stored = self.backend.put([names[i] for i in missing], [pages[i] for i in missing])
                for i, ok in zip(missing, stored, strict=True):
                    held[i] = ok
                with self._lock:
                    self.pages_refused += len(stored) - sum(stored)
                if not any(stored):
                    # a store with no memory to give, say: it answers but stores nothing
                    raise RuntimeError(f"the store refused all {len(stored)} pages of a put")
            return held

        return self._submit(self._puts, "put", self.config.get_timeout(len(pages) * self.page_bytes), run)

    def _submit(
        self, lane: _Lane, op: str, timeout: float, run: Callable[[], Any],
        failed: Callable[[Any], int] = lambda result: 0,
    ) -> Future:
        def call() -> Any:
            start = self.clock()
            lane.overdue_at = start + timeout
            try:
                result = run()
            except Exception:
                self._failed(op, "failed", exc_info=True)
                raise
            finally:
                lane.overdue_at = None
                self._finished()
            if self.clock() - start > timeout:
                self._failed(op, "ran past its deadline", exc_info=False)
            elif count := failed(result):
                self._failed(op, f"failed for {count} of its keys", exc_info=False)
            else:
                with self._lock:
                    self._failures[op] = 0
            return result

        return lane.submit(call)

    def _failed(self, op: str, what: str, exc_info: bool) -> None:
        now = self.clock()
        with self._lock:
            self.errors += 1
            failures = self._failures[op] = self._failures.get(op, 0) + 1
            skipping = failures >= _FAILURES_TO_SKIP
            if skipping:
                # one short of the count, so the first failure after the skip skips again; a success resets it
                self._failures[op] = _FAILURES_TO_SKIP - 1
                self._skip_until["write" if op == "put" else "read"] = now + _SKIP_SECONDS
            if not skipping and self._logged_at is not None and now - self._logged_at < _LOG_INTERVAL:
                self._unlogged += 1
                return
            unlogged, self._unlogged, self._logged_at = self._unlogged, 0, now
        logger.warning(
            "KV store: a call to %s %s (%d more failures since the last warning)%s",
            op, what, unlogged,
            f"; skipping store {'writes' if op == 'put' else 'reads'} for {_SKIP_SECONDS:.0f} s" if skipping else "",
            exc_info=exc_info,
        )

    def close(self, wait_s: float = 1.0) -> None:
        """Stop the threads once their queued calls finish, waiting at most ``wait_s`` each, since a call to
        a store that has gone away may never return. The backend is closed only once no thread still uses it."""
        lanes = (self._exists, self._gets, self._puts)
        for lane in lanes:
            lane.close()
        for lane in lanes:
            lane.thread.join(wait_s)
        if any(lane.thread.is_alive() for lane in lanes):
            logger.warning("KV store: a call was still running at shutdown, so the store connection is left open")
            return
        self.backend.close()


def _leading(found: Sequence[bool]) -> int:
    """How many of ``found``, from the first, are True."""
    count = 0
    for ok in found:
        if not ok:
            break
        count += 1
    return count
