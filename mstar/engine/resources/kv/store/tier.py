"""The KV store under one KV cache's host pool: which pages go to the store, and which come back.

Writing: a cached page filed on the host (`HostWrites` landed its copy) is queued, and after each step the
store thread is handed the next few queued pages, one put at a time, and puts the ones the store lacks.

Reading: when a new prompt continues past what the GPU and host indexes hold, the store thread asks how
many of the next pages the store holds. A hit of at least ``min_tokens`` then waits for its turn: one read
at a time takes host pages and the store thread reads them in, so a store that hangs mid-read holds one
read's pages, not every waiting read's. Once they land they're filed in the host index, and the KV cache's
copy from the host brings them to the GPU as it does any host hit. A request whose pages a read already
covers waits on that read. A read that has started lands even if its requests stopped waiting; one still
waiting its turn with nobody waiting on it is dropped, so stale reads never run ahead of wanted ones.

The store threads touch host pages from the CPU, outside the CUDA stream order that keeps the host pool's
copies consistent, so:
- a put reads only pages the host index filed after their copy landed, held from then until the put ends;
- a get starts only once every copy the host pool queued before its pages were taken has finished, since a
  reload releases its host pages with the copy still queued;
- every page a store thread uses stays held until it reports back. A get that returns has stopped its
  transfers (the backend makes sure), so its pages come back then; the pages of a get that raised may still
  be written, and stay held until the store closes.

Every method but `due` and `_call_done` (which the store threads call) runs under the KV cache's lock.
Which streams wait on a read is this module's; the streams' own fields are the KV cache's, which `advance`
reports the finished streams to.
"""

from __future__ import annotations

import enum
import logging
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mstar.engine.resources.kv.store.base import GetResult
from mstar.engine.resources.kv.store.config import MAX_WAIT_S
from mstar.engine.resources.kv.store.io import StoreIO

if TYPE_CHECKING:
    from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, FiledPage
    from mstar.engine.resources.kv.manager import CacheStream

logger = logging.getLogger(__name__)

# One put moves about this many bytes (at least a page, at most `_MAX_PUT_PAGES`), since a page runs from
# tens of KB to tens of MB across models.
_PUT_BYTES = 32 << 20
_MAX_PUT_PAGES = 64
# Queued pages are held, so the bytes a put reads are the ones the copy landed, never a page the host
# evicted and reused. The queue and the put in flight hold at most `write_share` of the host pool, so a slow
# or skipped store can't take the room offloads need. Past it the newest page is dropped, and every page of
# its chain after it: a lookup matches a chain from its start, so the queue avoids storing a tail without it.
# a new prompt doesn't ask while this many lookups wait to start: past it, a slow store would only make each
# new prompt wait out its deadline
_MAX_LOOKUPS_WAITING = 4
# how many dropped keys are remembered, to drop the pages filed under them
_MAX_DROPPED = 4096
_LOG_INTERVAL = 5.0


class Phase(enum.Enum):
    LOOKUP = "lookup"    # the store thread is asking how many of ``keys`` the store holds
    QUEUED = "queued"    # the store holds ``held`` of them; waiting for its turn to read, holding no pages
    COPIES = "copies"    # took host pages; waiting for the host pool's earlier copies before the get
    GETTING = "getting"  # the store thread is reading into ``pages``


# compared by identity, as are the streams waiting on it: two reads of one prompt hold equal fields
@dataclass(slots=True, eq=False)
class StoreRead:
    keys: list[bytes]
    # the key of the page before ``keys[0]`` in the chain, or None at its root
    parent: bytes | None
    # the lookup's in LOOKUP, the get's in GETTING
    future: Future | None
    # when the streams stop waiting for it; it still lands later, for whichever request asks next
    deadline: float
    # when the first request asked, which every deadline counts from
    asked: float = 0.0
    # the streams waiting on it; empty once none does, and the read only has to land or release its pages
    streams: list[CacheStream] = field(default_factory=list)
    phase: Phase = Phase.LOOKUP
    held: int = 0
    pages: list[int] = field(default_factory=list)
    # events for the host pool's copies queued before ``pages`` were taken
    copies: list = field(default_factory=list)


@dataclass(slots=True, eq=False)
class StorePut:
    pages: list[int]
    future: Future


class StoreTier:
    def __init__(self, io: StoreIO, pool: CPUPagePool, tokens_per_page: int):
        self.io = io
        self.pool = pool
        # the fewest pages worth a store lookup or read
        self.min_pages = max(1, -(-io.config.min_tokens // tokens_per_page))
        held = max(2, int(pool.arena.max_num_pages * io.config.write_share))
        self.put_pages = max(1, min(_MAX_PUT_PAGES, _PUT_BYTES // io.page_bytes, held // 2))
        self.max_queued = held - self.put_pages
        # key -> the host page filed under it, held until handed to the store thread, oldest first
        self.queued: OrderedDict[bytes, int] = OrderedDict()
        self._dropped: OrderedDict[bytes, None] = OrderedDict()
        self.pages_dropped = 0
        # totals for the line `close` logs, until metrics report them
        self.pages_put = self.pages_read = self.pages_missing = self.pages_failed = 0
        self._dropped_logged_at: float | None = None
        self.reads: list[StoreRead] = []
        self.put: StorePut | None = None
        # Host pages of gets that raised, which may still be written, held until the store closes. They count
        # against the reads' share, so a backend that keeps raising ends up holding that share and isn't read.
        self.quarantine: list[int] = []
        # What `due` reads, without the lock: set by each store call that finishes, and by each `advance`
        # from the reads it leaves.
        self._changed = False
        self._next_deadline = float("inf")
        self._polling = False
        self._waiting = False
        io.on_finish = self._call_done
        # host pages reads may hold at once, quarantined ones included, as writes are held to a share
        self.max_read_pages = max(1, int(pool.arena.max_num_pages * io.config.read_share))
        self._cut_logged = False

    # ------------------------------------------------------------------ writing

    def filed(self, filed: list[FiledPage]) -> None:
        """Queue pages the host index just filed after their copy landed: what `HostWrites.land` returns, which
        files each chain from its start, handed over by `KVManager._land_host_writes`."""
        dropped = 0
        for key, page, parent in filed:
            if key in self.queued:
                continue
            # the chain's own parent: the host index may have filed the page as a root
            if len(self.queued) >= self.max_queued or parent in self._dropped:
                self._dropped[key] = None
                if len(self._dropped) > _MAX_DROPPED:
                    self._dropped.popitem(last=False)
                dropped += 1
                continue
            self._dropped.pop(key, None)
            self.pool.arena.retain([page])
            self.queued[key] = page
        if dropped:
            self._count_dropped(dropped)

    def _count_dropped(self, dropped: int) -> None:
        self.pages_dropped += dropped
        now = time.monotonic()
        if self._dropped_logged_at is None or now - self._dropped_logged_at >= _LOG_INTERVAL:
            self._dropped_logged_at = now
            logger.warning(
                "KV store: %d cached pages left out of the store so far, as the store falls behind the host's "
                "copies (a full queue drops the newest pages, and the rest of their prompts)", self.pages_dropped,
            )

    def launch_puts(self) -> None:
        """Once the last put has finished, hand the store thread the next few queued pages."""
        put = self.put
        if put is not None:
            if not put.future.done():
                return
            # a put that raised was logged by the store thread; its pages just stay out of the store
            self.pages_put += sum(_result(put.future, []))
            self.pool.arena.release(put.pages)
            self.put = None
        if not self.queued or not self.io.writable():
            return
        keys, pages = [], []
        while self.queued and len(keys) < self.put_pages:
            key, page = self.queued.popitem(last=False)
            keys.append(key)
            pages.append(page)
        # the queue's hold on each page passes to the put
        self.put = StorePut(pages, self.io.put(keys, pages))

    # ------------------------------------------------------------------ reading

    def wants(self, pages: int) -> bool:
        """Whether a read of ``pages`` whole pages past what the GPU and host hold is worth asking about."""
        return pages >= self.min_pages and self.io.available() and self.io.lookups_waiting() < _MAX_LOOKUPS_WAITING

    def due(self, now: float) -> bool:
        """Whether `advance` has anything to do, read without the KV cache's lock and without walking the
        reads, since every waiting row's readiness check asks."""
        return (
            self._changed or now > self._next_deadline or self._polling
            or (self._waiting and not self.io.available())
        )

    def _call_done(self) -> None:
        self._changed = True

    def ask(self, stream: CacheStream, keys: list[bytes], parent: bytes | None, now: float) -> StoreRead:
        """The read ``stream`` waits on for its prompt pages ``keys``, which the GPU and host don't hold:
        one already reading all of them, or a new one."""
        for read in self.reads:
            # a read past its deadline would leave the joiner no wait of its own
            if read.keys[:len(keys)] == keys and now <= read.deadline:
                read.streams.append(stream)
                self._next_deadline = min(self._next_deadline, read.deadline)
                self._waiting = True
                return read
        lookup = self.io.exists(keys)
        read = StoreRead(
            keys, parent, lookup, now + self.io.config.timeout_ms / 1000, streams=[stream], asked=now,
        )
        self.reads.append(read)
        self._next_deadline = min(self._next_deadline, read.deadline)
        self._waiting = True
        return read

    def advance(self, now: float) -> list[tuple[CacheStream, StoreRead]]:
        """Move each read on, and let the next one read once none is. Return the (stream, read) pairs whose
        streams stopped waiting: their read finished, passed its deadline, or the store stopped answering."""
        released: list[tuple[CacheStream, StoreRead]] = []
        self._changed = False
        pending = []
        for read in self.reads:
            if read.phase is Phase.LOOKUP and read.future.done():
                held = _result(read.future, 0)
                # a hit under ``min_tokens`` isn't read, as SGLang's HiCache skips such a prefetch
                if held < self.min_pages or not read.streams or not self.io.available():
                    released += self._release(read)
                    continue
                read.phase, read.held = Phase.QUEUED, min(held, len(read.keys))
                # the request's wait still counts from when it asked, plus time for the bytes, at most MAX_WAIT_S
                read.deadline = min(
                    read.asked + MAX_WAIT_S,
                    read.deadline + read.held * self.io.page_bytes / (self.io.config.floor_gbps * 1e9),
                )
            elif read.phase is Phase.QUEUED and not read.streams:
                continue  # holds no pages; a later request asks again
            elif read.phase is Phase.COPIES:
                self._submit_get(read)
            elif read.phase is Phase.GETTING and read.future.done():
                self._land(read)
                released += self._release(read)
                continue
            if read.streams and (now > read.deadline or not self.io.available()):
                released += self._release(read)
            if read.phase is Phase.LOOKUP and not read.streams and read.future.cancel():
                continue  # nobody waits on it, and the store thread hadn't started it
            pending.append(read)
        self.reads = pending
        if self._reading() is None:
            released += self._start_next(now)
        self._next_deadline = min((read.deadline for read in self.reads if read.streams), default=float("inf"))
        self._polling = any(read.phase is Phase.COPIES for read in self.reads)
        self._waiting = any(read.streams for read in self.reads)
        return released

    def _reading(self) -> StoreRead | None:
        return next((read for read in self.reads if read.phase in (Phase.COPIES, Phase.GETTING)), None)

    def _start_next(self, now: float) -> list[tuple[CacheStream, StoreRead]]:
        """Take host pages for the oldest queued read someone waits on, and start its get. Only now, so a prompt
        the store doesn't hold, and a read still waiting its turn, leave the host's cached pages alone."""
        read = next((read for read in self.reads if read.phase is Phase.QUEUED and read.streams), None)
        if read is None or not self.io.available():
            return []
        # pages another read has put on the host since this one's lookup aren't read again
        found = len(self.pool.index.pages_for(read.keys[:read.held]))
        if found:
            read.parent, read.keys, read.held = read.keys[found - 1], read.keys[found:], read.held - found
        room = self.max_read_pages - len(self.quarantine)
        if read.held > room and not self._cut_logged:
            self._cut_logged = True
            logger.warning(
                "KV store: a hit of %d pages was cut to %d, the reads' share of the host pool (store.read_share "
                "of cpu_offload_pages, less held pages); the rest is recomputed. Logged once.",
                read.held, max(0, room),
            )
        read.held = min(read.held, room)
        if read.held < self.min_pages:
            self.reads.remove(read)
            return self._release(read)
        # short when offloads and holds leave the host pool few pages; a read under the threshold isn't made
        pages = self.pool.take(read.held) or []
        if len(pages) < self.min_pages:
            self.pool.arena.release(pages)
            self.reads.remove(read)
            return self._release(read)
        read.keys = read.keys[:len(pages)]
        read.pages = pages
        read.phase = Phase.COPIES
        read.copies = self.pool.copies_queued()
        self._submit_get(read)
        return []

    def _submit_get(self, read: StoreRead) -> None:
        """Hand the store thread the get once the host pool's earlier copies have finished. Polled here, on the
        worker's thread, rather than waited for on the store's, which has no CUDA device of its own."""
        if all(event.query() for event in read.copies):
            read.copies = []
            read.phase = Phase.GETTING
            read.future = self.io.get(read.keys, read.pages)

    def _land(self, read: StoreRead) -> None:
        results = _result(read.future, None)
        if results is None:
            # a get that raised may still write any of its pages
            self.pages_failed += len(read.pages)
            self.quarantine += read.pages
            read.pages = []
            if self.max_read_pages - len(self.quarantine) < self.min_pages:
                logger.warning(
                    "KV store: %d host pages of reads that raised are held until shutdown, which leaves reads "
                    "too few pages: the store is no longer read", len(self.quarantine),
                )
            return
        arrived = 0
        for result in results:
            if result is not GetResult.ARRIVED:
                break
            arrived += 1
        if arrived:
            # filed under the chain's previous page where the host holds it, else as a root: a lookup walks
            # the keys either way, and only eviction order reads the parent
            parent = None if read.parent is None else self.pool.index.page_for(read.parent)
            self.pool.index.insert_run(read.keys[:arrived], read.pages[:arrived], parent)
        self.pages_read += arrived
        self.pages_missing += sum(result is GetResult.MISSING for result in results)
        self.pages_failed += sum(result is GetResult.FAILED for result in results)
        # the get stopped its transfers before returning, failed ones included
        self.pool.arena.release(read.pages)
        read.pages = []

    @staticmethod
    def _release(read: StoreRead) -> list[tuple[CacheStream, StoreRead]]:
        streams, read.streams = read.streams, []
        return [(stream, read) for stream in streams]

    def detach(self, stream: CacheStream, read: StoreRead) -> None:
        """Stop ``stream`` waiting on ``read``, for a request being removed; the read lands on its own, and a
        lookup nobody waits on any more is cancelled if it hasn't started."""
        # by identity: two waiting streams of one prompt compare equal, and `remove` would take the wrong one
        read.streams = [waiting for waiting in read.streams if waiting is not stream]
        if read.phase is Phase.LOOKUP and not read.streams:
            read.future.cancel()
            self._changed = True

    # ------------------------------------------------------------------ bookkeeping

    @property
    def busy(self) -> bool:
        """Whether there are puts or reads for `launch_puts` and `advance` to move on after a step."""
        return bool(self.queued or self.put is not None or self.reads)

    def host_pages_held(self) -> list[int]:
        """The host pages the queue and the store threads hold, once per reference."""
        return (
            list(self.queued.values())
            + [page for read in self.reads for page in read.pages]
            + ([] if self.put is None else self.put.pages)
            + self.quarantine
        )

    def close(self) -> None:
        # the drop warning is rate-limited, so this is where the full count is sure to appear
        logger.info(
            "KV store: %d pages put (or already held), %d dropped before a put, %d read, %d missing at a read, "
            "%d failed, %d failed calls", self.pages_put, self.pages_dropped, self.pages_read, self.pages_missing,
            self.pages_failed, self.io.errors,
        )
        self.io.close()


def _result(future: Future, failed: Any) -> Any:
    """The future's result, or ``failed`` for one that raised; the store thread logged why."""
    try:
        return future.result()
    except Exception:
        return failed
