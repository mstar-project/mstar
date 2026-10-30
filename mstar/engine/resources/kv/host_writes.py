"""Copying cached device pages to the host before the device evicts them.

A page is queued once a release leaves it to the device's prefix index alone. After a step's outputs are
sent, up to a step's share of queued pages is copied to the host, each page's missing ancestors first, so
every page lands on the host under its parent. Eviction takes queued pages last, and cancels the copy of a
page it reuses mid-copy.

Every method but `launch` runs under the manager's lock.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, PageCopies
from mstar.engine.resources.kv.prefix_index import PrefixIndex

logger = logging.getLogger(__name__)

# What a step may queue for the host. The device has one device-to-host copy engine, which runs copies
# in the order they were queued whatever their stream or priority, and the writes queue behind the
# step's own output copy.
# `_WRITE_BYTES_PER_STEP` bounds how long the writes hold the copy engine before the next step's output
# copy can run (about 1 ms on an H100).
# `_WRITE_PAGES_PER_STEP` bounds the copies launched from the host, one per page at most (about 0.1 ms of host
# time at 16).
_WRITE_BYTES_PER_STEP = 32 << 20
_WRITE_PAGES_PER_STEP = 16


@dataclass
class HostWrite:
    """Device ``pages`` being copied to the host's ``host_pages``, not yet seen to land.

    Once ``event`` completes, ``host_pages[i]`` is filed under ``keys[i]``, below the host page of
    ``parents[i]``. A key in ``cancelled`` is not filed: eviction reused its device page during the copy,
    or the copy failed to launch. The write holds ``anchors``, the host pages its root-most pages are filed
    under, so the host can't evict them first.
    """
    keys: list[bytes]
    pages: list[int]
    host_pages: list[int]
    parents: list[bytes | None]
    anchors: list[int]
    event: torch.cuda.Event | None = None
    cancelled: set[bytes] = field(default_factory=set)


class HostWrites:
    def __init__(self, pool: CPUPagePool, index: PrefixIndex, page_bytes: int):
        self._pool = pool
        self._index = index
        self._per_step = min(_WRITE_PAGES_PER_STEP, max(1, _WRITE_BYTES_PER_STEP // page_bytes))
        # Keys waiting for their copy, in release order, each with the commit count when it was queued. A
        # key queued again keeps its place and its count, since its page is sealed and never written again.
        self.queued: OrderedDict[bytes, int] = OrderedDict()
        # Counts the manager's commits. A page queued at count c is final once the step committed at c has
        # run on the device.
        self.commits = 0
        # cached image ends, which no lookup on the host reads, so they are never queued
        self._image_ends: set[bytes] = set()
        self._writing: dict[bytes, HostWrite] = {}
        self._in_flight: list[HostWrite] = []

    def queue(self, key: bytes | None) -> None:
        if key is not None and key not in self._image_ends:
            self.queued.setdefault(key, self.commits)

    def add_image_end(self, key: bytes) -> None:
        self._image_ends.add(key)

    def awaits(self, key: bytes) -> bool:
        """Whether the page under ``key`` is queued or being copied and has no copy on the host yet. A key
        the host already holds can stay queued until `take` skips it, so the host is checked too."""
        return (key in self.queued or key in self._writing) and self._pool.index.page_for(key) is None

    def evicted(self, keys: list[bytes]) -> None:
        """Forget ``keys``, which eviction dropped from the device, and cancel any copy still running."""
        for key in keys:
            self.queued.pop(key, None)
            self._image_ends.discard(key)
            write = self._writing.get(key)
            if write is not None and (write.event is None or not write.event.query()):
                write.cancelled.add(key)

    def take(self, done: int) -> HostWrite | None:
        """Up to a step's share of queued pages to copy, among those queued by commit ``done`` or earlier.
        Each page comes after its ancestors that have no copy on the host and none on the way."""
        keys: list[bytes] = []
        taken: set[bytes] = set()
        anchors: set[int] = set()
        while self.queued and len(keys) < self._per_step:
            key, commits = next(iter(self.queued.items()))
            if commits > done:
                break
            chain, anchor = self._unwritten_chain(key, taken)
            room = self._per_step - len(keys)
            if len(chain) > room:
                # the ancestors that fit go now, and the key waits for a later step
                chain = chain[:room]
            else:
                del self.queued[key]
            if chain and anchor is not None:
                anchors.add(anchor)
            keys += chain
            taken.update(chain)
        if not keys:
            return None
        # held before `take` evicts, so the eviction passes over them
        held = list(anchors)
        self._pool.arena.retain(held)
        host = self._pool.take(len(keys))
        if not host:
            self._pool.arena.release(held)
            return None
        del keys[len(host):]
        pages = [self._index.page_for(key) for key in keys]
        write = HostWrite(
            keys=keys, pages=pages, host_pages=host, parents=[self._index.parent_key(page) for page in pages],
            anchors=held,
        )
        # Listed before the copies launch, so `evicted` cancels the copy of a page another thread evicts and
        # reuses while they launch.
        for key in keys:
            self._writing[key] = write
        self._in_flight.append(write)
        return write

    def _unwritten_chain(self, key: bytes, taken: set[bytes]) -> tuple[list[bytes], int | None]:
        """``key`` and its ancestors, root first, up to the first that is on the host, on its way there, or
        in ``taken``, and the host page of the ancestor the walk stopped at on the host. An ancestor is
        sealed, and was indexed by the commit that indexed ``key`` or an earlier one, so it is final
        whenever ``key`` is."""
        chain = []
        while key is not None and key not in taken:
            host_page = self._pool.index.page_for(key)
            if host_page is not None:
                return chain[::-1], host_page
            write = self._writing.get(key)
            if write is not None and key not in write.cancelled:
                break
            chain.append(key)
            key = self._index.parent_key(self._index.page_for(key))
        return chain[::-1], None

    def launch(self, write: HostWrite, gpu_kv_cache: torch.Tensor, lock) -> None:
        """Queue ``write``'s copies, without the lock. A copy that fails to launch cancels the write, since
        the pages stay cached on the device."""
        try:
            self._pool.write(gpu_kv_cache, PageCopies(gpu=write.pages, cpu=write.host_pages))
            failed = False
        except Exception:
            logger.exception(
                "Copying %d cached pages to the host failed, so they stay on the device only", len(write.keys),
            )
            failed = True
        # recorded even after a failure, since the copies that did launch still read and write these pages
        event = self._pool.record_write()
        with lock:
            write.event = event
            if failed:
                write.cancelled.update(write.keys)

    def land(self) -> None:
        """File each write that has landed on the host."""
        in_flight = []
        for write in self._in_flight:
            if write.event is None or not write.event.query():
                in_flight.append(write)
                continue
            for key in write.keys:
                if self._writing.get(key) is write:
                    del self._writing[key]
            self._pool.file_writes(write.keys, write.host_pages, write.parents, write.cancelled)
            self._pool.arena.release(write.anchors)
        self._in_flight = in_flight

    @property
    def landing(self) -> bool:
        return bool(self._in_flight)

    def host_pages_held(self) -> list[int]:
        """The host pages the writes in flight hold, once per reference."""
        return [page for write in self._in_flight for page in write.host_pages + write.anchors]
