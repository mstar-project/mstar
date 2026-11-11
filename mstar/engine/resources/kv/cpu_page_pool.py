"""Host-side page pool for KV cache offloading.

When device pages run out, a cold request's pages move here, into pinned host
memory, freeing device pages for the requests that are actually running. They
move back when it is scheduled again.

Pages the prefix index names are filed here under the same keys, so a prefix
that several requests share is stored once.

The pool stores pages and the stream metadata that has to survive the round
trip; which request to evict, and when, is the worker's decision.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.kv.cache import GroupPages, KVCache, PagedKVConfig, held_pages, on_pages
from mstar.engine.resources.kv.page_copier import make_page_copier
from mstar.engine.resources.kv.prefix_index import PrefixIndex

if TYPE_CHECKING:
    from mstar.engine.resources.kv.arena import PageArena

logger = logging.getLogger(__name__)

# a full pool refuses every eviction attempt, which can be thousands a second
_EXHAUSTED_LOG_INTERVAL = 5.0


@dataclass
class OffloadedStream:
    """One (request, label) stream living on the host."""
    # its pages and lengths in each group (`CacheStream.groups`), on host pages; the first
    # group's first len(keys) pages are filed under ``keys`` in the host index
    groups: tuple[GroupPages, ...]
    position: int
    protected_prefix: int = 0
    keys: list[bytes] = field(default_factory=list)

    @property
    def host_pages(self) -> list[int]:
        """Every host page the record holds, group by group."""
        return held_pages(self.groups) if len(self.groups) > 1 else self.groups[0].page_indices


class PageCopies(NamedTuple):
    gpu: list[int]
    cpu: list[int]


class CPUPagePool:
    """Host mirror of the paged KV cache, holding whatever is offloaded.

    ``arena`` and ``index`` take no lock of their own, so every call that
    changes them holds the manager's lock.
    """

    def __init__(
        self,
        config: PagedKVConfig,
        kv_cache: KVCache,
        arena: "PageArena",
    ):
        self.config = config
        self.arena = arena
        self.index = PrefixIndex(arena)
        # the arena reports owner changes to the index, so `room` reads the evictable set
        arena.index = self.index

        # same layout as the device cache, on pinned memory so the copies can
        # be async: one tensor per storage (`KVCache.storages`), the primary first
        self.cpu_storages = [
            torch.empty(
                tensor.shape[0],
                arena.max_num_pages,
                *tensor.shape[2:],  # one page: K/V axis under NHD, latent under MLA
                dtype=kv_cache.dtype,
                device="cpu",
                pin_memory=True,
            )
            for tensor in kv_cache.storages
        ]
        self.cpu_kv_cache = self.cpu_storages[0]

        # {request_id: {label: OffloadedStream}}
        self.offloaded: dict[str, dict[str, OffloadedStream]] = {}
        # request -> the pages set aside for its move here, for a request that no drop can free
        # `_unmoved` sums the pages set aside that no request has moved into yet.
        self.reserved: dict[str, int] = {}
        self._unmoved = 0

        # copies run here so they don't serialize with compute
        self._stream: torch.cuda.Stream | None = None
        self._write_stream: torch.cuda.Stream | None = None
        self._copied_since_write = False
        self._copier = make_page_copier()

        self._refused_logged_at: float | None = None
        self._refused_unlogged = 0
        self.dropped = 0
        self.written = 0
        self.cancelled = 0
        self._dropped_logged_at: float | None = None
        self.guaranteed = 0
        self._guaranteed_logged_at: float | None = None

    def _paired(self, gpu) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each host storage with its device one: ``gpu`` is the `KVCache`, or a pool's one tensor."""
        devices = getattr(gpu, "storages", None) or [gpu]
        return list(zip(self.cpu_storages, devices, strict=True))

    def _get_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def is_offloaded(self, rid: str) -> bool:
        return bool(self.offloaded.get(rid))

    def retain_run(self, keys: list[bytes]) -> list[int]:
        pages = self.index.lookup(keys)
        self.arena.retain(pages)
        return pages

    def reserve(self, rid: str, pages: int) -> bool:
        """Set ``pages`` aside for ``rid``'s move here, in place of what was set aside for it, or return False when
        the room isn't there."""
        if pages > self.arena.num_free + len(self.index.evictable()) - self._set_aside(rid):
            return False
        unmoved = self._unmoved_by(rid)
        self.reserved[rid] = pages
        self._unmoved += self._unmoved_by(rid) - unmoved
        self._keep_free(0)
        return True

    def unreserve(self, rid: str) -> None:
        self._unmoved -= self._unmoved_by(rid)
        self.reserved.pop(rid, None)

    def count_guaranteed(self, rid: str) -> None:
        """Log that the pool had no room to set aside for ``rid``'s move, so ``rid`` reserves its whole footprint on
        the device."""
        self.guaranteed += 1
        now = time.monotonic()
        if self._guaranteed_logged_at is None or now - self._guaranteed_logged_at >= _EXHAUSTED_LOG_INTERVAL:
            logger.warning(
                "CPU page pool full: no room to set aside for %s's move here, so it reserves its whole KV footprint "
                "on the device (%d requests so far)", rid, self.guaranteed,
            )
            self._guaranteed_logged_at = now

    def _set_aside(self, rid: str | None) -> int:
        """The reserved pages no request has moved into yet, other than ``rid``'s."""
        return self._unmoved - (0 if rid is None else self._unmoved_by(rid))

    def _unmoved_by(self, rid: str) -> int:
        pages = self.reserved.get(rid)
        if pages is None:
            return 0
        return max(0, pages - sum(len(state.host_pages) for state in self.offloaded.get(rid, {}).values()))

    def _keep_free(self, n: int, rid: str | None = None) -> int:
        """Evict cached pages until ``n`` pages are free past the pages set aside for requests other than ``rid``, and
        return those. What is set aside stays free, not cached, since a host fetch could hold a cached page by the
        time its request moves here."""
        kept = self._set_aside(rid)
        if n + kept > self.arena.num_free:
            self.index.evict(n + kept - self.arena.num_free)
        return kept

    def take(self, n: int, rid: str | None = None) -> list[int]:
        """Up to ``n`` free pages, evicting cached pages when too few are free. The pages set aside for requests
        other than ``rid`` stay free."""
        kept = self._keep_free(n, rid)
        return self.arena.acquire(max(0, min(n, self.arena.num_free - kept)))

    def place(
        self,
        rid: str,
        label: str,
        groups: tuple[GroupPages, ...],
        keys: list[bytes],
        position: int,
        protected_prefix: int = 0,
    ) -> PageCopies | None:
        """None when the host can't hold the stream even after evicting cached pages. ``groups``
        are a copy of the stream's (`CacheStream.groups`), on its device pages."""
        # retained before evicting, so the eviction passes over them
        found = self.retain_run(keys)
        gpu_pages = held_pages(groups)
        n_pages = len(gpu_pages) - len(found)
        cpu_pages = self.take(n_pages, rid)
        if len(cpu_pages) < n_pages:
            self.arena.release(cpu_pages + found)
            self._log_refusal(rid, label, n_pages)
            return None
        unmoved = self._unmoved_by(rid)
        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            groups=on_pages(groups, found + cpu_pages),
            position=position,
            protected_prefix=protected_prefix,
            keys=list(keys),
        )
        self._unmoved += self._unmoved_by(rid) - unmoved
        return PageCopies(gpu=gpu_pages[len(found):], cpu=cpu_pages)

    def insert_keys(self, rid: str, label: str) -> None:
        """Call this only after the placed stream's copies are queued."""
        state = self.offloaded[rid][label]
        self.index.insert_run(state.keys, state.groups[0].page_indices[:len(state.keys)], None)

    def _log_refusal(self, rid: str, label: str, n_pages: int) -> None:
        now = time.monotonic()
        last = self._refused_logged_at
        if last is not None and now - last < _EXHAUSTED_LOG_INTERVAL:
            self._refused_unlogged += 1
            return
        logger.warning(
            "CPU page pool exhausted: cannot offload %d pages for %s/%s "
            "(%d of %d pages free; %d more offloads refused since the last "
            "warning)",
            n_pages, rid, label, self.arena.num_free,
            self.arena.max_num_pages, self._refused_unlogged,
        )
        self._refused_logged_at = now
        self._refused_unlogged = 0

    def count_dropped(self, keys: list[bytes]) -> None:
        dropped = sum(self.index.page_for(key) is None for key in keys)
        if dropped:
            self.dropped += dropped
            self._log_counts()

    def file_writes(
        self, keys: list[bytes], pages: list[int], parents: list[bytes | None], cancelled: set[bytes],
    ) -> None:
        """File each of ``pages`` under its key in ``keys``, except the keys in ``cancelled``, then
        release ``pages``. Where an offload filed a key during the copy, the index keeps the offload's
        page, and this write's page goes back to the free list."""
        filed = 0
        for key, page, parent in zip(keys, pages, parents, strict=True):
            if key not in cancelled:
                filed += self.index.insert(key, page, None if parent is None else self.index.page_for(parent))
        self.arena.release(pages)
        self.written += filed
        self.cancelled += len(cancelled)
        self._log_counts()

    def _log_counts(self) -> None:
        now = time.monotonic()
        if self._dropped_logged_at is None or now - self._dropped_logged_at >= _EXHAUSTED_LOG_INTERVAL:
            logger.info(
                "CPU page pool: %d cached device pages evicted with no copy here so far, %d cached "
                "pages copied here, %d copies cancelled",
                self.dropped, self.written, self.cancelled,
            )
            self._dropped_logged_at = now

    def copy_out(self, gpu_kv_cache: KVCache | torch.Tensor, copies: PageCopies) -> None:
        if not copies.gpu:
            return
        stream = self._get_stream()
        # The compute stream may still be running the step that last touched
        # these pages (a commit hands pages on right after the launch). Queue
        # the copies behind it, or they could read a page mid-write.
        stream.wait_stream(torch.cuda.current_stream())
        self._copied_since_write = True
        for host, device in self._paired(gpu_kv_cache):
            self._copier.copy(host, copies.cpu, device, copies.gpu, stream)

    def copy_in(self, gpu_kv_cache: KVCache | torch.Tensor, copies: PageCopies) -> None:
        if not copies.gpu:
            return
        stream = self._get_stream()
        # Same ordering as the offload: the device pages we write may have
        # been released by a commit whose kernels are still reading them.
        stream.wait_stream(torch.cuda.current_stream())
        self._copied_since_write = True
        for host, device in self._paired(gpu_kv_cache):
            self._copier.copy(device, copies.gpu, host, copies.cpu, stream)

    def _get_write_stream(self) -> torch.cuda.Stream:
        if self._write_stream is None:
            self._write_stream = torch.cuda.Stream()
        return self._write_stream

    def write(self, gpu_kv_cache: KVCache | torch.Tensor, copies: PageCopies) -> None:
        """Copy cached device pages here on a stream of their own, so `sync` never waits for them. The
        caller has seen the steps that wrote the pages finish, so the write adds no wait on the compute stream."""
        stream = self._get_write_stream()
        # Wait for the copies still queued on `_stream`, because `reload` and `_abandon_claims` release host
        # pages while those copies read or write them.
        if self._copied_since_write:
            self._copied_since_write = False
            stream.wait_stream(self._stream)
        for host, device in self._paired(gpu_kv_cache):
            self._copier.copy(host, copies.cpu, device, copies.gpu, stream)

    def record_write(self) -> torch.cuda.Event:
        """An event that completes once every write queued so far has landed."""
        event = torch.cuda.Event()
        event.record(self._get_write_stream())
        return event

    def record(self) -> torch.cuda.Event:
        """An event that completes once every copy queued so far has landed."""
        event = torch.cuda.Event()
        event.record(self._get_stream())
        return event

    def pop(self, rid: str, label: str) -> OffloadedStream | None:
        unmoved = self._unmoved_by(rid)
        state = self.offloaded.get(rid, {}).pop(label, None)
        if state is not None and not self.offloaded[rid]:
            del self.offloaded[rid]
        self._unmoved += self._unmoved_by(rid) - unmoved
        return state

    def discard(self, rid: str, label: str) -> None:
        """Release the host pages of a stream that was reloaded, or copied here by an offload that aborted."""
        state = self.pop(rid, label)
        if state is not None:
            self.arena.release(state.host_pages)
            # the room set aside for rid grows back by every page, and a keyed page goes to the index, not the free list
            self._keep_free(0)

    def sync(self) -> None:
        """Order the current stream behind the pending copies.

        Both directions need this before the pages they touched are reused:
        after an offload the device pages go back to the allocator, and after
        a reload the attention kernels read them.
        """
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)

    def remove_request(self, rid: str) -> None:
        self.unreserve(rid)
        for state in self.offloaded.pop(rid, {}).values():
            self.arena.release(state.host_pages)

    @property
    def num_free_pages(self) -> int:
        return self.arena.num_free

    def room(self) -> int:
        """Free pages plus the cached pages `place` can evict, less the pages set aside for requests."""
        return self.arena.num_free + len(self.index.evictable()) - self._set_aside(None)

    def takes(self, pages: int, keys: list[bytes]) -> int:
        """How many pages `place` takes out of `room` for a stream ``pages`` long, filed under ``keys``. A page
        another request holds under the same key costs nothing."""
        return pages - sum(owners > 1 for owners in self.arena.owners_of(self.index.pages_for(keys)))
