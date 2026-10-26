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

from mstar.engine.resources.kv.cache import KVCache, PagedKVConfig
from mstar.engine.resources.kv.page_copier import make_page_copier
from mstar.engine.resources.kv.prefix_index import PrefixIndex

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import PageArena

logger = logging.getLogger(__name__)

# a full pool refuses every eviction attempt, which can be thousands a second
_EXHAUSTED_LOG_INTERVAL = 5.0


@dataclass
class OffloadedStream:
    """One (request, label) stream living on the host."""
    # the first len(keys) pages are filed under ``keys`` in the host index
    cpu_page_indices: list[int]
    stored_len: int
    position: int
    released: int = 0
    protected_prefix: int = 0
    keys: list[bytes] = field(default_factory=list)


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

        # same layout as the device cache, on pinned memory so the copies can
        # be async
        self.cpu_kv_cache = torch.empty(
            config.num_layers,
            arena.allocator.max_num_pages,
            *kv_cache.tensor.shape[2:],  # one page: K/V axis under NHD, latent under MLA
            dtype=kv_cache.dtype,
            device="cpu",
            pin_memory=True,
        )

        # {request_id: {label: OffloadedStream}}
        self.offloaded: dict[str, dict[str, OffloadedStream]] = {}

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

    def take(self, n: int) -> list[int]:
        """Up to ``n`` free pages, evicting cached pages when too few are free."""
        if n > self.arena.num_free:
            self.index.evict(n - self.arena.num_free)
        return self.arena.acquire(min(n, self.arena.num_free))

    def place(
        self,
        rid: str,
        label: str,
        gpu_page_indices: list[int],
        keys: list[bytes],
        stored_len: int,
        position: int,
        released: int = 0,
        protected_prefix: int = 0,
    ) -> PageCopies | None:
        """None when the host can't hold the stream even after evicting cached pages."""
        # retained before evicting, so the eviction passes over them
        found = self.retain_run(keys)
        n_pages = len(gpu_page_indices) - len(found)
        cpu_pages = self.take(n_pages)
        if len(cpu_pages) < n_pages:
            self.arena.release(cpu_pages + found)
            self._log_refusal(rid, label, n_pages)
            return None
        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            cpu_page_indices=found + cpu_pages,
            stored_len=stored_len,
            position=position,
            released=released,
            protected_prefix=protected_prefix,
            keys=list(keys),
        )
        return PageCopies(gpu=gpu_page_indices[len(found):], cpu=cpu_pages)

    def insert_keys(self, rid: str, label: str) -> None:
        """Call this only after the placed stream's copies are queued."""
        state = self.offloaded[rid][label]
        self.index.insert_run(state.keys, state.cpu_page_indices[:len(state.keys)], None)

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
            self.arena.allocator.max_num_pages, self._refused_unlogged,
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

    def copy_out(self, gpu_kv_cache: torch.Tensor, copies: PageCopies) -> None:
        if not copies.gpu:
            return
        stream = self._get_stream()
        # The compute stream may still be running the step that last touched
        # these pages (a commit hands pages on right after the launch). Queue
        # the copies behind it, or they could read a page mid-write.
        stream.wait_stream(torch.cuda.current_stream())
        self._copied_since_write = True
        self._copier.copy(self.cpu_kv_cache, copies.cpu, gpu_kv_cache, copies.gpu, stream)

    def copy_in(self, gpu_kv_cache: torch.Tensor, copies: PageCopies) -> None:
        if not copies.gpu:
            return
        stream = self._get_stream()
        # Same ordering as the offload: the device pages we write may have
        # been released by a commit whose kernels are still reading them.
        stream.wait_stream(torch.cuda.current_stream())
        self._copied_since_write = True
        self._copier.copy(gpu_kv_cache, copies.gpu, self.cpu_kv_cache, copies.cpu, stream)

    def write(self, gpu_kv_cache: torch.Tensor, copies: PageCopies) -> None:
        """Copy cached device pages here on a stream of their own, so `sync` never waits for them. The
        caller has seen the steps that wrote the pages finish, so the write adds no wait on the compute stream."""
        if self._write_stream is None:
            self._write_stream = torch.cuda.Stream()
        # Wait for the copies still queued on `_stream`, because `reload` and `_abandon_claims` release host
        # pages while those copies read or write them.
        if self._copied_since_write:
            self._copied_since_write = False
            self._write_stream.wait_stream(self._stream)
        self._copier.copy(self.cpu_kv_cache, copies.cpu, gpu_kv_cache, copies.gpu, self._write_stream)

    def record_write(self) -> torch.cuda.Event:
        """An event that completes once every write queued so far has landed."""
        event = torch.cuda.Event()
        event.record(self._write_stream)
        return event

    def record_compute(self) -> torch.cuda.Event:
        """An event that completes once the work queued so far on the current stream has run."""
        event = torch.cuda.Event()
        event.record()
        return event

    def record(self) -> torch.cuda.Event:
        """An event that completes once every copy queued so far has landed."""
        event = torch.cuda.Event()
        event.record(self._get_stream())
        return event

    def pop(self, rid: str, label: str) -> OffloadedStream | None:
        state = self.offloaded.get(rid, {}).pop(label, None)
        if state is not None and not self.offloaded[rid]:
            del self.offloaded[rid]
        return state

    def discard(self, rid: str, label: str) -> None:
        """Drop a stream that was copied here but never committed to.

        The offload aborted after the copy, so the device pages stay live and
        these host pages go back to the pool.
        """
        state = self.pop(rid, label)
        if state is not None:
            self.arena.release(state.cpu_page_indices)

    def sync(self) -> None:
        """Order the current stream behind the pending copies.

        Both directions need this before the pages they touched are reused:
        after an offload the device pages go back to the allocator, and after
        a reload the attention kernels read them.
        """
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)

    def remove_request(self, rid: str) -> None:
        for state in self.offloaded.pop(rid, {}).values():
            self.arena.release(state.cpu_page_indices)

    @property
    def num_free_pages(self) -> int:
        return self.arena.num_free
