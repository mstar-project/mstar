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
    from mstar.engine.resources.kv.manager import PageArena

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

        # same layout as the device cache, on pinned memory so the copies can
        # be async
        self.cpu_kv_cache = torch.empty(
            kv_cache.tensor.shape[0],
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
        self._copier = make_page_copier()

        self._refused_logged_at: float | None = None
        self._refused_unlogged = 0
        self.dropped = 0
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
        cpu_pages = self.arena.acquire(n_pages)
        if cpu_pages is None:
            self.index.evict(n_pages - self.arena.num_free)
            cpu_pages = self.arena.acquire(n_pages)
        if cpu_pages is None:
            self.arena.release(found)
            self._log_refusal(rid, label, n_pages)
            return None
        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            groups=on_pages(groups, found + cpu_pages),
            position=position,
            protected_prefix=protected_prefix,
            keys=list(keys),
        )
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
            self.arena.allocator.max_num_pages, self._refused_unlogged,
        )
        self._refused_logged_at = now
        self._refused_unlogged = 0

    def count_dropped(self, keys: list[bytes]) -> None:
        dropped = sum(self.index.page_for(key) is None for key in keys)
        if not dropped:
            return
        self.dropped += dropped
        now = time.monotonic()
        if self._dropped_logged_at is None or now - self._dropped_logged_at >= _EXHAUSTED_LOG_INTERVAL:
            logger.info(
                "CPU page pool: %d cached device pages evicted with no copy here so far", self.dropped,
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
        self._copier.copy(self.cpu_kv_cache, copies.cpu, gpu_kv_cache, copies.gpu, stream)

    def copy_in(self, gpu_kv_cache: torch.Tensor, copies: PageCopies) -> None:
        if not copies.gpu:
            return
        stream = self._get_stream()
        # Same ordering as the offload: the device pages we write may have
        # been released by a commit whose kernels are still reading them.
        stream.wait_stream(torch.cuda.current_stream())
        self._copier.copy(gpu_kv_cache, copies.gpu, self.cpu_kv_cache, copies.cpu, stream)

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
            self.arena.release(state.host_pages)

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
            self.arena.release(state.host_pages)

    @property
    def num_free_pages(self) -> int:
        return self.arena.num_free
