"""Host-side page pool for KV cache offloading.

When device pages run out, a cold request's pages move here, into pinned host
memory, freeing device pages for the requests that are actually running. They
move back when it is scheduled again.

The pool stores pages and the stream metadata that has to survive the round
trip; which request to evict, and when, is the worker's decision.
"""

import logging
import time
from dataclasses import dataclass

import torch

from mstar.engine.resources.kv.cache import KVCache, PageAllocator, PagedKVConfig
from mstar.engine.resources.kv.page_copier import make_page_copier

logger = logging.getLogger(__name__)

# a full pool refuses every eviction attempt, which can be thousands a second
_EXHAUSTED_LOG_INTERVAL = 5.0


@dataclass
class OffloadedStream:
    """One (request, label) stream living on the host."""
    cpu_page_indices: list[int]
    stored_len: int
    position: int
    released: int = 0
    protected_prefix: int = 0


class CPUPagePool:
    """Host mirror of the paged KV cache, holding whatever is offloaded."""

    def __init__(
        self,
        config: PagedKVConfig,
        kv_cache: KVCache,
        max_cpu_pages: int,
    ):
        self.config = config
        self.page_allocator = PageAllocator(max_cpu_pages)

        # same layout as the device cache, on pinned memory so the copies can
        # be async
        self.cpu_kv_cache = torch.zeros(
            config.num_layers,
            max_cpu_pages,
            *kv_cache.tensor.shape[2:],  # one page: K/V axis under NHD, latent under MLA
            dtype=kv_cache.dtype,
            device="cpu",
        ).pin_memory()

        # {request_id: {label: OffloadedStream}}
        self.offloaded: dict[str, dict[str, OffloadedStream]] = {}

        # copies run here so they don't serialize with compute
        self._stream: torch.cuda.Stream | None = None
        self._copier = make_page_copier()

        self._refused_logged_at: float | None = None
        self._refused_unlogged = 0

    def _get_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def is_offloaded(self, rid: str) -> bool:
        return bool(self.offloaded.get(rid))

    def labels(self, rid: str) -> list[str]:
        return list(self.offloaded.get(rid, {}))

    def offload_stream(
        self,
        rid: str,
        label: str,
        gpu_kv_cache: torch.Tensor,
        gpu_page_indices: list[int],
        stored_len: int,
        position: int,
        released: int = 0,
        protected_prefix: int = 0,
    ) -> bool:
        """Copy device pages to the host. False when the host pool is full,
        in which case nothing moved and the caller keeps its device pages."""
        n_pages = len(gpu_page_indices)
        if n_pages == 0:
            return False

        cpu_pages = self.page_allocator.try_allocate(n_pages)
        if cpu_pages is None:
            self._log_refusal(rid, label, n_pages)
            return False

        stream = self._get_stream()
        # The compute stream may still be running the step that last touched
        # these pages (a commit hands pages on right after the launch). Queue
        # the copies behind it, or they could read a page mid-write.
        stream.wait_stream(torch.cuda.current_stream())
        self._copier.copy(
            self.cpu_kv_cache, cpu_pages, gpu_kv_cache, gpu_page_indices, stream
        )

        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            cpu_page_indices=cpu_pages,
            stored_len=stored_len,
            position=position,
            released=released,
            protected_prefix=protected_prefix,
        )
        return True

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
            n_pages, rid, label, self.page_allocator.num_free,
            self.cpu_kv_cache.shape[1], self._refused_unlogged,
        )
        self._refused_logged_at = now
        self._refused_unlogged = 0

    def reload_stream(
        self,
        rid: str,
        label: str,
        gpu_kv_cache: torch.Tensor,
        gpu_page_indices: list[int],
    ) -> OffloadedStream:
        """Copy the host pages back onto ``gpu_page_indices`` and release them.

        Returns the stream metadata saved at offload, for the caller to
        restore.
        """
        state = self.offloaded[rid].pop(label)
        if not self.offloaded[rid]:
            del self.offloaded[rid]

        stream = self._get_stream()
        # Same ordering as the offload: the device pages we write may have
        # been released by a commit whose kernels are still reading them.
        stream.wait_stream(torch.cuda.current_stream())
        self._copier.copy(
            gpu_kv_cache, gpu_page_indices,
            self.cpu_kv_cache, state.cpu_page_indices, stream,
        )

        self.page_allocator.free(state.cpu_page_indices)
        return state

    def discard(self, rid: str, label: str) -> None:
        """Drop a stream that was copied here but never committed to.

        The offload aborted after the copy, so the device pages stay live and
        these host pages go back to the pool.
        """
        state = self.offloaded.get(rid, {}).pop(label, None)
        if state is None:
            return
        if not self.offloaded[rid]:
            del self.offloaded[rid]
        self.page_allocator.free(state.cpu_page_indices)

    def num_pages(self, rid: str, label: str) -> int:
        state = self.offloaded.get(rid, {}).get(label)
        return 0 if state is None else len(state.cpu_page_indices)

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
            self.page_allocator.free(state.cpu_page_indices)

    @property
    def num_free_pages(self) -> int:
        return self.page_allocator.num_free
