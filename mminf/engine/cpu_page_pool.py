"""CPU-side page pool for KV cache offloading.

When GPU pages are exhausted, cold requests' KV cache pages can be swapped to
CPU pinned memory here, freeing GPU pages for active requests. Pages are
reloaded back to GPU when the request is re-scheduled.
"""
import logging
from dataclasses import dataclass

import torch

from mminf.engine.kv_store import KVCacheConfig, PageAllocator

logger = logging.getLogger(__name__)


@dataclass
class OffloadedState:
    """Tracks a single (request, label) that has been offloaded to CPU."""
    cpu_page_indices: list[int]
    seq_len: int
    position_id_start: int


class CPUPagePool:
    """CPU-side mirror of the GPU paged KV cache for offloading."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_cpu_pages: int,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = kv_cache_config
        self.page_allocator = PageAllocator(max_cpu_pages)

        # Same shape as the GPU KV cache but on pinned CPU memory
        self.cpu_kv_cache = torch.zeros(
            kv_cache_config.num_layers,
            max_cpu_pages,
            2,  # K and V
            kv_cache_config.page_size,
            kv_cache_config.num_kv_heads,
            kv_cache_config.head_dim,
            dtype=kv_cache_dtype,
            device="cpu",
        ).pin_memory()

        # {request_id: {label: OffloadedState}}
        self.offloaded: dict[str, dict[str, OffloadedState]] = {}

        # Dedicated CUDA stream for async GPU↔CPU copies
        self._stream: torch.cuda.Stream | None = None

    def _get_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def is_offloaded(self, request_id: str) -> bool:
        return request_id in self.offloaded and len(self.offloaded[request_id]) > 0

    def offload_pages(
        self,
        request_id: str,
        label: str,
        gpu_kv_cache: torch.Tensor,
        gpu_page_indices: list[int],
        seq_len: int,
        position_id_start: int,
    ) -> None:
        """Copy GPU pages → CPU pages (async on dedicated stream)."""
        n_pages = len(gpu_page_indices)
        if n_pages == 0:
            return

        cpu_pages = self.page_allocator.try_allocate(n_pages)
        if cpu_pages is None:
            logger.warning(
                "CPU page pool exhausted: cannot offload %d pages for %s/%s",
                n_pages, request_id, label,
            )
            return

        stream = self._get_stream()
        with torch.cuda.stream(stream):
            for gpu_idx, cpu_idx in zip(gpu_page_indices, cpu_pages, strict=True):
                # Copy all layers at once: cpu[:, cpu_idx] = gpu[:, gpu_idx]
                self.cpu_kv_cache[:, cpu_idx].copy_(
                    gpu_kv_cache[:, gpu_idx], non_blocking=True
                )

        self.offloaded.setdefault(request_id, {})[label] = OffloadedState(
            cpu_page_indices=cpu_pages,
            seq_len=seq_len,
            position_id_start=position_id_start,
        )

    def reload_pages(
        self,
        request_id: str,
        label: str,
        gpu_kv_cache: torch.Tensor,
        gpu_page_indices: list[int],
    ) -> tuple[int, int]:
        """Copy CPU pages → GPU pages (async), free CPU pages.

        Returns (seq_len, position_id_start) that were saved during offload.
        """
        state = self.offloaded[request_id][label]

        stream = self._get_stream()
        with torch.cuda.stream(stream):
            for cpu_idx, gpu_idx in zip(state.cpu_page_indices, gpu_page_indices, strict=True):
                gpu_kv_cache[:, gpu_idx].copy_(
                    self.cpu_kv_cache[:, cpu_idx], non_blocking=True
                )

        self.page_allocator.free(state.cpu_page_indices)
        seq_len, pos_id = state.seq_len, state.position_id_start
        del self.offloaded[request_id][label]
        if not self.offloaded[request_id]:
            del self.offloaded[request_id]
        return seq_len, pos_id

    def sync(self) -> None:
        """Wait for all pending GPU↔CPU copies to complete."""
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)

    def remove_request(self, request_id: str) -> None:
        """Free any CPU pages held by this request (e.g., on request removal)."""
        if request_id not in self.offloaded:
            return
        for _label, state in self.offloaded[request_id].items():
            self.page_allocator.free(state.cpu_page_indices)
        del self.offloaded[request_id]

    @property
    def num_free_pages(self) -> int:
        return self.page_allocator.num_free
