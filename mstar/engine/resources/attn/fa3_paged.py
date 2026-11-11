"""Paged attention on FlashAttention-3 (the ``flash_attn_3`` wheel) over rows
of one fixed span: each row's ``span`` queries attend to its stored length plus
``kv_extent`` keys, bottom-right causal or all of them. Speculative decoding's
verify step runs on it (`FlashInferManager`, ``verify_backend``).

FA3's paged kernel reads each row's KV length from the device
(``cache_seqlens``), and its split schedule is the kernel's own, so on a
resident step the lengths come straight off the resident tables in the graph:
an asynchronous verify step, planned from upper bounds while the step before
it runs, attends to exactly the context the device holds. A host-planned step
(synchronous mode, or an eager one) stages the same buffers from the KV plan.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

from mstar.platforms import current_platform

try:
    from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache
except Exception as _exc:  # noqa: BLE001
    flash_attn_with_kvcache = None
    _IMPORT_ERROR: str | None = f"{type(_exc).__name__}: {_exc}"
else:
    _IMPORT_ERROR = None

# the head dims the wheel builds FA3 kernels for
_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})


@functools.cache
def fa3_paged_unavailable_reason(
    device: torch.device | None, head_dim: int | None = None, dtype: torch.dtype | None = None,
) -> str | None:
    """Why paged FA3 can't run here, or None if it can. ``device`` None asks
    about the current GPU (the conductor's question)."""
    if _IMPORT_ERROR is not None:
        return f"flash_attn_3 doesn't import ({_IMPORT_ERROR})"
    try:
        from flash_attn_3.flash_attn_config import CONFIG
    except Exception:  # noqa: BLE001
        CONFIG = {}
    if CONFIG.get("build_flags", {}).get("FLASHATTENTION_DISABLE_PAGEDKV"):
        return "flash_attn_3 is built without paged KV"
    capability = current_platform.cuda_capability(device)
    if capability != (9, 0):
        # the wheel's paged kernels are Hopper's
        return f"paged FA3 runs on SM90, not compute capability {capability}"
    if head_dim is not None and head_dim not in _HEAD_DIMS:
        return f"flash_attn_3 has no kernel for head_dim {head_dim}"
    if dtype is not None and dtype not in (torch.bfloat16, torch.float16):
        return f"paged FA3 reads a bf16 or fp16 cache, not {dtype}"
    return None


class FA3PagedWrapper:
    """Paged FA3 over rows of one fixed span, from its own buffers (static
    under a captured graph): a page table ``table_width`` wide (the pool's
    block-table width, `PagedKVConfig.block_table_width`), each row's KV
    length, and the query offsets. The table is that wide in every plan, eager
    or captured, so eager and replayed steps split alike. Each row's ``span``
    queries attend to its stored length plus ``kv_extent`` keys (``span`` by
    default; fewer hides the block's last rows), bottom-right causal or, with
    ``causal`` False, all of them. Takes the calls `FlashInferPrefillWrapper`
    takes from the manager, and main's in-graph `gather_resident`."""

    plans_in_graph = True
    supports_gather_resident = False

    def __init__(
        self, *, page_size: int, table_width: int, device: torch.device, span: int, kv_extent: int | None = None,
        causal: bool = True, batch_size: int | None = None, accelerator_graph: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from mstar.utils.h2d import PinnedStager

        kv_extent = span if kv_extent is None else kv_extent
        if not 0 < kv_extent <= span:
            raise ValueError(f"FA3PagedWrapper: kv_extent {kv_extent} outside a span of {span}")
        self.device = device
        self.page_size = page_size
        self.width = table_width
        self.span = span
        self.kv_extent = kv_extent
        self.causal = causal
        self.accelerator_graph = accelerator_graph
        self.batch_size = batch_size
        # the cache's: a resident step never plans on the host, so this is the one q is cast to
        self.dtype = dtype
        # bound at the first run, so a wrapper builds where the wheel doesn't import
        self._kernel = None
        self._rows = 0
        self._stager = PinnedStager(torch.int32)
        self._page_table = self._seqused = self._qo_indptr_buf = None
        if accelerator_graph:
            assert batch_size is not None, "batch_size required for accelerator graph mode"
            self._rows = batch_size
            self._page_table = torch.zeros(batch_size, self.width, dtype=torch.int32, device=device)
            self._seqused = torch.ones(batch_size, dtype=torch.int32, device=device)
            self._qo_indptr_buf = torch.arange(0, (batch_size + 1) * span, span, dtype=torch.int32, device=device)

    @torch.compiler.disable
    def plan(
        self, qo_indptr: torch.Tensor, paged_kv_indptr: torch.Tensor, paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor, causal: bool = True, dtype: torch.dtype = torch.bfloat16,
        window_left: int = -1, **kwargs,
    ) -> None:
        """A host plan, from the KV plan's arrays (a synchronous or eager step).
        H2D copies only, so it may run beside a live graph."""
        del kwargs, causal  # the mask is fixed by the wrapper
        if window_left >= 0:
            raise ValueError("FA3PagedWrapper plans no sliding window")
        qo = qo_indptr.numpy()
        spans = np.diff(qo)
        rows = len(spans)
        if rows and (spans != self.span).any():
            raise ValueError(f"FA3PagedWrapper plans rows of {self.span} queries; got {spans.tolist()}")
        kvp = paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        if rows and int(npages.max()) > self.width:
            raise ValueError(f"a row of {int(npages.max())} pages, past the page table's {self.width}")
        # the KV plan already counts the whole span: keys past kv_extent come off here
        lens = (
            np.maximum(npages - 1, 0) * self.page_size + paged_kv_last_page_len.numpy() - (self.span - self.kv_extent)
        )
        host = np.zeros((rows, self.width), dtype=np.int32)
        row_of = np.repeat(np.arange(rows), npages)
        host[row_of, np.arange(row_of.shape[0]) - kvp[row_of]] = paged_kv_indices.numpy()
        self.dtype = dtype
        if self.accelerator_graph:
            if rows != self.batch_size:
                raise ValueError(f"FA3PagedWrapper captured for {self.batch_size} rows, planned {rows}")
            # the stager writes flat values through the table's row-major view
            self._stager.copy_(self._page_table, host.reshape(-1))
            self._stager.copy_(self._seqused, lens.astype(np.int32))
            return
        from mstar.utils.h2d import to_device_nonblocking

        self._rows = rows
        self._page_table = to_device_nonblocking(torch.from_numpy(host), self.device)
        self._seqused = to_device_nonblocking(torch.from_numpy(lens.astype(np.int32)), self.device)
        self._qo_indptr_buf = to_device_nonblocking(torch.from_numpy(qo.astype(np.int32)), self.device)

    def gather_resident(
        self, store: torch.Tensor, slots: torch.Tensor, kv_len: torch.Tensor, stride: int, col: int,
    ) -> None:
        """A resident step's page table and lengths off the tables, in the graph:
        each row's block table, ``table_width`` columns from ``col``, which is
        exactly the table, and its device KV length plus ``kv_extent``. A padding
        row reads the trash row, all SINK_PAGE at length 0, and attends to its
        own span."""
        columns = store.view(-1, stride)[:, col: col + self.width]
        self._page_table.copy_(columns.index_select(0, slots))
        self._seqused.copy_((kv_len + self.kv_extent).clamp(min=1))

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer) -> torch.Tensor:
        """``q`` [tokens, qo heads, head dim] over a layer's pages."""
        if isinstance(kv_cache_layer, (tuple, list)):
            k, v = kv_cache_layer
        else:
            k, v = kv_cache_layer[:, 0], kv_cache_layer[:, 1]
        if q.shape[0] != self._rows * self.span:
            # past the plan's rows the kernel would read the wrong queries
            raise ValueError(f"FA3PagedWrapper planned {self._rows} rows of {self.span}, got {q.shape[0]} queries")
        kernel = self._kernel
        if kernel is None:
            if flash_attn_with_kvcache is None:
                raise RuntimeError(f"paged FA3 can't run: flash_attn_3 doesn't import ({_IMPORT_ERROR})")
            kernel = self._kernel = flash_attn_with_kvcache
        return kernel(
            q.to(self.dtype), k, v, cache_seqlens=self._seqused, page_table=self._page_table,
            cu_seqlens_q=self._qo_indptr_buf, max_seqlen_q=self.span, causal=self.causal,
        )
