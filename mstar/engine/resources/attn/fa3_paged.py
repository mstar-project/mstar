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

from mstar.engine.resources.attn.ragged.wrappers import zero_unplanned_rows
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
    ``causal`` False, all of them. ``plan_span``: the KV plan's rows a request,
    when it counts more than the wrapper's queries (a lagged drafter's pass,
    whose extent may pass its own rows); ``span`` by default. Takes the calls
    `FlashInferPrefillWrapper` takes from the manager, and main's in-graph
    `gather_resident`."""

    plans_in_graph = True
    supports_gather_resident = False
    # a verify walk never repeats within a launch (`Engine.repeat_refusal`)
    supports_repeats = False

    def __init__(
        self, *, page_size: int, table_width: int, device: torch.device, span: int, kv_extent: int | None = None,
        causal: bool = True, batch_size: int | None = None, accelerator_graph: bool = False,
        dtype: torch.dtype = torch.bfloat16, plan_span: int | None = None,
    ):
        from mstar.utils.h2d import PinnedStager

        plan_span = span if plan_span is None else plan_span
        kv_extent = span if kv_extent is None else kv_extent
        # a lagged pass's queries attend past their own rows, never past the plan's
        if not 0 < kv_extent <= plan_span:
            raise ValueError(f"FA3PagedWrapper: kv_extent {kv_extent} outside a plan span of {plan_span}")
        self.device = device
        self.page_size = page_size
        self.width = table_width
        self.span = span
        self.kv_extent = kv_extent
        # the KV plan's rows a request: more than the queries a lagged pass runs (`Lag`)
        self.plan_span = plan_span
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
        spans = np.diff(qo_indptr.numpy())
        rows = len(spans)
        if rows and (spans != self.plan_span).any():
            raise ValueError(f"FA3PagedWrapper plans rows of {self.plan_span} queries; got {spans.tolist()}")
        kvp = paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        if rows and int(npages.max()) > self.width:
            raise ValueError(f"a row of {int(npages.max())} pages, past the page table's {self.width}")
        # the KV plan counts the whole plan span: keys past kv_extent come off here
        lens = (
            np.maximum(npages - 1, 0) * self.page_size + paged_kv_last_page_len.numpy()
            - self.plan_span + self.kv_extent
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
        qo = np.arange(rows + 1, dtype=np.int32) * self.span
        self._qo_indptr_buf = to_device_nonblocking(torch.from_numpy(qo), self.device)

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


# Pages of this many tokens or fewer keep FA3's paged kernel inside each row's length. On larger pages, flash_attn_3
# 3.0.1 (an observed property of the wheel, not a documented contract) reads a row's last page past its length once
# the step's longest row has about 64 queries, and masks those scores to zero probability: 0 times a stale NaN or Inf
# left there by an earlier owner's broken forward still poisons the row. Decode and verify rows (8 queries at most)
# never take that path; a mixed step's prompt row can.
FA3_TAIL_SAFE_PAGE_SIZE = 32


class FA3RaggedWrapper:
    """Paged FA3 over rows of any span, each bottom-right causal or, with ``causal`` False, attending all its
    keys: a mixed step's verify rows of ``k + 1`` queries and its prompt rows (`MixedRows`), from the KV plan's
    arrays as FlashInfer's prefill wrapper takes them. The caller hides each verify row's last keys in those
    arrays (`hide_tail_views`), so a row attends to exactly its arrays' length. Planned on the host. Under a
    captured graph its buffers are static: ``batch_size`` rows of a ``table_width`` page table, query offsets,
    and up to ``max_tokens`` queries, of which those past the rows' (a replay's padding) come out zero."""

    plans_in_graph = False
    supports_gather_resident = False
    supports_repeats = False
    # Pinned, not FA3's heuristics: those read max_seqlen_q, which a capture fixes at its bucket's tokens and an
    # eager step takes from its rows, so the two would split the keys and pack the heads differently and reduce
    # in a different order. One split, query heads packed per KV head (the verify rows' shape: few queries a row).
    NUM_SPLITS = 1
    PACK_GQA = True

    def __init__(
        self, *, page_size: int, table_width: int, device: torch.device, batch_size: int | None = None,
        max_tokens: int | None = None, accelerator_graph: bool = False, dtype: torch.dtype = torch.bfloat16,
        causal: bool = True, verify_extent: int = 0,
    ):
        from mstar.utils.h2d import PinnedStager

        self.causal = causal
        # the keys a verify row attends past its stored length, ``k + 1 - hide_last``, for `gather_resident`
        self.verify_extent = verify_extent
        self.device = device
        self.page_size = page_size
        self.width = table_width
        self.accelerator_graph = accelerator_graph
        self.batch_size = batch_size
        self.dtype = dtype
        self._kernel = None
        self._stager = PinnedStager(torch.int32)
        self._rows = 0
        self._max_q = 0
        self._page_table = self._seqused = self._qo_indptr_buf = None
        if accelerator_graph:
            assert batch_size is not None and max_tokens is not None, "batch_size and max_tokens for a capture"
            self._rows, self._max_q = batch_size, max_tokens
            self._page_table = torch.zeros(batch_size, self.width, dtype=torch.int32, device=device)
            self._seqused = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

    @torch.compiler.disable
    def plan(
        self, qo_indptr: torch.Tensor, paged_kv_indptr: torch.Tensor, paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor, causal: bool = True, dtype: torch.dtype = torch.bfloat16,
        window_left: int = -1, **kwargs,
    ) -> None:
        """A host plan: H2D copies only, so it may run beside a live graph."""
        del kwargs, causal  # the mask is fixed by the wrapper
        if window_left >= 0:
            raise ValueError("FA3RaggedWrapper plans no sliding window")
        qo = qo_indptr.numpy().astype(np.int32)
        rows = len(qo) - 1
        kvp = paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        if rows and int(npages.max()) > self.width:
            raise ValueError(f"a row of {int(npages.max())} pages, past the page table's {self.width}")
        # a row of no pages (a replay's padding) attends to nothing
        lens = np.where(npages > 0, np.maximum(npages - 1, 0) * self.page_size + paged_kv_last_page_len.numpy(), 0)
        host = np.zeros((rows, self.width), dtype=np.int32)
        row_of = np.repeat(np.arange(rows), npages)
        host[row_of, np.arange(row_of.shape[0]) - kvp[row_of]] = paged_kv_indices.numpy()
        self.dtype = dtype
        if self.accelerator_graph:
            if rows != self.batch_size:
                raise ValueError(f"FA3RaggedWrapper captured for {self.batch_size} rows, planned {rows}")
            if int(qo[-1]) > self._max_q:
                raise ValueError(f"FA3RaggedWrapper captured for {self._max_q} queries, planned {int(qo[-1])}")
            self._stager.copy_(self._page_table, host.reshape(-1))
            self._stager.copy_(self._seqused, lens.astype(np.int32))
            self._stager.copy_(self._qo_indptr_buf, qo)
            return
        from mstar.utils.h2d import to_device_nonblocking

        self._rows = rows
        self._max_q = int(np.diff(qo).max()) if rows else 0
        self._page_table = to_device_nonblocking(torch.from_numpy(host), self.device)
        self._seqused = to_device_nonblocking(torch.from_numpy(lens.astype(np.int32)), self.device)
        self._qo_indptr_buf = to_device_nonblocking(torch.from_numpy(qo), self.device)

    def gather_resident(
        self, store: torch.Tensor, slots: torch.Tensor, kv_len: torch.Tensor, stride: int, col: int,
    ) -> None:
        """A mixed step's verify rows off the resident tables, in the graph: each row with a slot reads its block
        table, ``table_width`` columns from ``col``, and attends to its device length plus ``verify_extent``. A row
        on the trash row (a prompt row, or padding) keeps its host plan. The host plan's query offsets stand:
        the step's rows are the host's to choose."""
        live = slots != 0
        rows = store.view(-1, stride)[:, col: col + self.width].index_select(0, slots)
        self._page_table.copy_(torch.where(live[:, None], rows.to(torch.int32), self._page_table))
        lens = (kv_len + self.verify_extent).to(torch.int32)
        self._seqused.copy_(torch.where(live, lens, self._seqused))

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer) -> torch.Tensor:
        """``q`` [tokens, qo heads, head dim] over a layer's pages."""
        if isinstance(kv_cache_layer, (tuple, list)):
            k, v = kv_cache_layer
        else:
            k, v = kv_cache_layer[:, 0], kv_cache_layer[:, 1]
        kernel = self._kernel
        if kernel is None:
            if flash_attn_with_kvcache is None:
                raise RuntimeError(f"paged FA3 can't run: flash_attn_3 doesn't import ({_IMPORT_ERROR})")
            kernel = self._kernel = flash_attn_with_kvcache
        # a captured graph's max_seqlen_q stays the bucket's: the kernel's grid is fixed at capture
        out = kernel(
            q.to(self.dtype), k, v, cache_seqlens=self._seqused, page_table=self._page_table,
            cu_seqlens_q=self._qo_indptr_buf, max_seqlen_q=max(self._max_q, 1), causal=self.causal,
            num_splits=self.NUM_SPLITS, pack_gqa=self.PACK_GQA,
        )
        if self.accelerator_graph:
            # the queries past every row's (a replay's padding) are never written by the kernel: one in-place
            # kernel zeroes them from the planned total on the device, so on every replay
            zero_unplanned_rows(out, self._qo_indptr_buf)
        return out
