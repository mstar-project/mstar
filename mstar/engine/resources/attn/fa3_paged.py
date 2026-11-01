"""A verify block's paged attention on FlashAttention-3 (the ``flash_attn_3`` wheel), for a node
that takes its verdicts back on the device (``DraftPolicy.device_rewind``).

FA3's paged kernel reads each row's KV length from the device (``cache_seqlens``) and takes
nothing else that depends on it from its plan: the split schedule
(``get_scheduler_metadata``) is computed on the device too. So a rewind shortens the lengths,
computes the schedule from them once for the step's layers, and the step then attends exactly
as an eager call at the shortened lengths does.
"""

import functools

import numpy as np
import torch

from mstar.platforms import current_platform

try:
    from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
except Exception as _exc:  # noqa: BLE001
    flash_attn_with_kvcache = get_scheduler_metadata = None
    _IMPORT_ERROR: str | None = f"{type(_exc).__name__}: {_exc}"
else:
    _IMPORT_ERROR = None

# the head dims the wheel builds FA3 kernels for
_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})


@functools.cache
def fa3_paged_unavailable_reason(device: torch.device, head_dim: int, dtype: torch.dtype) -> str | None:
    """Why a verify block on ``device`` can't run paged FA3, or None if it can."""
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
        # measured on SM90 only; the wheel has no Blackwell kernels
        return f"paged FA3 runs on SM90, not compute capability {capability}"
    if head_dim not in _HEAD_DIMS:
        return f"flash_attn_3 has no kernel for head_dim {head_dim}"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"paged FA3 reads a bf16 or fp16 cache, not {dtype}"
    return None


class FA3PagedWrapper:
    """A paged FA3 plan over rows that each run one block of the same few tokens (a verify
    step): the rows' pages as a page table, their KV lengths and query offsets, in this
    wrapper's own buffers (static under a captured graph). Takes the calls
    `FlashInferPrefillWrapper` takes from the manager.

    The page table is as wide as the pool's longest row (``max_seq_len``) in every plan, eager
    or captured: FA3 bounds its split count by that width, so eager and replayed steps split
    alike."""

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_seq_len: int,
        device: torch.device,
        batch_size: int | None = None,
        accelerator_graph: bool = False,
    ):
        from mstar.utils.h2d import PinnedStager

        self.device = device
        self.accelerator_graph = accelerator_graph
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.width = -(-max_seq_len // page_size) + 1
        self.dtype = torch.bfloat16
        self._causal = True
        self._rows = self._span = 0
        self._stager = PinnedStager(torch.int32)
        # the schedule `correct_kv_lens` computes; None runs FA3's own in every call
        self._meta: torch.Tensor | None = None
        self._page_table = self._seqused = self._qo_indptr_buf = None
        if accelerator_graph:
            assert batch_size is not None, "batch_size required for accelerator graph mode"
            self._buffer = torch.zeros(self._layout(batch_size)[-1], dtype=torch.int32, device=device)
            self._views(self._buffer, batch_size)

    def _layout(self, rows: int) -> tuple[int, int, int]:
        """Where the lengths and the query offsets start in a plan's one int32 buffer, behind
        the page table, and its size; each at a multiple of 16 elements."""
        lens_at = -(-rows * self.width // 16) * 16
        qo_at = lens_at + -(-rows // 16) * 16
        return lens_at, qo_at, qo_at + rows + 1

    def _views(self, buffer: torch.Tensor, rows: int) -> None:
        lens_at, qo_at, end = self._layout(rows)
        self._page_table = buffer[: rows * self.width].view(rows, self.width)
        self._seqused = buffer[lens_at:qo_at][:rows]
        self._qo_indptr_buf = buffer[qo_at:end]

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        kv_lens: torch.Tensor | None = None,
        window_left: int = -1,
        **kwargs,
    ):
        """From the KV plan's host arrays: H2D copies only, so it may run beside a live graph."""
        del kwargs  # FA2's split options
        if window_left >= 0:
            raise ValueError("FA3PagedWrapper plans no sliding window")
        qo = qo_indptr.numpy()
        spans = np.diff(qo)
        rows = len(spans)
        span = int(spans[0]) if rows else 0
        if (spans != span).any():
            raise ValueError(f"FA3PagedWrapper plans rows of one block each; got spans {spans.tolist()}")
        kvp = paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        if kv_lens is None:
            lens = np.maximum(npages - 1, 0) * self.page_size + paged_kv_last_page_len.numpy()
        else:
            lens = np.asarray(kv_lens, dtype=np.int64)
        if rows and int(npages.max()) > self.width:
            raise ValueError(f"a row of {int(npages.max())} pages, past the page table's {self.width}")
        # one host array, one copy: the page table (zeros past a row's pages name SINK_PAGE),
        # the lengths and the query offsets
        lens_at, qo_at, end = self._layout(rows)
        host = np.zeros(end, dtype=np.int32)
        row_of = np.repeat(np.arange(rows), npages)
        host[row_of * self.width + np.arange(row_of.shape[0]) - kvp[row_of]] = paged_kv_indices.numpy()
        host[lens_at:lens_at + rows] = lens
        host[qo_at:end] = qo
        if self.accelerator_graph:
            if rows != self.batch_size:
                # a graph replays the rows it was captured with
                raise ValueError(f"FA3PagedWrapper captured for {self.batch_size} rows, planned {rows}")
            self._stager.copy_(self._buffer, host)
        else:
            from mstar.utils.h2d import to_device_nonblocking

            self._views(to_device_nonblocking(torch.from_numpy(host), self.device), rows)
        self._rows, self._span = rows, span
        self._causal, self.dtype = causal, dtype
        if not self.accelerator_graph:
            # a captured plan keeps the schedule it was captured with alive: its graph writes and
            # reads it in every replay
            self._meta = None

    def locate_kv_lens(self) -> None:
        """The rows' lengths are on the device already (`correct_kv_lens`)."""

    def correct_kv_lens(self, rejected: torch.Tensor) -> None:
        """Each row's KV length less its ``rejected`` (``[rows]`` int32), on the device, and the
        split schedule from the shortened lengths, once for the step's layers. Capturable."""
        self._seqused.sub_(rejected)
        self._meta = get_scheduler_metadata(
            self._rows, self._span, self.width * self.page_size, self.num_qo_heads, self.num_kv_heads,
            self.head_dim, self._seqused, qkv_dtype=self.dtype, cu_seqlens_q=self._qo_indptr_buf,
            page_size=self.page_size, causal=self._causal,
        )

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer) -> torch.Tensor:
        """``q`` [tokens, qo heads, head dim] over a layer's pages ``[pages, 2, page, kv heads,
        head dim]`` (or its ``(k, v)``)."""
        if isinstance(kv_cache_layer, (tuple, list)):
            k, v = kv_cache_layer
        else:
            k, v = kv_cache_layer[:, 0], kv_cache_layer[:, 1]
        if q.shape[0] != self._rows * self._span:
            # past the plan's rows the kernel would read the wrong queries
            raise ValueError(f"FA3PagedWrapper planned {self._rows} rows of {self._span}, got {q.shape[0]} queries")
        return flash_attn_with_kvcache(
            q.to(self.dtype), k, v, cache_seqlens=self._seqused, page_table=self._page_table,
            cu_seqlens_q=self._qo_indptr_buf, max_seqlen_q=self._span, causal=self._causal,
            scheduler_metadata=self._meta,
        )
