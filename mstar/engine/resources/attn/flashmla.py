"""DeepSeek's FlashMLA as the decode kernel of the MLA attention resource.

FlashInfer's Hopper MLA kernel costs about 15 us per call whatever the batch, which on Kimi K3's
24 MLA layers is 0.4-1.0 ms of every decode step at small batch; FlashMLA (``flash_mla_with_kvcache``,
sm90) reads the same paged latent cache, ``[pages, page_size, 576]`` viewed with one KV head, through
a block table and per-row lengths. This wrapper has the FlashInfer wrapper's ``plan`` / ``run``
surface for the plans it can take: every row the same number of queries (one in a plain decode
step, ``k + 1`` in a verify step), causal, no context-only reads. The manager picks it per plan.
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch

FLASHMLA_HEAD_DIM = 576  # latent + rope
FLASHMLA_HEAD_DIM_V = 512
FLASHMLA_PAGE = 64


@lru_cache(maxsize=1)
def flashmla_available() -> bool:
    try:
        import flash_mla  # noqa: F401
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def flashmla_wanted() -> bool:
    """``MSTAR_MLA_DECODE_BACKEND`` = ``flashmla`` or ``flashinfer`` (the default). On an H100 the two
    kernels cost about the same per call (15 us at one row, 26 against 31 us at 32 rows, 48 against
    50 at 64: bench/kernels/mla_decode_backends.py), so FlashMLA stays an option until it wins somewhere."""
    return os.environ.get("MSTAR_MLA_DECODE_BACKEND", "flashinfer") == "flashmla"


def flashmla_supports(kv_lora_rank: int, qk_rope_head_dim: int, page_size: int = FLASHMLA_PAGE) -> bool:
    """The kernel's fixed shape: latent 512 + rope 64 in pages of 64 tokens."""
    return (kv_lora_rank == FLASHMLA_HEAD_DIM_V and kv_lora_rank + qk_rope_head_dim == FLASHMLA_HEAD_DIM
            and page_size == FLASHMLA_PAGE)


class FlashMLAWrapper:
    """One plan's block table and lengths (static buffers under a CUDA graph), and the call.

    ``max_pages_per_row`` bounds the block table's width (``max_seq_len / page_size``)."""

    def __init__(
        self,
        num_qo_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        sm_scale: float,
        max_pages_per_row: int,
        batch_size: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
    ):
        from flash_mla import get_mla_metadata

        assert flashmla_supports(kv_lora_rank, qk_rope_head_dim, page_size), (kv_lora_rank, qk_rope_head_dim, page_size)
        self.num_qo_heads, self.kv_lora_rank, self.qk_rope_head_dim = num_qo_heads, kv_lora_rank, qk_rope_head_dim
        self.page_size, self.sm_scale, self.device = page_size, float(sm_scale), torch.device(device)
        self.use_cuda_graph, self.batch_size, self.max_pages = use_cuda_graph, batch_size, int(max_pages_per_row)
        self._get_meta = get_mla_metadata
        self._sched = None  # FlashMLASchedMeta: fixed (rows, queries, heads) per wrapper under a graph
        self.s_q = 1
        pin = dict(pin_memory=True) if self.device.type == "cuda" else {}
        if use_cuda_graph:
            assert batch_size is not None
            self._block_table = torch.zeros(batch_size, self.max_pages, dtype=torch.int32, device=self.device)
            self._kv_len_buf = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            self._block_table_host = torch.zeros(batch_size, self.max_pages, dtype=torch.int32, **pin)
            self._kv_len_host = torch.zeros(batch_size, dtype=torch.int32, **pin)
        else:
            self._block_table = self._kv_len_buf = self._qo_indptr_buf = None
        self.dtype = torch.bfloat16

    @staticmethod
    def plan_fits(qo_indptr: torch.Tensor, causal: bool, context_only: bool) -> int:
        """The rows' common query count when this kernel can take the plan, else 0."""
        if context_only or not causal:
            return 0
        q = qo_indptr[1:] - qo_indptr[:-1]
        if q.numel() == 0:
            return 0
        s_q = int(q[0])
        return s_q if s_q >= 1 and bool((q == s_q).all()) else 0

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """Host tensors from the KV plan, as the FlashInfer wrapper takes them."""
        rows = qo_indptr.numel() - 1
        s_q = self.plan_fits(qo_indptr, causal, False)
        assert s_q, "FlashMLA takes causal plans with the same query count in every row"
        num_pages = (paged_kv_indptr[1:] - paged_kv_indptr[:-1]).to(torch.int64)
        kv_len = (num_pages - 1).clamp_min(0) * self.page_size + paged_kv_last_page_len.to(torch.int64)
        # a row without pages (a padding row before its first token) attends to one token of page 0,
        # the sink page, so the kernel never sees an empty row
        kv_len = torch.where(num_pages > 0, kv_len, torch.ones_like(kv_len))
        table = torch.zeros(rows, self.max_pages, dtype=torch.int32)
        starts = paged_kv_indptr[:-1].to(torch.int64)
        for r in range(rows):
            n = int(num_pages[r])
            if n:
                table[r, :n] = paged_kv_indices[starts[r]:starts[r] + n]
        if self.use_cuda_graph:
            assert rows == self.batch_size, (rows, self.batch_size)
            self._block_table_host.copy_(table)
            self._kv_len_host.copy_(kv_len.to(torch.int32))
            self._block_table.copy_(self._block_table_host, non_blocking=True)
            self._kv_len_buf.copy_(self._kv_len_host, non_blocking=True)
            self._qo_indptr_buf.copy_(qo_indptr.to(torch.int32), non_blocking=True)
        else:
            self._block_table = table.to(self.device, non_blocking=True)
            self._kv_len_buf = kv_len.to(self.device, dtype=torch.int32, non_blocking=True)
            self._qo_indptr_buf = qo_indptr.to(self.device, dtype=torch.int32, non_blocking=True)
            self.batch_size = rows
        # FlashMLA works its tile schedule out on the device at the first call made with a
        # FlashMLASchedMeta and keeps it for the calls after, so a plan with other lengths needs a
        # new one (the old schedule gave wrong rows after a replan). The captured path gets its new
        # one inside the capture (see run), so a replay recomputes the schedule from the lengths buffer.
        if not self.use_cuda_graph:
            self._sched, _ = self._get_meta()
        self.s_q = s_q
        self.dtype = dtype

    @torch.compiler.disable
    def run(self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_cache_layer: torch.Tensor, return_lse: bool = False):
        """``q_nope [T, H, 512]``, ``q_pe [T, H, 64]`` with ``T = rows * s_q``; ``kv_cache_layer
        [pages, page_size, 576]`` -> ``[T, H, 512]`` (and the natural log-sum-exp ``[T, H]`` fp32)."""
        from flash_mla import flash_mla_with_kvcache

        rows = self.batch_size
        if self._sched is None or torch.cuda.is_current_stream_capturing():
            self._sched, _ = self._get_meta()  # a fresh schedule, computed by the captured kernels on every replay
        q = torch.cat([q_nope, q_pe], dim=-1).view(rows, self.s_q, self.num_qo_heads, FLASHMLA_HEAD_DIM)
        k = kv_cache_layer.view(kv_cache_layer.shape[0], self.page_size, 1, FLASHMLA_HEAD_DIM)
        out, lse = flash_mla_with_kvcache(
            q, k, self._block_table, self._kv_len_buf, FLASHMLA_HEAD_DIM_V, self._sched, None, self.sm_scale,
            causal=True,
        )
        out = out.view(rows * self.s_q, self.num_qo_heads, FLASHMLA_HEAD_DIM_V)
        if return_lse:
            return out, lse.permute(0, 2, 1).reshape(rows * self.s_q, self.num_qo_heads)
        return out
