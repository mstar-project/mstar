"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm).

Same layouts as the FlashInfer wrappers, without a workspace or head-dim padding.
"""

import torch
import triton
import triton.language as tl

from mstar.engine.resources.attn.dense import flash_attn_varlen_func
from mstar.engine.resources.attn.ragged.wrappers import (
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    _RaggedPrefillBase,
)

_ZERO_BLOCK = 4096


@triton.jit
def _zero_rows_from(out_ptr, first_row_ptr, numel, row_numel, BLOCK: tl.constexpr):
    """Zero a contiguous tensor in place from row ``*first_row_ptr`` onward.
    Chunks that land entirely before that row return immediately."""
    start = tl.program_id(0).to(tl.int64) * BLOCK
    first = tl.load(first_row_ptr).to(tl.int64) * row_numel
    if start + BLOCK > first:
        offs = start + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, 0.0, mask=(offs >= first) & (offs < numel))


def _max_span(cu_seqlens: torch.Tensor) -> int:
    """Longest segment in a host ``cu_seqlens``. Zero if there are no tokens."""
    if cu_seqlens.numel() < 2:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max())


class _DenseRaggedPrefillBase(_RaggedPrefillBase):
    """FA3 varlen over packed q/k/v.

    Graph mode reads the layout from the static buffers and launches at the
    bucket ceiling, so every replay of that bucket runs the same kernel.
    """

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, **kwargs):
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        self._causal = False
        self._cu_q: torch.Tensor | None = None
        self._cu_k: torch.Tensor | None = None
        self._max_q = 0
        self._max_k = 0

        # varlen writes only the planned rows, so in a replay the rest hold
        # whatever the graph pool had there (a previous replay's, possibly NaN).
        # ``run`` zeros from ``_n_valid`` (the planned total) onward.
        self._n_valid = (
            torch.zeros((), dtype=torch.int32, device=self.device) if self.accelerator_graph else None
        )

    def _plan_kernel(
        self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool,
    ) -> None:
        self._causal = causal
        if self.accelerator_graph:
            self._qo_indptr_buf.copy_(cu, non_blocking=True)
            self._cu_q = self._qo_indptr_buf
            if kv_cu is cu:
                self._cu_k = self._cu_q
            else:
                self._kv_indptr_buf.copy_(kv_cu, non_blocking=True)
                self._cu_k = self._kv_indptr_buf
            self._max_q, self._max_k = self.max_total_tokens, self.max_total_kv_tokens
            self._n_valid.fill_(total_tokens)
            return
        self._cu_q = cu.to(self.device, non_blocking=True)
        self._cu_k = self._cu_q if kv_cu is cu else kv_cu.to(self.device, non_blocking=True)
        self._max_q, self._max_k = _max_span(cu.cpu()), _max_span(kv_cu.cpu())

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = (t.to(self.q_data_type).contiguous() for t in (q, k, v))
        out = flash_attn_varlen_func(
            q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )
        out = out[0] if isinstance(out, tuple) else out
        if self._n_valid is None:
            return out
        self._check_rows(q.shape[0])
        assert out.is_contiguous(), "the tail is zeroed by flat offset"
        numel = out.numel()
        _zero_rows_from[(triton.cdiv(numel, _ZERO_BLOCK),)](
            out, self._n_valid, numel, out[0].numel(), BLOCK=_ZERO_BLOCK,
        )
        return out


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA3 varlen."""


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA3 varlen."""
