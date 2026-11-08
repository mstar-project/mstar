"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm), or
FlashAttention 4's on Blackwell.

Same layouts as the FlashInfer wrappers, without a workspace or head-dim padding.
"""

import torch

from mstar.engine.resources.attn.dense import flash_attn_varlen_func
from mstar.engine.resources.attn.ragged.wrappers import (
    PrefixedPrefillWrapper,
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    _RaggedPrefillBase,
    zero_unplanned_rows,
)
from mstar.platforms import current_platform


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
            return
        self._cu_q = cu.to(self.device, non_blocking=True)
        self._cu_k = self._cu_q if kv_cu is cu else kv_cu.to(self.device, non_blocking=True)
        self._max_q, self._max_k = _max_span(cu.cpu()), _max_span(kv_cu.cpu())

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False) -> torch.Tensor:
        if return_lse:
            raise NotImplementedError("the DENSE ragged backend does not return the log-sum-exp; use FLASHINFER")
        q, k, v = (
            t if self._strided and t.stride(-1) == 1 else t.contiguous()
            for t in (t.to(self.q_data_type) for t in (q, k, v))
        )
        out = self._varlen(q, k, v)
        out = out[0] if isinstance(out, tuple) else out
        if not self.accelerator_graph:
            return out
        self._check_rows(q.shape[0])
        return zero_unplanned_rows(out, self._qo_indptr_buf)

    # FA3 reads rows by stride, so q / k / v sliced out of a fused projection
    # go in as they are; AITER's varlen is handed contiguous ones
    _strided = not current_platform.is_rocm()

    def _varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return flash_attn_varlen_func(
            q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )


# eager non-causal plans with at most this many segments run FA4 dense, a call per segment
_FA4_DENSE_MAX_SEGMENTS = 8


class _Fa4RaggedPrefillBase(_DenseRaggedPrefillBase):
    """FlashAttention 4 (``flash_attn.cute``): the same plan and graph-mode tail handling as FA3, another kernel
    call. An eager non-causal plan of a few segments runs one dense call per segment, each writing its rows of the
    packed output: FA4 schedules a dense non-causal call statically persistent, which varlen can't, and the result
    is varlen's bit for bit (MiniMax-H3's 2-head calls over 38K rows: 6% faster on B300; 1 head: 11%)."""

    def _plan_kernel(self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool) -> None:
        super()._plan_kernel(cu, kv_cu, total_tokens, causal)
        self._segments = None
        if not self.accelerator_graph and not causal and cu.numel() - 1 <= _FA4_DENSE_MAX_SEGMENTS:
            q_cu, k_cu = cu.cpu().tolist(), kv_cu.cpu().tolist()
            self._segments = [(q_cu[i], q_cu[i + 1], k_cu[i], k_cu[i + 1]) for i in range(len(q_cu) - 1)]

    _strided = False

    def _varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self._segments is not None:
            return self._dense(q, k, v)
        from flash_attn.cute import flash_attn_varlen_func as fa4_varlen

        return fa4_varlen(
            q, k, v, cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_k,
            max_seqlen_q=self._max_q, max_seqlen_k=self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )

    def _dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        for q0, q1, k0, k1 in self._segments:
            if q1 > q0:
                _fa4_dense_into(q[q0:q1][None], k[k0:k1][None], v[k0:k1][None], out[q0:q1][None], self.sm_scale)
        return out


def _fa4_dense_into(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """``out`` = FA4 dense non-causal attention of ``[1, S, H, D]`` q/k/v: in place through the forward the public
    ``flash_attn_func`` wraps, or (if that private signature moves) through ``flash_attn_func`` and a copy."""
    from flash_attn.cute import interface

    try:
        interface._flash_attn_fwd(q, k, v, softmax_scale=scale, causal=False, out=out)
    except TypeError:
        result = interface.flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
        out.copy_(result[0] if isinstance(result, tuple) else result)


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA3 varlen."""


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA3 varlen."""


class DensePrefixedPrefillWrapper(_DenseRaggedPrefillBase, PrefixedPrefillWrapper):
    """Prefixed attention through FA3 varlen."""


class Fa4RaggedPrefillWrapper(_Fa4RaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA4 varlen."""


class Fa4RaggedCrossPrefillWrapper(_Fa4RaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA4 varlen."""
