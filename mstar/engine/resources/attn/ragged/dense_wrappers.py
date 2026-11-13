"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm), or
FlashAttention 4's on Blackwell, or Dao's FlashAttention-3 build (``flash_attn_3``) where
the ``ragged_attention`` table names ``fa3``.

Same layouts as the FlashInfer wrappers, without a workspace or head-dim padding.
"""

import torch

from mstar.engine.resources.attn.dense import flash_attn_varlen_func
from mstar.engine.resources.attn.ragged.wrappers import (
    PrefixedPrefillWrapper,
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    _KeyPrefixMixin,
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


# eager non-causal plans with at most this many segments run a dense call per segment
_DENSE_MAX_SEGMENTS = 8


class _PerSegmentDenseBase(_DenseRaggedPrefillBase):
    """The dense backend's plan and graph-mode tail handling over another kernel library's varlen call
    (``_varlen_call``). An eager non-causal plan of a few segments runs one dense call per segment
    (``_dense_into``), each writing its rows of the packed output, where the library's dense kernel is the faster
    one; the result is its varlen kernel's bit for bit."""

    def _plan_kernel(self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool) -> None:
        super()._plan_kernel(cu, kv_cu, total_tokens, causal)
        self._segments = None
        if not self.accelerator_graph and not causal and cu.numel() - 1 <= _DENSE_MAX_SEGMENTS:
            q_cu, k_cu = cu.cpu().tolist(), kv_cu.cpu().tolist()
            self._segments = [(q_cu[i], q_cu[i + 1], k_cu[i], k_cu[i + 1]) for i in range(len(q_cu) - 1)]

    def _varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self._segments is not None:
            return self._dense(q, k, v)
        return self._varlen_call(q, k, v)

    def _dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        for q0, q1, k0, k1 in self._segments:
            if q1 > q0:
                self._dense_into(q[q0:q1][None], k[k0:k1][None], v[k0:k1][None], out[q0:q1][None])
        return out

    def _varlen_call(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError

    def _dense_into(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
        """``out`` = dense non-causal attention of ``[1, S, H, D]`` q/k/v."""
        raise NotImplementedError


class _Fa4RaggedPrefillBase(_PerSegmentDenseBase):
    """FlashAttention 4 (``flash_attn.cute``). Its dense non-causal call is statically persistent, which varlen
    can't be (MiniMax-H3's 2-head calls over 38K rows: 6% faster on B300; 1 head: 11%)."""

    _strided = False

    def _varlen_call(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from flash_attn.cute import flash_attn_varlen_func as fa4_varlen

        return fa4_varlen(
            q, k, v, cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_k,
            max_seqlen_q=self._max_q, max_seqlen_k=self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )

    def _dense_into(self, q, k, v, out) -> None:
        _fa4_dense_into(q, k, v, out, self.sm_scale)


def _fa4_dense_into(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """``out`` = FA4 dense non-causal attention of ``[1, S, H, D]`` q/k/v: in place through the forward the public
    ``flash_attn_func`` wraps, or (if that private signature moves) through ``flash_attn_func`` and a copy."""
    from flash_attn.cute import interface

    try:
        interface._flash_attn_fwd(q, k, v, softmax_scale=scale, causal=False, out=out)
    except TypeError:
        result = interface.flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
        out.copy_(result[0] if isinstance(result, tuple) else result)


class _Fa3RaggedPrefillBase(_PerSegmentDenseBase):
    """Dao's FlashAttention-3 (``flash_attn_3``, the PyTorch index's wheel for torch's CUDA build), the
    ``ragged_attention`` table's ``fa3``. Its dense kernel runs a long segment ~3% faster than its varlen one or
    FlashInfer's FA3 backend (MiniMax-H3's 28 heads over 38K rows on an H200 at its power cap). It reads the head
    count off q / k / v, so a Ulysses pipeline attends a head group through the full call's plan, those heads'
    bits unchanged."""

    any_heads = True
    _strided = True

    def _varlen_call(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from flash_attn_interface import flash_attn_varlen_func as fa3_varlen

        return fa3_varlen(
            q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
            causal=self._causal, softmax_scale=self.sm_scale,
        )

    def _dense_into(self, q, k, v, out) -> None:
        import flash_attn_interface  # noqa: F401  # registers torch.ops.flash_attn_3

        # the kernel op itself: the public wrapper is a custom op, which refuses an output it didn't allocate.
        # One split, as flash_attn_func and the varlen call run it
        torch.ops.flash_attn_3.fwd(q, k, v, out=out, softmax_scale=self.sm_scale, is_causal=False, num_splits=1)


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA3 varlen."""


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA3 varlen."""


class DensePrefixedPrefillWrapper(_DenseRaggedPrefillBase, PrefixedPrefillWrapper):
    """Prefixed attention through FA3 varlen."""


class DenseKeyPrefixPrefillWrapper(_KeyPrefixMixin, DenseRaggedCrossPrefillWrapper):
    """Key-prefix self-attention through FA3."""


class Fa4RaggedPrefillWrapper(_Fa4RaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA4 varlen."""


class Fa4RaggedCrossPrefillWrapper(_Fa4RaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA4 varlen."""


class Fa3RaggedPrefillWrapper(_Fa3RaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through Dao's FA3 varlen (the ``ragged_attention`` table's ``fa3``)."""
