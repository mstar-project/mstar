"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm), or FA3's
fixed-length kernel for an eager plan of one non-causal segment.

Same layouts as the FlashInfer wrappers, without a workspace or head-dim padding.
"""

import torch

from mstar.engine.resources.attn.dense import flash_attn_func, flash_attn_varlen_func
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
    """FA3 varlen over packed q/k/v; an eager plan of one non-causal segment runs
    FA3's fixed-length kernel instead, with the same output bit for bit.

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
        # (q rows, kv rows) of a plan that is one segment, for FA3's fixed-length kernel; else None
        self._one_segment: tuple[int, int] | None = None

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
            self._one_segment = None
            return
        self._cu_q = cu.to(self.device, non_blocking=True)
        self._cu_k = self._cu_q if kv_cu is cu else kv_cu.to(self.device, non_blocking=True)
        cu_host, kv_host = cu.cpu(), kv_cu.cpu()
        self._max_q, self._max_k = _max_span(cu_host), _max_span(kv_host)
        # FA3's fixed-length kernel beats varlen on one non-causal segment (H200: 1-12%
        # at Cosmos3's and the vision towers' shapes) but loses on short causal ones
        one = flash_attn_func is not None and not causal and cu_host.numel() == 2 and kv_host.numel() == 2
        self._one_segment = (self._max_q, self._max_k) if one else None

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False) -> torch.Tensor:
        if return_lse:
            raise NotImplementedError("the DENSE ragged backend does not return the log-sum-exp; use FLASHINFER")
        # FA3 reads rows by stride, so q / k / v sliced out of a fused projection
        # go in as they are; AITER's varlen is handed contiguous ones
        strided = not current_platform.is_rocm()
        q, k, v = (
            t if strided and t.stride(-1) == 1 else t.contiguous()
            for t in (t.to(self.q_data_type) for t in (q, k, v))
        )
        # the fixed-length kernel reads every row it is given, so only where the plan covers them all
        if self._one_segment is not None and self._one_segment == (q.shape[0], k.shape[0]):
            out = flash_attn_func(q[None], k[None], v[None], causal=self._causal, softmax_scale=self.sm_scale)
            out = (out[0] if isinstance(out, tuple) else out)[0]
        else:
            out = flash_attn_varlen_func(
                q, k, v, self._cu_q, self._cu_k, self._max_q, self._max_k,
                causal=self._causal, softmax_scale=self.sm_scale,
            )
            out = out[0] if isinstance(out, tuple) else out
        if not self.accelerator_graph:
            return out
        self._check_rows(q.shape[0])
        return zero_unplanned_rows(out, self._qo_indptr_buf)


class DenseRaggedPrefillWrapper(_DenseRaggedPrefillBase, RaggedPrefillWrapper):
    """Ragged self-attention through FA3 varlen."""


class DenseRaggedCrossPrefillWrapper(_DenseRaggedPrefillBase, RaggedCrossPrefillWrapper):
    """Ragged cross-attention through FA3 varlen."""


class DensePrefixedPrefillWrapper(_DenseRaggedPrefillBase, PrefixedPrefillWrapper):
    """Prefixed attention through FA3 varlen."""
