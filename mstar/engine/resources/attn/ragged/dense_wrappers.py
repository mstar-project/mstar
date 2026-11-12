"""Cacheless varlen prefill through FA3 ``flash_attn_varlen`` (AITER on ROCm), or FA3's
fixed-length kernel per segment for an eager non-causal plan of one segment or of long ones.

Same layouts as the FlashInfer wrappers, without a workspace or head-dim padding.
"""

from typing import NamedTuple

import torch

from mstar.engine.resources.attn.dense import fa3_fixed_into, flash_attn_func, flash_attn_varlen_func
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


# Query rows from which every segment of a plan runs on its own fixed-length call:
# H200, FA3 at 32/8 heads of 128, two to six segments: 6-9% under one varlen call
# at 1536-3552 rows, while at 1024 and under the launches cost more than they save.
_FIXED_SEGMENT_MIN_ROWS = 1536


class _FixedSegments(NamedTuple):
    """A plan's segment boundaries (host ``cu_seqlens``) for FA3's fixed-length kernel."""
    q: tuple[int, ...]
    k: tuple[int, ...]


class _DenseRaggedPrefillBase(_RaggedPrefillBase):
    """FA3 varlen over packed q/k/v. An eager non-causal plan of one segment, or
    of segments all at least ``_FIXED_SEGMENT_MIN_ROWS`` long, runs FA3's
    fixed-length kernel once per segment instead, with the same output bit for bit.

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
        # the segments of a plan FA3's fixed-length kernel runs, else None
        self._fixed: _FixedSegments | None = None

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
            self._fixed = None
            return
        self._cu_q = cu.to(self.device, non_blocking=True)
        self._cu_k = self._cu_q if kv_cu is cu else kv_cu.to(self.device, non_blocking=True)
        cu_host, kv_host = cu.cpu(), kv_cu.cpu()
        self._max_q, self._max_k = _max_span(cu_host), _max_span(kv_host)
        # FA3's fixed-length kernel beats varlen on one non-causal segment (H200: 1-12%
        # at Cosmos3's and the vision towers' shapes) and on each of several long ones,
        # but loses on short causal ones
        self._fixed = None
        if flash_attn_func is not None and not causal and cu_host.numel() == kv_host.numel() >= 2:
            if cu_host.numel() == 2 or (
                fa3_fixed_into is not None and int((cu_host[1:] - cu_host[:-1]).min()) >= _FIXED_SEGMENT_MIN_ROWS
            ):
                self._fixed = _FixedSegments(tuple(cu_host.tolist()), tuple(kv_host.tolist()))

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
        fixed = self._fixed
        if fixed is not None and (fixed.q[-1], fixed.k[-1]) == (q.shape[0], k.shape[0]):
            if len(fixed.q) == 2:
                out = flash_attn_func(q[None], k[None], v[None], causal=False, softmax_scale=self.sm_scale)
                out = (out[0] if isinstance(out, tuple) else out)[0]
            else:
                # each segment writes its rows of one output, so nothing is concatenated after
                out = q.new_empty(q.shape[0], q.shape[1], v.shape[-1])
                scale = self.sm_scale if self.sm_scale is not None else q.shape[-1] ** -0.5
                for i in range(len(fixed.q) - 1):
                    q0, q1, k0, k1 = fixed.q[i], fixed.q[i + 1], fixed.k[i], fixed.k[i + 1]
                    fa3_fixed_into(q[None, q0:q1], k[None, k0:k1], v[None, k0:k1], out[None, q0:q1], scale)
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


class DenseKeyPrefixPrefillWrapper(_KeyPrefixMixin, DenseRaggedCrossPrefillWrapper):
    """Key-prefix self-attention through FA3."""
