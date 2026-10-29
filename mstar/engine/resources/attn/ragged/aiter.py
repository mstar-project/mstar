"""Cacheless varlen attention on AMD GPUs with AITER's flash attention.

The ROCm counterpart of ``ragged.flashinfer``. Each layer is one
``aiter.flash_attn_varlen_func`` over the packed q/k/v, with the segment layout
as cumulative lengths (separate ones for the query and key sides of a cross
pair). CK takes any head dim that is a multiple of 8 up to 256, so nothing is
zero-padded to FlashInfer's 64/128/256.

A captured step stages its lengths into static buffers per (bucket, slot,
label), padded to the bucket's segment count with empty segments. The kernel
grid is sized on the host from ``max_seqlen_q``/``max_seqlen_k``, so those are
fixed at the bucket's token ceiling and ``run`` reads nothing back from the
device. The kernel allocates each call's output itself: under capture that is
the graph pool's memory, a distinct block per layer, so one layer's result is
never copied out of the way of the next.
"""

import functools
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.attn.aiter import _PlanImage
from mstar.engine.resources.attn.base import EagerSlotKey
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.ragged.base import RaggedAttnManager, RaggedCrossAttnManager
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.step import Segment, SlotLease, StepContext

# in int32s, so the second length array starts 64-byte aligned
_ALIGN = 16
_MAX_HEAD_DIM = 256


@functools.cache
def aiter_varlen_unavailable_reason() -> str | None:
    """Why AITER's varlen flash attention can't be imported, or None."""
    try:
        from aiter import flash_attn_varlen_func  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def _varlen():
    from aiter import flash_attn_varlen_func

    return flash_attn_varlen_func


class _RaggedPlan(NamedTuple):
    """One label's layout for this step. Built per label per step, hence a NamedTuple."""
    cu_seqlens_q: torch.Tensor  # device, [segments + 1]
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    num_segments: int  # real, before padding to a bucket
    num_tokens: int  # real query tokens
    causal: bool


def _cu(segments: list[Segment]) -> np.ndarray:
    cu = np.zeros(len(segments) + 1, dtype=np.int32)
    np.cumsum([seg.span for seg in segments], out=cu[1:])
    return cu


class _AiterRaggedBase:
    """Staging, preplan and ``run``, shared by the self- and cross-attention
    managers. A subclass lays out its step in ``_plan_layouts``."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
    ):
        if config.head_dim > _MAX_HEAD_DIM:
            raise ValueError(
                f"ragged attention backend 'aiter': head_dim {config.head_dim} is "
                f"above the {_MAX_HEAD_DIM} AITER's varlen kernels take"
            )
        reason = aiter_varlen_unavailable_reason()
        if reason is not None:
            raise RuntimeError(
                f"ragged attention backend 'aiter' needs AITER's varlen flash attention ({reason})"
            )
        self._config = config
        self._device = device
        self._dtype = dtype

        self._cg_images: dict[CGSlotKey, _PlanImage] = {}
        self._eager_images: dict[EagerSlotKey, _PlanImage] = {}

        self._current_plans: dict[str, _RaggedPlan] = {}
        self._preplan_plans: dict[str, _RaggedPlan] = {}
        self._preplanned = False

    @property
    def max_segments_per_request(self) -> int:
        """Segments one request may bring to a captured plan, as configured
        (after YAML overrides); a captured plan holds this many per row."""
        return self._config.max_segments_per_request

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # plan copies into device buffers that a replay reads
        return True

    @staticmethod
    def _group_segments_by_label(segments: Iterable[Segment]) -> dict[str, list[Segment]]:
        res: dict[str, list[Segment]] = {}
        for seg in segments:
            res.setdefault(seg.label, []).append(seg)
        return res

    def plan(self, step, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager buffers are per slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "ragged attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            self._current_plans = self._preplan_plans
            self._preplan_plans = {}
            self._preplanned = False
            return
        # a fresh dict, so a label this step doesn't declare can't find last
        # step's plan
        plans: dict[str, _RaggedPlan] = {}
        self._plan_layouts(step, ctx, plans)
        if ctx.is_preplan:
            self._preplan_plans = plans
            self._preplanned = True
        else:
            self._current_plans = plans

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_plans = {}

    def _plan_layouts(self, step, ctx: StepContext, plans: dict[str, _RaggedPlan]) -> None:
        raise NotImplementedError

    def _stage(
        self, label: str, q_cu: np.ndarray, k_cu: np.ndarray, causal: bool,
        lease: SlotLease | None, slot: int,
    ) -> _RaggedPlan:
        n_seg = q_cu.shape[0] - 1
        q_tokens, k_tokens = int(q_cu[-1]), int(k_cu[-1])
        if lease is None:
            seg_cap = n_seg
            max_q = int(np.diff(q_cu).max()) if n_seg else 0
            max_k = int(np.diff(k_cu).max()) if n_seg else 0
            key = EagerSlotKey(label=label, slot=slot)
            image = self._eager_images.get(key)
            numel = 2 * _round(seg_cap + 1)
            if image is None or image.numel < numel:
                if image is not None:
                    image.wait()
                image = self._eager_images[key] = _PlanImage(max(numel, 256), self._device)
        else:
            bucket = lease.bucket
            seg_cap = self._config.max_segments_for(bucket.bs)
            # the bucket's token count is the ceiling. The per-request one is
            # for a runner that buckets by batch size only.
            max_tokens = max(bucket.num_tokens, self._config.max_tokens_for(bucket.bs) or 0)
            if n_seg > seg_cap or q_tokens > max_tokens or k_tokens > max_tokens:
                raise ValueError(
                    f"ragged attention {label!r}: {n_seg} segments / {q_tokens} query / "
                    f"{k_tokens} key tokens exceed bucket {bucket}'s {seg_cap} / {max_tokens}"
                )
            max_q = max_k = max_tokens
            key = CGSlotKey(bucket=bucket, slot=lease.slot, label=label)
            image = self._cg_images.get(key)
            if image is None:
                image = self._cg_images[key] = _PlanImage(2 * _round(seg_cap + 1), self._device)
        k_off = _round(seg_cap + 1)
        image.wait()
        host = image.np
        # pad to the bucket's fixed segment count with empty segments
        host[:n_seg + 1] = q_cu
        host[n_seg + 1:seg_cap + 1] = q_tokens
        host[k_off:k_off + n_seg + 1] = k_cu
        host[k_off + n_seg + 1:k_off + seg_cap + 1] = k_tokens
        image.send(k_off + seg_cap + 1)
        return _RaggedPlan(
            cu_seqlens_q=image.dev[:seg_cap + 1],
            cu_seqlens_k=image.dev[k_off:k_off + seg_cap + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            num_segments=n_seg,
            num_tokens=q_tokens,
            causal=causal,
        )

    ### Submodule-level functionality

    def num_segments(self, label: str | None = None) -> int:
        """Real (unpadded) segment count this step planned under ``label``."""
        return self._plan(label).num_segments

    def _plan(self, label: str | None) -> _RaggedPlan:
        if label is None:
            label = self._default_label
        plan = self._current_plans.get(label)
        if plan is None:
            raise KeyError(
                f"ragged attention has no plan for label {label!r}; this step "
                f"planned {sorted(self._current_plans)}. Every label a "
                "forward attends must carry a segment in the step declaration."
            )
        return plan

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """One layer's attention with the plan for ``label``.

        ``q`` is [total_tokens, Hq, D] in the query layout, ``k`` and ``v`` are
        [total_kv_tokens, Hkv, D] in the key layout (the query layout, for
        self-attention). The output is in the resource's dtype, like
        FlashInfer's, and is the kernel's own allocation: a fresh tensor per
        call, which under capture is a block of the graph pool that a replay
        writes at the same address. Rows past the planned tokens (a padded
        bucket's empty segments) are not written and are not a result.
        """
        plan = self._plan(label)
        qk, kk, vk = (t if t.dtype == self._dtype else t.to(self._dtype) for t in (q, k, v))
        out = _varlen()(
            qk, kk, vk, plan.cu_seqlens_q, plan.cu_seqlens_k,
            plan.max_seqlen_q, plan.max_seqlen_k,
            softmax_scale=self._config.sm_scale, causal=plan.causal,
        )
        # the kernel returns (out, lse) when asked for the log-sum-exp; plain out otherwise
        return out[0] if isinstance(out, tuple) else out


def _round(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class AiterRaggedManager(_AiterRaggedBase, RaggedAttnManager):
    """Self-attention. Each label in the step is its own layout, and each
    segment attends within itself."""

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        super().plan(step, ctx)

    def _plan_layouts(self, step: AttentionStep, ctx: StepContext, plans) -> None:
        for label, segments in self._group_segments_by_label(step.segments or ()).items():
            cu = _cu(segments)
            plans[label] = self._stage(label, cu, cu, step.causal, ctx.slot_lease, ctx.slot)


class AiterRaggedCrossManager(_AiterRaggedBase, RaggedCrossAttnManager):
    """Cross-attention. Each ``(q_label, kv_label)`` pair of the step is one
    plan, run under ``cross_label(q_label, kv_label)``. Never causal."""

    def _plan_layouts(self, step: RaggedCrossAttentionStep, ctx: StepContext, plans) -> None:
        by_label = self._group_segments_by_label(step.segments or ())
        for q_label, kv_label in step.pairs:
            q_segments, kv_segments = by_label[q_label], by_label[kv_label]
            if [s.request_id for s in q_segments] != [s.request_id for s in kv_segments]:
                raise ValueError(
                    f"ragged cross-attention {q_label!r} <- {kv_label!r}: the two labels "
                    "must be declared for the same requests in the same order"
                )
            label = cross_label(q_label, kv_label)
            plans[label] = self._stage(
                label, _cu(q_segments), _cu(kv_segments), False, ctx.slot_lease, ctx.slot,
            )
