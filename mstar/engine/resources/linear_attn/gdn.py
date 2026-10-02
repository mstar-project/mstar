"""Gated delta rule through FlashInfer's GDN kernels.

One kernel per batch: all-one-token rows take the recurrent decode path,
anything else the chunked path, which also handles span-1 and zero-span padding.

The pool's plan result arrives through ``ctx.plan_results`` and its per-layer
block as a plain tensor argument to ``run``.
"""

from __future__ import annotations

import logging

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.linear_attn.wrappers import (
    GDNDecodeWrapper,
    GDNPrefillWrapper,
    GDNWrapper,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT, RecurrentAddressing
from mstar.engine.resources.step import Segment, SlotLease, StepContext
from mstar.utils.causal_conv1d import PAD_SLOT_ID

logger = logging.getLogger(__name__)

# What each arch's chunked prefill kernel will read as its packed state.
# SM90 is fp32 only; Blackwell takes the narrower ones, so a bf16 pool is not
# forced through an fp32 round trip there.
_PREFILL_STATE_DTYPES: dict[int, tuple[torch.dtype, ...]] = {
    9: (torch.float32,),
    10: (
        torch.float32, torch.bfloat16, torch.float16,
        torch.float8_e4m3fn, torch.float8_e5m2,
    ),
    12: (torch.float32,),
}


class GDNManager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        state_dtype: torch.dtype,
        has_sink: bool,
        device: torch.device,
    ):
        self.config = config
        self.geometry = geometry
        self.num_layers = num_layers
        self.state_dtype = state_dtype
        self._device = device
        self._pool_key = config.recurrent_state
        self._has_sink = has_sink

        major = (
            torch.cuda.get_device_capability(device)[0]
            if device.type == "cuda" else 9
        )
        allowed = _PREFILL_STATE_DTYPES.get(major, (torch.float32,))
        # Cast only where the arch cannot read the pool as is: a bf16 pool on
        # SM90 round-trips through fp32 for prefill (`GDNPrefillWrapper.run`).
        self._prefill_dtype = (
            state_dtype if state_dtype in allowed else torch.float32
        )
        self._check_state_dtype(state_dtype, geometry, has_sink)

        # See `build_cuda_graph_buffers`.
        self._cg_max_bs = 0
        self._cg_wrappers: dict[CGSlotKey, GDNWrapper] = {}

        # (label, is_decode) -> wrapper
        self._eager_wrappers: dict[tuple[str, bool], GDNWrapper] = {}

        self._current: dict[str, GDNWrapper] = {}

        # Pre-planning mutates no live state, so staging is just caching the
        # result.
        self._preplanned = False
        self._cached_plan_output: dict[str, GDNWrapper] | None = None

    @staticmethod
    def _check_state_dtype(
        state_dtype: torch.dtype, geometry: DeltaNetGeometry, has_sink: bool,
    ) -> None:
        if state_dtype is torch.float32:
            return
        if state_dtype is not torch.bfloat16:
            raise NotImplementedError(
                f"gdn state dtype {state_dtype} is not supported: FlashInfer's "
                "pool decode paths are fp32 and bf16."
            )
        # The bf16 decode kernel is pool-only and K=V=128 only.
        if geometry.head_k_dim != 128 or geometry.head_v_dim != 128:
            raise NotImplementedError(
                f"a bf16 gdn state needs K=V=128, got K={geometry.head_k_dim} "
                f"V={geometry.head_v_dim}; declare an fp32 pool instead."
            )
        from flashinfer import gdn_decode

        if not gdn_decode._GDN_DECODE_BF16_STATE_AVAILABLE:
            raise NotImplementedError(
                "a bf16 gdn state needs FlashInfer's bf16 decode backend, "
                "which this build does not have; declare an fp32 pool instead."
            )
        if not has_sink:
            # That kernel writes a -1 index to slot 0, which without a sink
            # belongs to a request.
            raise ValueError(
                "a bf16 gdn state needs the pool's sink slot: its decode "
                "kernel writes padding rows to slot 0 rather than skipping "
                "them. Unset RecurrentStateConfig.disable_sink_slot."
            )

    def depends_on(self) -> set[str]:
        # the pool plans first; its addressing arrives via `ctx.plan_results`
        return {self._pool_key}

    # Step lifecycle

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "linear-attn preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            self._current = self._cached_plan_output
            self.clear_preplan()
            return self._current

        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]

        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_plan(
                label, segments, addressing[label], ctx
            )
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        # The wrappers' plan state is overwritten by whatever plans next, so
        # an abandoned stage leaves nothing to rewind.
        self._preplanned = False
        self._cached_plan_output = None

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _get_wrapper(
        self, label: str, is_decode: bool, lease: SlotLease | None = None,
    ) -> GDNWrapper:
        """The wrapper this (label, walk) plans into, built once and reused.

        Under capture it is keyed per (bucket, slot, label) and sized to the
        bucket, since the graph holds its buffers' addresses.
        """
        if lease is None:
            # both walks can run eagerly under one label
            key, store = (label, is_decode), self._eager_wrappers
            bs, tok = None, None
        else:
            key = CGSlotKey(lease.bucket, lease.slot, label)
            store = self._cg_wrappers
            bs = max(self._cg_max_bs, lease.bucket.bs)
            tok = lease.bucket.num_tokens

        if key not in store:
            null_slot_id = SINK_SLOT if self._has_sink else NO_SLOT
            if is_decode:
                store[key] = GDNDecodeWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    qk_l2norm=self.config.qk_l2norm,
                    bs=bs,
                    cuda_graph=lease is not None,
                    null_slot_id=null_slot_id,
                )
            else:
                store[key] = GDNPrefillWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    qk_l2norm=self.config.qk_l2norm,
                    prefill_dtype=self._prefill_dtype,
                    bs=bs,
                    num_tokens=tok,
                    cuda_graph=lease is not None,
                    has_sink_state=self._has_sink,
                    null_slot_id=null_slot_id,
                )
        return store[key]

    def _build_plan(
        self,
        label: str,
        segments: list[Segment],
        addressing: RecurrentAddressing,
        ctx: StepContext,
    ) -> GDNWrapper:
        spans = [seg.span for seg in segments]
        num_rows = len(spans)
        # a narrow, not a gather: padding rows keep pointing at the sink
        slots = addressing.slot_indices[:num_rows]

        # One walk owns the whole step; splitting a mixed batch would be a
        # change here, not to callers.
        is_decode = bool(spans) and all(s == 1 for s in spans)
        wrapper = self._get_wrapper(label, is_decode, ctx.slot_lease)
        if is_decode:
            wrapper.plan(spans, slots)
        else:
            wrapper.plan(spans, slots, addressing.has_state[:num_rows])
        return wrapper

    def current_plan(self, label: str | None = None) -> GDNWrapper:
        if label is None:
            label = self._default_label
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"gdn has no plan for label {label!r}; this step planned "
                f"{sorted(self._current)}. Every label a forward runs must "
                "carry a segment in the step declaration."
            )
        return found

    # Submodule-level functionality

    @torch.compiler.disable
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state_layer: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """One layer's gated delta rule over this step's packed tokens.

        ``q``/``k`` are ``[total_tokens, H, K]``, ``v`` ``[total_tokens, HV,
        V]``, and the return ``[total_tokens, HV, V]``. ``state_layer`` is this
        layer's ``[max_slots, HV, V, K]`` view of the pool, updated in place.

        ``a``/``b`` are the raw gate projections, not the decay and learning
        rate. q and k are L2-normalized here only where the kernel will not do
        it (see `GDNWrapper.qk_l2norm_in_kernel`).
        """
        plan = self.current_plan(label)
        if self.config.qk_l2norm and not plan.qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
            k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        # both kernels demand contiguous inputs; a split projection gives views
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        return plan.run(q, k, v, a, b, state_layer, a_log, dt_bias)

    @torch.compiler.disable
    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        label: str | None = None,
    ) -> torch.Tensor:
        """The depthwise conv over ``[q|k|v]``, before the delta rule.

        ``x`` is ``[total_tokens, conv_dim]``, ``weight``
        ``[conv_dim, kernel_size]``, and ``conv_layer`` this layer's
        ``[max_slots, conv_dim, width]`` view of the pool's conv block, updated
        in place. Padding rows carry the pool's null slot, which both conv
        kernels skip.
        """
        return self.current_plan(label).run_conv(
            x=x, conv_layer=conv_layer, weight=weight, bias=bias,
            activation=activation,
        )

    # Engine lifecycle

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        # A floor under every capture wrapper's row count, so none reallocates
        # if a wider layout reaches it; announced before any bucket plans.
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._cg_wrappers.clear()
        self._eager_wrappers.clear()
        self._current.clear()
