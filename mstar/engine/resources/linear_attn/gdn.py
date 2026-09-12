"""Gated delta rule through FlashInfer's GDN kernels.

Two kernels, and a plan picks one for the whole batch. Every row of one token
takes the recurrent decode path, which addresses the state pool directly by
slot. Anything else takes the chunked path, which handles span-1 rows and
zero-span padding alongside long ones — so a mixed batch goes through it whole
rather than being split, and neither kernel needs the tokens re-packed.

The pool is reached two ways, and never as an object: its plan result arrives
through ``ctx.plan_results`` (this resource names it in ``depends_on``), and
its per-layer block arrives as a plain tensor argument to ``run``, the way
``AttentionCallable`` hands ``kv.layer_view()`` to ``attn.run``.
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
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
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

        major = (
            torch.cuda.get_device_capability(device)[0]
            if device.type == "cuda" else 9
        )
        allowed = _PREFILL_STATE_DTYPES.get(major, (torch.float32,))
        # Cast only where the arch cannot read the pool as it stands. A bf16
        # pool on SM90 round-trips through fp32 for prefill and is read
        # directly by decode, so the scatter casts back — see
        # `GDNPrefillWrapper.run`.
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
            # That kernel reads a -1 index as slot 0 and writes there anyway.
            # With no sink, slot 0 belongs to a request and a padding row would
            # land on top of it.
            raise ValueError(
                "a bf16 gdn state needs the pool's sink slot: its decode "
                "kernel writes padding rows to slot 0 rather than skipping "
                "them. Unset RecurrentStateConfig.disable_sink_slot."
            )

    def depends_on(self) -> set[str]:
        # so the pool plans first and its addressing reaches us through
        # `ctx.plan_results`; see `StepRunner.topo_sort`
        return {self._pool_key}

    # Step lifecycle

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        self.reset_default_cursors()
        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]

        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_plan(
                label, segments, addressing[label], ctx
            )
        return self._current

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
        bucket: the captured graph holds the addresses of the wrapper's
        buffers, so one that reallocated on a later, larger layout would leave
        the replay reading freed memory.
        """
        if lease is None:
            # Both walks can run eagerly under one label, and they hold
            # different wrapper types — so `is_decode` has to be in the key.
            key, store = (label, is_decode), self._eager_wrappers
            bs, tok = None, None
        else:
            key = CGSlotKey(lease.bucket, lease.slot, label)
            store = self._cg_wrappers
            bs = max(self._cg_max_bs, lease.bucket.bs)
            tok = lease.bucket.num_tokens

        if key not in store:
            if is_decode:
                store[key] = GDNDecodeWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    bs=bs,
                    cuda_graph=lease is not None,
                )
            else:
                store[key] = GDNPrefillWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    prefill_dtype=self._prefill_dtype,
                    bs=bs,
                    num_tokens=tok,
                    cuda_graph=lease is not None,
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
        # Both walks address every row, so slots are a narrow of the pool's
        # addressing — no gather, and padding rows keep pointing at the sink.
        slots = addressing.slot_indices[:num_rows]

        # Without chunked prefill a batch is either all single-token rows or
        # not, so one walk owns the whole step. Splitting a mixed batch — the
        # faster decode kernel over its single-token rows — would be a change
        # here and not to callers.
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

        ``q``/``k`` are ``[total_tokens, H, K]`` and ``v`` ``[total_tokens, HV,
        V]``. ``a`` and ``b`` are the raw ``[total_tokens, HV]`` gate
        projections, not the decay and the learning rate: the decode kernel
        forms those itself from ``a_log``/``dt_bias``, while the chunked one
        wants them made, so the two are marshalled apart below.

        ``state_layer`` is this layer's ``[max_slots, HV, V, K]`` view of the
        pool — the caller reads it off the pool, as ``AttentionCallable`` reads
        ``layer_view()`` — and is updated in place.

        Returns ``[total_tokens, HV, V]``. Neither path re-packs tokens, so
        there is no output buffer to stitch: each kernel already writes the
        batch in order.

        q and k are L2-normalized here rather than in the kernels. Both take a
        flag for it, but the SM90 prefill kernel silently ignores its one and
        returns NaN for any sequence long enough to matter.
        """
        plan = self.current_plan(label)
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        # both kernels demand contiguous inputs, and a caller that split one
        # projection into q/k/v hands over views that are not. A no-op when
        # they already are.
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        # Both wrappers take the gates raw: prefill forms the decay and the
        # learning rate itself, decode hands them to a kernel that does.
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

        ``x`` is ``[total_tokens, conv_dim]`` and ``conv_layer`` this layer's
        ``[max_slots, conv_dim, width]`` view of the pool's conv block, updated
        in place. ``weight`` is ``[conv_dim, kernel_size]``.

        Splits the same way ``run`` does, and on the same plan. Padding rows
        need nothing special either way: pointed at the sink they write there,
        and with the sink off they carry -1, which is this kernel's own
        ``PAD_SLOT_ID``.
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
        # A floor under every capture wrapper's row count, applied when one is
        # built. The engine announces the widest batch it will capture before
        # any bucket plans, and a wrapper that sized itself to its own bucket
        # alone would reallocate if a wider layout ever reached it.
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._cg_wrappers.clear()
        self._eager_wrappers.clear()
        self._current.clear()
