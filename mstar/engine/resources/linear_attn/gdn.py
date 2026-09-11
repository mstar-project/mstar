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
from dataclasses import dataclass

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.step import Segment, StepContext

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


@dataclass(frozen=True)
class GDNDecodePlan:
    """Every row is one token; row i is token i."""

    slots: torch.Tensor  # [n] int32, straight off the pool's addressing
    num_rows: int


@dataclass(frozen=True)
class GDNPrefillPlan:
    """Rows of any span, including 1 and 0."""

    slots: torch.Tensor       # [n] int32
    cu_seqlens: torch.Tensor  # [n + 1] int32
    has_state: torch.Tensor   # [n] bool, False where the slot reads as zeros
    num_rows: int


@dataclass(frozen=True)
class GDNPlan:
    """One label's layout for this step.

    Exactly one half is set today: without chunked prefill a batch is either
    all single-token rows or not, and the chunked kernel takes the rest whole.
    The pair is kept so a later split — running the faster decode kernel over
    the single-token rows of a mixed batch — is a change here and not to
    callers.
    """

    decode: GDNDecodePlan | None
    prefill: GDNPrefillPlan | None
    total_tokens: int


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
        # directly by decode, so the scatter casts back (see `_run_prefill`).
        self._prefill_dtype = (
            state_dtype if state_dtype in allowed else torch.float32
        )
        self._check_state_dtype(state_dtype, geometry, has_sink)

        self._cg_max_bs = 0
        self._cg_plans: dict[CGSlotKey, torch.Tensor] = {}
        self._eager_plans: dict[str, torch.Tensor] = {}
        self._current: dict[str, GDNPlan] = {}

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

    def _build_plan(
        self,
        label: str,
        segments: list[Segment],
        addressing: RecurrentAddressing,
        ctx: StepContext,
    ) -> GDNPlan:
        spans = [seg.span for seg in segments]
        num_rows = len(spans)
        total = sum(spans)
        # Both halves address every row, so slots are a narrow of the pool's
        # addressing — no gather, and padding rows keep pointing at the sink.
        slots = addressing.slot_indices[:num_rows]

        if spans and all(s == 1 for s in spans):
            return GDNPlan(
                decode=GDNDecodePlan(slots=slots, num_rows=num_rows),
                prefill=None,
                total_tokens=total,
            )

        cu = [0]
        for span in spans:
            cu.append(cu[-1] + span)
        buf = self._cu_buffer(label, ctx, num_rows)
        buf[: len(cu)].copy_(
            torch.tensor(
                cu, dtype=torch.int32, pin_memory=torch.cuda.is_available()
            ),
            non_blocking=True,
        )
        return GDNPlan(
            decode=None,
            prefill=GDNPrefillPlan(
                slots=slots,
                cu_seqlens=buf[: len(cu)],
                has_state=addressing.has_state[:num_rows],
                num_rows=num_rows,
            ),
            total_tokens=total,
        )

    def _cu_buffer(
        self, label: str, ctx: StepContext, num_rows: int,
    ) -> torch.Tensor:
        """The cu_seqlens buffer this step stages into.

        Under capture it is static and per (bucket, slot, label), built on the
        first plan for that key and sized to the bucket rather than to this
        plan's row count — a bucket replays at several layouts.
        """
        lease = ctx.slot_lease
        if lease is None:
            key, store, rows = label, self._eager_plans, num_rows
        else:
            key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
            store = self._cg_plans
            rows = max(self._cg_max_bs, lease.bucket.bs, num_rows)

        found = store.get(key)
        if found is None or found.numel() < rows + 1:
            found = store[key] = torch.zeros(
                rows + 1, dtype=torch.int32, device=self._device
            )
        return found

    def current_plan(self, label: str | None = None) -> GDNPlan:
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

        if plan.decode is not None:
            return self._run_decode(
                plan.decode, q, k, v, a, b, state_layer, a_log, dt_bias
            )
        return self._run_prefill(
            plan.prefill, q, k, v, a, b, state_layer, a_log, dt_bias
        )

    def _run_decode(self, plan, q, k, v, g, beta, state, a_log, dt_bias):
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        # row i is token i, so the token axis is just unsqueezed
        out, _ = gated_delta_rule_decode_pretranspose(
            q=q.unsqueeze(1), k=k.unsqueeze(1), v=v.unsqueeze(1),
            state=None,
            A_log=a_log,
            a=g.unsqueeze(1),
            dt_bias=dt_bias,
            b=beta.unsqueeze(1),
            scale=self.config.sm_scale,
            initial_state=state,
            initial_state_indices=plan.slots,
            use_qk_l2norm=False,
        )
        return out.squeeze(1)

    def _run_prefill(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        # The chunked kernel takes the decay and the learning rate already
        # formed, where the decode kernel takes `a`/`b` raw and forms them from
        # the same two weights. Keep the formula here so the paths agree.
        g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
            a.float() + dt_bias.float()
        )
        beta = torch.sigmoid(b.float())

        slots = plan.slots.to(torch.int64)
        # SM90 takes packed, sequence-ordered state — `state_indices` is
        # SM100/SM103 only — so gather here and scatter back. Once per prefill
        # step rather than per token, and prefill stays eager-cheap.
        initial = torch.index_select(state, 0, slots).to(self._prefill_dtype)
        # zero the rows that start fresh, by multiply rather than boolean mask:
        # `initial[~mask] = 0` is a data-dependent shape and cannot be captured
        initial.mul_(plan.has_state.to(initial.dtype).view(-1, 1, 1, 1))

        out, final = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            # FlashInfer wants the decay exponentiated; FLA's takes log space
            g=torch.exp(g),
            beta=beta,
            scale=self.config.sm_scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=plan.cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
        state.index_copy_(0, slots, final.to(state.dtype))
        return out

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
        from mstar.utils.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        plan = self.current_plan(label)
        if plan.decode is not None:
            return causal_conv1d_update(
                x=x,
                conv_state=conv_layer,
                weight=weight,
                bias=bias,
                activation=activation,
                conv_state_indices=plan.decode.slots,
            )
        # the varlen kernel is feature-major
        out = causal_conv1d_fn(
            x=x.transpose(0, 1),
            weight=weight,
            bias=bias,
            conv_states=conv_layer,
            query_start_loc=plan.prefill.cu_seqlens,
            cache_indices=plan.prefill.slots,
            has_initial_state=plan.prefill.has_state,
            activation=activation,
        )
        return out.transpose(0, 1)

    # Engine lifecycle

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._cg_plans.clear()
        self._eager_plans.clear()
        self._current.clear()
