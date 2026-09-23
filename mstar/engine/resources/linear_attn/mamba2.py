"""Mamba-2 (SSD) planned against the recurrent state pool.

Same shape as ``gdn.py``: the pool's addressing arrives through
``ctx.plan_results`` and its per-layer blocks as plain tensor arguments, one
wrapper per (label, walk) plans the batch, and the layer reaches both through
``Mamba2Callable``. The conv half reuses ``mstar.utils.causal_conv1d``; the
recurrence is ``mamba2_kernels``: an in-place slot-indexed Triton step for
decode and the sequential scan for prefill.

Decode is the hot path of a full-duplex model (one token per session per tick,
every layer, every 80 ms) and is CUDA-graph safe: the step kernel reads the
pool through the plan's slot buffer and writes it back in place. Prefill is a
system prompt, short and rare, and runs eagerly on gathered state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.linear_attn.mamba2_kernels import (
    PAD_SLOT_ID,
    mamba2_scan,
    mamba2_state_update,
)
from mstar.engine.resources.recurrent.config import Mamba2Geometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.step import Segment, SlotLease, StepContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mamba2Plan:
    """Where this step's rows live and how long each is."""

    slots: torch.Tensor      # [rows] int32, straight off the pool's addressing
    has_state: torch.Tensor  # [rows] bool, False where the slot reads as zeros
    spans: tuple[int, ...]
    num_rows: int

    @property
    def is_decode(self) -> bool:
        return bool(self.spans) and all(s == 1 for s in self.spans)


class Mamba2Manager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: Mamba2Geometry,
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
        self._time_step_limit = tuple(config.time_step_limit)

        self._cg_max_bs = 0
        # label -> this step's plan, for `run`/`run_conv` to read
        self._current: dict[str, Mamba2Plan] = {}
        # Pre-planning: the plan is a narrow of the pool's addressing plus the
        # spans, so staging it is just caching it.
        self._preplanned = False
        self._cached_plan_output: dict[str, Mamba2Plan] | None = None

    def depends_on(self) -> set[str]:
        # so the pool plans first and its addressing reaches us through
        # `ctx.plan_results`; see `StepRunner.topo_sort`
        return {self._pool_key}

    # Step lifecycle

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "mamba2 preplan is already pending; clear_preplan before planning a "
            "different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            self._current = self._cached_plan_output
            self.clear_preplan()
            return self._current

        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]
        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            spans = tuple(seg.span for seg in segments)
            num_rows = len(spans)
            self._current[label] = Mamba2Plan(
                # every row addresses a slot (padding rows keep pointing at the
                # sink, or carry NO_SLOT), so the plan is a narrow of the pool's
                slots=addressing[label].slot_indices[:num_rows],
                has_state=addressing[label].has_state[:num_rows],
                spans=spans,
                num_rows=num_rows,
            )
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        self._preplanned = False
        self._cached_plan_output = None

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def current_plan(self, label: str | None = None) -> Mamba2Plan:
        if label is None:
            label = self._default_label
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"mamba2 has no plan for label {label!r}; this step planned "
                f"{sorted(self._current)}. Every label a forward runs must "
                "carry a segment in the step declaration."
            )
        return found

    # Submodule-level functionality

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
        """The depthwise causal conv over ``[x | B | C]`` (``[total_tokens,
        conv_dim]``), reading and updating this layer's ``[max_slots, conv_dim,
        width]`` pool block in place. Decode rows go through the single-token
        update, anything else through the varlen kernel."""
        from mstar.utils.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        plan = self.current_plan(label)
        if plan.is_decode:
            return causal_conv1d_update(
                x=x, conv_state=conv_layer, weight=weight, bias=bias,
                activation=activation, conv_state_indices=plan.slots,
            )
        cu = [0]
        for span in plan.spans:
            cu.append(cu[-1] + span)
        query_start_loc = torch.tensor(cu, dtype=torch.int32, device=x.device)
        out = causal_conv1d_fn(
            x=x.transpose(0, 1),  # the varlen kernel is feature-major
            weight=weight, bias=bias,
            conv_states=conv_layer,
            query_start_loc=query_start_loc,
            cache_indices=plan.slots,
            has_initial_state=plan.has_state,
            activation=activation,
            pad_slot_id=PAD_SLOT_ID,
            seqlens_cpu=torch.tensor(plan.spans, dtype=torch.int32),
        )
        return out.transpose(0, 1)

    @torch.compiler.disable
    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        ssm_layer: torch.Tensor,
        a_log: torch.Tensor,
        d: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        label: str | None = None,
    ) -> torch.Tensor:
        """One layer's SSD recurrence over this step's packed tokens.

        ``x`` ``[total_tokens, H, P]``, ``dt`` ``[total_tokens, H]`` (raw, the
        bias / softplus / clamp are applied inside), ``b``/``c``
        ``[total_tokens, G, N]``; ``ssm_layer`` is this layer's
        ``[max_slots, H, P, N]`` view of the pool, updated in place. Returns
        ``[total_tokens, H, P]`` in ``x``'s dtype (``D * x`` included).
        """
        plan = self.current_plan(label)
        a = -torch.exp(a_log.float())
        if plan.is_decode:
            return mamba2_state_update(
                x, dt, a, b, c, ssm_layer, plan.slots, d=d, dt_bias=dt_bias,
                dt_softplus=True, time_step_limit=self._time_step_limit,
                pad_slot_id=PAD_SLOT_ID,
            )
        # Prefill: gather each row's state (zeros where it starts fresh), run
        # the sequential scan over its span, scatter the final state back.
        # Eager and per row: a system prompt is short and there is one per
        # request.
        out = torch.empty_like(x)
        slots = plan.slots.tolist()
        has_state = plan.has_state.tolist()
        start = 0
        for span, slot, carried in zip(plan.spans, slots, has_state, strict=True):
            if span <= 0 or slot == PAD_SLOT_ID:
                continue
            h0 = ssm_layer[slot] if carried else None
            y, final = mamba2_scan(
                x[start:start + span], dt[start:start + span], a,
                b[start:start + span], c[start:start + span], h0,
                d=d, dt_bias=dt_bias, dt_softplus=True,
                time_step_limit=self._time_step_limit,
            )
            out[start:start + span] = y
            ssm_layer[slot].copy_(final.to(ssm_layer.dtype))
            start += span
        return out

    # Engine lifecycle

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._current.clear()


# kept for symmetry with gdn.py's per-(bucket, slot, label) wrapper stores; the
# Mamba-2 plan owns no device buffers of its own (it narrows the pool's), so a
# capture bucket needs nothing sized ahead
_ = CGSlotKey, SlotLease
