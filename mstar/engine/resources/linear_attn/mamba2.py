"""Mamba-2 (SSD) planned against the recurrent state pool.

Same shape as ``gdn.py``: the pool's addressing arrives through
``ctx.plan_results`` and its per-layer blocks as plain tensor arguments, one
wrapper per (label, walk) plans the batch, and the layer reaches both through
the contract's ``causal_conv`` and ``mamba2``. The conv half reuses ``mstar.utils.causal_conv1d``; the
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

from mstar.distributed.communication import gather_sequence
from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn.base import (
    LinearAttnManager,
    SplitRows,
    gather_rows,
    scatter_rows,
)
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.linear_attn.mamba2_kernels import (
    PAD_SLOT_ID,
    _effective_dt,
    _expand_groups,
    mamba2_scan,
    mamba2_state_update,
)
from mstar.engine.resources.recurrent.config import Mamba2Geometry
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT, RecurrentAddressing
from mstar.engine.resources.sequence_shard import state_passing, ulysses
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
    supports_split_steps = True
    # the state is the recurrent pool's, which a drop zeroes
    supports_drop = True

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
        # The pool's slot for rows with no request; the conv kernels skip it.
        # Their default of 0 would skip a real request when the sink is off.
        self._null_slot_id = SINK_SLOT if has_sink else NO_SLOT
        self._time_step_limit = tuple(config.time_step_limit)

        self._cg_max_bs = 0
        # label -> this step's plan, for `run`/`run_conv` to read
        self._current: dict[str, Mamba2Plan] = {}
        # label -> this rank's share of it, for a label the step splits over SP ranks
        self._split_rows: dict[str, SplitRows] = {}
        # Pre-planning: the plan is a narrow of the pool's addressing plus the
        # spans, so staging it is just caching it.
        self._preplanned = False
        self._cached_plan_output: tuple[dict[str, Mamba2Plan], dict[str, SplitRows]] | None = None

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
            self._current, self._split_rows = self._cached_plan_output
            self.clear_preplan()
            return self._current

        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]
        self._current, self._split_rows = {}, {}
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
            split = self._plan_split(label, list(spans), addressing[label], ctx)
            if split is not None:
                self._split_rows[label] = split
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current, self._split_rows
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
        label = self._default_label if label is None else label
        if label in self._split_rows:
            return self._conv_split(label, x, conv_layer, weight, bias, activation)
        if plan.is_decode:
            return causal_conv1d_update(
                x=x, conv_state=conv_layer, weight=weight, bias=bias,
                activation=activation, conv_state_indices=plan.slots,
                null_block_id=self._null_slot_id,
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
            null_block_id=self._null_slot_id,
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
        split = self._split_rows.get(self._default_label if label is None else label)
        if split is not None:
            exchange = self._ulysses if self._split_exchange == "ulysses" else self._state_passing
            return exchange(plan, split, x, dt, b, c, ssm_layer, a, d, dt_bias)
        if plan.is_decode:
            return mamba2_state_update(
                x, dt, a, b, c, ssm_layer, plan.slots, d=d, dt_bias=dt_bias,
                dt_softplus=True, time_step_limit=self._time_step_limit,
                pad_slot_id=PAD_SLOT_ID,
            )
        return self._scan_rows(plan, x, dt, b, c, ssm_layer, a, d, dt_bias)

    def _scan_rows(self, plan: Mamba2Plan, x, dt, b, c, ssm_layer, a, d, dt_bias) -> torch.Tensor:
        """Prefill: gather each row's state (zeros where it starts fresh), run the sequential scan
        over its span, scatter the final state back. Eager and per row: a system prompt is short
        and there is one per request. NO_SLOT rows are skipped and read as zeros, like the decode
        kernel's. ``ssm_layer`` may be a block of the pool's heads, with ``a``, ``d`` and
        ``dt_bias`` cut to it."""
        out = torch.zeros_like(x)
        slots = plan.slots.tolist()
        has_state = plan.has_state.tolist()
        start = 0
        for span, slot, carried in zip(plan.spans, slots, has_state, strict=True):
            if span <= 0 or slot == PAD_SLOT_ID:
                start += max(span, 0)
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

    def _ulysses(self, plan: Mamba2Plan, split: SplitRows, x, dt, b, c, ssm_layer, a, d, dt_bias):
        """A split step by head: every token of a block of heads (B/C repeated to one group a
        rank where there are fewer than ranks), the scan over the replica's head block, the
        block's final states gathered into every replica, and the output back by token. It splits
        memory, not time: the scan is a loop over tokens."""
        group = split.seq_range.group
        world = group.world_size
        heads = ssm_layer.shape[1] // world
        lo, hi = group.rank * heads, (group.rank + 1) * heads
        if b.shape[1] % world:
            b, c = (t.repeat_interleave(world // t.shape[1], dim=1) for t in (b, c))

        def run(x, dt, b, c):
            block = ssm_layer[:, lo:hi]
            out = self._scan_rows(
                plan, x, dt, b, c, block, a[lo:hi], None if d is None else d[lo:hi],
                None if dt_bias is None else dt_bias[lo:hi],
            )
            mine = block.index_select(0, plan.slots.long().clamp_min(0))
            scatter_rows(ssm_layer, plan.slots, group.all_gather(mine, dim=1))
            return out

        return ulysses(run, x, dt, b, c, r=split.seq_range)

    def _state_passing(self, plan: Mamba2Plan, split: SplitRows, x, dt, b, c, ssm_layer, a, d, dt_bias):
        """A split step by token (``sequence_shard.state_passing``), as planned (``HandOff``): each
        rank scans every row it holds, from its slot where it starts here and from zeros otherwise,
        and hands off the row it passes on (its final, or for a middle piece its decay
        ``exp(sum dt A)`` and its final from zeros). A row continuing into the rank is then
        corrected in closed form for the state it enters with, ``y_t += C_t . (exp(sum_{s<=t} dt_s
        A) h0)`` and ``final += exp(sum dt A) h0``, and every row's final state is gathered into
        every replica."""
        group, hand = split.seq_range.group, split.hand_off
        decay = _effective_dt(dt, dt_bias, True, self._time_step_limit) * a
        out = torch.zeros_like(x)
        finals: dict[int, torch.Tensor] = {}

        def span(i: int) -> slice:
            return slice(int(hand.offsets[i]), int(hand.offsets[i] + hand.spans[i]))

        for i in hand.rows:
            h0 = None
            if i in hand.starts:
                h0 = gather_rows(ssm_layer, split.slots[i:i + 1], split.has_state[i:i + 1])[0]
            rows = span(i)
            out[rows], finals[i] = mamba2_scan(
                x[rows], dt[rows], a, b[rows], c[rows], h0, d=d, dt_bias=dt_bias, dt_softplus=True,
                time_step_limit=self._time_step_limit,
            )
        heads, head_dim, state_size = ssm_layer.shape[1:]
        transition = torch.ones(heads, device=x.device)
        offset = torch.zeros(heads, head_dim, state_size, device=x.device)
        if hand.passes is not None:
            last = hand.passes
            transition = decay[span(last)].sum(0).exp() if last not in hand.starts else torch.zeros_like(transition)
            offset = finals[last]
        entering = state_passing(group, transition, offset, lambda state, y: state * y[:, None, None])
        if hand.first is not None:
            first = hand.rows[0]
            rows = span(first)
            carried = decay[rows].cumsum(0).exp()
            cx = _expand_groups(c[rows].float(), heads)
            out[rows] = (out[rows].float() + torch.einsum("lh,hpn,lhn->lhp", carried, entering, cx)).to(out.dtype)
            finals[first] = finals[first] + carried[-1][:, None, None] * entering
        local = [finals[i] for i in hand.rows if i in hand.ends]
        mine = torch.stack(local) if local else offset.new_zeros(0, heads, head_dim, state_size)
        scatter_rows(ssm_layer, hand.ended_slots, gather_sequence(group, mine, hand.counts))
        return out

    # Engine lifecycle

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._current.clear()
        self._split_rows.clear()


# kept for symmetry with gdn.py's per-(bucket, slot, label) wrapper stores; the
# Mamba-2 plan owns no device buffers of its own (it narrows the pool's), so a
# capture bucket needs nothing sized ahead
_ = CGSlotKey, SlotLease
