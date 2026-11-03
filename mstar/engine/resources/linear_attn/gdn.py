"""Gated delta rule through FlashInfer's GDN kernels.

One kernel per batch: all-one-token rows take the recurrent decode path,
anything else the chunked path, which also handles span-1 and zero-span padding.

The pool's plan result arrives through ``ctx.plan_results`` and its per-layer
block as a plain tensor argument to ``run``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import torch

from mstar.distributed.communication import gather_sequence
from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, Resource
from mstar.engine.resources.linear_attn.base import (
    LinearAttnManager,
    SplitRows,
    finals_by_rank,
    gather_rows,
    scatter_rows,
)
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.linear_attn.gdn_stage import gdn_stage_applies
from mstar.engine.resources.linear_attn.wrappers import (
    GDNDecodeWrapper,
    GDNMixedWrapper,
    GDNPrefillWrapper,
    GDNWrapper,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import (
    NO_SLOT,
    SINK_SLOT,
    RecurrentAddressing,
    RecurrentStatePool,
    checkpoint_rows,
)
from mstar.engine.resources.sequence_shard import state_passing, ulysses
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import Segment, SlotLease, StepContext
from mstar.utils.causal_conv1d import PAD_SLOT_ID
from mstar.utils.h2d import to_device_nonblocking

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
    supports_split_steps = True

    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        state_dtype: torch.dtype,
        has_sink: bool,
        device: torch.device,
        max_rows_in_combined: Mapping[str, int | None] | None = None,
    ):
        self.config = config
        self._splits = {split.combined_walk: split for split in config.splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
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
        self._stage_prefill = gdn_stage_applies(device, geometry)

        # See `build_cuda_graph_buffers`.
        self._cg_max_bs = 0
        self._cg_max_tokens = 0
        self._cg_wrappers: dict[CGSlotKey, GDNWrapper] = {}
        # The prefills' state checkpoints, sized once to the largest bucket and
        # shared: one forward runs at a time, and each graph holds the address.
        # An eager step fits it (`RecurrentStatePool._checkpoint_every`), and
        # without graphs it grows to the largest step.
        self._checkpoints: torch.Tensor | None = None
        self._pool: RecurrentStatePool | None = None

        # (label, is_decode) -> wrapper
        self._eager_wrappers: dict[tuple[str, bool], GDNWrapper] = {}

        self._current: dict[str, GDNWrapper] = {}
        # label -> this rank's share of it, for a label the step splits over SP ranks
        self._split_rows: dict[str, SplitRows] = {}

        # Pre-planning mutates no live state, so staging is just caching the
        # result.
        self._preplanned = False
        self._cached_plan_output: tuple[dict[str, GDNWrapper], dict[str, SplitRows]] | None = None

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

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        super().bind_dependencies(resources)
        self._pool = resources.get(self._pool_key)

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
            self._current, self._split_rows = self._cached_plan_output
            self.clear_preplan()
            return self._current

        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]

        self._current, self._split_rows = {}, {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            wrapper = self._current[label] = self._build_plan(
                label, segments, addressing[label], ctx
            )
            # an eager split walk's step runs its leading decode rows apart (`GDNMixedWrapper`)
            run = wrapper.halves[2] if isinstance(wrapper, GDNMixedWrapper) else 0
            split = self._plan_split(label, [seg.span for seg in segments], addressing[label], ctx, run)
            if split is not None:
                # a leased step plans copied where a linear layer splits (`captures_split_steps`)
                assert ctx.slot_lease is None, "a leased step split for a linear attention"
                self._split_rows[label] = split
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current, self._split_rows
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
        split: CombinedSplit | None = None,
    ) -> GDNWrapper:
        """The wrapper this (label, walk) plans into, built once and reused.

        Under capture it is keyed per (bucket, slot, label) and sized to the
        bucket, since the graph holds its buffers' addresses.
        """
        if lease is None:
            # both walks can run eagerly under one label
            key, store = (label, "split" if split is not None else is_decode), self._eager_wrappers
            bs, tok = None, None
        else:
            key = CGSlotKey(lease.bucket, lease.slot, label)
            store = self._cg_wrappers
            # A split's decode half runs on as many rows as it is sized for,
            # so it takes the bucket's; the floor is for the other wrappers,
            # whose kernels run on the step's rows whatever their buffers hold.
            bs = lease.bucket.bs if split is not None else max(self._cg_max_bs, lease.bucket.bs)
            tok = lease.bucket.num_tokens

        if key not in store:
            null_slot_id = SINK_SLOT if self._has_sink else NO_SLOT
            if split is not None:
                store[key] = GDNMixedWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    qk_l2norm=self.config.qk_l2norm,
                    prefill_dtype=self._prefill_dtype,
                    bs=bs,
                    num_tokens=tok,
                    accelerator_graph=lease is not None,
                    has_sink_state=self._has_sink,
                    null_slot_id=null_slot_id,
                    max_prefill_rows=(
                        split.max_prefill_rows(self._max_rows_in_combined, lease.bucket.bs)
                        if lease is not None else None
                    ),
                    stage_inputs=self._stage_prefill,
                )
            elif is_decode:
                store[key] = GDNDecodeWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    sm_scale=self.config.sm_scale,
                    qk_l2norm=self.config.qk_l2norm,
                    bs=bs,
                    accelerator_graph=lease is not None,
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
                    accelerator_graph=lease is not None,
                    has_sink_state=self._has_sink,
                    null_slot_id=null_slot_id,
                    stage_inputs=self._stage_prefill,
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

        # One kernel for the step: the chunked one unless every row is one
        # token, or always under a packed capture, which replays mixed steps.
        # A split walk's step (decode rows beside prompt rows) takes both.
        is_decode = not ctx.force_prefill and bool(spans) and all(s == 1 for s in spans)
        walk = ctx.slot_lease.bucket.graph_walk if ctx.slot_lease is not None else ctx.graph_walk
        split = self._splits.get(walk)
        every = addressing.snapshot_every
        walks = split.prefill_walks if split is not None else (walk,)
        if every and self._pool.keyed_walks(label).isdisjoint(walks):
            # a walk the KV cache doesn't key keeps no snapshots, so its graphs skip the checkpoints too
            every = 0
        assert every or not addressing.snapshots, (
            f"gdn: a {walk!r} step keeps snapshots for {label!r}, which its KV cache doesn't key there"
        )
        checkpoints = None
        if every and ctx.slot_lease is not None:
            checkpoints = self._checkpoint_buffer(checkpoint_rows(self._cg_max_tokens, self._cg_max_bs, every))
        elif every and addressing.snapshots:
            checkpoints = self._checkpoint_buffer(sum(max(span, 1) // every for span in spans))
        if split is not None and addressing.host_slot_indices is not None:
            wrapper = self._get_wrapper(label, False, ctx.slot_lease, split=split)
            wrapper.plan(
                spans, addressing.host_slot_indices[:num_rows],
                addressing.host_has_state[:num_rows],
                addressing.snapshots, every, checkpoints,
            )
            return wrapper
        wrapper = self._get_wrapper(label, is_decode, ctx.slot_lease)
        if is_decode:
            wrapper.plan(spans, slots)
        else:
            wrapper.plan(
                spans, slots, addressing.has_state[:num_rows],
                addressing.snapshots, every, checkpoints,
            )
        return wrapper

    def _checkpoint_buffer(self, rows: int) -> torch.Tensor:
        if self._checkpoints is None or self._checkpoints.shape[0] < rows:
            geometry = self.geometry
            self._checkpoints = torch.empty(
                (rows, geometry.num_v_heads, geometry.head_v_dim, geometry.head_k_dim),
                dtype=torch.float32, device=self._device,
            )
        return self._checkpoints

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
        split = self._split_rows.get(self._default_label if label is None else label)
        # a split step's hand-off runs the chunked kernel itself, which normalizes nothing
        in_kernel = plan.qk_l2norm_in_kernel and (split is None or self._split_exchange == "ulysses")
        if self.config.qk_l2norm and not in_kernel:
            q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
            k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        # a split projection gives column views; copy only for a kernel that
        # needs them contiguous
        if not plan.takes_strided_inputs:
            v = v.contiguous()
            a = a.contiguous()
            b = b.contiguous()
        if split is None:
            return plan.run(q, k, v, a, b, state_layer, a_log, dt_bias)
        decode, prefill, _ = plan.halves if isinstance(plan, GDNMixedWrapper) else (None, plan, 0)
        if split.run:
            # the decode run, copied: every rank runs it on every head, in place on its replica
            decode = decode.run(*self._decode_slab(split, q, k, v, a, b), state_layer, a_log, dt_bias)
        else:
            decode = None
        exchange = self._ulysses if self._split_exchange == "ulysses" else self._state_passing
        out = exchange(prefill, split, q, k, v, a, b, state_layer, a_log, dt_bias)
        if decode is not None:
            # this rank's tokens of the run lead its share
            rank, counts = split.range.group.rank, split.decode_tokens
            start = sum(counts[:rank])
            out[: counts[rank]] = decode[start:start + counts[rank]].to(out.dtype)
        return out

    def _ulysses(self, prefill: GDNWrapper, split: SplitRows, q, k, v, a, b, state_layer, a_log, dt_bias):
        """A split step's prompt rows by head (``sequence_shard.ulysses``): every token of a block
        of heads, the chunked half run on it over the replica's block of the state, and the
        block's final states gathered into every replica. The decode run's tokens ride the
        exchange; the chunked half (planned eagerly past the run) doesn't read them."""
        group = split.range.group
        heads = state_layer.shape[1] // group.world_size
        lo, hi = group.rank * heads, (group.rank + 1) * heads
        front, slots = split.run, split.prompt_slots

        def run(q, k, v, a, b):
            if not split.has_prompt:
                # nothing past the run (from the shared plan, so alike on every rank)
                return torch.zeros_like(v)
            block = state_layer[:, lo:hi]
            out = prefill.run(
                q[front:], k[front:], v[front:], a[front:], b[front:], block, a_log[lo:hi], dt_bias[lo:hi],
            )
            finals = group.all_gather(block.index_select(0, slots.long().clamp_min(0)), dim=1)
            scatter_rows(state_layer, slots, finals)
            return torch.cat([out.new_zeros(front, *out.shape[1:]), out]) if front else out

        return ulysses(run, q, k, v, a, b, r=split.range)

    def _state_passing(self, prefill: GDNWrapper, split: SplitRows, q, k, v, a, b, state_layer, a_log, dt_bias):
        """A split step's prompt rows by token (``sequence_shard.state_passing``): each rank runs
        the rows starting on it from their slots, hands off the row it passes on (its final, or for
        a middle piece the piece's transition from the identity with no values and its offset from
        zeros, which needs equal key and value dims), runs a row continuing into it from the state
        it gets, and every row's final state is gathered into every replica."""
        group, rank = split.range.group, split.range.group.rank
        pieces, run = split.pieces, split.run
        mine = pieces[rank]
        offsets = np.cumsum(mine) - mine
        rows = [i for i in range(run, len(mine)) if mine[i]]
        starts = {i: not pieces[:rank, i].any() for i in rows}
        ends = {i: not pieces[rank + 1:, i].any() for i in rows}
        alpha = torch.exp(-torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float()))
        beta = torch.sigmoid(b.float())
        heads, dim_v, dim_k = state_layer.shape[1:]
        out = v.new_zeros(v.shape[0], heads, dim_v)
        finals: dict[int, torch.Tensor] = {}

        def chunked(chosen: list[int], initial: torch.Tensor, zero_values: bool = False) -> torch.Tensor:
            # the chunked kernel over `chosen` rows' pieces here, a sequence each (twice over with
            # `zero_values`: the second time with no values); never on an empty set, on which the
            # kernel deadlocks while the other ranks wait in the hand-off
            index = np.concatenate([np.arange(offsets[i], offsets[i] + mine[i]) for i in chosen])
            spans = [mine[i] for i in chosen]
            if zero_values:
                index, spans = np.concatenate([index, index]), spans * 2
            index_t = to_device_nonblocking(index, q.device, torch.int64)
            values = v.index_select(0, index_t)
            if zero_values:
                values[len(index) // 2:] = 0
            cu = to_device_nonblocking(np.concatenate([[0], np.cumsum(spans)]), q.device, torch.int64)
            got, final = _chunked_kernel(q.device)(
                q=q.index_select(0, index_t), k=k.index_select(0, index_t), v=values,
                g=alpha.index_select(0, index_t), beta=beta.index_select(0, index_t), scale=self.config.sm_scale,
                initial_state=initial.to(self._prefill_dtype), output_final_state=True, cu_seqlens=cu,
                use_qk_l2norm_in_kernel=False,
            )
            if not zero_values:
                for i, start in zip(chosen, np.cumsum([0, *spans[:-1]]), strict=True):
                    out[offsets[i]:offsets[i] + mine[i]] = got[start:start + mine[i]].to(out.dtype)
                    finals[i] = final[chosen.index(i)].float()
            return final.float()

        fresh = [i for i in rows if starts[i]]
        if fresh:
            index = to_device_nonblocking(fresh, q.device, torch.int64)
            chunked(fresh, gather_rows(state_layer, split.slots[index], split.has_state[index]))
        identity = torch.eye(dim_k, device=q.device).expand(heads, dim_k, dim_k)
        transition, offset = identity, state_layer.new_zeros(heads, dim_v, dim_k, dtype=torch.float32)
        if rows and not ends[rows[-1]]:
            last = rows[-1]
            if starts[last]:
                transition, offset = torch.zeros_like(identity), finals[last]
            else:
                # a middle piece: its offset from zeros, its transition from the identity with no values
                offset, transition = chunked([last], torch.stack([torch.zeros_like(identity), identity]), True)
        entering = state_passing(group, transition, offset, torch.matmul)
        if rows and not starts[rows[0]]:
            chunked([rows[0]], entering[None])
        # every prompt row's final state, gathered from the rank it ends on
        ended, counts = finals_by_rank(pieces, run)
        local = [finals[i] for i in rows if ends[i]]
        mine_finals = torch.stack(local) if local else offset.new_zeros(0, heads, dim_v, dim_k)
        slots = split.slots.index_select(0, to_device_nonblocking(ended, q.device, torch.int64))
        scatter_rows(state_layer, slots, gather_sequence(group, mine_finals, counts))
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

        ``x`` is ``[total_tokens, conv_dim]``, ``weight``
        ``[conv_dim, kernel_size]``, and ``conv_layer`` this layer's
        ``[max_slots, conv_dim, width]`` view of the pool's conv block, updated
        in place. Padding rows carry the pool's null slot, which both conv
        kernels skip.
        """
        label = self._default_label if label is None else label
        if label in self._split_rows:
            return self._conv_split(label, x, conv_layer, weight, bias, activation)
        return self.current_plan(label).run_conv(
            x=x, conv_layer=conv_layer, weight=weight, bias=bias,
            activation=activation,
        )

    # Engine lifecycle

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots
        # A floor under every capture wrapper's row count, so none reallocates
        # if a wider layout reaches it; announced before any bucket plans.
        self._cg_max_bs = max(self._cg_max_bs, max_bs)
        self._cg_max_tokens = max(self._cg_max_tokens, max_seq_len)

    def cleanup(self):
        self._cg_wrappers.clear()
        self._eager_wrappers.clear()
        self._current.clear()
        self._split_rows.clear()
        self._checkpoints = None


def _chunked_kernel(device: torch.device):
    """FlashInfer's chunked GDN prefill, or its torch stand-in on a CPU tensor."""
    if device.type == "cpu":
        from mstar.engine.resources.linear_attn.gdn_cpu import chunk_gated_delta_rule
    else:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    return chunk_gated_delta_rule
