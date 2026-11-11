"""The linear-attention resource's spec-time factory, and what its managers share on a node whose
steps split over SP ranks.

``LinearAttnManager.build`` picks the manager for the configured variant (`gdn`,
`kda` or `mamba2`), by deferred import so naming one does not load the others.

On a split step (``sequence_shard``) each rank holds a contiguous range of the step's tokens. The
recurrent pool stays copied over SP: every rank holds every head its TP rank does, and after each
step every SP replica of a slot holds the same bytes. The conv runs on each rank's own tokens with
the left context the earlier ranks hold (``conv_halo``); the recurrence's prompt rows take the
manager's exchange (``ulysses`` or ``state_passing``), its leading one-token decode rows run copied.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import gather_sequence
from mstar.engine.resources.base import AttentionResource, EngineResourceInfo, Resource
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnSpec,
    LinearAttnVariant,
    LinearSpExchange,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, Mamba2Geometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.sequence_shard import (
    ConvHalo,
    SeqRange,
    SequenceShard,
    check_shard_plan,
    conv_halo,
    conv_halo_plan,
)
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import to_device_nonblocking

if TYPE_CHECKING:
    from mstar.engine.resources.linear_attn.wrappers import ConvMetadata

logger = logging.getLogger(__name__)


class Chunk(NamedTuple):
    """One call of the chunked kernel over some prompt rows' pieces on this rank, a sequence each:
    their tokens (``index`` into this rank's) and ``cu_seqlens``, on the device."""

    rows: tuple[int, ...]
    index: torch.Tensor
    cu: torch.Tensor


class HandOff(NamedTuple):
    """A ``state_passing`` split's prompt rows on this rank, the same in every layer, so planned
    once a step: the ``rows`` with a piece here, in order, those that ``start`` or ``end`` here,
    and each row's piece (``offsets`` into this rank's tokens, ``spans``); the chunked calls,
    ``fresh`` (the rows starting here, from their slots ``fresh_slots``), ``first`` (a row
    continuing into this rank, from the state it enters with) and ``middle`` (the last row twice
    over where it neither starts nor ends here: its offset and transition); and every prompt
    row's slot in the order the finals gather lays them out, ``counts`` a rank."""

    rows: tuple[int, ...]
    starts: frozenset[int]
    ends: frozenset[int]
    offsets: np.ndarray
    spans: np.ndarray
    fresh: Chunk | None
    fresh_slots: torch.Tensor | None
    fresh_has_state: torch.Tensor | None
    first: Chunk | None
    middle: Chunk | None
    ended_slots: torch.Tensor
    counts: list[int]

    @property
    def passes(self) -> int | None:
        """The row this rank hands on (its last, not ending here), or None: it hands on nothing."""
        return self.rows[-1] if self.rows and self.rows[-1] not in self.ends else None


class SplitRows(NamedTuple):
    """A split step's label on this rank: its ranges, each rank's piece of each row
    (``SeqRange.row_pieces``), the conv halo on the device, the local conv's layout, and the
    rows' slots and whether each carries state. ``run`` leading rows are one-token decode rows,
    which run copied. ``hand_off`` is the prompt rows' plan under ``state_passing``."""

    seq_range: SeqRange
    pieces: np.ndarray
    halo: ConvHalo
    # the rows with a piece here, as the conv kernel lays them out, each from its scratch window
    conv_cu: torch.Tensor
    conv_spans: torch.Tensor
    conv_metadata: ConvMetadata
    conv_windows: torch.Tensor
    conv_has_state: torch.Tensor
    slots: torch.Tensor
    has_state: torch.Tensor
    run: int
    hand_off: HandOff | None

    @property
    def decode_tokens(self) -> tuple[int, ...]:
        """Each rank's tokens of the decode run, which every rank gathers."""
        return tuple(self.pieces[:, : self.run].sum(1).tolist())

    @property
    def prompt_slots(self) -> torch.Tensor:
        """The slots of the rows past the decode run."""
        return self.slots[self.run:]

    @property
    def has_prompt(self) -> bool:
        """Whether any row past the decode run holds a token (a step of zero-span rows behind the
        run would leave the chunked kernel nothing, on which it deadlocks)."""
        return bool(self.pieces[:, self.run:].any())


class LinearAttnManager(AttentionResource):
    # Abstract except for `build`, which dispatches on the variant.
    _pool_key: str
    # the sequence shard over its node (`LinearAttnSpec.derive`), set by `build`; and the shard
    # itself where it can split (`bind_dependencies`)
    _shard_key: str | None = None
    _shard: SequenceShard | None = None
    # how a split step's prompt rows reach the recurrence (`LinearAttnSpec.split_exchange`), read only
    # where the node's shard splits (`bind_dependencies`)
    _split_exchange: LinearSpExchange = "ulysses"
    # label -> this rank's share of it, for each label this step splits (the managers' `plan`)
    _split_rows: dict[str, SplitRows]

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # from the pool's unsharded head counts, so alike on every rank, before the build shards them
        group = info.joint_comm_group
        tp, sp = (1, 1) if group is None else (group.tp_group.world_size, group.sp_group.world_size)
        exchange = spec.split_exchange(info.dependency(spec.config.recurrent_state).config, tp, sp)
        manager = cls._build_variant(spec, info)
        manager._shard_key = spec.shard_key
        manager._split_exchange = exchange
        return manager

    @classmethod
    def _build_variant(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; `shard` is idempotent so both builders can call it.
        # Heads over TP alone, the pool's being copied over SP ranks (`head_shards`)
        pool_config = info.dependency(spec.config.recurrent_state).config
        pool_config.shard(info.head_shards)

        backend = spec.config.backend
        variant = spec.config.variant
        if variant is LinearAttnVariant.MAMBA2:
            if backend is not LinearAttnBackend.FLASHINFER:
                raise ValueError(f"Unknown linear attention backend {backend!r}")
            # Its own kernels (Triton), planned against the same pool.
            from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager

            return Mamba2Manager(
                config=spec.config,
                geometry=Mamba2Geometry.from_blocks(pool_config.blocks),
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["ssm"].resolved_dtype(info.kv_dtype),
                has_sink=not pool_config.disable_sink_slot,
                device=info.device,
            )

        # also checks the pool's shapes: `from_blocks` raises on another family's
        geometry = DeltaNetGeometry.from_blocks(pool_config.blocks)
        if variant is LinearAttnVariant.KDA:
            if backend is LinearAttnBackend.FLASHINFER:
                # the config's default, which is GDN's; KDA has no FlashInfer path
                backend = spec.config.backend = LinearAttnBackend.TRITON
            if backend is not LinearAttnBackend.TRITON:
                raise ValueError(f"KDA runs on the Triton backend, not {backend!r}")
            if pool_config.disable_sink_slot:
                raise ValueError(
                    "KDA needs the pool's sink slot: its kernels address padding "
                    "rows there. Unset RecurrentStateConfig.disable_sink_slot."
                )
            from mstar.engine.resources.linear_attn.kda import KDAManager

            return KDAManager(
                config=spec.config,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["state"].resolved_dtype(info.kv_dtype),
                device=info.device,
                speculative_tokens=DeltaNetGeometry.speculative_tokens_of(pool_config.blocks),
                conv_dtype=pool_config.blocks["conv"].resolved_dtype(info.kv_dtype),
            )

        if backend is not LinearAttnBackend.FLASHINFER:
            raise ValueError(f"Unknown linear attention backend {backend!r}")
        if variant is LinearAttnVariant.GDN:
            from mstar.engine.resources.linear_attn.gdn import GDNManager

            return GDNManager(
                config=spec.config,
                max_rows_in_combined=info.max_rows_in_combined,
                geometry=geometry,
                num_layers=pool_config.num_layers,
                state_dtype=pool_config.blocks["state"].resolved_dtype(info.kv_dtype),
                has_sink=not pool_config.disable_sink_slot,
                device=info.device,
            )
        raise ValueError(f"Unknown linear attention variant {variant!r}")

    def depends_on(self) -> set[str]:
        # the pool plans first, its addressing arriving via `ctx.plan_results`; and the shard,
        # whose plan says whether a step splits
        keys = {self._pool_key}
        if self._shard_key is not None:
            keys.add(self._shard_key)
        return keys

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        if self._shard_key is None or not resources[self._shard_key].can_split:
            return
        self._shard = resources[self._shard_key]
        logger.info(
            "linear attention over pool %r takes split steps over %d SP ranks: prompt rows by %s",
            self._pool_key, self._shard.group.world_size, self._split_exchange,
        )

    def _plan_split(
        self, label: str, spans: list[int], addressing: RecurrentAddressing, ctx: StepContext, run: int = 0,
    ) -> SplitRows | None:
        """``label``'s split on this rank, or None where the step runs it copied. Raises, at every
        SP size, for a step that doesn't declare the node's shard."""
        from mstar.engine.resources.linear_attn.wrappers import conv_metadata

        if self._shard_key is None:
            return None
        plan = check_shard_plan(f"linear attention over pool {self._pool_key!r}", self._shard_key, ctx)
        r = plan.ranges.get(label) if self._shard is not None and plan.split else None
        if r is None:
            return None
        # the node never splits a captured step (`SeqShardSpec.eager`)
        assert ctx.slot_lease is None, "a leased step split for a linear attention"
        if addressing.snapshots:
            # the pool keeps snapshots at one rank only (`RecurrentStateSpec.keeps_prefix_cache`)
            raise ValueError(f"a split step of {label!r} keeps prefix snapshots, which split steps don't write")
        pieces = r.row_pieces(spans)
        taps = self.conv_taps
        host = conv_halo_plan(pieces, r.group.rank, taps)
        device = self._device
        rows, local = len(spans), len(host.rows)
        slots = addressing.slot_indices[:rows]
        return SplitRows(
            seq_range=r,
            pieces=pieces,
            halo=host.to(device),
            conv_cu=to_device_nonblocking(np.concatenate([[0], np.cumsum(host.spans)]), device, torch.int32),
            conv_spans=torch.from_numpy(host.spans.astype(np.int32)),
            conv_metadata=conv_metadata(host.spans.tolist(), device),
            conv_windows=torch.arange(local, dtype=torch.int32, device=device),
            conv_has_state=torch.ones(local, dtype=torch.bool, device=device),
            slots=slots,
            has_state=addressing.has_state[:rows],
            run=run,
            hand_off=(
                _hand_off(pieces, r.group.rank, run, slots, addressing.has_state[:rows], device)
                if self._split_exchange == "state_passing" else None
            ),
        )

    @property
    def conv_taps(self) -> int:
        """The inputs the conv window keeps: its kernel size less one."""
        return self.geometry.conv_kernel_size - 1

    def _conv_split(
        self, label: str, x: torch.Tensor, conv_layer: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None, activation: str | None,
    ) -> torch.Tensor:
        """The conv over this rank's tokens ``x`` of split ``label`` (``conv_halo``): every row's
        final window written into this replica of its slot."""
        from mstar.utils.causal_conv1d import causal_conv1d_fn

        split = self._split_rows[label]
        # before the collective, alike on every rank
        self._shard.check_scattered(label)

        def conv(x: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
            # the rows with a piece here, each from its scratch window
            return causal_conv1d_fn(
                x=x.transpose(0, 1), weight=weight, bias=bias, conv_states=initial,
                query_start_loc=split.conv_cu, cache_indices=split.conv_windows,
                has_initial_state=split.conv_has_state, activation=activation, pad_slot_id=-1,
                null_block_id=-1, seqlens_cpu=split.conv_spans, metadata=split.conv_metadata,
            ).transpose(0, 1)

        stored = gather_rows(conv_layer, split.slots, split.has_state)
        out, final = conv_halo(conv, x, stored, split.halo, split.seq_range.group)
        scatter_rows(conv_layer, split.slots, final)
        return out

    def _decode_slab(self, split: SplitRows, *xs: torch.Tensor) -> list[torch.Tensor]:
        """The step's decode run, every rank's tokens of it gathered: the first tokens of each of
        ``xs``, this rank's, and the whole run back, one collective per dtype sized by the step."""
        counts = split.decode_tokens
        mine = counts[split.seq_range.group.rank]
        out: list[torch.Tensor | None] = [None] * len(xs)
        for dtype in dict.fromkeys(x.dtype for x in xs):
            chosen = [i for i, x in enumerate(xs) if x.dtype == dtype]
            flat = torch.cat([xs[i][:mine].flatten(1) for i in chosen], dim=1)
            whole = gather_sequence(split.seq_range.group, flat, counts)
            parts = whole.split([xs[i][0].numel() for i in chosen], dim=1)
            for i, part in zip(chosen, parts, strict=True):
                out[i] = part.unflatten(1, xs[i].shape[1:]).contiguous()
        return out


def _hand_off(
    pieces: np.ndarray, rank: int, run: int, slots: torch.Tensor, has_state: torch.Tensor, device: torch.device,
) -> HandOff:
    """``rank``'s ``HandOff`` for prompt rows (from ``run``) split ``pieces``."""
    mine = pieces[rank].astype(np.int64)
    offsets = np.cumsum(mine) - mine
    rows = tuple(i for i in range(run, len(mine)) if mine[i])
    starts = frozenset(i for i in rows if not pieces[:rank, i].any())
    ends = frozenset(i for i in rows if not pieces[rank + 1:, i].any())

    def chunk(chosen: list[int]) -> Chunk:
        index = np.concatenate([np.arange(offsets[i], offsets[i] + mine[i]) for i in chosen])
        cu = np.concatenate([[0], np.cumsum([mine[i] for i in chosen])])
        return Chunk(tuple(chosen), to_device_nonblocking(index, device, torch.int64),
                     to_device_nonblocking(cu, device, torch.int64))

    fresh = [i for i in rows if i in starts]
    fresh_index = to_device_nonblocking(fresh, device, torch.int64) if fresh else None
    last = rows[-1] if rows else None
    # every prompt row's final, from the rank it ends on: in row order, as the ranks hold them
    ended = run + np.flatnonzero(pieces[:, run:].any(0))
    ends_on = pieces.shape[0] - 1 - np.argmax(pieces[::-1, ended] > 0, axis=0)
    return HandOff(
        rows=rows, starts=starts, ends=ends, offsets=offsets, spans=mine,
        fresh=chunk(fresh) if fresh else None,
        fresh_slots=None if fresh_index is None else slots.index_select(0, fresh_index),
        fresh_has_state=None if fresh_index is None else has_state.index_select(0, fresh_index),
        first=chunk([rows[0]]) if rows and rows[0] not in starts else None,
        middle=chunk([last, last]) if last is not None and last not in starts | ends else None,
        ended_slots=slots.index_select(0, to_device_nonblocking(ended, device, torch.int64)),
        counts=np.bincount(ends_on, minlength=pieces.shape[0]).tolist(),
    )


def gather_rows(layer: torch.Tensor, slots: torch.Tensor, has_state: torch.Tensor) -> torch.Tensor:
    """Each row's block of a pool ``layer`` by ``slots``, zeros for a row without state or on
    ``NO_SLOT`` (-1, the sink off)."""
    live = slots >= 0
    rows = layer.index_select(0, slots.long().clamp_min(0))
    keep = (has_state & live).view(-1, *([1] * (rows.dim() - 1)))
    return torch.where(keep, rows, torch.zeros((), dtype=rows.dtype, device=rows.device))


def scatter_rows(layer: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """``values``, a block a row, into a pool ``layer`` at ``slots``. A padding row on ``NO_SLOT``
    (-1, the sink off) writes a live row's slot with that row's own value, or with no live row
    what the slot holds: no host sync, no other slot touched."""
    slots = slots.long()
    live = slots >= 0
    ref = torch.argmax(live.to(torch.uint8)).view(1)
    ref_slot = slots.index_select(0, ref).clamp_min(0)
    addr = torch.where(live, slots, ref_slot)
    shape = (-1, *([1] * (values.dim() - 1)))
    padded = torch.where(live.any(), values.index_select(0, ref), layer.index_select(0, ref_slot).to(values.dtype))
    layer.index_copy_(0, addr, torch.where(live.view(shape), values, padded).to(layer.dtype))
