"""The linear-attention resource's spec-time factory, and what its managers share on a node whose
steps split over SP ranks.

``LinearAttnManager.build`` picks the manager for the configured variant.

On a split step (``sequence_shard``) each rank holds a contiguous range of the step's tokens. The
recurrent pool stays copied over SP: every rank holds every head its TP rank does, and after each
step every SP replica of a slot holds the same bytes. The conv runs on each rank's own tokens with
the left context the earlier ranks hold (``conv_halo``); the recurrence's prompt rows take the
manager's exchange (``ulysses`` or ``state_passing``), its leading one-token decode rows run copied.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import NamedTuple

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

logger = logging.getLogger(__name__)


class SplitRows(NamedTuple):
    """A split step's label on this rank: its ranges, each rank's piece of each row
    (``SeqRange.row_pieces``), the conv halo on the device, the local conv's layout, and the
    rows' slots and whether each carries state. ``run`` leading rows are one-token decode rows,
    which run copied."""

    range: SeqRange
    pieces: np.ndarray
    halo: ConvHalo
    # the rows with a piece here, as the conv kernel lays them out
    conv_cu: torch.Tensor
    conv_spans: torch.Tensor
    slots: torch.Tensor
    has_state: torch.Tensor
    run: int

    @property
    def decode_tokens(self) -> tuple[int, ...]:
        """Each rank's tokens of the decode run, which every rank gathers."""
        return tuple(self.pieces[:, : self.run].sum(1).tolist())


class LinearAttnManager(AttentionResource):
    # Abstract except for `build`, which dispatches on the variant.
    _pool_key: str
    # the sequence shard over its node (`LinearAttnSpec.derive`), set by `build`; and the shard
    # itself where it can split (`bind_dependencies`)
    _shard_key: str | None = None
    _shard: SequenceShard | None = None
    # how a split step's prompt rows reach the recurrence (`LinearAttnSpec.sp_exchange`), read only
    # where the node's shard splits (`bind_dependencies`)
    _split_exchange: LinearSpExchange = "ulysses"
    # label -> this rank's share of it, for each label this step splits (the managers' `plan`)
    _split_rows: dict[str, SplitRows]

    @classmethod
    def build(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        manager = cls._build_variant(spec, info)
        manager._shard_key = spec.shard_key
        manager._split_exchange = spec.sp_exchange or spec.sp_exchanges[0]
        return manager

    @classmethod
    def _build_variant(cls, spec: LinearAttnSpec, info: EngineResourceInfo):
        # the pool's own config, not a copy; `shard` is idempotent so both builders can call it.
        # Heads over TP alone, the pool's being copied over SP ranks (`head_shards`)
        pool_config = info.dependency(spec.config.recurrent_state).config
        pool_config.shard(info.head_shards)

        backend = spec.config.backend
        if backend is not LinearAttnBackend.FLASHINFER:
            raise ValueError(f"Unknown linear attention backend {backend!r}")

        variant = spec.config.variant
        if variant is LinearAttnVariant.MAMBA2:
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
        if variant is LinearAttnVariant.KDA:
            raise NotImplementedError(
                "KDA shares this pool's state layout, but its kernels take a "
                "per-K-channel gate the GDN marshalling does not build, and "
                "FlashInfer ships no chunked KDA prefill. Follow-up."
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
        if self._shard_key is None:
            return None
        plan = check_shard_plan(f"linear attention over pool {self._pool_key!r}", self._shard_key, ctx)
        r = plan.ranges.get(label) if self._shard is not None and plan.split else None
        if r is None:
            return None
        if addressing.snapshots:
            # the pool keeps snapshots at one rank only (`RecurrentStateSpec.keeps_prefix_cache`)
            raise ValueError(f"a split step of {label!r} keeps prefix snapshots, which split steps don't write")
        pieces = r.row_pieces(spans)
        taps = self.conv_taps
        host = conv_halo_plan(pieces, r.group.rank, taps)
        device = self._device
        halo = host.to(device)
        rows = len(spans)
        return SplitRows(
            range=r,
            pieces=pieces,
            halo=halo,
            conv_cu=to_device_nonblocking(np.concatenate([[0], np.cumsum(host.spans)]), device, torch.int32),
            conv_spans=torch.from_numpy(host.spans.astype(np.int32)),
            slots=addressing.slot_indices[:rows],
            has_state=addressing.has_state[:rows],
            run=run,
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
        rows = len(split.halo.rows)

        def conv(x: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
            # the rows with a piece here, each from its scratch window
            return causal_conv1d_fn(
                x=x.transpose(0, 1), weight=weight, bias=bias, conv_states=initial,
                query_start_loc=split.conv_cu,
                cache_indices=torch.arange(rows, dtype=torch.int32, device=x.device),
                has_initial_state=torch.ones(rows, dtype=torch.bool, device=x.device),
                activation=activation, pad_slot_id=-1, null_block_id=-1, seqlens_cpu=split.conv_spans,
            ).transpose(0, 1)

        stored = gather_rows(conv_layer, split.slots, split.has_state)
        out, final = conv_halo(conv, x, stored, split.halo, split.range.group)
        scatter_rows(conv_layer, split.slots, final)
        return out

    def _decode_slab(self, split: SplitRows, *xs: torch.Tensor) -> list[torch.Tensor]:
        """The step's decode run, every rank's tokens of it gathered: the first tokens of each of
        ``xs``, this rank's, and the whole run back, one collective sized by the step."""
        counts = split.decode_tokens
        mine = counts[split.range.group.rank]
        flat = torch.cat([x[:mine].flatten(1) for x in xs], dim=1)
        whole = gather_sequence(split.range.group, flat, counts)
        sizes = [x[0].numel() for x in xs]
        parts = whole.split(sizes, dim=1)
        return [part.unflatten(1, x.shape[1:]).contiguous() for part, x in zip(parts, xs, strict=True)]


def finals_by_rank(pieces: np.ndarray, first: int) -> tuple[np.ndarray, list[int]]:
    """The rows from ``first`` holding tokens, whose final states every replica gathers from the
    rank each ends on, and how many end on each rank: in row order, as the ranks hold them."""
    rows = first + np.flatnonzero(pieces[:, first:].any(0))
    ends_on = pieces.shape[0] - 1 - np.argmax(pieces[::-1, rows] > 0, axis=0)
    return rows, np.bincount(ends_on, minlength=pieces.shape[0]).tolist()


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
    what the slot holds, as ``GDNPrefillWrapper.run`` does: no host sync, no other slot touched."""
    slots = slots.long()
    live = slots >= 0
    ref = torch.argmax(live.to(torch.uint8)).view(1)
    ref_slot = slots.index_select(0, ref).clamp_min(0)
    addr = torch.where(live, slots, ref_slot)
    shape = (-1, *([1] * (values.dim() - 1)))
    padded = torch.where(live.any(), values.index_select(0, ref), layer.index_select(0, ref_slot).to(values.dtype))
    layer.index_copy_(0, addr, torch.where(live.view(shape), values, padded).to(layer.dtype))
