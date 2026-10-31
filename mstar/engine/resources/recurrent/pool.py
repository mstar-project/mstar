"""A pool of fixed-size recurrent state slots.

Storage only: the pool hands out slots, keeps the per-layer tensors they index,
and says which slot each row of a step addresses. As ``kv/`` is to ``attn/``,
a linear-attention resource is the wrapper planned against it.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.base import (
    CachedPrefix,
    CGSlotKey,
    CGSlotSpec,
    EngineResourceInfo,
    KeyedWalks,
    Resource,
)
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.manager import _DEBUG_ASSERTS
from mstar.engine.resources.recurrent.config import (
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    Segment,
    StepContext,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import KVManager

logger = logging.getLogger(__name__)

# Where a padding tail or zero-span row points. SINK_SLOT is held out of
# circulation like SINK_PAGE and absorbs writes nothing reads (the one place
# slot ids are not unique). NO_SLOT asks the backend to skip the row, which
# only some kernels honour.
SINK_SLOT = 0
NO_SLOT = -1


def checkpoint_rows(max_tokens: int, max_bs: int, every: int) -> int:
    """The checkpoints a captured step of up to ``max_tokens`` tokens over
    ``max_bs`` rows can write, one each ``every`` tokens of a row."""
    return (max_tokens + max_bs) // every + 1


@dataclass
class SlotState:
    index: int
    # whether a later step resumes from the slot; a fresh slot reads as zeros
    has_state: bool = False
    # Counts forks into this slot, like `CacheStream.generation`; nothing in
    # the engine reads it yet (the pre-plan tests use it to see a fork land).
    generation: int = 0
    # tokens the state has taken in, which places its page boundaries
    tokens: int = 0


class SnapshotLease(NamedTuple):
    """A probe's hold on the snapshot it restores, from the probe to its walk's admit.

    ``slot`` is None when no snapshot fits, and ``hit`` is how much the KV
    cache held before the pool capped it to the snapshot.
    """
    label: str
    slot: int | None
    tokens: int
    hit: int


class SnapshotWrite(NamedTuple):
    """A snapshot slot that admit reserved for a prefill row's state at
    ``offset`` tokens into the row. Commit files it under ``key``."""
    slot: int
    offset: int
    key: bytes


class SnapshotTarget(NamedTuple):
    row: int
    offset: int
    slot: int


class DecodeSnapshot(NamedTuple):
    slot: int
    tokens: int
    # None until the KV chain keys its page
    key: bytes | None = None
    # the filed snapshot this one replaces once the chain keys it
    replaces: DecodeSnapshot | None = None


@dataclass(frozen=True)
class RecurrentAddressing:
    """Where one label's rows live this step: the pool's whole plan output.

    Token layout (cu_seqlens, chunk metadata) is the backend's to build.
    """

    slot_indices: torch.Tensor  # [rows] int32, `pad_index` where nothing is addressed
    has_state: torch.Tensor     # [rows] bool, False for a slot with no saved state (fresh, or the sink)
    num_rows: int               # the rows the step declared, a replay's padding rows included
    # The same two on the host, as built: a backend that splits rows between
    # kernels by them must not read the device copies back
    host_slot_indices: Sequence[int] | None = None
    host_has_state: Sequence[bool] | None = None
    # The prefill states to save into snapshot slots, and the checkpoint
    # interval they sit on: a page or a multiple of one, or 0 while the pool
    # keeps no snapshots.
    snapshots: Sequence[SnapshotTarget] = ()
    snapshot_every: int = 0
    # scatter_'s int64 copy of slot_indices, keyed by whether a graph capture made it
    scatter_index: dict[bool, torch.Tensor] = field(default_factory=dict, compare=False, repr=False)


class RecurrentStatePool(Resource):
    @classmethod
    def build(
        cls, spec: RecurrentStateSpec, info: EngineResourceInfo,
    ) -> "RecurrentStatePool":
        config = spec.config
        world_size = 1
        if info.joint_comm_group is not None:
            world_size = info.joint_comm_group.world_size
            config.shard(world_size)
        kv = info.dependencies.get(config.kv_cache)
        if world_size > 1 or kv is None or not kv.config.prefix_cache:
            # no snapshot slots where the prefix cache can't open
            config = replace(config, prefix_slots=0)
        # resolved here, not written back: the config is the model's, shared by its specs
        return cls(
            device=info.device, config=config, name=spec.resource_key, kv_dtype=info.kv_dtype,
            world_size=world_size,
        )

    def __init__(
        self, device: torch.device, config: RecurrentStateConfig, name: str = "recurrent_state",
        kv_dtype: torch.dtype | None = None, world_size: int = 1,
    ):
        """``kv_dtype`` (the engine's) stands in for a block's ``dtype=None``."""
        if config.prefix_slots and config.disable_sink_slot:
            raise ValueError(
                "prefix_slots needs the sink slot: a captured step's padding "
                "rows write their snapshots there"
            )
        self._device = device
        self.config = config
        # the resource key, for errors
        self._name = name

        # One [num_layers, max_slots + prefix_slots, *shape] tensor per block. Layer-major so
        # `blocks[name][layer]` is contiguous: FlashInfer's pool paths assert
        # the slot-major view is K-contiguous (stride(-1) == 1).
        self._blocks: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (config.num_layers, config.max_slots + config.prefix_slots, *block.shape),
                dtype=block.resolved_dtype(kv_dtype),
                device=device,
            )
            for name, block in config.blocks.items()
        }

        # rid -> label -> slot. Mirrors `KVManager._streams`.
        self._slots: dict[str, dict[str, SlotState]] = {}
        # reversed so `pop` hands out ascending ids, as the page allocator does
        self._free: list[int] = list(reversed(range(config.max_slots)))
        if not config.disable_sink_slot:
            sink = self._free.pop()
            assert sink == SINK_SLOT, f"expected slot {SINK_SLOT} first, got {sink}"
        # guards `_slots`/`_free` across threads; see `KVManager._lock`
        self._lock = threading.RLock()
        # (rid, label) -> slot the live admit took fresh, and whether it took the
        # request's reservation, for `rollback_admit`. Per row like
        # `KVManager._admit_reserved_pages`, and handle-keyed the same way, so a
        # released request's entries are purged with it.
        self._admit_taken: dict[tuple[str, str], tuple[int, bool]] = {}
        # Admission, as `KVManager` decides it: only at one rank, since a
        # reservation held on rank 0 alone would make the ranks refuse different
        # allocations. `_reserved` holds, per admitted request, the slots kept
        # for labels it hasn't taken yet, and `_kept` their sum.
        self._decides = world_size == 1
        self._reserved: dict[str, int] = {}
        self._kept = 0

        self._cg_max_bs = 0
        self._cg_max_tokens = 0
        self._cg_addressing: dict[CGSlotKey, RecurrentAddressing] = {}
        # (slot_indices, has_state) mirrors per captured buffer: a steady batch
        # addresses the same slots every step, so its copies are skipped
        self._cg_mirrors: dict[CGSlotKey, tuple[H2DMirror, H2DMirror]] = {}
        # copy-only H2D for the addressing, see `_build_addressing`
        self._index_stager = PinnedStager(torch.int32)
        self._state_stager = PinnedStager(torch.bool)
        self._eager_addressing: dict[str, RecurrentAddressing] = {}
        # label -> this step's addressing, for the backend to read
        self._current: dict[str, RecurrentAddressing] = {}

        # Pre-planning stages the addressing (host bookkeeping plus two H2D
        # copies). Forks are not staged; see `_pending_fork_state`.
        self._preplanned = False
        self._cached_plan_output: dict[str, RecurrentAddressing] | None = None

        # Snapshots:
        # `_kv` is the KV cache whose page keys name them, None while they are off.
        # `_snap_meta` maps a filed slot to its key, least recently filed or restored first.
        # `_snap_refs` counts the leases and restores holding a slot, which eviction skips.
        self._kv: KVManager | None = None
        first = config.max_slots
        self._snap_free = list(reversed(range(first, first + config.prefix_slots)))
        self._snap_by_key: dict[bytes, int] = {}
        self._snap_meta: OrderedDict[int, bytes] = OrderedDict()
        self._snap_refs: dict[int, int] = {}
        # Handle-keyed, purged by `remove_request`.
        # `_keyed` holds the (rid, label) streams the KV cache keys.
        # `_leases` holds a probe's snapshot until admit, and `_restores` from
        # admit to plan.
        # `_snap_writes` holds the slots admit reserved for a prefill's states,
        # by (rid, label, boundary), until commit files them.
        # `_decode` holds each stream's newest decode snapshot, and `_unkeyed`
        # those whose page the KV chain doesn't key yet.
        self._keyed: set[tuple[str, str]] = set()
        self._leases: dict[str, SnapshotLease] = {}
        self._restores: dict[tuple[str, str], SnapshotLease] = {}
        self._snap_writes: dict[tuple[str, str, int], SnapshotWrite] = {}
        self._decode: dict[tuple[str, str], DecodeSnapshot] = {}
        self._unkeyed: set[tuple[str, str]] = set()
        # what admits took for prefix reuse, for `rollback_admit`
        self._admit_restores: list[tuple[str, str]] = []
        self._admit_snaps: list[tuple[str, str, int]] = []

        # from the blocks, not the config: a dtype=None block is sized as resolved
        total_bytes = sum(t.nbytes for t in self._blocks.values())
        logger.info(
            "recurrent state pool: %d usable slots (+%d sink, +%d snapshot) x "
            "%d layers, %.2f MiB (%.2f KiB/slot), blocks=%s",
            config.usable_slots, config.max_slots - config.usable_slots,
            config.prefix_slots, config.num_layers,
            total_bytes / 2**20, total_bytes / (config.max_slots + config.prefix_slots) / 2**10,
            {n: tuple(b.shape) for n, b in config.blocks.items()},
        )

    # Storage access

    def block(self, name: str, layer_idx: int) -> torch.Tensor:
        """One layer's contiguous slot-major view of a block: ``[max_slots + prefix_slots, *shape]``."""
        return self._blocks[name][layer_idx]

    def gather(
        self, name: str, layer_idx: int = 0, *, label: str = "main", start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block ``name`` of this step's rows under ``label``, ``[rows, *shape]``
        (a copy), in the order the step declared them.

        It addresses rows through the step's plan, so a captured and an eager
        forward read the same slots. A row with no saved state (a request's
        first step, or a padding row on the sink) reads ``start``, zeros when
        none is given. A row must declare span >= 1 to reach its own slot; a
        zero-span row addresses the sink, so it reads ``start``. A pool without
        a sink refuses (``_need_sink``)."""
        if self.config.disable_sink_slot:
            self._need_sink("gather")
        rows = self.addressing(label)
        out = self._blocks[name][layer_idx].index_select(0, rows.slot_indices[: rows.num_rows])
        fresh = ~rows.has_state[: rows.num_rows].view(-1, *(1,) * (out.dim() - 1))
        return torch.where(fresh, 0 if start is None else start, out)

    def scatter_(self, name: str, value: torch.Tensor, layer_idx: int = 0, *, label: str = "main") -> None:
        """Write ``value [rows, *shape]`` into block ``name`` of this step's rows
        under ``label``, in the order ``gather`` reads them; padding rows write
        the sink, which a pool without one refuses."""
        if self.config.disable_sink_slot:
            self._need_sink("scatter_")
        rows = self.addressing(label)
        if value.shape[0] != rows.num_rows:
            raise ValueError(
                f"recurrent state pool {self._name!r} label {label!r}: {value.shape[0]} rows "
                f"for a step that declared {rows.num_rows}"
            )
        # index_copy_ needs int64: cast once per step, not per block and layer. A capture
        # records its own cast, so a replay recomputes it from that replay's plan.
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        index = rows.scatter_index.get(capturing)
        if index is None:
            index = rows.scatter_index[capturing] = rows.slot_indices[: rows.num_rows].long()
        self._blocks[name][layer_idx].index_copy_(0, index, value)

    def _need_sink(self, op: str) -> None:
        # Without a sink, rows with no slot address NO_SLOT, which index_select
        # and index_copy_ refuse: an IndexError on the CPU, a device-side
        # assert that kills the worker on CUDA.
        raise ValueError(
            f"recurrent state pool {self._name!r} has no sink slot "
            f"(disable_sink_slot), so {op} has no slot for a padding row; "
            "unset RecurrentStateConfig.disable_sink_slot to use gather / scatter_"
        )

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    @property
    def pad_index(self) -> int:
        """What an unaddressed row points at."""
        return NO_SLOT if self.config.disable_sink_slot else SINK_SLOT

    # Request lifecycle

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None):
        del overrides
        with self._lock:
            # Idempotent: the conductor sends one NewRequest per partition, so
            # a worker serving two ingests twice.
            self._slots.setdefault(rid, {})

    def remove_request(self, rid: str):
        self.reset_request(rid, free=True)
        with self._lock:
            self._slots.pop(rid, None)
            self._kept -= self._reserved.pop(rid, 0)
            self._drop_lease(rid)
            for key in [k for k in self._restores if k[0] == rid]:
                self._unref(self._restores.pop(key).slot)
            # reserved by an admit whose step never committed (its forward raised)
            for key in [k for k in self._snap_writes if k[0] == rid]:
                self._snap_free.append(self._snap_writes.pop(key).slot)
            # a filed decode snapshot stays, for the next turn of the chat
            for key in [k for k in self._decode if k[0] == rid]:
                decode = self._decode.pop(key)
                if key in self._unkeyed:
                    self._unkeyed.discard(key)
                    self._snap_free.append(decode.slot)
            self._keyed -= {k for k in self._keyed if k[0] == rid}
            self._forget_prefix_admit(lambda key: key[0] == rid)

    def reset_request(self, rid: str, free: bool = False):
        """Drop a request's slots. ``free`` hands them back to the pool.

        Both paths zero what they release, or a backend could resume on another
        request's state.
        """
        with self._lock:
            self._forget_taken(lambda r: r == rid)
            labels = self._slots.get(rid)
            if not labels:
                return
            for slot in labels.values():
                self._zero_slot(slot.index)
                if free:
                    self._free.append(slot.index)
            if free:
                labels.clear()
            else:
                for slot in labels.values():
                    slot.has_state = False
                    slot.tokens = 0

    def _zero_slot(self, index: int) -> None:
        for tensor in self._blocks.values():
            tensor[:, index].zero_()

    def _alloc(self, rid: str, label: str) -> SlotState | None:
        with self._lock:
            labels = self._slots.setdefault(rid, {})
            slot = labels.get(label)
            if slot is not None:
                return slot
            kept = self._reserved.get(rid, 0)
            if label not in self.config.labels or len(self._free) - self._kept + bool(kept) <= 0:
                # an undeclared label, or every free slot is kept for an admitted request
                return None
            if kept:
                self._kept -= 1
                if kept == 1:
                    del self._reserved[rid]
                else:
                    self._reserved[rid] = kept - 1
            # Slot ids must be unique across the batch: kernels assume it rather
            # than check, and a scatter with two rows on one slot is
            # nondeterministic. The free list guarantees it.
            slot = labels[label] = SlotState(index=self._free.pop())
            self._admit_taken[(rid, label)] = (slot.index, bool(kept))
            return slot

    def _forget_taken(self, is_mine) -> None:
        """Drop the rollback record of the rows ``is_mine`` accepts. Caller
        holds the lock."""
        taken = self._admit_taken
        for key in [k for k in taken if is_mine(k[0])]:
            del taken[key]

    # Admission: a slot for each declared label, for a request that holds none

    def ask(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        if not self._decides:
            return None
        with self._lock:
            if self._slots.get(rid) or rid in self._reserved:
                return None
        return ASKS

    def reserve(self, rid: str, node_name: str, graph_walk: str, published: None) -> int | None:
        with self._lock:
            need, room = len(self.config.labels), len(self._free) - self._kept
            if need > room:
                return room
            self._reserved[rid] = need
            self._kept += need
            return None

    def unreserve(self, rid: str) -> None:
        with self._lock:
            self._kept -= self._reserved.pop(rid, 0)

    def grew_since(self, mark: int) -> bool:
        return len(self._free) - self._kept > mark

    # Step lifecycle

    def admit(self, step: RecurrentStep, ctx: StepContext) -> AdmitOutcome:
        """Reserve a slot per addressed (rid, label), plus fork targets.

        Capture rows and a replay's padding rows reserve nothing and address the
        sink instead; otherwise the runner's dummy names would drain a pool
        sized for the real concurrency.
        """
        if ctx.capture:
            return ADMIT_OK
        if self._admit_taken or self._admit_restores or self._admit_snaps:
            with self._lock:
                self._forget_taken(ctx.is_real_row)
                self._forget_prefix_admit(lambda key: ctx.is_real_row(key[0]))
        page = self._kv.config.tokens_per_page if self._kv is not None else 0
        every: dict[str, int] = {}
        for segment in step.segments or ():
            if segment.span <= 0 or ctx.is_padding_row(segment.request_id):
                # a zero-span row is addressed at the sink, like a padding row:
                # it neither reads nor writes its own slot
                continue
            if self._alloc(segment.request_id, segment.label) is None:
                return self._refused(segment.request_id, segment.label)
            if page and (segment.request_id in self._leases or segment.span >= page):
                label = segment.label
                if label not in every:
                    every[label] = self._checkpoint_every(
                        [seg for seg in step.segments if seg.label == label], ctx,
                    )
                self._admit_prefix(segment.request_id, label, segment.span, every[label], ctx.is_preplan)

        for rid in self._fork_rids(step):
            if ctx.is_padding_row(rid):
                continue
            for _, to_label in (*step.pre_forks, *step.post_forks):
                if self._alloc(rid, to_label) is None:
                    return self._refused(rid, to_label)
        return ADMIT_OK

    def rollback_admit(self, step: RecurrentStep, ctx: StepContext) -> None:
        """Free the slots this step's admit took fresh; slots a request already
        held stay with it. Without this, two forking requests can each keep
        the slot the other's fork needs, and the pool has no eviction to break
        that."""
        del step
        if ctx.capture or not (self._admit_taken or self._admit_restores or self._admit_snaps):
            return
        with self._lock:
            self._undo_prefix_admit(lambda key: ctx.is_real_row(key[0]))
            taken = self._admit_taken
            for rid, label in [k for k in taken if ctx.is_real_row(k[0])]:
                index, reserved = taken.pop((rid, label))
                labels = self._slots.get(rid)
                slot = labels.get(label) if labels else None
                if slot is None or slot.index != index:
                    continue  # reset or removed under us; the slot went with it
                del labels[label]
                self._zero_slot(index)
                self._free.append(index)
                if reserved:
                    self._reserved[rid] = self._reserved.get(rid, 0) + 1
                    self._kept += 1

    def _refused(self, rid: str, label: str) -> AdmitOutcome:
        """Why `_alloc` gave ``rid`` no slot for ``label``."""
        if label not in self.config.labels:
            return AdmitOutcome(ok=False, reason=AdmitRuntimeError(
                message=(
                    f"recurrent state pool {self._name!r}: {rid} opens label {label!r}, which isn't one of "
                    f"the pool's labels {self.config.labels}, so admission kept no slot for it"
                ),
                request_ids=frozenset((rid,)),
            ))
        return AdmitOutcome(
            ok=False,
            reason=AllocationFailed(
                message=(
                    f"recurrent state pool is full: {self.config.usable_slots} "
                    f"usable slots, none free for {rid}/{label}"
                ),
                # named for the KV cache's unit; one slot is what is short here
                pages_short=1,
                label=label,
                request_id=rid,
            ),
        )

    @staticmethod
    def _fork_rids(step: RecurrentStep) -> set[str]:
        if not (step.pre_forks or step.post_forks):
            return set()
        return {seg.request_id for seg in step.segments or ()}

    @property
    def supports_preplan(self):
        return True

    def _maybe_apply_forks(self, step: RecurrentStep, ctx: StepContext):
        for rid in self._fork_rids(step):
            if ctx.capture or ctx.is_padding_row(rid):
                continue
            for from_label, to_label in step.pre_forks:
                self._apply_fork(rid, from_label, to_label)

    def _pending_fork_state(self, step: RecurrentStep) -> dict[tuple[str, str], bool]:
        """``has_state`` each pre-fork will hand its target, without applying it.

        Staging cannot apply the fork (the copy would race the default stream),
        but a fork target inherits its source's ``has_state``, so the flag is
        read ahead and the copy happens at promotion.
        """
        pending: dict[tuple[str, str], bool] = {}
        for rid in self._fork_rids(step):
            for from_label, to_label in step.pre_forks:
                src = self._slots.get(rid, {}).get(from_label)
                if src is not None:
                    pending[(rid, to_label)] = src.has_state
        return pending

    def plan(self, step: RecurrentStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "recurrent preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # Promotion: apply the fork copies staging left undone; see
            # `_pending_fork_state`.
            self._current = self._cached_plan_output
            self._maybe_apply_forks(step, ctx)
            self._drop_stage()
            return self._current

        # Forks run ahead of the addressing, which records `has_state`; staging
        # only reads the flag the fork will land.
        if ctx.is_preplan:
            pending = self._pending_fork_state(step)
        else:
            pending = {}
            self._maybe_apply_forks(step, ctx)
            self._apply_restores(step)

        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_addressing(
                label, segments, ctx, pending,
            )
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        """Drop the staged plan; staging touched no live state, so nothing to rewind."""
        self._drop_stage()

    def _drop_stage(self) -> None:
        self._preplanned = False
        self._cached_plan_output = None

    def commit(self, step: RecurrentStep, ctx: StepContext) -> None:
        if ctx.capture:
            return
        with self._lock:
            # committed, so there is nothing left to unwind
            if self._admit_taken:
                self._forget_taken(ctx.is_real_row)
            if self._admit_restores or self._admit_snaps:
                self._forget_prefix_admit(lambda key: ctx.is_real_row(key[0]))
            page = self._kv.config.tokens_per_page if self._kv is not None else 0
            for segment in step.segments or ():
                if segment.span <= 0 or ctx.is_padding_row(segment.request_id):
                    continue
                slot = self._slots.get(segment.request_id, {}).get(segment.label)
                if slot is not None:
                    # the backend already wrote the pool; a later step resumes
                    slot.has_state = True
                    slot.tokens += segment.span
                    # a row too short to checkpoint that ends on a page (a decode step) is saved here, after its forward
                    if page and segment.span < page and not slot.tokens % page:
                        self._save_decode(segment.request_id, segment.label, slot)
            if self._snap_writes:
                self._file_writes(step)
            if self._unkeyed:
                self._file_decodes()

        for rid in self._fork_rids(step):
            if ctx.is_padding_row(rid):
                continue
            for from_label, to_label in step.post_forks:
                self._apply_fork(rid, from_label, to_label)

    def _apply_fork(
        self, rid: str, from_label: str, to_label: str,
    ) -> None:
        """Copy one slot onto its fork target.

        A real copy, not page aliasing: the state is mutated in place, so two
        labels cannot share it.
        """
        with self._lock:
            labels = self._slots.get(rid)
            if not labels or from_label not in labels:
                return
            src = labels[from_label]
            dst = labels.get(to_label)
            assert dst is not None, (
                f"fork target {rid}/{to_label} was never reserved; admit "
                "should have allocated it"
            )
            for tensor in self._blocks.values():
                tensor[:, dst.index].copy_(tensor[:, src.index])
            dst.has_state = src.has_state
            dst.tokens = src.tokens
            dst.generation += 1

    # Prefix reuse: a hit restores the state a snapshot holds at the agreed length

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        if self.config.kv_cache in resources:
            self.bind_kv(resources[self.config.kv_cache])

    def keyed_walks(self, label: str) -> frozenset[str]:
        """The walks the KV cache keys under ``label``, none while snapshots are off."""
        return self._kv.keyed_walks(label) if self._kv is not None else frozenset()

    def bind_kv(self, kv: "KVManager") -> None:
        """Key snapshots by the page chains of ``kv``, the cache `config.kv_cache` names."""
        self._kv = kv

    def enable_prefix_cache(self, root: bytes, walks: dict[str, KeyedWalks] | None = None) -> bool:
        del root, walks
        if not self.config.prefix_slots:
            self._kv = None
        return self._kv is not None

    def close_prefix_cache(self) -> None:
        self._kv = None

    def fingerprint(self) -> bytes:
        """The blocks' shapes and dtypes. The state's dtype changes what the
        recurrent layers output, and so every KV page written after them."""
        return fingerprint(sorted(
            (name, block.shape, str(block.dtype)) for name, block in self.config.blocks.items()
        ))

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
    ) -> CachedPrefix | None:
        """Note which streams the KV cache keys, and return None.
        `cap_cached_prefix` then fits the KV cache's answer to a snapshot."""
        del inputs
        kv = self._kv
        if kv is None:
            return None
        label = kv.keyed_label(rid, node_name, graph_walk)
        if label is not None:
            with self._lock:
                if rid in self._slots:
                    self._keyed.add((rid, label))
        return None

    def cap_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, prefix: CachedPrefix,
    ) -> CachedPrefix:
        """Cut ``prefix`` to the longest snapshot at or under it, and lease that
        snapshot until admit. A stream that already holds state restores nothing."""
        del node_name, graph_walk
        if self._kv is None:
            return prefix
        label, page = prefix.label, self._kv.config.tokens_per_page
        keys = self._kv.chain_keys(rid, label, prefix.tokens // page)
        with self._lock:
            if rid not in self._slots:
                return self._cut(prefix, 0)
            slot = self._slots[rid].get(label)
            if (slot is not None and slot.has_state) or (rid, label) in self._restores:
                return self._cut(prefix, 0)
            lease = self._leases.get(rid)
            if lease is not None and lease.tokens <= prefix.tokens:
                return self._cut(prefix, lease.tokens)
            self._drop_lease(rid)
            pages = next((n for n in range(len(keys), 0, -1) if keys[n - 1] in self._snap_by_key), 0)
            snap = self._snap_by_key[keys[pages - 1]] if pages else None
            if snap is not None:
                self._snap_refs[snap] = self._snap_refs.get(snap, 0) + 1
            self._leases[rid] = SnapshotLease(label, snap, pages * page, prefix.tokens)
        return self._cut(prefix, pages * page)

    @staticmethod
    def _cut(prefix: CachedPrefix, tokens: int) -> CachedPrefix:
        if tokens == prefix.tokens:
            return prefix
        return prefix._replace(tokens=tokens, position=prefix.position - prefix.tokens + tokens)

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
        prefix: CachedPrefix | None,
    ) -> None:
        del node_name, graph_walk, inputs
        with self._lock:
            lease = self._leases.get(rid)
            if lease is None:
                return
            if prefix is None:
                self._drop_lease(rid)
                return
            if prefix.tokens != lease.tokens:
                raise RuntimeError(
                    f"recurrent state: {rid} holds a snapshot of {lease.tokens} tokens, "
                    f"but the agreed prefix is {prefix.tokens}"
                )

    def _drop_lease(self, rid: str) -> None:
        lease = self._leases.pop(rid, None)
        if lease is not None and lease.slot is not None:
            self._unref(lease.slot)

    def _unref(self, snap: int) -> None:
        self._snap_refs[snap] -= 1
        if not self._snap_refs[snap]:
            del self._snap_refs[snap]

    def _checkpoint_every(self, segments: list[Segment], ctx: StepContext) -> int:
        """The checkpoint interval of a step over ``segments``: a page, or for a
        step past the largest captured bucket the smallest page multiple whose
        checkpoints fit that bucket's buffer, which an eager step writes too."""
        page = self._kv.config.tokens_per_page
        if not self._cg_max_tokens:
            return page
        spans = [seg.span for seg in segments if seg.span > 0 and not ctx.is_padding_row(seg.request_id)]
        rows = checkpoint_rows(self._cg_max_tokens, self._cg_max_bs, page)
        every = page
        while sum(span // every for span in spans) > rows:
            every += page
        return every

    def _admit_prefix(self, rid: str, label: str, span: int, every: int, preplan: bool) -> None:
        """Turn a hit's lease into a restore for plan to copy in, and reserve
        snapshot slots for the boundaries a prefill row keeps: the row's last
        boundary, and the KV hit's last boundary when the hit runs past the
        restored length, both on the step's checkpoint interval ``every``. A row
        that starts past token 0 and restores nothing keeps none. A pre-plan
        stages only an unprepared speculative step, which takes neither, and
        `clear_preplan` gives nothing back."""
        page = self._kv.config.tokens_per_page
        with self._lock:
            start = hit = 0
            lease = self._leases.get(rid)
            if lease is not None and lease.label == label:
                assert not preplan, "a pre-plan took a prefix lease"
                del self._leases[rid]
                hit = lease.hit
                if lease.slot is not None:
                    self._restores[(rid, label)] = lease
                    self._admit_restores.append((rid, label))
                    start = lease.tokens
            elif self._slots[rid][label].has_state:
                return
            if span < page or (rid, label) not in self._keyed:
                return
            for boundary in sorted({start + (span - 1) // every * every, start + (hit - start) // every * every}):
                if start < boundary <= start + span:
                    assert not preplan, "a pre-plan reserved a snapshot write"
                    self._reserve_write(rid, label, boundary, boundary - start)

    def _reserve_write(self, rid: str, label: str, boundary: int, offset: int) -> None:
        """Caller holds the lock."""
        key = self._kv.chain_key(rid, label, boundary // self._kv.config.tokens_per_page)
        if key is None or key in self._snap_by_key or (rid, label, boundary) in self._snap_writes:
            return
        snap = self._reserve_snapshot()
        if snap is not None:
            self._snap_writes[(rid, label, boundary)] = SnapshotWrite(snap, offset, key)
            self._admit_snaps.append((rid, label, boundary))

    def _file_writes(self, step: RecurrentStep) -> None:
        """Index the states this step's forward saved. Caller holds the lock."""
        rids = {seg.request_id for seg in step.segments or ()}
        for key in [k for k in self._snap_writes if k[0] in rids]:
            write = self._snap_writes.pop(key)
            if _DEBUG_ASSERTS:
                chained = self._kv.chain_key(key[0], key[1], key[2] // self._kv.config.tokens_per_page)
                assert write.key == chained, (
                    f"recurrent state: the snapshot of {key[0]}/{key[1]} at {key[2]} tokens is keyed "
                    "apart from the KV chain's key there"
                )
            self._index(write.slot, write.key)

    def _apply_restores(self, step: RecurrentStep) -> None:
        """Copy each restore's snapshot into its slot before the step reads it,
        and mark the snapshot most recently used. The copy is queued ahead of
        any write that could reuse the snapshot slot, so the reference can be
        dropped here."""
        if not self._restores:
            return
        rids = {seg.request_id for seg in step.segments or ()}
        with self._lock:
            for key in [k for k in self._restores if k[0] in rids]:
                restore = self._restores.pop(key)
                slot = self._slots.get(key[0], {}).get(key[1])
                if slot is not None:
                    for tensor in self._blocks.values():
                        tensor[:, slot.index].copy_(tensor[:, restore.slot])
                    slot.has_state = True
                    slot.tokens = restore.tokens
                    self._snap_meta.move_to_end(restore.slot)
                self._unref(restore.slot)

    def _reserve_snapshot(self) -> int | None:
        """A free snapshot slot, else the least recently used one nothing
        holds. Caller holds the lock."""
        if self._snap_free:
            return self._snap_free.pop()
        snap = next((snap for snap in self._snap_meta if snap not in self._snap_refs), None)
        if snap is not None:
            del self._snap_by_key[self._snap_meta.pop(snap)]
        return snap

    def _index(self, snap: int, key: bytes) -> bool:
        """File ``snap`` under ``key`` and return True, or free it and return
        False if an earlier copy stands. Caller holds the lock."""
        if key in self._snap_by_key:
            self._snap_free.append(snap)
            return False
        self._snap_by_key[key] = snap
        self._snap_meta[snap] = key
        return True

    def _save_decode(self, rid: str, label: str, slot: SlotState) -> None:
        """Copy ``slot`` into a snapshot that replaces the stream's previous
        decode snapshot. Caller holds the lock."""
        if (rid, label) not in self._keyed:
            return
        previous = self._decode.pop((rid, label), None)
        if previous is not None and (rid, label) in self._unkeyed:
            self._drop_decode((rid, label), previous)
            previous = previous.replaces
        snap = self._reserve_snapshot()
        if snap is None:
            if previous is not None:
                self._decode[(rid, label)] = previous
            return
        for tensor in self._blocks.values():
            tensor[:, snap].copy_(tensor[:, slot.index])
        # a filed `previous` stays until this one is filed: a step whose token never
        # reaches the chain (the one speculated past the reply's end) never is
        self._decode[(rid, label)] = DecodeSnapshot(snap, slot.tokens, replaces=previous)
        self._unkeyed.add((rid, label))

    def _drop_decode(self, key: tuple[str, str], decode: DecodeSnapshot) -> None:
        """Put a decode snapshot's slot back on the free list. One that eviction
        already took is left alone, and one a lease or restore holds is left for
        eviction. Caller holds the lock."""
        if key in self._unkeyed:
            self._unkeyed.discard(key)
        elif self._snap_by_key.get(decode.key) != decode.slot or decode.slot in self._snap_refs:
            return
        else:
            del self._snap_by_key[decode.key]
            del self._snap_meta[decode.slot]
        self._snap_free.append(decode.slot)

    def _file_decodes(self) -> None:
        """File each decode snapshot whose page the KV chain keys by now: its
        ids reach the chain after the step that wrote them. Caller holds the lock."""
        page = self._kv.config.tokens_per_page
        for key in list(self._unkeyed):
            decode = self._decode[key]
            name = self._kv.chain_key(*key, decode.tokens // page)
            if name is None:
                continue
            self._unkeyed.discard(key)
            if decode.replaces is not None:
                self._drop_decode(key, decode.replaces)
            if self._index(decode.slot, name):
                self._decode[key] = DecodeSnapshot(decode.slot, decode.tokens, name)
            else:
                del self._decode[key]

    def _undo_prefix_admit(self, is_mine) -> None:
        """For each record ``is_mine`` accepts, turn its restore back into a
        lease and free its reserved snapshot slot. Caller holds the lock."""
        for rid, label in [k for k in self._admit_restores if is_mine(k)]:
            restore = self._restores.pop((rid, label), None)
            if restore is not None:
                self._leases[rid] = restore
        for key in [k for k in self._admit_snaps if is_mine(k)]:
            write = self._snap_writes.pop(key, None)
            if write is not None:
                self._snap_free.append(write.slot)
        self._forget_prefix_admit(is_mine)

    def _forget_prefix_admit(self, is_mine) -> None:
        """Drop the rollback record of the restores and snapshot slots
        ``is_mine`` accepts, as `_forget_taken` does slots. Caller holds the lock."""
        gone = {k for k in (*self._admit_restores, *self._admit_snaps) if is_mine(k)}
        if gone:
            self._admit_restores = [k for k in self._admit_restores if k not in gone]
            self._admit_snaps = [k for k in self._admit_snaps if k not in gone]

    # Addressing

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _build_addressing(
        self, label: str, segments: list[Segment], ctx: StepContext,
        pending_forks: dict[tuple[str, str], bool] | None = None,
    ) -> RecurrentAddressing:
        pad = self.pad_index
        pending_forks = pending_forks or {}
        indices = []
        has_state = []
        for seg in segments:
            slot = (
                None if ctx.capture or ctx.is_padding_row(seg.request_id)
                else self._slots.get(seg.request_id, {}).get(label)
            )
            if slot is None or seg.span <= 0:
                # no slot, a zero-span row, a capture row or a padding row: the
                # sink (or the sentinel), which nothing reads back
                indices.append(pad)
                has_state.append(False)
            else:
                indices.append(slot.index)
                has_state.append(
                    pending_forks.get((seg.request_id, label), slot.has_state)
                )

        snapshots = ()
        if self._snap_writes and not ctx.capture:
            with self._lock:
                by_rid: dict[str, list[SnapshotWrite]] = {}
                for (rid, written, _), write in self._snap_writes.items():
                    if written == label:
                        by_rid.setdefault(rid, []).append(write)
            snapshots = tuple(
                SnapshotTarget(row, write.offset, write.slot)
                for row, seg in enumerate(segments) for write in by_rid.get(seg.request_id, ())
            )

        num_rows = len(indices)
        target, mirrors = self._addressing_buffers(label, ctx, num_rows)
        # Built on the host and copied in. The tail is cleared too: a recurrent
        # kernel reads every row, so a stale index would write a slot whose
        # request is not in this step. The clear rides the same H2D copy since
        # this runs in the pre-plan beside a live graph: no device-side fill.
        self._index_stager.copy_(
            target.slot_indices, indices, pad_value=pad, mirror=mirrors[0],
        )
        self._state_stager.copy_(
            target.has_state, has_state, pad_value=False, mirror=mirrors[1],
        )
        return RecurrentAddressing(
            slot_indices=target.slot_indices,
            has_state=target.has_state,
            num_rows=num_rows,
            # the lists as built, not copies: this call made them and nothing
            # changes them after
            host_slot_indices=indices,
            host_has_state=has_state,
            snapshots=snapshots,
            snapshot_every=(
                self._checkpoint_every(segments, ctx) if snapshots
                else self._kv.config.tokens_per_page if self._kv is not None else 0
            ),
        )

    def _addressing_buffers(
        self, label: str, ctx: StepContext, num_rows: int,
    ) -> tuple[RecurrentAddressing, tuple[H2DMirror | None, H2DMirror | None]]:
        """The buffers this step fills, and their mirrors (captured buffers only).

        Under capture they are static per (bucket, slot, label) and sized to the
        largest batch any runner asked for, not the first plan's row count.
        """
        lease = ctx.slot_lease
        if lease is None:
            buf = self._eager_addressing.get(label)
            if buf is None or buf.slot_indices.numel() < num_rows:
                buf = self._eager_addressing[label] = self._new_buffers(num_rows)
            return buf, (None, None)

        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        buf = self._cg_addressing.get(key)
        if buf is None:
            size = max(self._cg_max_bs, lease.bucket.bs, num_rows)
            buf = self._cg_addressing[key] = self._new_buffers(size)
            self._cg_mirrors[key] = (H2DMirror(), H2DMirror())
        assert buf.slot_indices.numel() >= num_rows, (
            f"recurrent addressing for {key} holds "
            f"{buf.slot_indices.numel()} rows but this step planned {num_rows}"
        )
        return buf, self._cg_mirrors[key]

    def _new_buffers(self, size: int) -> RecurrentAddressing:
        return RecurrentAddressing(
            slot_indices=torch.full(
                (size,), self.pad_index, dtype=torch.int32, device=self._device
            ),
            has_state=torch.zeros(size, dtype=torch.bool, device=self._device),
            num_rows=0,
        )

    def addressing(self, label: str = "main") -> RecurrentAddressing:
        """This step's addressing under ``label``, for a backend to run against."""
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"recurrent state has no plan for label {label!r}; this step "
                f"planned {sorted(self._current)}. Every label a forward runs "
                "must carry a segment in the step declaration."
            )
        return found

    # Engine lifecycle

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots
        # per-key buffers are built on first plan; every runner calls in, so
        # keep the largest
        self._cg_max_bs = max(self._cg_max_bs, max_bs)
        self._cg_max_tokens = max(self._cg_max_tokens, max_seq_len)

    def cleanup(self):
        self._blocks.clear()
        self._cg_addressing.clear()
        self._cg_mirrors.clear()
        self._eager_addressing.clear()
        self._current.clear()
