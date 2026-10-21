"""A pool of fixed-size recurrent state slots.

Storage only: the pool hands out slots, keeps the per-layer tensors they index,
and says which slot each row of a step addresses. As ``kv/`` is to ``attn/``,
a linear-attention resource is the wrapper planned against it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.recurrent.config import (
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    Segment,
    StepContext,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

logger = logging.getLogger(__name__)

# Where a padding tail or zero-span row points. SINK_SLOT is held out of
# circulation like SINK_PAGE and absorbs writes nothing reads (the one place
# slot ids are not unique). NO_SLOT asks the backend to skip the row, which
# only some kernels honour.
SINK_SLOT = 0
NO_SLOT = -1


@dataclass
class SlotState:
    index: int
    # whether a later step resumes from the slot; a fresh slot reads as zeros
    has_state: bool = False
    # Counts forks into this slot, like `CacheStream.generation`; nothing in
    # the engine reads it yet (the pre-plan tests use it to see a fork land).
    generation: int = 0


@dataclass(frozen=True)
class RecurrentAddressing:
    """Where one label's rows live this step: the pool's whole plan output.

    Token layout (cu_seqlens, chunk metadata) is the backend's to build.
    """

    slot_indices: torch.Tensor  # [rows] int32, `pad_index` where nothing is addressed
    has_state: torch.Tensor     # [rows] bool, False where the slot reads as zeros
    num_rows: int               # real rows, before a capture bucket's padding
    # The same two on the host, as built: a backend that splits rows between
    # kernels by them must not read the device copies back
    host_slot_indices: Sequence[int] | None = None
    host_has_state: Sequence[bool] | None = None


class RecurrentStatePool(Resource):
    @classmethod
    def build(
        cls, spec: RecurrentStateSpec, info: EngineResourceInfo,
    ) -> "RecurrentStatePool":
        config = spec.config
        if info.joint_comm_group is not None:
            config.shard(info.joint_comm_group.world_size)
        return cls(
            device=info.device, config=config, comm_group=info.joint_comm_group,
        )

    def __init__(
        self, device: torch.device, config: RecurrentStateConfig,
        comm_group: JointGroups | None = None,
    ):
        self._device = device
        self.config = config
        self._comm_group = comm_group

        # One [num_layers, max_slots, *shape] tensor per block. Layer-major so
        # `blocks[name][layer]` is contiguous: FlashInfer's pool paths assert
        # the slot-major view is K-contiguous (stride(-1) == 1). A block that
        # is not per layer is [max_slots, *shape].
        self._blocks: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (*((config.num_layers,) if block.per_layer else ()),
                 config.max_slots + config.prefix_slots, *block.shape),
                dtype=block.dtype,
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
        # (rid, label) -> slot the live admit took fresh, for `rollback_admit`.
        # Per row like `KVManager._admit_reserved_pages`, and handle-keyed the
        # same way, so a released request's entries are purged with it.
        self._admit_taken: dict[tuple[str, str], int] = {}
        # the part of it the pre-plan pass took: all `clear_preplan` gives back
        self._preplan_taken: set[tuple[str, str]] = set()

        self._cg_max_bs = 0
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

        # Prefix snapshots, in the `prefix_slots` past the request slots: the state a prompt
        # reached at a page boundary, under the KV chain's key there. Reserved at admit,
        # written by the forward, indexed at commit, and stamped by the commits that read
        # them: each TP rank does all of it in step order, so they evict alike.
        first = config.max_slots
        self._snap_free = list(reversed(range(first, first + config.prefix_slots)))
        self._snap_by_key: dict[bytes, int] = {}
        # snapshot slot -> (key, tokens, stamp)
        self._snap_meta: dict[int, tuple[bytes, int, int]] = {}
        # snapshot slot -> leases and restores holding it
        self._snap_refs: dict[int, int] = {}
        self._snap_clock = 0
        self._prefix_root: bytes | None = None
        # label -> the walks its keys describe
        self._keyed: dict[str, frozenset[str]] = {}
        # rid -> label -> the prompt's page keys, unrooted
        self._prefix_keys: dict[str, dict[str, list[bytes]]] = {}
        # rid -> label -> the first of them rooted, hashed once for a TP follower's adopt polls
        self._prefix_rooted: dict[str, dict[str, list[bytes]]] = {}
        # under TP rank 0 decides each hit (`adopt_cached_prefix`)
        self._follower = (
            comm_group is not None and comm_group.world_size > 1
            and getattr(comm_group, "rank", 0) != 0
        )
        # rid -> (label, snapshot slot, tokens): a hit held from the probe until admit
        self._leases: dict[str, tuple[str, int, int]] = {}
        # (rid, label) -> (snapshot slot, tokens) the request's first step copies in at plan
        self._restores: dict[tuple[str, str], tuple[int, int]] = {}
        # (rid, label) -> snapshot slot restored, stamped by the step's commit
        self._restored: dict[tuple[str, str], int] = {}
        # (rid, label, tokens) -> snapshot slot reserved for the forward to write
        self._snap_writes: dict[tuple[str, str, int], int] = {}
        self._snap_written: set[int] = set()
        # what admits took for prefix reuse, for `rollback_admit` / `clear_preplan`: per row
        # like `_admit_taken`, restores by (rid, label) and snapshot slots by (rid, label,
        # tokens), and the part of it the pre-plan pass took
        self._admit_restores: list[tuple[str, str]] = []
        self._admit_snaps: list[tuple[str, str, int]] = []
        self._preplan_prefix: set[tuple] = set()

        logger.info(
            "recurrent state pool: %d usable slots (+%d sink) x %d layers, "
            "%.2f MiB (%.2f KiB/slot), blocks=%s",
            config.usable_slots, config.max_slots - config.usable_slots,
            config.num_layers,
            config.total_bytes / 2**20, config.slot_bytes / 2**10,
            {n: tuple(b.shape) for n, b in config.blocks.items()},
        )

    # Storage access

    def block(self, name: str, layer_idx: int | None = None) -> torch.Tensor:
        """One layer's contiguous slot-major view of a block: ``[max_slots, *shape]``.
        A block that is not per layer takes no ``layer_idx``."""
        per_layer = self.config.blocks[name].per_layer
        if per_layer == (layer_idx is None):
            raise ValueError(
                f"recurrent block {name!r} is "
                f"{'per layer' if per_layer else 'per slot'}; got layer_idx={layer_idx}"
            )
        tensor = self._blocks[name]
        return tensor[layer_idx] if per_layer else tensor

    def _slot_views(self, index: int):
        """Every block's rows of slot ``index``, all layers."""
        for name, tensor in self._blocks.items():
            yield tensor[:, index] if self.config.blocks[name].per_layer else tensor[index]

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    @property
    def pad_index(self) -> int:
        """What an unaddressed row points at."""
        return NO_SLOT if self.config.disable_sink_slot else SINK_SLOT

    # Request lifecycle

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None):
        with self._lock:
            # Idempotent: the conductor sends one NewRequest per partition, so
            # a worker serving two ingests twice.
            self._slots.setdefault(rid, {})
            # a declared stream's config is the conductor's KVReqConfig
            keys = getattr(overrides, "prefix_keys", None)
            if keys and getattr(overrides, "prefix_cache", True):
                self._prefix_keys.setdefault(rid, keys)

    def remove_request(self, rid: str):
        self.reset_request(rid, free=True)
        with self._lock:
            self._slots.pop(rid, None)
            self._prefix_keys.pop(rid, None)
            self._prefix_rooted.pop(rid, None)
            self._drop_lease(rid)
            for key in [k for k in self._restores if k[0] == rid]:
                self._unref(self._restores.pop(key)[0])
            for key in [k for k in self._restored if k[0] == rid]:
                del self._restored[key]
            # reserved by an admit whose step never committed (its forward raised)
            for key in [k for k in self._snap_writes if k[0] == rid]:
                snap = self._snap_writes.pop(key)
                self._snap_written.discard(snap)
                self._snap_free.append(snap)
            # handle-keyed: what is left of its rollback record would name the next request
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

    def _zero_slot(self, index: int) -> None:
        for rows in self._slot_views(index):
            rows.zero_()

    def _alloc(self, rid: str, label: str) -> SlotState | None:
        with self._lock:
            labels = self._slots.setdefault(rid, {})
            slot = labels.get(label)
            if slot is not None:
                return slot
            if not self._free:
                return None
            # Slot ids must be unique across the batch: kernels assume it rather
            # than check, and a scatter with two rows on one slot is
            # nondeterministic. The free list guarantees it.
            slot = labels[label] = SlotState(index=self._free.pop())
            self._admit_taken[(rid, label)] = slot.index
            return slot

    def _forget_taken(self, is_mine) -> None:
        """Drop the rollback record of the rows ``is_mine`` accepts. Caller
        holds the lock."""
        taken = self._admit_taken
        for key in [k for k in taken if is_mine(k[0])]:
            del taken[key]
            self._preplan_taken.discard(key)

    # Step lifecycle

    def admit(self, step: RecurrentStep, ctx: StepContext) -> AdmitOutcome:
        """Reserve a slot per addressed (rid, label), plus fork targets.

        Capture rows and a replay's padding rows reserve nothing and address the
        sink instead; otherwise the runner's dummy names would drain a pool
        sized for the real concurrency.
        """
        if ctx.capture:
            return ADMIT_OK
        # A pre-planned step was reserved on its pre-plan pass, and a refusal
        # after this no-op must still find that reservation.
        if (self._admit_taken or self._admit_restores or self._admit_snaps) and not (
                self._preplanned and not ctx.is_preplan):
            with self._lock:
                self._forget_taken(ctx.is_real_row)
                self._forget_prefix_admit(lambda key: ctx.is_real_row(key[0]))
        for segment in step.segments or ():
            if segment.span <= 0 or ctx.is_padding_row(segment.request_id):
                # reads its state without extending it, or a padding row
                continue
            if self._alloc(segment.request_id, segment.label) is None:
                return self._out_of_slots(segment.request_id, segment.label)
            with self._lock:
                lease = self._leases.get(segment.request_id)
                if lease is not None and lease[0] == segment.label:
                    # a hit's first step: plan copies the snapshot into its slot
                    del self._leases[segment.request_id]
                    key = (segment.request_id, segment.label)
                    self._restores[key] = lease[1:]
                    self._admit_restores.append(key)
        with self._lock:
            for rid, label, tokens in step.snapshots:
                if (ctx.is_padding_row(rid) or self._prefix_root is None
                        or (rid, label, tokens) in self._snap_writes):  # a step admitted again
                    continue
                snap = self._reserve_snapshot()
                if snap is not None:
                    self._snap_writes[(rid, label, tokens)] = snap
                    self._admit_snaps.append((rid, label, tokens))

        for rid in self._fork_rids(step):
            if ctx.is_padding_row(rid):
                continue
            for _, to_label in (*step.pre_forks, *step.post_forks):
                if self._alloc(rid, to_label) is None:
                    return self._out_of_slots(rid, to_label)
        if ctx.is_preplan:
            with self._lock:
                self._preplan_taken = {k for k in self._admit_taken if ctx.is_real_row(k[0])}
                self._preplan_prefix = {
                    k for k in (*self._admit_restores, *self._admit_snaps)
                    if ctx.is_real_row(k[0])
                }
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
            self._release([k for k in self._admit_taken if ctx.is_real_row(k[0])])
            self._undo_prefix_admit(lambda key: ctx.is_real_row(key[0]))

    def _release(self, keys: list[tuple[str, str]]) -> None:
        """Free the slots ``keys`` took, zeroed like every free slot. Newest first, so
        the free list is back in its old order and every rank of a TP group hands
        out the same slots next. Caller holds the lock."""
        for rid, label in reversed(keys):
            index = self._admit_taken.pop((rid, label))
            self._preplan_taken.discard((rid, label))
            labels = self._slots.get(rid)
            slot = labels.get(label) if labels else None
            if slot is None or slot.index != index:
                continue  # reset or removed under us; the slot went with it
            del labels[label]
            self._zero_slot(index)
            self._free.append(index)

    def _out_of_slots(self, rid: str, label: str) -> AdmitOutcome:
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
            self._apply_restores(step)
            self._drop_stage()
            # the live step's now: commit or roll back
            self._preplan_taken, self._preplan_prefix = set(), set()
            return self._current

        # Forks run ahead of the addressing, which records `has_state`; staging
        # only reads the flag the fork will land.
        if ctx.is_preplan:
            pending = self._pending_fork_state(step)
            # a restore lands at promotion too, and its slot then holds state
            pending.update({key: True for key in self._restores
                            if key[0] in {seg.request_id for seg in step.segments or ()}})
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
        """Drop the staged plan and give back the slots its pre-plan pass took.

        Staging builds addressing only (forks apply at promotion); a live step's
        reservation is the runner's to commit or roll back.
        """
        self._drop_stage()
        with self._lock:
            # in the order the pre-plan pass took them, so `_release` goes newest first
            self._release([k for k in self._admit_taken if k in self._preplan_taken])
            self._undo_prefix_admit(self._preplan_prefix.__contains__)

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
            for segment in step.segments or ():
                if segment.span <= 0 or ctx.is_padding_row(segment.request_id):
                    continue
                slot = self._slots.get(segment.request_id, {}).get(segment.label)
                if slot is not None:
                    # the backend already wrote the pool; a later step resumes
                    slot.has_state = True

        for rid in self._fork_rids(step):
            if ctx.is_padding_row(rid):
                continue
            for from_label, to_label in step.post_forks:
                self._apply_fork(rid, from_label, to_label)
        self._commit_snapshots(step)

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
            for to, from_ in zip(
                self._slot_views(dst.index), self._slot_views(src.index), strict=True,
            ):
                to.copy_(from_)
            dst.has_state = src.has_state
            dst.generation += 1

    # Prefix reuse: a hit restores the state a snapshot holds at the agreed length

    def enable_prefix_cache(
        self, root: bytes, walks: dict[str, tuple[str, str | None]] | None = None,
    ) -> bool:
        self._keyed = {
            label: frozenset(walk for walk in named if walk is not None)
            for label, named in (walks or {}).items()
        }
        if not (self.config.prefix_slots and self.config.prefix_page_size):
            # keyed but keeping nothing: every probe answers 0, so no hit skips its state
            return False
        self._prefix_root = root
        return True

    def close_prefix_cache(self) -> None:
        self._prefix_root = None

    @property
    def keeps_snapshots(self) -> bool:
        """Whether a prefill should save its state for prefix reuse."""
        return self._prefix_root is not None

    def fingerprint(self) -> bytes:
        """What a snapshot's bytes depend on beyond its tokens: the blocks, as sharded."""
        world = self._comm_group.world_size if self._comm_group is not None else 1
        blocks = sorted((n, b.shape, str(b.dtype)) for n, b in self.config.blocks.items())
        return fingerprint(blocks, world, self.config.prefix_page_size)

    def _keyed_label(self, rid: str, graph_walk: str) -> str | None:
        keys = self._prefix_keys.get(rid) or {}
        for label, walks in self._keyed.items():
            if graph_walk in walks and keys.get(label):
                return label
        return None

    def _rooted(self, keys: list[bytes]) -> list[bytes]:
        return [fingerprint(self._prefix_root, key) for key in keys]

    def _find_snapshot(self, rooted: list[bytes], limit: int) -> tuple[int, int] | None:
        """The longest snapshot along this chain of at most ``limit`` tokens: (slot, tokens)."""
        size = self.config.prefix_page_size
        for pages in range(min(len(rooted), limit // size), 0, -1):
            snap = self._snap_by_key.get(rooted[pages - 1])
            if snap is not None and self._snap_meta[snap][1] == pages * size:
                return snap, pages * size
        return None

    def _lease(self, rid: str, label: str, snap: int, tokens: int) -> None:
        self._snap_refs[snap] = self._snap_refs.get(snap, 0) + 1
        self._leases[rid] = (label, snap, tokens)

    def _unref(self, snap: int) -> None:
        self._snap_refs[snap] -= 1
        if not self._snap_refs[snap]:
            del self._snap_refs[snap]

    def _drop_lease(self, rid: str) -> None:
        lease = self._leases.pop(rid, None)
        if lease is not None:
            self._unref(lease[1])

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
    ) -> int | None:
        """The longest prefix a snapshot holds for this request, held until admit."""
        del node_name
        with self._lock:
            label = self._keyed_label(rid, graph_walk)
            if label is None:
                return None
            if rid in self._leases:
                return self._leases[rid][2]
            slot = self._slots.get(rid, {}).get(label)
            if (slot is not None and slot.has_state) or (rid, label) in self._restores:
                return None
            if self._prefix_root is None or self._follower:
                return 0
            keys = list(self._prefix_keys[rid][label])
        rooted = self._rooted(keys)
        with self._lock:
            if rid not in self._slots:
                return None
            # one key short, as the KV cache: a fully cached prompt still runs a token
            found = self._find_snapshot(rooted, (len(rooted) - 1) * self.config.prefix_page_size)
            if found is None:
                return 0
            self._lease(rid, label, *found)
            return found[1]

    def cap_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, length: int,
    ) -> int:
        """Snapshots sit at some page boundaries only: the longest at or under ``length``."""
        del node_name, graph_walk
        with self._lock:
            lease = self._leases.get(rid)
            if lease is None:
                return length
            if lease[2] <= length:
                return lease[2]
            label = lease[0]
            keys = list(self._prefix_keys[rid][label])
        rooted = self._rooted(keys)
        with self._lock:
            self._drop_lease(rid)
            found = self._find_snapshot(rooted, length)
            if found is None:
                return 0
            self._lease(rid, label, *found)
            return found[1]

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs, matched_len: int,
    ) -> None:
        del node_name, graph_walk, inputs
        with self._lock:
            lease = self._leases.get(rid)
            if lease is None:
                return
            if matched_len <= 0:
                self._drop_lease(rid)
            elif lease[2] != matched_len:
                raise RuntimeError(
                    f"recurrent state: {rid} holds a snapshot of {lease[2]} tokens, but the "
                    f"agreed prefix is {matched_len}")

    def cached_prefix_pages(
        self, rid: str, node_name: str, graph_walk: str, matched_len: int,
    ) -> list[int] | None:
        del node_name, graph_walk, matched_len
        with self._lock:
            lease = self._leases.get(rid)
            return None if lease is None else [lease[1]]

    def adopt_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, matched_len: int,
        pages: list[int] | None = None,
    ) -> bool:
        """On a TP follower, take rank 0's snapshot of ``matched_len`` tokens: False while
        this rank's commit of the step that wrote it trails rank 0's."""
        del node_name
        if pages is None:  # rank 0 restored no snapshot for this request
            return True
        with self._lock:
            label = self._keyed_label(rid, graph_walk)
            if label is None or rid in self._leases:
                return True
            if self._prefix_root is None:
                raise RuntimeError(
                    f"recurrent state: rank 0 restored {matched_len} tokens for {rid}, but "
                    "this rank keeps no snapshots")
            need = matched_len // self.config.prefix_page_size
            rooted = self._prefix_rooted.get(rid, {}).get(label, [])
            # polled until this rank's commit catches up: hash the keys once
            keys = None if len(rooted) >= need else list(self._prefix_keys[rid][label][:need])
        if keys is not None:
            rooted = self._rooted(keys)
        with self._lock:
            if rid in self._prefix_keys:
                self._prefix_rooted.setdefault(rid, {})[label] = rooted
            rooted = rooted[:need]
            snap = self._snap_by_key.get(rooted[-1]) if rooted else None
            if snap is None or self._snap_meta[snap][1] != matched_len:
                return False
            if pages and pages[0] != snap:
                raise RuntimeError(
                    f"recurrent state: snapshots diverged from rank 0 for {rid}: slot {snap} "
                    f"here, {pages[0]} there")
            self._lease(rid, label, snap, matched_len)
            return True

    def write_snapshot(self, rid: str, label: str, tokens: int) -> None:
        """Called by the forward once ``rid``'s state stands at ``tokens``: copy it into the
        snapshot slot admit reserved, if one was."""
        snap = self._snap_writes.get((rid, label, tokens))
        if snap is None:
            return
        slot = self._slots[rid][label]
        for to, from_ in zip(self._slot_views(snap), self._slot_views(slot.index), strict=True):
            to.copy_(from_)
        self._snap_written.add(snap)

    def _reserve_snapshot(self) -> int | None:
        """A free snapshot slot, else the oldest nothing holds. Caller holds the lock."""
        if self._snap_free:
            return self._snap_free.pop()
        idle = [(meta[2], snap) for snap, meta in self._snap_meta.items()
                if snap not in self._snap_refs]
        if not idle:
            return None
        _, snap = min(idle)
        del self._snap_by_key[self._snap_meta.pop(snap)[0]]
        return snap

    def _apply_restores(self, step: RecurrentStep) -> None:
        """Copy each hit's snapshot into its slot before its first step reads it."""
        rids = {seg.request_id for seg in step.segments or ()}
        with self._lock:
            for key in [k for k in self._restores if k[0] in rids]:
                snap, _ = self._restores.pop(key)
                slot = self._slots.get(key[0], {}).get(key[1])
                if slot is not None:
                    for to, from_ in zip(
                        self._slot_views(slot.index), self._slot_views(snap), strict=True,
                    ):
                        to.copy_(from_)
                    slot.has_state = True
                    self._restored[key] = snap
                # the copy is queued ahead of anything that could overwrite the snapshot
                self._unref(snap)

    def _commit_snapshots(self, step: RecurrentStep) -> None:
        """Index what this step's forward saved, and stamp what it restored."""
        rids = {seg.request_id for seg in step.segments or ()}
        with self._lock:
            for key in [k for k in self._restored if k[0] in rids]:
                snap = self._restored.pop(key)
                meta = self._snap_meta.get(snap)
                if meta is not None:
                    self._snap_clock += 1
                    self._snap_meta[snap] = (*meta[:2], self._snap_clock)
            for key in [k for k in self._snap_writes if k[0] in rids]:
                snap = self._snap_writes.pop(key)
                rid, label, tokens = key
                keys = self._prefix_keys.get(rid, {}).get(label) or []
                pages = tokens // self.config.prefix_page_size
                written = snap in self._snap_written
                self._snap_written.discard(snap)
                if not written or not 0 < pages <= len(keys):
                    self._snap_free.append(snap)
                    continue
                name = fingerprint(self._prefix_root, keys[pages - 1])
                if name in self._snap_by_key:
                    self._snap_free.append(snap)  # an earlier request's copy stands
                    continue
                self._snap_clock += 1
                self._snap_by_key[name] = snap
                self._snap_meta[snap] = (name, tokens, self._snap_clock)

    def _undo_prefix_admit(self, is_mine) -> None:
        """Hand back what admits took for prefix reuse, for the records ``is_mine`` accepts:
        restores become leases again, and reserved snapshot slots go back. Caller holds
        the lock."""
        for rid, label in [k for k in self._admit_restores if is_mine(k)]:
            restore = self._restores.pop((rid, label), None)
            if restore is not None:
                self._leases[rid] = (label, *restore)
        for key in [k for k in self._admit_snaps if is_mine(k)]:
            snap = self._snap_writes.pop(key, None)
            if snap is not None:
                self._snap_written.discard(snap)
                self._snap_free.append(snap)
        self._forget_prefix_admit(is_mine)

    def _forget_prefix_admit(self, is_mine) -> None:
        """Drop the rollback record of the restores and snapshot slots ``is_mine`` accepts
        (a key's first item is its row), as `_forget_taken` does slots. Caller holds the lock."""
        gone = {k for k in (*self._admit_restores, *self._admit_snaps) if is_mine(k)}
        if gone:
            self._admit_restores = [k for k in self._admit_restores if k not in gone]
            self._admit_snaps = [k for k in self._admit_snaps if k not in gone]
            self._preplan_prefix -= gone

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
        del slots, max_seq_len
        # per-key buffers are built on first plan; every runner calls in, so
        # keep the largest
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def post_warmup_validate(self):
        """Every rank of the group must hold the same free slots after warmup.

        Each rank admits from its own pool, so identical free lists are what
        make their verdicts agree; see ``KVManager.post_warmup_validate``.
        """
        if self._comm_group is None or self._comm_group.world_size == 1:
            return
        local = torch.tensor(
            [self.num_free_slots], dtype=torch.int64, device=self._device,
        )
        for group in (self._comm_group.tp_group, self._comm_group.sp_group):
            values = group.all_gather(local, dim=0).cpu().tolist()
            if any(v != values[0] for v in values):
                raise RuntimeError(
                    f"recurrent state pool has asymmetric free slots across "
                    f"ranks: {values}. Check the YAML for a per-rank max_slots, "
                    "and any code that admits before warmup completes."
                )

    def cleanup(self):
        self._blocks.clear()
        self._cg_addressing.clear()
        self._cg_mirrors.clear()
        self._eager_addressing.clear()
        self._current.clear()
