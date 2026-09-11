"""A pool of fixed-size recurrent state slots.

Storage only. The pool hands out slots, keeps the per-layer tensors those slots
index into, and says which slot each row of a step addresses; what a backend
writes there, and with which kernel, is the backend's business.

The split mirrors ``kv/`` and ``attn/``: this is the cache, and a linear-
attention resource is the wrapper planned against it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, EngineResourceInfo, Resource
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

logger = logging.getLogger(__name__)

# A row addressing no slot. Every backend the pool serves has to treat this as
# "skip", which FlashInfer's pool paths already do (see the -1 handling in
# `gated_delta_rule_decode_pretranspose`). Padding rows from a capture bucket
# and zero-span segments both land here.
NO_SLOT = -1


@dataclass
class SlotState:
    index: int
    # Whether the slot holds state a later step should resume from. False for a
    # freshly allocated slot, which reads as zeros.
    has_state: bool = False
    # Bumped on fork, like `CacheStream.generation`: lets a stale plan be
    # spotted rather than silently attended through.
    generation: int = 0


@dataclass(frozen=True)
class RecurrentAddressing:
    """Where one label's rows live this step.

    The pool's whole plan output. Token layout (cu_seqlens, chunk metadata) is
    the backend's to build from the same segments — it is about tokens, not
    slots.
    """

    slot_indices: torch.Tensor  # [rows] int32, NO_SLOT where nothing is addressed
    has_state: torch.Tensor     # [rows] bool, False where the slot reads as zeros
    num_rows: int               # real rows, before a capture bucket's padding


class RecurrentStatePool(Resource):
    @classmethod
    def build(
        cls, spec: RecurrentStateSpec, info: EngineResourceInfo,
    ) -> "RecurrentStatePool":
        config = spec.config
        if info.joint_comm_group is not None:
            config.shard(info.joint_comm_group.world_size)
        return cls(device=info.device, config=config)

    def __init__(self, device: torch.device, config: RecurrentStateConfig):
        self._device = device
        self.config = config

        # One [num_layers, max_slots, *shape] tensor per block. Layer-major so
        # `blocks[name][layer]` is contiguous: FlashInfer's pool paths assert
        # the slot-major view is K-contiguous (stride(-1) == 1).
        self._blocks: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (config.num_layers, config.max_slots, *block.shape),
                dtype=block.dtype,
                device=device,
            )
            for name, block in config.blocks.items()
        }

        # rid -> label -> slot. Mirrors `KVManager._streams`.
        self._slots: dict[str, dict[str, SlotState]] = {}
        self._free: list[int] = list(range(config.max_slots))
        # guards `_slots`/`_free` against a concurrent admit/plan/commit or
        # reset/remove on another thread; see `KVManager._lock`
        self._lock = threading.RLock()

        self._cg_max_bs = 0
        self._cg_addressing: dict[CGSlotKey, RecurrentAddressing] = {}
        self._eager_addressing: dict[str, RecurrentAddressing] = {}
        # label -> this step's addressing, for the backend to read
        self._current: dict[str, RecurrentAddressing] = {}

        logger.info(
            "recurrent state pool: %d slots x %d layers, %.2f MiB "
            "(%.2f KiB/slot), blocks=%s",
            config.max_slots, config.num_layers,
            config.total_bytes / 2**20, config.slot_bytes / 2**10,
            {n: tuple(b.shape) for n, b in config.blocks.items()},
        )

    # Storage access

    def block(self, name: str, layer_idx: int) -> torch.Tensor:
        """One layer's slot-major view of a block: ``[max_slots, *shape]``.

        Contiguous, so a backend can hand it straight to a kernel that indexes
        the pool itself rather than gathering rows out of it.
        """
        return self._blocks[name][layer_idx]

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

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

    def reset_request(self, rid: str, free: bool = False):
        """Drop a request's slots. ``free`` hands them back to the pool.

        Both paths zero what they release: a slot is handed out as zeros, and
        the alternative is a backend resuming on another request's state.
        """
        with self._lock:
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
        for tensor in self._blocks.values():
            tensor[:, index].zero_()

    def _alloc(self, rid: str, label: str) -> SlotState | None:
        with self._lock:
            labels = self._slots.setdefault(rid, {})
            slot = labels.get(label)
            if slot is not None:
                return slot
            if not self._free:
                return None
            # Slot ids must be unique across the batch: an indexed scatter with
            # two rows naming one slot leaves it nondeterministic, and the
            # kernels take that as a caller precondition rather than checking
            # it (which would cost a host sync). A free list gives it for free.
            slot = labels[label] = SlotState(index=self._free.pop())
            return slot

    # Step lifecycle

    def admit(self, step: RecurrentStep, ctx: StepContext) -> AdmitOutcome:
        """Reserve a slot per addressed (rid, label), plus fork targets."""
        del ctx
        for segment in step.segments or ():
            if segment.span <= 0:
                # reads its state without extending it, or a padding row
                continue
            if self._alloc(segment.request_id, segment.label) is None:
                return self._out_of_slots(segment.request_id, segment.label)

        for rid in self._fork_rids(step):
            for _, to_label in (*step.pre_forks, *step.post_forks):
                if self._alloc(rid, to_label) is None:
                    return self._out_of_slots(rid, to_label)
        return ADMIT_OK

    def _out_of_slots(self, rid: str, label: str) -> AdmitOutcome:
        return AdmitOutcome(
            ok=False,
            reason=AllocationFailed(
                message=(
                    f"recurrent state pool is full: {self.config.max_slots} "
                    f"slots, none free for {rid}/{label}"
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

    def plan(self, step: RecurrentStep, ctx: StepContext):
        for rid in self._fork_rids(step):
            for from_label, to_label in step.pre_forks:
                self._apply_fork(rid, from_label, to_label)

        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_addressing(label, segments, ctx)
        return self._current

    def commit(self, step: RecurrentStep, ctx: StepContext) -> None:
        del ctx
        with self._lock:
            for segment in step.segments or ():
                if segment.span <= 0:
                    continue
                slot = self._slots.get(segment.request_id, {}).get(segment.label)
                if slot is not None:
                    # the backend has already written the pool; this only says a
                    # later step resumes rather than starting from zeros
                    slot.has_state = True

        for rid in self._fork_rids(step):
            for from_label, to_label in step.post_forks:
                self._apply_fork(rid, from_label, to_label)

    def _apply_fork(self, rid: str, from_label: str, to_label: str) -> None:
        """Copy one slot onto its fork target.

        A real copy, not the KV cache's page aliasing: the state is mutated in
        place, so two labels cannot share it. Fixed-size, which is why this
        needs none of ``KVManager._apply_fork``'s length arithmetic.
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
            dst.generation += 1

    # Addressing

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _build_addressing(
        self, label: str, segments: list[Segment], ctx: StepContext,
    ) -> RecurrentAddressing:
        indices = []
        has_state = []
        for seg in segments:
            slot = self._slots.get(seg.request_id, {}).get(label)
            if slot is None or seg.span <= 0:
                indices.append(NO_SLOT)
                has_state.append(False)
            else:
                indices.append(slot.index)
                has_state.append(slot.has_state)

        num_rows = len(indices)
        target = self._addressing_buffers(label, ctx, num_rows)
        # Built on the host and copied in: the values come from Python
        # bookkeeping, so building them on device would sync.
        pin = torch.cuda.is_available()
        target.slot_indices[:num_rows].copy_(
            torch.tensor(indices, dtype=torch.int32, pin_memory=pin),
            non_blocking=True,
        )
        target.has_state[:num_rows].copy_(
            torch.tensor(has_state, dtype=torch.bool, pin_memory=pin),
            non_blocking=True,
        )
        # Clear the tail rather than leaving last step's. Attention keeps a
        # replay off its padding with the plan's indptrs; a recurrent kernel
        # has no such thing and reads every row of the batch, so a stale index
        # here is a write to a slot whose request is not in this step.
        target.slot_indices[num_rows:].fill_(NO_SLOT)
        target.has_state[num_rows:].fill_(False)
        return RecurrentAddressing(
            slot_indices=target.slot_indices,
            has_state=target.has_state,
            num_rows=num_rows,
        )

    def _addressing_buffers(
        self, label: str, ctx: StepContext, num_rows: int,
    ) -> RecurrentAddressing:
        """The buffers this step fills.

        Under capture they are static and per (bucket, slot, label), built on
        the first plan for that key and sized to the largest batch any runner
        asked for — a bucket replays at several row counts, so this must not be
        sized off the first plan's.
        """
        lease = ctx.slot_lease
        if lease is None:
            buf = self._eager_addressing.get(label)
            if buf is None or buf.slot_indices.numel() < num_rows:
                buf = self._eager_addressing[label] = self._new_buffers(num_rows)
            return buf

        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        buf = self._cg_addressing.get(key)
        if buf is None:
            size = max(self._cg_max_bs, lease.bucket.bs, num_rows)
            buf = self._cg_addressing[key] = self._new_buffers(size)
        assert buf.slot_indices.numel() >= num_rows, (
            f"recurrent addressing for {key} holds "
            f"{buf.slot_indices.numel()} rows but this step planned {num_rows}"
        )
        return buf

    def _new_buffers(self, size: int) -> RecurrentAddressing:
        return RecurrentAddressing(
            slot_indices=torch.full(
                (size,), NO_SLOT, dtype=torch.int32, device=self._device
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

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        # The per-key buffers are built on first plan (which labels a walk
        # plans under is the step's to declare). Every runner capturing against
        # this node calls in, so keep the largest.
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._blocks.clear()
        self._cg_addressing.clear()
        self._eager_addressing.clear()
        self._current.clear()
