"""Slot-pooled fixed-size per-request state (recurrent / conv state of
linear-attention layers) as an engine resource.
"""

from __future__ import annotations

import logging
import threading

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.slot_state.config import (
    SINK_SLOT,
    SlotSpan,
    SlotStateConfig,
    SlotStatePlan,
    SlotStateSpec,
    SlotStateStep,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    StepContext,
)
from mstar.utils.pinned_staging import pinned

logger = logging.getLogger(__name__)


class SlotStateManager(Resource):
    def __init__(
        self,
        cfg: SlotStateConfig,
        name: str,
        device: torch.device,
        joint_comm_group: JointGroups | None = None,
    ):
        self.config = cfg
        self.name = name
        self._device = device
        self._comm_group = joint_comm_group
        if joint_comm_group is not None:
            cfg.shard(joint_comm_group.world_size)
        self._pools: dict[str, torch.Tensor] = {
            tname: torch.zeros(
                spec.pool_shape(cfg.max_slots), dtype=spec.dtype, device=device,
            )
            for tname, spec in cfg.tensors.items()
        }
        self._slot_of: dict[str, int] = {}
        self._committed: dict[str, int] = {}
        # LIFO from 1: SINK_SLOT stays out of circulation
        self._free_slots: list[int] = list(range(cfg.max_slots, SINK_SLOT, -1))
        self._lock = threading.RLock()

        self._current_plan: SlotStatePlan | None = None
        # eager steps stage their slot index here (grown on demand); leased
        # steps into the per-cg-slot static buffers the captures recorded
        self._eager_index = torch.zeros(
            cfg.max_slots + 1, dtype=torch.int64, device=device,
        )
        self._cg_index: dict[int, torch.Tensor] = {}

    @classmethod
    def build(cls, spec: SlotStateSpec, info: EngineResourceInfo):
        return cls(
            cfg=spec.config,
            name=spec.resource_key,
            device=info.device,
            joint_comm_group=info.joint_comm_group,
        )

    # -- introspection ----------------------------------------------------

    @property
    def max_slots(self) -> int:
        return self.config.max_slots

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def device(self) -> torch.device:
        return self._device

    def pool(self, name: str) -> torch.Tensor:
        """The whole pool tensor for ``name``; the slot axis sits at the
        spec's ``slot_dim`` and holds ``max_slots + 1`` rows."""
        return self._pools[name]

    def slot_view(self, name: str, slot: int) -> torch.Tensor:
        """One slot's state for ``name``, an in-place view of the pool with
        the per-request shape the spec declared."""
        spec = self.config.tensors[name]
        return self._pools[name].select(spec.slot_dim, slot)

    def slot_of(self, rid: str) -> int | None:
        return self._slot_of.get(rid)

    def committed(self, rid: str) -> int:
        return self._committed.get(rid, 0)

    def set_committed(self, rid: str, tokens: int) -> None:
        """Rewind or restore a request's committed length (speculative undo)."""
        with self._lock:
            if self.slot_of(rid) is None:
                raise KeyError(f"request {rid!r} holds no slot")
            self._committed[rid] = int(tokens)

    def tracked_requests(self) -> set[str]:
        return set(self._slot_of)

    def current_plan(self) -> SlotStatePlan:
        if self._current_plan is None:
            raise RuntimeError(
                f"slot state {self.name!r}: no step planned; the forward runs "
                "between plan and commit"
            )
        return self._current_plan

    # -- slots ------------------------------------------------------------

    def _zero_slot(self, slot: int) -> None:
        for name, spec in self.config.tensors.items():
            self._pools[name].select(spec.slot_dim, slot).zero_()

    def _alloc(self, rid: str) -> bool:
        """Lease a zeroed slot to ``rid``; False when the pool is exhausted."""
        with self._lock:
            if rid in self._slot_of:
                return True
            if not self._free_slots:
                return False
            slot = self._free_slots.pop()
            self._slot_of[rid] = slot
            self._committed.setdefault(rid, 0)
        # zero state = the recurrence from nothing; zero conv tail = the
        # first chunk's zero left-padding, so a fresh slot goes through the
        # same continue path as a resumed one
        self._zero_slot(slot)
        return True

    def _release(self, rid: str) -> None:
        with self._lock:
            slot = self._slot_of.pop(rid, None)
            if slot is not None:
                self._free_slots.append(slot)

    # -- request lifecycle ------------------------------------------------

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None):
        # a slot is leased at the first admit with tokens, not here: the
        # runner ingests every capture dummy row up front and a pool sized
        # for the serve batch must not be drained by them
        with self._lock:
            self._committed.setdefault(rid, 0)

    def remove_request(self, rid: str):
        with self._lock:
            self._release(rid)
            self._committed.pop(rid, None)

    def reset_request(self, rid: str, free: bool = False):
        """Padding rows between captures / after a replay: forget what they
        committed and hand the slot back.
        """
        del free
        with self._lock:
            self._committed[rid] = 0
            self._release(rid)

    # -- step lifecycle ---------------------------------------------------

    def admit(self, step: SlotStateStep, ctx: StepContext) -> AdmitOutcome:
        real = set(ctx.request_ids)
        for segment in step.segments:
            if segment.span <= 0 or segment.request_id not in real:
                continue
            if not self._alloc(segment.request_id):
                # AllocationFailed, like the KV cache: it is what the worker's
                # hold/backoff path keys on. Nothing here is evictable, so the
                # batch is held until a resident request releases a slot.
                return AdmitOutcome(
                    ok=False,
                    ready=True,
                    reason=AllocationFailed(
                        f"slot state {self.name!r}: pool of {self.max_slots} "
                        f"slots exhausted admitting {segment.request_id!r}",
                        pages_short=1,
                        label=segment.label,
                        request_id=segment.request_id,
                    ),
                )
        return ADMIT_OK

    def _index_buffer(self, ctx: StepContext, rows: int) -> torch.Tensor:
        lease = ctx.slot_lease
        if lease is not None and lease.bucket is not None:
            buf = self._cg_index.get(lease.slot)
            if buf is None or buf.shape[0] < rows:
                raise RuntimeError(
                    f"slot state {self.name!r}: cg slot {lease.slot} has no "
                    f"static index buffer for {rows} rows; "
                    "build_cuda_graph_buffers must size it before capture"
                )
            return buf[:rows]
        if self._eager_index.shape[0] < rows:
            self._eager_index = torch.zeros(
                rows, dtype=torch.int64, device=self._device,
            )
        return self._eager_index[:rows]

    def plan(self, step: SlotStateStep, ctx: StepContext) -> SlotStatePlan:
        real = set(ctx.request_ids)
        spans: list[SlotSpan] = []
        slots: list[int] = []
        q_start = 0
        for segment in step.segments:
            rid = segment.request_id
            slot = self._slot_of.get(rid)
            is_real = rid in real and slot is not None
            if not is_real:
                slot = SINK_SLOT
            ctx_start = self._committed.get(rid, 0) if is_real else 0
            # A capture records a single-token step on dummy rows that never
            # ran a chunk: their zero state is fine to record against (replay
            # reads real rows), so the guard is for real steps only.
            if is_real and step.mode == "step" and ctx_start == 0 and not ctx.capture:
                raise RuntimeError(
                    f"slot state {self.name!r}: single-token step for "
                    f"{rid!r} with no committed tokens — a step before its "
                    "first chunk is a scheduling bug"
                )
            spans.append(SlotSpan(
                request_id=rid, slot=slot, q_start=q_start,
                q_len=segment.span, ctx_start=ctx_start, real=is_real,
            ))
            slots.append(slot)
            q_start += segment.span

        index = self._index_buffer(ctx, len(slots))
        if slots:
            host = pinned(slots, dtype=torch.int64)
            if host.device == index.device:
                index.copy_(host)
            else:
                index.copy_(host, non_blocking=True)
        plan = SlotStatePlan(mode=step.mode, slot_index=index, spans=spans)
        self._current_plan = plan
        return plan

    def commit(self, step: SlotStateStep, ctx: StepContext) -> None:
        if not step.commit:
            return
        with self._lock:
            for span in self._current_plan.spans if self._current_plan else ():
                if span.real:
                    self._committed[span.request_id] = (
                        self._committed.get(span.request_id, 0) + span.q_len
                    )

    # -- engine lifecycle -------------------------------------------------

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del max_seq_len
        for spec in slots:
            buf = self._cg_index.get(spec.slot)
            if buf is None or buf.shape[0] < max_bs:
                self._cg_index[spec.slot] = torch.zeros(
                    max_bs, dtype=torch.int64, device=self._device,
                )

    def post_warmup_validate(self):
        """Every rank must hold the same number of free slots after warmup —
        TP leases slots in lockstep, so a divergence here means a rank
        admitted something the others did not."""
        if self._comm_group is None or self._comm_group.world_size == 1:
            return
        local = torch.tensor([self.num_free], dtype=torch.int64, device=self._device)
        for group in [self._comm_group.tp_group, self._comm_group.sp_group]:
            values = group.all_gather(local, dim=0).cpu().tolist()
            if any(v != values[0] for v in values):
                raise RuntimeError(
                    f"slot state {self.name!r} has asymmetric free slots "
                    f"across ranks: {values}"
                )

    def cleanup(self):
        self._pools.clear()
        self._cg_index.clear()
