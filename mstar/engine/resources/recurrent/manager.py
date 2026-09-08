"""The recurrent-state resource: slot allocation, per-step addressing, offload.

Lifecycle, mirroring ``KVManager`` so the engine and worker treat it the same way:

* ``ingest_request``: nothing is allocated yet.
* ``admit``: every segment with a non-zero span gets a slot per label (or the request
  keeps the one it has); exhaustion returns ``AllocationFailed`` so the worker's
  eviction path applies. Zero-span segments (replay padding) reserve nothing.
* ``plan``: builds ``RecurrentPlanOutput`` (slot ids + has-state flags) in declaration
  order, into the static buffers of the leased CUDA-graph slot when there is one, and
  publishes it on ``ctx.plan_results``. Pre-planning is supported like KV.
* ``commit``: marks the rows as holding state (``IN_PLACE``/``CHECKPOINT``); the
  kernels wrote the slots during the forward. ``DEFERRED`` steps are committed through
  ``commit_deferred`` once the accepted lengths are known.
* Eviction: a request's slots can be parked in pinned host memory and brought back.

Layers reach the storage through ``part(name)`` (``[num_layers, slots + 1, *shape]``)
or ``layer_view(name, layer_idx)`` with the same layer-index cursor the attention
resources use.
"""
from __future__ import annotations

import logging
import threading

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import AttentionResource, CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.recurrent.config import (
    CommitMode,
    RecurrentPlanOutput,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStateStep,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    Segment,
    SlotLease,
    StepContext,
)

logger = logging.getLogger(__name__)

SCRATCH_SLOT = 0


class RecurrentStateManager(AttentionResource):
    def __init__(
        self,
        cfg: RecurrentStateConfig,
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
        self._parts: dict[str, torch.Tensor] = {
            part_name: torch.zeros(
                (cfg.num_layers, cfg.max_num_slots + 1, *part.shape),
                dtype=part.dtype, device=device,
            )
            for part_name, part in cfg.parts.items()
        }
        # slot ids 1..max_num_slots; 0 is scratch
        self._allocator = PageAllocator(cfg.max_num_slots + 1)
        assert self._allocator.allocate(1) == [SCRATCH_SLOT]
        # rid -> label -> slot
        self._slots: dict[str, dict[str, int]] = {}
        # (rid, label) holding a written state
        self._resident: set[tuple[str, str]] = set()
        self._in_flight: set[tuple[str, str]] = set()
        self._offloaded: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        self._lock = threading.RLock()

        self._cg_max_bs = 0
        self._static: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._current_plan: RecurrentPlanOutput | None = None
        self._preplan: RecurrentPlanOutput | None = None
        self._preplanned = False
        self._preplan_new: list[tuple[str, str]] = []
        self._preplan_marked: list[tuple[str, str]] = []
        self.reset_default_cursors()

    @classmethod
    def build(cls, spec: RecurrentStateSpec, info: EngineResourceInfo):
        return cls(
            cfg=spec.config, name=spec.resource_key, device=info.device,
            joint_comm_group=info.joint_comm_group,
        )

    # ------------------------------------------------------------------ storage
    def part(self, name: str) -> torch.Tensor:
        return self._parts[name]

    @torch.compiler.disable
    def layer_view(self, name: str, layer_idx: int | None = None) -> torch.Tensor:
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        assert layer_idx is not None, "set_default_layer_idx first or pass layer_idx"
        return self._parts[name][layer_idx]

    @property
    def plan_output(self) -> RecurrentPlanOutput:
        assert self._current_plan is not None, "plan before reading the step's addressing"
        return self._current_plan

    @property
    def num_free_slots(self) -> int:
        return self._allocator.num_free

    # ------------------------------------------------------------- request life
    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None):
        with self._lock:
            self._slots.setdefault(rid, {})

    def remove_request(self, rid: str):
        with self._lock:
            slots = self._slots.pop(rid, {})
            for label, slot in slots.items():
                self._allocator.free([slot])
                self._resident.discard((rid, label))
                self._in_flight.discard((rid, label))
            self._offloaded.pop(rid, None)

    def reset_request(self, rid: str, free: bool = False):
        with self._lock:
            slots = self._slots.get(rid)
            if slots is None:
                return
            for label in list(slots):
                self._resident.discard((rid, label))
                self._in_flight.discard((rid, label))
                if free:
                    self._allocator.free([slots.pop(label)])

    # ---------------------------------------------------------------- the step
    def _ensure_slot(self, rid: str, label: str) -> int | None:
        slots = self._slots.setdefault(rid, {})
        slot = slots.get(label)
        if slot is not None:
            return slot
        got = self._allocator.try_allocate(1)
        if got is None:
            return None
        slots[label] = got[0]
        # a fresh slot starts from zeros in every layer, so the kernels never need to mask a
        # first step's initial state (the decode path relies on this: it reads the slots
        # unconditionally); a reload copies the parked state over it afterwards
        self._zero_slot(got[0])
        return got[0]

    def _zero_slot(self, slot: int) -> None:
        for part in self._parts.values():
            part[:, slot].zero_()

    def admit(self, step: RecurrentStateStep, ctx: StepContext) -> AdmitOutcome:
        if self._preplanned and not ctx.is_preplan:
            return ADMIT_OK
        if getattr(ctx, "capture", False):
            # CUDA-graph capture: the dummy rows address the scratch slot (``_addressing``
            # maps rows without a slot there), so a capture never consumes state slots and
            # the captured kernels are the same slot-indexed ones the replay runs
            return ADMIT_OK
        padding = self._padding_rows(ctx)
        with self._lock:
            if ctx.is_preplan:
                self._preplan_new = []
                self._preplan_marked = []
            for seg in step.segments:
                if seg.span == 0 or seg.request_id in padding:
                    continue
                if seg.request_id in self._offloaded and seg.label in self._offloaded[seg.request_id]:
                    # the worker reloads before re-driving the step
                    from mstar.engine.resources.step import RequestOffloading

                    return AdmitOutcome(ok=False, reason=RequestOffloading(
                        message=f"{self.name}: {seg.request_id}/{seg.label} is offloaded",
                        label=seg.label, request_id=seg.request_id,
                    ))
                had = seg.label in self._slots.get(seg.request_id, {})
                slot = self._ensure_slot(seg.request_id, seg.label)
                if slot is None:
                    return AdmitOutcome(ok=False, reason=AllocationFailed(
                        message=f"{self.name}: no free state slot for {seg.request_id}/{seg.label}",
                        pages_short=1, label=seg.label, request_id=seg.request_id,
                    ))
                if ctx.is_preplan and not had:
                    self._preplan_new.append((seg.request_id, seg.label))
            for seg in step.segments:
                key = (seg.request_id, seg.label)
                if seg.label in self._slots.get(seg.request_id, {}):
                    self._in_flight.add(key)
                    if ctx.is_preplan:
                        self._preplan_marked.append(key)
        return ADMIT_OK

    @staticmethod
    def _padding_rows(ctx: StepContext) -> set[str]:
        """The rows a captured replay pads the batch with (the runner's dummy rids). Unlike
        cache pages, a state slot is tens of MB, so padding rows never get one: they run
        against the scratch slot and their (discarded) outputs are computed from zeros."""
        padded = ctx.padded_request_ids
        if padded is ctx.request_ids or len(padded) == len(ctx.request_ids):
            return set()
        real = set(ctx.request_ids)
        return {rid for rid in padded if rid not in real}

    def _static_buffers(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One device buffer per capture slot holding ``[slot_ids | has_state | cu_seqlens]``
        (int32, ``3 * max_bs + 1`` entries) plus its pinned host staging twin, so a plan is a
        single asynchronous copy. Returns ``(ids, flags, cu, (device, host))`` views."""
        bufs = self._static.get(slot)
        if bufs is None:
            n = self._cg_max_bs
            dev = torch.zeros((3 * n + 1,), dtype=torch.int32, device=self._device)
            host = torch.zeros((3 * n + 1,), dtype=torch.int32, pin_memory=self._device.type == "cuda")
            bufs = self._static[slot] = (dev[:n], dev[n:2 * n], dev[2 * n:], (dev, host))
        return bufs

    def build_cuda_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int):
        del slots, max_seq_len
        if max_bs > self._cg_max_bs:
            self._cg_max_bs = max_bs
            self._static = {}

    def _addressing(self, segments: list[Segment]) -> tuple[list[int], list[bool], list[int], bool]:
        slot_ids, has_state, cu = [], [], [0]
        is_decode = True
        for seg in segments:
            slot = self._slots.get(seg.request_id, {}).get(seg.label)
            if slot is None or seg.span == 0:
                slot_ids.append(SCRATCH_SLOT)
                has_state.append(False)
            else:
                slot_ids.append(slot)
                has_state.append((seg.request_id, seg.label) in self._resident)
            cu.append(cu[-1] + seg.span)
            if seg.span != 1:
                is_decode = False
        return slot_ids, has_state, cu, is_decode

    def plan(self, step: RecurrentStateStep, ctx: StepContext) -> RecurrentPlanOutput:
        assert not (self._preplanned and ctx.is_preplan), (
            f"{self.name}: a preplan is already pending; clear_preplan first"
        )
        self.reset_default_cursors()
        if self._preplanned:
            res = self._preplan
            self._current_plan = res
            self._preplan_new = []
            self._preplan_marked = []
            self.clear_preplan()
            return res
        with self._lock:
            slot_ids, has_state, cu, is_decode = self._addressing(list(step.segments))
        lease: SlotLease | None = ctx.slot_lease
        rows = len(slot_ids)
        if is_decode and (SCRATCH_SLOT in slot_ids or (lease is not None and rows < lease.bucket.bs)):
            # padding rows (and rows without a slot) run against the scratch slot; keep it at
            # zeros so their discarded outputs stay finite: two small memsets per step
            self._zero_slot(SCRATCH_SLOT)
        if lease is not None:
            ids_buf, flags_buf, cu_buf, (dev, host) = self._static_buffers(lease.slot)
            n = ids_buf.numel()
            assert rows <= n, (rows, n)
            # fill the pinned twin on the host (numpy slice assignment, no tensor constructions),
            # then one asynchronous copy; padding rows address the scratch slot and their
            # boundaries repeat the last real one
            arr = host.numpy()
            arr[:rows] = slot_ids
            arr[rows:n] = SCRATCH_SLOT
            arr[n:n + rows] = has_state
            arr[n + rows:2 * n] = 0
            arr[2 * n:2 * n + rows + 1] = cu
            arr[2 * n + rows + 1:] = cu[-1]  # cu region is n + 1 entries
            dev.copy_(host, non_blocking=True)
            ids_t, flags_t, cu_t = ids_buf, flags_buf, cu_buf
        else:
            packed = torch.tensor([*slot_ids, *[int(h) for h in has_state], *cu], dtype=torch.int32)
            packed = packed.to(self._device, non_blocking=True)
            ids_t, flags_t, cu_t = packed[:rows], packed[rows:2 * rows], packed[2 * rows:]
        res = RecurrentPlanOutput(
            slot_ids=ids_t, has_state=flags_t, cu_seqlens=cu_t,
            slot_ids_cpu=slot_ids, has_state_cpu=has_state, cu_seqlens_cpu=cu,
            is_decode=is_decode, num_rows=rows, num_tokens=cu[-1],
        )
        if ctx.is_preplan:
            self._preplan = res
            self._preplanned = True
        else:
            self._current_plan = res
        return res

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        with self._lock:
            for rid, label in reversed(self._preplan_new):
                slot = self._slots.get(rid, {}).pop(label, None)
                if slot is not None:
                    self._allocator.free([slot])
            self._preplan_new = []
            for key in self._preplan_marked:
                self._in_flight.discard(key)
            self._preplan_marked = []
        self._preplanned = False
        self._preplan = None

    def commit(self, step: RecurrentStateStep, ctx: StepContext, outcome=None) -> None:
        del outcome
        if step.commit_mode is CommitMode.DEFERRED:
            # the forward left the resident slots untouched; commit_deferred lands the
            # accepted prefix once the sampler decides it
            return
        with self._lock:
            for seg in step.segments:
                key = (seg.request_id, seg.label)
                self._in_flight.discard(key)
                if seg.span > 0 and seg.label in self._slots.get(seg.request_id, {}):
                    self._resident.add(key)

    def commit_deferred(self, request_ids: list[str], accepted: list[int], label: str = "main") -> None:
        """Mark the rows of a ``DEFERRED`` step resident once their accepted prefixes were
        replayed into the slots by the model (speculative decoding)."""
        with self._lock:
            for rid, n in zip(request_ids, accepted, strict=True):
                self._in_flight.discard((rid, label))
                if n > 0 and label in self._slots.get(rid, {}):
                    self._resident.add((rid, label))

    # ---------------------------------------------------------------- eviction
    @property
    def supports_eviction(self):
        return self.config.cpu_offload_slots > 0

    def is_offloaded(self, rid: str) -> bool:
        return rid in self._offloaded

    def reclaimable(self, rid: str) -> int:
        with self._lock:
            slots = self._slots.get(rid, {})
            if not slots or any((rid, label) in self._in_flight for label in slots):
                return 0
            return len(slots)

    def offload(self, rid: str) -> int:
        with self._lock:
            slots = self._slots.get(rid, {})
            if not slots or self.reclaimable(rid) == 0:
                return 0
            if sum(len(v) for v in self._offloaded.values()) >= self.config.cpu_offload_slots:
                return 0
            parked: dict[str, dict[str, torch.Tensor]] = {}
            for label, slot in list(slots.items()):
                parked[label] = {
                    name: t[:, slot].to("cpu", non_blocking=False).pin_memory()
                    if t.is_cuda else t[:, slot].clone()
                    for name, t in self._parts.items()
                }
                self._allocator.free([slot])
                del slots[label]
                self._resident.discard((rid, label))
            self._offloaded[rid] = parked
            return len(parked)

    def reload(self, rid: str) -> bool:
        with self._lock:
            parked = self._offloaded.get(rid)
            if parked is None:
                return True
            needed = len(parked)
            if self._allocator.num_free < needed:
                return False
            for label, tensors in parked.items():
                slot = self._ensure_slot(rid, label)
                assert slot is not None
                for name, t in tensors.items():
                    self._parts[name][:, slot].copy_(t, non_blocking=True)
                self._resident.add((rid, label))
            del self._offloaded[rid]
            return True

    def get_offload_priority(self, rid: str) -> float:
        return float(self.reclaimable(rid))

    # ------------------------------------------------------------------ misc
    def post_warmup_validate(self):
        if self._comm_group is None or self._comm_group.world_size == 1:
            return
        local = torch.tensor([self._allocator.num_free], dtype=torch.int64, device=self._device)
        for group in (self._comm_group.tp_group, self._comm_group.sp_group):
            values = group.all_gather(local, dim=0).cpu().tolist()
            if any(v != values[0] for v in values):
                raise RuntimeError(
                    f"recurrent state {self.name!r} has asymmetric free slots across ranks: {values}"
                )

    def slot_of(self, rid: str, label: str = "main") -> int | None:
        return self._slots.get(rid, {}).get(label)

    def has_state(self, rid: str, label: str = "main") -> bool:
        return (rid, label) in self._resident
