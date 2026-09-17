"""Acceptance counts of a speculating node, from the device to the host bookkeeping.

A verify step processes ``k + 1`` tokens per request and keeps ``accepted + 1`` of them; the
resources that grew by ``k + 1`` at commit (host bookkeeping, done as the step is enqueued) have
to take the rejected tail back before the next step is planned. The counts are known on the
device only, mid-step, so this resource carries them across:

* ``stage(accepted)`` runs inside the forward, right after the verification (captured into the
  step's CUDA graph): the counts go to a static per-slot device buffer and from there to a pinned
  host mirror, and a step counter is bumped and mirrored the same way, in that order, so a host
  that sees the counter at ``seq`` also sees the counts of step ``seq``.
* ``note_step(request_ids)`` runs on the host once the step is enqueued (replayed or eager) and
  remembers, per real request, which step and row hold its verdict.
* ``plan`` (first in the runner's order: it depends on nothing) settles every pending verdict
  whose step is enqueued, waiting on the host mirror of the counter (the wait overlaps the running
  step's tail, the draft pass) and copying the counts out before their slot buffer is reused, then
  publishes ``{rid: SpecAccepted}`` for the requests in this step. The KV manager reads it under
  ``SPEC_ACCEPTANCE`` and trims its lengths before building the sequence views.

Storage only: no kernels, no request state beyond the pending verdicts.
"""

from __future__ import annotations

import logging
import time

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.speculative.config import SpecAccepted, SpecAcceptanceSpec, SpecStep
from mstar.engine.resources.step import StepContext

logger = logging.getLogger(__name__)

# a step whose counts never arrive within this long is a hang somewhere else; say so
WAIT_TIMEOUT_S = 120.0


class SpecAcceptance(Resource):
    def __init__(self, device: torch.device, num_speculative: int):
        self.device = device
        self.k = int(num_speculative)
        on_gpu = torch.device(device).type == "cuda"
        pin = dict(pin_memory=True) if on_gpu and torch.cuda.is_available() else {}
        self._pin = pin
        self._counter_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self._counter_host = torch.zeros(1, dtype=torch.int32, **pin)
        # per CUDA-graph slot (None: eager, sized per step): device buffer and pinned mirror
        self._acc_dev: dict[int | None, torch.Tensor] = {}
        self._acc_host: dict[int | None, torch.Tensor] = {}
        self._cg_max_bs = 0
        self._current_slot: int | None = None
        self._seq_enqueued = 0  # verify steps staged so far, host view
        # rid -> (seq, slot, row) until its counts are read, then -> int accepted
        self._pending: dict[str, tuple[int, int | None, int] | int] = {}
        self._max_rows_seen = 0

    @classmethod
    def build(cls, spec: SpecAcceptanceSpec, info: EngineResourceInfo):
        return cls(info.device, spec.num_speculative)

    # ----------------------------------------------------------------- lifecycle
    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._pending.pop(rid, None)

    def remove_request(self, rid: str):
        self._pending.pop(rid, None)

    def reset_request(self, rid: str, free: bool = False):
        del free
        self._pending.pop(rid, None)

    def build_cuda_graph_buffers(self, slots, max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._acc_dev.clear()
        self._acc_host.clear()
        self._pending.clear()

    # ---------------------------------------------------------------------- step
    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        # what a plan published is a fact about an earlier step; nothing to rewind
        return

    def plan(self, step: SpecStep, ctx: StepContext) -> dict[str, SpecAccepted]:
        self._current_slot = ctx.slot_lease.slot if ctx.slot_lease is not None else None
        # the buffers `stage` writes are made here, before a capture step's forward records them
        # (an allocation inside the capture would be the graph's, and a pinned host allocation there
        # is not allowed at all)
        self._buffers(self._current_slot, max(len(ctx.padded_request_ids), 1))
        self._settle()
        out: dict[str, SpecAccepted] = {}
        for segment in step.segments or ():
            rid = segment.request_id
            if ctx.is_padding_row(rid):
                continue
            verdict = self._pending.get(rid)
            if isinstance(verdict, int):
                out[rid] = SpecAccepted(accepted=verdict, rejected=self.k - verdict, label=segment.label)
                del self._pending[rid]
        return out

    def commit(self, step: SpecStep, ctx: StepContext) -> None:
        return

    # ------------------------------------------------------------ submodule side
    def _buffers(self, slot: int | None, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The slot's device buffer and pinned mirror, at least ``rows`` wide (a capture slot's are
        sized to the widest bucket any runner announced and never shrink; the eager slot, None, is
        resized as needed and rewritten every step)."""
        size = max(self._cg_max_bs, rows) if slot is not None else rows
        dev = self._acc_dev.get(slot)
        if dev is None or dev.shape[0] < size:
            dev = torch.zeros(size, dtype=torch.int32, device=self.device)
            host = torch.zeros(size, dtype=torch.int32, **self._pin)
            self._acc_dev[slot], self._acc_host[slot] = dev, host
        return self._acc_dev[slot], self._acc_host[slot]

    def stage(self, accepted: torch.Tensor) -> None:
        """Inside the forward, after the verification: publish ``accepted [rows]`` (int32, real rows
        first) to the host mirror of the current slot and bump the step counter after it. Capturable:
        two device copies, two device-to-host copies into pinned memory."""
        rows = int(accepted.shape[0])
        dev, host = self._buffers(self._current_slot, rows)  # made in plan; a no-op here
        dev[:rows].copy_(accepted.to(torch.int32))
        host[:rows].copy_(dev[:rows], non_blocking=True)
        self._counter_dev.add_(1)
        self._counter_host.copy_(self._counter_dev, non_blocking=True)

    def note_step(self, request_ids: list[str]) -> int:
        """On the host once the step is enqueued: row ``i`` of the current slot's mirror will hold
        ``request_ids[i]``'s verdict when the counter reaches the returned sequence number."""
        self._seq_enqueued += 1
        for row, rid in enumerate(request_ids):
            self._pending[rid] = (self._seq_enqueued, self._current_slot, row)
        return self._seq_enqueued

    def accepted_for(self, request_ids: list[str]) -> list[int]:
        """The verdicts of these requests' last verify step (waits for it). For slicing the
        emitted tokens post-replay; the pending entries stay for the next plan to publish."""
        self._settle()
        out = []
        for rid in request_ids:
            verdict = self._pending.get(rid)
            if not isinstance(verdict, int):
                raise RuntimeError(f"no settled verdict for {rid}: note_step before accepted_for")
            out.append(verdict)
        return out

    def _settle(self) -> None:
        """Read every pending verdict whose step has been staged, after waiting for its counter.
        Values are copied out here, before a later step reuses the slot's buffers."""
        latest = 0
        for verdict in self._pending.values():
            if not isinstance(verdict, int):
                latest = max(latest, verdict[0])
        if latest == 0:
            return
        self._wait(latest)
        for rid, verdict in list(self._pending.items()):
            if isinstance(verdict, int):
                continue
            seq, slot, row = verdict
            self._pending[rid] = int(self._acc_host[slot][row])

    def _wait(self, seq: int) -> None:
        if self.device.type != "cuda":
            return  # copies are synchronous off the GPU
        deadline = time.monotonic() + WAIT_TIMEOUT_S
        while int(self._counter_host[0]) < seq:
            if time.monotonic() > deadline:
                raise RuntimeError(f"speculative step {seq} never reported its acceptance counts "
                                   f"(host counter at {int(self._counter_host[0])})")
            time.sleep(0)
