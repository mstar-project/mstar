"""Acceptance counts of a speculating node, from the device to the host bookkeeping.

A verify step processes ``k + 1`` tokens per request and keeps ``accepted + 1`` of them; the
resources that grew by ``k + 1`` at commit (host bookkeeping, done as the step is enqueued) have
to take the rejected tail back before the next step is planned. The counts are known on the
device only, mid-step, so this resource carries them across:

* ``stage(accepted, tokens)`` runs inside the forward, right after the verification (captured into
  the step's CUDA graph): the counts and the verified tokens go to static per-slot device buffers
  and from there to pinned host mirrors, and a step counter is bumped and mirrored the same way, in
  that order, so a host that sees the counter at ``seq`` also sees the counts and tokens of step
  ``seq``. The tokens let the post-replay cut of a row's output stop at a stop token without a sync.
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
import threading
import time
from typing import NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.speculative.config import SpecAccepted, SpecAcceptanceSpec, SpecStep
from mstar.engine.resources.step import StepContext

logger = logging.getLogger(__name__)

# a step whose counts never arrive within this long is a hang somewhere else; say so
WAIT_TIMEOUT_S = 120.0


class Verdict(NamedTuple):
    """One request's settled verdict: how many drafts the target accepted, and the ``k + 1``
    tokens it verified (the ones at ``[:accepted + 1]`` are the emitted ones)."""
    accepted: int
    tokens: list[int]


class SpecAcceptance(Resource):
    def __init__(self, device: torch.device, num_speculative: int):
        self.device = device
        self.k = int(num_speculative)
        on_gpu = torch.device(device).type == "cuda"
        pin = dict(pin_memory=True) if on_gpu and torch.cuda.is_available() else {}
        self._pin = pin
        self._counter_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self._counter_host = torch.zeros(1, dtype=torch.int32, **pin)
        # per CUDA-graph slot (None: eager, sized per step): device buffers and pinned mirrors of the
        # accepted counts [rows] and the verified tokens [rows, k + 1]
        self._acc_dev: dict[int | None, torch.Tensor] = {}
        self._acc_host: dict[int | None, torch.Tensor] = {}
        self._tok_dev: dict[int | None, torch.Tensor] = {}
        self._tok_host: dict[int | None, torch.Tensor] = {}
        self._cg_max_bs = 0
        self._current_slot: int | None = None
        self._seq_enqueued = 0  # verify steps committed so far, host view (= the device counter's target)
        # The plan thread pre-plans step N + 1 while the main thread still reads step N's outputs, so
        # everything below is guarded: rid -> (seq, slot, row) from the step's commit until its
        # mirrors are read; rid -> (seq, Verdict) from then until the request's next verify step or
        # its removal; rid -> the seq whose verdict a plan already published (the KV manager applies
        # a published trim once and does not undo it when a pre-plan is discarded).
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[int, int | None, int]] = {}
        self._settled: dict[str, tuple[int, Verdict]] = {}
        self._published: dict[str, int] = {}
        self._max_rows_seen = 0
        # acceptance statistics over the settled verdicts (every real row of every verify step)
        self.rows_settled = 0
        self.accepted_total = 0
        self._log_every = 2000

    @classmethod
    def build(cls, spec: SpecAcceptanceSpec, info: EngineResourceInfo):
        return cls(info.device, spec.num_speculative)

    # ----------------------------------------------------------------- lifecycle
    def _forget(self, rid: str) -> None:
        with self._lock:
            self._pending.pop(rid, None)
            self._settled.pop(rid, None)
            self._published.pop(rid, None)

    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._forget(rid)

    def remove_request(self, rid: str):
        self._forget(rid)

    def reset_request(self, rid: str, free: bool = False):
        del free
        self._forget(rid)

    def build_cuda_graph_buffers(self, slots, max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        if self.rows_settled:
            logger.info("speculation, final: %s", self.stats())
        for d in (self._acc_dev, self._acc_host, self._tok_dev, self._tok_host):
            d.clear()
        with self._lock:
            self._pending.clear()
            self._settled.clear()
            self._published.clear()

    # ---------------------------------------------------------------------- step
    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        # what a plan published is a fact about an earlier step; nothing to rewind
        return

    def plan(self, step: SpecStep, ctx: StepContext) -> dict[str, SpecAccepted]:
        """Publish each row's verdict from its last verify step, once: a pre-plan and the plan that
        promotes it, or a discarded pre-plan and the fresh plan after it, publish it a single time
        between them, since the KV manager's trim is applied once and never undone."""
        self._current_slot = ctx.slot_lease.slot if ctx.slot_lease is not None else None
        # the buffers `stage` writes are made here, before a capture step's forward records them
        # (an allocation inside the capture would be the graph's, and a pinned host allocation there
        # is not allowed at all)
        self._buffers(self._current_slot, max(len(ctx.padded_request_ids), 1))
        self._settle()
        out: dict[str, SpecAccepted] = {}
        with self._lock:
            for segment in step.segments or ():
                rid = segment.request_id
                if ctx.is_padding_row(rid):
                    continue
                entry = self._settled.get(rid)
                if entry is None or self._published.get(rid) == entry[0]:
                    continue
                seq, verdict = entry
                out[rid] = SpecAccepted(accepted=verdict.accepted, rejected=self.k - verdict.accepted, label=segment.label)
                self._published[rid] = seq
        return out

    def commit(self, step: SpecStep, ctx: StepContext) -> None:
        """A verify step's rows are registered as the step is committed, on the thread that enqueued
        its forward and before the plan thread may pre-plan the next step against it: row ``i`` of
        the slot's mirrors belongs to ``padded_request_ids[i]`` and settles when the device counter
        reaches this step's number (``stage`` bumps it once per verify forward)."""
        if not step.verify:
            return
        rows = ctx.padded_request_ids if ctx.padded_request_ids is not None else ctx.request_ids
        slot = ctx.slot_lease.slot if ctx.slot_lease is not None else None
        with self._lock:
            self._seq_enqueued += 1
            for row, rid in enumerate(rows):
                if not ctx.is_padding_row(rid):
                    self._pending[rid] = (self._seq_enqueued, slot, row)

    # ------------------------------------------------------------ submodule side
    def _buffers(self, slot: int | None, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The slot's device buffer and pinned mirror, at least ``rows`` wide (a capture slot's are
        sized to the widest bucket any runner announced and never shrink; the eager slot, None, is
        resized as needed and rewritten every step)."""
        size = max(self._cg_max_bs, rows) if slot is not None else rows
        dev = self._acc_dev.get(slot)
        if dev is None or dev.shape[0] < size:
            self._acc_dev[slot] = torch.zeros(size, dtype=torch.int32, device=self.device)
            self._acc_host[slot] = torch.zeros(size, dtype=torch.int32, **self._pin)
            self._tok_dev[slot] = torch.zeros(size, self.k + 1, dtype=torch.int32, device=self.device)
            self._tok_host[slot] = torch.zeros(size, self.k + 1, dtype=torch.int32, **self._pin)
        return self._acc_dev[slot], self._acc_host[slot]

    def stage(self, accepted: torch.Tensor, tokens: torch.Tensor | None = None) -> None:
        """Inside the forward, after the verification: publish ``accepted [rows]`` (int32, real rows
        first) and, when given, the verified ``tokens [rows, k + 1]`` to the host mirrors of the
        current slot, then bump the step counter. Capturable: device copies and device-to-host
        copies into pinned memory, in stream order."""
        rows = int(accepted.shape[0])
        dev, host = self._buffers(self._current_slot, rows)  # made in plan; a no-op here
        dev[:rows].copy_(accepted.to(torch.int32))
        host[:rows].copy_(dev[:rows], non_blocking=True)
        if tokens is not None:
            tdev, thost = self._tok_dev[self._current_slot], self._tok_host[self._current_slot]
            tdev[:rows].copy_(tokens.to(torch.int32))
            thost[:rows].copy_(tdev[:rows], non_blocking=True)
        self._counter_dev.add_(1)
        self._counter_host.copy_(self._counter_dev, non_blocking=True)

    def verdicts_for(self, request_ids: list[str]) -> list["Verdict"]:
        """The verdicts (accepted count and verified tokens) of these requests' last verify step,
        waiting for it. For the post-replay cut of the emitted tokens; whether the next step's plan
        has already published them (the plan thread may be ahead) makes no difference here."""
        self._settle()
        with self._lock:
            out = []
            for rid in request_ids:
                entry = self._settled.get(rid)
                if entry is None:
                    raise RuntimeError(f"no verdict for {rid}: its verify step was not committed before its outputs were read")
                out.append(entry[1])
        return out

    def accepted_for(self, request_ids: list[str]) -> list[int]:
        return [v.accepted for v in self.verdicts_for(request_ids)]

    def _settle(self) -> None:
        """Read every registered row whose step has been committed so far, after waiting for the
        counter to reach it. Values are copied out here, before a later step reuses the slot's
        buffers. Rows registered by another thread while this one waited are left for the next call:
        their step's counter may not have been reached."""
        with self._lock:
            latest = max((entry[0] for entry in self._pending.values()), default=0)
        if latest == 0:
            return
        self._wait(latest)
        with self._lock:
            for rid, (seq, slot, row) in list(self._pending.items()):
                if seq > latest:
                    continue
                accepted = int(self._acc_host[slot][row])
                self._settled[rid] = (seq, Verdict(accepted, self._tok_host[slot][row].tolist()))
                del self._pending[rid]
                self.rows_settled += 1
                self.accepted_total += accepted
                if self.rows_settled % self._log_every == 0:
                    logger.info("speculation: %s", self.stats())

    @property
    def mean_accepted(self) -> float:
        """Drafts accepted per verify row so far (0..k); tokens per step = this + 1."""
        return self.accepted_total / self.rows_settled if self.rows_settled else 0.0

    def stats(self) -> str:
        return (f"{self.rows_settled} verify rows, {self.mean_accepted:.2f} of {self.k} drafts accepted per row "
                f"({self.mean_accepted + 1:.2f} tokens per step)")

    def _wait(self, seq: int) -> None:
        if self.device.type != "cuda":
            return  # copies are synchronous off the GPU
        deadline = time.monotonic() + WAIT_TIMEOUT_S
        while int(self._counter_host[0]) < seq:
            if time.monotonic() > deadline:
                raise RuntimeError(f"speculative step {seq} never reported its acceptance counts "
                                   f"(host counter at {int(self._counter_host[0])})")
            time.sleep(0)
