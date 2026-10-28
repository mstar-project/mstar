"""A verify step's verdict, from the forward to the host.

A verify step runs ``k + 1`` rows per request, the last emitted token and ``k`` drafts, and
keeps the drafts the target agrees with. Its resources grow by the whole block at commit; the
verdict, known on the device only and mid-step, says how much of it to take back. This
resource carries it across:

- ``plan`` numbers the step and writes the number into the step's slot (its capture slot, or
  the eager one).
- ``verify``, in the forward and capturable, runs the sampler's verify, settles the node's
  in-place state from the verdict on the device (``settle_verify``), and copies each row's
  verdict, then the step's number, to the slot's pinned host mirror. A host that reads the
  number has the verdict.
- ``commit`` notes which mirror row holds each real request's verdict.
- ``settle``, from the runner before a request's next step admits anywhere, reads the rows'
  verdicts (waiting for the number) and hands the rewinds each one once. An engine-side
  proposer is handed each verdict as it is read (``DraftProposer.observe``).
- ``emitted_rows``, from the engine after the commit, cuts each row's tokens at a stop id and
  at ``max_tokens`` and says which rows stop.

The maps are keyed by the worker's request handle; ``ingest_request`` and ``remove_request``
clear them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource, RewindKind
from mstar.engine.resources.draft_verify.config import DraftPolicy, DraftVerifySpec, VerdictBatch, VerifyStep
from mstar.engine.resources.step import StepContext

if TYPE_CHECKING:
    from mstar.distributed.communication import JointGroups
    from mstar.engine.resources.draft_verify.proposer import DraftProposer
    from mstar.engine.resources.sampler.resource import SamplerResource

logger = logging.getLogger(__name__)

# a verdict that never lands within this long is a hang somewhere else; say so
WAIT_TIMEOUT_S = 120.0
# verified rows between acceptance log lines
_LOG_EVERY = 4096
# `_harvest`'s "every slot" default
_ANY_SLOT = object()


class VerdictTensors(NamedTuple):
    """A verify step's verdict on the device, in the forward."""
    tokens: torch.Tensor  # [n, k + 1]: the target's token at each row of the block
    accepted: torch.Tensor  # [n]: leading drafts equal to it; a row emits tokens[:accepted + 1]
    next_inputs: torch.Tensor  # [n, 1]: tokens[accepted], the next step's input


class EmittedRows(NamedTuple):
    """One verify step's emitted tokens on the host, columnar: row i is ``request_ids[i]``,
    whose tokens are the next ``counts[i]`` of ``tokens``."""
    request_ids: tuple[int, ...]
    tokens: np.ndarray
    counts: np.ndarray
    done: np.ndarray  # the row's loop stops after this step


@dataclass
class _Mirror:
    """One slot's verdict rows, on the device and in pinned host memory."""
    rows: int
    width: int
    # [rows, width] int64: accepted, the row's input token, then the k + 1 target tokens
    dev: torch.Tensor
    host: torch.Tensor
    host_np: np.ndarray
    # the number of the step planned on this slot, and of the last one whose rows landed
    seq_dev: torch.Tensor
    seq_host: torch.Tensor
    seq_np: np.ndarray
    # a capture slot's buffers are baked into its graphs, so they are never replaced
    captured: bool


class _Pending(NamedTuple):
    seq: int
    slot: int | None
    row: int
    label: str
    drafts: int


class _StepRecord(NamedTuple):
    seq: int
    slot: int | None
    rows: tuple[int, ...]  # the real rows' handles
    positions: tuple[int, ...]  # and their rows in the mirror


class DraftVerifyResource(Resource):
    rewind_kind = RewindKind.NONE
    # keeps no per-token state; a request's verdicts are cleared at ingest
    prefix_skip_safe = True

    def __init__(
        self, device: torch.device, joint_comm_group: JointGroups | None = None,
        sampler: str = "sampler",
    ):
        self._device = torch.device(device)
        # device-to-host copies land later only on an accelerator
        self._async = self._device.type != "cpu"
        self._pin = self._device.type == "cuda" and torch.cuda.is_available()
        self._group = (
            joint_comm_group
            if joint_comm_group is not None and joint_comm_group.world_size > 1 else None
        )
        self.sampler_key = sampler
        self._sampler: SamplerResource | None = None
        # the walk's engine-side proposer, bound by the engine at load
        self._proposer: DraftProposer | None = None
        # the node's resources that `verify` settles on the device (RewindKind.DEVICE)
        self._settlers: list[Resource] = []
        # capture slot (None: eager) -> its mirror
        self._mirrors: dict[int | None, _Mirror] = {}
        self._cg_rows = 0
        self._seq = 0
        # (seq, slot, drafts, label) of the step planned last, until its commit
        self._planned: tuple[int, int | None, int, str] | None = None
        # the step an eager forward last staged; a replay runs no Python
        self._verified = 0
        # The plan thread settles a step ahead while the gpu thread commits and emits, so
        # everything below is under the lock: handle -> where its last verdict is, until read;
        # -> (label, rejected) once read, until a rewind takes it; -> its last verdict's mirror
        # row once read; -> tokens emitted so far (None: none yet), which also marks the
        # request live.
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._settled: dict[int, tuple[str, int]] = {}
        self._rows: dict[int, np.ndarray] = {}
        self._emitted: dict[int, int | None] = {}
        # steps committed since the last `emitted_rows`
        self._steps: list[_StepRecord] = []
        self.rows_verified = 0
        self.drafts_accepted = 0
        # rows by how many drafts they accepted, which separates a weak first draft from a
        # chain that degrades
        self.accepted_hist = np.zeros(1, dtype=np.int64)
        self._logged = 0

    @classmethod
    def build(cls, spec: DraftVerifySpec, info: EngineResourceInfo) -> DraftVerifyResource:
        return cls(info.device, info.joint_comm_group, sampler=spec.sampler)

    def bind_sampler(self, sampler: SamplerResource) -> None:
        """The node's sampler, which `verify` runs; bound by the engine at load."""
        self._sampler = sampler

    def bind_proposer(self, proposer: DraftProposer) -> None:
        """The engine-side proposer `propose` asks, which sees every verdict; bound at load."""
        self._proposer = proposer

    def bind_settlers(self, resources: Sequence[Resource]) -> None:
        """The node's ``RewindKind.DEVICE`` resources, which `verify` settles; bound by the
        engine at load, once per walk that verifies."""
        for resource in resources:
            if not any(resource is bound for bound in self._settlers):
                self._settlers.append(resource)

    # ── request lifecycle ──

    def ingest_request(self, rid: int, overrides=None):
        del overrides
        with self._lock:
            self._pending.pop(rid, None)
            self._settled.pop(rid, None)
            self._rows.pop(rid, None)
            self._emitted[rid] = None

    def ingest_capture_row(self, rid: int):
        return

    def remove_request(self, rid: int):
        with self._lock:
            self._pending.pop(rid, None)
            self._settled.pop(rid, None)
            self._rows.pop(rid, None)
            self._emitted.pop(rid, None)

    def reset_request(self, rid: int, free: bool = False):
        del free
        self.remove_request(rid)

    def build_accelerator_graph_buffers(self, slots, max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_rows = max(self._cg_rows, max_bs)

    def cleanup(self):
        if self.rows_verified:
            logger.info("draft-verify, final: %s", self.stats())

    # ── step ──

    def plan(self, step: VerifyStep, ctx: StepContext) -> None:
        if step.sampler != self.sampler_key:
            raise ValueError(
                f"a verify step's sampler step is under {step.sampler!r}, but the node's "
                f"DraftVerifySpec names {self.sampler_key!r}; pass it to with_draft_verify"
            )
        label = "main"
        if self._settlers:
            labels = {segment.label for segment in step.segments or ()}
            if len(labels) > 1:
                # a settle addresses one label's rows, in the verdict's row order
                raise ValueError(
                    f"a verify step settles in-place state under one label; this one has {sorted(labels)}"
                )
            label = next(iter(labels), label)
        lease = ctx.slot_lease
        slot = None if lease is None else lease.slot
        # this step's forward writes the slot's mirror: read the rows still pending there first
        self._harvest(slot=slot)
        mirror = self._mirror(slot, len(ctx.padded_request_ids), step.drafts + 3, lease is not None)
        self._seq += 1
        # a kernel argument, not a host buffer, so no later plan can change it under the step
        mirror.seq_dev.fill_(self._seq)
        self._planned = (self._seq, slot, step.drafts, label)

    def commit(self, step: VerifyStep, ctx: StepContext) -> None:
        planned, self._planned = self._planned, None
        if planned is None:
            return
        seq, slot, drafts, _ = planned
        if ctx.slot_lease is None and self._verified != seq:
            raise RuntimeError("the forward of a verify step did not call DraftVerifyResource.verify")
        if ctx.capture:
            return
        labels = {segment.request_id: segment.label for segment in step.segments}
        rows: list[int] = []
        positions: list[int] = []
        with self._lock:
            for row, rid in enumerate(ctx.padded_request_ids):
                if ctx.is_padding_row(rid):
                    continue
                if rid in self._pending or rid in self._settled:
                    # the runner rewinds a request's last verdict before its next step admits
                    raise RuntimeError(
                        f"draft-verify: request {rid} verified again before its last verdict was rewound"
                    )
                self._pending[rid] = _Pending(seq, slot, row, labels[rid], drafts)
                rows.append(rid)
                positions.append(row)
            self._steps.append(_StepRecord(seq, slot, tuple(rows), tuple(positions)))

    # ── forward side; capturable ──

    @torch.compiler.disable
    def agree(self, drafts: torch.Tensor) -> torch.Tensor:
        """Rank 0's drafts on every rank of a lockstep instance; the drafts as they are on one.
        The drafts are the verify forward's inputs, and a drafter whose collectives don't sum
        bitwise alike on every rank can draft apart."""
        if self._group is None:
            return drafts
        return self._group.broadcast(drafts.contiguous(), src=0)

    @staticmethod
    def block_ids(inputs: torch.Tensor, drafts: torch.Tensor) -> torch.Tensor:
        """``[n, k + 1]``: each row's input token, then its drafts."""
        return torch.cat([inputs.reshape(-1, 1).to(drafts.dtype), drafts], 1)

    @torch.compiler.disable
    def propose(
        self, request_ids: list[int], inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The block of a verify step whose walk drafts with an engine-side proposer
        (``DraftPolicy.proposer``): ``[n, k + 1]``, each row's input token and the drafts the
        proposer staged for it, agreed across a lockstep instance; and their distributions,
        for ``verify``'s ``draft_probs`` (None: taken as certain). ``inputs`` holds a token per
        row the forward runs. Capturable."""
        if self._proposer is None:
            raise RuntimeError("DraftVerifyResource.propose on a walk whose policy names no proposer")
        if self._planned is None:
            raise RuntimeError("DraftVerifyResource.propose outside a planned verify step")
        rows = inputs.reshape(-1).shape[0]
        drafts, probs = self._proposer.propose(request_ids, rows)
        expected = self._planned[2]
        if drafts.shape != (rows, expected):
            raise ValueError(f"the proposer drafted {tuple(drafts.shape)} for {rows} rows of {expected} drafts")
        return self.block_ids(inputs, self.agree(drafts)), probs

    @torch.compiler.disable
    def draft_sample(
        self, request_ids: list[int], logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """For a drafter that samples: one draft per row of ``logits`` ``[n, V]``, drawn as the
        request samples (temperature, top-k, top-p), and the distribution it came from, which
        ``verify`` takes stacked as ``draft_probs``. A node whose verify is greedy drafts the
        argmax and returns no distribution. Capturable; call it in a verify step's forward."""
        if self._planned is None:
            raise RuntimeError("DraftVerifyResource.draft_sample outside a planned verify step")
        if self._sampler is None:
            raise RuntimeError("DraftVerifyResource has no sampler bound; the engine binds it at load")
        return self._sampler.draft_sample(request_ids, logits)

    @torch.compiler.disable
    def verify(
        self, request_ids: list[int], block: torch.Tensor, logits: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
    ) -> VerdictTensors:
        """Verify ``block`` (``block_ids``) against the target's ``logits`` over its rows,
        ``[n * (k + 1), V]``, settle the node's in-place state by the verdict, and stage the
        verdict for the host. ``request_ids`` are the rows the forward ran, padding rows
        included. ``draft_probs`` ``[n, k, V]``: the distributions a sampling drafter drew the
        drafts from (``draft_sample``); None takes each draft as certain. Call it once the
        target has run over the whole block: a recurrent layer's settle reads what it ran."""
        if self._planned is None:
            raise RuntimeError("DraftVerifyResource.verify outside a planned verify step")
        if self._sampler is None:
            raise RuntimeError("DraftVerifyResource has no sampler bound; the engine binds it at load")
        seq, slot, drafts, label = self._planned
        if block.shape[1] != drafts + 1:
            raise ValueError(f"a verify block of {block.shape[1]} tokens where the step declared {drafts} drafts")
        tokens, accepted = self._sampler.verify(request_ids, block, logits, draft_probs)
        n = tokens.shape[0]
        mirror = self._mirrors[slot]
        mirror.dev[:n].copy_(torch.cat([accepted[:, None], block[:, :1], tokens], 1))
        mirror.host[:n].copy_(mirror.dev[:n], non_blocking=True)
        # after the rows, on the same stream: the number lands last
        mirror.seq_host.copy_(mirror.seq_dev, non_blocking=True)
        for resource in self._settlers:
            resource.settle_verify(accepted, label)
        self._verified = seq
        return VerdictTensors(tokens, accepted, tokens.gather(1, accepted[:, None]))

    # ── host side ──

    def settle(self, request_ids: Sequence[int]) -> VerdictBatch | None:
        """The rejected rows of these requests' last verify steps that no rewind has taken yet,
        each handed out once; None when there are none. Waits for a verdict that hasn't landed.
        A fact about an earlier step, so a discarded pre-plan doesn't give it back."""
        pending, settled = self._pending, self._settled
        if not pending and not settled:
            return None
        with self._lock:
            want = [rid for rid in request_ids if rid in pending or rid in settled]
        if not want:
            return None
        self._harvest(rids=want)
        out = VerdictBatch([], [], [])
        with self._lock:
            for rid in want:
                verdict = settled.pop(rid, None)
                if verdict is not None and verdict[1]:
                    out.request_ids.append(rid)
                    out.labels.append(verdict[0])
                    out.rejected.append(verdict[1])
        return out if out.request_ids else None

    def emitted_rows(
        self, request_ids: Sequence[int], max_tokens: Sequence[int],
        ignore_eos: Sequence[bool], policy: DraftPolicy,
    ) -> EmittedRows:
        """The tokens each row of the verify step just committed emits, and whether its loop
        stops. A row emits its accepted drafts and the target's next token, up to and
        including the first stop id (nothing when its input is one: the reply ended a step
        earlier), and no more than ``max_tokens`` counting what it emitted before. Once per
        verify step, after the commit."""
        steps, self._steps = self._steps, []
        rids = tuple(request_ids)
        n = len(rids)
        last = steps[-1] if steps else None
        if last is not None and last.rows == rids and last.positions == tuple(range(n)):
            self._wait(last.slot, last.seq)
            table = self._mirrors[last.slot].host_np[:n].copy()
        else:
            # a per-request exec ran a step per row, each over the last one's mirror rows
            ran = {rid for record in steps for rid in record.rows}
            missing = [rid for rid in rids if rid not in ran]
            if missing:
                raise RuntimeError(f"draft-verify: no verify step of requests {missing} to emit")
            self._harvest(rids=rids)
            with self._lock:
                table = np.stack([self._rows[rid] for rid in rids])
        width = table.shape[1] - 2
        accepted, inputs, tokens = table[:, 0], table[:, 1], table[:, 2:]
        with self._lock:
            before = np.fromiter(
                (policy.emitted_before if (e := self._emitted.get(rid)) is None else e for rid in rids),
                np.int64, n,
            )
        limit = np.asarray(max_tokens, dtype=np.int64)
        counts = np.minimum(accepted + 1, np.maximum(limit - before, 0))
        stopped = np.zeros(n, dtype=bool)
        if policy.stop_ids:
            keep = ~np.asarray(ignore_eos, dtype=bool)
            stop_ids = np.fromiter(policy.stop_ids, np.int64, len(policy.stop_ids))
            ended = np.isin(inputs, stop_ids) & keep
            hits = np.isin(tokens, stop_ids) & keep[:, None] & (np.arange(width) < counts[:, None])
            hit = hits.any(1)
            counts = np.where(hit, hits.argmax(1) + 1, counts)
            counts[ended] = 0
            stopped = hit | ended
        total = before + counts
        with self._lock:
            emitted = self._emitted
            for rid, value in zip(rids, total.tolist(), strict=True):
                # a request removed while its step ran stays removed
                if rid in emitted:
                    emitted[rid] = value
        self.rows_verified += n
        self.drafts_accepted += int(accepted.sum())
        counts_by_accepted = np.bincount(accepted, minlength=width)
        grow = counts_by_accepted.shape[0] - self.accepted_hist.shape[0]
        if grow > 0:
            self.accepted_hist = np.pad(self.accepted_hist, (0, grow))
        self.accepted_hist[: counts_by_accepted.shape[0]] += counts_by_accepted
        if self.rows_verified - self._logged >= _LOG_EVERY:
            self._logged = self.rows_verified
            logger.info("draft-verify: %s", self.stats())
        return EmittedRows(
            rids, tokens[np.arange(width) < counts[:, None]], counts, stopped | (total >= limit),
        )

    def stats(self) -> str:
        """Rows verified, drafts accepted per row, and the rate at which each draft is
        accepted given the ones before it were."""
        mean = self.drafts_accepted / self.rows_verified if self.rows_verified else 0.0
        reached = self.accepted_hist[::-1].cumsum()[::-1]
        rates = " ".join(
            f"{reached[i] / reached[i - 1]:.2f}" if reached[i - 1] else "-" for i in range(1, len(reached))
        )
        return (
            f"{self.rows_verified} verify rows, {mean:.2f} drafts accepted per row "
            f"({mean + 1:.2f} tokens per step); accepted by position {rates or '-'}"
        )

    # ── mirrors ──

    def _mirror(self, slot: int | None, rows: int, width: int, captured: bool) -> _Mirror:
        mirror = self._mirrors.get(slot)
        if mirror is not None and mirror.rows >= rows and mirror.width == width:
            return mirror
        if mirror is not None and mirror.captured:
            raise RuntimeError(
                f"draft-verify: slot {slot}'s verdict mirror holds {mirror.rows} rows of {mirror.width}, "
                f"a step needs {rows} of {width}, and graphs captured against it read it in place"
            )
        if captured:
            rows = max(rows, self._cg_rows)
        elif mirror is not None:
            rows = max(rows, 2 * mirror.rows)
        dev = torch.zeros(rows, width, dtype=torch.long, device=self._device)
        host = torch.zeros(rows, width, dtype=torch.long, pin_memory=self._pin)
        seq_dev = torch.zeros(1, dtype=torch.long, device=self._device)
        seq_host = torch.zeros(1, dtype=torch.long, pin_memory=self._pin)
        mirror = self._mirrors[slot] = _Mirror(
            rows, width, dev, host, host.numpy(), seq_dev, seq_host, seq_host.numpy(), captured,
        )
        return mirror

    def _harvest(self, rids: Sequence[int] | None = None, slot=_ANY_SLOT) -> None:
        """Read pending verdicts out of the mirrors (these requests', or every one in
        ``slot``), waiting for their steps to stage them."""
        with self._lock:
            if rids is None:
                entries = [(rid, p) for rid, p in self._pending.items() if slot is _ANY_SLOT or p.slot == slot]
            else:
                entries = [(rid, self._pending[rid]) for rid in rids if rid in self._pending]
        if not entries:
            return
        newest: dict[int | None, int] = {}
        for _, p in entries:
            newest[p.slot] = max(newest.get(p.slot, 0), p.seq)
        for s, seq in newest.items():
            self._wait(s, seq)
        with self._lock:
            proposer = self._proposer
            for rid, p in entries:
                if self._pending.get(rid) is not p:
                    continue  # read by another thread, or removed
                del self._pending[rid]
                row = self._rows[rid] = self._mirrors[p.slot].host_np[p.row].copy()
                self._settled[rid] = (p.label, p.drafts - int(row[0]))
                if proposer is not None:
                    proposer.observe(rid, row)

    def harvest(self, request_ids: Sequence[int]) -> None:
        """Read these requests' pending verdicts, waiting for their steps to stage them: a
        reader that needs them before the runner's rewinds take them (a proposer's plan)."""
        if self._pending:
            self._harvest(rids=request_ids)

    def _wait(self, slot: int | None, seq: int) -> None:
        """Until the verdict of step ``seq`` is in ``slot``'s mirror. A host spin, so the
        thread waiting is the plan thread or the gpu thread, never the main loop."""
        flag = self._mirrors[slot].seq_np
        if self._async and flag[0] < seq:
            deadline = time.monotonic() + WAIT_TIMEOUT_S
            while flag[0] < seq:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"draft-verify: the verdict of step {seq} never landed (slot {slot})")
                time.sleep(0)
        if flag[0] != seq:
            # a slot's next step reads what is pending there first, so this is a bug
            what = f"overwritten by step {int(flag[0])}" if flag[0] > seq else "never staged"
            raise RuntimeError(f"draft-verify: the verdict of step {seq} in slot {slot} was {what}")
