"""The engine's state for one node that drafts and verifies: its mode, its
ledger of unapplied results, and the token cut its postprocess runs."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.draft.config import DraftMode
from mstar.engine.resources.draft.drafter import VERIFY_ACCEPTED, VERIFY_STOPPED, VERIFY_TOKENS, VerifyNode
from mstar.engine.resources.draft.ledger import VerifyLedger
from mstar.engine.resources.step import ResourceStep

logger = logging.getLogger(__name__)

# seconds between acceptance log lines
_LOG_EVERY_S = 60.0


class DrafterShare(NamedTuple):
    """What the engine adds to a verifying node's steps for its drafter (`drafter_step`), bound at load from
    the node's `VerifyNode`, so no model declares it."""
    verify_walk: str
    prompt_walks: frozenset[str]
    # the drafter's own resources' steps (`VerifyNode.prefill_steps`, `VerifyNode.verify_steps`)
    prompt_steps: Mapping[str, ResourceStep]
    verify_steps: Mapping[str, ResourceStep]


def drafter_step(step, graph_walk: str, share: DrafterShare | None):
    """``step`` with its drafter's share, as a new step: on a prompt or verify step, the drafter's resource
    steps. The engine derives them from the drafter's declaration, so a captured graph and each step it
    replays agree by construction."""
    if share is None or step is None:
        return step
    if graph_walk == share.verify_walk:
        own = share.verify_steps
    elif graph_walk in share.prompt_walks:
        own = share.prompt_steps
    else:
        return step
    steps = dict(step.steps)
    if clash := sorted(steps.keys() & own.keys()):
        raise RuntimeError(f"a {graph_walk!r} step declares {clash}, its drafter's steps, which the engine adds")
    # fresh each step, as a declaration's are: a resource may stamp its own step (KV's commit count)
    steps.update((key, replace(resource_step)) for key, resource_step in own.items())
    return replace(step, steps=steps)


class TokenCut:
    """A verify step's emitted run per row, as columns over the step's rows."""

    __slots__ = ("request_ids", "counts", "stopped")

    def __init__(self, request_ids: Sequence[int], counts: np.ndarray, stopped: np.ndarray):
        self.request_ids = request_ids
        self.counts = counts
        self.stopped = stopped


class NodeDraft:
    """One verifying node on this engine. The token cut keeps each request's
    emitted count, handle-keyed and purged on removal, since a verify step's
    budget is in tokens and the decode loop counts iterations."""

    def __init__(
        self, decl: VerifyNode, mode: DraftMode, reason: str, sampler_key: str,
        token_stop, take_back_keys: Sequence[str],
    ):
        self.decl = decl
        self.node = decl.node
        self.walk = decl.walk
        self.mode = mode
        self.reason = reason
        self.sampler_key = sampler_key
        self.token_stop = token_stop
        self.take_back_keys = tuple(take_back_keys)
        self.ledger = VerifyLedger(decl.node, synchronous=mode == DraftMode.SYNCHRONOUS)
        self._stop_ids = np.array(sorted(token_stop.stop_ids), dtype=np.int64)
        # the reply's first token comes from a prefill walk and counts toward the budget
        self._prefill_tokens = 1 if token_stop.prefill_walks else 0
        # rid -> tokens its verify steps emitted so far
        self._emitted: dict[int, int] = {}
        # (rid, label) -> position counter minus KV length, under MSTAR_KV_DEBUG_ASSERTS (`check_offsets`)
        self._offsets: dict[tuple[int, str], int] = {}
        # (rid, KV label) -> the position label of the same stream, from the steps' segments
        self._rope_labels: dict[tuple[int, str], str] = {}
        # acceptance counters for the periodic log line and the low-acceptance warning
        self.drafted = 0
        self.accepted = 0
        self.verify_rows = 0
        self.accepted_hist = np.zeros(decl.config.num_draft_tokens + 1, dtype=np.int64)
        self._warned_low = False
        self._logged_at = time.monotonic()

    @property
    def asynchronous(self) -> bool:
        return self.mode == DraftMode.ASYNCHRONOUS

    def committed(self, graph_walk: str, rids: Sequence[int], step, resources: Mapping) -> int:
        """Right after the runner's commit of any step of the node, on the GPU
        thread: number it, and for a verify step what each corrected resource
        counted."""
        if graph_walk != self.walk or not rids or not self.take_back_keys:
            # a node no resource of which takes back records no rows: the step only counts
            return self.ledger.committed(None)
        labels: dict[str, str] = {}
        counted: list[int] | None = None
        for key in self.take_back_keys:
            resource_step = step.get(key)
            segments = (resource_step.segments if resource_step is not None else None) or step.segments
            seg = segments[0]
            labels[key] = seg.label
            spans = resources[key].committed_spans(rids, seg.label, seg.span)
            if counted is None:
                counted = list(spans)
                continue
            for i, (have, span) in enumerate(zip(counted, spans, strict=True)):
                if have and span and have != span:
                    raise RuntimeError(f"verify step on {self.decl.node}: {key!r} counted {span} for {rids[i]}, "
                                       f"another resource {have}")
                # a span one resource didn't count (KV's, offloaded mid-step) is taken back by none,
                # so positions stay with the KV length
                if not span:
                    counted[i] = 0
        return self.ledger.committed(rids, labels, counted)

    def refuse_step(self, step) -> None:
        """A verify step's resources take back a length: a fork, a retention
        policy or a sink would copy or move the reserved one. Forks are declared
        per step, so this is checked per step, not only at startup. And every
        row of a stream that takes back counts the block, ``k + 1``: whatever
        path prepared the rows (a model's own, the engine's uniform rows, a
        pre-plan's template), a shorter span would reserve too few slots."""
        for key, resource_step in step.steps.items():
            for name in ("pre_forks", "post_forks", "retention", "sink", "advance"):
                if getattr(resource_step, name, None):
                    raise RuntimeError(f"a verify step's {key!r} declares {name}, which a take-back can't follow")
        block, seen = self.decl.config.block, []
        for key in self.take_back_keys:
            resource_step = step.get(key)
            segments = (resource_step.segments if resource_step is not None else None) or step.segments or ()
            if any(segments is other for other in seen):
                continue  # the step's default segments, already checked
            seen.append(segments)
            if any(segment.span != block for segment in segments):
                spans = sorted({segment.span for segment in segments})
                raise RuntimeError(f"a verify step's {key!r} rows span {spans}; each runs k + 1 = {block}")

    def check_offsets(self, resources: Mapping, rids, step=None) -> None:
        """Debug only (``MSTAR_KV_DEBUG_ASSERTS``): each of ``rids``' position
        counters minus its KV length stays what it was at its first verify step.
        Commit adds the span to both and the take-back the same count to both, so
        the resident offset cell is a fixed per-request offset. ``step`` (a
        verify step of these rows) pairs each KV label with its position label.
        An offloaded stream is skipped: the offload resets its KV length, not the
        counter, and a reload rewrites it."""
        from mstar.engine.resources.kv.manager import KVManager
        from mstar.engine.resources.position.manager import RopeManager

        kv_key, kv = next(((k, r) for k, r in resources.items() if isinstance(r, KVManager)), (None, None))
        rope_key, rope = next(((k, r) for k, r in resources.items() if isinstance(r, RopeManager)), (None, None))
        if kv is None or rope is None:
            return
        if step is not None:
            self._pair_labels(step, kv_key, rope_key)
        offsets, labels = self._offsets, self._rope_labels
        for rid in rids:
            for label, stream in (kv._streams.get(rid) or {}).items():
                rope_label = labels.get((rid, label))
                if rope_label is None or stream.offloaded or not stream.page_indices:
                    continue
                offset = rope.position(rid, rope_label) - stream.stored_len
                held = offsets.setdefault((rid, label), offset)
                assert held == offset, f"{self.node}: position offset of {rid}/{label} moved from {held} to {offset}"

    def _pair_labels(self, step, kv_key: str, rope_key: str) -> None:
        """Each row's KV segments and position segments, in order, are one stream each."""
        def by_row(key):
            resource_step = step.get(key)
            rows: dict = {}
            for seg in (resource_step.segments if resource_step is not None else None) or step.segments or ():
                rows.setdefault(seg.request_id, []).append(seg.label)
            return rows

        rope_rows = by_row(rope_key)
        for rid, kv_labels in by_row(kv_key).items():
            for kv_label, rope_label in zip(kv_labels, rope_rows.get(rid, ()), strict=False):
                self._rope_labels[(rid, kv_label)] = rope_label

    def remove_request(self, rid: int) -> None:
        self.ledger.remove_request(rid)
        self._emitted.pop(rid, None)
        for held in (self._offsets, self._rope_labels):
            for key in [key for key in held if key[0] == rid]:
                del held[key]

    def cut(
        self, seq: int, host_rows, request_infos: Mapping[int, object],
    ) -> TokenCut:
        """Record step ``seq``'s result and cut each row's run of ``accepted + 1``
        tokens at its first stop id (unless ``ignore_eos``) and at the budget
        left (``max_tokens``). Adds the stopped column to ``host_rows``, which
        the stop check reads. The take-back reads ``accepted``, never the cut."""
        request_ids = host_rows.request_ids
        tokens = host_rows.buffers[VERIFY_TOKENS].numpy()
        accepted = host_rows.buffers[VERIFY_ACCEPTED].reshape(-1).numpy()
        self.ledger.record(seq, request_ids, accepted)
        n = len(request_ids)
        tokens, accepted = tokens[:n], accepted[:n].astype(np.int64)
        counts = accepted + 1
        sampler = self.token_stop.sampler
        live = [i for i, rid in enumerate(request_ids) if rid in request_infos]
        budget = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        ignore = np.zeros(n, dtype=bool)
        emitted = self._emitted
        for i in live:
            rid = request_ids[i]
            info = request_infos[rid]
            budget[i] = info.max_tokens - self._prefill_tokens - emitted.get(rid, 0)
            ignore[i] = info.resource_configs[sampler].ignore_eos
        np.minimum(counts, np.maximum(budget, 1), out=counts)
        stopped = counts >= budget
        if self._stop_ids.size:
            within = np.arange(tokens.shape[1])[None, :] < counts[:, None]
            hit = np.isin(tokens, self._stop_ids) & within & ~ignore[:, None]
            rows = hit.any(axis=1)
            if rows.any():
                counts[rows] = hit[rows].argmax(axis=1) + 1
                stopped |= rows
        for i in live:
            rid = request_ids[i]
            emitted[rid] = emitted.get(rid, 0) + int(counts[i])
        host_rows.buffers[VERIFY_STOPPED] = torch.from_numpy(stopped)
        self._count(accepted[live] if len(live) < n else accepted)
        return TokenCut(request_ids, counts, stopped)

    def _count(self, accepted: np.ndarray) -> None:
        k = self.decl.config.num_draft_tokens
        rows = accepted.size
        self.verify_rows += rows
        self.drafted += rows * k
        self.accepted += int(accepted.sum())
        self.accepted_hist += np.bincount(accepted, minlength=k + 1)[: k + 1]
        now = time.monotonic()
        if now - self._logged_at >= _LOG_EVERY_S:
            self._logged_at = now
            logger.info(
                "speculative decoding on %s: tau %.2f over %d verify rows, %.1f%% of drafts accepted, "
                "accepted per step %s", self.node, self.acceptance_length(), self.verify_rows,
                100 * self.accepted / max(self.drafted, 1), self.accepted_hist.tolist(),
            )
        if not self._warned_low and self.drafted >= 4096 and self.accepted < 0.02 * self.drafted:
            self._warned_low = True
            logger.warning(
                "speculative decoding on %s accepts %.1f%% of %d drafts: a mismatched drafter "
                "gives correct output and no speedup", self.node, 100 * self.accepted / self.drafted, self.drafted,
            )

    def acceptance_length(self) -> float:
        """τ, the mean of accepted + 1 per verify row."""
        return (self.accepted + self.verify_rows) / self.verify_rows if self.verify_rows else 0.0
