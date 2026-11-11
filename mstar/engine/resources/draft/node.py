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
from mstar.engine.resources.step import Lag, ResourceStep

logger = logging.getLogger(__name__)

# seconds between acceptance log lines
_LOG_EVERY_S = 60.0


class DraftLag(NamedTuple):
    """Where a lagged drafter's lag goes on its node's verify steps (`DrafterNeeds.lag`): the KV pool its
    layer set ``layer_set`` is in, the positions over that pool, and the drafter's attention binding."""
    walk: str
    lag: Lag
    kv_key: str
    layer_set: str
    pos_key: str
    attn_key: str


class DrafterShare(NamedTuple):
    """What the engine binds into a verifying node once at load
    (`NodeSubmodule.bind_draft`), from its `VerifyNode` and the draft config, so
    no model declares it: the walk it verifies on, the sampler that verifies, k,
    and what it adds to the node's steps for its drafter (`drafter_step`): the
    drafter's resource steps and its carry's loop-back writes. The same on every rank."""
    verify_walk: str
    sampler_key: str
    num_draft_tokens: int
    prompt_walks: frozenset[str]
    # the sampler loop-back state the drafter's state is stored through: its carry on both walks
    # (`carry_writes`), and on a verify step a lagged drafter's keep flag too (`verify_writes`)
    prompt_writes: frozenset[str]
    verify_writes: frozenset[str]
    # the drafter's own resources' steps (`VerifyNode.prefill_steps`, `VerifyNode.verify_steps`)
    prompt_steps: Mapping[str, ResourceStep]
    verify_steps: Mapping[str, ResourceStep]
    # the node's resources that correct a verify step's state in its graph, in dependency order
    correcting_keys: tuple[str, ...] = ()
    lag: DraftLag | None = None

    @property
    def block(self) -> int:
        """Tokens a verify step runs a row: its last emitted token and the drafts."""
        return self.num_draft_tokens + 1


def drafter_step(step, graph_walk: str, share: DrafterShare | None):
    """``step`` with its drafter's share, as a new step: on a prompt or verify step, the drafter's resource
    steps and its loop-back writes on the node's sampler step, and on a verify step a lagged drafter's lag in
    its KV, position and attention steps. The engine derives all of it from the drafter's declaration, so a
    captured graph and each step it replays agree by construction."""
    if share is None or step is None:
        return step
    for key, resource_step in step.steps.items():
        if getattr(resource_step, "lag", None):
            raise RuntimeError(f"a {graph_walk!r} step's {key!r} declares a lag, which only the engine sets "
                               "(from the drafter's DrafterNeeds.lag)")
    if graph_walk == share.verify_walk:
        writes, own = share.verify_writes, share.verify_steps
    elif graph_walk in share.prompt_walks:
        writes, own = share.prompt_writes, share.prompt_steps
    else:
        return step
    steps = dict(step.steps)
    if clash := sorted(steps.keys() & own.keys()):
        raise RuntimeError(f"a {graph_walk!r} step declares {clash}, its drafter's steps, which the engine adds")
    if writes:
        sampler_step = steps.get(share.sampler_key)
        if sampler_step is None:
            raise RuntimeError(f"a {graph_walk!r} step declares no {share.sampler_key!r} step, which its "
                               f"drafter's loop-back state {sorted(writes)} is stored through")
        steps[share.sampler_key] = replace(sampler_step, loopback_writes=sampler_step.loopback_writes | writes)
    # fresh each step, as a declaration's are: a resource may stamp its own step (KV's commit count)
    steps.update((key, replace(resource_step)) for key, resource_step in own.items())
    lag = share.lag
    if lag is not None and graph_walk == lag.walk:
        for key, value in ((lag.kv_key, {lag.layer_set: lag.lag}), (lag.pos_key, lag.lag), (lag.attn_key, lag.lag)):
            resource_step = steps.get(key)
            if resource_step is None:
                raise RuntimeError(f"a {graph_walk!r} step declares no {key!r} step, which the drafter's lag rides on")
            steps[key] = replace(resource_step, lag=value)
    return replace(step, steps=steps)


def check_lag(step, graph_walk: str, lag: DraftLag | None) -> None:
    """The lag a step carries is the drafter's declared one, on exactly its KV, position and attention steps
    (`drafter_step`), and none on any other step or walk: at capture, and at each verify step."""
    on = lag is not None and graph_walk == lag.walk
    for key, resource_step in step.steps.items():
        got = getattr(resource_step, "lag", None) or None
        want = None
        if on and key == lag.kv_key:
            want = {lag.layer_set: lag.lag}
        elif on and key in (lag.pos_key, lag.attn_key):
            want = lag.lag
        if got != want:
            raise RuntimeError(f"a {graph_walk!r} step's {key!r} lags {got}; the drafter declares {want} there")


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
        token_stop, take_back_keys: Sequence[str], lag: DraftLag | None = None,
    ):
        self.decl = decl
        # where a lagged drafter's lag goes (`draft_lag`), checked on every verify step (`check_lag`)
        self.lag = lag
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
        # the log line's window opens at the first verify step, not at load, which can take longer
        # than a window; each line reports the window's own counts (`_window_start`)
        self._logged_at: float | None = None
        self._window_start = (0, 0, 0, self.accepted_hist.copy())

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
            spans = resources[key].committed_spans(resource_step, rids, seg.label, seg.span)
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
        check_lag(step, self.walk, self.lag)
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
        now = time.monotonic()
        if self._logged_at is None:
            self._logged_at = now
        rows = accepted.size
        self.verify_rows += rows
        self.drafted += rows * k
        self.accepted += int(accepted.sum())
        self.accepted_hist += np.bincount(accepted, minlength=k + 1)[: k + 1]
        if now - self._logged_at >= _LOG_EVERY_S:
            rows0, drafted0, accepted0, hist0 = self._window_start
            rows, drafted, kept = self.verify_rows - rows0, self.drafted - drafted0, self.accepted - accepted0
            logger.info(
                "speculative decoding on %s, the last %.0f s: tau %.2f over %d verify rows, %.1f%% of drafts "
                "accepted, accepted per step %s", self.node, now - self._logged_at, (kept + rows) / max(rows, 1),
                rows, 100 * kept / max(drafted, 1), (self.accepted_hist - hist0).tolist(),
            )
            self._logged_at = now
            self._window_start = (self.verify_rows, self.drafted, self.accepted, self.accepted_hist.copy())
        if not self._warned_low and self.drafted >= 4096 and self.accepted < 0.02 * self.drafted:
            self._warned_low = True
            logger.warning(
                "speculative decoding on %s accepts %.1f%% of %d drafts: a mismatched drafter "
                "gives correct output and no speedup", self.node, 100 * self.accepted / self.drafted, self.drafted,
            )

    def acceptance_length(self) -> float:
        """τ, the mean of accepted + 1 per verify row."""
        return (self.accepted + self.verify_rows) / self.verify_rows if self.verify_rows else 0.0
