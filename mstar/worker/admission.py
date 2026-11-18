"""Whether a row or a step fits the engine's resources now, and what happens
when admit refuses one.

Doesn't import the scheduler. A refused step's caller passes in what only it
knows: the requests in arrival order, who is last in line, and whether other
rows could refill the step (the last two as callables into the scheduler).
Reads the request bookkeeping only to unpark rows, starts the rebuild of a row
its engine dropped (``Rebuilds.start``), and fails a request a resource refused
for good (``RequestLifecycle.fail``).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Iterable
from enum import Enum

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AllocationFailed, Residency
from mstar.model.submodule_base import chunk_loop_name
from mstar.worker.rebuild import Prefilling
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.services import WorkerLayout, WorkerServices

logger = logging.getLogger(__name__)

# Seconds between logs of one (node, walk)'s refusals with nothing to evict
_HOLD_LOG_INTERVAL = 5.0


class EvictionPolicy(Enum):
    """Strategy for choosing which request to offload to CPU on OOM."""
    LAST_IN_LINE = "last_in_line"  # the row a step of the refused key would reach last
    # TODO: PRIORITY — ask a named resource how much it wants each candidate
    # gone (``Engine.offload_priority``), for cases where the line is the wrong
    # question. Needs eviction-policy metadata naming which resource to consult.


class Admission:
    """Parked rows, holds and batch limits. A terminal refusal fails its
    request at once, which every scan then leaves out."""

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05

    def __init__(self, services: WorkerServices, layout: WorkerLayout, lifecycle: RequestLifecycle):
        self._lifecycle = lifecycle
        self._engine_manager = services.engine_manager
        self._request_state = services.request_state
        self._runtime = services.graph_runtime
        self._rebuilds = services.rebuilds
        self._combined_walk_of = layout.combined_walk_of
        self._follows = layout.followed_nodes
        # node -> the walks whose prefill runs as a chunk Loop there
        self._chunk_loop_walks = layout.chunk_loops.walks_by_node
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[int, float] = {}
        # node -> rids waiting for capacity there, which readiness answers
        # without asking the engine until `Engine.admit_waiting` admits them
        self.parked: dict[str, set[int]] = {}
        # (node, graph walk) -> most rows a batch may take while its admits are
        # refused with nothing to evict; see `limit_batch`
        self.batch_limits: dict[tuple[str, str], int] = {}
        # (node, graph walk) -> most rows the KV pool can hold at once even
        # with nothing else resident; see `cap_batch`
        self.batch_ceilings: dict[tuple[str, str], int] = {}
        # (node, graph walk) -> the resource whose refusal set its cap, if named
        self.ceiling_resources: dict[tuple[str, str], str | None] = {}
        self.eviction_policy = EvictionPolicy.LAST_IN_LINE
        # (node, walk) -> when its last hold was logged, and the holds since
        self._hold_logged: dict[tuple[str, str], tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def check_ready(
        self, node_name: str, rid: int, fwd_info: CurrentForwardPassInfo,
        allow_reload: bool = True,
    ) -> bool:
        """Engine-level readiness. Retryable not-ready (an in-flight KV read, a
        reload that doesn't fit) just comes back False; a ``terminal`` reason
        never will, so it fails the request instead of being rescanned
        forever. A rid waiting for capacity is parked on its node until
        `let_rows_back` finds it admitted, and a dropped rid waits until its
        rebuild lands."""
        if rid in self.parked.get(node_name, ()) or self._rebuilds.rebuilding(rid, node_name):
            return False
        engine = self._engine_manager.get_engine(node_name)
        outcome = engine.check_ready(node_name, rid, fwd_info, allow_reload=allow_reload)
        if outcome.queued:
            self.parked.setdefault(node_name, set()).add(rid)
            return False
        if outcome.rebuild:
            # A follower starts a rebuild only at its first row in rank 0's step, in `Lockstep._start_follow_rebuilds`.
            # The request stays dropped here from the rebuild's last chunk until this rank replays the rebuild landing
            # on rank 0, and starting a rebuild then would run it a second time on this rank alone.
            if node_name not in self._follows:
                self.start_rebuild(node_name, rid, fwd_info.graph_walk)
            return False
        if outcome.reason is not None and outcome.reason.terminal:
            logger.error(
                "Request %s cannot be served on node %s by resource %s: %s",
                self._runtime.get_rid_string(rid), node_name, outcome.failed_resource, outcome.reason.message,
            )
            self._lifecycle.fail({
                rid: f"resource {outcome.failed_resource} rejected the request: {outcome.reason.message}",
            }, status=getattr(outcome.reason, "status", 500))
            return False
        return outcome.ok and outcome.ready

    def start_rebuild(self, node_name: str, rid: int, graph_walk: str) -> None:
        engine = self._engine_manager.get_engine(node_name)
        rebuild = self._rebuilds.start(rid, node_name, self.prefilling(node_name, rid, graph_walk))
        if rebuild is not None:
            engine.prepare_rebuild(node_name, rid, rebuild)
        elif not self._rebuilds.cursor(rid, node_name).resumes_prefill:
            # the rows before its prompt's step in progress prepare their inputs where that step's input is kept
            engine.set_aside_chunk_inputs(rid, node_name)

    def prefilling(self, node_name: str, rid: int, graph_walk: str) -> Prefilling | None:
        """Where ``rid``'s prompt stands while it is mid-prefill in ``graph_walk`` on ``node_name``, read off its
        chunk Loop as `MicroScheduler._chunk_loop_progress` reads it, else None."""
        if graph_walk not in self._chunk_loop_walks.get(node_name, ()):
            return None
        (done,), (total,) = self._runtime.loop_progress(
            node_name, [rid], [graph_walk], [chunk_loop_name(node_name, graph_walk)],
        )
        return Prefilling(graph_walk, done, total) if 0 < done < total else None

    def let_rows_back(self) -> None:
        """Before a scan: drop the holds that have run out, and unpark the
        rows whose capacity has freed."""
        now = time.monotonic()
        self.held_until = {rid: t for rid, t in self.held_until.items() if t > now}
        self._unpark()

    def _unpark(self) -> None:
        """Let readiness ask the engine again about each parked request it has since admitted."""
        if not any(self.parked.values()):
            return
        request_state = self._request_state

        def published_of(rid: int, node_name: str):
            fwd_info = request_state.get_fwd_info(rid, request_state.get_partition_for_node(node_name))
            return fwd_info.resource_publish_info

        for engine in {*map(self._engine_manager.get_engine, self.parked)}:
            for node_name, rid in engine.admit_waiting(published_of):
                # a request asked again on each node of its pool may have parked on only one of them
                self.parked.get(node_name, set()).discard(rid)

    def clear(self, rid: int) -> None:
        """Forget ``rid``; called from its teardown."""
        self.held_until.pop(rid, None)
        for rids in self.parked.values():
            rids.discard(rid)

    # ------------------------------------------------------------------
    # Batch sizes
    # ------------------------------------------------------------------

    def hold(self, request_ids: Iterable[int]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

    def has_caps(self) -> bool:
        return bool(self.batch_limits or self.batch_ceilings)

    def cap(self, node_name: str, graph_walk: str, cap: int | None) -> int | None:
        """``cap`` lowered by any limit or cap from a refused admit (see
        `limit_batch` and `cap_batch`)."""
        key = (node_name, graph_walk)
        for limit in (self.batch_limits.get(key), self.batch_ceilings.get(key)):
            if limit is not None:
                cap = limit if cap is None else min(cap, limit)
        return cap

    def limit_batch(self, node_name: str, graph_walk: str, rows: int) -> None:
        """Keep ``(node_name, graph_walk)``'s batches to ``rows`` (at least
        one) after an admit refused with nothing to evict, until a step of it
        admits. ``rows`` is what the free pages held, so one refusal sizes it.

        Holding the blamed rows alone does not shrink the retry: the ready
        queue refills the batch to the engine's cap, so a cap's worth of rows
        that cannot fit together is refused on every pass.
        """
        key = (node_name, graph_walk)
        limit = max(rows, 1)
        self.batch_limits[key] = min(limit, self.batch_limits.get(key, limit))

    def cap_batch(
        self, node_name: str, graph_walk: str, rows: int, resource: str | None = None,
    ) -> None:
        """Keep ``(node_name, graph_walk)``'s batches to ``rows`` (at least
        one), the most the KV pool ``resource`` holds even with nothing else
        resident.

        Unlike `limit_batch`, a step that admits at this size does not lift it:
        the next full batch could never fit either, so it would be refused,
        and with a victim to evict, evict one for nothing, on every other
        step. Lifted by a step that admits fewer rows than the cap, where the
        queue no longer fills it and the next batch may be of smaller rows,
        or by a teardown that hands pages of ``resource`` back (see
        `lift_caps`).
        """
        key = (node_name, graph_walk)
        cap = max(rows, 1)
        if cap <= self.batch_ceilings.get(key, cap):
            self.batch_ceilings[key] = cap
            self.ceiling_resources[key] = resource

    def lift_caps(self, resources: Collection[str]) -> None:
        """A teardown handed pages of ``resources`` back: lift the `cap_batch`
        caps their refusals set, and a cap whose refusal named no resource.

        A cap is sized from the rows refused then. On a busy walk every step
        is at the cap, so a smaller step never comes to lift it, and it would
        outlast the rows that set it.
        """
        for key, resource in list(self.ceiling_resources.items()):
            if resource is None or resource in resources:
                del self.batch_ceilings[key], self.ceiling_resources[key]

    def clear_batch_limit(
        self, node_name: str, graph_walk: str, rows: int, walks: Iterable[str] = (),
    ) -> None:
        """A step of ``rows`` rows of ``(node_name, graph_walk)`` admitted:
        lift its `limit_batch` limit, and its `cap_batch` cap if the step was
        smaller than both. ``walks`` are the real walks the step ran rows of.

        A step of a real walk lifts its combined walk's limit too: a mixed
        batch refused at two rows leaves a limit of one, which no mixed step
        can meet, so no step under the combined label would ever lift it. A
        mixed step lifts the limits of the walks it ran, since the combined
        walk's own cap sized it; their caps stay, as the combined cap, not
        their queues, set how many of their rows it took.
        """
        keys = [(node_name, graph_walk)]
        combined = self._combined_walk_of.get((node_name, graph_walk))
        if combined is not None:
            keys.append((node_name, combined))
        for key in keys:
            limit = self.batch_limits.pop(key, None)
            ceiling = self.batch_ceilings.get(key)
            if ceiling is not None and rows < min(ceiling, limit or ceiling):
                del self.batch_ceilings[key], self.ceiling_resources[key]
        for walk in walks:
            if walk != graph_walk:
                self.batch_limits.pop((node_name, walk), None)

    # ------------------------------------------------------------------
    # Refusals
    # ------------------------------------------------------------------

    def requeue(self, batch: ScheduledBatch, node_batch) -> None:
        """Put a batch whose admit refused it back on its queues, so the step
        can be retried. The caller runs any ready teardowns after this, which
        can't then push a torn-down rid onto queues already dismantled, and
        then `relieve`s the refusal.

        A rid prepare_inputs dropped (vetoed, as a loop's extra step past the
        rid's own count is, or failed) has no step to retry: on success the
        routing drops it too. Re-queued, it would run again with its loop-back
        inputs already spent. A row the cache served, or one that declined its
        stream's final chunk, is retried, because its node completes only on a
        step that runs.

        A ``terminal`` reason would refuse the same step again on every pass,
        so the requests it names fail and the rest are re-queued;
        the whole batch when it names no row of it. A TP follower only re-queues:
        rank 0 refused the same step, fails the same requests, and their
        teardown reaches this rank as a REMOVE.
        """
        prepared = set(node_batch.request_ids) | node_batch.no_forward_rids
        for rid in list(batch.request_to_worker_graph):
            if rid not in prepared:
                batch.discard_rid(rid)

        reason = node_batch.admit_error
        rids = batch.request_to_worker_graph
        doomed: set[int] = set()
        if reason.terminal and not isinstance(reason, AllocationFailed):
            named = getattr(reason, "request_ids", ())
            doomed = {rid for rid in rids if rid in named} or set(rids)
            if batch.node_name in self._follows:
                doomed = set()
        rows = batch.graph_rows()
        survivors = [rid for rid in rows if rid not in doomed]
        if survivors:
            self._runtime.push_back_node(batch.node_name, survivors, [rows[rid] for rid in survivors])
        if doomed:
            logger.error(
                "Admit refused node=%s walk=%s terminally (%s): failing %d of "
                "%d requests", batch.node_name, batch.graph_walk,
                reason.message, len(doomed), len(rids),
            )
            self._lifecycle.fail(
                dict.fromkeys(doomed, f"Error in worker: {reason.message}"), status=getattr(reason, "status", 500),
            )
        elif not reason.terminal and not isinstance(reason, AllocationFailed):
            logger.info(
                "Admit refused node=%s walk=%s (%s): re-queued %d requests",
                batch.node_name, batch.graph_walk, type(reason).__name__, len(batch),
            )

    def relieve(
        self, batch: ScheduledBatch, node_batch,
        arrivals: Iterable[int],
        last_in_line: Callable[[Collection[int]], int],
        could_refill: Callable[[], bool],
    ) -> None:
        """After `requeue` and the teardowns it lets run, make room for a step
        refused with ``AllocationFailed``: evict, hold or cap. Any other refusal
        has nothing to relieve. ``RequestOffloading`` means the rid is already
        on its way to the host, so evicting anything else is wasted work: the
        retry is gated on ``check_ready`` reloading it.

        Eviction reads this rank's own scheduler stamps, so it is the one
        decision not derivable from the broadcast. Only rank 0 picks a victim;
        its choice rides out on the next ``ScheduleTPNode`` as a resident-set
        delta, and a follow rank neither evicts nor holds: it has no scheduling
        decision a backoff could improve, and rank 0 hit the same refusal on the
        same batch and is already driving the retry.

        ``arrivals``: this worker's requests, earliest first. ``last_in_line``:
        of some rids, the one a step of this key reaches last. ``could_refill``:
        whether other rows wait to run; asked lazily, since it scans the queues.
        """
        reason = node_batch.admit_error
        if not isinstance(reason, AllocationFailed) or batch.node_name in self._follows:
            return
        batch_ids = set(batch.request_to_worker_graph)

        # A batch the pool cannot hold even with nothing else resident does not
        # fit after evicting a request outside it: that request only comes back
        # into the pages it freed. Only a row of the batch makes the retry smaller.
        # Compared with the rows the resource sized: a row the prefix cache
        # served whole is in the batch but runs no step and holds no pages.
        alone = getattr(reason, "rows_fit_alone", None)
        sized = getattr(reason, "rows_sized", None)
        too_big = alone is not None and alone < (len(batch_ids) if sized is None else sized)

        # scope the eviction to whichever resource actually ran out, when the
        # admit named one
        failed = node_batch.failed_resource
        victim_id = self._evict(
            node_batch.node_name, batch_ids, arrivals, last_in_line,
            affected_resources=None if failed is None else {failed},
            inside_only=too_big,
        )

        if victim_id is not None:
            self.hold([victim_id])
            # the victim is a row of the batch; rows waiting to run would refill
            # the retry to the size the pool cannot hold, and each refusal would
            # offload another running row
            if too_big and could_refill():
                self.cap_batch(batch.node_name, batch.graph_walk, alone, failed)
            logger.warning(
                "OOM on node=%s walk=%s: offloaded victim=%s, "
                "retrying %d remaining requests",
                batch.node_name, batch.graph_walk, victim_id,
                len(batch_ids) - (1 if victim_id in batch_ids else 0),
            )
            return
        # Nothing could be evicted or dropped, so the batch has to get smaller or it will
        # refuse identically for ever. Hold the one request the admit named,
        # not the whole batch, so the retry is immediate. When other rows
        # wait to run, the batch also gets a limit, or they refill it to the
        # same size: the rows the free pages held, or half when the resource
        # does not say. With none waiting a limit only leaves running rows
        # out of the step, where the next eviction picks them first.
        # Symmetric without coordinating: both ranks refuse the same batch
        # for the same request.
        blamed = getattr(reason, "request_id", None)
        shed = [blamed] if blamed in batch_ids else list(batch_ids)
        self.hold(shed)
        if could_refill():
            if too_big:
                self.cap_batch(batch.node_name, batch.graph_walk, alone, failed)
            fit = getattr(reason, "rows_fit", None)
            self.limit_batch(
                batch.node_name, batch.graph_walk,
                len(batch_ids) // 2 if fit is None else fit,
            )
        self.log_throttled(
            self._hold_logged, (batch.node_name, batch.graph_walk),
            "OOM on node=%s walk=%s: no offload possible, holding %s of %d "
            "requests so the retry is smaller (%d earlier holds not logged)",
            batch.node_name, batch.graph_walk,
            blamed if len(shed) == 1 else f"all {len(shed)}",
            len(batch_ids),
        )

    def log_throttled(self, logged: dict, key, message: str, *args) -> None:
        """Log ``message`` at most once per `_HOLD_LOG_INTERVAL` for ``key``, with the lines skipped since
        as its last argument."""
        now = time.monotonic()
        last, unlogged = logged.get(key, (None, 0))
        if last is not None and now - last < _HOLD_LOG_INTERVAL:
            logged[key] = (last, unlogged + 1)
            return
        logged[key] = (now, 0)
        logger.warning(message, *args, unlogged)

    def _evict(
        self, node_name: str, batch_ids: set[int], arrivals: Iterable[int],
        last_in_line: Callable[[Collection[int]], int],
        affected_resources: set[str] | None = None,
        inside_only: bool = False,
    ) -> int | None:
        """Offload one request's state for a refused step, freeing room to retry.

        Prefers a victim outside *batch_ids*; falls back to one inside it (the
        caller then excludes it from execution). ``inside_only`` takes only
        the latter. Returns the victim, or None when nothing could be
        reclaimed.
        """
        engine = self._engine_manager.get_engine(node_name)
        # quietly: a resource that never evicts (a LoRA pool) ran out, which is
        # backpressure for the scheduler, not a working set that does not fit
        if not engine.evictable(node_name, affected_resources):
            return None

        active = [
            rid for rid in arrivals
            if engine.residency(node_name, rid) is Residency.RESIDENT
            and (not inside_only or rid in batch_ids)
        ]

        # a request admitted but not yet run holds no pages: offloading it
        # frees nothing and the retry would re-pick it
        candidates = [
            rid for rid in active
            if engine.reclaimable(node_name, rid, affected_resources) and engine.can_offload(node_name, rid)
        ]

        if not candidates:
            if inside_only:
                return None
            logger.warning(
                "No eviction candidate on %s for %s: %d active requests, none "
                "with pages of it to reclaim now.", node_name,
                "any resource" if affected_resources is None
                else ", ".join(sorted(affected_resources)),
                len(active),
            )
            return None

        # prefer evicting requests that aren't currently executing
        external = [rid for rid in candidates if rid not in batch_ids]
        victim_id = last_in_line(external or candidates)
        freed = engine.offload_request(node_name, victim_id)
        if freed <= 0:
            # A victim was found and would not move. The usual cause is the host
            # pool being full (see ``CPUPagePool.place``) - different
            # from having no candidate, and fixed by a larger
            # ``cpu_offload_pages`` rather than by scheduling less.
            logger.warning(
                "Eviction victim %s on %s freed nothing — the host pool is "
                "most likely full. Raise cpu_offload_pages for the resource "
                "that ran out.", self._runtime.get_rid_string(victim_id), node_name,
            )
            return None
        logger.info(
            "Offloaded request %s from %s (%d reclaimed, policy=%s, in_batch=%s)",
            self._runtime.get_rid_string(victim_id), node_name, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id
