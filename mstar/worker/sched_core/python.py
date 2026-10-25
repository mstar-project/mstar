"""The scheduling pass in Python (``PythonSchedCore``): the core for a worker the Rust
graph runtime cannot schedule alone (a node that splits batches by capture
key, a speculation chain that mixes walks, or the Python runtime).

A pass is a generator that yields each ``CheckRequest`` and returns the step;
``run`` drives it.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Iterable
from typing import Any

from mstar.graph.runtime.base import GraphRuntime, RequestWalks
from mstar.model.submodule_base import InputMetadata
from mstar.worker.batch_builder import BaseBatchBuilder, BatchBuildRequest, BatchBuildResult
from mstar.worker.engine_manager import EngineManager
from mstar.worker.micro_scheduler import ReadyNodeEntry, ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.sched_core.base import (
    READY,
    UNCHECKED,
    CheckRequest,
    PassRequest,
    PassStep,
    RoundRobin,
    SchedCore,
    SchedKeys,
)

logger = logging.getLogger(__name__)


class PythonSchedCore(SchedCore):
    def __init__(
        self, runtime: GraphRuntime, engine_manager: EngineManager, keys: SchedKeys,
        batch_builder: BaseBatchBuilder, rr: RoundRobin, leader_nodes: Iterable[str],
    ):
        self.runtime = runtime
        self.engine_manager = engine_manager
        self.keys = keys
        self.batch_builder = batch_builder
        self.rr = rr
        self.leader_nodes = set(leader_nodes)
        # Batches already assembled and waiting their turn: what is left of a
        # ready set too big for one step. Taken before scanning the queues, so
        # a split batch finishes before anything new starts.
        # (node, graph walk) -> ScheduledBatch
        self.backlog: dict[tuple[str, str], ScheduledBatch] = {}
        # the pass in progress, and its generator
        self._req: PassRequest | None = None
        self._steps = None

    # -- the pass ---------------------------------------------------------------

    def begin(self, request_state: RequestStateManager, request: PassRequest) -> PassStep:
        self._req = request
        self._steps = self._pass(request_state, request)
        return self._advance(None)

    def resume(self, verdicts: bytes) -> PassStep:
        return self._advance(verdicts)

    def _advance(self, verdicts: bytes | None) -> PassStep:
        try:
            asked = self._steps.send(verdicts)
        except StopIteration as done:
            self._req = self._steps = None
            return PassStep(None, done.value)
        except BaseException:
            self._req = self._steps = None
            raise
        return PassStep(asked)

    def _pass(self, request_state: RequestStateManager, request: PassRequest):
        target, exclude_target = request.target, request.exclude_target
        pre_existing_batch_size = request.pre_existing_batch_size
        keys = self.keys
        # a targeted merge's pre-existing rows are of the target's own walk
        pre_existing_walk = target[1] if target is not None and pre_existing_batch_size else None
        # A chain merges and yields away within its own real walk, or its whole
        # combined walk under `speculation_mixes`.
        mixes = pre_existing_walk is not None and keys.speculation_mixes(*target)
        own_walk = None if mixes else pre_existing_walk
        skip_walk = None
        if exclude_target is not None and not keys.speculation_mixes(*exclude_target):
            skip_walk = exclude_target[1]
        scan_target = target if own_walk else None
        scan_exclude = exclude_target if skip_walk else None
        target = None if target is None else keys.key(*target)
        exclude_target = None if exclude_target is None else keys.key(*exclude_target)
        scan_target = scan_target or target
        scan_exclude = scan_exclude or exclude_target
        if target is not None and self._room_left(
            target, pre_existing_walk, pre_existing_batch_size,
        ) == 0:
            if mixes:
                # the chain never yields for its combined walk's other rows,
                # so park them: `room_for_continuing` gives them first claim
                yield from self._park_other_walks(request_state, target, pre_existing_walk)
            # otherwise another walk's rows wait on their queues for the yield
            return None
        # scanned only once a step could take fresh rows
        ready: dict[tuple[str, str], list[ReadyNodeEntry]] | None = None

        def schedule(key, entries):
            return (yield from self._schedule_key(
                request_state, key, entries,
                pre_existing_batch_size=pre_existing_batch_size,
                capture_group_of=request.capture_group_of,
                pre_existing_walk=pre_existing_walk,
                only_walk=own_walk,
                # a yield-away runs the other walks: the chain's own rows
                # parked under the combined walk wait for its next merge
                skip_walk=skip_walk,
            ))

        # A backlogged key goes first, oldest first, so a split set drains
        # before anything else starts; its fresh rows ride along. A key whose
        # step comes out empty (every backlogged row blocked) is passed over
        # rather than leaving the worker idle behind it. `exclude_target` is
        # only a fairness hint, and finishing a split set beats fairness.
        backlogged = [k for k in self.backlog if target is None or k == target]
        for key in backlogged:
            # no fresh row could join a step its backlog fills, unless every backlogged row is blocked
            if self._backlog_fills(key, pre_existing_walk, pre_existing_batch_size):
                scheduled = yield from schedule(key, [])
                if scheduled is not None:
                    return scheduled
            if ready is None:
                ready = yield from self._scan_ready(scan_target, scan_exclude)
            scheduled = yield from schedule(key, ready.get(key, []))
            if scheduled is not None:
                return scheduled
        if ready is None:
            ready = yield from self._scan_ready(scan_target, scan_exclude)
        # A walk with no room (a cap of 0) takes nothing, so it stays the least
        # recent; picked anyway, it would starve the walk that frees the room.
        rr_key = self.rr.pick(key for key in ready if keys.walk_caps(key)[key[1]] != 0)
        if rr_key is None or rr_key in backlogged:
            return None
        return (yield from schedule(rr_key, ready.get(rr_key, [])))

    def _park_other_walks(
        self, request_state: RequestStateManager,
        key: tuple[str, str], own_walk: str,
    ):
        """Backlog ``key``'s ready rows of walks other than ``own_walk``, for a
        mixing chain at its cap."""
        if key not in self.keys.walks_of_key:
            return
        # Excluded in the scan, not filtered after: own_walk's rows would
        # otherwise each pay a `check_ready` only to be dropped here.
        entries = (yield from self._scan_ready(key, (key[0], own_walk))).get(key, [])
        if not entries:
            return
        parked = self._assemble_batch(*key, entries)
        if parked:
            self._backlog(key, parked)

    def _scan_ready(
        self,
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
    ):
        """Engine-ready rows on the queues, by scheduling key, in scan order:
        a generator that yields every graph-ready row as one check request."""
        ready: dict[tuple[str, str], list[ReadyNodeEntry]] = {}
        rids: list[int] = []
        node_idx: list[int] = []
        nodes: list[str] = []
        rows: list[tuple[tuple[str, str], int, str]] = []
        # The runtime matches ``target`` and ``exclude_target`` against both
        # real and combined walks.
        # failed rows read live: a check earlier in this pass can fail one
        exclude = self._req.waiting.union(self._req.failed)
        for spec in self.runtime.get_ready_nodes(
            exclude, target=target, exclude_target=exclude_target,
        ):
            if spec.node_name not in self.leader_nodes:
                continue  # only rank 0 can initiate scheduling!
            wg_id = self.runtime.get_worker_graph_id_for_node(
                spec.node_name, spec.graph_walk,
            )
            if spec.node_name not in nodes:
                nodes.append(spec.node_name)
            n = nodes.index(spec.node_name)
            row = (self.keys.key(spec.node_name, spec.graph_walk), wg_id, spec.graph_walk)
            rids.extend(spec.rids)
            node_idx.extend([n] * len(spec.rids))
            rows.extend([row] * len(spec.rids))
        if not rids:
            return ready
        # whether the node is ready on the engine level (e.g., for AR, whether
        # the kv cache is read in)
        verdicts = yield CheckRequest(rids, node_idx, nodes)
        for rid, verdict, (key, wg_id, walk) in zip(rids, verdicts, rows, strict=True):
            if verdict == READY:
                ready.setdefault(key, []).append(ReadyNodeEntry(rid, wg_id, walk))
        return ready

    def _schedule_key(
        self, request_state: RequestStateManager,
        node_walk: tuple[str, str], entries: list[ReadyNodeEntry],
        pre_existing_batch_size: int,
        capture_group_of: int | None,
        pre_existing_walk: str | None = None,
        only_walk: str | None = None,
        skip_walk: str | None = None,
    ):
        """One step for ``node_walk`` out of its backlog and its ready
        ``entries``, composed by the batch builder. Under a combined walk,
        backlogged rows of any walk but ``only_walk`` (a speculation merge),
        or of ``skip_walk`` (a yield-away from that walk), stay parked."""
        remaining = self._room_left(node_walk, pre_existing_walk, pre_existing_batch_size)
        if remaining == 0:
            # The caller already holds a full batch. Assembling would pop these
            # nodes off their ready queues with nowhere to run them, so leave
            # them and the backlog as they are for the next pass.
            return None
        backlogged = self.backlog.pop(node_walk, None)
        blocked = set()
        if backlogged is not None:
            # bounded by what this step could take, so a long backlog is not
            # swept in full on every pass. Not `remaining`: that is the
            # caller's own walk's leftover and is None when it holds no rows.
            blocked = yield from self._unready_backlog_rids(
                backlogged, request_state,
                self._scan_limit(node_walk, pre_existing_batch_size),
                capture_group_of, only_walk, skip_walk,
            )
            if not backlogged:
                backlogged = None  # every row failed
        fresh = self._assemble_batch(*node_walk, entries) if entries else None
        if backlogged is None and not fresh:
            return None
        return self._build_and_schedule(request_state, BatchBuildRequest(
            node_name=node_walk[0], graph_walk=node_walk[1],
            backlog=backlogged, fresh=fresh or None, blocked_rids=blocked,
            capture_group_of=capture_group_of,
            walk_caps=self.keys.walk_caps(node_walk),
            walk_row_caps=self.keys.walk_row_caps(node_walk),
            pre_existing_batch_size=pre_existing_batch_size,
            pre_existing_walk=pre_existing_walk,
        ))

    def _room_left(
        self, node_walk: tuple[str, str], pre_existing_walk: str | None,
        pre_existing_batch_size: int,
    ) -> int | None:
        """What is left of the caller's own walk's cap in a ``node_walk`` step
        once its rows are counted; None when uncapped or the caller has none."""
        if not pre_existing_batch_size:
            return None
        own_cap = self.keys.walk_caps(node_walk).get(pre_existing_walk or node_walk[1])
        return None if own_cap is None else max(own_cap - pre_existing_batch_size, 0)

    def _scan_limit(
        self, node_walk: tuple[str, str], pre_existing_batch_size: int,
    ) -> int | None:
        """An upper bound on the rows this step can take, for bounding the
        backlog's readiness scan.

        The builder caps a mixed step at the min over the walks actually in
        it, which nothing knows before the scan; the max over the key's caps
        bounds that from above. None when any of them is uncapped.
        """
        caps = list(self.keys.walk_caps(node_walk).values())
        if not caps or any(cap is None for cap in caps):
            return None
        return max(max(caps) - pre_existing_batch_size, 0)

    def _capture_group(
        self, request_state: RequestStateManager,
        node_name: str, graph_walk: str, rid: int,
    ) -> Any | None:
        engine = self.engine_manager.get_engine(node_name)
        if not engine.splits_by_capture_key(node_name, graph_walk):
            return None
        fwd_info = request_state.get_fwd_info(
            rid, request_state.get_partition_for_node(node_name),
        )
        # a streamed input's chunk can decide the capture (a codec's window size)
        chunks = request_state.ingested_stream_chunks(rid, node_name)
        return engine.capture_group(
            node_name, graph_walk, rid, fwd_info,
            input_metadata=InputMetadata(stream_chunks=chunks) if chunks else None,
        )

    def _backlog_fills(
        self, key: tuple[str, str], pre_existing_walk: str | None,
        pre_existing_batch_size: int,
    ) -> bool:
        """Whether ``key``'s live backlog alone fills any step it could run,
        beside the caller's own rows, so a scan for fresh rows is wasted."""
        room = self._room_left(key, pre_existing_walk, pre_existing_batch_size)
        if room is None and not pre_existing_batch_size:
            caps = self.keys.walk_caps(key).values()
            room = None if None in caps else max(caps)
        req = self._req
        return room is not None and len(self._live_backlog(key, req.waiting, req.failed)) >= room

    def _live_backlog(
        self, node_walk: tuple[str, str], waiting: Iterable[int], failed: Iterable[int],
    ) -> set[int]:
        """``node_walk``'s backlogged rids that are not failed or waiting."""
        backlogged = self.backlog.get(node_walk)
        if backlogged is None:
            return set()
        return set(backlogged.request_to_worker_graph).difference(waiting, failed)

    def _unready_backlog_rids(
        self, batch: ScheduledBatch, request_state: RequestStateManager,
        max_bs: int | None = None, capture_group_of: int | None = None,
        only_walk: str | None = None, skip_walk: str | None = None,
    ):
        """The rows of a backlogged ``batch`` that cannot run this step; failed
        rows are dropped from it instead. Rows of a walk other than
        ``only_walk``, or of ``skip_walk``, are parked. A generator: the
        checks are yielded as one request."""
        failed = self._req.failed
        # Held (OOM backoff) and pending-remove rows wait without asking the
        # engine, as on the fresh path: asking reloads an offloaded row, which
        # would OOM it again straight away. They do not count against the
        # step's worth below either -- they were never candidates for it.
        waiting = self._req.waiting.intersection(batch.request_to_worker_graph)
        # Parked by walk, likewise without a check and outside the step's worth.
        if batch.request_walks and (only_walk or skip_walk) is not None:
            waiting |= {
                rid for rid in batch.request_to_worker_graph
                if (only_walk is not None and batch.walk_of(rid) != only_walk)
                or batch.walk_of(rid) == skip_walk
            }
        # Check readiness in backlog order and stop once a step's worth is
        # found: the builder takes the first `max_bs` rids not excluded, which
        # are exactly the ones checked and found ready, and the rids past that
        # point keep their place for the next pass.
        #
        # The capture group and the walk are settled here, not left to the
        # builder: an excluded row must not consume the step's worth, or the
        # scan would stop with rows the builder then takes unchecked.
        limit = len(batch) if max_bs is None else max(max_bs, 0)
        off_group: set[int] = set()
        counts = None
        if self.engine_manager.get_engine(batch.node_name).splits_by_capture_key(
            batch.node_name, batch.graph_walk,
        ):
            group = functools.partial(
                self._capture_group, request_state,
                batch.node_name, batch.graph_walk,
            )
            # `is not None`: handle 0 is a real rid
            anchor = [capture_group_of is not None,
                      None if capture_group_of is None else group(capture_group_of)]

            def counts(rid: int) -> bool:
                rid_group = group(rid)
                if not anchor[0]:
                    # the first eligible rid sets the group, as before
                    anchor[:] = [True, rid_group]
                if rid_group != anchor[1]:
                    off_group.add(rid)
                    return False
                return True
        not_ready_rids: set[int] = set(waiting)
        candidates = [rid for rid in batch.request_to_worker_graph if rid not in waiting]
        if candidates and limit > 0:
            verdicts = yield CheckRequest(candidates, None, [batch.node_name], limit, counts)
            not_ready_rids.update(
                rid for rid, verdict in zip(candidates, verdicts, strict=True)
                if verdict not in (READY, UNCHECKED)
            )
        not_ready_rids |= off_group
        # A failed rid is not "not ready yet": excluding it would put it
        # straight back in the backlog. This chunk is out of `self.backlog`
        # right now, so `drop_rids` cannot reach it.
        dropped = not_ready_rids & failed
        if dropped:
            for rid in dropped:
                batch.discard_rid(rid)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        return not_ready_rids - failed

    def _build_and_schedule(
        self, request_state: RequestStateManager, request: BatchBuildRequest,
    ) -> ScheduledBatch | None:
        """Have the batch builder compose one step for ``request``'s (node,
        walk), and carry out its verdict.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (request.node_name, request.graph_walk)
        request.capture_group = functools.partial(
            self._capture_group, request_state, *node_walk,
        )
        # Read before the build: `_merged` folds `fresh` into `backlog` in place.
        expected = set()
        if request.backlog is not None:
            expected |= request.backlog.request_to_worker_graph.keys()
        if request.fresh is not None:
            expected |= request.fresh.request_to_worker_graph.keys()
        result: BatchBuildResult = self.batch_builder.build_batch(request)
        self._refuse_lost_rows(node_walk, expected, result)
        # Whatever was popped past this step has to be remembered here, or it
        # would never run.
        if result.backlog is not None:
            self._backlog(node_walk, result.backlog)
        if result.returned:
            self._set_in_flight(request.node_name, result.returned, False)
            self.runtime.push_back_node(
                request.node_name, list(result.returned),
                list(result.returned.values()),
            )
        if result.scheduled is not None:
            logger.debug(
                "MicroScheduler scheduling node %s with graph walk %s for %d requests",
                *node_walk, len(result.scheduled),
            )
            self.rr.note(node_walk)
            result.scheduled.relabel_if_uniform()
        return result.scheduled

    @staticmethod
    def _refuse_lost_rows(
        node_walk: tuple[str, str], expected: set[int], result: BatchBuildResult,
    ) -> None:
        """Every row handed to a builder must come back out of it.

        A row missing from all three of ``scheduled``, ``backlog`` and
        ``returned`` has already come off its ready queue and is marked in
        flight, so nothing will ever schedule it again and the request hangs
        for good. Raising here costs one set per step and names the rows; the
        alternative is a silent hang with no evidence of where it started.
        """
        got = set()
        for part in (result.scheduled, result.backlog):
            if part is not None:
                got |= part.request_to_worker_graph.keys()
        if result.returned:
            got |= result.returned.keys()
        if got != expected:
            raise AssertionError(
                f"batch builder dropped rows for {node_walk}: "
                f"lost {sorted(expected - got)}, invented {sorted(got - expected)}"
            )

    def _backlog(self, node_walk: tuple[str, str], batch: ScheduledBatch) -> None:
        """Park ``batch``, folding it into whatever already waits under this key.

        Replacing would drop the rids already there — their graph nodes came
        off the ready queues when they were first assembled, so nothing would
        ever schedule them again and the requests hang.
        """
        existing = self.backlog.get(node_walk)
        if existing is None:
            self.backlog[node_walk] = batch
        else:
            existing.merge(batch)
        # Claimed like a running batch: an input arriving while it waits must
        # not put its node back on the ready queue, where it would be popped
        # a second time.
        self._set_in_flight(batch.node_name, batch.request_to_worker_graph, True)

    def _set_in_flight(
        self, node_name: str, rid_to_wg: dict[int, int], in_flight: bool,
    ) -> None:
        self.runtime.set_in_flight(
            node_name, list(rid_to_wg), list(rid_to_wg.values()), in_flight,
        )

    def _assemble_batch(
        self, node_name: str, graph_walk: str, entries: list[ReadyNodeEntry],
    ) -> ScheduledBatch | None:
        combined = (node_name, graph_walk) in self.keys.walks_of_key
        rids = [entry.request_id for entry in entries]
        walks = {entry.graph_walk for entry in entries} if combined else {graph_walk}
        rows = None
        if len(walks) == 1:  # the common case: one walk, the plain pop
            walk = walks.pop()
            popped = self.runtime.pop_rids(node_name, walk, rids)
        else:
            rows = RequestWalks.from_walks(rids, [entry.graph_walk for entry in entries])
            popped = self.runtime.pop_walk_rids(node_name, rows)
        if popped is None or not popped.wg_ids.keys:
            return None
        batch_rids = popped.wg_ids.keys
        batch = ScheduledBatch(
            node_name=node_name,
            graph_walk=graph_walk,
            request_to_worker_graph=dict(zip(batch_rids, popped.wg_ids.values, strict=True)),
            input_edges=popped.input_edges,
            output_signals=list(popped.output_signals),
        )
        if not combined:
            return batch
        if rows is None:
            batch.request_walks = dict.fromkeys(batch_rids, walk)
            batch.walk_output_signals = {walk: list(popped.output_signals)}
        else:
            batch.request_walks = {
                rid: rows.walks[w] for rid, w in zip(batch_rids, popped.rid_walk_idx, strict=True)
            }
            batch.walk_output_signals = {
                walk: list(signals)
                for walk, signals in zip(rows.walks, popped.walk_output_signals, strict=True)
            }
        return batch

    # -- the backlog, outside a pass ---------------------------------------------

    def room_for_continuing(self, target: tuple[str, str]) -> int | None:
        keys = self.keys
        key = keys.key(*target)
        backlog = self.backlog.get(key)
        cap = keys.max_batch_size(*target)
        if backlog is not None and backlog.request_walks:
            walks = set(backlog.request_walks.values())
            if keys.speculation_mixes(*target):
                # the chain's merge takes every backlogged row
                if walks != {target[1]}:
                    cap = keys.walk_set_cap(key, walks | {target[1]})
                return self.batch_builder.room_for_continuing(cap, backlog)
            # the chain's merge takes only its own walk's backlogged rows
            if cap is None:
                return None
            own = sum(1 for r in backlog.request_walks.values() if r == target[1])
            return max(0, cap - own)
        return self.batch_builder.room_for_continuing(cap, backlog)

    def backlog_has_schedulable(
        self, exclude_key: tuple[str, str] | None, exclude_walk: str | None,
        waiting: Iterable[int], failed: Iterable[int],
    ) -> bool:
        waiting, failed = set(waiting), set(failed)
        for node_walk, batch in self.backlog.items():
            only_other = node_walk == exclude_key and exclude_walk is not None
            if node_walk == exclude_key and not only_other:
                continue
            if only_other and (
                set(batch.request_walks.values()) if batch.request_walks else {batch.graph_walk}
            ) == {exclude_walk}:
                continue  # every row is of the excluded walk
            for rid in batch.request_to_worker_graph:
                if only_other and batch.walk_of(rid) == exclude_walk:
                    continue
                if rid in failed or rid in waiting:
                    continue
                return True
        return False

    def backlog_rows(self, exclude: Iterable[int]) -> list[tuple[str, int]]:
        exclude = set(exclude)
        return [
            (node, sum(1 for rid in batch.request_to_worker_graph if rid not in exclude))
            for (node, _), batch in self.backlog.items()
        ]

    def drop_rids(self, rids: Iterable[int]) -> None:
        for rid in rids:
            for batch in self.backlog.values():
                # Membership, not a falsy pop: a worker graph id of 0 is real.
                if rid not in batch.request_to_worker_graph:
                    continue
                batch.discard_rid(rid)
                batch.input_edges = batch.input_edges.select_rids(
                    batch.request_to_worker_graph.keys()
                )
        self.backlog = {k: v for k, v in self.backlog.items() if len(v) > 0}

    def backlog_splits_from(
        self, request_state: RequestStateManager, target: tuple[str, str], rid: int,
        waiting: Iterable[int], failed: Iterable[int],
    ) -> bool:
        target = self.keys.key(*target)
        live = self._live_backlog(target, waiting, failed)
        if not live:
            return False
        return self.batch_builder.chain_must_yield(
            self.backlog[target], live, rid,
            functools.partial(self._capture_group, request_state, *target),
        )
