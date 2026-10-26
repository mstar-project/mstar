"""The scheduling pass in Python (``PythonSchedCore``): the core for the Python
graph runtime, and the reference the Rust core follows.

A pass is explicit state (``_Pass``): the backlogged keys it tries, oldest
first, which attempt is current (``_Stage``), the scan once taken, and the
check it waits on. ``begin`` and ``resume`` advance it (``_run``) until it
needs rows checked or is done; the Rust core's ``Pass`` has the same shape.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

from mstar.graph.runtime.base import GraphRuntime, RequestWalks
from mstar.worker.batch_builder import BaseBatchBuilder, BatchBuildRequest, BatchBuildResult
from mstar.worker.micro_scheduler import ReadyNodeEntry, ScheduledBatch
from mstar.worker.sched_core.base import (
    NO_GROUP,
    CheckRequest,
    CheckResult,
    PassRequest,
    PassStep,
    RoundRobin,
    SchedCore,
    SchedKeys,
    Verdict,
)

logger = logging.getLogger(__name__)

Key = tuple[str, str]


class _Stage(Enum):
    """The pass's current attempt. Each backlogged key is tried on its
    backlog alone (when that fills a step), then with the scan's fresh rows;
    after them, the round-robin pick over the scan."""
    BACKLOG_ONLY = auto()
    WITH_FRESH = auto()
    PICK = auto()
    # a mixing chain at its cap: park its combined walk's other ready rows
    PARK = auto()


class _Scan(NamedTuple):
    """A ready scan's result: engine-ready rows by key, in scan order, and
    the group of ``capture_group_of`` when the scan named it."""
    ready: dict[Key, list[ReadyNodeEntry]]
    anchor_group: int | None


class _ScanCheck(NamedTuple):
    """The scan's check, and per requested row its key, worker graph and
    real walk."""
    request: CheckRequest
    rows: list[tuple[Key, int, str]]


class _BacklogCheck(NamedTuple):
    """A key's backlog recheck, and what the attempt needs once answered."""
    request: CheckRequest
    key: Key
    entries: list[ReadyNodeEntry]
    # the group the step must share, as known before the check
    anchor_group: int | None
    batch: ScheduledBatch
    # backlogged rows that wait without a check
    waiting: set[int]


@dataclass(slots=True)
class _Pass:
    req: PassRequest
    # the caller's own rows' walk (a speculation merge's)
    pre_walk: str | None
    # a merge takes only this walk's backlogged rows: `pre_walk` unless the
    # chain mixes walks
    own_walk: str | None
    # a yield-away leaves this walk's backlogged rows parked
    skip_walk: str | None
    # what the ready scan is narrowed to and kept from
    scan_target: Key | None
    scan_exclude: Key | None
    stage: _Stage
    # the backlogged keys to try, oldest first, and the current one
    keys: list[Key] = field(default_factory=list)
    ki: int = 0
    scan: _Scan | None = None
    awaiting: _ScanCheck | _BacklogCheck | None = None

    @property
    def pre(self) -> int:
        return self.req.pre_existing_batch_size


class PythonSchedCore(SchedCore):
    def __init__(
        self, runtime: GraphRuntime, keys: SchedKeys, batch_builder: BaseBatchBuilder,
        rr: RoundRobin, leader_nodes: Iterable[str],
    ):
        self.runtime = runtime
        self.keys = keys
        self.batch_builder = batch_builder
        self.rr = rr
        self.leader_nodes = set(leader_nodes)
        # Batches already assembled and waiting their turn: what is left of a
        # ready set too big for one step. Taken before scanning the queues, so
        # a split batch finishes before anything new starts.
        # (node, graph walk) -> ScheduledBatch
        self.backlog: dict[Key, ScheduledBatch] = {}
        # the pass waiting on a check
        self._pass: _Pass | None = None

    # -- the pass ---------------------------------------------------------------

    def begin(self, request: PassRequest) -> PassStep:
        keys = self.keys
        target, exclude_target = request.target, request.exclude_target
        # a targeted merge's pre-existing rows are of the target's own walk
        pre_walk = target[1] if target is not None and request.pre_existing_batch_size else None
        # A chain merges and yields away within its own real walk, or its whole
        # combined walk under `speculation_mixes`.
        mixes = pre_walk is not None and keys.speculation_mixes(*target)
        own_walk = None if mixes else pre_walk
        skip_walk = None
        if exclude_target is not None and not keys.speculation_mixes(*exclude_target):
            skip_walk = exclude_target[1]
        target_key = None if target is None else keys.key(*target)
        exclude_key = None if exclude_target is None else keys.key(*exclude_target)
        # The runtime matches a scan's target against real and combined walks.
        p = _Pass(
            request, pre_walk, own_walk, skip_walk,
            scan_target=target if own_walk else target_key,
            scan_exclude=exclude_target if skip_walk else exclude_key,
            stage=_Stage.BACKLOG_ONLY,
            # A backlogged key goes first, oldest first, so a split set drains
            # before anything else starts; its fresh rows ride along. A key
            # whose step comes out empty (every backlogged row blocked) is
            # passed over rather than leaving the worker idle behind it.
            # `exclude_target` is only a fairness hint, and finishing a split
            # set beats fairness.
            keys=[key for key in self.backlog if target_key is None or key == target_key],
        )
        if target_key is not None and self._room_left(target_key, pre_walk, p.pre) == 0:
            if not (mixes and target_key in keys.walks_of_key):
                # another walk's rows wait on their queues for the yield
                return PassStep(None)
            # The chain never yields for its combined walk's other rows, so
            # park them: `room_for_continuing` gives them first claim. Excluded
            # in the scan, not filtered after: the chain's own rows would
            # otherwise each pay a `check_ready` only to be dropped.
            p.stage = _Stage.PARK
            p.scan_target, p.scan_exclude = target_key, (target_key[0], pre_walk)
        return self._run(p)

    def resume(self, result: CheckResult) -> PassStep:
        p, self._pass = self._pass, None
        if p is None:
            raise RuntimeError("resume: no pass is waiting on a check")
        return self._run(p, result)

    def _run(self, p: _Pass, result: CheckResult | None = None) -> PassStep:
        """Advance ``p`` until it waits on a check or is done; ``result``
        answers the check it was waiting on."""
        awaiting, p.awaiting = p.awaiting, None
        if isinstance(awaiting, _ScanCheck):
            p.scan = self._read_scan(awaiting, result)
        elif awaiting is not None:
            done = self._attempted(p, self._finish_key(p, awaiting, result))
            if done is not None:
                return done
        while True:
            if p.stage is _Stage.BACKLOG_ONLY:
                if p.ki == len(p.keys):
                    p.stage = _Stage.PICK
                    continue
                # no fresh row could join a step its backlog fills, unless
                # every backlogged row is blocked
                if not self._backlog_fills(p, p.keys[p.ki]):
                    p.stage = _Stage.WITH_FRESH
                    continue
                key, entries, anchor_group = p.keys[p.ki], [], None
            else:
                if p.scan is None:  # scanned once, when a step could first take fresh rows
                    asked = self._scan_request(p)
                    if asked is not None:
                        return self._wait(p, asked)
                    p.scan = _Scan({}, None)
                if p.stage is _Stage.PARK:
                    self._park_other_walks(p)
                    return PassStep(None)
                key = p.keys[p.ki] if p.stage is _Stage.WITH_FRESH else self._rr_pick(p)
                if key is None:
                    return PassStep(None)
                entries, anchor_group = p.scan.ready.get(key, []), p.scan.anchor_group
            started = self._start_key(p, key, entries, anchor_group)
            if isinstance(started, _BacklogCheck):
                return self._wait(p, started)
            done = self._attempted(p, started)
            if done is not None:
                return done

    def _wait(self, p: _Pass, asked: _ScanCheck | _BacklogCheck) -> PassStep:
        p.awaiting = asked
        self._pass = p
        return PassStep(asked.request)

    @staticmethod
    def _attempted(p: _Pass, scheduled: ScheduledBatch | None) -> PassStep | None:
        """Where an attempt leaves the pass: done with its step (the
        round-robin pick is the last attempt), else on to the key's fresh
        attempt or the next key."""
        if scheduled is not None or p.stage is _Stage.PICK:
            return PassStep(None, scheduled)
        if p.stage is _Stage.BACKLOG_ONLY:
            p.stage = _Stage.WITH_FRESH
        else:
            p.ki += 1
            p.stage = _Stage.BACKLOG_ONLY
        return None

    def _rr_pick(self, p: _Pass) -> Key | None:
        """The least recently scheduled key with fresh rows, unless it is one
        of the backlogged keys the pass already tried. A walk with no room (a
        cap of 0) takes nothing, so it stays the least recent; picked anyway,
        it would starve the walk that frees the room."""
        key = self.rr.pick(
            key for key in p.scan.ready if self.keys.walk_caps(key)[key[1]] != 0
        )
        return None if key is None or key in p.keys else key

    # -- the checks ---------------------------------------------------------------

    def _scan_request(self, p: _Pass) -> _ScanCheck | None:
        """The check for every graph-ready row, or None when there is none."""
        rids: list[int] = []
        key_idx: list[int] = []
        keys: list[Key] = []
        rows: list[tuple[Key, int, str]] = []
        # failed rows read live: a check earlier in this pass can fail one
        exclude = p.req.waiting.union(p.req.failed)
        for spec in self.runtime.get_ready_nodes(
            exclude, target=p.scan_target, exclude_target=p.scan_exclude,
        ):
            if spec.node_name not in self.leader_nodes:
                continue  # only rank 0 can initiate scheduling!
            wg_id = self.runtime.get_worker_graph_id_for_node(spec.node_name, spec.graph_walk)
            key = self.keys.key(spec.node_name, spec.graph_walk)
            if key not in keys:
                keys.append(key)
            rids.extend(spec.rids)
            key_idx.extend([keys.index(key)] * len(spec.rids))
            rows.extend([(key, wg_id, spec.graph_walk)] * len(spec.rids))
        if not rids:
            return None
        return _ScanCheck(CheckRequest(
            rids, key_idx, [node for node, _ in keys], walks=[label for _, label in keys],
            # the anchor's group, for a step the scan was narrowed to
            anchor=p.req.capture_group_of if p.scan_target is not None else None,
        ), rows)

    @staticmethod
    def _read_scan(asked: _ScanCheck, result: CheckResult) -> _Scan:
        ready: dict[Key, list[ReadyNodeEntry]] = {}
        groups = result.groups
        for i, (rid, verdict, (key, wg_id, walk)) in enumerate(
            zip(asked.request.rids, result.verdicts, asked.rows, strict=True),
        ):
            if verdict == Verdict.READY:
                group = NO_GROUP if groups is None else groups[i]
                ready.setdefault(key, []).append(ReadyNodeEntry(rid, wg_id, walk, group))
        # the anchor's group only when the scan named one; otherwise the step
        # anchors on its own first row (see `_off_group`)
        anchored = asked.request.anchor is not None
        return _Scan(ready, result.anchor_group if anchored else None)

    # -- one key's attempt ---------------------------------------------------------

    def _start_key(
        self, p: _Pass, key: Key, entries: list[ReadyNodeEntry], anchor_group: int | None,
    ) -> _BacklogCheck | ScheduledBatch | None:
        """One step for ``key`` out of its backlog and its fresh ``entries``:
        the step, or the backlog's recheck to answer first. Under a combined
        walk, backlogged rows of any walk but ``p.own_walk`` (a speculation
        merge), or of ``p.skip_walk`` (a yield-away from that walk), stay
        parked."""
        if self._room_left(key, p.pre_walk, p.pre) == 0:
            # The caller already holds a full batch. Assembling would pop these
            # nodes off their ready queues with nowhere to run them, so leave
            # them and the backlog as they are for the next pass.
            return None
        backlogged = self.backlog.pop(key, None)
        if backlogged is None:
            return self._build_key(p, key, entries, None, set(), anchor_group)
        asked, waiting = self._backlog_request(p, backlogged)
        if asked is None:
            return self._build_key(p, key, entries, backlogged, waiting, anchor_group)
        return _BacklogCheck(asked, key, entries, anchor_group, backlogged, waiting)

    def _finish_key(
        self, p: _Pass, chk: _BacklogCheck, result: CheckResult,
    ) -> ScheduledBatch | None:
        blocked = self._backlog_blocked(p, chk, result)
        anchor_group = chk.anchor_group if result.anchor_group is None else result.anchor_group
        backlogged = chk.batch if len(chk.batch) else None  # empty: every row failed
        return self._build_key(p, chk.key, chk.entries, backlogged, blocked, anchor_group)

    def _build_key(
        self, p: _Pass, key: Key, entries: list[ReadyNodeEntry],
        backlogged: ScheduledBatch | None, blocked: set[int], anchor_group: int | None,
    ) -> ScheduledBatch | None:
        fresh = self._assemble_batch(*key, entries) if entries else None
        if backlogged is None and not fresh:
            return None
        if fresh:
            blocked |= self._off_group(fresh, entries, anchor_group)
        return self._build_and_schedule(BatchBuildRequest(
            node_name=key[0], graph_walk=key[1],
            backlog=backlogged, fresh=fresh or None, blocked_rids=blocked,
            walk_caps=self.keys.walk_caps(key),
            walk_row_caps=self.keys.walk_row_caps(key),
            pre_existing_batch_size=p.pre,
            pre_existing_walk=p.pre_walk,
        ))

    def _backlog_request(
        self, p: _Pass, batch: ScheduledBatch,
    ) -> tuple[CheckRequest | None, set[int]]:
        """The readiness check for a backlogged ``batch``, and its rows that
        wait without one. Held (OOM backoff) and pending-remove rows wait, as
        on the fresh path: asking reloads an offloaded row, which would OOM it
        again straight away; so do rows of a walk other than ``p.own_walk``,
        or of ``p.skip_walk``. None of them counts against the step's worth.

        The check stops once a step's worth is found, bounded by what this
        step could take so a long backlog is not swept in full on every pass:
        the builder takes the first rows not blocked, which are exactly the
        ones checked and found ready, and the rows past that point keep their
        place for the next pass. Under a capture-split node only rows of the
        step's group count toward that, or the check would stop with rows the
        builder then takes unchecked (``CheckRequest.by_anchor``)."""
        waiting = set(p.req.waiting.intersection(batch.request_to_worker_graph))
        if batch.request_walks and (p.own_walk or p.skip_walk) is not None:
            waiting |= {
                rid for rid in batch.request_to_worker_graph
                if (p.own_walk is not None and batch.walk_of(rid) != p.own_walk)
                or batch.walk_of(rid) == p.skip_walk
            }
        max_bs = self._scan_limit((batch.node_name, batch.graph_walk), p.pre)
        limit = len(batch) if max_bs is None else max(max_bs, 0)
        candidates = [rid for rid in batch.request_to_worker_graph if rid not in waiting]
        if not candidates or limit <= 0:
            return None, waiting
        return CheckRequest(
            candidates, None, [batch.node_name], limit,
            walks=[batch.graph_walk], anchor=p.req.capture_group_of, by_anchor=True,
        ), waiting

    @staticmethod
    def _backlog_blocked(p: _Pass, chk: _BacklogCheck, result: CheckResult) -> set[int]:
        """The backlogged rows that cannot run this step: waiting, not ready
        or off the step's capture group. A failed row is not "not ready yet":
        excluding it would put it straight back in the backlog, and this chunk
        is out of ``self.backlog`` right now, so ``drop_rids`` cannot reach
        it -- it is dropped from the batch here."""
        failed = p.req.failed
        batch = chk.batch
        blocked = chk.waiting | {
            rid for rid, verdict in zip(chk.request.rids, result.verdicts, strict=True)
            if verdict not in (Verdict.READY, Verdict.UNCHECKED)
        }
        dropped = blocked & failed
        if dropped:
            for rid in dropped:
                batch.discard_rid(rid)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        return blocked - failed

    @staticmethod
    def _off_group(
        fresh: ScheduledBatch, entries: list[ReadyNodeEntry], anchor_group: int | None,
    ) -> set[int]:
        """The fresh rows outside the step's capture group: the anchor's, or
        with none (no ``capture_group_of`` and no ready backlogged row) the
        first fresh row's. They park for a later step."""
        group_of = {entry.request_id: entry.group for entry in entries}
        rids = list(fresh.request_to_worker_graph)
        if all(group_of[rid] == NO_GROUP for rid in rids):
            return set()
        wanted = group_of[rids[0]] if anchor_group is None else anchor_group
        return {rid for rid in rids if group_of[rid] != wanted}

    def _park_other_walks(self, p: _Pass) -> None:
        """Backlog the scan's rows of the target's combined walk, for a mixing
        chain at its cap (the scan left out the chain's own walk)."""
        key = p.scan_target
        entries = p.scan.ready.get(key, [])
        if not entries:
            return
        parked = self._assemble_batch(*key, entries)
        if parked:
            self._backlog(key, parked)

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

    def _backlog_fills(self, p: _Pass, key: Key) -> bool:
        """Whether ``key``'s live backlog alone fills any step it could run,
        beside the caller's own rows, so a scan for fresh rows is wasted."""
        room = self._room_left(key, p.pre_walk, p.pre)
        if room is None and not p.pre:
            caps = self.keys.walk_caps(key).values()
            room = None if None in caps else max(caps)
        return room is not None and len(self.live_backlog(key, p.req.waiting, p.req.failed)) >= room

    def _build_and_schedule(self, request: BatchBuildRequest) -> ScheduledBatch | None:
        """Have the batch builder compose one step for ``request``'s (node,
        walk), and carry out its verdict.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (request.node_name, request.graph_walk)
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

    def live_backlog(
        self, key: tuple[str, str], waiting: Iterable[int], failed: Iterable[int],
    ) -> list[int]:
        backlogged = self.backlog.get(key)
        if backlogged is None:
            return []
        skip = set(waiting).union(failed)
        return [rid for rid in backlogged.request_to_worker_graph if rid not in skip]
