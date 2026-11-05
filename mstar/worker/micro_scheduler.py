import functools
import itertools
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from mstar.engine.chunking import ChunkRange
from mstar.graph.runtime.base import ReadyNodeSpec, RequestWalks
from mstar.model.submodule_base import InputMetadata, StepOrder, chunk_loop_name
from mstar.worker.admission import Admission
from mstar.worker.batch_builder import (
    BatchBuilderType,
    BatchBuildRequest,
    BatchBuildResult,
    LineOrder,
    StepCandidates,
    StepPick,
    make_batch_builder,
)
from mstar.worker.lockstep import Lockstep
from mstar.worker.scheduled_batch import (  # noqa: F401 (re-exported)
    REMOVED_RID,
    ContinuingRows,
    MidPrefillRow,
    PopReadyResult,
    ScheduledBatch,
    cap_chunk_loops,
    chunk_loop_stride,
    runs_final_chunk,
    runs_first_chunk,
    tp_chunk_ranges,
    unfinished_chunk_rids,
)
from mstar.worker.services import WorkerLayout, WorkerServices

logger = logging.getLogger(__name__)

# The lane of the worker's main chain of steps, and of the one side lane, which
# runs its keys' steps beside it (``mstar.worker.side_steps``)
CHAIN_LANE = 0
SIDE_LANE = 1

# Below every turn and arrival: a sort key offset that puts a row ahead of them all
_AHEAD = 1 << 62
# between `_AHEAD`'s rows and the rest: a row mid-prefill under a token budget
_MID_PREFILL = 1 << 61


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: int
    worker_graph_id: int
    graph_walk: str





class SchedulingType(Enum):
    ROUND_ROBIN = "round_robin"
    # TODO: priority. It used to key off a per-engine-type table, which no
    # longer exists — every node runs on the same engine now. The replacement
    # is for the model to declare a (node, graph_walk) priority, since only it
    # knows which walk is latency-sensitive. Worth weighing against
    # head-of-line blocking: a busy high-priority walk starves the rest.


class ChainPick(NamedTuple):
    """A speculation chain's next step at its target, as ``pick_chain_step``
    chose it; ``take_chain_step`` pops the rows it took off the queues."""
    key: tuple[str, str]
    pick: StepPick
    # the target's ready-queue scan the pick ran over
    specs: list
    # the caller's rows mid-prefill that the step took (``pick_chain_step``)
    mid_prefill: Sequence[MidPrefillRow] = ()



def _pre_existing_tokens(seq_lens: Collection[int], n: int) -> int:
    """The tokens a speculation chain's own ``n`` rows will run, as the
    submodule estimates them (``speculative_input_seq_len``): their inputs do
    not exist yet."""
    # A row with no estimate (-1) guesses one token, a decode row.
    return sum(s if s >= 0 else 1 for s in seq_lens) + n - len(seq_lens)


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    def __init__(
        self, services: WorkerServices, layout: WorkerLayout,
        lockstep: Lockstep, admission: Admission, model_config: dict | None = None,
        sched_type=SchedulingType.ROUND_ROBIN,
        max_consec_tp_follower_batches: int = 1,
    ):
        model_config = model_config or {}
        self.engine_manager = services.engine_manager
        self.request_state = services.request_state
        self.lockstep = lockstep
        self.admission = admission
        self._orders: dict[tuple[str, str], StepOrder] = {}
        # (node, real walk) -> combined walk; scheduling keys use the combined one
        self.combined_walk_of = layout.combined_walk_of
        self._walks_of_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (node, walk), combined in self.combined_walk_of.items():
            self._walks_of_key[(node, combined)].append(walk)
        # (node, walk) -> the lane its steps run on, for keys off the chain: a
        # lane schedules only its own keys (``next_batch``), so the chain
        # neither runs nor yields to another lane's
        self.lane_of: dict[tuple[str, str], int] = dict.fromkeys(layout.side_keys, SIDE_LANE)
        # (node, walk) -> whether its speculation chain merges the other walks
        # of its combined walk: the config's settings, then each default as
        # `speculation_mixes` first decides it
        self._mixing_chains = self._parse_speculation_mixes_walks(
            model_config.get("speculation_mixes_walks") or {},
        )
        # scheduling key -> whether its steps run under a token budget; see `chunks`
        self._budgeted: dict[tuple[str, str], bool] = {}
        self._max_bs: dict[tuple[str, str], int | None] = {}
        self._walk_caps_of_key: dict[tuple[str, str], dict[str, int | None]] = {}
        # (node, walk) -> whether its cap can move between passes
        self._cap_varies: dict[tuple[str, str], bool] = {}
        self._row_caps_of_key: dict[tuple[str, str], dict[str, int]] = {}
        self.batch_builder = make_batch_builder(BatchBuilderType(model_config.get("batch_builder", "fifo")))
        self.batch_number = 0
        # Interning and the (walk, node) -> worker graph index both live here
        self.runtime = services.graph_runtime
        # node -> the walks whose prefill runs as a chunk Loop there
        self.chunk_loop_walks = layout.chunk_loops.walks_by_node
        # Resolves a tensor uuid, for row lengths read from popped inputs.
        # None skips the room check.
        self.get_tensor: Callable[[int], Any] | None = services.tensor_manager.get_tensor
        # (rid, node) -> when its prompt's first chunk was cut, for serving
        # prompts in start order; held while the prompt is mid-prefill, and
        # dropped once its last chunk is cut (`_assign_chunks`).
        self.chunk_started: dict[tuple[int, str], int] = {}
        self._chunk_starts = itertools.count()
        # (rid, node) -> cached prefix a measured, not yet chunked row starts past
        self.cached_prefix: dict[tuple[int, str], int] = {}
        self.sched_type = sched_type

        # RIDs that have failed but have not gone through the cleanup procedure;
        # these cannot be scheduled (unless this is a TP follower node)
        self.failed_rids: set[int] = set()
        # lockstep-parallel (TP / SP instance) scheduling
        self.parallel_leader_nodes = layout.parallel_leader_nodes
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        # Batches already assembled and waiting their turn: what is left of a
        # ready set too big for one step. Taken before scanning the queues, so
        # a split batch finishes before anything new starts.

        # (node, graph walk) -> ScheduledBatch
        self.backlog: dict[tuple[str, str], ScheduledBatch] = {}

        self.node_and_walk_to_last_batch_num = {}
        # scheduling key -> rid -> its turn there: a counter over every row of
        # every admitted step, given out in the order the step took its rows
        # (`note_ran`), so sorting by it is first come, first served. Kept per
        # key: a row that runs another node meanwhile has still waited here.
        self.last_ran: dict[tuple[str, str], dict[int, int]] = {}
        self._turns = 0
        # scheduling key -> the first turn of its latest admitted step: the
        # rows at or past it hold the key's step
        self._latest_step: dict[tuple[str, str], int] = {}
        # rid -> the order requests arrived at this worker in (`note_arrival`)
        self.arrival: dict[int, int] = {}
        self._arrivals = 0
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: dict[int, None] = {}

    def _parse_speculation_mixes_walks(
        self, setting: Mapping[str, Any],
    ) -> dict[tuple[str, str], bool]:
        """The serving config's ``speculation_mixes_walks``: per node, one bool
        for every walk of its combined walks, or a mapping of walk -> bool."""
        walks_of_node: dict[str, set[str]] = defaultdict(set)
        for node, walk in self.combined_walk_of:
            walks_of_node[node].add(walk)
        out: dict[tuple[str, str], bool] = {}
        for node, value in setting.items():
            if node not in walks_of_node:
                raise ValueError(
                    f"speculation_mixes_walks names {node!r}, which has no combined walk"
                )
            per_walk = value if isinstance(value, Mapping) else dict.fromkeys(walks_of_node[node], value)
            for walk, mixes in per_walk.items():
                if walk not in walks_of_node[node]:
                    raise ValueError(
                        f"speculation_mixes_walks names walk {walk!r} of {node!r}, "
                        "which is in none of its combined walks"
                    )
                if not isinstance(mixes, bool):
                    raise ValueError(
                        f"speculation_mixes_walks for {node}/{walk}: {mixes!r} is not a bool"
                    )
                out[(node, walk)] = mixes
        return out

    def speculation_mixes(self, node_name: str, graph_walk: str) -> bool:
        """Whether a speculation chain on ``(node_name, graph_walk)`` merges
        ready rows of the other walks of its combined walk. Off, the chain
        takes only its own walk's rows, and another walk's ready rows make it
        yield; steps then mix walks only when the scheduler builds one fresh,
        which plans inline.

        Unless the config says otherwise, on wherever the combined walk has a
        token budget: chunked prompts then ride in decode steps rather than
        decode sitting out every chunk step. Without one a merged prompt runs
        whole and stretches the decode step it joins."""
        key = (node_name, graph_walk)
        mixes = self._mixing_chains.get(key)
        if mixes is None:
            mixes = key in self.combined_walk_of and self.chunks(node_name, graph_walk)
            self._mixing_chains[key] = mixes
        return mixes

    def chunks(self, node_name: str, graph_walk: str) -> bool:
        """Whether steps of ``(node_name, graph_walk)`` run under a token
        budget: its scheduling key's, which is the combined walk's under one.
        A walk without one runs its prompts whole, even inside a chunk Loop."""
        key = self._key(node_name, graph_walk)
        budgeted = self._budgeted.get(key)
        if budgeted is None:
            budgeted = self.engine_manager.get_engine(key[0]).get_max_batch_tokens(*key) is not None
            self._budgeted[key] = budgeted
        return budgeted

    def _node_rr_order(
        self, ready: dict[tuple[str, str], list[ReadyNodeEntry]],
    ) -> list[tuple[str, str]]:
        """The (node, walk) keys with a ready row, least recently scheduled
        first; ties go to the first scanned."""
        return sorted(
            ready,
            key=lambda key: self.node_and_walk_to_last_batch_num.get(key, 0),
        )

    def note_ran(self, batch: ScheduledBatch) -> None:
        """Give ``batch``'s rows their turn under its scheduling key, in the
        order its step took them. Called once the step's admit has taken it,
        so the rows of a refused step keep their place in line."""
        key = self._key(batch.node_name, batch.graph_walk)
        turns = self.last_ran.get(key)
        if turns is None:
            turns = self.last_ran[key] = {}
        rows = batch.request_to_worker_graph
        if batch.turn_order is not None:
            listed = [rid for rid in batch.turn_order if rid in rows]
            if len(listed) < len(rows):
                seen = set(listed)
                listed += [rid for rid in rows if rid not in seen]
            rows = listed
        first = self._turns + 1
        self._turns += len(rows)
        turns.update(zip(rows, range(first, self._turns + 1), strict=True))
        self._latest_step[key] = first

    def last_in_line(self, node_name: str, graph_walk: str, rids: Iterable[int]) -> int:
        """Of ``rids``, the one a step of ``(node_name, graph_walk)`` would
        reach last: the line a capped step's pick reads."""
        return max(rids, key=self._line_order(self._key(node_name, graph_walk)).key)

    def note_arrival(self, rid: int) -> None:
        """A request reached this worker; a request with several partitions
        here keeps its first arrival."""
        if rid not in self.arrival:
            self._arrivals += 1
            self.arrival[rid] = self._arrivals

    def _order_of(self, key: tuple[str, str]) -> StepOrder:
        """``key``'s `StepOrder`, as its submodule declares it. A combined
        walk's step holds its walks' rows, so it takes turns if any of its
        walks does."""
        order = self._orders.get(key)
        if order is None:
            node, walk = key
            engine = self.engine_manager.get_engine(node)
            orders = [engine.get_step_order(node, real) for real in self._walks_of_key.get(key) or (walk,)]
            order = StepOrder.TURNS if StepOrder.TURNS in orders else StepOrder.ARRIVAL
            self._orders[key] = order
        return order

    def _line_order(self, key: tuple[str, str]) -> LineOrder:
        """``StepCandidates.line_order`` for ``key``."""
        turns = self.last_ran.get(key, {})
        arrived = self.arrival.get
        if self._order_of(key) is StepOrder.TURNS:
            # longest-waiting first; a row new to the key ahead of every row
            # that ran here, the earliest request first
            def waited(rid: int) -> int:
                turn = turns.get(rid)
                return arrived(rid, 0) - _AHEAD if turn is None else turn
            return LineOrder(waited, strict=False)
        holding = self._latest_step.get(key, _AHEAD)

        # the rows holding the step, then the earliest request
        def arrival(rid: int) -> int:
            return arrived(rid, 0) - (_AHEAD if turns.get(rid, 0) >= holding else 0)
        if not self.chunks(*key):
            return LineOrder(arrival, strict=True)
        # Under a token budget a row mid-prefill comes next, in the order the
        # prompts started, so prompts that haven't started (back from an OOM
        # hold, say) can't take a mixed step's prompt rows ahead of it.
        started, node = self.chunk_started, key[0]

        def mid_prefill_next(rid: int) -> int:
            at = started.get((rid, node))
            if at is None or turns.get(rid, 0) >= holding:
                return arrival(rid)
            return at - _MID_PREFILL
        return LineOrder(mid_prefill_next, strict=True)

    def could_refill(self, node_name: str, graph_walk: str, batch_rids: set[int]) -> bool:
        """Whether rows of ``(node_name, graph_walk)``'s scheduling key other
        than ``batch_rids`` wait to run, so they could refill a retry of that
        batch to its size. When none do, the batch took every row it could,
        and holding the blamed one is what makes the retry smaller.

        Asked of the graph, not the engine: an engine readiness check would
        start reloading an offloaded row. Offloaded rows are left out; they
        come back only once their pages fit. Refusal path only.
        """
        key = self._key(node_name, graph_walk)
        if self._live_backlog(key) - batch_rids:
            return True
        exclude = self.failed_rids.union(self.pending_removes, self.admission.held_until, batch_rids)
        engine = self.engine_manager.get_engine(node_name)
        return any(
            not engine.is_offloaded(spec.node_name, rid)
            for spec in self._ready_specs(exclude, key, None)
            if spec.node_name == node_name
            for rid in spec.rids
        )

    def _try_schedule_tp_follow(
        self, exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        """The follow head's step, unless the turn goes to a colocated node."""
        head = self.lockstep.follow_head()
        if head is None:
            return None
        head_key = self._key(head.node_name, head.graph_walk)
        if exclude_target is not None and head_key == self._key(*exclude_target):
            return None
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(head_key):
            return None
        batch = self.lockstep.build_follow_step()
        if batch is not None:
            self.note_follow_pop(head_key)
        return batch

    def note_follow_pop(self, key: tuple[str, str]) -> None:
        """The follow head of scheduling key ``key`` was popped, which counts for
        round-robin order as a scheduled step does."""
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[key] = self.batch_number

    def next_batch(
        self,
        exclude_target: tuple[str, str] | None = None,
        lane: int = CHAIN_LANE,
    ) -> ScheduledBatch | None:
        """
        One step's worth: a pending TP follow batch if there is one, else
        the batch builder's pick from one (node, walk)'s backlog and ready rows.

        A (node, walk) with a backlog — the remainder of a ready set too big
        for one forward — goes first, so a split set drains before anything
        else starts; its fresh ready rows join the same step.

        A batch holds one capture group (``Engine.capture_group``); rids of
        another group wait in the backlog for a batch of their own.

        A speculation chain's step at its own target is ``pick_chain_step``'s.

        Args:
            exclude_target: If set, skip this (node_name, graph_walk) pair:
                a yield-away from the speculation chain running it.
            lane: whose keys to schedule (``lane_of``); TP follow steps are
                the chain's.
        """
        # done before any early returns
        self.admission.let_rows_back()

        # Note: a TP follow batch has to be scheduled irrespective of failure.
        # Rank 0 already committed to this batch and will sit on the collective
        # inside the forward until every follower joins it, so a follower that
        # skipped the batch because one of its rids failed locally would hang
        # the whole TP group. A popped ScheduleTPNode has no re-queue path
        # and must be submitted unconditionally.
        if lane == CHAIN_LANE:
            tp_follow_batch = self._try_schedule_tp_follow(exclude_target)
            if tp_follow_batch is None:
                self.num_consec_tp_follower_batches = 0
            else:
                self.num_consec_tp_follower_batches += 1
                return tp_follow_batch

        if self.sched_type != SchedulingType.ROUND_ROBIN:
            raise NotImplementedError(f"Unknown scheduling type {self.sched_type}")
        # A chain merges and yields away within its own real walk, or its whole
        # combined walk under `speculation_mixes`.
        skip_walk = None
        if exclude_target is not None and not self.speculation_mixes(*exclude_target):
            skip_walk = exclude_target[1]
        scan_exclude = exclude_target if skip_walk else None
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
        scan_exclude = scan_exclude or exclude_target
        # A queued row can be first in line even where a backlog fills the
        # step on its own (a row holding the step, back on its queue).
        ready = self._scan_ready(None, scan_exclude, lane)

        def schedule(key, entries):
            # a yield-away runs the other walks: the chain's own rows parked
            # under the combined walk wait for its next step
            return self._schedule_key(key, entries, skip_walk=skip_walk)

        # A backlogged key goes first, oldest first, so a split set drains
        # before anything else starts; its fresh rows ride along. A key whose
        # step comes out empty (every backlogged row blocked) is passed over
        # rather than leaving the worker idle behind it. `exclude_target` is
        # only a fairness hint, and finishing a split set beats fairness --
        # except under ARRIVAL, where the rows holding a key's step keep it: a
        # yield-away from the chain running it leaves that backlog to it.
        keys = [
            k for k in self.backlog
            if self._lane(k) == lane and not self._left_to_its_chain(k, exclude_target, skip_walk)
        ]
        for key in keys:
            scheduled = schedule(key, ready.get(key, []))
            if scheduled is not None:
                return scheduled
        # A walk with no room (a cap of 0) takes nothing, so it stays the least
        # recent; picked anyway, it would starve the walk that frees the room.
        ready = {
            key: rows for key, rows in ready.items() if self._walk_caps(key)[key[1]] != 0
        }
        # Keep going down the round-robin order rather than idling behind the
        # first node whose step comes out empty. That node is not marked
        # scheduled, so round robin picks it again next pass anyway -- and
        # under a colocated model the node behind it is the one that frees
        # its pages (snac_decoder behind Orpheus's LLM).
        for rr_key in self._node_rr_order(ready):
            if rr_key in keys:
                continue  # already tried above, off its backlog
            scheduled = schedule(rr_key, ready.get(rr_key, []))
            if scheduled is not None:
                return scheduled
        return None

    def _lane(self, key: tuple[str, str]) -> int:
        return self.lane_of.get(key, CHAIN_LANE)

    def _left_to_its_chain(
        self, key: tuple[str, str], exclude_key: tuple[str, str] | None,
        skip_walk: str | None,
    ) -> bool:
        """Whether a yield-away from ``exclude_key`` leaves ``key``'s backlog
        alone: under ARRIVAL, when the chain it yields from holds that step.
        A combined walk's other walks still run (their rows are the
        yield-away's to take; ``skip_walk`` parks the chain's own)."""
        if key != exclude_key or self._order_of(key) is not StepOrder.ARRIVAL:
            return False
        return skip_walk is None or key not in self._walks_of_key

    def _scan_ready(
        self, target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
        lane: int = CHAIN_LANE,
    ) -> dict[tuple[str, str], list[ReadyNodeEntry]]:
        """Engine-ready rows on the queues, by scheduling key, in scan order."""
        ready: dict[tuple[str, str], list[ReadyNodeEntry]] = {}
        # Do not schedule a request that was removed between scheduling
        # cycles, has its remove deferred for in-flight safety, is in OOM
        # backoff, or recently failed.
        exclude = self.failed_rids.union(self.pending_removes, self.admission.held_until)
        check_ready = self.admission.check_ready
        # only rank 0 can initiate scheduling: a followed node's head is
        # served by _try_schedule_tp_follow
        for spec in self._rows_here(self._ready_specs(exclude, target, exclude_target, lane)):
            node_partition = self.request_state.get_partition_for_node(
                spec.node_name
            )
            wg_id = self.runtime.get_worker_graph_id_for_node(
                spec.node_name, spec.graph_walk,
            )
            entries = None
            for rid in spec.rids:
                fwd_info = self.request_state.get_fwd_info(rid, node_partition)
                # check if the node is ready on the engine level
                # (e.g., for AR, whether the kv cache is read in)
                if not check_ready(spec.node_name, rid, fwd_info):
                    continue
                if entries is None:
                    entries = ready.setdefault(self._key(spec.node_name, spec.graph_walk), [])
                entries.append(ReadyNodeEntry(rid, wg_id, spec.graph_walk))
        return ready

    def _key(self, node_name: str, graph_walk: str) -> tuple[str, str]:
        """The scheduling key: the walk's combined walk if it has one."""
        return node_name, self.combined_walk_of.get((node_name, graph_walk), graph_walk)

    def label_for(
        self, node_name: str, request_walks: dict[int, str],
    ) -> tuple[str, dict[int, str]]:
        """A batch's walk label and ``request_walks`` for rows with these real
        walks. The batch takes ``request_walks`` itself, not a copy."""
        walks = set(request_walks.values())
        if len(walks) == 1:
            return walks.pop(), {}
        return self._key(node_name, next(iter(walks)))[1], request_walks

    def _ready_specs(
        self, exclude: set[int],
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
        lane: int = CHAIN_LANE,
    ):
        """``runtime.get_ready_nodes`` for scheduling keys, which the runtime
        matches against both real and combined walks, of ``lane`` only."""
        specs = self.runtime.get_ready_nodes(
            exclude, target=target, exclude_target=exclude_target,
        )
        if self.lane_of or lane != CHAIN_LANE:
            specs = [spec for spec in specs if self._lane((spec.node_name, spec.graph_walk)) == lane]
        return specs

    def _rows_here(
        self, specs: Iterable[ReadyNodeSpec],
        head_node: str | None = None, head_rids: Collection[int] = (),
    ) -> Iterator[ReadyNodeSpec]:
        """Of graph-ready ``specs``, the rows that are work this rank can take:
        every row of a node it leads, and of a node rank 0 leads only the rows
        of rank 0's FIFO head (``head_node``'s ``head_rids``), since such a node
        runs here only as rank 0 sends it."""
        leads = self.parallel_leader_nodes
        for spec in specs:
            if spec.node_name in leads:
                yield spec
            elif spec.node_name == head_node:
                rids = [rid for rid in spec.rids if rid in head_rids]
                if rids:
                    yield spec._replace(rids=rids)

    def _schedule_key(
        self, node_walk: tuple[str, str], entries: list[ReadyNodeEntry],
        skip_walk: str | None = None,
    ) -> ScheduledBatch | None:
        """One step for ``node_walk`` out of its backlog and its ready
        ``entries`` (already found ready), as the batch builder picks it.
        Under a combined walk, backlogged rows of ``skip_walk`` (a yield-away
        from that walk) stay parked."""
        backlogged = self.backlog.pop(node_walk, None)
        live = [] if backlogged is None else self._backlog_candidates(
            backlogged, skip_walk=skip_walk,
        )
        found = {entry.request_id for entry in entries}
        walk_of = None
        if node_walk in self._walks_of_key:
            walks = {entry.request_id: entry.graph_walk for entry in entries}
            walk_of = lambda rid: walks.get(rid) or backlogged.walk_of(rid)  # noqa: E731
        check = self._readiness(node_walk[0])
        pick = self.batch_builder.pick(self._step_candidates(
            node_walk, backlog=live,
            queued=[entry.request_id for entry in entries],
            walk_of=walk_of,
            ready=lambda rid: rid in found or check(rid),
        ))
        fresh = self._assemble_batch(*node_walk, entries) \
            if entries else None
        return self._build_step(node_walk, backlogged, fresh or None, pick)

    def _build_step(
        self, node_walk: tuple[str, str],
        backlogged: ScheduledBatch | None, fresh: ScheduledBatch | None,
        pick: StepPick, mid_prefill: Collection[MidPrefillRow] = (),
        pre_existing_batch_size: int = 0,
        pre_existing_seq_lens: Callable[[], Collection[int]] | None = None,
    ) -> ScheduledBatch | None:
        """The step ``pick`` chose out of ``backlogged``, ``fresh`` and the
        caller's ``mid_prefill`` rows, under the node's token budget if it has
        one. ``pre_existing_batch_size`` rows of a speculation chain run in it
        too; ``pre_existing_seq_lens`` estimates their spans, asked for only
        when the budget needs them (estimating walks every chain row, and a
        steady decode step usually brings no other row)."""
        engine = self.engine_manager.get_engine(node_walk[0])
        max_tokens = engine.get_max_batch_tokens(*node_walk)
        row_tokens: dict[int, int] = {}
        chunkable: set[int] = set()
        started: dict[int, int] = {}
        # rid -> (start, total) of a popped or backlogged row mid-prefill
        resumed: dict[int, tuple[int, int]] = {}
        continuing, starts = self._mid_prefill_rows(
            node_walk, mid_prefill, row_tokens, chunkable, started,
        )
        if backlogged is None and not fresh and continuing is None:
            return None
        # only for the chunk sizing below, never to gate admission, so a walk
        # with no budget skips it: measuring a row costs an input-tensor build
        # and a cached-prefix probe (memoized for a backlogged row, `_row_lens`)
        if max_tokens is not None:
            taken = set(pick.taken)
            for batch in (backlogged, fresh):
                if batch:
                    tokens, chunks = self._measure_rows(batch, started, resumed, taken)
                    row_tokens.update(tokens)
                    chunkable |= chunks
        min_chunk = 1
        if max_tokens is not None and row_tokens:
            min_chunk = engine.min_chunk_tokens(max_tokens)
            seq_lens = pre_existing_seq_lens() if pre_existing_seq_lens is not None else ()
            max_tokens -= _pre_existing_tokens(seq_lens, pre_existing_batch_size)
        budgeted = max_tokens is not None and bool(row_tokens)
        return self._build_and_schedule(BatchBuildRequest(
            node_name=node_walk[0], graph_walk=node_walk[1],
            backlog=backlogged, fresh=fresh, pick=pick, continuing=continuing,
            max_batch_tokens=max_tokens if budgeted else None,
            walk_max_tokens=engine.walk_token_caps(*node_walk) if budgeted else None,
            row_tokens=row_tokens,
            chunkable=chunkable,
            min_chunk_tokens=min_chunk,
            start_order=started,
        ), starts, resumed)

    def _mid_prefill_rows(
        self, node_walk: tuple[str, str], rows: Collection[MidPrefillRow],
        row_tokens: dict[int, int], chunkable: set[int], started: dict[int, int],
    ) -> tuple[ScheduledBatch | None, dict[int, tuple[int, int]]]:
        """``rows`` as a batch to build from, measured from where each next
        chunk starts; not popped, since their nodes are in flight. Returns the
        batch, or None for no row, and rid -> (its next chunk's start, its
        prompt's length)."""
        node = node_walk[0]
        rows = [row for row in rows if row.start < row.total]
        if not rows:
            return None, {}
        batch = ScheduledBatch(
            node_name=node, graph_walk=node_walk[1],
            request_to_worker_graph={row.request_id: row.worker_graph_id for row in rows},
        )
        if node_walk in self._walks_of_key:
            batch.request_walks = {row.request_id: row.graph_walk for row in rows}
        for row in rows:
            rid = row.request_id
            row_tokens[rid] = row.total - row.start
            started[rid] = self.chunk_started.get((rid, node), 0)
            chunkable.add(rid)
        return batch, {row.request_id: (row.start, row.total) for row in rows}


    def pick_chain_step(
        self, target: tuple[str, str], continuing: Sequence[int],
        mid_prefill: Collection[MidPrefillRow] = (), next_iter_of: str | None = None,
    ) -> "ChainPick":
        """Who runs the step a speculation chain speculates into at
        ``target``: its ``continuing`` rows (ready, in step order), the rows
        backlogged there and the ones on ``target``'s queues, as the batch
        builder picks them. Nothing is popped; ``take_chain_step`` takes the
        backlogged and queued rows it chose once the chain's rows are prepped.

        The chain only ever continues its own rows, so this is the one place
        a row waiting at the chain's target can get the step while the chain
        runs. Unless the chain mixes walks (`speculation_mixes`), only rows of
        its own walk are in it; another walk's rows make it yield instead.

        ``mid_prefill``: the chain's step's rows whose chunk is in flight.
        Their nodes are off the ready queue until it lands, so without them
        the next chunk would wait a step. They wait first in line, ahead of
        the backlog, and are never popped.

        ``next_iter_of``: the loop the chain's step runs its rows one
        iteration on in, if it does; their capture group is the one they have
        there (``_chain_capture_group``).
        """
        self.admission.let_rows_back()
        key = self._key(*target)
        own_walk = None if self.speculation_mixes(*target) else target[1]
        backlogged = self.backlog.get(key)
        live = [] if backlogged is None else self._backlog_candidates(
            backlogged, only_walk=own_walk,
        )
        mid = [
            row for row in mid_prefill
            if (own_walk is None or row.graph_walk == own_walk)
            and self._key(target[0], row.graph_walk) == key
            and row.start < row.total
        ] if mid_prefill else []
        if mid:
            live = [row.request_id for row in mid] + live
        specs = []
        # a chain holding every request here has nothing waiting on the queues
        if len(self.request_state.per_request_info) > len(continuing) + len(mid):
            exclude = self.failed_rids.union(self.pending_removes, self.admission.held_until)
            specs = [
                spec for spec in self._rows_here(self._ready_specs(
                    exclude, target if own_walk is not None else key, None,
                ))
                if spec.rids
            ]
        queued = specs[0].rids if len(specs) == 1 else [rid for spec in specs for rid in spec.rids]
        walk_of = None
        if key in self._walks_of_key:
            walks = {rid: spec.graph_walk for spec in specs for rid in spec.rids}
            walks.update((row.request_id, row.graph_walk) for row in mid)
            walk_of = lambda rid: walks.get(rid) or backlogged.walk_of(rid)  # noqa: E731
        ready = self._readiness(target[0])
        if mid:
            check, in_flight = ready, {row.request_id for row in mid}
            ready = lambda rid: rid in in_flight or check(rid)  # noqa: E731
        pick = self.batch_builder.pick(self._step_candidates(
            key, backlog=live, queued=queued, walk_of=walk_of,
            ready=ready,
            continuing=continuing, own_walk=target[1],
            one_walk=own_walk, next_iter_of=next_iter_of,
        ))
        if mid:
            took = set(pick.taken)
            mid = [row for row in mid if row.request_id in took]
        return ChainPick(key, pick, specs, mid)

    def take_chain_step(
        self, chain: "ChainPick",
        pre_existing_batch_size: int = 0,
        pre_existing_seq_lens: Callable[[], Collection[int]] | None = None,
    ) -> ScheduledBatch | None:
        """The backlogged and queued rows ``chain`` chose, popped as one batch:
        the rows that join the chain's step, or the step itself when the
        chain gave all of it away. Queued rows it held back (another capture
        group, a walk the step can't mix) are popped and parked, so the chain
        yields to them (``backlog_splits_from``) rather than run past them.
        The chain's rows mid-prefill that it took join from their held inputs;
        the chain's own ``pre_existing_batch_size`` rows count against a token
        budget (``_build_step``)."""
        key = chain.key
        own = set(chain.pick.continuing)
        taken = [rid for rid in chain.pick.taken if rid not in own]
        if not taken and not chain.pick.held_back:
            return None
        backlogged = self.backlog.get(key)
        parked = set() if backlogged is None else backlogged.request_to_worker_graph.keys()
        wanted = {rid for rid in (*taken, *chain.pick.held_back) if rid not in parked}
        entries = []
        if wanted:
            for spec in chain.specs:
                wg_id = self.runtime.get_worker_graph_id_for_node(spec.node_name, spec.graph_walk)
                entries.extend(
                    ReadyNodeEntry(rid, wg_id, spec.graph_walk)
                    for rid in spec.rids if rid in wanted
                )
        if backlogged is not None and not parked.isdisjoint(taken):
            self.backlog.pop(key)
        else:
            backlogged = None
        fresh = self._assemble_batch(*key, entries) if entries else None
        return self._build_step(
            key, backlogged, fresh or None, chain.pick,
            chain.mid_prefill, pre_existing_batch_size, pre_existing_seq_lens,
        )

    def _step_candidates(
        self, key: tuple[str, str], *,
        backlog: Sequence[int], queued: Sequence[int],
        walk_of: Callable[[int], str] | None, ready: Callable[[int], bool],
        continuing: Sequence[int] = (), own_walk: str | None = None,
        one_walk: str | None = None, next_iter_of: str | None = None,
    ) -> StepCandidates:
        node, walk = key
        if key not in self._walks_of_key:
            one_walk = walk
        engine = self.engine_manager.get_engine(node)
        capture_group = None
        if engine.splits_by_capture_key(node, walk):
            capture_group = self._chain_capture_group(node, walk, continuing, next_iter_of)
        return StepCandidates(
            graph_walk=walk,
            continuing=continuing, own_walk=own_walk,
            backlog=backlog, queued=queued,
            walk_of=walk_of or (lambda _: walk),
            one_walk=one_walk,
            ready=ready,
            capture_group=capture_group,
            walk_caps=self._walk_caps(key),
            walk_row_caps=self._walk_row_caps(key),
            line_order=functools.partial(self._line_order, key),
        )

    def _readiness(
        self, node_name: str,
    ) -> Callable[[int], bool]:
        """`Admission.check_ready` for one waiting row of ``node_name``."""
        partition = self.request_state.get_partition_for_node(node_name)
        check_ready = self.admission.check_ready
        return lambda rid: check_ready(
            node_name, rid, self.request_state.get_fwd_info(rid, partition),
        )

    def _capture_group(
        self, node_name: str, graph_walk: str, rid: int, next_iter_of: str | None = None,
    ) -> Any | None:
        """``rid``'s capture group at ``graph_walk``; with ``next_iter_of``, at
        the next iteration of that loop."""
        engine = self.engine_manager.get_engine(node_name)
        if not engine.splits_by_capture_key(node_name, graph_walk):
            return None
        fwd_info = self.request_state.get_fwd_info(
            rid, self.request_state.get_partition_for_node(node_name),
        )
        # a streamed input's chunk can decide the capture (a codec's window size)
        chunks = self.request_state.ingested_stream_chunks(rid, node_name)
        if next_iter_of is None:
            metadata = InputMetadata(stream_chunks=chunks) if chunks else None
        else:
            loops = dict(fwd_info.dynamic_loop_iter_counts)
            loops[next_iter_of] = loops.get(next_iter_of, 0) + 1
            metadata = InputMetadata(stream_chunks=chunks or {}, dynamic_loop_iter_counts=loops)
        return engine.capture_group(node_name, graph_walk, rid, fwd_info, input_metadata=metadata)

    def _chain_capture_group(
        self, node_name: str, graph_walk: str,
        chain: Collection[int], next_iter_of: str | None,
    ) -> Callable[[int], Any]:
        """``_capture_group`` for the rows of a speculation chain's step. The
        chain's own rows run it one iteration on in ``next_iter_of`` (as the
        step's input metadata has them), while their stored loop counters are
        still the step in flight's; a group that changes with the iteration
        (a step schedule's dtype) is the one they'll run in. Other rows run
        the step where they stand."""
        if next_iter_of is None or not chain:
            return functools.partial(self._capture_group, node_name, graph_walk)
        chain = frozenset(chain)
        return lambda rid: self._capture_group(
            node_name, graph_walk, rid, next_iter_of if rid in chain else None,
        )

    def backlog_splits_from(
        self, target: tuple[str, str], rid: int, next_iter_of: str | None = None,
    ) -> bool:
        """Whether ``target``'s backlog holds a live rid a speculation chain
        anchored at ``rid`` would never merge (``chain_must_yield``), so the
        chain has to yield for it to run. ``next_iter_of`` as in
        ``pick_chain_step``.
        """
        target = self._key(*target)
        live = self._live_backlog(target)
        if not live:
            return False
        return self.batch_builder.chain_must_yield(
            self.backlog[target], live, rid,
            self._chain_capture_group(*target, (rid,), next_iter_of),
        )

    def _live_backlog(self, node_walk: tuple[str, str]) -> set[int]:
        """``node_walk``'s backlogged rids that are not failed, removed or held."""
        waiting = self.backlog.get(node_walk)
        if waiting is None:
            return set()
        now = time.monotonic()
        held = self.admission.held_until
        return {
            r for r in waiting.request_to_worker_graph
            if r not in self.failed_rids and r not in self.pending_removes
            and held.get(r, 0.0) <= now
        }

    def _row_lens(
        self, batch: ScheduledBatch,
        rids: list[int],
    ) -> dict[int, Any] | None:
        """``rids``' lengths, for submodules that report them before prepare;
        measured once per row and kept on ``batch``. None if not reported."""
        engine = self.engine_manager.get_engine(batch.node_name)
        if self.get_tensor is None or not engine.declares_input_sequence_len(batch.node_name):
            return None
        unmeasured = [rid for rid in rids if rid not in batch.row_lens]
        if unmeasured:
            partition = self.request_state.get_partition_for_node(batch.node_name)
            inputs = batch.input_edges.to_input_tensors(
                self.get_tensor, unmeasured, skip_missing=True,
            ).by_rid
            for rid in unmeasured:
                batch.row_lens[rid] = engine.input_sequence_len(
                    batch.node_name, batch.walk_of(rid),
                    self.request_state.get_fwd_info(rid, partition), inputs.get(rid, {}),
                    rid=rid,
                )
        return batch.row_lens

    def _measure_rows(
        self, batch: ScheduledBatch,
        started: dict[int, int], resumed: dict[int, tuple[int, int]],
        only: Collection[int],
    ) -> tuple[dict[int, int], set[int]]:
        """For a node that reports row lengths: the remaining tokens of each
        of ``batch``'s rows in ``only`` (the step's), and the ones that may be
        chunked. A row mid-prefill goes in ``started`` with its start order and
        in ``resumed`` with where its chunk Loop stands
        (``_chunk_loop_progress``)."""
        node = batch.node_name
        engine = self.engine_manager.get_engine(node)
        rids = [rid for rid in batch.request_to_worker_graph if rid in only]
        progress = self._chunk_loop_progress(batch, rids)
        lens = self._row_lens(
            batch, [rid for rid in rids if rid not in progress],
        )
        if lens is None:
            return {}, set()
        tokens: dict[int, int] = {}
        chunkable: set[int] = set()
        for rid in rids:
            walk = batch.walk_of(rid)
            at = progress.get(rid)
            if at is not None:
                tokens[rid] = at[1] - at[0]
                started[rid] = self.chunk_started.get((rid, node), 0)
                resumed[rid] = at
            else:
                info = lens[rid]
                if info is None:
                    continue
                if info.cached_prefix:
                    self.cached_prefix[(rid, node)] = info.cached_prefix
                tokens[rid] = info.seq_len
            if engine.supports_chunked_prefill(node, walk):
                chunkable.add(rid)
        return tokens, chunkable

    def _chunk_loop_progress(
        self, batch: ScheduledBatch, rids: list[int],
    ) -> dict[int, tuple[int, int]]:
        """rid -> (tokens run, the prompt's length) for the rows of ``rids``
        mid-prefill, read off their chunk Loops. Nothing of theirs is in
        flight: a row whose chunk runs is off the ready queue and out of the
        backlog until the chunk lands, when the Loop has counted it."""
        node = batch.node_name
        walks = self.chunk_loop_walks.get(node)
        if not walks:
            return {}
        rows = [rid for rid in rids if batch.walk_of(rid) in walks]
        if not rows:
            return {}
        row_walks = [batch.walk_of(rid) for rid in rows]
        counts, caps = self.runtime.loop_progress(
            node, rows, row_walks, [chunk_loop_name(node, walk) for walk in row_walks],
        )
        return {
            rid: (count, cap)
            for rid, count, cap in zip(rows, counts, caps, strict=True) if count > 0
        }

    def _backlog_candidates(
        self, batch: ScheduledBatch,
        only_walk: str | None = None, skip_walk: str | None = None,
    ) -> list[int]:
        """The rows of a backlogged ``batch`` a step could take, in backlog
        order; the rest stay parked without asking the engine. Rows of a walk
        other than ``only_walk``, or of ``skip_walk``, are parked."""
        # Held (OOM backoff) and pending-remove rows wait without asking the
        # engine, as on the fresh path: asking reloads an offloaded row, which
        # would OOM it again straight away.
        now = time.monotonic()
        held, removing, failed = self.admission.held_until, self.pending_removes, self.failed_rids
        rows = [
            rid for rid in batch.request_to_worker_graph
            if rid not in removing and rid not in failed and held.get(rid, 0.0) <= now
        ]
        if batch.request_walks and (only_walk or skip_walk) is not None:
            rows = [
                rid for rid in rows
                if (only_walk is None or batch.walk_of(rid) == only_walk)
                and batch.walk_of(rid) != skip_walk
            ]
        return rows

    def _build_and_schedule(
        self, request: BatchBuildRequest,
        mid_prefill_starts: dict[int, tuple[int, int]] | None = None,
        resumed: dict[int, tuple[int, int]] | None = None,
    ) -> ScheduledBatch | None:
        """Have the batch builder split what was popped for ``request``'s
        (node, walk) by its pick, and carry out its verdict.

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
        if request.continuing is not None:
            expected |= request.continuing.request_to_worker_graph.keys()
        result: BatchBuildResult = self.batch_builder.build_batch(request)
        self._refuse_lost_rows(node_walk, expected, result)
        backlog, returned = result.backlog, result.returned
        if mid_prefill_starts:
            # left out, a row mid-prefill goes back to its queue when its chunk lands
            if backlog is not None:
                backlog, _ = backlog.split_off_first(None, exclude_rids=set(mid_prefill_starts))
            if returned:
                returned = {rid: wg for rid, wg in returned.items() if rid not in mid_prefill_starts}
        # Whatever was popped past this step has to be remembered here, or it
        # would never run.
        if backlog is not None:
            self._backlog(node_walk, backlog)
        if returned:
            self._set_in_flight(request.node_name, returned, False)
            self.runtime.push_back_node(
                request.node_name, list(returned), list(returned.values()),
            )
        if result.scheduled is not None:
            self._mark_scheduled(*node_walk, len(result.scheduled))
            self._assign_chunks(
                result.scheduled, result.chunk_tokens or {}, request.row_tokens,
                {**(resumed or {}), **(mid_prefill_starts or {})},
            )
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

    def _assign_chunks(
        self, batch: ScheduledBatch, chunk_tokens: dict[int, int],
        row_tokens: dict[int, int], resumed: dict[int, tuple[int, int]],
    ) -> None:
        """Stamp the ``ChunkRange`` each chunked row runs this step.

        A row mid-prefill resumes at ``resumed``'s (start, total): where its
        chunk Loop stands, or for a speculation the end of the chunk in
        flight. A row first cut here starts past its cached prefix and caps
        its chunk Loop at its whole input's length, which is what ends it.

        A row whose walk has no token budget is never measured, so it gets no
        range at all: it runs whole, and its stride ends its chunk Loop
        (``chunk_loop_stride``).
        """
        if not (chunk_tokens or resumed or self.cached_prefix):
            return
        node = batch.node_name
        for rid in batch.request_to_worker_graph:
            at = resumed.get(rid)
            if at is not None:
                start, total = at
                end = start + chunk_tokens.get(rid, total - start)
                batch.chunk_ranges[rid] = ChunkRange(start, end, total, end - start)
            else:
                key = (rid, node)
                cached = self.cached_prefix.pop(key, 0)
                if rid not in chunk_tokens:
                    continue
                total = cached + row_tokens[rid]
                end = cached + chunk_tokens[rid]
                batch.chunk_ranges[rid] = ChunkRange(cached, end, total, end)
                self.chunk_started[key] = next(self._chunk_starts)
            if end < total:
                batch.incomplete_node_rids.add(rid)
            else:
                self.chunk_started.pop((rid, node), None)
        cap_chunk_loops(batch, self.runtime, self.chunk_loop_walks)

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

    def _mark_scheduled(
        self, node_name: str, graph_walk: str, num_requests: int,
    ) -> None:
        """Record that this (node, walk) just ran, for round-robin ordering."""
        logger.debug(
            "MicroScheduler scheduling node %s with graph walk %s for %d requests",
            node_name, graph_walk, num_requests,
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(node_name, graph_walk)] = self.batch_number

    def _drop_backlogged_rid(self, rid: int) -> None:
        """Take a request out of anything still queued for it.

        Its node was popped off the ready queue when the batch was assembled,
        so this is the only place holding it.
        """
        for batch in self.backlog.values():
            # Membership, not a falsy pop: a worker graph id of 0 is real.
            if rid not in batch.request_to_worker_graph:
                continue
            batch.discard_rid(rid)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        self.backlog = {
            k: v for k, v in self.backlog.items() if len(v) > 0
        }

    def _walk_caps(self, key: tuple[str, str]) -> dict[str, int | None]:
        """``BatchBuildRequest.walk_caps`` for ``key``: its walk's cap, or a
        combined key's walks' caps and its own under its label, each lowered
        by any limit from a refused admit. The engine's caps are cached unless
        one varies (see `_engine_cap`); the limits come and go, so they are
        applied on every call."""
        caps = self._walk_caps_of_key.get(key)
        if caps is None:
            walks = self._walks_of_key.get(key, ())
            caps = {walk: self._engine_cap(key[0], walk) for walk in walks}
            caps[key[1]] = self._engine_cap(*key)
            if not any(self._varies(key[0], walk) for walk in caps):
                self._walk_caps_of_key[key] = caps
        if not self.admission.has_caps():
            return caps
        return {walk: self.admission.cap(key[0], walk, cap) for walk, cap in caps.items()}

    def _walk_row_caps(self, key: tuple[str, str]) -> dict[str, int]:
        """``BatchBuildRequest.walk_row_caps`` for ``key``: empty unless it is
        a combined walk. Fixed by the model, so asked once."""
        caps = self._row_caps_of_key.get(key)
        if caps is None:
            engine = self.engine_manager.get_engine(key[0])
            caps = {}
            for walk in self._walks_of_key.get(key, ()):
                cap = engine.get_max_rows_in_combined(key[0], walk)
                if cap is not None:
                    caps[walk] = cap
            self._row_caps_of_key[key] = caps
        return caps

    def _max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one, lowered by
        any limit from a refused admit (see `limit_batch`)."""
        return self.admission.cap(node_name, graph_walk, self._engine_cap(node_name, graph_walk))

    def _engine_cap(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one. A combined
        walk's cap is the model's to declare, via its captures or max_batch_size.
        Fixed once the graphs are captured, so asked once, unless
        ``NodeSubmodule.batch_cap_varies`` says it moves: a stale cap overfills
        the step."""
        key = (node_name, graph_walk)
        if key in self._max_bs:
            return self._max_bs[key]
        cap = self.engine_manager.get_engine(node_name).get_max_batch_size(node_name, graph_walk)
        if not self._varies(node_name, graph_walk):
            self._max_bs[key] = cap
        return cap

    def _varies(self, node_name: str, graph_walk: str) -> bool:
        key = (node_name, graph_walk)
        varies = self._cap_varies.get(key)
        if varies is None:
            engine = self.engine_manager.get_engine(node_name)
            varies = self._cap_varies[key] = engine.batch_cap_varies(node_name, graph_walk)
        return varies

    def _assemble_batch(
        self,
        node_name: str,
        graph_walk: str,
        entries: list[ReadyNodeEntry],
    ) -> ScheduledBatch | None:
        combined = (node_name, graph_walk) in self._walks_of_key
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


    def _backlog_has_schedulable(
        self, exclude_target: tuple[str, str] | None,
        exclude_walk: str | None = None,
    ) -> bool:
        """Whether a backlogged chunk outside ``exclude_target`` holds a rid
        still worth running; with ``exclude_walk``, only that walk's rows
        under ``exclude_target`` are outside it."""
        now = time.monotonic()
        held = self.admission.held_until
        for node_walk, batch in self.backlog.items():
            if self._lane(node_walk) != CHAIN_LANE:
                continue
            only_other = node_walk == exclude_target and exclude_walk is not None
            if node_walk == exclude_target and not only_other:
                continue
            if only_other and (
                set(batch.request_walks.values()) if batch.request_walks else {batch.graph_walk}
            ) == {exclude_walk}:
                continue  # every row is of the excluded walk
            for rid in batch.request_to_worker_graph:
                if only_other and batch.walk_of(rid) == exclude_walk:
                    continue
                if rid in self.failed_rids or rid in self.pending_removes:
                    continue
                if held.get(rid, 0.0) > now:
                    continue
                return True
        return False

    def has_ready_excluding(
        self,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: any worker-graph queue ready with a (node, walk) other
        than `exclude_target`? Used by the speculation path to decide whether
        breaking the spec chain for fairness is actually warranted on this
        worker — on single-walk workers (e.g. Orpheus LLM) the answer is
        always False, so speculation can run every iter.

        Does NOT pop or modify queue state. Mirrors the ready-scan in
        next_batch but stops at the first match. Its engine readiness
        checks are real ones, though: a resource may act on them (a LoRA pool
        claims a slot for the row it calls ready, held until that row's admit
        or removal), as it would for a row the scan schedules.

        Backlogged chunks count too: their graph nodes came off the queues
        when they were assembled, so the scan below cannot see them. A chunk
        under `exclude_target` is left out like any other — the speculative
        merge takes that one before its own rids, so it needs no yield.

        Only `exclude_target`'s own real walk is excluded: a ready row of
        another walk under the same combined walk is another target, which a
        speculation chain on that walk never merges, so it counts. A chain that
        mixes (`speculation_mixes`) merges those, so its whole combined walk is
        excluded.
        """
        mixes = exclude_target is not None and self.speculation_mixes(*exclude_target)
        exclude_walk = exclude_target[1] if exclude_target is not None and not mixes else None
        # the runtime scan below takes the real walk as given, or the combined one
        scan_exclude = exclude_target
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
        if mixes:
            scan_exclude = exclude_target
        if self._backlog_has_schedulable(exclude_target, exclude_walk):
            return True

        # A failed rid is normally invisible here — next_batch refuses to
        # schedule it, so reporting it as ready would break the spec chain for
        # work that never materializes. The exception is a rid sitting in the
        # head TP follow batch: next_batch *will* schedule that one (rank 0
        # is waiting on it), so it counts as real ready work.
        tp_pend_rids: set[int] = set()
        head_node: str | None = None
        pend = self.lockstep.follow_head()
        if pend is not None:
            if self._key(pend.node_name, pend.graph_walk) != exclude_target:
                tp_pend_rids = set(self.lockstep.tp_rids(pend))
                head_node = pend.node_name
        # Don't bother expiring holds here: we only read them, and the next
        # next_batch call will refresh.
        now = time.monotonic()
        exclude = {
            rid for rid in self.failed_rids if rid not in tp_pend_rids
        } | {
            rid for rid, t in self.admission.held_until.items() if t > now
        }

        # Graph readiness is necessary but not sufficient, so a False here is
        # final and skips the engine pass.
        if not self.runtime.has_ready_excluding(exclude, scan_exclude):
            return False
        for spec in self._rows_here(self._ready_specs(exclude, None, scan_exclude), head_node, tp_pend_rids):
            node_partition = self.request_state.get_partition_for_node(
                spec.node_name
            )
            for rid in spec.rids:
                fwd_info = self.request_state.get_fwd_info(
                    rid, node_partition
                )
                if self.admission.check_ready(spec.node_name, rid, fwd_info):
                    return True
        return False

    def pending_rows(self) -> Counter[str]:
        """Per node this rank schedules, rows ready for a step but not in one:
        its backlog, and graph-ready rows still on the queues. Read-only and
        graph-level (no engine readiness check, which can reload), so a held
        or not yet reloaded row counts; a failed or removed one does not."""
        exclude = self.failed_rids.union(self.pending_removes)
        rows: Counter[str] = Counter()
        for (node, _), waiting in self.backlog.items():
            rows[node] += sum(1 for rid in waiting.request_to_worker_graph if rid not in exclude)
        for spec in self._rows_here(self.runtime.get_ready_nodes(exclude)):
            rows[spec.node_name] += len(spec.rids)
        return rows

    def fail_rids(self, rids: set[int]) -> None:
        """Stop scheduling new work for requests reported to the conductor as
        failed. Cleared by ``clear_rid`` when the removal comes back."""
        self.failed_rids.update(rids)
        for rid in rids:
            self._drop_backlogged_rid(rid)

    def clear_rid(self, rid: int) -> None:
        """Forget all per-request scheduler state; called on REMOVE_REQUEST."""
        self.failed_rids.discard(rid)
        for key in [key for key in self.chunk_started if key[0] == rid]:
            del self.chunk_started[key]
        for key in [key for key in self.cached_prefix if key[0] == rid]:
            del self.cached_prefix[key]
        for turns in self.last_ran.values():
            turns.pop(rid, None)
        self.arrival.pop(rid, None)
        self._drop_backlogged_rid(rid)
