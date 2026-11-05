"""Every rank of a TP instance lands a dropped request's rebuild between the same two of rank 0's steps.

Under async scheduling rank 0 builds step N+1 while N runs. A rebuild that runs its first chunk in N runs its next
one in N+1 as one of N's rows mid-prefill, and rank 0 marks it as a rebuild row from its log. A follower that runs
that chunk as a prompt chunk never lands the rebuild, and from then on holds one dropped request more than rank 0.
Under MSTAR_KV_DEBUG_ASSERTS the follower's page trail then differs from rank 0's on the next step, and the follower
stops.

Two ranks with real KV pools and rebuild logs drop v, rebuild it over N and N+1, and the follower checks rank 0's
trail on the step after. The follower builds N+1 with the real `Worker._try_follow_speculation`, or with
`MicroScheduler._try_schedule_tp_follow` when rank 0 sends N+1 after N, and both ranks post-process with the real
`Worker._finish_rebuild_rows`.
"""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest
import torch
from test_resident_delta import NODE, PAGE_SIZE, _both_short_then_drop, _counts, manager_mod
from test_worker_rebuild import POLICY, PREFILL, _graphs, _kept, _record

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Residency
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.host_pool import HostPool
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.rebuild import KeptIds, LayoutRow, Rebuilds
from mstar.worker.speculation import SpeculationBuilder, SpeculationContext
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

# v's rebuild writes its 5-page prompt: three pages in N and the rest in N+1
FIRST, WHOLE = 3 * PAGE_SIZE, 5 * PAGE_SIZE
CHUNKS = {1: ChunkRange(0, FIRST, WHOLE, FIRST), 2: ChunkRange(FIRST, WHOLE, WHOLE, WHOLE - FIRST)}


class _Rank:
    """One rank: its engine over a real pool, a real rebuild log keeping v's ids, and a scheduler and worker over
    them with stubs for the graph runtime."""

    def __init__(self, kv, engine):
        self.kv, self.engine = kv, engine
        self.rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 12, torch.device("cpu")))
        self.rebuilds._kept[NODE]["v"] = _kept(0, prompt=[0] * WHOLE)
        self.scheduler = MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda node: engine))
        self.scheduler.rebuilds = self.rebuilds
        self.scheduler.runtime = SimpleNamespace(get_worker_graph_id_for_node=lambda node, walk: 0)
        w = self.worker = Worker.__new__(Worker)
        w._rebuilds, w.scheduler, w.enable_nvtx = self.rebuilds, self.scheduler, False
        w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
        w.request_state = SimpleNamespace(
            get_partition_for_node=lambda node: "p",
            get_fwd_info=lambda rid, partition: CurrentForwardPassInfo(rid, "decode", 0, 0, 8),
        )
        w._wait_for_step = lambda batch_N: None
        w._preemption = SimpleNamespace(rebuilt=lambda rid, node, tokens: None)
        w._in_flight_rids, w._walk_drivers, w._walk_driven_edges = set(), {}, {}
        tensors = SimpleNamespace(get_tensor=lambda uuid: None, get_tensors=lambda uuids: [])
        w.streams = StreamBufferManager(SimpleNamespace(per_request_info={}), tensors, [])
        w._graph_runtime = SimpleNamespace(
            get_spec_target=lambda *args: SimpleNamespace(
                node_name=NODE, graph_walk=PREFILL, output_signals=[], is_new_loop_iter=False, loop_name=None,
            ),
            prep_follow_spec_rids=lambda prep: SpeculationPrepOutput([], list(prep.rids), [0] * len(prep.rids),
                                                                      ColumnarEdgeSpecs.empty()),
            get_consumed_edges=lambda node, spec_node, walk: frozenset(),
        )
        w._spec_builder = SpeculationBuilder(SpeculationContext(
            graph_runtime=w._graph_runtime, tensor_manager=tensors, request_state=w.request_state,
            scheduler=self.scheduler, streams=w.streams, engine_manager=w.engine_manager,
        ))

    def post_process(self, batch: ScheduledBatch) -> None:
        rows = list(batch.request_to_worker_graph)
        node_batch = SimpleNamespace(
            request_ids=rows, skipped_rids=set(), cached_rids=set(), per_request_info=dict.fromkeys(rows),
        )
        pending = SimpleNamespace(batch=batch, node_batch=node_batch, node_name=NODE)
        Worker._finish_rebuild_rows(self.worker, pending, {})


def _step(seq: int, chunks=CHUNKS) -> ScheduledBatch:
    """Rank 0's step ``seq``, running the chunk of v's rebuild its log offered."""
    chunk = chunks[seq]
    return ScheduledBatch(
        node_name=NODE, graph_walk=PREFILL, request_to_worker_graph={"v": 0}, rebuild_rids=frozenset({"v"}),
        chunk_ranges={"v": chunk}, incomplete_node_rids=set() if chunk.is_final else {"v"}, tp_seq=seq,
    )


def _head(lead, seq: int, rids=(), chunks=CHUNKS, **kwargs) -> ScheduleTPNode:
    """Rank 0's broadcast of step ``seq``: its moves and its page trail as of now."""
    chunk = chunks.get(seq)
    return ScheduleTPNode(
        NODE, PREFILL, list(rids), spec_seq=seq, resident_delta=lead.take_resident_delta(),
        page_trails=lead.page_trails(NODE),
        **({} if chunk is None else dict(
            chunk_starts=[chunk.start], chunk_ends=[chunk.end], chunk_totals=[chunk.total],
            chunk_strides=[chunk.stride],
        )),
        **kwargs,
    )


@pytest.mark.parametrize("speculated", [True, False], ids=["N+1 speculated from N", "N+1 sent after N"])
def test_a_rebuild_over_two_steps_lands_on_the_follower_with_rank_0s(monkeypatch, speculated):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    zero, one = _Rank(leader, lead), _Rank(follower, follow)
    sched, request_state = one.scheduler, one.worker.request_state

    # the step after the refusal carries v's drop
    sched.register_tp_follow(_head(lead, 0))
    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    sched.pop_tp_follow_head()

    # N: rank 0's log starts v's rebuild and offers its first chunk. The follower knows v from its own KV.
    zero.scheduler._start_rebuild(NODE, "v", "decode")
    sched.register_tp_follow(_head(lead, 1, ["v"]))
    n = sched._try_schedule_tp_follow(request_state)
    assert n.rebuild_rids == {"v"}

    if speculated:
        sched.register_tp_follow(_head(lead, 2, ["v"], speculative=True, spec_from_seq=1))
        pending = SimpleNamespace(tp_seq=1, node_name=NODE, graph_walk=PREFILL, batch=n, partition="p",
                                  node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)))
        n_1 = Worker._try_follow_speculation(one.worker, pending).scheduled_batch
        zero.post_process(_step(1))
        one.post_process(n)
    else:
        zero.post_process(_step(1))
        sched.register_tp_follow(_head(lead, 2, ["v"]))
        one.post_process(n)
        n_1 = sched._try_schedule_tp_follow(request_state)
    zero.post_process(_step(2))
    one.post_process(n_1)
    sched.register_tp_follow(_head(lead, 3))

    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    assert follower.residency("v") is Residency.RESIDENT and _counts(leader) == _counts(follower)
    assert not one.rebuilds.rebuilding("v", NODE), "the follower would take v's decode row for a rebuild row"


def test_a_rebuild_over_two_rows_lands_on_the_follower_with_rank_0s(monkeypatch):
    """v prefilled in two steps, so its log rebuilds it in two rows, one a step. Each rank runs them from its own
    log in N and N+1, and the follower lands v when it replays rank 0's land."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    zero, one = _Rank(leader, lead), _Rank(follower, follow)
    for rank in (zero, one):
        rank.rebuilds._kept[NODE]["v"] = KeptIds(
            (LayoutRow(PREFILL, FIRST), LayoutRow(PREFILL, WHOLE - FIRST)),
            [_record([0] * FIRST), _record([0] * (WHOLE - FIRST))], array("i", [0]),
        )
    rows = {1: ChunkRange(0, FIRST, FIRST, FIRST), 2: ChunkRange(0, WHOLE - FIRST, WHOLE - FIRST, WHOLE - FIRST)}
    sched, request_state = one.scheduler, one.worker.request_state
    sched.register_tp_follow(_head(lead, 0))
    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    sched.pop_tp_follow_head()

    zero.scheduler._start_rebuild(NODE, "v", "decode")
    for seq in (1, 2):
        sched.register_tp_follow(_head(lead, seq, ["v"], chunks=rows))
        step = sched._try_schedule_tp_follow(request_state)
        assert step.rebuild_rids == {"v"}, f"the follower took v's row {seq} for one of its own"
        zero.post_process(_step(seq, rows))
        one.post_process(step)
    sched.register_tp_follow(_head(lead, 3, chunks=rows))

    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    assert follower.residency("v") is Residency.RESIDENT and _counts(leader) == _counts(follower)
    assert not one.rebuilds.rebuilding("v", NODE), "the follower's rebuild of v outlived rank 0's"


def test_a_follower_whose_rebuild_cant_start_stops(monkeypatch):
    """The follower replays rank 0's drop of v holding the input of v's prompt and no ids, and its chunk loop has
    v's prompt run whole, so v's rebuild has nothing to start from. The error came back on every pass over rank 0's
    step while rank 0 waited in it."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    zero, one = _Rank(leader, lead), _Rank(follower, follow)
    del one.rebuilds._kept[NODE]["v"]
    follow._chunking.prepared["v", NODE] = SimpleNamespace()
    sched, request_state = one.scheduler, one.worker.request_state
    sched.register_tp_follow(_head(lead, 0))
    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    sched.pop_tp_follow_head()
    zero.scheduler._start_rebuild(NODE, "v", "decode")
    sched.register_tp_follow(_head(lead, 1, ["v"]))

    with pytest.raises(SystemExit, match="TP ranks diverged"):
        sched.settle_tp_follow_delta(request_state, caught_up=True)


def test_a_follower_starts_the_rebuild_of_a_request_busy_on_another_node(monkeypatch):
    """v's drop here waits only for a step of this node, so v can be in a step of another node as rank 0 starts its
    rebuild. A follower that skipped every request in flight on any node didn't start v's rebuild, and missed the keys
    the rebuild lays out before it took rank 0's lease."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    zero, one = _Rank(leader, lead), _Rank(follower, follow)
    sched, request_state = one.scheduler, one.worker.request_state
    sched.register_tp_follow(_head(lead, 0))
    assert sched.settle_tp_follow_delta(request_state, caught_up=True)
    sched.pop_tp_follow_head()
    zero.scheduler._start_rebuild(NODE, "v", "decode")
    sched.register_tp_follow(_head(lead, 1, ["v"]))
    sched.in_flight_rids.add("v")

    assert sched.settle_tp_follow_delta(request_state, caught_up=True)

    assert one.rebuilds.rebuilding("v", NODE), "the follower didn't start v's rebuild at rank 0's row of it"
