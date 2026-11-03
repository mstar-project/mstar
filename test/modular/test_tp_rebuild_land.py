"""Every rank of a TP instance lands a dropped request's rebuild between the same two of rank 0's steps.

Under async scheduling rank 0 builds step N+1 while N runs. A rebuild that runs its first chunk in N runs its next
one in N+1 as one of N's rows mid-prefill, and rank 0 marks it as a rebuild row from its log. A follower that runs
that chunk as a prompt chunk never lands the rebuild, and from then on holds one dropped request more than rank 0.
Under MSTAR_KV_DEBUG_ASSERTS the follower's page trail then differs from rank 0's on the next step, and the follower
stops.

Two ranks with real KV pools and rebuild logs drop v, rebuild it over N and N+1, and the follower checks rank 0's
trail on the step after. The follower builds N+1 with the real `Worker._try_follow_speculation`, and both ranks
post-process with the real `Worker._finish_rebuild_rows`.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from test_resident_delta import NODE, PAGE_SIZE, _both_short_then_drop, _counts, manager_mod
from test_worker_rebuild import POLICY, _graphs, _kept

from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Residency
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.host_pool import HostPool
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.rebuild import Rebuilds
from mstar.worker.speculation import Speculation
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

PREFILL = POLICY.walk
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
            get_partition_for_node=lambda node: "p", get_fwd_info=lambda rid, partition: SimpleNamespace(
                graph_walk="decode",
            ),
        )
        w._wait_for_step = lambda batch_N: None
        w._preemption = SimpleNamespace(rebuilt=lambda rid, node, tokens: None)
        w._in_flight_rids, w._walk_drivers, w._walk_driven_edges = set(), {}, {}
        tensors = SimpleNamespace(get_tensor=lambda uuid: None, get_tensors=lambda uuids: [])
        w.streams = StreamBufferManager(SimpleNamespace(per_request_info={}), tensors, [])
        w._graph_runtime = SimpleNamespace(
            get_spec_target=lambda *args: SimpleNamespace(),
            prep_follow_spec_rids=lambda prep: SpeculationPrepOutput([], list(prep.rids), [0] * len(prep.rids),
                                                                      ColumnarEdgeSpecs.empty()),
        )
        w._spec_builder = SimpleNamespace(assemble=_assemble)

    def post_process(self, batch: ScheduledBatch) -> None:
        rows = list(batch.request_to_worker_graph)
        node_batch = SimpleNamespace(
            request_ids=rows, skipped_rids=set(), cached_rids=set(), per_request_info=dict.fromkeys(rows),
        )
        pending = SimpleNamespace(batch=batch, node_batch=node_batch, node_name=NODE)
        Worker._finish_rebuild_rows(self.worker, pending, {})


def _assemble(pending, info, rows, inputs, streamed, continuing, fresh_final, *, chunk_ranges=None,
              incomplete_node_rids=None, rebuild_rids=frozenset(), **kwargs):
    """The scheduled half of the speculation the follower built, as `SpeculationBuilder.assemble` carries it."""
    return Speculation(
        scheduled_batch=ScheduledBatch(
            node_name=NODE, graph_walk=PREFILL, request_to_worker_graph=dict(rows), rebuild_rids=rebuild_rids,
            chunk_ranges=dict(chunk_ranges or {}), incomplete_node_rids=set(incomplete_node_rids or ()),
        ),
        node_batch=None, consumed_edges=set(), continuing_rids=set(), partition="p", is_new_iter=False,
        is_same_node=True,
    )


def _step(seq: int) -> ScheduledBatch:
    """Rank 0's step ``seq``, running the chunk of v's rebuild its log offered."""
    chunk = CHUNKS[seq]
    return ScheduledBatch(
        node_name=NODE, graph_walk=PREFILL, request_to_worker_graph={"v": 0}, rebuild_rids=frozenset({"v"}),
        chunk_ranges={"v": chunk}, incomplete_node_rids=set() if chunk.is_final else {"v"}, tp_seq=seq,
    )


def _head(lead, seq: int, rids=(), **kwargs) -> ScheduleTPNode:
    """Rank 0's broadcast of step ``seq``: its moves and its page trail as of now."""
    chunk = CHUNKS.get(seq)
    return ScheduleTPNode(
        NODE, PREFILL, list(rids), spec_seq=seq, resident_delta=lead.take_resident_delta(),
        page_trails=lead.page_trails(NODE),
        **({} if chunk is None else dict(
            chunk_starts=[chunk.start], chunk_ends=[chunk.end], chunk_totals=[chunk.total],
            chunk_strides=[chunk.stride],
        )),
        **kwargs,
    )


def test_a_rebuild_over_a_speculated_step_lands_on_the_follower_with_rank_0s(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    zero, one = _Rank(leader, lead), _Rank(follower, follow)
    sched = one.scheduler

    # the step after the refusal carries v's drop
    sched.register_tp_follow(_head(lead, 0))
    assert sched.settle_tp_follow_delta(caught_up=True)
    sched.pop_tp_follow_head()

    # N: rank 0's log starts v's rebuild and offers its first chunk. The follower knows v from its own KV.
    zero.scheduler._start_rebuild(NODE, "v", "decode")
    sched.register_tp_follow(_head(lead, 1, ["v"]))
    assert sched.settle_tp_follow_delta(caught_up=True)
    assert sched.follow_rebuilds(one.worker.request_state, NODE, ["v"]) == {"v"}
    sched.pop_tp_follow_head()
    n = _step(1)

    # N+1, speculated from N
    sched.register_tp_follow(_head(lead, 2, ["v"], speculative=True, spec_from_seq=1))
    pending = SimpleNamespace(tp_seq=1, node_name=NODE, graph_walk=PREFILL, batch=n,
                              node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)))
    followed = Worker._try_follow_speculation(one.worker, pending)
    assert followed is not None

    # each rank post-processes N, then N+1, and rank 0 sends the step after
    zero.post_process(_step(1))
    zero.post_process(_step(2))
    one.post_process(n)
    one.post_process(followed.scheduled_batch)
    sched.register_tp_follow(_head(lead, 3))

    assert sched.settle_tp_follow_delta(caught_up=True)
    assert follower.residency("v") is Residency.RESIDENT and _counts(leader) == _counts(follower)
    assert not one.rebuilds.rebuilding("v", NODE), "the follower would take v's decode row for a rebuild row"
