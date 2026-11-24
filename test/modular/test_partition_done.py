"""A partition ends on the worker's ``partition_done`` signal.

The worker sets ``partition_done`` on the worker-graphs-done report of the pass
that consumed a stream's final chunk, or whose stop check returned
``END_PARTITION`` (a prefill that sampled a stop token). The conductor must
treat that as the partition's ``request_done`` (so ``producer_done`` reaches
its downstream connections) whatever is left of its plan: a stream consumer's
step runs once per chunk and never ends on its own, and a prefill's plan still
has its decode step.
"""
from dataclasses import dataclass, field

import pytest

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
    WalkStep,
)
from mstar.utils.ipc_format import WorkerGraphsDone


@dataclass
class _Recorder:
    producer_done: list = field(default_factory=list)
    partition_inputs: list = field(default_factory=list)


def _conductor(rid="r"):
    c = object.__new__(Conductor)
    c._client_streams = {}
    rec = _Recorder()
    c._send_producer_done = lambda request_id, frm, to: rec.producer_done.append((request_id, frm, to))
    c._send_inputs = lambda request_id, pname, inputs, walk, step_metadata: rec.partition_inputs.append(pname)
    c._un_persist_tensors = lambda request_id, tensors: None
    c._set_partition_worker_graph_ids = lambda request_id, pname, walk: None
    c.producer_triggered_partitions = set()
    # A prefills, then decodes from the token it persists; B and C consume streams, one pass a chunk
    c._walk_inputs = {"a": (), "a_decode": (("text_inputs", "llm"),), "b": (), "c": ()}
    c._walk_persists = dict.fromkeys(c._walk_inputs, frozenset())
    c._repeating_walks = frozenset({"b", "c"})
    plans = {"A": [WalkStep("a"), WalkStep("a_decode", bind={"text_inputs": "new_token"})],
             "B": [WalkStep("b")], "C": [WalkStep("c")]}
    conns = {
        "A->B": StreamingConnectionState(from_partition="A", to_partition="B", edge_name="tok"),
        "B->C": StreamingConnectionState(from_partition="B", to_partition="C", edge_name="codes"),
    }
    req = RequestData(
        persist_signals={}, persist_signal_ref_cnt={}, worker_graph_to_workers={}, all_worker_graph_ids=set(),
        max_output_tokens=10_000, random_seed=0, resource_configs={},
        partition_states={p: PartitionState(partition_name=p, graph_walk=p.lower(), plan=plans[p]) for p in "ABC"},
        partition_definitions={p: PartitionDefinition(name=p, graph_walks={p.lower()}) for p in "ABC"},
        streaming_connections=conns,
    )
    c.requests = {rid: req}
    return c, req, rec


def _done_forward(c, req, partition, partition_done=False):
    """``_process_done_forward`` after a pass whose report carried ``partition_done``."""
    req.partition_states[partition].worker_reported_done = partition_done
    return c._process_done_forward("r", partition)


def test_worker_final_chunk_signal_ends_the_partition_and_propagates_producer_done():
    c, req, rec = _conductor()
    # B consumed the final chunk of A->B: B is done, and C must learn its producer is done
    all_done = _done_forward(c, req, "B", partition_done=True)
    assert req.partition_states["B"].is_done is True
    assert req.streaming_connections["B->C"].producer_done is True
    assert rec.producer_done == [("r", "B", "C")]
    assert all_done is False          # A and C are still running


def test_counter_heuristics_are_not_needed_and_an_ordinary_pass_does_not_end_it():
    c, req, rec = _conductor()
    # B finished a pass but did not consume the final chunk: it waits for more stream data
    all_done = _done_forward(c, req, "B")
    assert req.partition_states["B"].is_done is False
    assert req.streaming_connections["B->C"].producer_done is False
    assert rec.producer_done == [] and rec.partition_inputs == []
    assert all_done is False


@pytest.mark.parametrize("flag_on", ["first", "last"])
def test_the_flag_on_any_report_ends_a_partition_without_a_stream_in(flag_on):
    """A producer-only partition whose prefill sampled a stop token: the flag
    comes on the LLM's report, and the pass's other worker graph (the
    encoder's) can report before or after it."""
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["A"]
    pstate.current_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {0: ["enc"], 1: ["llm"]}
    reports = [
        WorkerGraphsDone(request_id="r", partition_name="A", worker_graph_ids=[wg], is_first_tp_rank=True,
                         partition_done=wg == 1)
        for wg in ((1, 0) if flag_on == "first" else (0, 1))
    ]
    assert c._process_worker_graphs_done(reports[0]) == []
    assert c._process_worker_graphs_done(reports[1]) == ["A"]
    c._process_done_forward("r", "A")
    assert pstate.is_done is True
    assert rec.partition_inputs == [], "no further walk starts"
    assert rec.producer_done == [("r", "A", "B")]
    assert pstate.worker_reported_done is False, "reset for the next pass"


def test_two_passes_reported_in_one_cycle_end_the_partition_once():
    """A self-triggering partition's next pass can report before the conductor
    advances the last one. Pass 0 closes with ``partition_done=False`` and
    advances to Pass 1, which closes with ``partition_done=True`` once."""
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["B"]
    pstate.current_worker_graph_ids = {0}
    req.worker_graph_to_workers = {0: ["w"]}
    queued = [
        name
        for pass_id, done in ((0, False), (1, True))
        for name in c._process_worker_graphs_done(WorkerGraphsDone(
            request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
            partition_done=done, worker_id="w", pass_ids=[pass_id],
        ))
    ]
    assert queued == ["B", "B"]
    assert c._process_done_forward("r", "B") is False
    assert pstate.is_done is False
    assert pstate.fwd_pass_number == 1
    assert c._process_done_forward("r", "B") is False
    assert pstate.is_done is True
    assert pstate.fwd_pass_number == 2
    assert rec.producer_done == [("r", "B", "C")]


@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("report_order", [(0, 1), (1, 0)], ids=["consumer_first", "successor_first"])
def test_gated_stream_pass_across_workers_keeps_its_final_flag(n_ranks, report_order):
    """A gated next pass cannot start until both worker graphs finish this one.

    The consumer may be on another worker and report the final chunk before or
    after its successor. Every rank must report before the conductor closes the
    pass; the final-stream flag then ends the partition exactly once.
    """
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["B"]
    pstate.current_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {
        0: [f"consumer_{rank}" for rank in range(n_ranks)],
        1: [f"successor_{rank}" for rank in range(n_ranks)],
    }

    reports = [
        WorkerGraphsDone(
            request_id="r", partition_name="B", worker_graph_ids=[wg],
            is_first_tp_rank=(rank == 0), partition_done=(wg == 0 and rank == 0),
        )
        for wg in report_order for rank in range(n_ranks)
    ]
    flag_seen = False
    for report in reports[:-1]:
        assert c._process_worker_graphs_done(report) == []
        flag_seen |= report.partition_done
        assert pstate.worker_reported_done == flag_seen

    assert c._process_worker_graphs_done(reports[-1]) == ["B"]
    assert c._process_done_forward("r", "B") is False
    assert pstate.is_done is True
    assert pstate.fwd_pass_number == 1
    assert rec.partition_inputs == []
    assert rec.producer_done == [("r", "B", "C")]


def test_chain_closes_partition_by_partition():
    c, req, rec = _conductor()
    req.partition_states["A"].plan = [WalkStep("a")]
    assert c._process_done_forward("r", "A") is False           # A ends itself: its plan has run
    assert req.streaming_connections["A->B"].producer_done is True
    assert _done_forward(c, req, "B", partition_done=True) is False
    assert _done_forward(c, req, "C", partition_done=True) is True   # every partition done
    assert rec.producer_done == [("r", "A", "B"), ("r", "B", "C")]


def test_a_finished_consumer_is_not_told_its_producer_is_done():
    """A consumer done from the start (e.g. Qwen3-Omni's Talker with no audio
    out) must not get producer_done: a continue_after_done stream would then
    hand it empty chunks to run on."""
    c, req, rec = _conductor()
    req.partition_states["C"].is_done = True
    _done_forward(c, req, "B", partition_done=True)
    assert req.streaming_connections["B->C"].producer_done is True
    assert rec.producer_done == []


def test_pipelined_passes_across_workers_keep_partition_done_per_pass():
    """Issue #554 reproducer: vocoder (WG 0) reports pass 0 (False) and pass 1
    (True) before post (WG 1) reports pass 0 (False) and pass 1 (False).
    Pass 0 must close without ending the partition; Pass 1 closes with done."""
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["B"]
    pstate.current_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {0: ["voc"], 1: ["post"]}

    reports = [
        WorkerGraphsDone(request_id="r", partition_name="B", worker_graph_ids=[0],
                         is_first_tp_rank=True, partition_done=False, worker_id="voc", pass_ids=[0]),
        WorkerGraphsDone(request_id="r", partition_name="B", worker_graph_ids=[0],
                         is_first_tp_rank=True, partition_done=True, worker_id="voc", pass_ids=[1]),
        WorkerGraphsDone(request_id="r", partition_name="B", worker_graph_ids=[1],
                         is_first_tp_rank=True, partition_done=False, worker_id="post", pass_ids=[0]),
        WorkerGraphsDone(request_id="r", partition_name="B", worker_graph_ids=[1],
                         is_first_tp_rank=True, partition_done=False, worker_id="post", pass_ids=[1]),
    ]
    assert c._process_worker_graphs_done(reports[0]) == []
    assert c._process_worker_graphs_done(reports[1]) == []
    assert not pstate.worker_reported_done, "Pass 1 final flag must not leak into Pass 0"

    assert c._process_worker_graphs_done(reports[2]) == ["B"]
    c._process_done_forward("r", "B")
    assert pstate.is_done is False
    assert pstate.fwd_pass_number == 1
    assert rec.producer_done == []

    assert c._process_worker_graphs_done(reports[3]) == ["B"]
    c._process_done_forward("r", "B")
    assert pstate.is_done is True
    assert pstate.fwd_pass_number == 2
    assert rec.producer_done == [("r", "B", "C")]


def test_tp2_duplicate_rank0_reports_cannot_close_pass_for_rank1():
    """In TP=2 across passes, rank 0 reporting pass 0 and pass 1 before rank 1
    reports pass 0 must not close pass 0 prematurely."""
    c, req, rec = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["B"]
    pstate.current_worker_graph_ids = {0}
    req.worker_graph_to_workers = {0: ["w0", "w1"]}

    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        partition_done=False, worker_id="w0", pass_ids=[0],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        partition_done=True, worker_id="w0", pass_ids=[1],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=False,
        partition_done=False, worker_id="w1", pass_ids=[0],
    )) == ["B"]
    c._process_done_forward("r", "B")
    assert pstate.is_done is False
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=False,
        partition_done=True, worker_id="w1", pass_ids=[1],
    )) == ["B"]
    c._process_done_forward("r", "B")
    assert pstate.is_done is True


def test_pipelined_persist_signals_do_not_clobber_earlier_pass():
    """Persist signals from Pass 1 arriving before Pass 0 closes must stay
    isolated until Pass 1's _process_done_forward applies them."""
    from mstar.utils.ipc_format import TensorPointerInfo

    c, req, _ = _conductor()
    c.enable_prof = False
    pstate = req.partition_states["B"]
    pstate.current_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {0: ["voc"], 1: ["post"]}
    t0 = TensorPointerInfo(
        dims=[1], dtype=None, nbytes=4, address=0, stride=[1],
        uuid="u0", source_session_id="s", source_entity="voc",
    )
    t1 = TensorPointerInfo(
        dims=[1], dtype=None, nbytes=4, address=0, stride=[1],
        uuid="u1", source_session_id="s", source_entity="voc",
    )
    c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        worker_id="voc", pass_ids=[0], persist_signals={"emb": [t0]},
    ))
    c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        worker_id="voc", pass_ids=[1], persist_signals={"emb": [t1]},
    ))
    assert req.persist_signals == {}
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        worker_id="post", pass_ids=[0],
    )) == ["B"]
    c._process_done_forward("r", "B")
    assert [i.uuid for i in req.persist_signals["emb"]] == ["u0"]
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        worker_id="post", pass_ids=[1], partition_done=True,
    )) == ["B"]
    c._process_done_forward("r", "B")
    assert [i.uuid for i in req.persist_signals["emb"]] == ["u1"]


def test_repeating_consumer_loop_walk_does_not_end_until_partition_done():
    """A consumer partition running a user Loop walk (not in _repeating_walks)
    must stay active on partition_done=False and close only on partition_done=True."""
    c, req, rec = _conductor()
    c._stream_walks = frozenset({"b"})
    c._repeating_walks = frozenset()  # e.g., user Loop excluded from _repeating_walks
    assert c._process_done_forward("r", "B") is False
    assert req.partition_states["B"].is_done is False
    assert rec.producer_done == []
    assert _done_forward(c, req, "B", partition_done=True) is False
    assert req.partition_states["B"].is_done is True
    assert rec.producer_done == [("r", "B", "C")]


def test_producer_triggered_pipelined_walk_transition_across_disjoint_wgs():
    """A producer-triggered partition switching from walk_a (WG 0) to walk_b
    (WG 1) while Pass 0 is still queued closes Pass 0 under walk_a and Pass 1
    under walk_b."""
    from types import SimpleNamespace

    c, req, rec = _conductor()
    c.enable_prof = False
    c.producer_triggered_partitions = {"B"}
    c._walk_inputs = {**c._walk_inputs, "walk_a": (), "walk_b": ()}
    c._walk_persists = {**c._walk_persists, "walk_a": frozenset(), "walk_b": frozenset()}
    c.worker_graphs = {
        0: SimpleNamespace(graph_walks={"walk_a"}),
        1: SimpleNamespace(graph_walks={"walk_b"}),
    }
    req.all_worker_graph_ids = {0, 1}
    req.worker_graph_to_workers = {0: ["w0"], 1: ["w1"]}
    pstate = req.partition_states["B"]
    pstate.plan = [WalkStep("walk_a"), WalkStep("walk_b")]
    pstate.graph_walk = "walk_a"
    pstate.current_worker_graph_ids = {0}
    c._set_partition_worker_graph_ids = lambda rid, pname, walk: setattr(
        pstate, "current_worker_graph_ids", {0} if walk == "walk_a" else {1},
    )

    queued = c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w0", pass_ids=[0],
    ))
    queued += c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w1", pass_ids=[0], partition_done=True,
    ))
    assert queued == ["B", "B"]
    assert c._process_done_forward("r", "B") is False
    assert pstate.is_done is False
    assert pstate.fwd_pass_number == 1
    assert c._process_done_forward("r", "B") is False
    assert pstate.graph_walk == "walk_b"
    assert pstate.is_done is True
    assert rec.producer_done == [("r", "B", "C")]


def test_producer_triggered_out_of_order_reports_overlapping_and_disjoint_wgs():
    """Out-of-order WGD reports from pipelined producer-triggered walk transitions
    must never flip head walk early or mix WGs from different walks into one pass."""
    from types import SimpleNamespace

    # Case 1: Overlapping WGs (walk_a={0,1}, walk_b={0,2}) across walk_a -> walk_a -> walk_b -> walk_a.
    c, req, _ = _conductor()
    c.enable_prof = False
    c.producer_triggered_partitions = {"B"}
    c._walk_inputs = {**c._walk_inputs, "walk_a": (), "walk_b": ()}
    c._walk_persists = {**c._walk_persists, "walk_a": frozenset(), "walk_b": frozenset()}
    c.worker_graphs = {
        0: SimpleNamespace(graph_walks={"walk_a", "walk_b"}),
        1: SimpleNamespace(graph_walks={"walk_a"}),
        2: SimpleNamespace(graph_walks={"walk_b"}),
    }
    req.all_worker_graph_ids = {0, 1, 2}
    req.worker_graph_to_workers = {0: ["w0"], 1: ["w1"], 2: ["w2"]}
    pstate = req.partition_states["B"]
    pstate.plan = [WalkStep("walk_a"), WalkStep("walk_b")]
    pstate.graph_walk = "walk_a"
    pstate.current_worker_graph_ids = {0, 1}
    c._set_partition_worker_graph_ids = lambda rid, pname, walk: setattr(
        pstate, "current_worker_graph_ids", {0, 1} if walk == "walk_a" else {0, 2},
    )

    # Subcase A: ready_passes empty, WG 0 reports Pass 0 (walk_a) & Pass 1 (walk_b),
    # and WG 2 reports Pass 0 (walk_b) before WG 1 reports Pass 0 (walk_a).
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w0", pass_ids=[0],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w0", pass_ids=[1],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[2], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w2", pass_ids=[0],
    )) == []
    assert pstate.graph_walk == "walk_a"
    assert pstate.current_worker_graph_ids == {0, 1}

    # Once WG 1 reports Pass 0 (walk_a), BOTH Pass 0 (walk_a) and Pass 1 (walk_b) close in order.
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w1", pass_ids=[0],
    )) == ["B", "B"]
    assert [(w, list(k)) for w, _, k in pstate.ready_passes] == [
        ("walk_a", [(0, 0), (1, 0)]),
        ("walk_b", [(0, 1), (2, 0)]),
    ]

    # Subcase B: ready_passes non-empty, Pass 2 is walk_a (WG 0 arrived, WG 1 pending)
    # and Pass 3 is walk_b (WG 0 + WG 2 arrived early) -> must NOT mix (0,2)+(2,1)!
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w0", pass_ids=[2],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w0", pass_ids=[3],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[2], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w2", pass_ids=[1],
    )) == []
    assert len(pstate.ready_passes) == 2

    # Completing WG 1 Pass 1 (walk_a) closes Pass 2 (walk_a) then Pass 3 (walk_b).
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w1", pass_ids=[1],
    )) == ["B", "B"]
    assert [(w, list(k)) for w, _, k in pstate.ready_passes] == [
        ("walk_a", [(0, 0), (1, 0)]),
        ("walk_b", [(0, 1), (2, 0)]),
        ("walk_a", [(0, 2), (1, 1)]),
        ("walk_b", [(0, 3), (2, 1)]),
    ]

    # Case 2: Disjoint WGs with TP=2 on WG 0 (`walk_a={0}`), WG 1 (`walk_b={1}`),
    # where w1 reports Pass 1 before w0_r1 finishes Pass 0.
    c2, req2, _ = _conductor()
    c2.enable_prof = False
    c2.producer_triggered_partitions = {"B"}
    c2._walk_inputs = {**c2._walk_inputs, "walk_a": (), "walk_b": ()}
    c2._walk_persists = {**c2._walk_persists, "walk_a": frozenset(), "walk_b": frozenset()}
    c2.worker_graphs = {
        0: SimpleNamespace(graph_walks={"walk_a"}),
        1: SimpleNamespace(graph_walks={"walk_b"}),
    }
    req2.all_worker_graph_ids = {0, 1}
    req2.worker_graph_to_workers = {0: ["w0_r0", "w0_r1"], 1: ["w1_r0"]}
    pstate2 = req2.partition_states["B"]
    pstate2.plan = [WalkStep("walk_a"), WalkStep("walk_b")]
    pstate2.graph_walk = "walk_a"
    pstate2.current_worker_graph_ids = {0}
    c2._set_partition_worker_graph_ids = lambda rid, pname, walk: setattr(
        pstate2, "current_worker_graph_ids", {0} if walk == "walk_a" else {1},
    )

    assert c2._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w0_r0", pass_ids=[0],
    )) == []
    assert c2._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w1_r0", pass_ids=[0], partition_done=True,
    )) == []
    assert pstate2.graph_walk == "walk_a"
    assert list(pstate2.ready_passes) == []

    assert c2._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=False,
        graph_walk="walk_a", worker_id="w0_r1", pass_ids=[0],
    )) == ["B", "B"]
    assert [(w, d, list(k)) for w, d, k in pstate2.ready_passes] == [
        ("walk_a", False, [(0, 0)]),
        ("walk_b", True, [(1, 0)]),
    ]


def test_producer_triggered_disjoint_three_walk_prefers_plan_order_over_wg_id():
    """When disjoint producer-triggered walks arrive out of order with
    wg_id(walk_b) > wg_id(walk_c) and walk_c has partition_done=True, Conductor
    must close walk_b before walk_c following pstate.plan order."""
    from types import SimpleNamespace

    c, req, _ = _conductor()
    c.enable_prof = False
    c.producer_triggered_partitions = {"B"}
    c._walk_inputs = {**c._walk_inputs, "walk_a": (), "walk_b": (), "walk_c": ()}
    c._walk_persists = {
        **c._walk_persists,
        "walk_a": frozenset(), "walk_b": frozenset(), "walk_c": frozenset(),
    }
    c.worker_graphs = {
        0: SimpleNamespace(graph_walks={"walk_a"}),
        1: SimpleNamespace(graph_walks={"walk_c"}),
        2: SimpleNamespace(graph_walks={"walk_b"}),
    }
    req.all_worker_graph_ids = {0, 1, 2}
    req.worker_graph_to_workers = {0: ["w0_0", "w0_1"], 1: ["w1_0"], 2: ["w2_0"]}
    pstate = req.partition_states["B"]
    pstate.plan = [WalkStep("walk_a"), WalkStep("walk_b"), WalkStep("walk_c")]
    pstate.graph_walk = "walk_a"
    pstate.current_worker_graph_ids = {0}
    wg_map = {"walk_a": {0}, "walk_b": {2}, "walk_c": {1}}
    c._set_partition_worker_graph_ids = lambda rid, pname, walk: setattr(
        pstate, "current_worker_graph_ids", wg_map[walk],
    )

    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=True,
        graph_walk="walk_a", worker_id="w0_0", pass_ids=[0],
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[1], is_first_tp_rank=True,
        graph_walk="walk_c", worker_id="w1_0", pass_ids=[0], partition_done=True,
    )) == []
    assert c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[2], is_first_tp_rank=True,
        graph_walk="walk_b", worker_id="w2_0", pass_ids=[0], partition_done=False,
    )) == []
    queued = c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", partition_name="B", worker_graph_ids=[0], is_first_tp_rank=False,
        graph_walk="walk_a", worker_id="w0_1", pass_ids=[0],
    ))
    assert queued == ["B", "B", "B"]
    assert [(w, d, list(k)) for w, d, k in pstate.ready_passes] == [
        ("walk_a", False, [(0, 0)]),
        ("walk_b", False, [(2, 0)]),
        ("walk_c", True, [(1, 0)]),
    ]

    for _ in range(3):
        c._process_done_forward("r", "B")
    assert pstate.graph_walk == "walk_c"
    assert pstate.cursor == 1
    assert pstate.fwd_pass_number == 3
    assert pstate.is_done is True
