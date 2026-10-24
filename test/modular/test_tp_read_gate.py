"""Rank 0 of a lockstep (TP/SP) instance runs a row only once every follower
has landed the row's inputs (``mstar/worker/read_gate.py``).

Rank 0 schedules from its own copy of a request's inputs. A follower whose read
of that input fails after rank 0 broadcast the step can never run it, and the
instance's collectives hang (#564). With the gate the row never leaves rank 0
until the follower reported its input landed; a failed read is never reported,
so only that request fails and the instance keeps serving.

Two ranks of a TP=2 instance, each with the real ``MicroScheduler`` and the
real ``Worker`` readiness, gate, failure and drain code; the follower reads
through a real ``SharedMemoryCommunicationManager`` whose reads complete when
the test says. Messages between the ranks go through the wire codec.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_failed_tensor_read import _read, _shm_manager
from test_micro_scheduler import NODE, WALK, _Engine, _Manager

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensors import StoredOutputs
from mstar.communication.wire import decode, encode
from mstar.engine.resources.step import StepContext
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.streaming.stream_buffer import StreamChunkInfo
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    MessageSource,
    OffloadDelta,
    ReadsLanded,
    RemoveRequest,
    WorkerMessageType,
)
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.read_gate import FollowerReadGate
from mstar.worker.worker import _REMOVED_RIDS_KEEP, Worker

WORKERS = ["w0", "w1"]
GRID = "image_grid_thw"


def _info(stream_buffers_by_consumer=None):
    return SimpleNamespace(
        stream_buffers_by_consumer=stream_buffers_by_consumer or {},
        stream_buffers={},
    )


def _rank(worker_id, rids, tensor_manager, consumes=None, workers=WORKERS):
    """Instance rank ``workers.index(worker_id)`` of a TP instance of NODE on
    ``workers``. ``consumes[rid]`` lists the (signal, uuids) a popped row of
    ``rid`` takes."""
    leader = worker_id == workers[0]
    manager = _Manager(rids)
    manager.per_request_info = {r: _info() for r in rids}
    if consumes is not None:
        pop = manager.runtime.pop_walk_rids

        def pop_walk_rids(node_name, rows, check_ready=False):
            popped = pop(node_name, rows, check_ready)
            if popped is None:
                return None
            block = ColumnarEdgeSpecs.empty()
            for rid in popped.wg_ids.keys:
                for signal, uuids in consumes[rid]:
                    block.add(rid, signal, uuids, False)
            return popped._replace(input_edges=block)

        manager.runtime.pop_walk_rids = pop_walk_rids
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda n: _Engine()),
        parallel_leader_nodes={NODE} if leader else set(),
    )
    sched.runtime = manager.runtime
    w = Worker.__new__(Worker)
    w.worker_id, w.scheduler = worker_id, sched
    w.parallel_nodes = {NODE}
    w.parallel_leader_nodes = {NODE} if leader else set()
    w.is_tp_follower = not leader
    w.tp_async_sched, w.tp_async_nodes = False, None
    w._tp_broadcast_seq = 0
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda dest, msg=None: w.sent.append((dest, decode(encode(msg)))),
    )
    group = SimpleNamespace(
        nodes={NODE}, tp_size=len(workers), _tp_rank=workers.index(worker_id), _workers=workers,
    )
    sharding = SimpleNamespace(groups=[group], get_sharding_group=lambda n, g: group)

    def ingest_inputs_batch(block, can_buffer):
        for rid in block.rids:
            manager.ingest(rid)

    w._graph_runtime = SimpleNamespace(
        get_rid_handle=lambda r: r if r in manager.per_request_info else None,
        get_rid_string=lambda h: h,
        get_sharding_config=lambda r: sharding,
        ingest_inputs_batch=ingest_inputs_batch,
    )
    w.request_state = manager
    w.tensor_manager = tensor_manager
    w.engine_manager = SimpleNamespace(get_engine=lambda n: SimpleNamespace(
        take_resident_delta=lambda n: OffloadDelta.new(),
        decide_cached_prefix=lambda b: {},
        page_trails=lambda n: {},
        prefix_cache_open=lambda n: False,
    ))
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w.enable_nvtx = False
    w._unprocessed_messages = {}
    w._in_flight_rids, w._pending_drains = set(), set()
    w._draining_rids, w._reads_done_sent = set(), set()
    w._removed_rids = RecentSet(_REMOVED_RIDS_KEEP)
    w._setup_read_gate()
    return w, manager


def _edge(uuid, name=GRID, node=NODE, is_streaming=False):
    src = torch.ones(3, dtype=torch.int64)
    info = TensorPointerInfo(
        list(src.shape), src.dtype, src.nbytes, 0, list(src.stride()), uuid, "s", "w0",
    )
    return GraphEdge(next_node=node, name=name, tensor_info=[info], is_streaming=is_streaming)


def _grid_edge(uuid):
    return _edge(uuid)


def _reading(tm, rid, edge, future):
    """A read ``tm`` started for ``edge``: its buffers are registered, and it
    lands or fails when ``future`` does."""
    for info in edge.tensor_info:
        tm.tensor_store.put_tensor(rid, info.uuid, torch.zeros(3, dtype=torch.int64), info)
    return _read(rid, edge, future)


def _landed(future=None):
    if future is None:
        future = Future()
        future.set_result(None)
    return future


def _pair(tmp_path):
    """Rank 0 has read both requests' image grids (its uuids 10 and 11); the
    follower is reading its own copies (uuids 60 and 61)."""
    leader_tm, _ = _shm_manager(tmp_path / "w0")
    leader, leader_mgr = _rank(
        "w0", ["r0", "r1"], leader_tm,
        consumes={"r0": [(GRID, [10])], "r1": [(GRID, [11])]},
    )
    leader_tm.pending = [
        _reading(leader_tm, rid, _grid_edge(uuid), _landed())
        for rid, uuid in (("r0", 10), ("r1", 11))
    ]
    leader._check_ready_tensors()
    tm, _comm = _shm_manager(tmp_path / "w1")
    follower, follower_mgr = _rank("w1", ["r0", "r1"], tm)
    follower_mgr.queues["wg0"]._ready = {}
    reads = {"r0": Future(), "r1": Future()}
    tm.pending = [
        _reading(tm, rid, _grid_edge(60 + i), reads[rid]) for i, rid in enumerate(reads)
    ]
    return leader, leader_mgr, follower, follower_mgr, reads


def _taken(w, dest, message_type):
    hits = [m for d, m in w.sent if d == dest and m.message_type == message_type]
    w.sent = [(d, m) for d, m in w.sent if m not in hits]
    return hits


def _deliver(src, dest, message_type):
    msgs = _taken(src, dest.worker_id, message_type)
    dest._process_message_list(msgs)
    return msgs


def _node_batch(rids):
    return SimpleNamespace(
        node_name=NODE, graph_walk=WALK, request_ids=rids,
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
    )


def _rows(batch):
    return None if batch is None else sorted(batch.request_to_worker_graph)


def test_a_late_follower_failure_on_the_image_grid_fails_only_that_request(tmp_path):
    """r1's image grid sets the size of r1's image span, so the follower cannot
    run r1's step on anything but the real values. Its read fails after rank 0
    read its own copy. Rank 0 runs only r0, the follower runs that step, and
    r1 is drained on both ranks without either of them hanging."""
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)

    reads["r0"].set_result(None)
    follower._check_ready_tensors()
    reports = _deliver(follower, leader, WorkerMessageType.READS_LANDED)
    assert [(m.body.request_id, m.body.counts) for m in reports] == [("r0", {GRID: 1})]

    batch = leader.scheduler.get_next_batch(leader_mgr)
    assert _rows(batch) == ["r0"]
    # r1 was popped and waits, claimed, so a late input cannot ready it twice
    assert ("r1", NODE) in leader_mgr.in_flight
    leader.maybe_send_zmq_to_tp_followers(_node_batch(["r0"]))
    _deliver(leader, follower, WorkerMessageType.SCHEDULE_TP)
    assert _rows(follower.scheduler.get_next_batch(follower.request_state)) == ["r0"]

    reads["r1"].set_exception(RuntimeError("Mooncake read failed. Status: -1"))
    follower._check_ready_tensors()
    assert not _taken(follower, "w0", WorkerMessageType.READS_LANDED)
    (failure,) = [m for d, m in follower.sent if d == "conductor"
                  and m.message_type == ConductorMessageType.FAIL_REQUESTS]
    assert list(failure.body.errors) == ["r1"]
    # rank 0 schedules nothing for r1 while the drain is on its way
    assert leader.scheduler.get_next_batch(leader_mgr) is None

    leader._drain_request(DrainRequest(request_id="r1", source=MessageSource.CONDUCTOR))
    _deliver(leader, follower, WorkerMessageType.DRAIN_REQUEST)
    for rank in (leader, follower):
        done = [m.body.request_id for d, m in rank.sent
                if m.message_type == ConductorMessageType.READS_DONE]
        assert done == ["r1"], rank.worker_id
    assert "r1" not in leader.scheduler.backlog.get((NODE, WALK), SimpleNamespace(
        request_to_worker_graph={})).request_to_worker_graph


def test_a_held_row_runs_once_the_follower_reports_its_input(tmp_path):
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)
    reads["r0"].set_result(None)
    follower._check_ready_tensors()
    _deliver(follower, leader, WorkerMessageType.READS_LANDED)
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]

    reads["r1"].set_result(None)
    follower._check_ready_tensors()
    _deliver(follower, leader, WorkerMessageType.READS_LANDED)
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r1"]


def test_a_follower_report_that_comes_before_rank_0s_own_input_still_counts(tmp_path):
    """The follower's read can land before rank 0's own copy does."""
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)
    leader._read_gate.forget("r0")
    reads["r0"].set_result(None)
    follower._check_ready_tensors()
    _deliver(follower, leader, WorkerMessageType.READS_LANDED)
    leader._note_landed({"r0": [_grid_edge(10)]})
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_ranks_holding_different_tensors_for_one_input_agree_on_the_count(tmp_path):
    """Each rank of a sharded or consolidated input holds its own tensor: the
    follower's uuid (60) is not rank 0's (10), and its report still counts."""
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)
    reads["r0"].set_result(None)
    reads["r1"].set_result(None)
    follower._check_ready_tensors()
    _deliver(follower, leader, WorkerMessageType.READS_LANDED)
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0", "r1"]


def test_rank_0_numbers_an_input_by_the_uuid_it_landed_under():
    """A consolidated input lands on rank 0 under a uuid minted there, not the
    shards' uuids it was read from; the row that consumes it is gated."""
    leader, leader_mgr = _rank(
        "w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [(GRID, [900])]},
    )
    leader._note_landed({"r0": [_grid_edge(900)]})
    assert leader.scheduler.get_next_batch(leader_mgr) is None
    leader._record_follower_reads(SimpleNamespace(
        request_id="r0", follower="w1", counts={GRID: 1}, stream_items={},
    ))
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_the_instances_own_outputs_never_wait():
    """A decode step consumes the token rank 0 produced itself; every rank
    produces its own and none reads it, so nothing is awaited."""
    leader, leader_mgr = _rank(
        "w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [("text_inputs", [99])]},
    )
    leader._note_landed({"r0": [_grid_edge(10)]})  # the prefill's grid
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_only_inputs_of_the_lockstep_node_are_numbered():
    leader, _ = _rank("w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False))
    pixels = _edge(5, name="pixel_values", node="vision_encoder")
    streamed = _edge(6, is_streaming=True)
    signal_only = GraphEdge(next_node=NODE, name="text_inputs")
    leader._note_landed({"r0": [pixels, streamed, signal_only, _grid_edge(7)]})
    assert leader._read_gate._external["r0"] == {7}


def test_a_colocated_encoders_outputs_are_awaited():
    """rank 0's LLM takes the encoder's vision_embeds locally; the follower reads them."""
    leader, leader_mgr = _rank(
        "w0", ["r0", "r1"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [("vision_embeds", [21])], "r1": [("vision_embeds", [22])]},
    )
    leader._count_local_outputs(
        StoredOutputs([21, 22], ["r0", "r1"], [0, 0], [1, 1]), ["vision_embeds"],
    )
    leader._record_follower_reads(SimpleNamespace(
        request_id="r1", follower="w1", counts={"vision_embeds": 1}, stream_items={},
    ))
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r1"]


def test_a_streamed_chunk_waits_for_the_followers_items():
    """The row's chunk ends at item 3 of the stream; the follower has to have
    landed its first three items, in order."""
    sbuf = SimpleNamespace(ingested_chunk=StreamChunkInfo(
        start_offset=1, context_items=0, num_items=2, is_final=False,
    ))
    leader, leader_mgr = _rank(
        "w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [("thinker_hidden", [31])]},
    )
    leader_mgr.per_request_info["r0"] = _info({NODE: {"thinker_hidden": sbuf}})
    def report(items):
        leader._record_follower_reads(SimpleNamespace(
            request_id="r0", follower="w1", counts={}, stream_items={"thinker_hidden": items},
        ))

    report(2)
    assert leader.scheduler.get_next_batch(leader_mgr) is None
    report(3)
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_the_follower_reports_streamed_items_landed_in_order(tmp_path):
    tm, _ = _shm_manager(tmp_path)
    follower, follower_mgr = _rank("w1", ["r0"], tm)
    sbuf = SimpleNamespace(landed_items=lambda: 4, put=lambda uuid, t: None)
    follower_mgr.per_request_info["r0"] = SimpleNamespace(
        stream_buffers={"thinker_hidden": sbuf}, stream_buffers_by_consumer={},
    )
    follower._route_streaming_tensor = lambda rid, edge: None
    done = Future()
    done.set_result(None)
    tm.pending = [_reading(tm, "r0", _edge(41, name="thinker_hidden", is_streaming=True), done)]
    follower._check_ready_tensors()
    (msg,) = _taken(follower, "w0", WorkerMessageType.READS_LANDED)
    assert (msg.body.counts, msg.body.stream_items) == ({}, {"thinker_hidden": 4})


def test_a_request_whose_follower_hosts_another_node_is_not_gated():
    """A node beside the follower feeds it without a read, which the follower
    cannot count; waiting for it would hold a healthy row for good."""
    leader, leader_mgr = _rank(
        "w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [(GRID, [10])]},
    )
    leader._note_landed({"r0": [_grid_edge(10)]})
    sharding = leader._graph_runtime.get_sharding_config("r0")
    sharding.node_to_worker = {
        NodeAndGraphWalk(NODE, WALK): ["w0", "w1"],
        NodeAndGraphWalk("vision_encoder", WALK): ["w1"],
    }
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_an_input_delivered_again_needs_the_followers_second_copy():
    """A persisted input comes back as a later walk's input under the same
    uuid; the follower has to land it again before that row runs."""
    gate = FollowerReadGate()
    gate.external(1, [10])
    assert gate.missing(1, ["w1"], lambda: [(GRID, (10,))], {})
    gate.record(1, "w1", {GRID: 1}, {})
    assert not gate.missing(1, ["w1"], lambda: [(GRID, (10,))], {})
    gate.external(1, [10])  # delivered again
    assert gate.missing(1, ["w1"], lambda: [(GRID, (10,))], {})
    gate.record(1, "w1", {GRID: 2}, {})
    assert not gate.missing(1, ["w1"], lambda: [(GRID, (10,))], {})


def test_a_loops_accumulated_outputs_are_one_delivery():
    """A colocated producer stored two tensors over two iterations and
    delivers them as one edge; the follower receives that one edge."""
    gate = FollowerReadGate()
    gate.external(1, [21])
    gate.external(1, [22])
    gate.record(1, "w1", {"vision_embeds": 1}, {})
    assert not gate.missing(1, ["w1"], lambda: [("vision_embeds", (21, 22))], {})


def test_an_output_routed_elsewhere_does_not_raise_the_count():
    """Rank 0 marks every colocated output; one never consumed by the lockstep
    node is never numbered, so the next one it does consume is number 1."""
    gate = FollowerReadGate()
    gate.external(1, [30])  # goes to another node
    gate.external(1, [31])
    gate.record(1, "w1", {"x": 1}, {})
    assert not gate.missing(1, ["w1"], lambda: [("x", (31,))], {})


def test_rows_are_numbered_in_the_order_rank_0_consumes_them():
    gate = FollowerReadGate()
    gate.external(1, [10, 11])
    gate.record(1, "w1", {GRID: 1}, {})
    gate.record(1, "w2", {GRID: 2}, {})
    assert not gate.missing(1, ["w1", "w2"], lambda: [(GRID, (10,)), ("token", (5,))], {})
    assert gate.missing(1, ["w1", "w2"], lambda: [(GRID, (11,))], {})
    # rechecking a held row keeps its number
    assert gate.missing(1, ["w1", "w2"], lambda: [(GRID, (11,))], {})
    gate.record(1, "w1", {GRID: 2}, {})
    # nothing unnumbered and nobody behind: the row is not even looked at
    assert not gate.missing(1, ["w1", "w2"], lambda: 1 / 0, {})
    gate.forget(1)
    assert not gate.missing(1, ["w1"], lambda: [(GRID, (11,))], {})


def test_a_decode_row_with_nothing_awaited_is_let_through_first():
    """The common row, a decode step taking only its own token, returns before
    any sharding lookup or index build."""
    leader, leader_mgr = _rank(
        "w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False),
        consumes={"r0": [("text_inputs", [99])]},
    )
    leader._graph_runtime.get_sharding_config = lambda r: 1 / 0
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


def test_a_colocated_output_the_lockstep_node_does_not_take_is_not_marked():
    leader, _ = _rank("w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False))
    leader._lockstep_inputs = frozenset({"vision_embeds"})
    leader._count_local_outputs(
        StoredOutputs([21, 22], ["r0", "r0"], [0, 1], [1, 1]), ["vision_embeds", "debug_map"],
    )
    assert leader._read_gate._external["r0"] == {21}


def test_a_removed_requests_late_report_is_dropped_not_parked(tmp_path):
    """A report can reach rank 0 after its REMOVE; parked, it would be replayed
    into a later request that reuses the id and open that request's gate early."""
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)
    leader._unprocessed_messages["r9"] = [SimpleNamespace(
        message_type=WorkerMessageType.READS_LANDED, body=SimpleNamespace(request_id="r9"),
    )]
    leader._mark_removed("r9")
    assert "r9" not in leader._unprocessed_messages
    reads["r0"].set_result(None)
    follower._check_ready_tensors()
    (late,) = _taken(follower, "w0", WorkerMessageType.READS_LANDED)
    late.body.request_id = "r9"
    leader._process_message_list([late])
    assert "r9" not in leader._unprocessed_messages



# --- which followers the gate waits on, on shipped configs' layouts ----------


class _AnyFlagFalse:
    """A model config with only what the graph builders read set; every
    feature flag reads as off."""
    max_inference_steps = 4
    max_position_embeddings = 4096

    def __getattr__(self, name):
        return False


def _walks(cls):
    model = cls.__new__(cls)
    model.config = _AnyFlagFalse()
    return SimpleNamespace(get_graph_walk_graphs=lambda: cls.get_graph_walk_graphs(model))


def _placement(config, moves=None):
    """The shipped config's node_groups as node -> workers, with ``moves``
    overriding a node's ranks."""
    import yaml
    groups = yaml.safe_load(open(f"configs/{config}"))["node_groups"]
    ranks = {n: g["ranks"] for g in groups for n in g["node_names"]}
    ranks.update(moves or {})
    return {n: [f"worker_{r}" for r in rs] for n, rs in ranks.items()}


def _leader_on(model, node, config, moves=None):
    leader, _ = _rank("w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False))
    leader.parallel_nodes = leader.parallel_leader_nodes = {node}
    leader._setup_read_gate(model)
    placement = _placement(config, moves)
    group = SimpleNamespace(nodes={node}, _workers=placement[node])
    leader._graph_runtime.get_sharding_config = lambda r: SimpleNamespace(
        get_sharding_group=lambda n, g: group,
        node_to_worker={NodeAndGraphWalk(n, "w"): ws for n, ws in placement.items()},
    )
    return leader


def test_a_follower_beside_a_node_that_does_not_feed_it_is_still_gated():
    """cosmos3_super_i2v_4step_tp2 puts vae_decoder on the dit pair's follower;
    it takes dit's output and feeds nothing into it, so the follower still
    reports dit's inputs and the request is gated."""
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    leader = _leader_on(_walks(Cosmos3Model), "dit", "cosmos3_super_i2v_4step_tp2.yaml")
    assert leader._gated_followers("r0", "dit", "video_gen") == ("worker_1",)


def test_a_follower_beside_a_node_that_feeds_it_is_not_gated():
    """qwen3_5_9b_tp2 is gated as shipped; with the vision encoder moved onto
    the follower, its embeddings would reach the follower's LLM without a
    read, which it cannot count, so that layout is not gated."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    model = _walks(Qwen3_5DenseModel)
    shipped = _leader_on(model, "LLM", "qwen3_5_9b_tp2.yaml")
    assert shipped._gated_followers("r0", "LLM", "prefill_vision") == ("worker_1",)
    moved = _leader_on(model, "LLM", "qwen3_5_9b_tp2.yaml", {"vision_encoder": [1]})
    assert moved._gated_followers("r0", "LLM", "prefill_vision") == ()


def test_the_instances_own_persisted_output_is_neither_marked_nor_counted(tmp_path):
    """The prefill's new_token comes back as the decode walk's text_inputs
    through the read path on every rank; every rank produced it, so the first
    decode step does not wait for a report."""
    leader, leader_mgr, follower, _follower_mgr, _reads = _pair(tmp_path)

    def token(uuid, entity):
        edge = _edge(uuid, name="text_inputs")
        edge.tensor_info[0]._source_node_name = NODE
        edge.tensor_info[0].source_entity = entity
        return edge

    leader._note_landed({"r0": [token(70, leader.tensor_manager.my_entity_id)]})
    follower._note_landed({"r0": [token(71, follower.tensor_manager.my_entity_id)]})
    assert 70 not in leader._read_gate._external["r0"]
    assert "text_inputs" not in follower._landed_counts.get("r0", {})
    # a prefill instance on other workers produced it: each decode rank reads
    # that copy, which can fail, so it is gated like any other input
    leader._note_landed({"r0": [token(80, "prefill_0")]})
    follower._note_landed({"r0": [token(80, "prefill_0")]})
    assert 80 in leader._read_gate._external["r0"]
    assert follower._landed_counts["r0"]["text_inputs"] == 1


def test_a_loops_terminal_output_into_the_lockstep_node_makes_its_nodes_feeders():
    from mstar.graph.base import GraphNode, Loop, Sequential
    from mstar.worker.read_gate import lockstep_feeders

    encoder = GraphNode(name="enc", input_names=["x"], outputs=[GraphEdge(next_node="enc", name="x")])
    walk = Sequential([
        Loop(section=encoder, max_iters=2, outputs=[GraphEdge(next_node="LLM", name="x")]),
        GraphNode(name="LLM", input_names=["x"], outputs=[]),
    ])
    assert lockstep_feeders(walk, frozenset({"LLM"})) == {"enc"}


def test_removing_a_request_this_rank_never_admitted_drops_its_reports():
    leader, _ = _rank("w0", ["r0"], SimpleNamespace(has_inflight_reads=lambda r: False))
    leader._unprocessed_messages["r9"] = [SimpleNamespace(
        message_type=WorkerMessageType.READS_LANDED, body=SimpleNamespace(request_id="r9"),
    )]
    leader._remove_request(RemoveRequest(request_id="r9"))
    assert "r9" not in leader._unprocessed_messages and "r9" in leader._removed_rids


@pytest.mark.parametrize("config", sorted(
    p.name for p in __import__("pathlib").Path("configs").glob("*.yaml")
    if "sp_size" in p.read_text()
))
def test_every_rank_of_an_sp_instance_gets_every_input_it_is_gated_on(config):
    """The gate holds a row until each follower counted as many deliveries of
    an input as rank 0 numbered, so a follower whose share of an input had no
    tensor would hold the row for good. On every shipped SP layout the
    lockstep node's inputs are replicated: each rank, rank 0 included, gets
    every delivery, from the API server and from every producer."""
    import yaml

    from mstar.distributed.base import ShardingConfig
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.registry import MODEL_REGISTRY

    spec = yaml.safe_load(open(f"configs/{config}"))
    module, name = MODEL_REGISTRY[spec["model"]]
    cls = getattr(__import__(module, fromlist=[name]), name)
    assert issubclass(cls, Cosmos3Model)  # the only SP model; extend the walks stub for another
    walks = _walks(cls).get_graph_walk_graphs()
    placement = _placement(config)
    (lockstep,) = [n for g in spec["node_groups"] if g.get("sp_size", 1) > 1 for n in g["node_names"]]
    instance = placement[lockstep]
    sharding: ShardingConfig = cls.get_sharding_config(cls.__new__(cls), f"configs/{config}")
    sharding.setup({
        NodeAndGraphWalk(node, walk): placement[node]
        for walk, section in walks.items() for node in section.get_nodes()
    })
    checked = 0
    for walk, section in walks.items():
        nodes = section.get_nodes()
        if lockstep not in nodes:
            continue
        group = sharding.get_sharding_group(lockstep, walk)
        assert group._workers == instance and len(instance) > 1
        sources = [(None, name) for name in nodes[lockstep].input_names]
        sources += [
            (node.name, edge.name) for node in nodes.values()
            for edge in getattr(node, "outputs", None) or ()
            if edge.next_node == lockstep and node.name != lockstep
        ]
        for source, signal in sources:
            src = sharding.get_sharding_group(source, walk) if source else None
            got = set()
            for rank in range(src.tp_size if src else 1):
                for dest in sharding.compute_fanout(
                    signal, walk, source or "api_server", lockstep, [8], walk,
                    source_tp_rank=rank if src else None,
                ):
                    if dest.full_tensor or dest.end_idxs[0] > dest.start_idxs[0]:
                        got.add(dest.worker)
            assert got == set(instance), (walk, source, signal, got)
            checked += 1
    # a persisted output (vae_encoder's cond_latents) re-enters a later walk
    # as the lockstep node's input, fanned out from the worker that kept it
    for pwalk, psection in walks.items():
        for node in psection.get_nodes().values():
            for edge in getattr(node, "outputs", None) or ():
                if not edge.persist or node.name == lockstep:
                    continue
                for walk, section in walks.items():
                    nodes = section.get_nodes()
                    if lockstep not in nodes or edge.name not in nodes[lockstep].input_names:
                        continue
                    got = {
                        dest.worker for dest in sharding.compute_fanout(
                            edge.name, pwalk, node.name, lockstep, [8], walk,
                        ) if dest.full_tensor
                    }
                    assert got == set(instance), (pwalk, node.name, edge.name, walk, got)
                    checked += 1
    assert checked


def test_at_tp4_a_row_waits_for_every_follower_and_one_failed_read_holds_only_it(tmp_path):
    """Three followers: rank 0 runs r0 only once all three reported it. w3's
    read of r1 fails, so r1 never runs, though w1 and w2 reported theirs."""
    workers = ["w0", "w1", "w2", "w3"]
    leader_tm, _ = _shm_manager(tmp_path / "w0")
    leader, leader_mgr = _rank(
        "w0", ["r0", "r1"], leader_tm, workers=workers,
        consumes={"r0": [(GRID, [10])], "r1": [(GRID, [11])]},
    )
    leader_tm.pending = [
        _reading(leader_tm, rid, _grid_edge(uuid), _landed())
        for rid, uuid in (("r0", 10), ("r1", 11))
    ]
    leader._check_ready_tensors()
    followers, reads = {}, {}
    for k, w in enumerate(workers[1:]):
        tm, _comm = _shm_manager(tmp_path / w)
        followers[w], mgr = _rank(w, ["r0", "r1"], tm, workers=workers)
        mgr.queues["wg0"]._ready = {}
        reads[w] = {"r0": Future(), "r1": Future()}
        tm.pending = [
            _reading(tm, rid, _grid_edge(100 * (k + 1) + i), reads[w][rid])
            for i, rid in enumerate(("r0", "r1"))
        ]

    def land(w, rid, error=None):
        if error is None:
            reads[w][rid].set_result(None)
        else:
            reads[w][rid].set_exception(error)
        followers[w]._check_ready_tensors()
        return _deliver(followers[w], leader, WorkerMessageType.READS_LANDED)

    for w in ("w1", "w2"):
        assert [m.body.request_id for m in land(w, "r0")] == ["r0"]
        assert [m.body.request_id for m in land(w, "r1")] == ["r1"]
    assert leader.scheduler.get_next_batch(leader_mgr) is None  # w3 has neither
    assert [m.body.request_id for m in land("w3", "r0")] == ["r0"]
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]
    assert land("w3", "r1", RuntimeError("Mooncake read failed. Status: -1")) == []
    assert [list(m.body.errors) for d, m in followers["w3"].sent if d == "conductor"
            and m.message_type == ConductorMessageType.FAIL_REQUESTS] == [["r1"]]
    assert leader.scheduler.get_next_batch(leader_mgr) is None


def test_a_report_parked_before_new_request_counts_once_it_is_replayed(tmp_path):
    """The follower's report can reach rank 0 before rank 0's NEW_REQUEST: it
    is parked under the wire id and replayed with the request, so the row
    does not wait for a second report."""
    leader, leader_mgr, follower, _follower_mgr, reads = _pair(tmp_path)
    reads["r0"].set_result(None)
    follower._check_ready_tensors()
    (report,) = _taken(follower, "w0", WorkerMessageType.READS_LANDED)
    info = leader_mgr.per_request_info.pop("r0")  # rank 0 has not admitted r0 yet
    leader._process_message_list([report])
    assert leader._unprocessed_messages["r0"] == [report]
    assert leader._read_gate._counts.get("r0") is None
    leader_mgr.per_request_info["r0"] = info
    leader._process_unprocessed("r0")  # what _add_new_request does last
    assert "r0" not in leader._unprocessed_messages
    assert _rows(leader.scheduler.get_next_batch(leader_mgr)) == ["r0"]


class _PastTheGateError(Exception):
    pass


def _speculating_leader(prepped_uuid):
    """Rank 0 about to continue r0 speculatively, with prep handing back r0's
    row with ``prepped_uuid`` on the grid edge. Past the gate's check the
    speculation raises _PastTheGateError; a dropped one records its commit."""
    leader, _ = _rank("w0", ["r0"], SimpleNamespace(
        has_inflight_reads=lambda r: False, get_tensor=None,
    ))
    commits = []
    block = ColumnarEdgeSpecs.empty()
    block.add("r0", GRID, [prepped_uuid], False)

    def to_input_tensors(*a, **k):
        raise _PastTheGateError

    block.to_input_tensors = to_input_tensors
    leader._graph_runtime.prep_spec_rids = lambda inp: SimpleNamespace(
        spec_id=7, ready_rids=list(inp.rids), input_edges=block, consumed_streaming_edge_idxs=[],
    )
    leader._graph_runtime.commit_speculation = lambda sid, ok, rids: commits.append((sid, ok, rids))
    leader.scheduler.backlog_splits_from = lambda *a: False
    leader.scheduler.room_for_continuing = lambda t: None
    leader._is_tearing_down = lambda r: False
    leader._streams_possible = lambda: False
    pending = SimpleNamespace(batch=SimpleNamespace(node_name=NODE))
    target = SimpleNamespace(node_name=NODE, graph_walk=WALK)
    return leader, commits, lambda: leader._try_speculate_next(pending, (target, ["r0"]))


def test_a_speculation_that_would_continue_a_row_ahead_of_a_followers_read_is_dropped():
    leader, commits, speculate = _speculating_leader(10)
    leader._read_gate.external("r0", [10])  # rank 0 read it; the follower has not reported
    assert speculate() is None
    assert commits == [(7, False, ["r0"])]


def test_a_speculation_continues_a_row_once_the_follower_reported_its_input():
    leader, commits, speculate = _speculating_leader(10)
    leader._read_gate.external("r0", [10])
    leader._record_follower_reads(ReadsLanded(
        request_id="r0", follower="w1", counts={GRID: 1}, stream_items={},
    ))
    try:
        speculate()
    except _PastTheGateError:
        pass
    else:
        raise AssertionError("the speculation stopped before the gate let it through")
    assert commits == []
