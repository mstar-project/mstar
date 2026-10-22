"""Client streams: a ``Connection`` from ``CLIENT`` feeds a running request.

The conductor forwards each ``CLIENT_STREAM_INPUT`` chunk to the consuming node's
workers as a streaming ``INPUT_SIGNALS`` edge and releases its persist hold
with the reader count, holds chunks for a request still waiting for
admission, drops them for one that has ended, and closes the stream with
``producer_done`` from ``CLIENT`` when the client's input ends.

Drives the real ``Conductor.__init__`` and ``Conductor.run`` with worker
launch and ZMQ stubbed, as ``test_ingest_unpersist.py`` does. The worker test
at the end checks the other half: those messages land in the worker's
``StreamBuffer`` with no worker change.
"""
import logging
from collections import deque
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

import mstar.communication.wire_types  # noqa: E402,F401 — registers the wire tags
import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.communication import wire  # noqa: E402
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore  # noqa: E402
from mstar.communication.tensors import TensorCommunicationManager  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
)
from mstar.distributed.base import ShardingConfig  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo  # noqa: E402
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput  # noqa: E402
from mstar.graph.runtime.python import PythonGraphRuntime  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.base import ForwardPassArgs, Model, WorkerGraph  # noqa: E402
from mstar.streaming.chunk_policy import (  # noqa: E402
    FixedChunkPolicy,
    LeftContextChunkPolicy,
    RampChunkPolicy,
    ScheduledLeftContextChunkPolicy,
    SlidingWindowChunkPolicy,
)
from mstar.streaming.stream_buffer import StreamBuffer  # noqa: E402
from mstar.streaming.topology import (  # noqa: E402
    CLIENT,
    ClientStreamLimits,
    Connection,
    PartitionTopology,
    client_stream_limits,
)
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    AbortRequest,
    ClientStreamConsumed,
    ClientStreamInput,
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequestConductor,
    ReadsDone,
    SetupDone,
    WorkerGraphsDone,
)
from mstar.worker.worker import Worker  # noqa: E402

PREPROCESS = "api_server_preprocess_worker"


class _StubComm:
    def __init__(self):
        self.sent = []
        self.batches = deque()

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return self.batches.popleft() if self.batches else []


class _ScriptDoneError(Exception):
    pass


class _DuplexToy(Model):
    """One partition whose loop consumes a stream the client feeds."""

    def __init__(self, consumer_input="audio_in"):
        self._consumer_input = consumer_input

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return {"listen": Loop(name="listen_loop", max_iters=1000, outputs=[], section=GraphNode(
            name="ear", input_names=[self._consumer_input], consumes_stream=True,
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="text", output_modality="text")],
        ))}

    def get_partitions(self):
        return [PartitionDefinition(name="Ear", graph_walks={"listen"}, initial_walk="listen")]

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear"], connections=[
            Connection(from_partition=CLIENT, to_partition="Ear", edge_name="audio_in",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
        ])

    def get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities,
                                      input_signals, model_kwargs=None):
        return _fwd()

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        return _fwd()

    def process_prompt(self, *a, **k):
        return {}

    def process_stream_chunk(self, chunk):
        return {"audio_in": chunk.tensors["audio_inputs"]}

    def postprocess(self, *a, **k):
        return b""

    def get_submodule(self, *a, **k):
        return None


def _fwd():
    return ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata(graph_walk="listen", is_prefill=False),
        inputs=[], unpersist_tensors=[],
    )


def _conductor(tmp_path, monkeypatch, model=None, **config) -> Conductor:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 1024, "node_groups": [{"node_names": ["ear"], "ranks": [0]}], **config,
    }))
    comm = _StubComm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    c = Conductor(model or _DuplexToy(), str(path), socket_path_prefix=str(tmp_path / "s"))
    c.comm = comm
    return c


def _run(c: Conductor, batches):
    """Run the real ``Conductor.run`` over ``batches``, one per iteration."""
    setup = [_msg(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in c.worker_ids]
    c.comm.batches.extend([setup, *batches])

    def poll():
        if not c.comm.batches:
            raise _ScriptDoneError

    c._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        c.run()


def _msg(kind, body):
    return ConductorMessage(message_type=kind, body=body)


def _new_request(rid):
    return _msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["audio"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    ))


def _info(uuid):
    return TensorPointerInfo(
        dims=[1280], dtype=torch.float32, nbytes=5120, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity=PREPROCESS,
    )


def _chunk(rid, *uuids, done=False, edge="audio_in", ended=None):
    """``uuids`` as items of ``edge``; ``done`` ends that stream, ``ended`` the ones it names."""
    signals = {edge: [_info(u) for u in uuids]} if uuids else {}
    ended = [edge] if done else list(ended or [])
    return _msg(ConductorMessageType.CLIENT_STREAM_INPUT,
                ClientStreamInput(request_id=rid, signals=signals, ended=ended))


def _sends(c, *kinds):
    out = []
    for entity, msg in c.comm.sent:
        kind = getattr(msg.message_type, "name", msg.message_type)
        if kind in kinds:
            out.append((kind, entity, msg.body))
    return out


def _stream_signals(c, rid):
    """(uuids, producer_done) of each INPUT_SIGNALS for ``rid``, in send order."""
    out = []
    for _kind, _entity, body in _sends(c, "INPUT_SIGNALS"):
        if body.request_id == rid:
            uuids = [info.uuid for edge in body.inputs for info in edge.tensor_info]
            out.append((uuids, set(body.producer_done)))
    return out


def _partition_done(c, rid):
    wg = c.node_walk_to_wg[("ear", "listen")].worker_graph_id
    return _msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
        request_id=rid, worker_graph_ids=[wg], is_first_tp_rank=True,
        partition_name="Ear", partition_done=True,
    ))


def _reads_done(rid, *entities):
    return [_msg(ConductorMessageType.READS_DONE, ReadsDone(request_id=rid, entity_id=e))
            for e in entities]


def test_a_chunk_reaches_the_consumer_as_a_streaming_edge_then_is_released(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [[_new_request("r")], [_chunk("r", 11, 12)]])

    sends = _sends(c, "INPUT_SIGNALS", "UNPERSIST_TENSORS")
    (kind, entity, body), (release_kind, release_entity, release) = sends
    assert (kind, entity, body.partition_name) == ("INPUT_SIGNALS", "worker_0", "Ear")
    (edge,) = body.inputs
    assert (edge.name, edge.next_node, edge.is_streaming) == ("audio_in", "ear", True)
    assert [info.uuid for info in edge.tensor_info] == [11, 12]
    # released after the send, with one reader each
    assert (release_kind, release_entity) == ("UNPERSIST_TENSORS", PREPROCESS)
    assert dict(release.uuid_to_ref_count) == {11: 1, 12: 1}
    assert c.requests["r"].persist_signal_ref_cnt == {}


def test_end_of_input_sends_producer_done_once_and_later_chunks_are_dropped(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [[_new_request("r")], [_chunk("r", 1)], [_chunk("r", done=True)],
             [_chunk("r", done=True)], [_chunk("r", 2)]])

    assert _stream_signals(c, "r") == [([1], set()), ([], {CLIENT})]


def test_end_of_input_reports_how_many_items_the_worker_was_sent(tmp_path, monkeypatch):
    # The consumer's stream buffer closes only once it has registered this
    # many items, so a done signal that overtook a chunk can't end it early.
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [[_new_request("r")], [_chunk("r", 1, 2)], [_chunk("r", 3, done=True)]])

    done = [body for _k, _e, body in _sends(c, "INPUT_SIGNALS")
            if body.request_id == "r" and body.producer_done]
    assert [dict(body.producer_done_items) for body in done] == [{"audio_in": 3}]


def test_a_chunk_and_the_end_in_one_message_arrive_in_that_order(tmp_path, monkeypatch):
    # The worker applies producer_done before a message's inputs, so the end
    # must travel in a message of its own, after the chunk.
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [[_new_request("r")], [_chunk("r", 5, done=True)]])

    assert _stream_signals(c, "r") == [([5], set()), ([], {CLIENT})]


def test_chunks_wait_for_admission_and_are_forwarded_in_order(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, max_concurrent_requests=1)
    _run(c, [
        [_new_request("a"), _new_request("b")],
        [_chunk("b", 1), _chunk("b", 2)],
        [_partition_done(c, "a")],
        _reads_done("a", PREPROCESS, "worker_0"),
    ])

    assert "b" in c.requests and c._held_stream_inputs == {}
    assert _stream_signals(c, "b") == [([1], set()), ([2], set())]
    # forwarded after b's own NEW_REQUEST
    kinds = [(k, body.request_id) for k, _e, body in _sends(c, "NEW_REQUEST", "INPUT_SIGNALS")]
    assert kinds.index(("NEW_REQUEST", "b")) < kinds.index(("INPUT_SIGNALS", "b"))


def test_held_chunks_of_an_aborted_queued_request_are_dropped(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, max_concurrent_requests=1)
    _run(c, [
        [_new_request("a"), _new_request("b")],
        [_chunk("b", 1)],
        [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("b"))],
    ])

    assert c._held_stream_inputs == {}
    assert _stream_signals(c, "b") == []


def test_chunks_for_a_draining_or_unknown_request_are_dropped(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [
        [_new_request("r")],
        [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r"))],
        [_chunk("r", 1), _chunk("never-seen", 2)],
    ])

    assert _stream_signals(c, "r") == []
    assert _stream_signals(c, "never-seen") == []


def test_completion_drains_the_workers_too(tmp_path, monkeypatch):
    # A chunk forwarded just before the model ended the request may still be
    # in a worker's read, so teardown waits for the workers' READS_DONE.
    c = _conductor(tmp_path, monkeypatch)
    _run(c, [[_new_request("r")], [_partition_done(c, "r")]])

    drains = {entity for _k, entity, body in _sends(c, "DRAIN_REQUEST") if body.request_id == "r"}
    assert drains == {"worker_0"}
    assert c.draining["r"].expected_acks == {PREPROCESS, "worker_0"}


def test_a_client_stream_needs_a_node_that_takes_it(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="audio_in"):
        _conductor(tmp_path, monkeypatch, model=_DuplexToy(consumer_input="something_else"))


def test_client_is_not_a_partition_name():
    with pytest.raises(ValueError, match="reserved"):
        PartitionTopology(partitions=[CLIENT])
    with pytest.raises(ValueError, match="only produces"):
        PartitionTopology(partitions=["A"], connections=[Connection(
            from_partition="A", to_partition=CLIENT, edge_name="x",
            chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
        )])


def test_stream_input_encodes_without_pickle():
    body = ClientStreamInput(request_id="r", signals={"audio_in": [_info(3)]}, ended=["audio_in"])
    msg = ConductorMessage(message_type=ConductorMessageType.CLIENT_STREAM_INPUT, body=body)
    # typed fields, not the pickle fallback; the envelope finds it by its tag
    assert wire._encoder(ClientStreamInput).__name__ == "_encode_dataclass"
    decoded = wire.decode(wire.encode(msg))
    assert decoded.body.request_id == "r" and decoded.body.ended == ["audio_in"]
    assert decoded.body.signals["audio_in"][0].uuid == 3


def _stub_worker(sbuf):
    """What ``Worker._process_new_inputs`` touches for a streaming-only signal."""
    reads = []
    req_info = SimpleNamespace(stream_buffers={"audio_in": sbuf})
    stub = SimpleNamespace(
        _draining_rids=set(), worker_id="worker_0", enable_nvtx=False,
        _rid=lambda rid: 7 if rid == "r" else None,
        request_state=SimpleNamespace(per_request_info={7: req_info}),
        tensor_manager=SimpleNamespace(
            # the worker also calls it, empty, for the non-streaming inputs
            start_read_tensors=lambda rid, edges, graph_walk=None: edges and reads.append(
                [info.uuid for edge in edges for info in edge.tensor_info]
            ) or [],
        ),
        wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
    )
    return stub, reads


def test_the_worker_buffers_client_chunks_and_closes_on_producer_done():
    sbuf = StreamBuffer(request_id=7, edge_name="audio_in", from_partition=CLIENT,
                        policy=FixedChunkPolicy(chunk_size=1))
    stub, reads = _stub_worker(sbuf)
    info = CurrentForwardPassInfo(
        request_id="r", graph_walk="listen", fwd_index=0, random_seed=0, max_tokens=8,
        partition_name="Ear",
    )
    chunk = GraphEdge(next_node="ear", name="audio_in", tensor_info=[_info(1), _info(2)],
                      is_streaming=True)

    Worker._process_new_inputs(stub, InputSignals(
        request_id="r", inputs=[chunk], request_info=info, partition_name="Ear",
    ))
    assert reads == [[1, 2]] and list(sbuf._tensor_ids_in_order) == [1, 2]
    assert not sbuf.producer_done

    Worker._process_new_inputs(stub, InputSignals(
        request_id="r", inputs=[], request_info=info, partition_name="Ear",
        producer_done={CLIENT}, producer_done_items={"audio_in": 2},
    ))
    assert sbuf.producer_done and sbuf._num_items_expected == 2


def test_a_queued_call_that_sends_too_much_is_turned_away(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, max_concurrent_requests=1)
    c.MAX_HELD_CLIENT_CHUNKS = 2
    _run(c, [[_new_request("a"), _new_request("b")], [_chunk("b", 1), _chunk("b", 2), _chunk("b", 3)]])

    assert c._held_stream_inputs == {} and all(w.request_id != "b" for w in c.waiting_queue)
    failed = [m.body for e, m in c.comm.sent if getattr(m, "message_type", None) == "request_failed"]
    assert [(f.request_id, f.status) for f in failed] == [("b", 503)]
    assert any(b.request_id == "b" for _k, _e, b in _sends(c, "REMOVE_REQUEST"))


def test_a_chunk_that_fails_fails_only_its_request(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch)
    real = c._send_inputs

    def flaky(request_id, *a, **k):
        if request_id == "bad":
            raise IndexError("no dims")
        return real(request_id, *a, **k)

    c._send_inputs = flaky
    _run(c, [[_new_request("bad"), _new_request("good")], [_chunk("bad", 1), _chunk("good", 2)]])

    assert _stream_signals(c, "good") == [([2], set())]
    assert c.draining["bad"].failure_error.startswith("client input failed: IndexError")
    assert c.requests["good"].streaming_connections[CLIENT, "Ear", "audio_in"].token_count == 1


def test_a_client_stream_has_one_consumer(tmp_path, monkeypatch):
    class _TwoEars(_DuplexToy):
        def get_graph_walk_graphs(self):
            walks = super().get_graph_walk_graphs()
            walks["listen"] = Loop(name="listen_loop", max_iters=1000, outputs=[], section=GraphNode(
                name="ear", input_names=["audio_in"], consumes_stream=True,
                outputs=[GraphEdge(next_node="ear2", name="x")],
            ))
            walks["listen2"] = GraphNode(name="ear2", input_names=["audio_in", "x"], outputs=[])
            return walks

        def get_partitions(self):
            return [PartitionDefinition(name="Ear", graph_walks={"listen", "listen2"}, initial_walk="listen")]

    with pytest.raises(ValueError, match="exactly one consuming node"):
        Conductor._find_client_streams(_TwoEars())


def _client_connection(edge_name, to_partition="Ear"):
    return Connection(from_partition=CLIENT, to_partition=to_partition, edge_name=edge_name,
                      chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1))


def test_client_streams_without_process_stream_chunk_fail_at_startup():
    class _NoHook(_DuplexToy):
        process_stream_chunk = Model.process_stream_chunk

    with pytest.raises(ValueError, match="process_stream_chunk"):
        Conductor._find_client_streams(_NoHook())


def test_a_client_stream_into_no_partition_fails_at_startup():
    class _Nowhere(_DuplexToy):
        def get_partition_topology(self):
            return PartitionTopology(partitions=["Ear"], connections=[_client_connection("audio_in", "Mouth")])

    with pytest.raises(ValueError, match="'Mouth', which is not a partition"):
        Conductor._find_client_streams(_Nowhere())


class _AudioAndVideo(_DuplexToy):
    """One partition whose loop consumes two client streams."""

    def get_graph_walk_graphs(self):
        return {"listen": Loop(name="listen_loop", max_iters=1000, outputs=[], section=GraphNode(
            name="ear", input_names=["audio_in", "video_in"], consumes_stream=True,
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="text", output_modality="text")],
        ))}

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear"], connections=[
            _client_connection("audio_in"), _client_connection("video_in"),
        ])


def test_two_client_streams_into_one_partition_route_and_end_independently(tmp_path, monkeypatch):
    """Connection state is per (from, to, edge): each stream keeps its own
    items and its own end, and ending one leaves the other open."""
    c = _conductor(tmp_path, monkeypatch, model=_AudioAndVideo())
    assert {name: s.partition for name, s in c._client_streams.items()} == {"audio_in": "Ear", "video_in": "Ear"}
    _run(c, [
        [_new_request("r")],
        [_chunk("r", 1, 2), _chunk("r", 3, edge="video_in")],
        [_chunk("r", ended=["video_in"])],
        [_chunk("r", 4), _chunk("r", 5, edge="video_in")],      # video's is late: dropped
        [_chunk("r", ended=["audio_in", "video_in"])],
    ])

    signals = [(body.inputs[0].name if body.inputs else None,
                [info.uuid for edge in body.inputs for info in edge.tensor_info],
                set(body.producer_done), dict(body.producer_done_items))
               for _k, _e, body in _sends(c, "INPUT_SIGNALS") if body.request_id == "r"]
    assert signals == [
        ("audio_in", [1, 2], set(), {}),
        ("video_in", [3], set(), {}),
        (None, [], {CLIENT}, {"video_in": 1}),              # ends video only
        ("audio_in", [4], set(), {}),
        (None, [], {CLIENT}, {"audio_in": 3}),              # then audio, once
    ]
    conns = c.requests["r"].streaming_connections
    assert conns[CLIENT, "Ear", "audio_in"].token_count == 3 and conns[CLIENT, "Ear", "video_in"].token_count == 1
    assert conns[CLIENT, "Ear", "audio_in"].producer_done and conns[CLIENT, "Ear", "video_in"].producer_done


def test_a_client_streams_consumer_must_be_its_partitions_and_the_only_one():
    """The worker routes a streamed edge to the one node, in any walk, that
    takes it; a consumer elsewhere would split the two."""
    class _ElsewhereToo(_DuplexToy):
        def get_graph_walk_graphs(self):
            walks = super().get_graph_walk_graphs()
            walks["speak"] = GraphNode(name="mouth", input_names=["audio_in"], outputs=[])
            return walks

    class _OnlyElsewhere(_DuplexToy):
        def get_graph_walk_graphs(self):
            return {"listen": GraphNode(name="ear", input_names=["x"], outputs=[]),
                    "speak": GraphNode(name="mouth", input_names=["audio_in"], outputs=[])}

    for model in (_ElsewhereToo(), _OnlyElsewhere()):
        with pytest.raises(ValueError, match="exactly one consuming node"):
            Conductor._find_client_streams(model)


def test_a_model_without_client_streams_is_not_inspected():
    class _Plain(_DuplexToy):
        def get_partition_topology(self):
            return PartitionTopology(partitions=["Ear"])

        def get_graph_walk_graphs(self):
            raise AssertionError("walked the graphs of a model with no client stream")

    assert Conductor._find_client_streams(_Plain()) == {}


@pytest.mark.parametrize("failed_here", [True, False])
def test_the_worker_acks_a_stream_it_has_no_buffer_for_unread(caplog, failed_here):
    """Its producer gets the ack it waits for to free the items. Expected
    only for a request whose ingest failed here; otherwise the stream was
    misrouted, which is worth a warning rather than a silent hang."""
    sbuf = StreamBuffer(request_id=7, edge_name="audio_in", from_partition=CLIENT,
                        policy=FixedChunkPolicy(chunk_size=1))
    stub, reads = _stub_worker(sbuf)
    stub.request_state.per_request_info[7].stream_buffers = {}
    stub.scheduler = SimpleNamespace(failed_rids={7} if failed_here else set())
    acked = []
    stub.tensor_manager.ack_unread_tensors = lambda rid, edges: acked.append((rid, edges))
    info = CurrentForwardPassInfo(request_id="r", graph_walk="listen", fwd_index=0, random_seed=0,
                                  max_tokens=8, partition_name="Ear")
    chunk = GraphEdge(next_node="ear", name="audio_in", tensor_info=[_info(1)], is_streaming=True)
    with caplog.at_level(logging.WARNING):
        Worker._process_new_inputs(stub, InputSignals(request_id="r", inputs=[chunk], request_info=info,
                                                      partition_name="Ear"))
    assert reads == []
    assert acked == [(7, [chunk])]
    assert ("no stream buffer" in caplog.text) is not failed_here


def test_a_client_end_closes_only_the_streams_it_counts():
    """A client's producer_done names the streams it ends; the others stay open."""
    audio, video = (StreamBuffer(request_id=7, edge_name=name, from_partition=CLIENT,
                                 policy=FixedChunkPolicy(chunk_size=1)) for name in ("audio_in", "video_in"))
    stub, _reads = _stub_worker(audio)
    stub.request_state.per_request_info[7].stream_buffers = {"audio_in": audio, "video_in": video}
    info = CurrentForwardPassInfo(request_id="r", graph_walk="listen", fwd_index=0, random_seed=0,
                                  max_tokens=8, partition_name="Ear")

    Worker._process_new_inputs(stub, InputSignals(
        request_id="r", inputs=[], request_info=info, partition_name="Ear",
        producer_done={CLIENT}, producer_done_items={"video_in": 2},
    ))
    assert video.producer_done and video._num_items_expected == 2
    assert not audio.producer_done


def test_a_producer_partitions_end_leaves_a_colocated_consumers_stream_open():
    """Two consumer partitions on one worker, fed by one producer: the end
    sent for one counts only its edges, so the other's stays open (and its
    own end later brings its own count)."""
    to_a = StreamBuffer(request_id=7, edge_name="to_a", from_partition="P", policy=FixedChunkPolicy(chunk_size=1))
    to_b = StreamBuffer(request_id=7, edge_name="to_b", from_partition="P", policy=FixedChunkPolicy(chunk_size=1))
    stub, _reads = _stub_worker(to_a)
    stub.request_state.per_request_info[7].stream_buffers = {"to_a": to_a, "to_b": to_b}
    info = CurrentForwardPassInfo(request_id="r", graph_walk="listen", fwd_index=0, random_seed=0,
                                  max_tokens=8, partition_name="A")

    Worker._process_new_inputs(stub, InputSignals(
        request_id="r", inputs=[], request_info=info, partition_name="A",
        producer_done={"P"}, producer_done_items={"to_a": 4},
    ))
    assert to_a.producer_done and to_a._num_items_expected == 4
    assert not to_b.producer_done


class _Bounded(_DuplexToy):
    """``audio_in`` with the given limits."""

    def __init__(self, limits):
        super().__init__()
        self._limits = limits

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear"], connections=[Connection(
            from_partition=CLIENT, to_partition="Ear", edge_name="audio_in",
            chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1), limits=self._limits,
        )])


def _consumed(rid, n):
    """The consumer worker's report after a step: it has consumed ``n`` audio_in items."""
    return _msg(ConductorMessageType.CLIENT_STREAM_CONSUMED, ClientStreamConsumed(
        request_ids=[rid], edge_names=["audio_in"], consumed=[n],
    ))


def test_a_call_too_far_ahead_of_its_consumer_is_refused_with_a_400(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, model=_Bounded(ClientStreamLimits(max_pending_items=2)))
    _run(c, [
        [_new_request("r")],
        [_chunk("r", 1, 2)],          # 2 pending: at the limit
        [_consumed("r", 1)],
        [_chunk("r", 3)],             # 2 pending again
        [_chunk("r", 4)],             # 3: refused
    ])

    assert _stream_signals(c, "r") == [([1, 2], set()), ([3], set())]
    drain = c.draining["r"]
    assert drain.failure_status == 400
    assert "client_streams.audio_in.max_pending_items" in drain.failure_error


def test_keep_newest_sends_what_fits_and_holds_the_newest_item_for_room(tmp_path, monkeypatch):
    """What fits is sent; past the limit the newest item waits in the
    conductor and the others, and any older one waiting, are freed at once;
    room (a consumption report) or the stream's end sends the one waiting."""
    c = _conductor(tmp_path, monkeypatch, model=_Bounded(
        ClientStreamLimits(max_pending_items=2, pending_overflow="keep_newest"),
    ))
    _run(c, [
        [_new_request("r")],
        [_chunk("r", 1, 2)],          # sent: 2 pending
        [_chunk("r", 3)],             # waits
        [_chunk("r", 4, 5)],          # 5 waits; 3 and 4 are dropped
        [_consumed("r", 1)],       # room for one: 5 goes
        [_chunk("r", 6)],             # waits
        [_chunk("r", done=True)],     # the end sends it first
    ])

    assert _stream_signals(c, "r") == [([1, 2], set()), ([5], set()), ([6], set()), ([], {CLIENT})]
    done = [b for _k, _e, b in _sends(c, "INPUT_SIGNALS") if b.request_id == "r" and b.producer_done]
    assert [dict(b.producer_done_items) for b in done] == [{"audio_in": 4}]
    freed = {u: n for _k, _e, b in _sends(c, "UNPERSIST_TENSORS") for u, n in b.uuid_to_ref_count.items()}
    assert {u: freed[u] for u in (3, 4)} == {3: 0, 4: 0}       # no reader: freed now
    assert all(freed[u] == 1 for u in (1, 2, 5, 6))
    conn = c.requests["r"].streaming_connections[CLIENT, "Ear", "audio_in"]
    assert conn.waiting is None and conn.token_count == 4


class _RuntimeTensorManager:
    """What a graph runtime's routing calls on the tensor manager, with the
    real refcount methods, as in ``test_stream_completion_order.py``."""
    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts
    set_persist = TensorCommunicationManager.set_persist
    dereference = TensorCommunicationManager.dereference
    dereference_batch = TensorCommunicationManager.dereference_batch
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    stamps_shm_placement = False

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def get_tensor(self, uuid):
        return self.tensor_store.get_tensor(uuid)

    def refresh_shm_placement(self, edges):
        pass

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)
        elif registered is None:
            self.tensor_store.bookkeeping.forget_tensor(uuid)


class _LoopConsumer:
    """The worker that consumes ``audio_in``: a real graph runtime (either one)
    whose ``ear`` reruns inside a dynamic loop, as a node reading its client
    stream does, so no pass finishes its worker graph; and after each pass the
    real ``Worker._stream_and_send``, with what it reads stubbed."""

    def __init__(self, kind):
        loop = Loop(name="listen_loop", max_iters=1000, outputs=[], section=GraphNode(
            name="ear", input_names={"state"}, outputs=[GraphEdge(name="state", next_node="ear")],
        ))
        common = dict(
            my_worker_id="worker_0",
            my_worker_graphs=[WorkerGraph(section=loop, graph_walks={"listen"}, ranks=[0], worker_graph_id=0)],
            all_wg_ids_to_graph_walks={0: {"listen"}}, all_wg_ids_to_dyn_loops={0: {"listen_loop"}},
            all_wg_ids_to_nodes={0: {"ear"}}, node_to_partition={"ear": "Ear"},
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        )
        if kind == "python":
            self.book = PythonTensorBookkeeping()
            self.rt = PythonGraphRuntime(tensor_manager=_RuntimeTensorManager(self.book), communicator=None,
                                         **common)
        else:
            pytest.importorskip("mstar_rust")
            from mstar.communication.tensor_store import RustTensorBookkeeping
            from mstar.graph.runtime.rust import RustGraphRuntime

            self.book = RustTensorBookkeeping()
            self.rt = RustGraphRuntime(bookkeeping=self.book, **common)
        self.rid = self.rt.add_request(
            request_id="r", partition="Ear", graph_walk="listen", partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
        )
        # what each send_outputs returned: the rids it sent WORKER_GRAPHS_DONE for
        self.graphs_done = []
        send_outputs = self.rt.send_outputs

        def recording_send_outputs(send_input):
            done = send_outputs(send_input)
            self.graphs_done.extend(done)
            return done

        self.rt.send_outputs = recording_send_outputs
        self.sbuf = StreamBuffer(request_id=self.rid, edge_name="audio_in", from_partition=CLIENT,
                                 policy=FixedChunkPolicy(chunk_size=1))
        fwd = CurrentForwardPassInfo(request_id="r", graph_walk="listen", fwd_index=0, random_seed=0,
                                     max_tokens=8, partition_name="Ear")
        w = Worker.__new__(Worker)
        w._graph_runtime = self.rt
        w._phase_period = 0
        w.enable_prof = False
        w.communicator = _StubComm()
        w.parallel_leader_nodes = {"ear"}
        w._paced_streams = frozenset({"audio_in"})
        w.tensor_manager = SimpleNamespace(dereference_batch_uniform=lambda uuids: None)
        w.request_state = SimpleNamespace(
            per_request_info={self.rid: SimpleNamespace(stream_buffers={"audio_in": self.sbuf})},
            get_fwd_info=lambda rid, partition: fwd,
            get_pending_publish_info=lambda rid, partition: {},
            buffer_publish_info=lambda rid, partition, published: None,
            flush_publish_info=lambda rid, partition: None,
        )
        self.worker = w
        self.batch = SimpleNamespace(node_name="ear", partition="Ear",
                                     node_batch=SimpleNamespace(resource_publish_info={}))
        self.engine = SimpleNamespace(materialize_publish=lambda node_batch, rids: None)
        self._uuids = iter(range(1000, 2000))
        block = ColumnarEdgeSpecs.empty()
        block.add(self.rid, "state", [], False, next_node="ear")
        self.rt.ingest_inputs_batch(block)

    def receive(self, *uuids):
        """Client items the conductor forwarded, landed in the stream buffer."""
        for uuid in uuids:
            self.sbuf.pre_read_register(uuid)
            self.sbuf.put(uuid, torch.zeros(1))

    def step(self, take=1):
        """One pass of ``ear`` that read ``take`` items: the messages it sent the conductor."""
        for _ in range(take):
            assert self.sbuf.has_chunk_ready()
            self.sbuf.pop_chunk()
        assert self.rt.pop_rids("ear", "listen", [self.rid], check_ready=True) is not None, "the loop stopped"
        uuid = next(self._uuids)
        self.book.put_tensor(uuid, _info(uuid))
        self.book.increment_ref(uuid, 1)  # the worker's safety hold
        out = self.rt.complete_and_route_batch(RouteInput(
            partition="Ear", graph_walk="listen", node_name="ear", output_signals=["state"],
            wg_ids=ParallelList([self.rid], [0]), tensors=[uuid], num_tensors=[1],
        ))
        sent = self.worker.communicator.sent
        sent.clear()
        Worker._stream_and_send(
            self.worker, self.batch, [self.rid],
            SimpleNamespace(flat_uuids=[uuid], flat_rids=[self.rid], signal_idxs=[0]),
            out, ["state"], engine=self.engine,
        )
        return [msg for entity, msg in sent if entity == "conductor"]


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_a_consumer_reading_inside_a_loop_makes_room_as_it_reads(tmp_path, monkeypatch, kind):
    """The loop sends no WORKER_GRAPHS_DONE until it ends, so the client's room
    comes from the worker's report after each step: a call that sends three
    times max_pending_items, never more than that ahead, is never refused."""
    consumer = _LoopConsumer(kind)
    reports = []
    for first in (1, 3, 5):
        consumer.receive(first, first + 1)
        reports.append(consumer.step() + consumer.step())
    assert consumer.graphs_done == [], "a pass finished the worker graph, so this is not the loop's case"
    rows = [(m.body.request_ids, m.body.edge_names, m.body.consumed) for batch in reports for m in batch]
    assert rows == [(["r"], ["audio_in"], [n]) for n in range(1, 7)]

    c = _conductor(tmp_path, monkeypatch, model=_Bounded(ClientStreamLimits(max_pending_items=2)))
    _run(c, [
        [_new_request("r")],
        [_chunk("r", 1, 2)], reports[0],
        [_chunk("r", 3, 4)], reports[1],
        [_chunk("r", 5, 6)], reports[2],
    ])

    assert "r" not in c.draining
    assert _stream_signals(c, "r") == [([1, 2], set()), ([3, 4], set()), ([5, 6], set())]
    assert c.requests["r"].streaming_connections[CLIENT, "Ear", "audio_in"].consumed_count == 6


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_a_consumer_that_stops_reading_still_gets_its_client_refused(tmp_path, monkeypatch, kind):
    """Its loop keeps stepping, but a step that read nothing reports nothing,
    so the client's room does not grow."""
    consumer = _LoopConsumer(kind)
    consumer.receive(1, 2)
    read_one = consumer.step()
    stalled = consumer.step(take=0) + consumer.step(take=0)
    assert [m.body.consumed for m in read_one] == [[1]] and stalled == []

    c = _conductor(tmp_path, monkeypatch, model=_Bounded(ClientStreamLimits(max_pending_items=2)))
    _run(c, [
        [_new_request("r")],
        [_chunk("r", 1, 2)], read_one,
        [_chunk("r", 3)],             # 2 pending again
        stalled,
        [_chunk("r", 4)],             # 3: refused
    ])

    assert _stream_signals(c, "r") == [([1, 2], set()), ([3], set())]
    assert c.draining["r"].failure_status == 400


def test_a_step_reports_only_moved_paced_counts_its_graphs_done_did_not_carry():
    """A WORKER_GRAPHS_DONE already carries every count, an unpaced stream is
    not the conductor's to pace, and only the lockstep leader reports."""
    paced, other = (StreamBuffer(request_id=7, edge_name=name, from_partition=CLIENT,
                                 policy=FixedChunkPolicy(chunk_size=1)) for name in ("audio_in", "video_in"))
    comm = _StubComm()
    stub = SimpleNamespace(
        parallel_leader_nodes={"ear"}, _paced_streams=frozenset({"audio_in"}), communicator=comm,
        _rid_str=lambda rid: "r",
        request_state=SimpleNamespace(per_request_info={
            7: SimpleNamespace(stream_buffers={"audio_in": paced, "video_in": other}),
        }),
    )

    Worker._report_paced_consumption(stub, "ear", [(7, {"audio_in": 3, "video_in": 9})], [7])
    Worker._report_paced_consumption(stub, "ear", [(7, {"audio_in": 3, "video_in": 10})], [])
    assert comm.sent == [] and paced.consumed_reported == 3

    stub.parallel_leader_nodes = set()  # a follower rank of ear's instance
    Worker._report_paced_consumption(stub, "ear", [(7, {"audio_in": 4})], [])
    assert comm.sent == []

    stub.parallel_leader_nodes = {"ear"}
    Worker._report_paced_consumption(stub, "ear", [(7, {"audio_in": 4, "video_in": 11})], [])
    [(entity, msg)] = comm.sent
    assert (entity, msg.message_type) == ("conductor", ConductorMessageType.CLIENT_STREAM_CONSUMED)
    assert (msg.body.request_ids, msg.body.edge_names, msg.body.consumed) == (["r"], ["audio_in"], [4])


def test_the_worker_paces_the_client_streams_it_consumes_with_a_pending_bound():
    open_ = _Bounded(ClientStreamLimits()).get_partition_topology()
    bounded = _Bounded(ClientStreamLimits(max_pending_items=4)).get_partition_topology()
    override = {"client_streams": {"audio_in": {"max_pending_items": 4}}}

    assert Worker._find_paced_streams(open_, open_.connections, {}) == frozenset()
    assert Worker._find_paced_streams(bounded, bounded.connections, {}) == {"audio_in"}
    assert Worker._find_paced_streams(open_, open_.connections, override) == {"audio_in"}
    assert Worker._find_paced_streams(bounded, [], {}) == frozenset()  # consumed on another worker


def test_a_consumption_report_encodes_without_pickle():
    body = ClientStreamConsumed(request_ids=["r", "s"], edge_names=["audio_in", "audio_in"], consumed=[3, 5])
    msg = ConductorMessage(message_type=ConductorMessageType.CLIENT_STREAM_CONSUMED, body=body)
    assert wire._encoder(ClientStreamConsumed).__name__ == "_encode_dataclass"
    decoded = wire.decode(wire.encode(msg)).body
    assert (decoded.request_ids, decoded.edge_names, decoded.consumed) == (["r", "s"], ["audio_in"] * 2, [3, 5])


def test_a_consumption_report_for_a_finished_or_unknown_request_is_ignored(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, model=_Bounded(ClientStreamLimits(max_pending_items=2)))
    _run(c, [
        [_new_request("r")],
        [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r"))],
        [_consumed("r", 1), _consumed("never-seen", 1)],
    ])

    assert c.requests["r"].streaming_connections[CLIENT, "Ear", "audio_in"].consumed_count == 0
    assert "never-seen" not in c.requests


def test_client_stream_limits_take_deployment_overrides():
    topology = _Bounded(ClientStreamLimits(max_items=10)).get_partition_topology()

    assert client_stream_limits(topology, None)["audio_in"].max_items == 10
    limits = client_stream_limits(topology, {"client_streams": {"audio_in": {
        "max_items": 6, "max_pending_items": 8, "pending_overflow": "keep_newest",
    }}})["audio_in"]
    assert (limits.max_items, limits.max_pending_items, limits.pending_overflow) == (6, 8, "keep_newest")


@pytest.mark.parametrize("max_items", [11, None])
def test_a_deployment_cannot_raise_or_lift_a_declared_bound(max_items):
    """A declared bound is what the model's consumer was sized for."""
    topology = _Bounded(ClientStreamLimits(max_items=10)).get_partition_topology()
    with pytest.raises(ValueError, match="may lower the model's bound of 10"):
        client_stream_limits(topology, {"client_streams": {"audio_in": {"max_items": max_items}}})


def _windowed(policy_factory):
    return PartitionTopology(partitions=["Ear"], connections=[Connection(
        from_partition=CLIENT, to_partition="Ear", edge_name="audio_in", chunk_policy_factory=policy_factory,
    )])


@pytest.mark.parametrize("policy_factory, window", [
    (lambda: FixedChunkPolicy(chunk_size=4), 4),
    (lambda: SlidingWindowChunkPolicy(window=28, stride=7), 28),
    (lambda: LeftContextChunkPolicy(chunk=8, left_context=2), 8 + 2),
    (lambda: RampChunkPolicy(first_chunk=2, chunk_size=4, growth=2.0, max_chunk=16), 16),
    (lambda: RampChunkPolicy(first_chunk=6, chunk_size=4), 6),
    # windows 4, 4+8, 12+16, then 25+25 once the context is full
    (lambda: ScheduledLeftContextChunkPolicy(schedule=(4, 8, 16), chunk=25, left_context=25), 50),
])
def test_max_pending_items_below_a_chunk_window_fails_loading(policy_factory, window):
    """A consumer held to fewer items than its widest chunk never forms that
    chunk, so it never reports room: the stream would stall (a call held
    forever under keep_newest, refused under refuse). The error names both."""
    topology = _windowed(policy_factory)
    fits = client_stream_limits(topology, {"client_streams": {"audio_in": {"max_pending_items": window}}})
    assert fits["audio_in"].max_pending_items == window
    with pytest.raises(ValueError, match=rf"'audio_in' stream's max_pending_items \({window - 1}\) is below the "
                                         rf"{window} items"):
        client_stream_limits(topology, {"client_streams": {"audio_in": {"max_pending_items": window - 1}}})


def test_max_pending_items_cannot_bound_chunks_that_grow_without_a_cap():
    topology = _windowed(lambda: RampChunkPolicy(first_chunk=2, chunk_size=4, growth=2.0))
    assert client_stream_limits(topology, None)["audio_in"].max_pending_items is None
    with pytest.raises(ValueError, match="grow without bound"):
        client_stream_limits(topology, {"client_streams": {"audio_in": {"max_pending_items": 1000}}})


@pytest.mark.parametrize("block, match", [
    ({"video_in": {"max_items": 1}}, "unknown client stream"),
    ({"audio_in": {"max_itmes": 1}}, "takes"),
    ({"audio_in": {"max_items": 0}}, "positive integer"),
    ({"audio_in": {"max_pending_items": True}}, "positive integer"),
    ({"audio_in": {"pending_overflow": "block"}}, "pending_overflow"),
    ({"audio_in": {"pending_overflow": "drop_oldest"}}, "pending_overflow"),
    ({"audio_in": 5}, "takes"),
])
def test_a_bad_client_streams_block_fails_loading(block, match):
    topology = _Bounded(ClientStreamLimits()).get_partition_topology()
    with pytest.raises(ValueError, match=match):
        client_stream_limits(topology, {"client_streams": block})


def test_only_a_client_stream_takes_limits():
    with pytest.raises(ValueError, match="not a client stream"):
        PartitionTopology(partitions=["A", "B"], connections=[Connection(
            from_partition="A", to_partition="B", edge_name="x",
            chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1), limits=ClientStreamLimits(max_items=1),
        )])
