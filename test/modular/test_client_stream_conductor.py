"""Client streams: a ``Connection`` from ``CLIENT`` feeds a running request.

The conductor forwards each ``CLIENT_STREAM_INPUT`` chunk to the consuming node's
workers as a streaming ``INPUT_SIGNALS`` edge and releases its persist hold
with the reader count, drops them for a request that has ended, and closes
the stream with ``producer_done`` from ``CLIENT`` when the client's input
ends. At ``max_concurrent_requests`` a call with open input is refused rather
than queued.

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
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.base import ForwardPassArgs, Model  # noqa: E402
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.streaming.stream_buffer import StreamBuffer  # noqa: E402
from mstar.streaming.topology import (  # noqa: E402
    CLIENT,
    ClientStreamLimits,
    Connection,
    PartitionTopology,
    client_stream_limits,
)
from mstar.utils.ipc_format import (  # noqa: E402
    AbortRequest,
    ClientStreamInput,
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequestConductor,
    ReadsDone,
    SetupDone,
    WorkerGraphsDone,
)
from mstar.validation.client_streams import check_client_streams  # noqa: E402
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


def _new_request(rid, client_input=None):
    """An opening; without ``client_input`` its client streams stay open (a live call)."""
    return _msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["audio"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
        client_input=client_input,
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


def test_a_one_shot_waits_for_a_slot_and_its_opening_input_follows_admission(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, max_concurrent_requests=1)
    one_shot = ClientStreamInput(request_id="b", signals={"audio_in": [_info(1)]}, ended=["audio_in"])
    _run(c, [
        [_new_request("a"), _new_request("b", one_shot)],
        [_partition_done(c, "a")],
        _reads_done("a", PREPROCESS, "worker_0"),
    ])

    assert "b" in c.requests
    assert _stream_signals(c, "b") == [([1], set()), ([], {CLIENT})]
    # forwarded after b's own NEW_REQUEST
    kinds = [(k, body.request_id) for k, _e, body in _sends(c, "NEW_REQUEST", "INPUT_SIGNALS")]
    assert kinds.index(("NEW_REQUEST", "b")) < kinds.index(("INPUT_SIGNALS", "b"))


@pytest.mark.parametrize("ended", [[], ["audio_in"]])
def test_a_call_with_open_input_is_refused_at_the_cap(tmp_path, monkeypatch, ended):
    """A live call that waited for a slot would start, and stay, that far
    behind its client; one stream left open is enough."""
    c = _conductor(tmp_path, monkeypatch, model=_AudioAndVideo(), max_concurrent_requests=1)
    _run(c, [[_new_request("a"), _new_request("b", ClientStreamInput(request_id="b", ended=ended))]])

    assert "b" not in c.requests and c.waiting_queue == []
    failed = [m.body for e, m in c.comm.sent if getattr(m, "message_type", None) == "request_failed"]
    assert [(f.request_id, f.status) for f in failed] == [("b", 503)]
    assert any(b.request_id == "b" for _k, _e, b in _sends(c, "REMOVE_REQUEST"))


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
    stub = Worker.__new__(Worker)
    vars(stub).update(
        _reads_done_sent=set(), worker_id="worker_0", enable_nvtx=False,
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


def test_a_chunk_that_fails_fails_only_its_request(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch)
    real = c._send_inputs

    def flaky(request_id, *a, **k):
        if request_id == "bad":
            raise IndexError("/srv/weights/secret no dims")
        return real(request_id, *a, **k)

    c._send_inputs = flaky
    _run(c, [[_new_request("bad"), _new_request("good")], [_chunk("bad", 1), _chunk("good", 2)]])

    assert _stream_signals(c, "good") == [([2], set())]
    error = c.draining["bad"].failure_error
    assert error.startswith("client input failed: internal error") and "secret" not in error
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

    assert _refused(_TwoEars(), "exactly one consuming node")


def _refused(model, text, config=None) -> bool:
    """The deployment is refused at startup, for ``text`` alone."""
    (problem,) = check_client_streams(model, config or {})
    return text in problem


def _client_connection(edge_name, to_partition="Ear"):
    return Connection(from_partition=CLIENT, to_partition=to_partition, edge_name=edge_name,
                      chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1))


def test_client_streams_without_process_stream_chunk_fail_at_startup():
    class _NoHook(_DuplexToy):
        process_stream_chunk = Model.process_stream_chunk

    assert _refused(_NoHook(), "process_stream_chunk")


def test_a_client_stream_into_no_partition_fails_at_startup():
    class _Nowhere(_DuplexToy):
        def get_partition_topology(self):
            return PartitionTopology(partitions=["Ear"], connections=[_client_connection("audio_in", "Mouth")])

    assert _refused(_Nowhere(), "'Mouth', which is not a partition")


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
        assert _refused(model, "exactly one consuming node")


def test_a_model_without_client_streams_is_not_inspected():
    class _Plain(_DuplexToy):
        def get_partition_topology(self):
            return PartitionTopology(partitions=["Ear"])

        def get_graph_walk_graphs(self):
            raise AssertionError("walked the graphs of a model with no client stream")

    assert Conductor._find_client_streams(_Plain()) == {} and check_client_streams(_Plain(), {}) == []


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
    # The ack goes on the wire, so it names the request by its string id,
    # never by this worker's handle.
    assert acked == [("r", [chunk])] and isinstance(acked[0][0], str)
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


def test_client_stream_limits_take_deployment_overrides():
    topology = _Bounded(ClientStreamLimits(max_items=10)).get_partition_topology()

    assert client_stream_limits(topology, None)["audio_in"].max_items == 10
    limits = client_stream_limits(topology, {"client_streams": {"audio_in": {"max_items": 6}}})["audio_in"]
    assert limits.max_items == 6


@pytest.mark.parametrize("max_items", [11, None])
def test_a_deployment_cannot_raise_or_lift_a_declared_bound(max_items):
    """A declared bound is what the model's consumer was sized for."""
    assert _refused(_Bounded(ClientStreamLimits(max_items=10)), "may lower the model's bound of 10",
                    {"client_streams": {"audio_in": {"max_items": max_items}}})


@pytest.mark.parametrize("config, match", [
    ({"client_streams": {"video_in": {"max_items": 1}}}, "unknown client stream"),
    ({"client_streams": {"audio_in": {"max_itmes": 1}}}, "takes"),
    ({"client_streams": {"audio_in": {"max_items": 0}}}, "positive integer"),
    ({"client_streams": {"audio_in": {"max_items": True}}}, "positive integer"),
    ({"client_streams": {"audio_in": 5}}, "takes"),
    ({"client_streams": ["audio_in"]}, "must map client stream names"),
    *(({"max_call_input_mb": mb}, "max_call_input_mb") for mb in ("1e3", 0, -4, True, float("nan"), float("inf"))),
])
def test_bad_client_stream_settings_fail_startup(config, match):
    assert _refused(_Bounded(ClientStreamLimits()), match, config)


def test_only_a_client_stream_takes_limits():
    with pytest.raises(ValueError, match="not a client stream"):
        PartitionTopology(partitions=["A", "B"], connections=[Connection(
            from_partition="A", to_partition="B", edge_name="x",
            chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1), limits=ClientStreamLimits(max_items=1),
        )])
