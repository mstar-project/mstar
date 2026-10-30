"""Client streams, API-server side: appends reach the consumer in order.

``/generate/ws`` keeps a client-stream request's input open and feeds its
``append`` messages to ``APIServer.stream_input``; the preprocess thread turns
each append into stream items with ``Model.process_stream_chunk``, stores
them, and sends them as ``INPUT_SIGNALS`` straight to the ranks the
conductor's ``CLIENT_STREAM_ROUTE`` names, then each stream's end with its
count. Tensors ``process_prompt`` returns under a client edge's name become
the stream's first chunk, so a one-shot request is a stream of one chunk.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from api_server_helpers import make_test_api_server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server import entrypoint  # noqa: E402
from mstar.api_server.data_worker import PreprocessWorkerThread  # noqa: E402
from mstar.api_server.entrypoint import PendingRequest  # noqa: E402
from mstar.api_server.request_types import (  # noqa: E402
    APIServerMessage,
    ClientAppend,
    PreprocessInput,
    RequestComplete,
)
from mstar.communication.communicator import OutboxCommunicator  # noqa: E402
from mstar.conductor.request_info import PartitionDefinition  # noqa: E402
from mstar.graph.base import GraphNode, TensorPointerInfo  # noqa: E402
from mstar.model.base import TensorAndMetadata  # noqa: E402
from mstar.profile.format import RequestProfile, RequestTiming  # noqa: E402
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.streaming.topology import (  # noqa: E402
    CLIENT,
    ClientStreamLimits,
    Connection,
    PartitionTopology,
)
from mstar.utils.ipc_format import (  # noqa: E402
    ClientStreamRoute,
    ConductorMessageType,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)

# the end of every stream _ListeningModel takes
END = frozenset({"audio_in"})


@pytest.fixture(autouse=True)
def _headerless_audio(monkeypatch):
    """These tests' audio files are raw float32 with no header to probe; the
    media caps have their own tests (test_request_media_totals.py)."""
    monkeypatch.setattr("mstar.api_server.data_worker.check_audio_duration", lambda filepath: None)


class _ListeningModel:
    """Raw float32 files in; ``audio_in`` stream items out."""

    def __init__(self):
        self.chunk_calls = []

    def max_input_files(self):
        return {}

    def kv_labels(self):
        return {}

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear"], connections=[
            Connection(from_partition=CLIENT, to_partition="Ear", edge_name="audio_in",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
        ])

    def get_graph_walk_graphs(self):
        streams = [conn.edge_name for conn in self.get_partition_topology().client_connections()]
        return {"listen": GraphNode(name="ear", input_names=streams, consumes_stream=True, outputs=[])}

    def get_partitions(self):
        return [PartitionDefinition(name="Ear", graph_walks={"listen"}, initial_walk="listen")]

    def load_audio(self, filepath, device):
        data = torch.frombuffer(bytearray(open(filepath, "rb").read()), dtype=torch.float32)
        return TensorAndMetadata(data=data.clone(), metadata={"sample_rate": 16000})

    def process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, **kwargs):
        out = {"text_inputs": [torch.tensor([1, 2, 3])]}
        if tensors and tensors.get("audio_inputs"):
            out["audio_in"] = tensors["audio_inputs"]
        return out

    def process_stream_chunk(self, chunk):
        self.chunk_calls.append(chunk)
        return {"audio_in": chunk.tensors.get("audio_inputs", [])}


class _TensorManager:
    def __init__(self):
        self.uuid = 0
        self.persisted = set()
        self.registered = []
        self.refs: dict[int, int] = {}

    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        out = {}
        for name, items in tensors.items():
            out[name] = []
            for item in items:
                self.uuid += 1
                out[name].append(TensorPointerInfo(
                    dims=list(item.shape), dtype=item.dtype, nbytes=item.nbytes, address=0,
                    stride=list(item.stride()), uuid=self.uuid, source_session_id="s",
                    source_entity="api_server_preprocess_worker",
                ))
        return out

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        self.registered.extend(info.uuid for info in tensor_infos)

    def set_persist(self, uuid, persist):
        (self.persisted.add if persist else self.persisted.discard)(uuid)

    def increment_ref_batch_uniform(self, uuids, n):
        for uuid in uuids:
            self.refs[uuid] = self.refs.get(uuid, 0) + n

    def dereference_batch_uniform(self, uuids, n=1):
        for uuid in uuids:
            self.refs[uuid] -= n

    # what the thread's loop polls between items
    def get_ready_tensors(self):
        return {}

    def take_failed_reads(self):
        return []

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        pass


def _thread(model=None, is_pending=None, model_config=None, routed=()):
    """``routed``: the requests whose route the conductor sends once their
    NEW_REQUEST has gone, as its dispatch does."""
    sent = []
    tm = _TensorManager()
    queues = {name: queue.Queue() for name in (
        "in_queue", "result_tensor_queue", "out_queue", "profile_queue", "cleanup_request_queue",
        "abort_request_queue", "reads_done_queue", "discard_tensor_queue",
    )}
    wt = PreprocessWorkerThread(
        **queues, stop_event=threading.Event(),
        communicator=OutboxCommunicator(SimpleNamespace(
            codec=SimpleNamespace(encode=lambda msg: msg),  # sends arrive as objects
            try_send_bytes=lambda entity, msg: sent.append((entity, msg)) or True,
            get_all_new_messages=_routed_once_opened(sent, *routed),
        )),
        tensor_manager=tm, model=model or _ListeningModel(), is_pending=is_pending,
        model_config=model_config,
    )
    return wt, sent, tm


def _audio_file(tmp_path, name, samples):
    path = tmp_path / name
    path.write_bytes(torch.arange(samples, dtype=torch.float32).numpy().tobytes())
    return str(path)


def _bodies(sent, kind):
    return [msg.body for _entity, msg in sent if msg.message_type == kind]


def _route_msg(rid="r", **workers):
    """The conductor's route for ``rid``; ``audio_in`` to ``w0`` by default."""
    return WorkerMessage(WorkerMessageType.CLIENT_STREAM_ROUTE, ClientStreamRoute(
        request_id=rid, workers=workers or {"audio_in": ["w0"]},
    ))


def _route(wt, rid="r", **workers):
    wt._route_client_streams(_route_msg(rid, **workers).body)


def _routed_once_opened(sent, *rids):
    """An inbox that delivers each of ``rids``' routes once its NEW_REQUEST
    has gone, as the conductor's dispatch does."""
    due = list(rids)

    def inbox():
        opened = {b.request_id for b in _bodies(sent, ConductorMessageType.NEW_REQUEST)}
        ready = [rid for rid in due if rid in opened]
        for rid in ready:
            due.remove(rid)
        return [_route_msg(rid) for rid in ready]

    return inbox


def _signals(sent, rid="r"):
    """Each INPUT_SIGNALS for ``rid`` in send order: (rank, edge, item dims,
    producer_done_items), the edge None for an end."""
    out = []
    for entity, msg in sent:
        if msg.message_type == WorkerMessageType.INPUT_SIGNALS and msg.body.request_id == rid:
            body = msg.body
            (edge,) = body.inputs or [None]
            out.append((
                entity, edge.name if edge else None,
                [info.dims for info in edge.tensor_info] if edge else [],
                dict(body.producer_done_items),
            ))
    return out


def _open(wt, files=None, ends=frozenset()):
    wt._process_input(PreprocessInput(
        request_id="r", text="hi", file_paths=files, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={"voice": "a"}, ends=ends,
    ))


def test_an_append_goes_straight_to_each_rank_of_its_consumer(tmp_path):
    """One INPUT_SIGNALS per rank, as a producer worker sends; each rank is a
    reader of the item, and the store's own hold is gone."""
    wt, sent, tm = _thread()
    _open(wt)
    _route(wt, audio_in=["w0", "w1"])
    path = _audio_file(tmp_path, "c.raw", 4)

    wt._process_stream_input(ClientAppend("r", file_paths={"audio": [path]}))

    assert _signals(sent) == [("w0", "audio_in", [[4]], {}), ("w1", "audio_in", [[4]], {})]
    (_entity, msg), _ = [(e, m) for e, m in sent if m.message_type == WorkerMessageType.INPUT_SIGNALS]
    (edge,) = msg.body.inputs
    assert (edge.next_node, edge.is_streaming, msg.body.partition_name) == ("ear", True, "Ear")
    assert msg.body.request_info is None
    (info,) = edge.tensor_info
    assert tm.refs[info.uuid] == 2 and info.uuid in tm.registered and info.uuid not in tm.persisted
    assert _bodies(sent, ConductorMessageType.NEW_REQUEST)[0].input_open
    (chunk,) = wt.model.chunk_calls
    assert torch.equal(chunk.tensors["audio_inputs"][0], torch.arange(4, dtype=torch.float32))
    assert chunk.request_kwargs["voice"] == "a" and chunk.model_kwargs == {}
    assert chunk.text is None and chunk.streams == {"audio_inputs": [None]}
    assert not (tmp_path / "c.raw").exists()  # the thread owns an append's files


def test_each_streams_end_goes_to_its_ranks_with_its_own_count(tmp_path):
    """Behind its items on each rank's FIFO; the consumer's buffer closes once
    it has that many. Ending one stream leaves the other open."""
    wt, sent, _tm = _thread(_TwoStreams())
    _open(wt)
    _route(wt, audio_in=["w0", "w1"], video_in=["w0"])
    audio = [_audio_file(tmp_path, f"a{i}.raw", 2) for i in range(2)]

    wt._process_stream_input(ClientAppend(
        "r", file_paths={"audio": [*audio, _audio_file(tmp_path, "v.raw", 3)]},
        file_streams={"audio": ["audio_in", "audio_in", "video_in"]}, ends=frozenset({"video_in"}),
    ))
    wt._process_stream_input(ClientAppend("r", ends=frozenset({"audio_in", "video_in"})))

    assert _signals(sent) == [
        ("w0", "audio_in", [[2], [2]], {}), ("w1", "audio_in", [[2], [2]], {}),
        ("w0", "video_in", [[3]], {}),
        ("w0", None, [], {"video_in": 1}),                 # the append's own items go first
        ("w0", None, [], {"audio_in": 2}), ("w1", None, [], {"audio_in": 2}),   # video's end only once
    ]


def test_input_before_the_route_is_held_then_sent_in_order(tmp_path):
    """The opening's chunk, the appends and an end that come before the
    conductor's route go out once it arrives, in the order they came."""
    wt, sent, tm = _thread()
    _open(wt, files={"audio": [_audio_file(tmp_path, "open.raw", 8)]})
    for i, samples in enumerate((2, 3)):
        wt._process_stream_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, f"{i}.raw", samples)]}))
    wt._process_stream_input(ClientAppend("r", ends=END))
    assert _signals(sent) == []

    _route(wt)

    assert _signals(sent) == [
        ("w0", "audio_in", [[8]], {}), ("w0", "audio_in", [[2]], {}), ("w0", "audio_in", [[3]], {}),
        ("w0", None, [], {"audio_in": 3}),
    ]
    assert set(tm.refs.values()) == {1}  # one reader each, the hold gone


def test_a_stream_whose_partition_ended_answers_later_appends_as_late(tmp_path):
    """The conductor's route names no rank for it any more."""
    wt, sent, tm = _thread()
    _open(wt)
    _route(wt)
    _route(wt, audio_in=[])
    late = []

    wt._process_stream_input(ClientAppend(
        "r", file_paths={"audio": [_audio_file(tmp_path, "c.raw", 4)]}, on_late=late.append,
    ))

    assert late == ["the call's 'audio_in' input has ended"]
    assert _signals(sent) == [] and tm.refs == {}


def test_a_call_that_drains_before_its_route_sends_nothing(tmp_path):
    """Its consumers are being torn down; REMOVE frees what it held."""
    wt, sent, tm = _thread()
    _open(wt, files={"audio": [_audio_file(tmp_path, "open.raw", 8)]})
    wt._begin_drain("r")

    _route(wt)

    assert _signals(sent) == [] and list(tm.refs.values()) == [1]
    wt._hard_cleanup("r")
    assert "r" not in wt._client_calls


def test_a_route_that_fails_fails_its_call_alone(tmp_path):
    """Its client is told; the rest of the thread's messages still apply."""
    wt, sent, tm = _thread()
    _open(wt, files={"audio": [_audio_file(tmp_path, "open.raw", 8)]})
    wt._process_input(PreprocessInput(
        request_id="q", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))

    def refuse(uuids, n):
        raise RuntimeError("the store refused")

    tm.increment_ref_batch_uniform = refuse
    batch = [_route_msg("r"), WorkerMessage(WorkerMessageType.REMOVE_REQUEST, RemoveRequest("q"))]
    wt.communicator.get_all_new_messages = lambda: batch
    wt._process_messages()

    assert "q" not in wt._client_calls  # the REMOVE behind it applied
    error = wt.out_queue.get_nowait()
    assert (error.request_id, error.modality, error.metadata["status"]) == ("r", "error", 500)
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.ABORT_REQUEST)] == ["r"]


def test_an_append_s_text_settings_and_target_streams_reach_the_hook(tmp_path):
    wt, sent, _tm = _thread()
    _open(wt)
    paths = [_audio_file(tmp_path, "a.raw", 2), _audio_file(tmp_path, "b.raw", 3)]

    wt._process_stream_input(ClientAppend(
        "r", file_paths={"audio": paths}, text="louder", model_kwargs={"gain": 2},
        file_streams={"audio": ["audio_in", None]},
    ))

    (chunk,) = wt.model.chunk_calls
    assert chunk.text == "louder" and chunk.model_kwargs == {"gain": 2}
    assert chunk.request_kwargs == {"voice": "a"}  # the opening's, beside the append's own
    assert chunk.streams == {"audio_inputs": ["audio_in", None]}
    assert [t.numel() for t in chunk.tensors["audio_inputs"]] == [2, 3]
    assert _signals(sent) == []  # held: no route yet
    _route(wt)
    assert _signals(sent) == [("w0", "audio_in", [[2], [3]], {})]


class _Transcribing(_ListeningModel):
    """Takes text appends too: each becomes one item of its byte values."""

    def process_stream_chunk(self, chunk):
        items = super().process_stream_chunk(chunk)
        if chunk.text:
            items["audio_in"] = [*items["audio_in"], torch.tensor(list(chunk.text.encode()), dtype=torch.float32)]
        return items


def test_a_text_only_append_reaches_a_model_that_takes_text():
    wt, sent, _tm = _thread(_Transcribing())
    _open(wt)

    wt._process_stream_input(ClientAppend("r", text="hi"))

    (chunk,) = wt.model.chunk_calls
    assert chunk.text == "hi" and chunk.tensors == {} and chunk.streams == {}
    _route(wt)
    assert _signals(sent) == [("w0", "audio_in", [[2]], {})]


def test_a_text_only_append_to_a_model_that_takes_none_is_refused_with_a_400():
    """The hook returns nothing for the text: dropped silently, the client
    would never learn its words went nowhere."""
    wt, sent, _tm = _thread()
    _open(wt)

    wt._process_waiting_input(ClientAppend("r", text="hello?"))

    error = wt.out_queue.get_nowait()
    assert error.modality == "error" and error.metadata["status"] == 400
    assert b"takes no text" in error.data
    _route(wt)
    assert _signals(sent) == []
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.ABORT_REQUEST)] == ["r"]


def test_prompt_tensors_named_after_a_client_stream_are_its_first_chunk(tmp_path):
    """A one-shot is a stream of one chunk, ended at once: its opening tells
    the conductor its input is closed, so it may wait for a slot."""
    wt, sent, _tm = _thread()
    _open(wt, files={"audio": [_audio_file(tmp_path, "clip.raw", 8)]}, ends=END)

    kinds = [msg.message_type for _entity, msg in sent]
    assert kinds == [ConductorMessageType.NEW_REQUEST]
    (new,) = _bodies(sent, ConductorMessageType.NEW_REQUEST)
    assert "audio_in" not in new.initial_signals and "text_inputs" in new.initial_signals
    assert not new.input_open
    _route(wt)
    assert _signals(sent) == [("w0", "audio_in", [[8]], {}), ("w0", None, [], {"audio_in": 1})]


def test_an_append_for_a_finished_request_is_dropped(tmp_path):
    wt, sent, _tm = _thread()
    path = _audio_file(tmp_path, "late.raw", 4)

    wt._process_stream_input(ClientAppend("gone", file_paths={"audio": [path]}, ends=END))

    assert sent == [] and not (tmp_path / "late.raw").exists()


def test_a_bad_append_fails_and_aborts_the_request(tmp_path):
    class _Confused(_ListeningModel):
        def process_stream_chunk(self, chunk):
            return {"not_a_stream": chunk.tensors["audio_inputs"]}

    wt, sent, _tm = _thread(_Confused())
    _open(wt)
    wt.in_queue.put(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "x.raw", 4)]}))
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    try:
        error = wt.out_queue.get(timeout=10)
    finally:
        wt.stop_event.set()
        runner.join(timeout=10)

    assert error.modality == "error" and b"internal error" in error.data  # #646: kept from the client
    assert error.metadata["status"] == 500  # the model's bug, not the client's
    (abort,) = _bodies(sent, ConductorMessageType.ABORT_REQUEST)
    assert abort.request_id == "r"
    # the drain began: with no read in flight it acknowledged at once
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.READS_DONE)] == ["r"]
    assert _signals(sent) == []


def test_queued_appends_keep_their_order_behind_the_request(tmp_path):
    wt, sent, _tm = _thread(routed=("r",))
    wt.in_queue.put(PreprocessInput(
        request_id="r", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    for i, samples in enumerate((2, 3, 4)):
        wt.in_queue.put(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, f"{i}.raw", samples)]}))
    wt.in_queue.put(ClientAppend("r", ends=END))
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    deadline = time.time() + 10
    while len(_signals(sent)) < 4 and time.time() < deadline:
        time.sleep(0.01)
    wt.stop_event.set()
    runner.join(timeout=10)

    assert sent[0][1].message_type == ConductorMessageType.NEW_REQUEST
    assert _signals(sent) == [
        ("w0", "audio_in", [[2]], {}), ("w0", "audio_in", [[3]], {}), ("w0", "audio_in", [[4]], {}),
        ("w0", None, [], {"audio_in": 3}),
    ]


class _CappedModel(_ListeningModel):
    """``audio_in`` carries at most 5 samples over a call's life."""

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear"], connections=[
            Connection(from_partition=CLIENT, to_partition="Ear", edge_name="audio_in",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
                       limits=ClientStreamLimits(max_items=5, item_size=lambda t: t.numel(), unit="samples")),
        ])


def test_a_call_past_its_streams_max_items_ends_with_a_400_naming_the_limit(tmp_path):
    wt, sent, _tm = _thread(_CappedModel())
    _open(wt)
    _route(wt)
    wt._process_waiting_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "a.raw", 3)]}))
    wt._process_waiting_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "b.raw", 3)]}))

    error = wt.out_queue.get_nowait()
    assert error.modality == "error" and error.metadata["status"] == 400
    assert b"passed 5 samples" in error.data and b"client_streams.audio_in.max_items" in error.data
    # the first append went on; the one past the limit was never sent
    assert _signals(sent) == [("w0", "audio_in", [[3]], {})]
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.ABORT_REQUEST)] == ["r"]
    assert list(tmp_path.iterdir()) == []


def test_the_opening_s_first_chunk_counts_against_max_items(tmp_path):
    wt, sent, _tm = _thread(_CappedModel())
    _open(wt, files={"audio": [_audio_file(tmp_path, "clip.raw", 4)]})
    assert wt._client_calls["r"].charged == {"audio_in": 4}

    wt._process_waiting_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "c.raw", 2)]}))

    assert wt.out_queue.get_nowait().metadata["status"] == 400
    wt._hard_cleanup("r")
    assert "r" not in wt._client_calls  # dropped with the request's other state


def test_an_opening_past_max_items_is_refused():
    wt, sent, _tm = _thread(_CappedModel())
    wt._process_waiting_input(PreprocessInput(
        request_id="r", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    assert _bodies(sent, ConductorMessageType.NEW_REQUEST)  # 0 samples: fine

    class _LongOpening(_CappedModel):
        def process_prompt(self, *a, **k):
            return {"text_inputs": [torch.tensor([1])], "audio_in": [torch.zeros(6)]}

    wt, sent, _tm = _thread(_LongOpening())
    wt._process_waiting_input(PreprocessInput(
        request_id="r", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    error = wt.out_queue.get_nowait()
    assert error.metadata["status"] == 400 and b"max_items" in error.data
    assert _bodies(sent, ConductorMessageType.NEW_REQUEST) == []


def test_a_deployment_lowers_a_streams_max_items(tmp_path):
    """The serving config's ``client_streams`` block may lower the declared
    bound; what the model counts per item stays the model's."""
    wt, sent, _tm = _thread(_CappedModel(), model_config={"client_streams": {"audio_in": {"max_items": 2}}})
    _open(wt)
    wt._process_waiting_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "a.raw", 3)]}))

    assert b"passed 2 samples" in wt.out_queue.get_nowait().data


class _TwoStreams(_ListeningModel):
    """``audio_in`` into Ear and ``video_in`` into Eye; a file that names no
    stream goes to video_in."""

    def get_partition_topology(self):
        return PartitionTopology(partitions=["Ear", "Eye"], connections=[
            Connection(from_partition=CLIENT, to_partition=partition, edge_name=name,
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
                       limits=ClientStreamLimits(max_items=10))
            for name, partition in (("audio_in", "Ear"), ("video_in", "Eye"))
        ])

    def get_graph_walk_graphs(self):
        return {walk: GraphNode(name=node, input_names=[name], consumes_stream=True, outputs=[])
                for walk, node, name in (("listen", "ear", "audio_in"), ("look", "eye", "video_in"))}

    def get_partitions(self):
        return [PartitionDefinition(name="Ear", graph_walks={"listen"}, initial_walk="listen"),
                PartitionDefinition(name="Eye", graph_walks={"look"}, initial_walk="look")]

    def process_stream_chunk(self, chunk):
        self.chunk_calls.append(chunk)
        items = {"audio_in": [], "video_in": []}
        for tensor, name in zip(chunk.tensors.get("audio_inputs", []), chunk.streams.get("audio_inputs", []),
                                strict=True):
            items[name or "video_in"].append(tensor)
        return items


def test_items_the_model_routes_to_an_ended_stream_are_late_and_neither_charged_nor_sent(tmp_path):
    """The API server can't tell where a file that names no stream goes; the
    preprocess thread can. Such items are dropped and the append is answered
    as late, while what it carried for an open stream goes on."""
    wt, sent, _tm = _thread(_TwoStreams())
    _open(wt)
    _route(wt, audio_in=["w0"], video_in=["w0"])
    late = []
    # an append's own items go ahead of the end it carries
    wt._process_stream_input(ClientAppend(
        "r", file_paths={"audio": [_audio_file(tmp_path, "v.raw", 2)]}, ends=frozenset({"video_in"}),
        on_late=late.append,
    ))
    assert late == []

    wt._process_stream_input(ClientAppend(
        "r", file_paths={"audio": [_audio_file(tmp_path, "a.raw", 3), _audio_file(tmp_path, "b.raw", 4)]},
        file_streams={"audio": ["audio_in", None]}, on_late=late.append,
    ))

    assert late == ["the call's 'video_in' input has ended"]
    assert _signals(sent) == [
        ("w0", "video_in", [[2]], {}), ("w0", None, [], {"video_in": 1}), ("w0", "audio_in", [[3]], {}),
    ]
    assert wt._client_calls["r"].charged == {"video_in": 1, "audio_in": 1}
    assert list(tmp_path.iterdir()) == []


def test_ends_that_reach_one_rank_together_each_name_their_partition():
    """Two streams into two partitions, ended at once, with one rank
    consuming both: one end per partition, so neither names the other's."""
    wt, sent, _tm = _thread(_TwoStreams())
    _open(wt, ends=frozenset({"audio_in", "video_in"}))
    _route(wt, audio_in=["w0"], video_in=["w0"])

    ends = sorted((b.partition_name, b.producer_done_items) for b in _bodies(sent, WorkerMessageType.INPUT_SIGNALS))
    assert ends == [("Ear", {"audio_in": 0}), ("Eye", {"video_in": 0})]


class _RecordingPreprocess:
    def __init__(self):
        self.items = []
        self.aborted = []

    def new_request(self, item):
        self.items.append(item)

    def stream_input(self, item):
        self.items.append(item)

    def abort_request(self, key):
        self.aborted.append(key)

    forward_abort = abort_request

    def has_pending_tensors(self, key):
        return True  # a completed call's last chunks are still on their way


def _server(takes_stream_input=True, timeout_seconds=600.0):
    return make_test_api_server(
        timeout_seconds=timeout_seconds,
        takes_stream_input=takes_stream_input,
        client_streams=frozenset({"audio_in"} if takes_stream_input else ()),
        max_call_input_bytes=1 << 30,
        preprocess_worker=_RecordingPreprocess(),
    )


def _submit(server, request_id="r", **kwargs):
    """Submit under ``request_id``; returns the submission's key."""
    server.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id=request_id, **kwargs,
    )
    return server.submission_key(request_id)


def test_a_one_shot_request_ends_its_input_at_once(tmp_path):
    server = _server()
    key = _submit(server)

    (opening,) = server.preprocess_worker.items
    assert isinstance(opening, PreprocessInput) and opening.request_id == key
    assert opening.ends == END
    assert server.stream_input(key, {"audio": ["a.wav"]}) is False
    assert len(server.preprocess_worker.items) == 1


def test_an_open_request_takes_appends_until_it_ends(tmp_path):
    server = _server()
    key = _submit(server, end_input=False)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert server.stream_input(key, {"audio": [str(wav)]}) is True
    assert server.stream_input(key, end_input=True) is True

    assert server.preprocess_worker.items[1:] == [
        ClientAppend(key, file_paths={"audio": [str(wav)]}),
        ClientAppend(key, ends=END),
    ]
    assert server.stream_input(key, {"audio": ["b.wav"]}) is False
    assert server.stream_input("unknown#9", {"audio": ["b.wav"]}) is False
    assert len(server.preprocess_worker.items) == 3


def test_a_finished_calls_appends_never_reach_a_newer_call_under_its_id(tmp_path):
    """Once a call has finished its id is free, and a new call may take it
    while the old call's socket still holds it; that socket's appends name the
    old call's key, so they are refused rather than fed to the new call."""
    server = _server()
    old = _submit(server, "call", end_input=False)
    server.pending_requests[old].event.set()            # the model ended the call
    new = _submit(server, "call", end_input=False)       # a new call takes the free id
    assert new != old
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    assert server.stream_input(old, {"audio": [str(wav)]}) is False
    assert server.stream_input(new, {"audio": [str(wav)]}) is True

    appends = [i for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]
    assert appends == [ClientAppend(new, file_paths={"audio": [str(wav)]})]
    assert server.pending_requests[old].input_bytes == 0


def test_a_model_without_client_streams_ignores_end_input():
    server = _server(takes_stream_input=False)
    key = _submit(server, end_input=False)

    (only,) = server.preprocess_worker.items
    assert isinstance(only, PreprocessInput)
    assert server.pending_requests[key].open_streams == frozenset()


def _pending(server, key="r#1"):
    """An open call under ``key``, as submit_request leaves it."""
    server._live["r"] = key
    server._client_ids[key] = "r"
    server.pending_requests[key] = PendingRequest(
        streaming=True, input_modalities=["audio"], output_modalities=["text"],
        profile=RequestProfile(rid="r", timing=RequestTiming(recv_time=0.0)), open_streams=END,
    )
    return key


def test_input_activity_holds_off_the_request_timeout(tmp_path):
    server = _server(timeout_seconds=0.3)
    key = _pending(server)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    async def consume():
        async for _batch in server.iter_result_chunk_batches("r"):
            pass

    async def keep_talking():
        for _ in range(12):  # 0.6 s of input, twice the timeout
            server.stream_input(key, {"audio": [str(wav)]})
            await asyncio.sleep(0.05)
        server.pending_requests[key].event.set()

    async def main():
        await asyncio.gather(consume(), keep_talking())

    asyncio.run(main())  # finishes without "Request timed out"


def test_a_silent_open_request_still_times_out():
    server = _server(timeout_seconds=0.2)
    _pending(server)
    server.abort_submission = lambda key, outcome="abort": None

    async def consume():
        async for _batch in server.iter_result_chunk_batches("r"):
            pass

    with pytest.raises(HTTPException):
        asyncio.run(consume())


class _StreamingFakeServer:
    takes_stream_input = True

    def __init__(self, upload_dir):
        self.upload_dir = upload_dir
        self.calls = []
        self.appended = []  # each append's text, file_streams and model_kwargs
        self.finish = asyncio.Event()

    def submit_request(self, **kwargs):
        self.calls.append(("submit", kwargs["request_id"], kwargs["end_input"]))
        return kwargs["request_id"]

    async def wait_for_request_id(self, request_id):
        pass

    def submission_key(self, request_id):
        return request_id  # a key per submission matters to the real server only

    def taking_input(self, key):
        return not any(c[0] == "append" and c[1] == key and c[-1] is True for c in self.calls)

    def stream_input(self, key, file_paths=None, end_input=False, on_late=None, **append):
        names = sorted(p.rsplit("_", 1)[-1] for ps in (file_paths or {}).values() for p in ps)
        self.appended.append(append)
        self.calls.append(("append", key, names, end_input))
        return True

    def abort_submission(self, key):
        self.calls.append(("abort", key))

    async def iter_result_chunks(self, request_id):
        # ends once the call's input has ended, at its opening or by an append
        while not any(c[1] == request_id and c[-1] is True and c[0] in ("append", "submit") for c in self.calls):
            await asyncio.sleep(0.01)
        return
        yield


def _recv_json(ws):
    return json.loads(ws.receive_text())


def test_generate_ws_opens_appends_and_ends_in_order(monkeypatch, tmp_path):
    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    wav = {"name": "chunk.wav", "data": "UklGRg=="}
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        ws.send_text(json.dumps({"request_id": "call", "append": [wav]}))
        ws.send_text(json.dumps({"request_id": "call", "append": [wav], "end_input": True}))
        assert _recv_json(ws) == {"request_id": "call", "finish": True}
        ws.send_text(json.dumps({"request_id": "call", "append": [wav]}))
        late = _recv_json(ws)

    assert fake.calls == [
        ("submit", "call", False),
        ("append", "call", ["chunk.wav"], False),
        ("append", "call", ["chunk.wav"], True),
    ]
    assert late["request_id"] == "call" and "not taking input" in late["error"]
    assert late["append"] is True  # a reply to the append, not the call's end


def test_generate_ws_appends_only_to_requests_this_socket_opened(monkeypatch, tmp_path):
    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "someone-else", "append": [
            {"name": "chunk.wav", "data": "UklGRg=="},
        ]}))
        reply = _recv_json(ws)

    assert "not taking input on this socket" in reply["error"] and reply["append"] is True
    assert fake.calls == []
    assert list(tmp_path.iterdir()) == []  # the rejected append left no file behind


class _HangingServer(_StreamingFakeServer):
    def __init__(self, upload_dir):
        super().__init__(upload_dir)
        self.aborted = []

    def abort_submission(self, key):
        self.aborted.append(key)

    async def iter_result_chunks(self, request_id):
        await asyncio.Event().wait()
        yield


def test_closing_the_socket_aborts_the_calls_it_opened(monkeypatch, tmp_path):
    fake = _HangingServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        ws.send_text(json.dumps({"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]}))
        deadline = time.time() + 5
        while len(fake.calls) < 2 and time.time() < deadline:
            time.sleep(0.01)
    deadline = time.time() + 5
    while "call" not in fake.aborted and time.time() < deadline:
        time.sleep(0.01)
    assert "call" in fake.aborted


def test_generate_ws_rejects_bad_ids_in_band_and_stays_up(monkeypatch, tmp_path):
    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": ["oops"], "append": []}))
        assert "must be a string" in _recv_json(ws)["error"]
        ws.send_text(json.dumps({"output_modalities": "text,audio"}))      # a live call with no id
        assert "needs a request_id" in _recv_json(ws)["error"]
        ws.send_text(json.dumps({"request_id": "ok", "output_modalities": "text", "end_input": True,
                                 "text": "hi"}))
        assert _recv_json(ws) == {"request_id": "ok", "finish": True}


def test_an_append_after_end_input_is_answered_and_the_call_goes_on(monkeypatch, tmp_path):
    fake = _HangingServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "b", "output_modalities": "text,audio"}))
        ws.send_text(json.dumps({"request_id": "b", "end_input": True}))
        ws.send_text(json.dumps({"request_id": "b", "append": [{"name": "c.wav", "data": "UklGRg=="}]}))
        assert "not taking input" in _recv_json(ws)["error"]
        assert fake.aborted == []


def _real_ws_server(monkeypatch, tmp_path, **kwargs):
    server = _server(**kwargs)
    server.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", server)
    return server


@pytest.mark.parametrize("why", ["unknown file type", "over the input budget"])
def test_a_rejected_append_ends_the_call_with_its_error_and_no_finish(monkeypatch, tmp_path, why):
    server = _real_ws_server(monkeypatch, tmp_path)
    if "budget" in why:
        server.max_call_input_bytes = 1
    bad = {"name": "x.unknown", "data": "AA=="} if "unknown" in why else {"name": "c.wav", "data": "UklGRg=="}
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        ws.send_text(json.dumps({"request_id": "call", "append": [bad]}))
        error = _recv_json(ws)
        # the next frame answers this probe: no finish came between
        ws.send_text(json.dumps({"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]}))
        probe = _recv_json(ws)

    assert error["request_id"] == "call" and "append rejected" in error["error"]
    assert error["status"] == 400
    assert "not taking input" in probe["error"]
    (key,) = [i.request_id for i in server.preprocess_worker.items if isinstance(i, PreprocessInput)]
    assert server.preprocess_worker.aborted[0] == key


def test_an_append_after_the_model_ended_the_call_is_late_not_rejected(monkeypatch, tmp_path):
    """Between the conductor's request_complete and the delivery of the
    call's last chunks, a bad append must not fail a reply that is done."""
    server = _real_ws_server(monkeypatch, tmp_path)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        assert _wait(lambda: "call" in server._live)
        key = server._live["call"]
        messages = [APIServerMessage("request_complete", RequestComplete(
            request_id=key, output_chunk_counts={}, conductor_ingest_time=0.0, conductor_finish_time=0.0,
        ))]

        def one_pass():
            if not messages:
                server.running = False
            return [messages.pop()] if messages else []

        server.communicator = SimpleNamespace(get_all_new_messages=one_pass)
        server.conductor_proc = None
        server._liveness_interval_s = 0.5
        server.running = True
        server._process_messages()

        ws.send_text(json.dumps({"request_id": "call", "append": [{"name": "x.unknown", "data": "AA=="}]}))
        late = _recv_json(ws)
        aborted = list(server.preprocess_worker.aborted)
        error = server.pending_requests[key].error

    assert "not taking input" in late["error"] and "status" not in late
    assert aborted == [] and error is None


def test_a_call_cannot_send_past_its_input_budget(tmp_path):
    server = _server()
    server.max_call_input_bytes = 10
    small, big = tmp_path / "s.wav", tmp_path / "b.wav"
    small.write_bytes(b"x" * 4)
    big.write_bytes(b"x" * 11)
    key = _submit(server, end_input=False, file_paths={"audio": [str(small)]})  # 4 of 10
    server.stream_input(key, {"audio": [str(small)]})                         # 8 of 10
    with pytest.raises(ValueError, match="max_call_input_mb"):
        server.stream_input(key, {"audio": [str(small)]})                     # 12 of 10
    assert server.stream_input("other#9", {"audio": [str(small)]}) is False
    with pytest.raises(ValueError, match="max_call_input_mb"):
        _submit(server, "big", end_input=False, file_paths={"audio": [str(big)]})
    assert "big" not in server._live


def test_an_append_naming_a_stream_the_model_lacks_is_refused(tmp_path):
    server = _server()
    key = _submit(server, end_input=False)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="no client stream named 'video_in'"):
        server.stream_input(key, {"audio": [str(wav)]}, file_streams={"audio": ["video_in"]})
    assert server.stream_input(key, {"audio": [str(wav)]}, file_streams={"audio": ["audio_in"]}) is True

    appended = server.preprocess_worker.items[-1]
    assert appended == ClientAppend(key, file_paths={"audio": [str(wav)]},
                                    file_streams={"audio": ["audio_in"]})


def test_an_opening_file_naming_a_stream_the_model_lacks_is_refused_with_a_400():
    """Checked as an append's file is; process_prompt routes the opening's
    files, so a stream the model has is only checked."""
    server = _server()
    with pytest.raises(HTTPException) as refused:
        _submit(server, end_input=False, file_streams={"audio": [None, "video_in"]})
    assert refused.value.status_code == 400 and "no client stream named 'video_in'" in refused.value.detail
    assert "r" not in server._live and server.preprocess_worker.items == []

    _submit(server, end_input=False, file_streams={"audio": ["audio_in"]})
    assert isinstance(server.preprocess_worker.items[0], PreprocessInput)


def test_an_append_s_text_counts_against_the_input_budget():
    server = _server()
    server.max_call_input_bytes = 8
    key = _submit(server, end_input=False)
    assert server.stream_input(key, text="four") is True                 # 4 of 8
    with pytest.raises(ValueError, match="max_call_input_mb"):
        server.stream_input(key, text="fives")                           # 9 of 8
    assert server.preprocess_worker.items[-1] == ClientAppend(key, text="four")


def _two_stream_server():
    server = _server()
    server.client_streams = frozenset({"audio_in", "video_in"})
    return server


def test_end_input_can_end_some_streams_and_leave_the_rest_open(tmp_path):
    server = _two_stream_server()
    key = _submit(server, end_input=False)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    assert server.stream_input(key, end_input=["video_in"]) is True
    assert server.pending_requests[key].open_streams == {"audio_in"} and server.taking_input(key)
    # a stream already ended takes nothing more: late, not an error
    assert server.stream_input(key, {"audio": [str(wav)]}, file_streams={"audio": ["video_in"]}) is False
    assert server.stream_input(key, end_input=["video_in"]) is False
    with pytest.raises(ValueError, match="no client stream named 'nope'"):
        server.stream_input(key, end_input=["nope"])
    assert server.stream_input(key, {"audio": [str(wav)]}, file_streams={"audio": ["audio_in"]}) is True
    assert server.stream_input(key, end_input=True) is True          # the rest
    assert not server.taking_input(key)

    appends = [i for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]
    assert [i.ends for i in appends] == [{"video_in"}, frozenset(), {"audio_in"}]


def test_an_opening_can_end_some_of_its_streams_at_once():
    server = _two_stream_server()
    key = _submit(server, end_input=["video_in"])

    (opening,) = server.preprocess_worker.items
    assert isinstance(opening, PreprocessInput) and opening.ends == {"video_in"}
    assert server.pending_requests[key].open_streams == {"audio_in"}
    with pytest.raises(ValueError, match="no client stream named 'nope'"):
        _submit(server, "other", end_input=["nope"])


def test_generate_ws_ends_the_streams_end_input_lists(monkeypatch, tmp_path):
    server = _real_ws_server(monkeypatch, tmp_path)
    server.client_streams = frozenset({"audio_in", "video_in"})
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "end_input": ["video_in"]},                     # an append: no input
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}], "end_input": ["audio_in"]},
        {"request_id": "call", "end_input": ["audio_in"]},                     # late now
    ])
    appends = lambda: [i for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]  # noqa: E731
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    assert [(bool(i.file_paths), i.ends) for i in appends()] == [
        (False, {"video_in"}), (True, {"audio_in"}),
    ]
    (late,) = ws.sent
    assert late["append"] is True and "not taking input" in late["error"]


def test_generate_ws_refuses_ending_a_stream_the_model_lacks(monkeypatch, tmp_path):
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "end_input": ["video_in"]},
        {"request_id": "bad", "text": "x", "end_input": [7]},
    ])
    asyncio.run(_serve(ws, until=lambda: len(ws.sent) >= 2))

    by_id = {reply["request_id"]: reply for reply in ws.sent}
    assert by_id["call"]["status"] == 400 and "no client stream named 'video_in'" in by_id["call"]["error"]
    assert "end_input must be" in by_id["bad"]["error"]


def test_generate_ws_takes_the_append_record_and_its_shorthand_alike(monkeypatch, tmp_path):
    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    wav = {"name": "chunk.wav", "data": "UklGRg=="}
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": [wav]},                                 # the shorthand
        {"request_id": "call", "append": {"files": [wav]}},                      # the same, as a record
        {"request_id": "call", "append": {"text": "hello"}},                     # text only
        {"request_id": "call", "append": {"files": [{**wav, "stream": "audio_in"}],
                                          "model_kwargs": {"gain": 2}, "text": None}},
        {"request_id": "call", "append": {"model_kwargs": '{"gain": 3}', "text": "x"}, "end_input": True},
    ])
    # until the call finishes: the socket's close aborts a call still running
    asyncio.run(_serve(ws, until=lambda: any(reply.get("finish") for reply in ws.sent)))

    assert ws.sent == [{"request_id": "call", "finish": True}]
    assert [c[2:] for c in fake.calls[1:]] == [
        (["chunk.wav"], False), (["chunk.wav"], False), ([], False), (["chunk.wav"], False), ([], True),
    ]
    assert fake.appended == [
        {"text": None, "file_streams": None, "model_kwargs": None},
        {"text": None, "file_streams": None, "model_kwargs": None},
        {"text": "hello", "file_streams": None, "model_kwargs": None},
        {"text": None, "file_streams": {"audio": ["audio_in"]}, "model_kwargs": {"gain": 2}},
        {"text": "x", "file_streams": None, "model_kwargs": {"gain": 3}},
    ]


@pytest.mark.parametrize("append", [
    {"request_id": "call", "append": {"files": [{"name": "c.wav", "data": "UklGRg==", "stream": 3}]}},
    {"request_id": "call", "append": {"text": 5}},
    {"request_id": "call", "append": {"volume": 2}},
    {"request_id": "call", "append": "c.wav"},
    {"request_id": "call", "append": {"model_kwargs": [1], "text": "x"}},
    {"request_id": "call", "append": {}},
    {"request_id": "call", "append": {"text": "x"}, "text": "y"},
    {"request_id": "call", "append": {"text": "x"}, "model_kwargs": {"gain": 2}},
], ids=["stream-type", "text-type", "unknown-field", "not-a-record", "kwargs-type", "empty",
        "text-beside", "kwargs-beside"])
def test_a_malformed_append_record_ends_the_call_with_a_400(monkeypatch, tmp_path, append):
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([{"request_id": "call", "output_modalities": "text"}, append])
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    (error,) = ws.sent
    assert error["status"] == 400 and "append rejected" in error["error"]
    assert not any(isinstance(i, ClientAppend) for i in server.preprocess_worker.items)
    assert list(tmp_path.iterdir()) == []  # a file decoded before the refusal is gone


def test_generate_ws_refuses_an_append_to_a_stream_the_model_lacks(monkeypatch, tmp_path):
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": {"files": [{"name": "c.wav", "data": "UklGRg==", "stream": "nope"}]}},
    ])
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    (error,) = ws.sent
    assert error["status"] == 400 and "no client stream named 'nope'" in error["error"]


def test_generate_ws_refuses_an_opening_file_naming_a_stream_the_model_lacks(monkeypatch, tmp_path):
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text",
         "files": [{"name": "c.wav", "data": "UklGRg==", "stream": "nope"}]},
    ])
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    (error,) = ws.sent
    assert error["status"] == 400 and "no client stream named 'nope'" in error["error"]
    assert server.preprocess_worker.items == []


def test_an_append_the_model_routes_to_an_ended_stream_is_answered_as_late(monkeypatch, tmp_path):
    """The preprocess thread answers through the append's ``on_late``, from
    its own thread; the call goes on."""

    class _RoutedToAnEndedStream(_StreamingFakeServer):
        def stream_input(self, key, file_paths=None, end_input=False, on_late=None, **append):
            threading.Thread(target=on_late, args=("the call's 'video_in' input has ended",)).start()
            return super().stream_input(key, file_paths, end_input, **append)

    fake = _RoutedToAnEndedStream(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]},
    ])
    calls_at_the_reply = []

    def answered():
        if ws.sent and not calls_at_the_reply:
            calls_at_the_reply.append(list(fake.calls))
        return ws.sent

    asyncio.run(_serve(ws, until=answered))

    (late,) = ws.sent
    assert late == {"request_id": "call", "append": True, "error": "the call's 'video_in' input has ended"}
    # the late reply leaves the call running; only the socket's close, after it, aborts the call
    assert not any(c[0] == "abort" for c in calls_at_the_reply[0])
    assert [c for c in fake.calls if c[0] == "abort"] == [("abort", "call")] and fake.calls[-1] == ("abort", "call")


def test_what_an_append_record_carries_never_opens_a_request(monkeypatch, tmp_path):
    """Routing reads the top level only: text, files and settings inside a
    record feed the call, whatever else the message holds."""
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "output_modalities": "text",
         "append": {"text": "more", "files": [{"name": "c.wav", "data": "UklGRg=="}],
                    "model_kwargs": {"gain": 2}}},
    ])
    appends = lambda: [i for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]  # noqa: E731
    asyncio.run(_serve(ws, until=appends))

    assert ws.sent == []
    assert [type(i).__name__ for i in server.preprocess_worker.items] == ["PreprocessInput", "ClientAppend"]
    (append,) = appends()
    assert append.text == "more" and append.model_kwargs == {"gain": 2} and append.file_paths["audio"]


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


def test_a_second_socket_cannot_take_over_a_live_call(monkeypatch, tmp_path):
    """Two sockets open a live call under one request_id: the second is
    refused with a 409, and the first socket's appends reach its own call."""
    server = _server()
    server.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", server)
    client = TestClient(entrypoint.app)
    with client.websocket_connect("/generate/ws") as a, client.websocket_connect("/generate/ws") as b:
        a.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        assert _wait(lambda: "call" in server._live)
        first = server._live["call"]
        b.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        refused = _recv_json(b)
        a.send_text(json.dumps({"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]}))
        assert _wait(lambda: any(isinstance(i, ClientAppend) for i in server.preprocess_worker.items))

    assert refused["status"] == 409 and refused["request_id"] == "call"
    opens = [i.request_id for i in server.preprocess_worker.items if isinstance(i, PreprocessInput)]
    appends = [i.request_id for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]
    assert opens == [first] and appends[0] == first


class _FailingOutputModel(_ListeningModel):
    def postprocess(self, output, modality, request_kwargs=None):
        raise RuntimeError("boom")


def test_an_output_failure_aborts_a_request_that_may_still_take_input():
    """The client's stream ends with the error, but a live call's engine
    request would never end on its own: the data worker aborts it."""
    from mstar.api_server.data_worker import InlineToken

    wt, sent, _tm = _thread(_FailingOutputModel())
    _open(wt)
    wt._emit_inline_token(InlineToken(request_id="r", value=5, loop_indices=None, modality="text"))

    assert wt.out_queue.get(timeout=5).modality == "error"
    (abort,) = _bodies(sent, ConductorMessageType.ABORT_REQUEST)
    assert abort.request_id == "r"
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.READS_DONE)] == ["r"]


def test_a_failed_append_does_not_release_a_read_still_in_flight(tmp_path):
    """The API server counts the failed append's error chunk as a delivered
    output, so it may report the call drained while its last read runs: the
    ACK still waits for that read."""
    class _Corrupt(_ListeningModel):
        def process_stream_chunk(self, chunk):
            raise ValueError("corrupt append")

    wt, sent, tm = _thread(_Corrupt())
    _open(wt)
    tm.has_inflight_reads = lambda request_id: True
    wt._process_waiting_input(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, "c.raw", 2)]}))
    wt._finish_reading("r", drained=True)
    assert _bodies(sent, ConductorMessageType.READS_DONE) == []

    tm.has_inflight_reads = lambda request_id: False
    wt._complete_drain_if_ready("r")
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.READS_DONE)] == ["r"]


def test_an_output_failing_after_its_request_was_removed_starts_no_teardown():
    """Its REMOVE_REQUEST came already: no abort goes out, and the id is not
    put back into the drain sets, which nothing would clear again."""
    wt, sent, _tm = _thread()
    _open(wt)
    wt._hard_cleanup("r")
    wt._fail_request("r", RuntimeError("segment gone"), "text output transfer")

    assert wt.out_queue.get_nowait().modality == "error"
    assert _bodies(sent, ConductorMessageType.ABORT_REQUEST) == []
    assert _bodies(sent, ConductorMessageType.READS_DONE) == []
    assert "r" not in wt._draining_rids and "r" not in wt._reads_done_sent


def test_a_rejected_call_completing_meanwhile_leaves_delivery_running():
    """reject_input ends the call in its stream, but its handler pops it only
    later; a completion landing meanwhile must not break the result thread
    (the data worker's per-request bookkeeping stays until the pop)."""
    import queue as _queue

    from mstar.api_server.data_worker import PreprocessWorker

    facade = PreprocessWorker.__new__(PreprocessWorker)  # the main-thread side, no thread
    for name in ("request_input_queue", "result_tensor_input_queue", "abort_request_queue",
                 "reads_done_queue", "output_queue", "profile_queue"):
        setattr(facade, name, _queue.Queue())
    facade._signal = lambda: None
    facade.per_request_reading_tensors, facade.received_chunk_counts = {}, {}
    server = _server()
    server.preprocess_worker = facade
    call = _submit(server, "call", end_input=False)
    other = _submit(server, "other")
    assert server.reject_input(call, "append rejected: bad file", 400)

    batches = [[APIServerMessage("request_complete", RequestComplete(
        request_id=rid, output_chunk_counts={}, conductor_ingest_time=0.0,
        conductor_finish_time=0.0,
    ))] for rid in (call, other)]

    server.communicator = SimpleNamespace(
        get_all_new_messages=lambda: batches.pop(0) if batches else [],
    )
    server.conductor_proc = None
    server._liveness_interval_s = 0.5
    server.running = True
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    _wait(lambda: other in server.recently_completed, timeout=2.0)
    server.running = False
    thread.join(timeout=2)

    assert other in server.recently_completed
    assert facade.abort_request_queue.get_nowait() == call  # the abort still went out


class _QueuedFramesSocket:
    """Hands back frames already received without suspending, as a read that
    delivered several frames at once does; then waits for the close."""

    def __init__(self, frames):
        self.frames = [json.dumps(f) for f in frames]
        self.sent: list[dict] = []
        self.closed = asyncio.Event()

    async def accept(self):
        pass

    async def receive(self):
        if self.frames:
            return {"type": "websocket.receive", "text": self.frames.pop(0)}
        await self.closed.wait()
        return {"type": "websocket.disconnect"}

    async def send_text(self, text):
        self.sent.append(json.loads(text))


async def _serve(ws, until, timeout=2.0):
    """Run /generate/ws on ``ws`` until ``until()`` or ``timeout``, then close it."""
    handler = asyncio.create_task(entrypoint.generate_ws(ws))
    deadline = time.time() + timeout
    while not until() and time.time() < deadline:
        await asyncio.sleep(0.01)
    ws.closed.set()
    await asyncio.wait_for(handler, 5)


def test_a_new_request_under_a_live_calls_id_is_refused_and_the_call_goes_on(monkeypatch, tmp_path):
    """Only an append (or a bare end_input) feeds a live call. A message that
    carries a request's fields under its id is a second request under an id
    in flight: refused with a 409, the call untouched."""
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text,audio"},
        {"request_id": "call", "output_modalities": "text,audio", "text": "hello again"},
        {"request_id": "call", "text": "a separate question", "end_input": True},
        {"request_id": "other", "end_input": True},
    ])
    asyncio.run(_serve(ws, until=lambda: len(ws.sent) >= 3))

    retried, one_shot, stray_end = sorted(ws.sent, key=lambda r: r["request_id"])
    assert retried["status"] == 409 and one_shot["status"] == 409
    assert "not taking input on this socket" in stray_end["error"]
    assert [i.request_id for i in server.preprocess_worker.items] == ["call#1"]
    assert server.preprocess_worker.aborted == ["call#1"]  # by the close, not before


def test_an_end_of_input_with_settings_or_null_fields_ends_the_call_s_input(monkeypatch, tmp_path):
    """An end of input carrying no text or files can't open a request, so it
    ends the live call's input whatever settings come with it; a null field
    is no field at all."""
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text,audio"},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}], "files": None},
        {"request_id": "call", "end_input": True, "model_kwargs": {}, "output_modalities": "text,audio",
         "text": None},
    ])
    appends = lambda: [i for i in server.preprocess_worker.items if isinstance(i, ClientAppend)]  # noqa: E731
    asyncio.run(_serve(ws, until=lambda: len(appends()) == 2))

    assert ws.sent == []
    assert [(bool(i.file_paths), i.ends) for i in appends()] == [(True, frozenset()), (False, END)]


def test_a_failed_request_cannot_take_over_the_next_calls_entry(monkeypatch, tmp_path):
    """A one-shot that fails and a live call under the same id, read before
    either task runs: each task owns only the call it was created for, so the
    live call streams, and closing the socket aborts it."""
    server = _real_ws_server(monkeypatch, tmp_path)
    ws = _QueuedFramesSocket([
        {"request_id": "A", "text": "x", "model_kwargs": "{not json", "end_input": True},
        {"request_id": "A", "output_modalities": "text"},
    ])

    def streaming():
        req = server.pending_requests.get("A#1")
        return bool(ws.sent) and req is not None and req.loop is not None

    asyncio.run(_serve(ws, until=streaming))
    (failed,) = ws.sent
    assert failed["request_id"] == "A" and "Expecting" in failed["error"]  # the JSON error
    assert server.preprocess_worker.aborted == ["A#1"]


def test_a_null_end_input_is_false(monkeypatch, tmp_path):
    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text", "end_input": None},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}], "end_input": None},
    ])
    asyncio.run(_serve(ws, until=lambda: len(fake.calls) >= 2))

    assert ws.sent == []
    assert fake.calls[:2] == [("submit", "call", False), ("append", "call", ["c.wav"], False)]


def test_a_bad_append_racing_the_end_of_the_call_is_answered_as_late(monkeypatch, tmp_path):
    """The call ends between the append's check and its refusal: the append
    was late, and the call (which reject_input leaves alone) isn't failed."""

    class _EndsMeanwhile(_StreamingFakeServer):
        def stream_input(self, key, file_paths=None, end_input=False, **append):
            raise ValueError("the call's input passed its limit")

        def reject_input(self, key, error, status):
            return False  # the model ended the call meanwhile

    fake = _EndsMeanwhile(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]},
    ])
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    (late,) = ws.sent
    assert "not taking input" in late["error"] and "status" not in late


def test_an_append_that_fails_internally_hides_the_error_from_the_client(monkeypatch, tmp_path):
    class _Fails(_StreamingFakeServer):
        def stream_input(self, key, file_paths=None, end_input=False, **append):
            raise RuntimeError("/srv/weights/secret broke")

        def reject_input(self, key, error, status):
            self.calls.append(("reject", error, status))
            return True

    fake = _Fails(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]},
    ])
    asyncio.run(_serve(ws, until=lambda: any(c[0] == "reject" for c in fake.calls)))

    (_, error, status), = [c for c in fake.calls if c[0] == "reject"]
    assert status == 500 and error.startswith("append rejected: internal error") and "secret" not in error


def test_an_append_the_call_stops_taking_meanwhile_is_answered_as_late(monkeypatch, tmp_path):
    """The call ends between the append's check and its feeding: the append
    is answered as late, and the call is left alone."""

    class _EndsMeanwhile(_StreamingFakeServer):
        def stream_input(self, key, file_paths=None, end_input=False, **append):
            return False  # the model ended the call meanwhile

        def reject_input(self, key, error, status):
            self.calls.append(("reject", key))
            return True

    fake = _EndsMeanwhile(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "call", "output_modalities": "text"},
        {"request_id": "call", "append": [{"name": "c.wav", "data": "UklGRg=="}]},
    ])
    asyncio.run(_serve(ws, until=lambda: ws.sent))

    (late,) = ws.sent
    assert late["append"] is True and "not taking input" in late["error"] and "status" not in late
    assert not any(c[0] == "reject" for c in fake.calls)


def test_replies_to_appends_never_hold_up_reading(monkeypatch, tmp_path):
    """A client that writes before it reads: the reply to a late append waits
    behind output the client hasn't read, and reading the socket must go on
    meanwhile, or the client's own writes would block."""

    class _UnreadSocket(_QueuedFramesSocket):
        async def send_text(self, text):
            await self.closed.wait()  # the client isn't reading yet

    fake = _StreamingFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    wav = {"name": "c.wav", "data": "UklGRg=="}
    ws = _UnreadSocket([
        {"request_id": "a", "output_modalities": "text"},
        {"request_id": "a", "end_input": True},
        {"request_id": "a", "append": [wav]},                  # late
        {"request_id": "nobody", "append": [wav]},             # not this socket's
        {"request_id": "b", "text": "next", "end_input": True},
    ])
    asyncio.run(_serve(ws, until=lambda: any(c[1] == "b" for c in fake.calls)))

    assert ("submit", "b", True) in fake.calls


@pytest.mark.parametrize("frames", [
    [{"request_id": "a", "output_modalities": "text"}],                  # a 409
    [{"request_id": 7, "text": "x"}],                                    # a bad id
    [{"output_modalities": "text"}],                                     # a live call without an id
    [{"request_id": "b", "output_modalities": "text"},                   # a call that fails to open,
     {"request_id": "b", "append": [{"name": "c.wav", "data": "UklGRg=="}]}],  # then its append
], ids=["in-use", "bad-id", "no-id", "failed-call-append"])
def test_refusals_never_hold_up_reading(monkeypatch, tmp_path, frames):
    """As above, for the receive loop's other replies, and for an append
    waiting on its call's opening, which failed and is still sending why."""

    class _UnreadSocket(_QueuedFramesSocket):
        async def send_text(self, text):
            await self.closed.wait()  # the client isn't reading yet

    class _Busy(_StreamingFakeServer):
        def submit_request(self, **kwargs):
            if kwargs["request_id"] == "b":
                raise HTTPException(status_code=503, detail="server busy")
            return super().submit_request(**kwargs)

    fake = _Busy(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _UnreadSocket([
        {"request_id": "a", "output_modalities": "text"},
        *frames,
        {"request_id": "c", "text": "next", "end_input": True},
    ])
    asyncio.run(_serve(ws, until=lambda: any(c[1] == "c" for c in fake.calls)))

    assert ("submit", "c", True) in fake.calls



def test_every_append_queued_behind_a_refused_opening_is_answered(monkeypatch, tmp_path):
    """Appends read before their call's opening was refused each get the late
    reply, the later ones too: each waits on the call it named when it was
    read, not on whatever holds that id once it runs."""

    class _Busy(_StreamingFakeServer):
        def submit_request(self, **kwargs):
            if kwargs["request_id"] == "b":
                raise HTTPException(status_code=503, detail="server busy")
            return super().submit_request(**kwargs)

    fake = _Busy(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    wav = {"name": "c.wav", "data": "UklGRg=="}
    ws = _QueuedFramesSocket([
        {"request_id": "b", "output_modalities": "text"},
        *[{"request_id": "b", "append": [wav]} for _ in range(3)],
    ])
    asyncio.run(_serve(ws, until=lambda: sum(1 for m in ws.sent if m.get("append")) == 3))

    late = [m for m in ws.sent if m.get("append")]
    assert len(late) == 3 and all("not taking input" in m["error"] for m in late)
    assert [m["status"] for m in ws.sent if not m.get("append")] == [503]


def test_an_append_file_that_cannot_be_deleted_does_not_end_the_socket(monkeypatch, tmp_path):
    class _Refuses(_StreamingFakeServer):
        def stream_input(self, key, file_paths=None, end_input=False, **append):
            raise ValueError("over the input budget")

        def reject_input(self, key, error, status):
            return True

    stuck = tmp_path / "stuck"
    stuck.mkdir()  # unlink fails, and not with "no such file"
    monkeypatch.setattr(entrypoint, "_decode_ws_files", lambda files, upload_dir: ({"audio": [str(stuck)]}, []))
    fake = _Refuses(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    ws = _QueuedFramesSocket([
        {"request_id": "a", "output_modalities": "text"},
        {"request_id": "a", "append": [{"name": "c.wav", "data": "UklGRg=="}]},
        {"request_id": "c", "text": "next", "end_input": True},
    ])
    asyncio.run(_serve(ws, until=lambda: any(c[1] == "c" for c in fake.calls)))

    assert ("submit", "c", True) in fake.calls


def _waiting_rig(tmp_path):
    """A thread whose API server holds the submissions in ``pending``, with
    the main-thread facade an abort goes through (it lets go first)."""
    from mstar.api_server.data_worker import PreprocessWorker

    pending: set[str] = set()
    wt, sent, _tm = _thread(is_pending=pending.__contains__)
    facade = PreprocessWorker.__new__(PreprocessWorker)  # the main-thread side
    facade.request_input_queue, facade.abort_request_queue = wt.in_queue, wt.abort_request_queue
    facade._signal = lambda: None
    facade.per_request_reading_tensors, facade.received_chunk_counts = {}, {}

    def abort(key):
        pending.discard(key)
        facade.abort_request(key)

    return wt, sent, pending, abort


def _run_until(wt, predicate):
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    _wait(predicate)
    time.sleep(0.2)
    wt.stop_event.set()
    runner.join(timeout=10)


def test_an_abort_drops_a_call_still_waiting_for_preprocessing(tmp_path):
    """The client closes while its call's opening still waits in the data
    worker: the call and its appends are never sent on, and the appends'
    files are deleted. The opening's own files are its handler's to delete."""
    wt, sent, pending, abort = _waiting_rig(tmp_path)
    opening_file = _audio_file(tmp_path, "open.raw", 4)
    append_file = _audio_file(tmp_path, "c.raw", 4)
    for rid in ("r#1", "r#2"):
        pending.add(rid)
        wt.in_queue.put(PreprocessInput(
            request_id=rid, text="hi", file_paths={"audio": [opening_file]} if rid == "r#1" else None,
            input_modalities=["text"], output_modalities=["text"], model_kwargs={},
        ))
    wt.in_queue.put(ClientAppend("r#1", file_paths={"audio": [append_file]}, ends=END))
    abort("r#1")
    _run_until(wt, lambda: _bodies(sent, ConductorMessageType.NEW_REQUEST))

    assert [b.request_id for b in _bodies(sent, ConductorMessageType.NEW_REQUEST)] == ["r#2"]
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.ABORT_REQUEST)] == ["r#1"]
    assert _signals(sent, "r#1") == []
    assert not (tmp_path / "c.raw").exists() and (tmp_path / "open.raw").exists()
    assert "r#1" not in wt.in_flight_requests


def test_an_abort_drops_a_running_calls_queued_appends(tmp_path):
    """The call is running and its appends still wait when the client
    closes: they are never sent, and their files are deleted."""
    wt, sent, pending, abort = _waiting_rig(tmp_path)
    pending.add("r")
    _open(wt)
    _route(wt)
    for i in range(2):
        wt.in_queue.put(ClientAppend("r", file_paths={"audio": [_audio_file(tmp_path, f"{i}.raw", 2)]}))
    wt.in_queue.put(ClientAppend("r", ends=END))
    abort("r")
    sent.clear()
    _run_until(wt, lambda: _bodies(sent, ConductorMessageType.ABORT_REQUEST))

    assert [b.request_id for b in _bodies(sent, ConductorMessageType.ABORT_REQUEST)] == ["r"]
    assert _signals(sent) == []
    assert list(tmp_path.iterdir()) == []


def test_an_abort_after_the_pending_check_tears_the_call_down(tmp_path):
    """The abort lands just after the thread found the opening still pending:
    the opening goes to the conductor, its abort after it from the same
    thread, and the thread drains the call as any running one."""
    wt, sent, pending, abort = _waiting_rig(tmp_path)
    pending.add("r")
    check = wt._is_pending

    def pending_then_aborted(key):
        held = check(key)
        if held:
            wt._is_pending = check
            abort(key)
        return held

    wt._is_pending = pending_then_aborted
    wt.in_queue.put(PreprocessInput(
        request_id="r", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    _run_until(wt, lambda: _bodies(sent, ConductorMessageType.ABORT_REQUEST))

    assert [(msg.message_type, msg.body.request_id) for _e, msg in sent[:2]] == [
        (ConductorMessageType.NEW_REQUEST, "r"), (ConductorMessageType.ABORT_REQUEST, "r"),
    ]
    # drained as a running call: with no read in flight it ACKs at once
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.READS_DONE)] == ["r"]


def test_the_api_server_holds_a_submission_until_it_ends():
    """is_pending, which the preprocess thread checks before sending queued
    input on: false once the request is aborted or has ended (a rejected
    append ends its call while the handler still holds it)."""
    server = _server()
    call = _submit(server, "call", end_input=False)
    other = _submit(server, "other")
    assert server.is_pending(call) and server.is_pending(other)
    assert server.reject_input(call, "append rejected: bad file", 400)
    server.abort_submission(other)
    assert not server.is_pending(call) and not server.is_pending(other)
    assert not server.is_pending("never#1")


def test_input_is_taken_in_arrival_order_up_to_one_request_per_pass():
    """Appends are cheap next to a new request's preprocess: a pass takes up
    to MAX_APPENDS_PER_PASS of them, in arrival order, and stops at the first
    new request."""
    wt, _sent, _tm = _thread()

    def opening(rid):
        return PreprocessInput(
            request_id=rid, text="hi", file_paths=None, input_modalities=["text"],
            output_modalities=["text"], model_kwargs={},
        )

    def take():
        return [(type(i).__name__, i.request_id) for i in wt._next_inputs()]

    for item in (ClientAppend("r"), opening("u"), ClientAppend("r", ends=END), ClientAppend("u")):
        wt.in_queue.put(item)
    assert take() == [("ClientAppend", "r"), ("PreprocessInput", "u")]
    assert take() == [("ClientAppend", "r"), ("ClientAppend", "u")]
    assert take() == []

    for i in range(wt.MAX_APPENDS_PER_PASS + 3):
        wt.in_queue.put(ClientAppend(f"c{i}"))
    assert len(take()) == wt.MAX_APPENDS_PER_PASS and len(take()) == 3


def test_one_failing_release_leaves_the_rest_of_the_pass_to_run(tmp_path):
    """A pass takes several calls' appends and a new request at once. An
    abort that cannot be sent, released while the first is processed, is
    that abort's failure alone: the others still reach the conductor."""
    class _Hooked(_ListeningModel):
        def process_stream_chunk(self, chunk):
            wt.abort_request_queue.put("unsendable")
            return super().process_stream_chunk(chunk)

    def encode(msg):
        if msg.body.request_id == "unsendable":
            raise UnicodeEncodeError("utf-8", "unsendable", 0, 1, "refused")
        return msg

    wt, sent, _tm = _thread(_Hooked())
    wt.communicator.codec = SimpleNamespace(encode=encode)
    for rid in ("a", "b"):
        wt._process_input(PreprocessInput(
            request_id=rid, text="hi", file_paths=None, input_modalities=["text"],
            output_modalities=["text"], model_kwargs={},
        ))
        _route(wt, rid)
    wt.in_queue.put(ClientAppend("a", file_paths={"audio": [_audio_file(tmp_path, "a.raw", 2)]}))
    wt.in_queue.put(ClientAppend("b", file_paths={"audio": [_audio_file(tmp_path, "b.raw", 3)]}, ends=END))
    wt.in_queue.put(PreprocessInput(
        request_id="d", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs={},
    ))
    sent.clear()
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    _wait(lambda: _bodies(sent, ConductorMessageType.NEW_REQUEST))
    wt.stop_event.set()
    runner.join(timeout=10)

    assert _signals(sent, "a") == [("w0", "audio_in", [[2]], {})]
    assert _signals(sent, "b") == [("w0", "audio_in", [[3]], {}), ("w0", None, [], {"audio_in": 1})]
    assert [b.request_id for b in _bodies(sent, ConductorMessageType.NEW_REQUEST)] == ["d"]
    assert list(tmp_path.iterdir()) == []


def test_max_call_input_mb_reads_in_mib():
    assert entrypoint._call_input_limit(None) == 512 << 20
    server = _server()
    server.max_call_input_bytes = entrypoint._call_input_limit({"max_call_input_mb": 0.5})
    with pytest.raises(ValueError, match=r"passed 0\.5 MiB"):
        server._check_call_input(1 << 20)


def test_websocket_deflate_is_a_serving_setting():
    """Per-message deflate on the WebSocket routes is the deployment's choice:
    on unless the serving config turns it off (as the duplex configs do, where
    compressing live audio cost the event loop more than it saved)."""
    app = object()
    assert entrypoint._uvicorn_config(app, "127.0.0.1", 8000, {}).ws_per_message_deflate is True
    off = entrypoint._uvicorn_config(app, "127.0.0.1", 8000, {"ws_per_message_deflate": False})
    assert off.ws_per_message_deflate is False
