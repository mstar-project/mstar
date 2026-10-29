"""The inline token path: a decode step's sampled tokens ride to the api server
as one RESULT_TOKENS frame (values, not tensors). Covers the wire type, the
api server's data worker, and the worker's value extraction."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import torch

from mstar.api_server.data_worker import InlineToken, PreprocessWorker, PreprocessWorkerThread, RequestOutputState
from mstar.api_server.request_types import APIServerMessage, ResultChunk, ResultTokens
from mstar.communication import wire
from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.submodule_base import HostRows
from mstar.worker.worker import Worker

# -- wire --------------------------------------------------------------------


def _loop(i):
    return NestedLoopIndices(loop_name_order=["decode_loop"], loop_indices={"decode_loop": i}, wg_fwd_pass_idx=i + 1)


def test_result_tokens_round_trips_through_the_codec():
    msg = APIServerMessage(
        message_type="result_tokens",
        body=ResultTokens(
            request_ids=["r1", "r2", "r3"], values=[5, 151645, 7],
            loop_indices=[_loop(3), None, _loop(9)],
            signal="text_inputs", modality="text",
        ),
    )
    back = wire.decode(wire.encode(msg))
    assert back.message_type == "result_tokens"
    assert isinstance(back.body, ResultTokens)
    assert back.body.request_ids == ["r1", "r2", "r3"]
    assert back.body.values == [5, 151645, 7]
    assert back.body.loop_indices[1] is None
    assert back.body.loop_indices[0].loop_indices == {"decode_loop": 3}
    assert back.body.loop_indices[2].wg_fwd_pass_idx == 10
    assert back.body.signal == "text_inputs" and back.body.modality == "text"


# -- api server --------------------------------------------------------------


class _Model:
    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])

    def postprocess(self, output, modality, request_kwargs=None):
        assert modality == "text"
        return f"<{int(output[0])}>".encode()


class _Comm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True


class _TM:
    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        return []

    def take_failed_reads(self):

        return []


    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass

    def ack_unread_tensors(self, request_id, graph_edges):
        pass


def _thread():
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=_Comm(),
        tensor_manager=_TM(), model=_Model(),
    )
    return worker, stop


def test_inline_tokens_are_detokenized_in_order_without_a_read():
    worker, stop = _thread()
    for i, v in enumerate((11, 12, 13)):
        worker.result_tensor_queue.put(InlineToken("req", v, _loop(i), "text"))
    worker.result_tensor_queue.put(InlineToken("other", 99, None, "text"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunks = [worker.out_queue.get(timeout=5) for _ in range(4)]
    finally:
        stop.set()
        thread.join(timeout=5)
    assert all(isinstance(c, ResultChunk) for c in chunks)
    assert [c.data for c in chunks if c.request_id == "req"] == [b"<11>", b"<12>", b"<13>"]
    assert [c.data for c in chunks if c.request_id == "other"] == [b"<99>"]
    state = worker.request_output_state["req"]
    assert state.next_sequence == 3 and state.next_emit == 3 and not state.pending


def test_turn_tokens_mark_their_text_chunks():
    """A model's declared turn tokens mark the chunks they arrive in, though
    postprocess may drop the tokens themselves; other chunks are unmarked.
    Every chunk carries its loop iteration as its step."""
    worker, stop = _thread()
    worker.model.turn_tokens = (1, 2)
    for i, v in enumerate((12, 1, 13, 2, 12)):
        worker.result_tensor_queue.put(InlineToken("req", v, _loop(i), "text"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunks = [worker.out_queue.get(timeout=5) for _ in range(5)]
    finally:
        stop.set()
        thread.join(timeout=5)
    assert [c.metadata.get("turn") for c in chunks] == [None, ["start"], None, ["end"], None]
    assert [c.metadata["step"] for c in chunks] == [0, 1, 2, 3, 4]
    # a chunk of several tokens keeps every mark, in order
    assert worker._turn_metadata([2, 12, 1], None) == {"turn": ["end", "start"]}


def test_turn_tokens_stamp_read_text_and_audio_with_their_step():
    """Chunks read as tensors (text without inline tokens, and all audio)
    carry their loop step as the inline path's do."""
    class _ReadModel(_Model):
        turn_tokens = (1, 2)

        def postprocess(self, output, modality, request_kwargs=None):
            return b"pcm" if modality == "audio" else super().postprocess(output, modality)

        def get_output_sample_rate(self, modality="audio"):
            return 24000

        def get_output_audio_channels(self, modality="audio"):
            return 1

    tensors = {"t": torch.tensor([1]), "a": torch.zeros(4, dtype=torch.int16)}

    class _ReadTM(_TM):
        def get_ready_tensors(self):
            return {"req": [GraphEdge(next_node="api", name=f"{modality}_output", tensor_info=[SimpleNamespace(uuid=u)])
                            for modality, u in (("text", "t"), ("audio", "a"))]}

        def get_tensor(self, uuid):
            return tensors[uuid]

        def dereference(self, uuid):
            pass

    worker, _ = _thread()
    worker.model, worker.tensor_manager = _ReadModel(), _ReadTM()
    worker.request_output_state = {"req": RequestOutputState(order={"t": (0, _loop(3)), "a": (1, _loop(5))},
                                                             next_sequence=2)}
    worker.tensor_uuid_to_metadata_per_request = {"req": {"t": {}, "a": {}}}
    assert worker._process_read_tensors() is True
    text, audio = worker.out_queue.get_nowait(), worker.out_queue.get_nowait()
    assert text.metadata == {"step": 3, "turn": ["start"]}
    assert audio.modality == "audio" and audio.metadata["step"] == 5


def test_a_draining_request_drops_its_inline_tokens():
    worker, stop = _thread()
    worker._draining_rids.add("gone")
    worker.result_tensor_queue.put(InlineToken("gone", 1, None, "text"))
    worker.result_tensor_queue.put(InlineToken("live", 2, None, "text"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunk = worker.out_queue.get(timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert chunk.request_id == "live" and chunk.data == b"<2>"
    assert worker.out_queue.empty()


def test_the_facade_accounts_an_inline_token_like_a_tensor():
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.result_tensor_input_queue = queue.Queue()
    facade.per_request_reading_tensors = {"r1": 0}
    facade.received_chunk_counts = {"r1": {}}
    facade.new_result_token("r1", 42, _loop(4), "text_inputs", "text")
    facade.new_result_token("r1", 43, _loop(5), "text_inputs", "text")
    facade.new_result_token("unknown", 1, _loop(0), "text_inputs", "text")  # dropped, no raise
    assert facade.per_request_reading_tensors["r1"] == 2
    assert facade.received_chunk_counts["r1"] == {"text_inputs": 2}
    items = [facade.result_tensor_input_queue.get_nowait() for _ in range(2)]
    assert [i.value for i in items] == [42, 43]
    assert facade.result_tensor_input_queue.empty()


# -- worker ------------------------------------------------------------------


class _Runtime:
    supports_inline_emit = True


class _Engine:
    def __init__(self, signals, walks=("decode",)):
        self._signals = signals
        self._walks = walks

    def inline_client_signals(self, node_name, graph_walk):
        # as the submodule: a walk it does not inline, and a combined label, get none
        return self._signals if graph_walk in self._walks else {}


class _Scheduled:
    def __init__(self, walks):
        self._walks = walks
        # rid -> real walk on a mixed step only, as ScheduledBatch keeps it
        self.request_walks = dict(walks) if len(set(walks.values())) > 1 else {}

    def walk_of(self, rid):
        return self._walks.get(rid, "decode")

    def rids_by_walk(self):
        out = {}
        for rid, walk in self._walks.items():
            out.setdefault(walk, []).append(rid)
        return out or {"decode": []}


class _Batch:
    node_name = "LLM"

    def __init__(self, walks=None, label="decode"):
        self.graph_walk = label
        self.batch = _Scheduled(walks or {})


def _worker(runtime=_Runtime()):
    w = Worker.__new__(Worker)
    w._graph_runtime = runtime
    return w


def test_inline_values_follow_the_batch_order_off_the_host_rows():
    rows = HostRows((7, 8, 9), {"new_token": torch.tensor([[70], [80], [90]])})
    sig, vals = _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), _Batch(), rows, [9, 7, 8], ["text_inputs", "kv"],
    )
    assert sig == "text_inputs" and vals == [90, 70, 80]


def test_a_row_that_ran_no_forward_takes_the_tensor_path_alone():
    # 8 ran no forward (the cache served it, or it declined its stream's final
    # chunk), so it has no host row; on a rows-only step the tensor path has
    # nothing for 7 and 9, so they must still go inline
    rows = HostRows((7, 9), {"new_token": torch.tensor([[70], [90]])})
    sig, vals = _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), _Batch(), rows, [7, 8, 9], ["text_inputs"],
    )
    assert sig == "text_inputs" and vals == [70, None, 90]


def test_inline_values_fall_back_when_nothing_applies():
    rows = HostRows((7, 8), {"new_token": torch.tensor([70, 80])})
    w = _worker()
    eng = _Engine({"text_inputs": "new_token"})
    none = (None, [])
    assert w._inline_emit_values(eng, _Batch(), rows, [7], ["other"]) == none
    assert w._inline_emit_values(_Engine({}), _Batch(), rows, [7], ["text_inputs"]) == none
    assert w._inline_emit_values(eng, _Batch(), None, [7], ["text_inputs"]) == none

    class _NoInline:
        supports_inline_emit = False
    assert _worker(_NoInline())._inline_emit_values(
        eng, _Batch(), rows, [7, 8], ["text_inputs"],
    ) == none


def test_a_mixed_step_inlines_its_decode_rows():
    # labelled by the combined walk, which the submodule does not inline; its
    # decode rows still go inline, and the prefill row carries its own token
    rows = HostRows((7, 8, 9), {"new_token": torch.tensor([[70], [80], [90]])})
    batch = _Batch({7: "decode", 8: "prefill_text", 9: "decode"}, label="mixed")
    sig, vals = _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), batch, rows, [7, 8, 9], ["text_inputs"],
    )
    assert sig == "text_inputs" and vals == [70, 80, 90]


def test_a_mixed_step_row_with_no_value_takes_the_tensor_path():
    rows = HostRows((7,), {"new_token": torch.tensor([[70]])})
    for walks in ({7: "prefill_text", 9: "decode"}, {7: "decode", 9: "prefill_text"}):
        assert _worker()._inline_emit_values(
            _Engine({"text_inputs": "new_token"}), _Batch(walks, label="mixed"), rows, [7, 9],
            ["text_inputs"],
        ) == ("text_inputs", [70, None])
