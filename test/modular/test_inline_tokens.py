"""The inline token path: a decode step's sampled tokens ride to the api server
as one RESULT_TOKENS frame (values, not tensors). Covers the wire type, the
api server's data worker, and the worker's value extraction."""
from __future__ import annotations

import queue
import threading

import torch

from mstar.api_server.data_worker import InlineToken, PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.request_types import APIServerMessage, ResultChunk, ResultTokens
from mstar.communication import wire
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
    def postprocess(self, output, modality, request_kwargs=None):
        assert modality == "text"
        return f"<{int(output[0])}>".encode()


class _Comm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


class _TM:
    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
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
    facade.output_loop_idxs = {"r1": {}}
    facade.new_result_token("r1", 42, _loop(4), "text_inputs", "text")
    facade.new_result_token("r1", 43, _loop(5), "text_inputs", "text")
    facade.new_result_token("unknown", 1, _loop(0), "text_inputs", "text")  # dropped, no raise
    assert facade.per_request_reading_tensors["r1"] == 2
    assert facade.output_loop_idxs["r1"]["text_inputs"].loop_indices == {"decode_loop": 5}
    items = [facade.result_tensor_input_queue.get_nowait() for _ in range(2)]
    assert [i.value for i in items] == [42, 43]
    assert facade.result_tensor_input_queue.empty()


# -- worker ------------------------------------------------------------------


class _Runtime:
    supports_inline_emit = True


class _Engine:
    def __init__(self, signals):
        self._signals = signals

    def inline_client_signals(self, node_name, graph_walk):
        return self._signals


class _Batch:
    node_name = "LLM"
    graph_walk = "decode"


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


def test_inline_values_fall_back_when_a_rid_has_no_row_or_nothing_applies():
    rows = HostRows((7, 8), {"new_token": torch.tensor([70, 80])})
    w = _worker()
    eng = _Engine({"text_inputs": "new_token"})
    none = (None, [])
    assert w._inline_emit_values(eng, _Batch(), rows, [7, 8, 9], ["text_inputs"]) == none
    assert w._inline_emit_values(eng, _Batch(), rows, [7], ["other"]) == none
    assert w._inline_emit_values(_Engine({}), _Batch(), rows, [7], ["text_inputs"]) == none
    assert w._inline_emit_values(eng, _Batch(), None, [7], ["text_inputs"]) == none

    class _NoInline:
        supports_inline_emit = False
    assert _worker(_NoInline())._inline_emit_values(
        eng, _Batch(), rows, [7, 8], ["text_inputs"],
    ) == none


def test_a_model_that_decodes_ids_gets_the_inline_value_as_a_list():
    """The per-token tensor build is skipped for a model that opts in; the
    default model still receives a tensor."""
    from mstar.api_server.data_worker import InlineToken, PreprocessWorkerThread

    class _Ids(_Model):
        inline_postprocess_takes_ids = True

        def postprocess(self, output, modality, request_kwargs=None):
            assert isinstance(output, list) and output == [5]
            return b"<5>"

    def _worker(model):
        w = object.__new__(PreprocessWorkerThread)
        w.model = model
        w.request_output_state = {}
        w.request_model_kwargs = {}
        w.queued = []
        w._queue_completed_output = lambda rid, seq, chunk: w.queued.append((rid, seq, chunk.data))
        w._fail_request = lambda *a, **kw: w.queued.append(("fail", a))
        return w

    item = InlineToken(request_id="r1", value=5, loop_indices=None, modality="text")
    w = _worker(_Ids())
    w._emit_inline_token(item)
    assert w.queued == [("r1", 0, b"<5>")]

    seen = {}

    class _Tensor(_Model):
        def postprocess(self, output, modality, request_kwargs=None):
            seen["type"] = type(output).__name__
            return b"x"

    w = _worker(_Tensor())
    w._emit_inline_token(item)
    assert seen["type"] == "Tensor" and w.queued == [("r1", 0, b"x")]
