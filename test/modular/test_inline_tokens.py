"""The inline token path: a decode step's sampled tokens ride to the api server
as one RESULT_TOKENS frame (values, not tensors). Covers the wire type, the
api server's data worker, and the worker's value extraction."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import torch

from mstar.api_server.data_worker import InlineTokens, PreprocessWorker, PreprocessWorkerThread, ResultQueue
from mstar.api_server.request_types import APIServerMessage, ResultChunk, ResultTokens
from mstar.communication import wire
from mstar.communication.event import EventWakeup
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.base import Model
from mstar.model.submodule_base import ARNodeSubmodule, DeviceLoopback, HostRows, TokenStop
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
    # the inline path's per-value default, over the stub's postprocess
    postprocess_inline = Model.postprocess_inline
    text_output = Model.text_output

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
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=ResultQueue(EventWakeup()),
        profile_queue=ResultQueue(EventWakeup()), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=_Comm(),
        tensor_manager=_TM(), model=_Model(),
    )
    return worker, stop


class _FailsOn(_Model):
    def __init__(self, bad):
        self.bad = bad

    def postprocess(self, output, modality, request_kwargs=None):
        if int(output[0]) == self.bad:
            raise ValueError("bad token")
        return super().postprocess(output, modality, request_kwargs)


def _run_until(worker, stop, n):
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        return [worker.out_queue.get(timeout=5) for _ in range(n)]
    finally:
        stop.set()
        thread.join(timeout=5)


def test_inline_tokens_are_detokenized_in_order_without_a_read():
    worker, stop = _thread()
    for v in (11, 12, 13):
        worker.result_tensor_queue.put(InlineTokens(["req", "other"], [v, v + 80], "text"))
    chunks = _run_until(worker, stop, 6)
    assert all(isinstance(c, ResultChunk) for c in chunks)
    assert [c.data for c in chunks if c.request_id == "req"] == [b"<11>", b"<12>", b"<13>"]
    assert [c.data for c in chunks if c.request_id == "other"] == [b"<91>", b"<92>", b"<93>"]
    state = worker.request_output_state["req"]
    assert state.next_sequence == 3 and state.next_emit == 3 and not state.pending


def test_a_draining_request_drops_its_inline_tokens():
    worker, stop = _thread()
    worker._draining_rids.add("gone")
    worker.result_tensor_queue.put(InlineTokens(["gone", "live"], [1, 2], "text"))
    [chunk] = _run_until(worker, stop, 1)
    assert chunk.request_id == "live" and chunk.data == b"<2>"
    assert worker.out_queue.empty()
    assert "gone" not in worker.request_output_state


def test_a_row_that_fails_postprocessing_fails_only_its_request():
    worker, stop = _thread()
    worker.model = _FailsOn(13)
    worker.result_tensor_queue.put(InlineTokens(["bad", "good"], [13, 2], "text"))
    chunks = {c.request_id: c for c in _run_until(worker, stop, 2)}
    assert chunks["good"].modality == "text" and chunks["good"].data == b"<2>"
    assert chunks["bad"].modality == "error" and b"bad token" in chunks["bad"].data
    assert chunks["bad"].metadata["status"] == 400


def test_the_facade_accounts_an_inline_token_like_a_tensor():
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.result_tensor_input_queue = queue.Queue()
    facade.per_request_reading_tensors = {"r1": 0, "r2": 0}
    facade.received_chunk_counts = {"r1": {}, "r2": {}}
    wakes = []
    facade._signal = lambda: wakes.append(1)
    facade.new_result_tokens(["r1", "r2"], [42, 52], "text_inputs", "text")
    # a request already cleaned up is dropped from its frame, without raising
    facade.new_result_tokens(["r1", "unknown"], [43, 1], "text_inputs", "text")
    # a frame with nothing known queues nothing and wakes no one
    facade.new_result_tokens(["gone"], [7], "text_inputs", "text")
    assert len(wakes) == 2, "one wake per frame that queued an item"
    assert facade.per_request_reading_tensors == {"r1": 2, "r2": 1}
    assert facade.received_chunk_counts == {"r1": {"text_inputs": 2}, "r2": {"text_inputs": 1}}
    items = [facade.result_tensor_input_queue.get_nowait() for _ in range(2)]
    assert [(i.request_ids, i.values) for i in items] == [(["r1", "r2"], [42, 52]), (["r1"], [43])]
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

    def __init__(self, walks=None, label="decode", withheld=None):
        self.graph_walk = label
        self.batch = _Scheduled(walks or {})
        self.node_batch = SimpleNamespace(withheld_outputs=withheld or {})


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


def test_a_withheld_output_rides_no_inline_value():
    """A token the stop check withheld (``StopCheck.withheld``, e.g. the step
    closing a reply that ended at prefill) is not the client's: that row gets
    None, which the runtime routes as the row's tensors, of which it has none."""
    rows = HostRows((7, 8), {"new_token": torch.tensor([[70], [80]])})
    sig, vals = _worker()._inline_emit_values(
        _Engine({"new_token": "new_token"}), _Batch(withheld={8: ("new_token",), 7: ("frame",)}),
        rows, [7, 8], ["new_token"],
    )
    assert sig == "new_token" and vals == [70, None]


def test_the_step_closing_a_loop_back_reply_that_ended_at_prefill_sends_no_token():
    """A loop-back decode's client edge carries its loop-back signal, inline off
    the token's buffer, so the step closing a reply that ended at prefill has
    to withhold the signal too, or the token sampled after the stop id reaches
    the client."""

    class _Decoder(ARNodeSubmodule):
        prepare_inputs = preprocess = forward = None

    sub = _Decoder()
    sub.token_stop = TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset({0}), sampler="sampler",
        walk="decode", prefill_walks=frozenset({"prefill"}),
    )
    sub.device_loopback = DeviceLoopback(walk="decode", signal="text_inputs", sampler="sampler", buffer="new_token")

    def info(walk):
        return SimpleNamespace(graph_walk=walk, resource_configs={"sampler": SimpleNamespace(ignore_eos=False)})

    # the prefill samples a stop id for 8, so its first decode step only closes the loop
    sub.check_stop_batched([7, 8], {7: info("prefill"), 8: info("prefill")},
                           HostRows((7, 8), {"new_token": torch.tensor([[5], [0]])}))
    rows = HostRows((7, 8), {"new_token": torch.tensor([[70], [80]])})
    check = sub.check_stop_batched([7, 8], {7: info("decode"), 8: info("decode")}, rows)
    assert check.stops == {8: {"decode_loop"}}
    sig, vals = _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), _Batch(withheld=check.withheld), rows, [7, 8], ["text_inputs"],
    )
    assert sig == "text_inputs" and vals == [70, None]


def test_a_withheld_output_skips_the_fused_route_and_send():
    """The fused device-loopback call reads every row's value off the column,
    so a withheld row would still ride inline: such a step takes the general
    path, which sends that row None."""
    class _Fused(_Runtime):
        supports_route_and_send_inline = True

        def route_and_send_inline(self, **kwargs):
            raise AssertionError("a withheld row would ride inline")

    class _LoopbackEngine(_Engine):
        def device_loopback_signals(self, node_name, graph_walk):
            return ["new_token"]

    w = _worker(_Fused())
    w.enable_prof, w._device_loopback = False, {}
    batch = _Batch(withheld={8: ("new_token",)})
    batch.partition = "p"
    batch.batch.output_signals, batch.batch.graph_walk = ["new_token"], "decode"
    batch.node_batch.stream_partition_done_rids = ()
    rows = HostRows((7, 8), {"new_token": torch.tensor([[70], [80]])})
    eng = _LoopbackEngine({"new_token": "new_token"})
    assert w._route_and_send_fast(batch, rows, eng, [7, 8], set()) is False


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


def test_a_mixed_step_inlines_its_decode_rows():
    # labelled by the combined walk, which the submodule does not inline; its
    # decode rows still go inline, and the prefill row carries its own token
    rows = HostRows((7, 8, 9), {"new_token": torch.tensor([[70], [80], [90]])})
    batch = _Batch({7: "decode", 8: "prefill_text", 9: "decode"}, label="mixed")
    sig, vals = _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), batch, rows, [7, 8, 9], ["text_inputs"],
    )
    assert sig == "text_inputs" and vals == [70, 80, 90]


def test_a_mixed_step_falls_back_when_a_decode_row_has_no_value():
    rows = HostRows((7,), {"new_token": torch.tensor([[70]])})
    batch = _Batch({7: "prefill_text", 9: "decode"}, label="mixed")
    assert _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), batch, rows, [7, 9], ["text_inputs"],
    ) == (None, [])
    # a row of a walk that sends no inline value needs none
    batch = _Batch({7: "decode", 9: "prefill_text"}, label="mixed")
    assert _worker()._inline_emit_values(
        _Engine({"text_inputs": "new_token"}), batch, rows, [7, 9], ["text_inputs"],
    ) == ("text_inputs", [70, 0])
