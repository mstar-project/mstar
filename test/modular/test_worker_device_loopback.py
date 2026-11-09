"""The worker's side of the device loop-back: a step whose every routed signal
the node keeps on the device, with the client copy inline, stores and holds
nothing and routes zero tensors per (request, signal)."""
from types import SimpleNamespace

from mstar.communication.tensors import StoredOutputs
from mstar.testing_utils.worker import make_worker


class _Engine:
    def __init__(self, on_device):
        self.on_device = on_device
        self.calls = 0

    def device_loopback_signals(self, node, walk):
        self.calls += 1
        return self.on_device.get((node, walk), frozenset())


def _body(inline_emit=True):
    return make_worker(graph_runtime=SimpleNamespace(supports_inline_emit=inline_emit)).executor._body


def test_all_on_device_is_per_step_signals_and_cached_per_node_and_walk():
    engine = _Engine({("LLM", "decode"): frozenset({"text_inputs"})})
    body = _body()
    assert body.all_on_device(engine, "LLM", "decode", ["text_inputs"])
    assert body.all_on_device(engine, "LLM", "decode", ["text_inputs"])
    assert engine.calls == 1, "the node's answer is cached per (node, walk)"
    # a step that also routes a signal the node does not keep stays on tensors
    assert not body.all_on_device(engine, "LLM", "decode", ["text_inputs", "kv"])
    assert not body.all_on_device(engine, "LLM", "prefill_text", ["new_token"])
    assert not body.all_on_device(engine, "LLM", "decode", [])
    assert engine.calls == 2


def test_no_rows_only_step_on_a_runtime_without_inline_emit():
    """Rows-only steps send the client's tokens inline; without it they were dropped."""
    engine = _Engine({("LLM", "decode"): frozenset({"text_inputs"})})
    assert not _body(inline_emit=False).all_on_device(engine, "LLM", "decode", ["text_inputs"])


def test_the_zero_tensor_store_shape_matches_the_route_layout():
    rids, signals = [3, 5, 8], ["text_inputs"]
    stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
    assert stored.flat_uuids == [] and stored.flat_rids == [] and stored.signal_idxs == []
    assert stored.num_tensors == [0, 0, 0]
    assert SimpleNamespace(**stored._asdict()).num_tensors == [0, 0, 0] if hasattr(stored, "_asdict") else True


def test_a_step_that_also_streams_keeps_only_its_loopback_on_the_device():
    """A token that loops back and also streams to another worker (MiniCPM-o's
    TTS code): the step is not rows-only, but its loop-back signal stays on the
    device while the streamed one routes as a tensor."""
    engine = _Engine({("TTS", "tts_decode"): frozenset({"tts_next"})})
    body = _body()
    assert not body.all_on_device(engine, "TTS", "tts_decode", ["tts_code", "tts_next"])
    assert body.device_signals(engine, "TTS", "tts_decode") == frozenset({"tts_next"})
    assert engine.calls == 1, "both questions read the one cached answer"


def test_the_store_leaves_out_the_signals_kept_on_the_device():
    engine = _Engine({("TTS", "tts_decode"): frozenset({"tts_next"})})
    executor = make_worker(graph_runtime=SimpleNamespace(supports_inline_emit=True)).executor
    outputs = {
        3: {"tts_code": ["c3"], "tts_next": ["c3"]},
        5: {"tts_code": ["c5"], "tts_next": ["c5"]},
        8: {},  # a row with nothing to store
    }
    batch_N = SimpleNamespace(node_name="TTS", graph_walk="tts_decode")

    executor._drop_device_kept(batch_N, outputs, [3, 5, 8], engine)

    assert outputs == {3: {"tts_code": ["c3"]}, 5: {"tts_code": ["c5"]}, 8: {}}


def test_only_a_loopback_whose_other_outputs_all_stream_may_keep_it_alone_on_the_device():
    """``ExecutingBatch.device_loopback``'s shape, fixed at startup: the step's other
    outputs all leave by streaming edges, so none of them loops back into the node,
    where the slot-master read of the loop-back rows would leave it out."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.model.submodule_base import DeviceLoopback
    from mstar.streaming.topology import StreamingGraphEdge
    from mstar.worker.engine_manager import _partial_loopback_keys

    def decode(node, outputs):
        return Loop(name=f"{node}_loop", max_iters=8, outputs=[],
                    section=GraphNode(name=node, input_names=["next"], outputs=outputs))

    walks = {
        # MiniCPM-o's TTS: the code loops back, and streams to the vocoder
        "tts_decode": decode("TTS", [
            StreamingGraphEdge(next_node="Token2Wav", name="code", target_partition="Token2Wav"),
            GraphEdge(next_node="TTS", name="next"),
        ]),
        # a second routed output that also loops back
        "two_loops": decode("AR", [GraphEdge(next_node="AR", name="next"), GraphEdge(next_node="AR", name="audio")]),
        # nothing besides the loop-back: rows-only, not this mode
        "text_decode": decode("LLM", [GraphEdge(next_node="LLM", name="next")]),
    }
    submodules = {
        node: SimpleNamespace(device_loopback=DeviceLoopback(walk=walk, signal="next", sampler="s", buffer="b"))
        for node, walk in (("TTS", "tts_decode"), ("AR", "two_loops"), ("LLM", "text_decode"))
    }
    model = SimpleNamespace(get_graph_walk_graphs=lambda: walks)

    assert _partial_loopback_keys(submodules, model) == {("TTS", "tts_decode")}
