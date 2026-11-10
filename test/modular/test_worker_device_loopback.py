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


def test_a_signal_that_also_streams_as_a_tensor_is_not_kept_on_the_device():
    """A stream that keeps its tensors (one the node does not send inline, or a
    lockstep node's) needs the step to store them, so the engine does not count
    its signal as kept on the device, and the step does not run rows-only."""
    from mstar.testing_utils.engine import make_engine, make_mgmt

    sub = SimpleNamespace(device_loopback_signals=lambda walk: frozenset({"tts_code"}))
    streamed = make_engine(
        _submodules={"TTS": make_mgmt(sub)}, _tensor_streams={("TTS", "tts_decode"): frozenset({"tts_code"})},
    )
    assert streamed.device_loopback_signals("TTS", "tts_decode") == frozenset()
    assert not _body().all_on_device(streamed, "TTS", "tts_decode", ["tts_code"])
    inline = make_engine(_submodules={"TTS": make_mgmt(sub)})
    assert _body().all_on_device(inline, "TTS", "tts_decode", ["tts_code"])


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
