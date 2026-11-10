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
    """A stream that keeps its tensors (not by value, or a lockstep node's)
    needs the step to store them, so its signal does not make the step
    rows-only, whatever the node keeps on the device."""
    engine = _Engine({("TTS", "tts_decode"): frozenset({"tts_code"})})
    streamed = make_worker(
        graph_runtime=SimpleNamespace(supports_inline_emit=True),
        tensor_streams={("TTS", "tts_decode"): frozenset({"tts_code"})},
    ).executor._body
    assert not streamed.all_on_device(engine, "TTS", "tts_decode", ["tts_code"])
    assert _body().all_on_device(engine, "TTS", "tts_decode", ["tts_code"])


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


SPOKEN = {"text_inputs": "new_token", "speech_ids": "new_token"}


class _SpokenEngine(_Engine):
    """A decode walk that loops ``text_inputs`` back, and a spoken one that
    also streams ``speech_ids``; ``inline`` is walk -> its inline signals."""

    def __init__(self, inline):
        super().__init__({("LLM", w): frozenset({"text_inputs"}) for w in ("decode", "decode_spoken")})
        self.inline = inline

    def inline_signals(self, node, walk):
        return self.inline.get(walk, {})


def _spoken_body(carried=True):
    return make_worker(
        graph_runtime=SimpleNamespace(supports_inline_emit=True),
        inline_carried={("LLM", "decode_spoken"): frozenset({"speech_ids"})} if carried else {},
    ).executor._body


def test_a_signal_the_inline_column_carries_needs_no_tensor():
    """A stream the step's column carries on every edge (``InlineStreams.carried``)
    makes the step rows-only beside its loop-back signal; without it, it stores."""
    engine = _SpokenEngine({"decode_spoken": SPOKEN})
    signals = ["text_inputs", "speech_ids"]
    assert _spoken_body().all_on_device(engine, "LLM", "decode_spoken", signals)
    assert not _spoken_body(carried=False).all_on_device(engine, "LLM", "decode_spoken", signals)
    # carried on its own walk only
    assert not _spoken_body().all_on_device(engine, "LLM", "decode", signals)


def test_a_mixed_step_loops_back_a_walk_whose_stream_rides_its_column():
    """A combined step of both walks keeps each walk's rows on the device when
    every walk's inline signals read one buffer, so its one column carries the
    spoken walk's stream; when the spoken walk's stream reads another buffer
    than the decode walk's token, the rows of both store."""
    batch = SimpleNamespace(
        node_name="LLM", request_walks={1: "decode", 2: "decode_spoken", 3: "decode_spoken"},
        walk_output_signals={"decode": ["text_inputs"], "decode_spoken": ["text_inputs", "speech_ids"]},
    )
    one = _SpokenEngine({"decode": {"text_inputs": "new_token"}, "decode_spoken": SPOKEN})
    assert _spoken_body()._loopback_rids(one, batch) == {1, 2, 3}
    two = _SpokenEngine({
        "decode": {"text_inputs": "new_token"},
        "decode_spoken": {"speech_ids": "speech_token", "text_inputs": "new_token"},
    })
    assert _spoken_body()._loopback_rids(two, batch) == frozenset()


def test_a_mixed_step_of_loop_back_walks_on_two_buffers_stores_them():
    """A loop-back-only walk's client emit rides the column too: when the
    step's walks read two buffers, the one column cannot carry both, so their
    rows store. The answer is cached per (node, walks)."""
    batch = SimpleNamespace(
        node_name="LLM", request_walks={1: "decode", 2: "decode_spoken"},
        walk_output_signals={"decode": ["text_inputs"], "decode_spoken": ["text_inputs"]},
    )
    engine = _SpokenEngine({"decode": {"text_inputs": "new_token"}, "decode_spoken": {"text_inputs": "alt"}})
    calls = []
    inline_signals = engine.inline_signals
    engine.inline_signals = lambda node, walk: calls.append(walk) or inline_signals(node, walk)
    body = _spoken_body(carried=False)
    assert body._loopback_rids(engine, batch) == frozenset()
    n = len(calls)
    assert body._loopback_rids(engine, batch) == frozenset()
    assert len(calls) == n, "cached per (node, walks)"
    one = _SpokenEngine({"decode": {"text_inputs": "new_token"}, "decode_spoken": {"text_inputs": "new_token"}})
    assert _spoken_body(carried=False)._loopback_rids(one, batch) == {1, 2}
