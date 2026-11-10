"""Loading the ``/v1/realtime`` turn-detection models: Silero VAD
(``server_vad``) and Smart Turn (``semantic_vad``), and refusing a file that is
not the expected export. The launcher hands the loaded models to mstar-server,
whose turn segmentation is tested in test/rust/ (test_rust_realtime.py).
"""

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from mstar.api_server.openai import realtime_vad

FRAME = realtime_vad.FRAME
_REAL_SILERO = realtime_vad.silero_weights() is not None and importlib.util.find_spec("onnxruntime") is not None


def test_missing_onnxruntime_disables_server_vad_with_a_warning(monkeypatch, tmp_path, caplog):
    weights = tmp_path / "silero_vad.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(weights))
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    realtime_vad.silero_factory.cache_clear()
    try:
        assert realtime_vad.silero_factory() is None
    finally:
        realtime_vad.silero_factory.cache_clear()
    assert "needs onnxruntime" in caplog.text


class _Arg:
    def __init__(self, name, type_):
        self.name, self.type = name, type_


_F, _I = "tensor(float)", "tensor(int64)"
_SILERO_IN = [("input", _F), ("state", _F), ("sr", _I)]


class _FakeSession:
    """An onnxruntime session with given inputs; ``run`` answers like Silero,
    or with the given output shapes, and refuses inputs it does not have."""

    def __init__(self, inputs, out_shapes=((1, 1), (2, 1, 128))):
        self.inputs, self.out_shapes = [_Arg(*a) for a in inputs], out_shapes

    def get_inputs(self):
        return self.inputs

    def run(self, _, feed):
        if set(feed) != {a.name for a in self.inputs}:
            raise ValueError(f"invalid input names {sorted(feed)}")
        return [np.zeros(s, dtype=np.float32) for s in self.out_shapes]


def _fake_onnxruntime(monkeypatch, tmp_path, session):
    weights = tmp_path / "silero_vad.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(weights))
    ort = type(sys)("onnxruntime")
    ort.SessionOptions = type("SessionOptions", (), {})
    ort.ExecutionMode = type("ExecutionMode", (), {"ORT_SEQUENTIAL": 0})
    ort.InferenceSession = lambda *a, **k: session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    realtime_vad.silero_factory.cache_clear()


@pytest.fixture
def _uncached_silero():
    yield
    realtime_vad.silero_factory.cache_clear()


@pytest.mark.parametrize("session,needle", [
    # the silero-vad package's silero_vad_half.onnx and silero_vad_openvino_16k.onnx
    (_FakeSession([("input", _F), ("state", _F)]), "its inputs are"),
    # its silero_vad_16k_sequence.onnx
    (_FakeSession([("input", _F), ("h", _F), ("c", _F)]), "its inputs are"),
    (_FakeSession([("input", _F), ("state", _F), ("sr", _F)]), "its inputs are"),
    (_FakeSession(_SILERO_IN, out_shapes=((1,), (2, 1, 128))), "its outputs have shapes"),
    (_FakeSession(_SILERO_IN, out_shapes=((1, 1),)), "its outputs have shapes"),
])
def test_a_model_with_other_inputs_or_outputs_disables_server_vad(
        monkeypatch, tmp_path, caplog, _uncached_silero, session, needle):
    _fake_onnxruntime(monkeypatch, tmp_path, session)
    assert realtime_vad.silero_factory() is None
    assert "is not the Silero VAD export" in caplog.text and needle in caplog.text


def test_the_silero_inputs_in_any_order_are_accepted(monkeypatch, tmp_path, _uncached_silero):
    """silero_vad_op18_ifless.onnx lists sr before state; inputs go by name."""
    _fake_onnxruntime(monkeypatch, tmp_path, _FakeSession([("input", _F), ("sr", _I), ("state", _F)]))
    make = realtime_vad.silero_factory()
    assert make is not None and make()(np.zeros(FRAME, dtype=np.float32)) == 0.0


def _tiny_onnx(path, inputs):
    """A model with the given (name, element type) inputs and Silero's
    outputs: the mean of ``input`` and ``state`` unchanged."""
    from onnx import TensorProto, helper, save

    types = {"float": TensorProto.FLOAT, "int64": TensorProto.INT64}
    shapes = {"input": [1, None], "state": [2, None, 128], "sr": []}
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["output"], axes=[1], keepdims=1),
         helper.make_node("Identity", ["state"], ["stateN"])],
        "tiny",
        [helper.make_tensor_value_info(n, types[t], shapes[n]) for n, t in inputs],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1]),
         helper.make_tensor_value_info("stateN", TensorProto.FLOAT, [2, None, 128])])
    # an IR version every onnxruntime release since 1.10 reads; onnx's default runs ahead of it
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
                    reason="needs onnx and onnxruntime")
def test_a_real_onnx_file_without_sr_is_refused(monkeypatch, tmp_path, _uncached_silero):
    """Through onnxruntime itself: the same graph is accepted with Silero's
    three inputs and refused without ``sr``, as silero_vad_half.onnx is."""
    good, bad = tmp_path / "good.onnx", tmp_path / "bad.onnx"
    _tiny_onnx(good, [("input", "float"), ("state", "float"), ("sr", "int64")])
    _tiny_onnx(bad, [("input", "float"), ("state", "float")])
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(good))
    realtime_vad.silero_factory.cache_clear()
    make = realtime_vad.silero_factory()
    assert make is not None and make()(np.zeros(FRAME, dtype=np.float32)) == 0.0
    monkeypatch.setenv("MSTAR_SILERO_VAD_PATH", str(bad))
    realtime_vad.silero_factory.cache_clear()
    assert realtime_vad.silero_factory() is None


@pytest.mark.skipif(not _REAL_SILERO, reason="no Silero VAD weights or onnxruntime")
def test_silero_streams_share_one_model_but_not_its_state():
    """Streams interleaved through the one ONNX session score each frame as
    they would alone, and reset starts a stream over."""
    realtime_vad.silero_factory.cache_clear()
    make = realtime_vad.silero_factory()
    with wave.open(str(Path(__file__).parents[1] / "qwen3-omni" / "audio.wav"), "rb") as wf:
        clip = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")[::3][: 40 * FRAME] / 32768.0
    frames = clip.astype(np.float32).reshape(-1, FRAME)
    alone = [p for v in [make()] for p in (v(f) for f in frames)]
    a, b = make(), make()
    together = []
    for f, g in zip(frames, frames[::-1], strict=True):
        together.append(a(f))
        b(g)
    assert together == alone and max(alone) > 0.5
    a.reset()
    assert [a(f) for f in frames] == alone


# ---------------------------------------------------------------- semantic_vad


def test_missing_smart_turn_is_a_warning_and_no_model(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(tmp_path / "missing.onnx"))
    realtime_vad.smart_turn.cache_clear()
    try:
        with caplog.at_level("WARNING", logger=realtime_vad.__name__):
            assert realtime_vad.smart_turn() is None
        assert realtime_vad.smart_turn_problem() == "no Smart Turn weights on this server"
    finally:
        realtime_vad.smart_turn.cache_clear()
    assert "semantic_vad is unavailable" in caplog.text


def test_smart_turn_without_onnxruntime_names_the_package(monkeypatch, tmp_path, _uncached_smart_turn):
    weights = tmp_path / "smart-turn-v3.2-cpu.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(weights))
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    realtime_vad.smart_turn.cache_clear()
    assert realtime_vad.smart_turn() is None
    assert realtime_vad.smart_turn_problem() == "onnxruntime or transformers is not installed"


def _fake_smart_turn_runtime(monkeypatch, tmp_path, session):
    weights = tmp_path / "smart-turn-v3.2-cpu.onnx"
    weights.write_bytes(b"onnx")
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(weights))
    ort = type(sys)("onnxruntime")
    ort.SessionOptions = type("SessionOptions", (), {})
    ort.InferenceSession = lambda *a, **k: session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    realtime_vad.smart_turn.cache_clear()


@pytest.fixture
def _uncached_smart_turn():
    yield
    realtime_vad.smart_turn.cache_clear()


_needs_transformers = pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                                         reason="needs transformers (Whisper's feature extractor)")


@_needs_transformers
@pytest.mark.parametrize("session,needle", [
    (_FakeSession(_SILERO_IN), "its inputs are"),     # MSTAR_SMART_TURN_PATH pointed at Silero
    (_FakeSession([("input_features", _I)], out_shapes=((1, 1),)), "its inputs are"),
    (_FakeSession([("input_features", _F)], out_shapes=((1,),)), "its outputs have shapes"),
    (_FakeSession([("input_features", _F)], out_shapes=()), "its outputs have shapes"),
])
def test_a_smart_turn_file_with_other_inputs_or_outputs_is_no_model(
        monkeypatch, tmp_path, caplog, _uncached_smart_turn, session, needle):
    _fake_smart_turn_runtime(monkeypatch, tmp_path, session)
    with caplog.at_level("WARNING", logger=realtime_vad.__name__):
        assert realtime_vad.smart_turn() is None
    assert "semantic_vad is unavailable" in caplog.text and needle in caplog.text
    assert realtime_vad.smart_turn_problem() == "the Smart Turn file is not the expected model"


@_needs_transformers
def test_the_smart_turn_signature_is_accepted(monkeypatch, tmp_path, _uncached_smart_turn):
    _fake_smart_turn_runtime(monkeypatch, tmp_path, _FakeSession([("input_features", _F)], out_shapes=((1, 1),)))
    model = realtime_vad.smart_turn()
    assert model is not None and model(np.zeros(16000, dtype=np.float32)) == 0.0


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in ("onnx", "onnxruntime", "transformers")),
                    reason="needs onnx, onnxruntime and transformers")
def test_a_real_onnx_file_with_silero_inputs_is_not_smart_turn(monkeypatch, tmp_path, _uncached_smart_turn):
    path = tmp_path / "silero_shaped.onnx"
    _tiny_onnx(path, [("input", "float"), ("state", "float"), ("sr", "int64")])
    monkeypatch.setenv("MSTAR_SMART_TURN_PATH", str(path))
    realtime_vad.smart_turn.cache_clear()
    assert realtime_vad.smart_turn() is None
