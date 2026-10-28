"""Per-model modality declarations and the intake check (#231).

Each model declares the modalities it takes and produces, and
``APIServer.submit_request`` refuses any other with ``UnsupportedModalityError``,
which every front end answers with a 400. The tables below hold the shapes the
front ends send each model (they must pass) and shapes it cannot serve (they
must not). A request that names no output gets the model's default.
"""

import asyncio
import base64
import io
import json
import time
import wave
from importlib import import_module
from types import SimpleNamespace

import pytest
from api_server_helpers import make_test_api_server
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import SUPPORTED_MODALITIES, UnsupportedModalityError
from mstar.api_server.request_types import ResultChunk
from mstar.api_server.rust_frontend import RustFrontendBridge
from mstar.model.base import Model
from mstar.model.registry import MODEL_REGISTRY


def _model_cls(name):
    module, cls = MODEL_REGISTRY[name]
    try:
        mod = import_module(module)
    except ModuleNotFoundError as e:
        # a model whose third-party deps are missing here is skipped; a broken
        # mstar import still fails
        if (e.name or "").split(".")[0] == "mstar":
            raise
        pytest.skip(f"{name} needs {e.name}")
    return getattr(mod, cls)


def _declared(name):
    """``name``'s class without its __init__: the class-level declarations."""
    cls = _model_cls(name)
    return cls.__new__(cls)


def _no_preprocess(*args, **kwargs):
    raise AssertionError("a refused request reached the data worker")


def _server(model, model_name, tmp_path=None, **fields):
    if tmp_path is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        fields["upload_dir"] = tmp_path
    return make_test_api_server(
        model=model, model_name=model_name,
        preprocess_worker=SimpleNamespace(new_request=_no_preprocess), **fields,
    )


def _wav(seconds=0.5, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(seconds * rate))
    return buf.getvalue()


# ── the check ───────────────────────────────────────────────────────────────


def test_unsupported_modalities_names_each_direction():
    bagel = _declared("bagel")
    assert bagel.unsupported_modalities(["text", "audio"], ["image", "audio"]) == [
        ("audio", "input"), ("audio", "output"),
    ]
    assert bagel.unsupported_modalities(["image", "text"], ["image"]) == []


def test_submit_request_refuses_what_the_model_lacks_before_registering_it():
    server = _server(_declared("bagel"), "bagel")
    with pytest.raises(UnsupportedModalityError) as e:
        server.submit_request(text="hi", input_modalities=["audio", "text"], output_modalities=["text"])
    assert e.value.status_code == 400
    assert e.value.detail == str(e.value) == (
        "bagel does not support 'audio' input; it takes image, text and produces image, text"
    )
    assert server.pending_requests == {} and server._live == {}


def test_submit_request_counts_the_uploads_too():
    # the data worker loads every file_paths key, whatever input_modalities says
    server = _server(_declared("bagel"), "bagel")
    with pytest.raises(UnsupportedModalityError, match="'audio' input"):
        server.submit_request(
            text="hi", file_paths={"audio": ["/uploads/x.wav"]},
            input_modalities=["text"], output_modalities=["text"],
        )
    assert server.pending_requests == {}


def test_submit_request_takes_what_the_model_declares():
    calls = []
    server = make_test_api_server(
        model=_declared("bagel"), model_name="bagel",
        preprocess_worker=SimpleNamespace(new_request=calls.append),
    )
    rid = server.submit_request(
        text="hi", file_paths={"image": ["/uploads/x.png"]},
        input_modalities=["image", "text"], output_modalities=["image"],
    )
    assert rid in server._live and len(calls) == 1


def test_without_a_model_only_the_known_modalities_are_checked():
    calls = []
    server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=calls.append))
    with pytest.raises(UnsupportedModalityError, match="'hologram'"):
        server.submit_request(input_modalities=["hologram"], output_modalities=["text"])
    server.submit_request(text="hi", input_modalities=["audio", "text"], output_modalities=["text"])
    assert len(calls) == 1


# ── per-model declarations ──────────────────────────────────────────────────

# Shapes the front ends send (openai/adapters.py, the Rust adapters, the SDK,
# the CLI hints, configs' warmup_requests, benchmark/): an over-narrow
# declaration turns a working request into a 400.
ACCEPTED = [
    ("bagel", ["text"], ["text"]),
    ("bagel", ["image", "text"], ["text"]),
    ("bagel", ["text"], ["image"]),
    ("bagel", ["image", "text"], ["image"]),
    # the transcription route keeps a text part next to the clip; the SDK's
    # transcribe drops it when the text is empty
    ("canary", ["audio", "text"], ["text"]),
    ("canary", ["audio"], ["text"]),
    ("canary_qwen", ["audio", "text"], ["text"]),
    ("canary_qwen", ["audio"], ["text"]),
    ("chatterbox", ["text"], ["audio"]),
    ("chatterbox", ["text", "audio"], ["audio"]),
    ("chatterbox_turbo", ["text", "audio"], ["audio"]),
    ("command_a_plus", ["text"], ["text"]),
    ("command_a_plus", ["image", "text"], ["text"]),
    ("cosmos3", ["text"], ["image"]),
    ("cosmos3", ["text"], ["video"]),
    ("cosmos3", ["image", "text"], ["video"]),
    ("cosmos3", ["video", "text"], ["video"]),
    ("cosmos3", ["text"], ["video", "audio"]),
    ("cosmos3_droid", ["image", "text"], ["action"]),
    # Edge's reasoner answers chat in text
    ("cosmos3_edge", ["image", "text"], ["text"]),
    ("cosmos3_edge", ["video", "text"], ["text"]),
    ("cosmos3_edge", ["image", "text"], ["video"]),
    ("cosmos3_edge_droid", ["image", "text"], ["action"]),
    ("fish_s2_pro", ["text"], ["audio"]),
    ("fish_s2_pro", ["text", "audio"], ["audio"]),
    # configs/fish_s2_pro.yaml's warmup: the file comes before the text
    ("fish_s2_pro", ["audio", "text"], ["audio"]),
    ("flux2_klein", ["text"], ["image"]),
    ("flux2_klein", ["image", "text"], ["image"]),
    ("glm52", ["text"], ["text"]),
    ("higgs_audio", ["audio", "text"], ["text"]),
    ("higgs_audio", ["audio"], ["text"]),
    ("kokoro", ["text"], ["audio"]),
    ("lingbot_world_1_3b", ["image", "text"], ["video_frame"]),
    ("ltx2_5", ["text"], ["video", "audio"]),
    ("ltx2_5", ["text"], ["video"]),
    # the transcription route keeps a text part next to the clip
    ("nemotron_asr", ["audio", "text"], ["text"]),
    ("nemotron_asr", ["audio"], ["text"]),
    # benchmark/nemotron_duplex/sessions.py; with --instructions, a system prompt
    ("nemotron_duplex", ["audio"], ["text", "audio"]),
    ("nemotron_duplex", ["audio", "text"], ["text", "audio"]),
    ("nemotron_duplex", ["audio"], ["audio"]),
    # a live call opens with no media, with or without a prompt
    # (test/nemotron_duplex/live_ws_request.py)
    ("nemotron_duplex", [], ["text", "audio"]),
    ("nemotron_duplex", ["text"], ["text", "audio"]),
    ("omnilingual_asr_ctc_300m", ["audio"], ["text"]),
    # benchmark/request.py and the SDK's transcribe(text=...) add a text part
    ("omnilingual_asr_ctc_300m", ["audio", "text"], ["text"]),
    ("omnivoice", ["text"], ["audio"]),
    ("omnivoice", ["text", "audio"], ["audio"]),
    ("orpheus", ["text"], ["audio"]),
    ("parakeet", ["audio", "text"], ["text"]),
    ("parakeet", ["audio"], ["text"]),
    ("pi05", ["image", "text"], ["action"]),
    ("qwen3_omni", ["text"], ["text"]),
    ("qwen3_omni", ["text"], ["text", "audio"]),
    ("qwen3_omni", ["image", "text"], ["text"]),
    ("qwen3_omni", ["audio", "text"], ["text"]),
    ("qwen3_omni", ["video", "text"], ["text"]),
    ("qwen3_omni", ["audio", "text"], ["audio"]),
    # the transcription and realtime routes keep a text part, the context;
    # the CLI hint's client.transcribe drops it when empty
    ("qwen3_asr", ["audio", "text"], ["text"]),
    ("qwen3_asr", ["audio"], ["text"]),
    ("qwen3_asr_realtime", ["audio", "text"], ["text"]),
    ("qwen3_tts", ["text"], ["audio"]),
    # client.tts(reference_audio=...) on the Base checkpoint
    ("qwen3_tts_base", ["audio", "text"], ["audio"]),
    ("qwen3_5_0.8b", ["text"], ["text"]),
    ("qwen3_5_0.8b", ["image", "text"], ["text"]),
    ("qwen3_5_0.8b", ["video", "text"], ["text"]),
    ("qwen3_8_27b", ["text", "video", "image"], ["text"]),
    ("vjepa2", ["video"], ["video"]),
    # benchmark/request.py appends text to every non-text input list
    ("vjepa2", ["video", "text"], ["video"]),
    ("vjepa2_ac", ["video", "text"], ["video"]),
    # test/vjepa2/video_request_mpc.sh
    ("vjepa2_ac", ["video"], ["scalar", "tensor", "video"]),
    ("wan22", ["text"], ["video"]),
    ("wan22", ["image", "text"], ["video"]),
    ("waypoint", ["image"], ["video_frame"]),
    ("whisper_large", ["audio", "text"], ["text"]),
    ("whisper_large", ["audio"], ["text"]),
    ("whisper_large_v3_turbo", ["audio", "text"], ["text"]),
    ("whisper_large_v3_turbo", ["audio"], ["text"]),
    ("z_image_turbo", ["text"], ["image"]),
]

# Shapes the model has no encoder, decoder or walk for.
REJECTED = [
    ("bagel", ["audio", "text"], ["text"]),
    ("bagel", ["video", "text"], ["text"]),
    ("bagel", ["text"], ["video"]),
    ("canary", ["image"], ["text"]),
    ("canary", ["audio"], ["audio"]),
    ("canary_qwen", ["audio"], ["audio"]),
    ("chatterbox", ["text"], ["text"]),
    ("chatterbox", ["image", "text"], ["audio"]),
    ("command_a_plus", ["audio", "text"], ["text"]),
    ("command_a_plus", ["video", "text"], ["text"]),
    ("command_a_plus", ["text"], ["image"]),
    ("cosmos3", ["audio", "text"], ["video"]),
    ("fish_s2_pro", ["text"], ["text"]),
    ("fish_s2_pro", ["image", "text"], ["audio"]),
    ("flux2_klein", ["text"], ["text"]),
    ("flux2_klein", ["video", "text"], ["image"]),
    ("glm52", ["image", "text"], ["text"]),
    ("glm52", ["text"], ["audio"]),
    ("higgs_audio", ["audio"], ["audio"]),
    # no voice cloning
    ("kokoro", ["text", "audio"], ["audio"]),
    ("kokoro", ["text"], ["text"]),
    # raw frames only, never an encoded video
    ("lingbot_world_1_3b", ["image", "text"], ["video"]),
    ("lingbot_world_1_3b", ["video", "text"], ["video_frame"]),
    ("ltx2_5", ["image", "text"], ["video"]),
    ("ltx2_5", ["text"], ["text"]),
    ("nemotron_asr", ["audio"], ["audio"]),
    ("nemotron_duplex", ["image", "audio"], ["audio"]),
    ("nemotron_duplex", ["audio"], ["video"]),
    ("omnilingual_asr_ctc_300m", ["audio"], ["audio"]),
    ("omnilingual_asr_ctc_300m", ["image"], ["text"]),
    ("omnivoice", ["image", "text"], ["audio"]),
    ("omnivoice", ["text"], ["text"]),
    ("orpheus", ["audio", "text"], ["audio"]),
    ("orpheus", ["text"], ["text"]),
    ("parakeet", ["image"], ["text"]),
    ("parakeet", ["audio"], ["audio"]),
    ("pi05", ["audio"], ["action"]),
    ("pi05", ["image", "text"], ["text"]),
    ("qwen3_omni", ["text"], ["video"]),
    ("qwen3_omni", ["text"], ["image"]),
    ("qwen3_asr", ["image", "audio"], ["text"]),
    ("qwen3_asr", ["audio"], ["audio"]),
    ("qwen3_asr_realtime", ["video", "text"], ["text"]),
    ("qwen3_5_0.8b", ["audio", "text"], ["text"]),
    ("qwen3_5_0.8b", ["text"], ["audio"]),
    ("qwen3_tts", ["image", "text"], ["audio"]),
    ("qwen3_tts", ["text"], ["text"]),
    ("vjepa2", ["image"], ["video"]),
    # only the AC predictor's MPC walk emits these
    ("vjepa2", ["video"], ["scalar"]),
    ("vjepa2", ["video"], ["tensor"]),
    ("vjepa2_ac", ["image"], ["video"]),
    ("vjepa2_ac", ["video"], ["action"]),
    ("wan22", ["video", "text"], ["video"]),
    ("wan22", ["text"], ["audio"]),
    # raw frames only, and the checkpoint takes no prompt
    ("waypoint", ["image"], ["video"]),
    ("waypoint", ["image", "text"], ["video_frame"]),
    ("whisper_large", ["image"], ["text"]),
    ("whisper_large", ["audio"], ["audio"]),
    ("whisper_large_v3_turbo", ["image", "audio"], ["text"]),
    # the shared image adapter opens /v1/images/edits; Z-Image-Turbo has no edit path
    ("z_image_turbo", ["image", "text"], ["image"]),
]


@pytest.mark.parametrize("name,in_mods,out_mods", ACCEPTED)
def test_front_end_request_shapes_are_accepted(name, in_mods, out_mods):
    assert _declared(name).unsupported_modalities(in_mods, out_mods) == []


@pytest.mark.parametrize("name,in_mods,out_mods", REJECTED)
def test_shapes_the_model_cannot_serve_are_refused(name, in_mods, out_mods):
    assert _declared(name).unsupported_modalities(in_mods, out_mods) != []


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_registered_model_narrows_the_defaults(name):
    """A model that doesn't declare inherits every modality, so the check is a
    no-op for it. New models must declare."""
    cls = _model_cls(name)
    assert cls.SUPPORTED_INPUT_MODALITIES != Model.SUPPORTED_INPUT_MODALITIES, (
        f"{name} does not declare SUPPORTED_INPUT_MODALITIES"
    )
    assert cls.SUPPORTED_OUTPUT_MODALITIES != Model.SUPPORTED_OUTPUT_MODALITIES, (
        f"{name} does not declare SUPPORTED_OUTPUT_MODALITIES"
    )


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_declarations_stay_inside_the_known_modalities(name):
    """A typo could never be satisfied and would refuse every request naming it."""
    cls = _model_cls(name)
    assert cls.SUPPORTED_INPUT_MODALITIES <= SUPPORTED_MODALITIES - {"video_frame"}
    assert cls.SUPPORTED_OUTPUT_MODALITIES <= SUPPORTED_MODALITIES


# ── per-checkpoint narrowing ────────────────────────────────────────────────


def _qwen3_tts(tmp_path, monkeypatch, tts_model_type):
    cls = _model_cls("qwen3_tts")
    (tmp_path / "speech_tokenizer").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"tts_model_type": tts_model_type, "talker_config": {}}))
    (tmp_path / "generation_config.json").write_text("{}")
    (tmp_path / "speech_tokenizer" / "config.json").write_text("{}")
    monkeypatch.setattr(
        "mstar.model.qwen3_tts.qwen3_tts_model.AutoTokenizer.from_pretrained", lambda *a, **k: None,
    )
    return cls(model_path_hf=str(tmp_path))


@pytest.mark.parametrize("tts_model_type, clones", [("custom_voice", False), ("voice_design", False), ("base", True)])
def test_qwen3_tts_takes_a_reference_clip_only_on_base(tmp_path, monkeypatch, tts_model_type, clones):
    model = _qwen3_tts(tmp_path, monkeypatch, tts_model_type)
    assert (model.unsupported_modalities(["audio", "text"], ["audio"]) == []) is clones
    assert model.unsupported_modalities(["text"], ["audio"]) == []


_QWEN3_5_TEXT = {
    "model_type": "qwen3_5_text", "attn_output_gate": True, "eos_token_id": 0,
    "head_dim": 32, "hidden_size": 64, "intermediate_size": 128,
    "layer_types": ["linear_attention", "full_attention"],
    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16, "linear_num_key_heads": 2,
    "linear_num_value_heads": 2, "linear_value_head_dim": 16,
    "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "vocab_size": 300,
}
_QWEN3_5_VISION = {
    "model_type": "qwen3_5", "deepstack_visual_indexes": [], "depth": 1,
    "hidden_act": "gelu_pytorch_tanh", "hidden_size": 32, "in_channels": 3,
    "intermediate_size": 64, "num_heads": 2, "num_position_embeddings": 16,
    "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
}


@pytest.mark.parametrize("vision", [True, False])
def test_qwen3_5_takes_attachments_only_with_a_vision_tower(tmp_path, monkeypatch, vision):
    cls = _model_cls("qwen3_5_0.8b")
    raw = {"model_type": "qwen3_5", "text_config": _QWEN3_5_TEXT}
    if vision:
        raw["vision_config"] = _QWEN3_5_VISION
    (tmp_path / "config.json").write_text(json.dumps(raw))
    monkeypatch.setattr(
        "mstar.model.qwen3_5.qwen3_5_model.AutoTokenizer.from_pretrained",
        lambda *a, **k: SimpleNamespace(eos_token_id=None),
    )
    model = cls(model_path_hf=str(tmp_path))
    for attachment in ("image", "video"):
        assert (model.unsupported_modalities([attachment, "text"], ["text"]) == []) is vision
    assert model.unsupported_modalities(["text"], ["text"]) == []


def test_cosmos3_produces_text_only_where_the_reasoner_is_served(tmp_path):
    cls = _model_cls("cosmos3")
    from test_cosmos3_edge_model import _fake_edge_dir

    edge = str(_fake_edge_dir(tmp_path))
    assert cls(model_path_hf=edge, skip_weight_loading=True).unsupported_modalities(["text"], ["text"]) == []
    for model in (
        cls(model_path_hf=edge, skip_weight_loading=True, enable_reasoner=False),
        cls(model_path_hf="unused", skip_weight_loading=True),  # Nano: no reasoner
    ):
        assert model.unsupported_modalities(["image", "text"], ["text"]) == [("text", "output")]
        assert model.unsupported_modalities(["image", "text"], ["video"]) == []
    assert "text" in cls.SUPPORTED_OUTPUT_MODALITIES


# ── the default output ──────────────────────────────────────────────────────

# One registry key per class: what a request that names no output gets.
# Cosmos3 depends on the checkpoint (test_cosmos3_defaults_to_chat_only_with_the_reasoner).
DEFAULTS = {
    "bagel": ("text",),
    "canary": ("text",),
    "canary_qwen": ("text",),
    "chatterbox": ("audio",),
    "command_a_plus": ("text",),
    "fish_s2_pro": ("audio",),
    "flux2_klein": ("image",),
    "glm52": ("text",),
    "higgs_audio": ("text",),
    "kokoro": ("audio",),
    "lingbot_world_1_3b": ("video_frame",),
    "ltx2_5": ("video", "audio"),
    "nemotron_duplex": ("text", "audio"),
    "omnilingual_asr_ctc_300m": ("text",),
    "omnivoice": ("audio",),
    "orpheus": ("audio",),
    "parakeet": ("text",),
    "pi05": ("action",),
    "qwen3_5_0.8b": ("text",),
    "qwen3_asr": ("text",),
    "qwen3_omni": ("text",),
    "qwen3_tts": ("audio",),
    "vjepa2": ("video",),
    "vjepa2_ac": ("video",),
    "wan22": ("video",),
    "waypoint": ("video_frame",),
    "whisper_large": ("text",),
    "z_image_turbo": ("image",),
}


@pytest.mark.parametrize("name,default", sorted(DEFAULTS.items()))
def test_the_default_output_per_class(name, default):
    assert _declared(name).default_output_modalities() == default


def test_the_defaults_table_covers_every_class():
    classes = {MODEL_REGISTRY[name] for name in [*DEFAULTS, "cosmos3"]}
    assert classes == set(MODEL_REGISTRY.values())


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_registered_model_has_a_default_it_produces(name):
    """A model with several outputs and no text must name its default, or a
    request that names none is a 400."""
    model = _declared(name)
    default = model.default_output_modalities()
    assert default, f"{name} has no default output"
    assert len(set(default)) == len(default)
    assert model.unsupported_modalities([], list(default)) == []


def test_cosmos3_defaults_to_chat_only_with_the_reasoner(tmp_path):
    # Edge's reasoner answered a request naming no output on main (as text);
    # elsewhere that request was refused, and now gets an image
    cls = _model_cls("cosmos3")
    from test_cosmos3_edge_model import _fake_edge_dir

    edge = str(_fake_edge_dir(tmp_path))
    assert cls(model_path_hf=edge, skip_weight_loading=True).default_output_modalities() == ("text",)
    for model in (
        cls(model_path_hf=edge, skip_weight_loading=True, enable_reasoner=False),
        cls(model_path_hf="unused", skip_weight_loading=True),  # Nano: no reasoner
    ):
        assert model.default_output_modalities() == ("image",)
        assert model.unsupported_modalities([], ["image"]) == []


def test_submit_request_fills_in_the_models_default():
    calls = []
    server = make_test_api_server(
        model=_declared("orpheus"), model_name="orpheus",
        preprocess_worker=SimpleNamespace(new_request=calls.append),
    )
    rid = server.submit_request(text="hi", input_modalities=["text"], output_modalities=[])
    (sent,) = calls
    assert sent.output_modalities == ["audio"]
    assert server.pending_requests[server.submission_key(rid)].output_modalities == ["audio"]
    # with no model to ask, text, as before
    calls.clear()
    server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=calls.append))
    server.submit_request(text="hi", input_modalities=["text"], output_modalities=[])
    assert calls[0].output_modalities == ["text"]


def test_a_model_without_a_default_refuses_a_request_naming_no_output():
    model = _declared("ltx2_5")
    model.DEFAULT_OUTPUT_MODALITIES = None  # two outputs, neither of them text
    server = _server(model, "ltx2_5")
    with pytest.raises(UnsupportedModalityError) as e:
        server.submit_request(text="a dog on a beach", input_modalities=["text"], output_modalities=[])
    assert e.value.status_code == 400
    assert e.value.detail == "ltx2_5 has no default output; name one of audio, video in output_modalities"
    assert server.pending_requests == {}


def test_a_default_goes_through_the_same_checks():
    # Waypoint's raw frames are streaming-only, named or not
    server = _server(_declared("waypoint"), "waypoint")
    with pytest.raises(UnsupportedModalityError, match="video_frame requires streaming=True"):
        server.submit_request(
            file_paths={"image": ["/uploads/seed.png"]}, input_modalities=["image"],
            output_modalities=[], streaming=False,
        )
    assert server.pending_requests == {}


# ── front ends ──────────────────────────────────────────────────────────────


@pytest.fixture
def bagel(monkeypatch, tmp_path):
    server = _server(_declared("bagel"), "bagel", tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)
    return server


def test_generate_answers_a_refusal_with_a_400(bagel):
    response = TestClient(entrypoint.app).post(
        "/generate", data={"text": "what is said here?", "output_modalities": "text"},
        files={"files": ("clip.wav", _wav())},
    )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("bagel does not support 'audio' input")


def test_generate_ws_answers_a_refusal_in_band_with_its_status(bagel):
    data = base64.b64encode(_wav()).decode()
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({
            "request_id": "r1", "text": "what is said here?", "files": [{"name": "clip.wav", "data": data}],
        }))
        reply = json.loads(ws.receive_text())
    assert reply["request_id"] == "r1" and reply["status"] == 400
    assert reply["error"].startswith("bagel does not support 'audio' input")


def test_chat_completions_answers_a_refusal_with_a_400(bagel):
    data = base64.b64encode(_wav()).decode()
    response = TestClient(entrypoint.app).post("/v1/chat/completions", json={
        "model": "bagel",
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
            {"type": "text", "text": "what is said here?"},
        ]}],
    })
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == 400 and error["message"].startswith("bagel does not support 'audio' input")


def test_speech_answers_a_refusal_with_a_400(tmp_path, monkeypatch):
    # a reference clip sent to a Qwen3-TTS checkpoint that does not clone
    pytest.importorskip("soundfile")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    model = _qwen3_tts(ckpt, monkeypatch, "custom_voice")
    monkeypatch.setattr(entrypoint, "api_server", _server(model, "qwen3_tts", tmp_path / "uploads"))
    ref = "data:audio/wav;base64," + base64.b64encode(_wav()).decode()
    response = TestClient(entrypoint.app).post(
        "/v1/audio/speech", json={"model": "qwen3_tts", "input": "Hello there", "ref_audio": ref},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"].startswith("qwen3_tts does not support 'audio' input")


def test_image_edit_answers_a_refusal_with_a_400(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _server(_declared("z_image_turbo"), "z_image_turbo", tmp_path))
    response = TestClient(entrypoint.app).post(
        "/v1/images/edits", data={"prompt": "make it a watercolor"}, files={"image": ("in.png", b"PNG")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("z_image_turbo does not support 'image' input")


class _LivePreprocess:
    """Records what reaches the preprocess worker."""

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
        return True


def _live_server(tmp_path):
    """NemotronDuplex's declarations and its one client stream, which
    ``__init__`` would read from the topology."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    return make_test_api_server(
        model=_declared("nemotron_duplex"), model_name="nemotron_duplex", upload_dir=tmp_path,
        client_streams=frozenset({"audio_in"}), takes_stream_input=True,
        preprocess_worker=_LivePreprocess(),
    )


def test_an_append_of_a_modality_the_model_lacks_is_refused_before_it_is_queued(tmp_path):
    # the preprocess thread loads an append's files as it loads an opening's
    server = _live_server(tmp_path)
    server.submit_request(
        input_modalities=[], output_modalities=["text", "audio"], request_id="call", end_input=False,
    )
    key = server.submission_key("call")
    png, wav = tmp_path / "frame.png", tmp_path / "chunk.wav"
    png.write_bytes(b"PNG")
    wav.write_bytes(_wav())
    with pytest.raises(UnsupportedModalityError, match="^nemotron_duplex does not support 'image' input"):
        server.stream_input(key, {"image": [str(png)]})
    assert len(server.preprocess_worker.items) == 1 and server.taking_input(key)
    assert server.stream_input(key, {"audio": [str(wav)]}) is True
    assert len(server.preprocess_worker.items) == 2


def test_generate_ws_ends_a_live_call_whose_append_the_model_cannot_take(monkeypatch, tmp_path):
    server = _live_server(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)
    png = base64.b64encode(b"PNG").decode()
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text,audio"}))
        ws.send_text(json.dumps({"request_id": "call", "append": [{"name": "frame.png", "data": png}]}))
        reply = json.loads(ws.receive_text())
    assert reply["request_id"] == "call" and reply["status"] == 400
    assert reply["error"].startswith("append rejected: nemotron_duplex does not support 'image' input")
    (opening,) = server.preprocess_worker.items  # the append never reached it
    assert opening.request_id in server.preprocess_worker.aborted
    assert list(tmp_path.iterdir()) == []  # nor is its file left behind


def test_rust_bridge_relays_a_refusal_in_band_with_its_status():
    # mstar-server does no modality intake of its own: it hands the request to
    # this bridge, and turns the in-band status into the HTTP status
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = _server(_declared("bagel"), "bagel")
    bridge._keys = {}
    sent = []
    bridge._send = sent.append

    async def submit():
        bridge._submit({
            "t": "submit", "rid": "r1", "sub": 3, "text": "what is said here?",
            "file_paths": {"audio": ["/uploads/clip.wav"]},
            "input_modalities": ["audio", "text"], "output_modalities": ["text"],
        })

    asyncio.run(submit())
    (reply,) = sent
    assert (reply["t"], reply["rid"], reply["sub"], reply["modality"], reply["metadata"]) == (
        "chunk", "r1", 3, "error", {"status": 400},
    )
    assert reply["data"].decode().startswith("bagel does not support 'audio' input")
    assert bridge._keys == {}


# Models that don't read output_modalities, so main served a request naming none
# (as text); each is served with its default here, not refused for text.
SERVED_UNNAMED = [
    # (registry name, the upload it takes or None, its inputs, its default)
    ("orpheus", None, ["text"], ["audio"]),
    ("pi05", ("frame.png", b"\x89PNG"), ["image", "text"], ["action"]),
    ("vjepa2", ("clip.mp4", b"\x00\x00\x00\x18ftyp"), ["video", "text"], ["video"]),
    ("vjepa2_ac", ("clip.mp4", b"\x00\x00\x00\x18ftyp"), ["video", "text"], ["video"]),
]


class _AnsweringDataWorker:
    """Stands in for the data worker and the engine: finishes each request with
    one chunk per output it asks for."""

    def __init__(self):
        self.server = None
        self.received = []

    def new_request(self, sent):
        self.received.append(sent)
        with self.server.request_lock:
            req = self.server.pending_requests[sent.request_id]
            req.chunks += [
                ResultChunk(request_id=sent.client_request_id, modality=m, data=b"x", metadata={})
                for m in sent.output_modalities
            ]
            req.event.set()
        req.notify()


def _answering(monkeypatch, tmp_path, name):
    worker = _AnsweringDataWorker()
    server = make_test_api_server(
        model=_declared(name), model_name=name, upload_dir=tmp_path, preprocess_worker=worker,
    )
    worker.server = server
    monkeypatch.setattr(entrypoint, "api_server", server)
    return server, worker


@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "aggregate"])
@pytest.mark.parametrize("name,upload,in_mods,default", SERVED_UNNAMED)
def test_generate_serves_a_request_naming_no_output(monkeypatch, tmp_path, name, upload, in_mods, default, streaming):
    _, worker = _answering(monkeypatch, tmp_path, name)
    response = TestClient(entrypoint.app).post(
        "/generate", data={"text": "hi", "streaming": str(streaming).lower()},
        files={"files": upload} if upload else None,
    )
    assert response.status_code == 200, response.text
    (sent,) = worker.received
    assert (sent.input_modalities, sent.output_modalities) == (in_mods, default)
    if streaming:
        assert [json.loads(line)["modality"] for line in response.text.splitlines()] == default
    else:
        assert list(response.json()["outputs"]) == default


@pytest.mark.parametrize("name,upload,in_mods,default", SERVED_UNNAMED)
def test_generate_ws_serves_a_request_naming_no_output(monkeypatch, tmp_path, name, upload, in_mods, default):
    _, worker = _answering(monkeypatch, tmp_path, name)
    message = {"request_id": "r1", "text": "hi"}
    if upload:
        message["files"] = [{"name": upload[0], "data": base64.b64encode(upload[1]).decode()}]
    replies = []
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps(message))
        while not (replies and (replies[-1].get("finish") or "error" in replies[-1])):
            replies.append(json.loads(ws.receive_text()))
    assert "error" not in replies[-1], replies[-1]
    assert [r["modality"] for r in replies[:-1]] == default
    (sent,) = worker.received
    assert (sent.input_modalities, sent.output_modalities) == (in_mods, default)


def test_a_live_call_naming_no_output_opens_with_the_models_default(monkeypatch, tmp_path):
    # NemotronDuplex refuses text alone in process_prompt, which such a call asked for
    server = _live_server(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call"}))
        for _ in range(500):
            if server.preprocess_worker.items:
                break
            time.sleep(0.01)
    (opening,) = server.preprocess_worker.items
    assert (opening.input_modalities, opening.output_modalities) == ([], ["text", "audio"])
    assert opening.ends == frozenset()  # still taking input


@pytest.mark.parametrize("name,upload,in_mods,default", SERVED_UNNAMED)
def test_rust_bridge_serves_a_request_naming_no_output(monkeypatch, tmp_path, name, upload, in_mods, default):
    # mstar-server's /generate sends an empty list when the form names no output
    server, worker = _answering(monkeypatch, tmp_path, name)
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server, bridge._keys, sent = server, {}, []
    bridge._send = sent.append
    file_paths = {in_mods[0]: [f"/uploads/{upload[0]}"]} if upload else {}

    async def submit():
        bridge._submit({
            "t": "submit", "rid": "r1", "sub": 1, "text": "hi", "file_paths": file_paths,
            "input_modalities": in_mods, "output_modalities": [],
        })
        for _ in range(200):
            if sent and sent[-1]["t"] in ("done", "err"):
                return
            await asyncio.sleep(0.01)

    asyncio.run(submit())
    assert [m["t"] for m in sent] == ["chunk"] * len(default) + ["done"], sent
    assert [m["modality"] for m in sent[:-1]] == default
    assert worker.received[0].output_modalities == default
