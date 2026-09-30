"""Per-model modality validation at intake (issue #231).

``SUPPORTED_MODALITIES`` used to be one global set checked for every model, so a
request declaring a modality the loaded model has no encoder/decoder for was
accepted at intake and only failed downstream. Each model now declares its own
supported input/output modalities, and ``submit_request`` rejects the rest with
``UnsupportedModalityError`` (mapped to a 400 by the HTTP layer).
"""

import threading
from importlib import import_module
from types import SimpleNamespace

import pytest

from mstar.api_server.entrypoint import (
    SUPPORTED_MODALITIES,
    APIServer,
    UnsupportedModalityError,
)
from mstar.model.base import Model
from mstar.model.registry import MODEL_REGISTRY


def _model_cls(name):
    module, cls = MODEL_REGISTRY[name]
    try:
        mod = import_module(module)
    except ModuleNotFoundError as e:
        # CI's CPU job installs only .[dev], so a model whose third-party deps
        # are missing is skipped there; a broken mstar import still fails
        if (e.name or "").split(".")[0] == "mstar":
            raise
        pytest.skip(f"{name} needs {e.name}")
    return getattr(mod, cls)


class _TextImage:
    """BAGEL's declaration, for the intake tests: no model import needed."""

    SUPPORTED_INPUT_MODALITIES = frozenset({"text", "image"})
    SUPPORTED_OUTPUT_MODALITIES = frozenset({"text", "image"})
    unsupported_modalities = Model.unsupported_modalities


def _server(model, model_name="bagel"):
    s = APIServer.__new__(APIServer)
    s.model = model
    s.model_name = model_name
    s.request_lock = threading.Lock()
    s.pending_requests = {}
    s.fatal_error = None
    s.preprocess_worker = SimpleNamespace(new_request=lambda *a, **k: None)
    return s


def test_bagel_declares_text_and_image_only():
    # Acceptance: BAGEL rejects audio/video input (it has no such encoder).
    bagel = _model_cls("bagel")
    assert bagel.SUPPORTED_INPUT_MODALITIES == _TextImage.SUPPORTED_INPUT_MODALITIES
    assert bagel.SUPPORTED_OUTPUT_MODALITIES == _TextImage.SUPPORTED_OUTPUT_MODALITIES


def test_unsupported_modalities_flags_input_and_output():
    model = _TextImage()
    assert model.unsupported_modalities(["text", "audio"], ["image", "audio"]) == [
        ("audio", "input"), ("audio", "output"),
    ]
    assert model.unsupported_modalities(["text", "image"], ["image"]) == []


def test_submit_request_rejects_unsupported_modality_for_the_model():
    server = _server(_TextImage())
    with pytest.raises(UnsupportedModalityError, match="audio"):
        server.submit_request(input_modalities=["audio"], output_modalities=["text"])
    assert server.pending_requests == {}  # nothing dispatched downstream


def test_submit_request_checks_the_uploaded_files_too():
    # the data worker loads every file_paths key, so a declared text-only
    # request carrying a .wav would still feed BAGEL audio
    server = _server(_TextImage())
    with pytest.raises(UnsupportedModalityError, match="'audio' \\(input\\)"):
        server.submit_request(
            text="hi", file_paths={"audio": ["/tmp/x.wav"]},
            input_modalities=["text"], output_modalities=["text"],
        )
    assert server.pending_requests == {}


def test_rejection_is_a_400_that_names_what_is_supported():
    server = _server(_TextImage())
    with pytest.raises(UnsupportedModalityError) as e:
        server.submit_request(input_modalities=["audio"], output_modalities=["text"])
    assert e.value.status_code == 400
    assert "it takes image, text in and image, text out" in str(e.value)


def test_submit_request_accepts_a_supported_combination():
    server = _server(_TextImage())
    rid = server.submit_request(
        text="hi", input_modalities=["text", "image"], output_modalities=["image"],
    )
    assert isinstance(rid, str) and rid in server.pending_requests


def test_falls_back_to_the_global_universe_without_a_model():
    server = _server(None, model_name="dummy")
    # A modality outside the universe is still rejected...
    with pytest.raises(UnsupportedModalityError, match="hologram"):
        server.submit_request(input_modalities=["hologram"], output_modalities=["text"])
    # ...but a known one goes through unchanged (no model to narrow it).
    rid = server.submit_request(input_modalities=["audio"], output_modalities=["text"])
    assert rid in server.pending_requests


class _SpeechOnly:
    """A TTS model's declaration: text in, audio out."""

    SUPPORTED_INPUT_MODALITIES = frozenset({"text"})
    SUPPORTED_OUTPUT_MODALITIES = frozenset({"audio"})
    DEFAULT_OUTPUT_MODALITIES = None
    unsupported_modalities = Model.unsupported_modalities
    default_output_modalities = Model.default_output_modalities


def test_an_unnamed_output_is_the_model_default():
    # a TTS model called without output_modalities speaks instead of a 400
    server = _server(_SpeechOnly(), model_name="orpheus")
    rid = server.submit_request(text="hi", input_modalities=["text"], output_modalities=[])
    assert server.pending_requests[rid].output_modalities == ["audio"]
    # and with no model to ask, text
    server = _server(None, model_name="dummy")
    rid = server.submit_request(text="hi", input_modalities=["text"], output_modalities=[])
    assert server.pending_requests[rid].output_modalities == ["text"]


# ── per-model declarations ──────────────────────────────────────────────────


# Every request shape the front ends actually emit, per registry name. These
# are asserted rather than only the rejections: an over-narrow declaration
# turns a working request into a 400, which is a worse failure than the
# permissiveness it replaced. Sources: mstar/api_server/openai/adapters.py and
# mstar/integrations/dynamo/bridges.py.
ACCEPTED = [
    ("bagel", ["text"], ["text"]),
    ("bagel", ["image", "text"], ["text"]),
    ("bagel", ["text"], ["image"]),
    ("bagel", ["image", "text"], ["image"]),
    ("qwen3_omni", ["text"], ["text", "audio"]),
    ("qwen3_omni", ["image", "text"], ["text"]),
    ("qwen3_omni", ["audio", "text"], ["text"]),
    ("qwen3_omni", ["video", "text"], ["text"]),
    # benchmark/base.py asks for audio alone; text still comes with it
    ("qwen3_omni", ["audio", "text"], ["audio"]),
    ("qwen3_omni", ["image", "text"], ["audio"]),
    ("orpheus", ["text"], ["audio"]),
    ("qwen3_tts", ["text"], ["audio"]),
    ("omnivoice", ["text"], ["audio"]),
    # speech route with ref_audio: the clip to clone
    ("omnivoice", ["text", "audio"], ["audio"]),
    ("cosmos3", ["text"], ["image"]),
    ("cosmos3", ["text"], ["video"]),
    ("cosmos3", ["image", "text"], ["video"]),
    ("cosmos3", ["video", "text"], ["video"]),
    ("cosmos3_droid", ["image", "text"], ["action"]),
    ("wan22", ["text"], ["video"]),
    ("wan22", ["image", "text"], ["video"]),
    ("vjepa2", ["video"], ["video"]),
    ("vjepa2_ac", ["video"], ["video"]),
    # benchmark/request.py appends text to every non-text input list
    ("vjepa2_ac", ["video", "text"], ["video"]),
    # test/vjepa2/video_request_mpc.sh
    ("vjepa2_ac", ["video"], ["scalar", "tensor", "video"]),
    ("whisper_large", ["audio"], ["text"]),
    # The chat adapter derives in_mods from the parts as written, so a text
    # prompt can ride along with the attachment. Whisper ignores it
    # (process_prompt builds the decoder prompt from language/task kwargs),
    # so it must be tolerated at intake, not rejected.
    ("whisper_large", ["audio", "text"], ["text"]),
    ("higgs_audio", ["audio"], ["text"]),
    ("higgs_audio", ["audio", "text"], ["text"]),
    ("pi05", ["image", "text"], ["action"]),
]

# Combinations each model has no encoder/decoder for. Only rejections that
# follow from the declared sets -- "audio is required" is not expressible here
# (higgs_audio takes ["text"] at intake and fails in process_prompt).
REJECTED = [
    ("bagel", ["audio"], ["text"]),
    ("bagel", ["video"], ["text"]),
    ("bagel", ["text"], ["video"]),
    ("whisper_large", ["image"], ["text"]),
    ("whisper_large", ["audio"], ["audio"]),
    ("higgs_audio", ["audio"], ["audio"]),
    ("orpheus", ["audio"], ["audio"]),
    ("qwen3_tts", ["text"], ["video"]),
    ("omnivoice", ["image", "text"], ["audio"]),
    ("omnivoice", ["text"], ["text"]),
    ("vjepa2", ["image"], ["video"]),
    # only the AC predictor's MPC walk emits these
    ("vjepa2", ["video"], ["scalar"]),
    ("vjepa2", ["video"], ["tensor"]),
    ("wan22", ["text"], ["audio"]),
    ("pi05", ["audio"], ["action"]),
    ("pi05", ["image", "text"], ["text"]),
    ("qwen3_omni", ["text"], ["video"]),
    ("cosmos3", ["audio"], ["video"]),
]


@pytest.mark.parametrize("name,in_mods,out_mods", ACCEPTED)
def test_front_end_request_shapes_are_accepted(name, in_mods, out_mods):
    model = _model_cls(name).__new__(_model_cls(name))
    assert model.unsupported_modalities(in_mods, out_mods) == []


@pytest.mark.parametrize("name,in_mods,out_mods", REJECTED)
def test_unsupported_combinations_are_rejected(name, in_mods, out_mods):
    model = _model_cls(name).__new__(_model_cls(name))
    assert model.unsupported_modalities(in_mods, out_mods) != []


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_registered_model_narrows_the_defaults(name):
    """A model that doesn't declare inherits the full universe, so the check is
    a no-op for it -- which is the bug #231 describes. New models must declare."""
    cls = _model_cls(name)
    assert cls.SUPPORTED_INPUT_MODALITIES != Model.SUPPORTED_INPUT_MODALITIES, (
        f"{name} does not declare SUPPORTED_INPUT_MODALITIES"
    )
    assert cls.SUPPORTED_OUTPUT_MODALITIES != Model.SUPPORTED_OUTPUT_MODALITIES, (
        f"{name} does not declare SUPPORTED_OUTPUT_MODALITIES"
    )


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_declarations_stay_inside_the_known_universe(name):
    """Catches typos: a modality no loader or emitter knows about can never be
    satisfied, so it would silently reject every request that asks for it."""
    cls = _model_cls(name)
    assert cls.SUPPORTED_INPUT_MODALITIES <= SUPPORTED_MODALITIES
    assert cls.SUPPORTED_OUTPUT_MODALITIES <= SUPPORTED_MODALITIES


def test_vjepa2_fails_a_request_with_no_video():
    # text passes intake (the benchmark sends it with the video); alone it must
    # fail in process_prompt, not hang the walk
    cls = _model_cls("vjepa2")
    with pytest.raises(ValueError, match="video"):
        cls.__new__(cls).process_prompt("describe", ["text"], ["video"], tensors={})


def test_cosmos3_audio_out_means_the_sound_walk():
    # audio is the sound band of a video, so asking for it turns sound on, and
    # asking for it without a video fails instead of returning an image
    cls = _model_cls("cosmos3")
    model = cls(model_path_hf="unused", skip_weight_loading=True)
    params = model._resolve_gen_params({"num_frames": 17}, ["text"], ["video", "audio"])
    assert params["generate_sound"] is True
    for mk, out in (
        ({}, ["audio"]),
        # multi-frame but no video out: the walk would be image_gen
        ({"num_frames": 17}, ["audio"]),
        ({"num_frames": 17}, ["image", "audio"]),
        ({"num_frames": 17, "generate_sound": True}, ["image"]),
    ):
        with pytest.raises(ValueError, match="sound"):
            model._resolve_gen_params(mk, ["text"], out)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_default_output_is_one_the_model_supports(name):
    """A model with several outputs and no text must name its default."""
    cls = _model_cls(name)
    model = cls.__new__(cls)
    assert set(model.default_output_modalities()) <= cls.SUPPORTED_OUTPUT_MODALITIES
