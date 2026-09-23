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
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.base import Model
from mstar.model.registry import MODEL_REGISTRY


def _server(model, model_name="bagel"):
    s = APIServer.__new__(APIServer)
    s.model = model
    s.model_name = model_name
    s.request_lock = threading.Lock()
    s.pending_requests = {}
    s.preprocess_worker = SimpleNamespace(new_request=lambda *a, **k: None)
    return s


def test_bagel_declares_text_and_image_only():
    # Acceptance: BAGEL rejects audio/video input (it has no such encoder).
    assert BagelModel.SUPPORTED_INPUT_MODALITIES == frozenset({"text", "image"})
    assert BagelModel.SUPPORTED_OUTPUT_MODALITIES == frozenset({"text", "image"})
    assert "audio" not in BagelModel.SUPPORTED_INPUT_MODALITIES
    assert "video" not in BagelModel.SUPPORTED_INPUT_MODALITIES


def test_unsupported_modalities_flags_input_and_output():
    model = BagelModel.__new__(BagelModel)  # class attrs only, no HF download
    assert model.unsupported_modalities(["text", "audio"], ["image", "audio"]) == [
        ("audio", "input"), ("audio", "output"),
    ]
    assert model.unsupported_modalities(["text", "image"], ["image"]) == []


def test_submit_request_rejects_unsupported_modality_for_the_model():
    server = _server(BagelModel.__new__(BagelModel))
    with pytest.raises(UnsupportedModalityError, match="audio"):
        server.submit_request(input_modalities=["audio"], output_modalities=["text"])
    assert server.pending_requests == {}  # nothing dispatched downstream


def test_submit_request_accepts_a_supported_combination():
    server = _server(BagelModel.__new__(BagelModel))
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


# ── per-model declarations ──────────────────────────────────────────────────


def _model_cls(name):
    module, cls = MODEL_REGISTRY[name]
    return getattr(import_module(module), cls)


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
    ("orpheus", ["text"], ["audio"]),
    ("qwen3_tts", ["text"], ["audio"]),
    ("cosmos3", ["text"], ["image"]),
    ("cosmos3", ["text"], ["video"]),
    ("cosmos3", ["image", "text"], ["video"]),
    ("cosmos3", ["video", "text"], ["video"]),
    ("cosmos3_droid", ["image", "text"], ["action"]),
    ("wan22", ["text"], ["video"]),
    ("wan22", ["image", "text"], ["video"]),
    ("vjepa2", ["video"], ["video"]),
    ("vjepa2_ac", ["video"], ["video"]),
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
    ("vjepa2", ["text"], ["video"]),
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
