"""Per-model modality validation at intake (issue #231).

``SUPPORTED_MODALITIES`` used to be one global set checked for every model, so a
request declaring a modality the loaded model has no encoder/decoder for was
accepted at intake and only failed downstream. Each model now declares its own
supported input/output modalities, and ``submit_request`` rejects the rest with
``UnsupportedModalityError`` (mapped to a 400 by the HTTP layer).
"""

import threading
from types import SimpleNamespace

import pytest

from mstar.api_server.entrypoint import APIServer, UnsupportedModalityError
from mstar.model.bagel.bagel_model import BagelModel


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
