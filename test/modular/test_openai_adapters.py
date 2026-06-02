"""Unit tests for the OpenAI request adapters (pydantic + numpy; no fastapi)."""

import base64
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("numpy")

from mminf.api_server.openai import adapters  # noqa: E402
from mminf.api_server.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ImageGenerationRequest,
    SpeechRequest,
)


def test_bagel_chat_maps_max_tokens(tmp_path):
    req = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "hi"}], max_tokens=32)
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert sa.text == "hi"
    assert sa.output_modalities == ["text"]
    assert sa.model_kwargs["max_output_tokens"] == 32


def test_bagel_chat_decodes_image_input(tmp_path):
    raw = b"\x89PNG-data"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    req = ChatCompletionRequest(
        model="bagel",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    )
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert "image" in sa.file_paths
    assert Path(sa.file_paths["image"][0]).read_bytes() == raw
    assert "image" in sa.input_modalities and "text" in sa.input_modalities


def test_qwen3_chat_audio_and_sampling(tmp_path):
    req = ChatCompletionRequest(
        model="qwen3_omni",
        messages=[{"role": "user", "content": "say hi"}],
        modalities=["text", "audio"],
        audio={"voice": "Ethan"},
        temperature=0.3,
        top_p=0.8,
    )
    sa = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path)
    assert sa.output_modalities == ["text", "audio"]
    assert sa.model_kwargs["thinker_temperature"] == 0.3
    assert sa.model_kwargs["thinker_top_p"] == 0.8
    assert sa.model_kwargs["voice"] == "Ethan"


def test_orpheus_speech_maps_voice_and_sampling(tmp_path):
    req = SpeechRequest(model="orpheus", input="hi", voice="tara", temperature=0.7, top_p=0.5)
    sa = adapters.OrpheusAdapter().speech_to_request(req, tmp_path)
    assert sa.output_modalities == ["audio"]
    assert sa.model_kwargs == {"voice": "tara", "temperature": 0.7, "top_p": 0.5}


def test_bagel_image(tmp_path):
    req = ImageGenerationRequest(model="bagel", prompt="a cat")
    sa = adapters.BagelAdapter().image_to_request(req, tmp_path)
    assert sa.text == "a cat" and sa.output_modalities == ["image"]


def test_extra_body_passthrough(tmp_path):
    # Unknown fields (OpenAI client extra_body) flow through as model_kwargs.
    req = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "x"}], think_mode=True)
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert sa.model_kwargs.get("think_mode") is True


def test_registry():
    assert {"bagel", "qwen3_omni", "orpheus"} <= set(adapters.ADAPTER_REGISTRY)
    assert adapters.get_adapter("pi05") is None
    assert adapters.get_adapter("bagel").supports_chat
