"""Unit tests for the OpenAI request adapters (pydantic + numpy; no fastapi)."""

import base64
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("numpy")

from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.protocol import (  # noqa: E402
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
    req = SpeechRequest(model="orpheus", input="hi", voice="tara", temperature=0.7, top_p=0.5, seed=11)
    sa = adapters.OrpheusAdapter().speech_to_request(req, tmp_path)
    assert sa.output_modalities == ["audio"]
    assert sa.model_kwargs == {"voice": "tara", "temperature": 0.7, "top_p": 0.5, "seed": 11}


def test_qwen3_speech_maps_talker_sampling(tmp_path):
    # talker_top_p was previously missing; the shared sampling helper adds it.
    req = SpeechRequest(model="qwen3_omni", input="hi", voice="Ethan", temperature=0.5, top_p=0.7, seed=123)
    sa = adapters.Qwen3OmniAdapter().speech_to_request(req, tmp_path)
    assert sa.output_modalities == ["text", "audio"]
    assert sa.model_kwargs["talker_temperature"] == 0.5
    assert sa.model_kwargs["talker_top_p"] == 0.7
    assert sa.model_kwargs["seed"] == 123
    assert sa.model_kwargs["voice"] == "Ethan"


def test_chat_and_image_honor_seed(tmp_path):
    chat = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "x"}], seed=7)
    assert adapters.BagelAdapter().chat_to_request(chat, tmp_path).model_kwargs["seed"] == 7
    img = ImageGenerationRequest(model="bagel", prompt="a cat", seed=9)
    assert adapters.BagelAdapter().image_to_request(img, tmp_path).model_kwargs["seed"] == 9


def test_extra_body_non_standard_knobs_pass_through(tmp_path):
    # top_k / repetition_penalty aren't OpenAI fields → flow via extra_body verbatim.
    req = ChatCompletionRequest(
        model="qwen3_omni", messages=[{"role": "user", "content": "x"}],
        talker_top_k=50, talker_repetition_penalty=1.05,
    )
    mk = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path).model_kwargs
    assert mk["talker_top_k"] == 50 and mk["talker_repetition_penalty"] == 1.05


def test_bagel_image(tmp_path):
    req = ImageGenerationRequest(model="bagel", prompt="a cat")
    sa = adapters.BagelAdapter().image_to_request(req, tmp_path)
    assert sa.text == "a cat" and sa.output_modalities == ["image"]


def test_extra_body_passthrough(tmp_path):
    # Unknown fields (OpenAI client extra_body) flow through as model_kwargs.
    req = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "x"}], think_mode=True)
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert sa.model_kwargs.get("think_mode") is True


def test_qwen3_chat_audio_url_input(tmp_path):
    raw = b"RIFFfakewavbytes"
    url = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    req = ChatCompletionRequest(
        model="qwen3_omni",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "audio_url", "audio_url": {"url": url}},
        ]}],
    )
    sa = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path)
    assert "audio" in sa.file_paths
    assert Path(sa.file_paths["audio"][0]).read_bytes() == raw
    assert "audio" in sa.input_modalities


def test_bagel_image_edit(tmp_path):
    image_path = str(tmp_path / "in.png")
    sa = adapters.BagelAdapter().image_edit_to_request("make it neon", image_path, {"cfg_img_scale": 2.0})
    assert sa.text == "make it neon"
    assert sa.file_paths == {"image": [image_path]}
    assert sa.input_modalities == ["image", "text"]
    assert sa.output_modalities == ["image"]
    assert sa.model_kwargs == {"cfg_img_scale": 2.0}


def test_registry():
    assert {"bagel", "qwen3_omni", "orpheus"} <= set(adapters.ADAPTER_REGISTRY)
    assert adapters.get_adapter("pi05") is None


def test_kimi_chat_preserves_message_structure_and_tool_fields(tmp_path):
    req = ChatCompletionRequest(
        model="kimi_k2_7",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "let me check the weather tool",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F and sunny"},
        ],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        tool_choice="auto",
        chat_template_kwargs={"x": 1},
        temperature=0.7,
        foo=1,
    )
    sa = adapters.KimiAdapter().chat_to_request(req, tmp_path)

    assert sa.text is None
    assert sa.output_modalities == ["text"]
    msgs = sa.model_kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["reasoning_content"] == "let me check the weather tool"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert msgs[3]["tool_call_id"] == "call_1"
    assert sa.model_kwargs["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
    assert sa.model_kwargs["tool_choice"] == "auto"
    assert sa.model_kwargs["chat_template_kwargs"] == {"x": 1}
    assert sa.model_kwargs["foo"] == 1
    assert sa.model_kwargs["temperature"] == 0.7


def test_kimi_chat_rejects_image_parts(tmp_path):
    req = ChatCompletionRequest(
        model="kimi_k2_7",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]}],
    )
    with pytest.raises(ValueError, match="text-only"):
        adapters.KimiAdapter().chat_to_request(req, tmp_path)


def test_kimi_adapter_registered_chat_only():
    adapter = adapters.get_adapter("kimi_k2_7")
    assert adapter is not None
    assert adapter.supports_chat is True
    assert adapter.supports_speech is False
    assert adapter.supports_images is False
    assert adapters.get_adapter("bagel").supports_chat
