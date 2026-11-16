import json

import pytest

from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai.protocol import ChatCompletionRequest, SpeechRequest
from mstar.api_server.unicode_input import reject_lone_surrogates
from mstar.model.multimodal import PromptPart


@pytest.mark.parametrize("value", [
    "\ud800", "\udfff", {"nested": ["ok", {"text": "\ud800"}]},
    {"\ud800": "key"}, SpeechRequest(input="hi", instructions={"text": "\ud800"}),
    ChatCompletionRequest(messages=[{"role": "user", "content": "\ud800"}]),
])
def test_rejects_lone_surrogates_in_decoded_client_text(value):
    with pytest.raises(ValueError, match="text is not valid UTF-8.*lone surrogate"):
        reject_lone_surrogates(value)


def test_accepts_a_valid_json_surrogate_pair_and_large_ascii_text():
    emoji = json.loads('"\\ud83d\\ude00"')
    reject_lone_surrogates({"text": [emoji, "a" * 100_000]})


@pytest.mark.parametrize("kwargs", [
    {"text": "\ud800"},
    {"model_kwargs": {"instructions": ["okay", {"text": "\ud800"}]}},
    {"prompt_parts": [PromptPart(modality="text", text="\ud800")]},
])
def test_submit_refuses_before_any_server_state_is_read(kwargs):
    server = object.__new__(APIServer)
    with pytest.raises(ValueError, match="text is not valid UTF-8"):
        server.submit_request(input_modalities=["text"], output_modalities=["text"], **kwargs)
