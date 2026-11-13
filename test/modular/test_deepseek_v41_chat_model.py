"""DeepSeek-V4.1's chat adapter and its model together: the request the adapter submits,
through the model's own ``process_prompt``, is the prompt the checkpoint's encoder writes."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("transformers")

from test_deepseek_v41_chat_template import CASES, byte_tokenizer  # noqa: E402

from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.protocol import ChatCompletionRequest  # noqa: E402
from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model  # noqa: E402

RENDERED = [c for c in CASES["cases"] if c["prompt"] is not None]


@pytest.fixture(scope="module")
def model():
    """The model class with a tokenizer whose ids are the prompt's bytes, under the
    checkpoint's template: no checkpoint is read."""
    model = DeepseekV41Model("deepseek-ai/DeepSeek-V4.1-Flash", max_seq_len=1 << 20)
    model._tokenizer = byte_tokenizer()
    return model


def _prompt(model, tmp_path, **body) -> str:
    args = adapters.DeepseekV41ChatAdapter().chat_to_request(
        ChatCompletionRequest(model="deepseek_v41", **body), tmp_path)
    out = model.process_prompt(args.text, ["text"], ["text"], prompt_parts=args.prompt_parts, **args.model_kwargs)
    return model.tokenizer.decode(out.new_input_tensors["text_inputs"][0])


def _as_clients_send(messages):
    """Each call's arguments as the JSON string an OpenAI client sends."""
    messages = json.loads(json.dumps(messages))
    for m in messages:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    return messages


@pytest.mark.parametrize("case", RENDERED, ids=lambda c: c["id"])
def test_a_chat_request_reaches_the_engine_as_the_encoders_prompt(model, case, tmp_path):
    got = _prompt(model, tmp_path, messages=_as_clients_send(case["messages"]),
                  tools=CASES["toolsets"].get(case["tools"]), enable_thinking=case["thinking"], **case["switches"])
    assert got == case["prompt"]


def test_a_plain_prompt_is_one_user_turn(model, tmp_path):
    plain = model.process_prompt("weather in Paris?", ["text"], ["text"])
    chat = _prompt(model, tmp_path, messages=[{"role": "user", "content": "weather in Paris?"}])
    assert model.tokenizer.decode(plain.new_input_tensors["text_inputs"][0]) == chat


def test_a_tokenizer_without_the_template_is_refused(model):
    bare = DeepseekV41Model("deepseek-ai/DeepSeek-V4.1-Flash")
    bare._tokenizer = byte_tokenizer()
    bare._tokenizer.chat_template = None
    with pytest.raises(RuntimeError, match="chat_template.jinja"):
        bare.process_prompt("hello", ["text"], ["text"])
