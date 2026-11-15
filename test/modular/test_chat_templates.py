"""A chat reaches each model's chat template as the messages the client sent.

The chat API used to join every message into one user turn, so a model read its
own earlier reply, and the client's system message, as the user's words. A
model with a chat template now hands it one message per turn, so a two-turn
chat must come out as exactly the ids the template gives the client's own
messages: a turn merged, dropped or given the wrong role changes them.

Needs each checkpoint's tokenizer and template, not its weights, so a model
skips unless its directory is set.
"""

from __future__ import annotations

import os

import pytest

from mstar.api_server.openai.adapters import flatten_messages

CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "My name is Ada."},
    {"role": "assistant", "content": "Hello Ada."},
    {"role": "user", "content": "What is my name?"},
]


def _qwen3_omni(path, text, in_mods, parts):
    """Through `process_prompt`, whose default persona the client's system message replaces."""
    from transformers import AutoProcessor

    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model._processor = AutoProcessor.from_pretrained(path, trust_remote_code=False)
    model.tokenizer = model._processor.tokenizer
    tensors = model.process_prompt(text, in_mods, ["text"], prompt_parts=parts).new_input_tensors
    [ids] = [t for name, ts in tensors.items() if name.startswith("text_inputs.") for t in ts]
    rendered = model._processor.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
    return ids.tolist(), model.tokenizer(rendered)["input_ids"]


def _qwen3_5(path, text, in_mods, parts):
    """Through `process_prompt`, which thinks unless told not to, where the template's default does not."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = Qwen3_5DenseModel(path)
    [ids] = model.process_prompt(text, in_mods, ["text"], prompt_parts=parts).new_input_tensors["text_inputs"]
    rendered = model.tokenizer.apply_chat_template(
        CHAT, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    return ids.tolist(), model.tokenizer(rendered)["input_ids"]


def _cosmos3_edge(path, text, in_mods, parts):
    """Through `render_chat`, the text `process_prompt` tokenizes for a chat with no attachment.

    `process_prompt` itself also wants the vision encoder's directory, which a
    template check has no use for. ``system_prompt`` is the default the
    client's system message replaces.
    """
    from transformers import AutoTokenizer

    from mstar.model.cosmos3.components.reasoner import render_chat
    from mstar.model.cosmos3.config import Cosmos3Config

    tok = AutoTokenizer.from_pretrained(path)
    reasoner = Cosmos3Config.from_pretrained(path).reasoner
    ours = render_chat(tok, parts, reasoner, enable_thinking=False, system_prompt="Drive carefully.")
    rendered = tok.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(ours, add_special_tokens=False)["input_ids"], tok(rendered, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize(("env", "render"), [
    ("MSTAR_QWEN3_OMNI_PATH", _qwen3_omni),
    ("QWEN3_5_CKPT", _qwen3_5),
    ("COSMOS3_EDGE_DIR", _cosmos3_edge),
], ids=["qwen3-omni", "qwen3.5", "cosmos3-edge"])
def test_a_two_turn_chat_renders_as_its_template_renders_its_messages(env, render, tmp_path):
    """Each message reaches the template as its own turn, the client's system message in place of any default."""
    if not os.environ.get(env):
        pytest.skip(f"set {env} to the checkpoint's directory")
    text, _, in_mods, parts = flatten_messages(CHAT, tmp_path)
    ours, template = render(os.environ[env], text, in_mods, parts)
    assert ours == template, "the chat reached the template as one user message, its turns glued together"


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]},
}}]
# a tool loop as a client sends it back: the reply's reasoning and calls (arguments as JSON text),
# then each result, then a question about them
TOOL_LOOP = [
    {"role": "system", "content": "Use tools when they help."},
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {"role": "assistant", "content": None, "reasoning": "I should check both cities.", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
        {"id": "c1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Rome", "days": 2}'}},
    ]},
    {"role": "tool", "tool_call_id": "c0", "content": "sunny, 24C"},
    {"role": "tool", "tool_call_id": "c1", "content": "rainy, 15C"},
]
# the same, as the template reads it: each call's arguments an object, the reasoning as reasoning_content
TOOL_LOOP_TEMPLATE = [
    TOOL_LOOP[0], TOOL_LOOP[1],
    {"role": "assistant", "content": "", "reasoning_content": "I should check both cities.", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}},
    ]},
    {"role": "tool", "content": "sunny, 24C"},
    {"role": "tool", "content": "rainy, 15C"},
]


@pytest.mark.parametrize("thinking", [True, False])
def test_a_tool_loop_renders_as_qwen3_5s_template_renders_its_messages(tmp_path, thinking):
    """Tools in the prompt, the assistant's calls and reasoning on its turn, and each tool result its
    own block: token for token the ids Qwen3.5's template gives the client's messages."""
    path = os.environ.get("QWEN3_5_CKPT")
    if not path:
        pytest.skip("set QWEN3_5_CKPT to the checkpoint's directory")
    from mstar.api_server.openai.adapters import Qwen3_5Adapter
    from mstar.api_server.openai.protocol import ChatCompletionRequest
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    args = Qwen3_5Adapter().chat_to_request(ChatCompletionRequest(messages=TOOL_LOOP, tools=TOOLS), tmp_path)
    model = Qwen3_5DenseModel(path)
    [ids] = model.process_prompt(
        args.text, args.input_modalities, ["text"], prompt_parts=args.prompt_parts,
        tools=args.model_kwargs["tools"], enable_thinking=thinking,
    ).new_input_tensors["text_inputs"]
    rendered = model.tokenizer.apply_chat_template(
        TOOL_LOOP_TEMPLATE, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )
    assert ids.tolist() == model.tokenizer(rendered)["input_ids"]
