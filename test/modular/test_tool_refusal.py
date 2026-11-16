"""A request's tool fields at `submit_request`, through the real method. A chat reaches it from an
adapter that serves several models (Qwen3.5's also serves Muse Glimmer and Phi-4-multimodal), so a
call is refused there for a model that declares no tool format; and ``chat_template_kwargs`` and
``/generate`` reach it past the chat adapter's check, so a model that calls tools checks them there."""

from types import SimpleNamespace

import pytest

from mstar.api_server.entrypoint import APIServer
from mstar.model.multimodal import PromptPart

TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]
REASONED = PromptPart(modality="text", text="Hi", role="assistant", extras='{"reasoning_content":"greet"}')


def _server(tool_format):
    return SimpleNamespace(model=SimpleNamespace(TOOL_FORMAT=tool_format))


@pytest.mark.parametrize(("kwargs", "parts"), [
    ({"tools": TOOLS}, None),
    ({"tools": TOOLS, "tool_choice": "required"}, None),
    ({}, [PromptPart(modality="text", text="sunny", role="tool")]),
], ids=["tools", "required", "tool-message"])
def test_a_chat_to_a_model_without_a_tool_format_refuses_a_call_or_a_tool_message(kwargs, parts):
    with pytest.raises(ValueError, match="^tool calling is not supported for this model$"):
        APIServer._serve_tool_fields(_server(None), dict(kwargs), parts, chat=True)


def test_a_chat_to_a_model_without_a_tool_format_drops_the_rest():
    """Tools with ``tool_choice`` none, and an assistant turn's ``tool_calls`` and reasoning, which
    ask for no call: served as before tools were, without them."""
    kwargs = {"tools": TOOLS, "tool_choice": "none", "parallel_tool_calls": True, "seed": 1}
    called = PromptPart(modality="text", text="", role="assistant", extras='{"tool_calls":[]}')
    parts = [PromptPart(modality="text", text="hi", role="user"), REASONED, called]
    kept = APIServer._serve_tool_fields(_server(None), kwargs, parts, chat=True)
    assert kwargs == {"seed": 1}
    assert [(p.role, p.text, p.extras) for p in kept] == [
        ("user", "hi", None), ("assistant", "Hi", None), ("assistant", "", None)]


def test_generate_on_a_model_without_a_tool_format_is_as_before():
    """Tool kwargs a model doesn't read: kept, and logged as ignored, as before tools were served."""
    kwargs = {"tools": TOOLS, "tool_choice": "required"}
    APIServer._serve_tool_fields(_server(None), kwargs, None, chat=False)
    assert kwargs == {"tools": TOOLS, "tool_choice": "required"}


@pytest.mark.parametrize("chat", [True, False], ids=["chat", "generate"])
def test_a_model_with_a_tool_format_keeps_checked_tools(chat):
    kwargs = {"tools": TOOLS, "tool_choice": "required"}
    parts = [PromptPart(modality="text", text="sunny", role="tool"), REASONED]
    assert APIServer._serve_tool_fields(_server("qwen_3_5"), kwargs, parts, chat=chat) == parts
    assert kwargs == {"tools": TOOLS, "tool_choice": "required"}


@pytest.mark.parametrize(("kwargs", "error"), [
    ({"tools": [{"function": {}}]}, "^each of tools must be"),
    ({"tools": ["get_weather"]}, "^each of tools must be"),
    ({"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "send_email"}}}, "doesn't list"),
    ({"tools": TOOLS, "structured_outputs": {"regex": "a"}}, "need tool_choice none"),
], ids=["no-name", "not-an-object", "unlisted-choice", "with-structured-outputs"])
@pytest.mark.parametrize("chat", [True, False], ids=["chat", "generate"])
def test_a_model_with_a_tool_format_checks_tools_that_skipped_the_adapter(kwargs, error, chat):
    """``chat_template_kwargs`` (chat) and ``model_kwargs`` (``/generate``) carry tools the chat
    adapter never checked: a 400, before they reach the template, the grammar or the reply parser."""
    with pytest.raises(ValueError, match=error):
        APIServer._serve_tool_fields(_server("qwen_3_5"), dict(kwargs), None, chat=chat)


def test_tools_with_tool_choice_none_take_a_json_response_format():
    """No call can come, so the reply's JSON grammar holds alone, as before tools were served."""
    kwargs = {"tools": TOOLS, "tool_choice": "none", "structured_outputs": {"json_object": True}}
    APIServer._serve_tool_fields(_server("qwen_3_5"), kwargs, None, chat=True)
    assert kwargs["tools"] == TOOLS
