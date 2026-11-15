"""A chat's tools reach `submit_request` from an adapter that serves several models (Qwen3.5's
also serves Muse Glimmer and Phi-4-multimodal), so the server refuses them for a model that
declares no tool format, through the real method."""

from types import SimpleNamespace

import pytest

from mstar.api_server.entrypoint import APIServer
from mstar.model.multimodal import PromptPart

TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]


def _server(tool_format):
    return SimpleNamespace(model=SimpleNamespace(TOOL_FORMAT=tool_format))


@pytest.mark.parametrize(("kwargs", "parts"), [
    ({"tools": TOOLS}, None),
    ({"tools": TOOLS, "tool_choice": "required"}, None),
    ({}, [PromptPart(modality="text", text="sunny", role="tool")]),
    ({}, [PromptPart(modality="text", text="", role="assistant", extras='{"tool_calls":[]}')]),
], ids=["tools", "required", "tool-turn", "tool-call-turn"])
def test_a_model_without_a_tool_format_refuses_a_call_or_a_tool_turn(kwargs, parts):
    with pytest.raises(ValueError, match="^tool calling is not supported for this model$"):
        APIServer._refuse_unserved_tools(_server(None), dict(kwargs), parts)


def test_a_model_without_a_tool_format_drops_tools_with_tool_choice_none():
    kwargs = {"tools": TOOLS, "tool_choice": "none", "parallel_tool_calls": True, "seed": 1}
    APIServer._refuse_unserved_tools(_server(None), kwargs, None)
    assert kwargs == {"seed": 1}


def test_a_model_with_a_tool_format_keeps_them():
    kwargs = {"tools": TOOLS, "tool_choice": "required"}
    parts = [PromptPart(modality="text", text="sunny", role="tool")]
    APIServer._refuse_unserved_tools(_server("qwen_3_5"), kwargs, parts)
    assert kwargs == {"tools": TOOLS, "tool_choice": "required"}
