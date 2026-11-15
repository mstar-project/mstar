"""A request's tool fields (``tools``, ``tool_choice``, ``parallel_tool_calls``) and tool turns.

A chat adapter that serves tools checks them first (`check_tools`); `APIServer.submit_request`
checks them again for a model that calls tools (``Model.TOOL_FORMAT``), since
``chat_template_kwargs`` and ``/generate`` reach the model without the adapter's check, and
refuses a call for a model that doesn't, as Dynamo's bridge does.
"""

from __future__ import annotations

import json
from dataclasses import replace

from mstar.engine.resources.sampler.grammar import STRUCTURED_OUTPUTS_KWARG
from mstar.model.multimodal import PromptPart

TOOL_KWARGS = ("tools", "tool_choice", "parallel_tool_calls")

_TOOL_CHOICE_SHAPE = 'tool_choice must be none, auto, required or {"type": "function", "function": {"name": ...}}'


def check_tools(mk: dict) -> None:
    """A request's tool fields, checked and kept. ``tools`` empty drops them all; ``tool_choice``
    ``none`` keeps ``tools`` in the prompt and asks for no call. The older ``functions`` /
    ``function_call`` aren't served. The Rust frontend's ``check_tools`` is the same, word for word."""
    function_call = mk.get("function_call")
    if (mk.get("functions") not in (None, []) and function_call != "none") \
            or function_call not in (None, "none", "auto"):
        raise ValueError("functions and function_call are not supported; send tools and tool_choice")
    tools, choice, parallel = mk.get("tools"), mk.get("tool_choice"), mk.get("parallel_tool_calls")
    if tools in (None, []):
        if choice == "required" or isinstance(choice, dict):
            raise ValueError("tool_choice asks for a call, but tools lists none")
        for name in TOOL_KWARGS:
            mk.pop(name, None)
        return
    shape = 'each of tools must be {"type": "function", "function": {"name": ...}}'
    if not isinstance(tools, list):
        raise ValueError(shape)
    names = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(shape)
        if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
            raise ValueError(f"tool {fn['name']!r}'s parameters must be a JSON schema object")
        if fn.get("strict") is not None and not isinstance(fn["strict"], bool):
            raise ValueError(f"tool {fn['name']!r}'s strict must be true or false")
        names.append(fn["name"])
    if isinstance(choice, dict):
        fn = choice.get("function") if choice.get("type") == "function" else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(_TOOL_CHOICE_SHAPE)
        if name not in names:
            raise ValueError(f"tool_choice names {name!r}, which tools doesn't list")
    elif choice not in (None, "none", "auto", "required"):
        raise ValueError(_TOOL_CHOICE_SHAPE)
    if parallel is not None and not isinstance(parallel, bool):
        raise ValueError("parallel_tool_calls must be true or false")
    if mk.get(STRUCTURED_OUTPUTS_KWARG) is not None:
        raise ValueError("send tools or a JSON response_format (structured_outputs), not both")


def asks_for_tools(model_kwargs: dict, parts: list[PromptPart] | None) -> bool:
    """Whether a request asks for a tool call (``tools`` that ``tool_choice`` doesn't turn off) or
    carries a tool turn (a ``tool`` message, or an assistant message's ``tool_calls``). An
    assistant message's reasoning alone is not one."""
    if model_kwargs.get("tools") not in (None, []) and model_kwargs.get("tool_choice") != "none":
        return True
    return any(
        p.role == "tool" or (p.extras is not None and "tool_calls" in json.loads(p.extras))
        for p in parts or ()
    )


def without_tools(model_kwargs: dict, parts: list[PromptPart] | None) -> list[PromptPart] | None:
    """A request that `asks_for_tools` not, for a model that calls none: its tool fields dropped,
    and its assistant turns' reasoning, as before tools were served."""
    for name in TOOL_KWARGS:
        model_kwargs.pop(name, None)
    if parts is None or all(p.extras is None for p in parts):
        return parts
    return [replace(p, extras=None) if p.extras is not None else p for p in parts]
