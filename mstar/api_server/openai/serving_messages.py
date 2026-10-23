"""Anthropic's ``POST /v1/messages`` (and ``/v1/messages/count_tokens``),
translated onto the chat path: the request becomes a chat request the model's
adapter takes, and the reply's ``reply`` events come back as an Anthropic
message, whole or streamed as its server-sent events.

What maps: ``system`` (a string or text blocks), ``messages`` with text,
``thinking``, ``tool_use`` and ``tool_result`` blocks, ``tools`` (their
``input_schema`` is the function's parameters), ``tool_choice`` (``auto``,
``any`` as ``required``, ``tool`` as a named function, ``none``, and
``disable_parallel_tool_use``), ``max_tokens``, ``stop_sequences``,
``temperature``, ``top_p``, ``top_k`` and ``thinking``. ``metadata`` and
``cache_control`` are hints and drop. Anything else is a 400: images and
documents (the model reads text), server tools, redacted thinking, a final
assistant turn to continue (prefill), and unknown fields.

Choices where the APIs differ:

* ``thinking`` absent leaves the model's default (GLM reasons), where Anthropic
  does not think unless asked; ``disabled`` turns it off where the model can.
  ``budget_tokens`` is checked as Anthropic checks it but not enforced: the
  model's reasoning ends where it ends, within ``max_tokens``.
* A ``tool_result`` with ``is_error`` reads ``Error: `` before its content, as
  the template has no flag for it.
* A reply's ``thinking`` block carries an empty ``signature``; one sent back
  is read for its text.
* A call cut off by ``max_tokens`` has no ``input`` to give: the whole message
  leaves it out, the stream ends inside its ``input_json_delta`` pieces, and
  both say ``stop_reason: max_tokens``.
* Stop sequences match the answer's text, not the thinking or a call's input.
"""

from __future__ import annotations

import asyncio
import json

from mstar.api_server.openai import reply
from mstar.api_server.openai._util import rid, sse
from mstar.api_server.openai.protocol import ChatCompletionRequest

_FIELDS = {"model", "messages", "system", "max_tokens", "stop_sequences", "stream", "temperature", "top_p",
           "top_k", "tools", "tool_choice", "thinking", "metadata"}
_COUNT_FIELDS = {"model", "messages", "system", "tools", "tool_choice", "thinking"}

_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens"}


def error_type(status: int) -> str:
    """Anthropic's error ``type`` for an HTTP status."""
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
            503: "overloaded_error", 529: "overloaded_error"}.get(
        status, "invalid_request_error" if 400 <= status < 500 else "api_error")


def error_body(status: int, message: str) -> dict:
    return {"type": "error", "error": {"type": error_type(status), "message": message}}


def to_chat_request(body, *, counting: bool = False) -> tuple[ChatCompletionRequest, list[str]]:
    """The chat request a Messages request asks for, and its stop sequences.
    ``counting`` reads a ``count_tokens`` body, which has no ``max_tokens``."""
    if not isinstance(body, dict):
        raise ValueError("the request body must be a JSON object")
    unknown = sorted(set(body) - (_COUNT_FIELDS if counting else _FIELDS))
    if unknown:
        raise ValueError(f"unsupported field(s) {unknown}")
    chat: dict = {"messages": _system(body.get("system")) + _messages(body.get("messages"))}
    if not counting:
        max_tokens = body.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f"max_tokens is required, a positive integer; got {max_tokens!r}")
        chat["max_tokens"] = max_tokens
        for key in ("temperature", "top_p", "top_k"):
            if body.get(key) is not None:
                chat[key] = body[key]
        chat["stream"] = bool(body.get("stream"))
    if body.get("tools"):
        chat["tools"] = _tools(body["tools"])
    if body.get("tool_choice") is not None:
        chat.update(_tool_choice(body["tool_choice"]))
    thinking = _thinking(body.get("thinking"), body.get("max_tokens"))
    if thinking is not None:
        chat["enable_thinking"] = thinking
    stops = body.get("stop_sequences") or []
    if not isinstance(stops, list) or not all(isinstance(s, str) and s for s in stops):
        raise ValueError(f"stop_sequences must be a list of non-empty strings; got {stops!r}")
    return ChatCompletionRequest(**chat), stops


def _text(blocks, where: str) -> str:
    """The text of a string or a list of text blocks, a block a line."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise ValueError(f"{where} must be a string or a list of content blocks")
    texts = []
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind != "text" or not isinstance(block.get("text"), str):
            raise ValueError(f"{where} takes text blocks only; got a {kind or type(block).__name__!r} block"
                             + (" (the model reads text)" if kind in ("image", "document") else ""))
        texts.append(block["text"])
    return "\n".join(texts)


def _system(system) -> list[dict]:
    if system is None:
        return []
    text = _text(system, "system")
    return [{"role": "system", "content": text}] if text else []


def _messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out: list[dict] = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant"):
            raise ValueError(f"a message's role must be user or assistant, not {role!r}")
        content = m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            raise ValueError(f"a message's content must be a string or a list of blocks; got {content!r}")
        out.extend(_user(blocks) if role == "user" else [_assistant(blocks)])
    if messages[-1].get("role") == "assistant":
        raise ValueError("a final assistant message (prefill) is not supported; end with a user turn")
    return out


def _user(blocks: list) -> list[dict]:
    """A user turn: each tool result is a tool turn, text between them a user turn."""
    out: list[dict] = []
    texts: list[str] = []
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            texts.append(_text([block], "a user message"))
        elif kind == "tool_result":
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
                texts = []
            if not isinstance(block.get("tool_use_id"), str):
                raise ValueError("a tool_result block needs its tool_use_id")
            result = _text(block.get("content") or "", "a tool_result's content")
            if block.get("is_error"):
                result = "Error: " + result  # the template has no flag for it
            out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": result})
        else:
            raise ValueError(f"a user message takes text and tool_result blocks; got {kind!r}"
                             + (" (the model reads text)" if kind in ("image", "document") else ""))
    if texts:
        out.append({"role": "user", "content": "\n".join(texts)})
    return out


def _assistant(blocks: list) -> dict:
    texts, thinking, calls = [], [], []
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            texts.append(_text([block], "an assistant message"))
        elif kind == "thinking":
            if not isinstance(block.get("thinking"), str):
                raise ValueError("a thinking block needs its thinking text")
            thinking.append(block["thinking"])
        elif kind == "tool_use":
            if not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str):
                raise ValueError("a tool_use block needs its id and name")
            if not isinstance(block.get("input"), dict):
                raise ValueError(f"a tool_use block's input must be an object; got {block.get('input')!r}")
            calls.append({"id": block["id"], "type": "function",
                          "function": {"name": block["name"], "arguments": block["input"]}})
        else:
            raise ValueError(f"an assistant message takes text, thinking and tool_use blocks; got {kind!r}")
    message: dict = {"role": "assistant", "content": "\n".join(texts) or None}
    if thinking:
        message["reasoning_content"] = "\n".join(thinking)
    if calls:
        message["tool_calls"] = calls
    return message


def _tools(tools) -> list[dict]:
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    out = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "custom") != "custom":
            kind = tool.get("type") if isinstance(tool, dict) else tool
            raise ValueError(f"tools take client tools with an input_schema; {kind!r} is a server tool")
        if not isinstance(tool.get("name"), str) or not isinstance(tool.get("input_schema"), dict):
            raise ValueError(f"a tool needs a name and an input_schema object; got {tool!r}")
        fn = {"name": tool["name"], "parameters": tool["input_schema"]}
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice(choice) -> dict:
    kind = choice.get("type") if isinstance(choice, dict) else None
    chat = {"auto": "auto", "any": "required", "none": "none"}.get(kind)
    if kind == "tool":
        if not isinstance(choice.get("name"), str):
            raise ValueError("tool_choice of type tool needs a name")
        chat = {"type": "function", "function": {"name": choice["name"]}}
    if chat is None:
        raise ValueError(f"tool_choice takes type auto, any, tool or none; got {choice!r}")
    out: dict = {"tool_choice": chat}
    if choice.get("disable_parallel_tool_use"):
        out["parallel_tool_calls"] = False
    return out


def _thinking(thinking, max_tokens) -> bool | None:
    if thinking is None:
        return None
    kind = thinking.get("type") if isinstance(thinking, dict) else None
    if kind == "disabled":
        return False
    if kind == "adaptive":
        return True
    if kind != "enabled":
        raise ValueError(f"thinking takes type enabled, adaptive or disabled; got {thinking!r}")
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1024:
        raise ValueError(f"thinking.budget_tokens must be an integer of at least 1024; got {budget!r}")
    if isinstance(max_tokens, int) and budget >= max_tokens:
        raise ValueError(f"thinking.budget_tokens ({budget}) must be less than max_tokens ({max_tokens})")
    # a target, not a bound: the model's reasoning is bounded by max_tokens only
    return True


async def create_message(api, model_name, adapter, body, raw_request=None):
    req, stops = to_chat_request(body)
    args = adapter.chat_to_request(req, api.upload_dir)
    parser = adapter.make_output_parser(req)
    request_id = rid("msg")
    api.submit_request(
        text=args.text,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        prompt_parts=args.prompt_parts,
        streaming=bool(req.stream),
        request_id=request_id,
    )
    usage = reply.Usage()
    events = reply.reply_events(api, request_id, parser, stops, usage)
    if req.stream:
        return _stream(model_name, request_id, events, usage)
    return _message(model_name, request_id, await reply.collect(events, raw_request))


async def count_tokens(api, adapter, body) -> dict:
    """``input_tokens`` of the prompt a Messages request renders, by the
    model's own prompt processing (chat template and tokenizer)."""
    req, _ = to_chat_request(body, counting=True)
    args = adapter.chat_to_request(req, api.upload_dir)
    tensors = await asyncio.to_thread(
        api.model.process_prompt, args.text, args.input_modalities, args.output_modalities, **args.model_kwargs)
    return {"input_tokens": sum(t.numel() for t in tensors.get("text_inputs", []))}


def _stop(finish: reply.Finish) -> dict:
    reason = _STOP_REASONS.get(finish.reason, "stop_sequence" if finish.stop_string is not None else "end_turn")
    return {"stop_reason": reason, "stop_sequence": finish.stop_string}


def _message(model_name, request_id, events) -> dict:
    finish = events[-1]
    content = []
    for run in reply.runs(events):
        if run.kind == "reasoning":
            content.append({"type": "thinking", "thinking": run.text, "signature": ""})
        elif run.kind == "content":
            content.append({"type": "text", "text": run.text})
        else:
            try:
                arguments = json.loads(run.call["arguments"])
            except json.JSONDecodeError:
                continue  # cut off by max_tokens: there is no input to give
            content.append({"type": "tool_use", "id": run.call["id"], "name": run.call["name"], "input": arguments})
    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content,
        **_stop(finish),
        "usage": {"input_tokens": finish.usage.prompt, "output_tokens": finish.usage.completion},
    }


async def _stream(model_name, request_id, events, usage: reply.Usage):
    """The Messages stream: ``message_start``, each block's start, deltas and
    stop, ``message_delta`` with the stop reason and usage, ``message_stop``.
    ``message_start`` goes out with the first block, as clients time the first
    token from the first event, and carries the prompt's token count."""
    started = False
    block = None  # the open block's kind: "thinking", "text" or "tool_use"
    index = -1

    def start() -> str:
        nonlocal started
        started = True
        message = {"id": request_id, "type": "message", "role": "assistant", "model": model_name, "content": [],
                   "stop_reason": None, "stop_sequence": None,
                   "usage": {"input_tokens": usage.prompt, "output_tokens": 0}}
        return sse({"type": "message_start", "message": message}, "message_start")

    def open_block(kind: str, content_block: dict) -> list[str]:
        nonlocal block, index
        out = close_block()
        block, index = kind, index + 1
        out.append(sse({"type": "content_block_start", "index": index, "content_block": content_block},
                       "content_block_start"))
        return out

    def close_block() -> list[str]:
        nonlocal block
        if block is None:
            return []
        block = None
        return [sse({"type": "content_block_stop", "index": index}, "content_block_stop")]

    def delta(d: dict) -> str:
        return sse({"type": "content_block_delta", "index": index, "delta": d}, "content_block_delta")

    async for batch in events:
        out: list[str] = []
        for e in batch:
            if isinstance(e, reply.Failure):
                # the HTTP status is committed: the failure travels in-band
                out.append(sse(error_body(e.status, e.message), "error"))
                continue
            if not started:
                out.append(start())
            if isinstance(e, reply.Finish):
                out += close_block()
                out.append(sse({"type": "message_delta", "delta": _stop(e),
                                "usage": {"output_tokens": e.usage.completion}}, "message_delta"))
                out.append(sse({"type": "message_stop"}, "message_stop"))
            elif "reasoning_content" in e:
                if block != "thinking":
                    out += open_block("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                out.append(delta({"type": "thinking_delta", "thinking": e["reasoning_content"]}))
            elif "content" in e:
                if block != "text":
                    out += open_block("text", {"type": "text", "text": ""})
                out.append(delta({"type": "text_delta", "text": e["content"]}))
            elif "tool_calls" in e:
                (call,) = e["tool_calls"]
                fn = call["function"]
                if "id" in call:
                    out += open_block("tool_use", {"type": "tool_use", "id": call["id"], "name": fn["name"],
                                                   "input": {}})
                if fn.get("arguments"):
                    out.append(delta({"type": "input_json_delta", "partial_json": fn["arguments"]}))
        if out:
            yield "".join(out)
