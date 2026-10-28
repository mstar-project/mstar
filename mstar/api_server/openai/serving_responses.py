"""OpenAI's ``POST /v1/responses``, stateless, translated onto the chat path:
the request becomes a chat request the model's adapter takes, and the reply's
``reply`` events come back as a response object, whole or streamed as its
server-sent events.

What maps: ``input`` (a string, or ``message``, ``function_call``,
``function_call_output`` and ``reasoning`` items), ``instructions`` (a system
turn first), function ``tools``, ``tool_choice``, ``parallel_tool_calls``,
``max_output_tokens``, ``reasoning.effort``, ``temperature``, ``top_p`` and
``stream``. ``metadata``, ``user``, ``prompt_cache_key``, ``safety_identifier``
and ``stream_options`` are hints and drop. A 400 for what needs state or a
server this does not have: ``previous_response_id``, ``conversation``,
``prompt``, ``store: true``, ``background``, built-in and custom tools, a
``strict`` schema (that needs constrained decoding), images and files, and
unknown fields.

Choices where the APIs differ:

* Reasoning comes back as a ``reasoning`` item's ``reasoning_text`` content
  (``response.reasoning_text.delta`` streamed), which Codex reads and sends
  back; there is no summary, so ``reasoning.summary`` other than ``auto`` is a
  400. ``include: ["reasoning.encrypted_content"]`` is taken, and served by
  that content: ``encrypted_content`` stays unset, as the reasoning is no
  secret. An input reasoning item is read for its content, or its summary.
* ``reasoning.effort`` maps onto the template's levels, least to most: the
  lowest takes ``minimal`` and ``low``, the middle one ``medium`` (of two, the
  lower), the highest ``high`` and ``xhigh`` (GLM-5.2's levels are high and
  max, GLM-5.3's low, high and max, Qwen3.8's low, medium and xhigh). A
  template with no levels takes the effort as a hint and drops it. ``none``
  turns thinking off where the model can.
* ``output_tokens_details.reasoning_tokens`` counts the tokens of the chunks
  that began inside the reasoning; ``input_tokens_details`` is left out.
* A reply cut off by ``max_output_tokens`` is ``incomplete``, and a call it
  cut keeps its ``arguments`` as far as they got, with ``status: incomplete``.
"""

from __future__ import annotations

from mstar.api_server.openai import reply
from mstar.api_server.openai._util import now, rid, sse
from mstar.api_server.openai.protocol import ChatCompletionRequest

_FIELDS = {"model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "max_output_tokens",
           "reasoning", "temperature", "top_p", "stream", "store", "include", "text", "truncation",
           "service_tier", "top_logprobs", "metadata", "user", "prompt_cache_key", "safety_identifier",
           "stream_options", "previous_response_id", "conversation", "prompt", "background", "max_tool_calls"}
# set to anything but their off value, these need what a stateless server does not have
_OFF = {"previous_response_id": (None,), "conversation": (None,), "prompt": (None,), "store": (None, False),
        "background": (None, False), "max_tool_calls": (None,), "top_logprobs": (None, 0),
        "truncation": (None, "disabled"), "service_tier": (None, "auto", "default")}
_EFFORT_RANK = {"minimal": 0, "low": 0, "medium": 1, "high": 2, "xhigh": 2}


def to_chat_request(body, efforts: tuple[str, ...]) -> ChatCompletionRequest:
    """The chat request a Responses request asks for. ``efforts`` are the
    template's reasoning levels, least first."""
    if not isinstance(body, dict):
        raise ValueError("the request body must be a JSON object")
    unknown = sorted(set(body) - _FIELDS)
    if unknown:
        raise ValueError(f"unsupported field(s) {unknown}")
    for key, off in _OFF.items():
        if body.get(key) not in off:
            raise ValueError(f"{key}={body[key]!r} is not served: responses are stateless and not stored")
    include = body.get("include") or []
    if not isinstance(include, list) or set(include) - {"reasoning.encrypted_content"}:
        raise ValueError(f"include takes only 'reasoning.encrypted_content'; got {include!r}")
    text = body.get("text") or {}
    if (text.get("format") or {"type": "text"}).get("type") != "text" or text.get("verbosity") not in (None, "medium"):
        raise ValueError(f"text takes the text format only, structured output needs constrained decoding; "
                         f"got {text!r}")

    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    messages += _input(body.get("input"))
    chat: dict = {"messages": messages, "stream": bool(body.get("stream"))}
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "parallel_tool_calls"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("tools"):
        chat["tools"] = _tools(body["tools"])
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = _tool_choice(body["tool_choice"])
    chat.update(_reasoning(body.get("reasoning"), efforts))
    return ChatCompletionRequest(**chat)


def _parts_text(content, where: str) -> str:
    """The text of a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"{where} must be a string or a list of content parts")
    texts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text") and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            texts.append(part["refusal"])
        else:
            raise ValueError(f"{where} takes text parts only; got {kind!r}"
                             + (" (the model reads text)" if kind in ("input_image", "input_file", "input_audio")
                                else ""))
    return "".join(texts)


def _input(items) -> list[dict]:
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ValueError("input must be a string or a non-empty list of items")
    out: list[dict] = []
    turn: dict | None = None  # the assistant turn being put together

    def flush():
        nonlocal turn
        if turn is not None:
            out.append(turn)
            turn = None

    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"input items are objects; got {item!r}")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ValueError(f"a message's role must be user, assistant, system or developer, not {role!r}")
            text = _parts_text(item.get("content"), f"a {role} message's content")
            if role == "assistant":
                if turn is not None and (turn.get("content") or turn.get("tool_calls")):
                    flush()
                turn = turn or {"role": "assistant"}
                turn["content"] = text
            else:
                flush()
                out.append({"role": role, "content": text})
        elif kind == "reasoning":
            texts = [p.get("text") for p in item.get("content") or [] if isinstance(p, dict)] or \
                [p.get("text") for p in item.get("summary") or [] if isinstance(p, dict)]
            if not all(isinstance(t, str) for t in texts):
                raise ValueError(f"a reasoning item's content and summary are text parts; got {item!r}")
            if not texts and item.get("encrypted_content"):
                raise ValueError("a reasoning item with only encrypted_content cannot be read; send its content")
            flush()
            turn = {"role": "assistant", "reasoning_content": "\n".join(texts)}
        elif kind == "function_call":
            if not all(isinstance(item.get(k), str) for k in ("call_id", "name", "arguments")):
                raise ValueError("a function_call item needs its call_id, name and arguments")
            turn = turn or {"role": "assistant"}
            turn.setdefault("tool_calls", []).append({"id": item["call_id"], "type": "function", "function": {
                "name": item["name"], "arguments": item["arguments"]}})
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise ValueError("a function_call_output item needs its call_id")
            flush()
            out.append({"role": "tool", "tool_call_id": item["call_id"],
                        "content": _parts_text(item.get("output", ""), "a function_call_output's output")})
        else:
            raise ValueError(f"input item type {kind!r} is not served; send message, function_call, "
                             "function_call_output or reasoning items")
    flush()
    if out[-1]["role"] == "assistant":
        raise ValueError("the input ends with an assistant turn; end with a user message or a function_call_output")
    return out


def _tools(tools) -> list[dict]:
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    out = []
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise ValueError(f"tools take function tools only; {kind!r} is not served")
        if not isinstance(tool.get("name"), str):
            raise ValueError(f"a function tool needs a name; got {tool!r}")
        if tool.get("strict"):
            raise ValueError(f"tool {tool['name']!r}: strict schemas need constrained decoding; send strict: false")
        fn = {"name": tool["name"], "parameters": tool.get("parameters") or {"type": "object", "properties": {}}}
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice(choice):
    if choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    raise ValueError(f"tool_choice takes 'auto', 'none', 'required' or a function by name; got {choice!r}")


def _reasoning(reasoning, efforts: tuple[str, ...]) -> dict:
    if reasoning is None:
        return {}
    if not isinstance(reasoning, dict):
        raise ValueError(f"reasoning must be an object; got {reasoning!r}")
    for key in ("summary", "generate_summary"):
        if reasoning.get(key) not in (None, "auto"):
            raise ValueError(f"reasoning.{key}={reasoning[key]!r}: no summary is made; the reasoning comes whole")
    effort = reasoning.get("effort")
    if effort is None:
        return {}
    if effort == "none":
        return {"enable_thinking": False}
    if effort not in _EFFORT_RANK:
        raise ValueError(f"reasoning.effort takes none, minimal, low, medium, high or xhigh; got {effort!r}")
    if not efforts:
        return {}
    n = len(efforts)
    return {"reasoning_effort": efforts[{0: 0, 1: (n - 1) // 2, 2: n - 1}[_EFFORT_RANK[effort]]]}


async def create_response(api, model_name, adapter, body, raw_request=None):
    req = to_chat_request(body, adapter.reasoning_efforts)
    args = adapter.chat_to_request(req, api.upload_dir)
    parser = adapter.make_output_parser(req)
    request_id = rid("resp")
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
    events = reply.reply_events(api, request_id, parser, None, usage)
    response = _Response(model_name, request_id, body)
    if req.stream:
        return _stream(response, events, usage)
    for e in await reply.collect(events, raw_request):
        response.add(e)
    return response.body()


class _Response:
    """A response object as its output items add up."""

    def __init__(self, model_name, request_id, body: dict):
        self.id, self.model_name, self.request = request_id, model_name, body
        self.created = now()
        self.output: list[dict] = []
        self.finish: reply.Finish | None = None
        self.failure: reply.Failure | None = None

    @property
    def item(self) -> dict | None:
        return self.output[-1] if self.output else None

    def add(self, e) -> list[tuple[str, dict]]:
        """Adds one event; returns the stream events it makes, (type, fields)."""
        out: list[tuple[str, dict]] = []
        if isinstance(e, reply.Failure):
            self.failure = e
            out += self._close("incomplete")
            out.append(("response.failed", {"response": self.body()}))
        elif isinstance(e, reply.Finish):
            self.finish = e
            out += self._close("incomplete" if e.reason == "length" else "completed")
            kind = "response.incomplete" if e.reason == "length" else "response.completed"
            out.append((kind, {"response": self.body()}))
        elif "reasoning_content" in e:
            out += self._text("reasoning", e["reasoning_content"])
        elif "content" in e:
            out += self._text("message", e["content"])
        elif "tool_calls" in e:
            (call,) = e["tool_calls"]
            fn = call["function"]
            if "id" in call:
                out += self._close("completed")
                out += self._open({"type": "function_call", "id": rid("fc"), "call_id": call["id"],
                                   "name": fn["name"], "arguments": "", "status": "in_progress"})
            if fn.get("arguments"):
                self.item["arguments"] += fn["arguments"]
                out.append(("response.function_call_arguments.delta", {
                    "item_id": self.item["id"], "output_index": len(self.output) - 1, "delta": fn["arguments"]}))
        return out

    def _open(self, item: dict) -> list[tuple[str, dict]]:
        self.output.append(item)
        index = len(self.output) - 1
        # a part arrives in content_part.added
        added = {**item, "content": []} if "content" in item else dict(item)
        out = [("response.output_item.added", {"output_index": index, "item": added})]
        part = _part(item)
        if part is not None:
            out.append(("response.content_part.added", {
                "item_id": item["id"], "output_index": index, "content_index": 0, "part": dict(part, text="")}))
        return out

    def _text(self, kind: str, text: str) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        if self.item is None or self.item["type"] != kind or self.item.get("status") != "in_progress":
            out += self._close("completed")
            content_type = "reasoning_text" if kind == "reasoning" else "output_text"
            part = {"type": content_type, "text": ""}
            if kind == "message":
                part.update(annotations=[], logprobs=[])
                item = {"type": "message", "id": rid("msg"), "status": "in_progress", "role": "assistant",
                        "content": [part]}
            else:
                item = {"type": "reasoning", "id": rid("rs"), "summary": [], "content": [part],
                        "status": "in_progress"}
            out += self._open(item)
        part = _part(self.item)
        part["text"] += text
        delta = "response.reasoning_text.delta" if kind == "reasoning" else "response.output_text.delta"
        fields = {"item_id": self.item["id"], "output_index": len(self.output) - 1, "content_index": 0,
                  "delta": text}
        if kind == "message":
            fields["logprobs"] = []
        out.append((delta, fields))
        return out

    def _close(self, status: str) -> list[tuple[str, dict]]:
        item = self.item
        if item is None or item.get("status") != "in_progress":
            return []
        index = len(self.output) - 1
        out: list[tuple[str, dict]] = []
        part = _part(item)
        if part is not None:
            done = "response.reasoning_text.done" if item["type"] == "reasoning" else "response.output_text.done"
            fields = {"item_id": item["id"], "output_index": index, "content_index": 0, "text": part["text"]}
            if item["type"] == "message":
                fields["logprobs"] = []
            out.append((done, fields))
            out.append(("response.content_part.done", {
                "item_id": item["id"], "output_index": index, "content_index": 0, "part": dict(part)}))
        else:
            out.append(("response.function_call_arguments.done", {
                "item_id": item["id"], "output_index": index, "name": item["name"], "arguments": item["arguments"]}))
        # what the token budget cut is incomplete: a call's arguments are not whole
        item["status"] = status
        out.append(("response.output_item.done", {"output_index": index, "item": _copy(item)}))
        return out

    def body(self, status: str | None = None) -> dict:
        req = self.request
        finish, failure = self.finish, self.failure
        if status is None:
            status = "failed" if failure else "in_progress" if finish is None else \
                "incomplete" if finish.reason == "length" else "completed"
        body = {
            "id": self.id, "object": "response", "created_at": self.created, "status": status,
            "background": False,
            "error": {"code": "server_error", "message": failure.message} if failure else None,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "instructions": req.get("instructions"), "max_output_tokens": req.get("max_output_tokens"),
            "metadata": req.get("metadata") or {}, "model": self.model_name,
            "output": [_copy(i) for i in self.output],
            "parallel_tool_calls": req.get("parallel_tool_calls", True), "previous_response_id": None,
            "reasoning": req.get("reasoning") or {"effort": None, "summary": None}, "store": False,
            "temperature": req.get("temperature"), "text": {"format": {"type": "text"}},
            "tool_choice": req.get("tool_choice", "auto"), "tools": req.get("tools") or [],
            "top_p": req.get("top_p"), "truncation": "disabled", "usage": None,
        }
        if finish is not None:
            u = finish.usage
            body["usage"] = {"input_tokens": u.prompt, "output_tokens": u.completion,
                             "output_tokens_details": {"reasoning_tokens": u.reasoning},
                             "total_tokens": u.prompt + u.completion}
        return body


def _part(item: dict) -> dict | None:
    return item["content"][0] if item.get("type") in ("message", "reasoning") else None


def _copy(item: dict) -> dict:
    return {**item, "content": [dict(p) for p in item["content"]]} if "content" in item else dict(item)


async def _stream(response: _Response, events, usage: reply.Usage):
    """The Responses stream: ``response.created`` with the first output, as
    clients time the first token from the first event; each item's events;
    then ``response.completed`` (or ``incomplete``, or ``failed``)."""
    seq = 0

    def write(kind: str, fields: dict) -> str:
        nonlocal seq
        seq += 1
        return sse({"type": kind, "sequence_number": seq - 1, **fields}, kind)

    started = False
    async for batch in events:
        out: list[str] = []
        for e in batch:
            if not started:
                started = True
                out.append(write("response.created", {"response": response.body("in_progress")}))
            out += [write(kind, fields) for kind, fields in response.add(e)]
        if out:
            yield "".join(out)
