"""GLM-5.2 / GLM-5.3 chat: the request's messages and tools in the shape the
checkpoint's chat template reads, and the reply split back into reasoning,
content and tool calls.

The template ends the prompt inside ``<think>`` when the model reasons, so the
reply opens with reasoning and ``</think>`` hands over to the answer. A call is
``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>``:
a string value arrives bare and any other as JSON, as the template renders them.
"""

from __future__ import annotations

import json
import math
import re
import uuid

THINK_START, THINK_END = "<think>", "</think>"
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

# what the templates read; others are dropped, as some would collide with
# apply_chat_template's own arguments
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "clear_thinking")


def template_messages(messages) -> list[dict]:
    """The messages with their roles, tool calls and tool results. The template
    iterates an assistant call's ``arguments``, which OpenAI clients send as a
    JSON string. A null field is left out: GLM-5.2's template prints ``None``.
    ``developer`` turns are ``system`` turns; the templates know only the latter."""
    out = []
    for m in messages:
        msg = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        msg = {k: v for k, v in msg.items() if v is not None}
        if msg.get("role") == "developer":
            msg["role"] = "system"
        if msg.get("tool_calls"):
            msg["tool_calls"] = [_call_for_template(c) for c in msg["tool_calls"]]
        out.append(msg)
    return out


def _call_for_template(call) -> dict:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise ValueError(f"tool_calls entries need a function object, got {call!r}")
    fn = dict(call["function"])
    if not isinstance(fn.get("name"), str) or not fn["name"]:
        raise ValueError(f"tool_calls entries need a function name, got {call!r}")
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            raise ValueError(f"tool call arguments are not JSON: {args!r}") from None
    if not isinstance(args, dict):
        raise ValueError(f"tool call arguments must be a JSON object, got {args!r}")
    fn["arguments"] = args
    return {**call, "function": fn}


def tool_parameters(tools) -> dict[str, dict]:
    """Each function tool's parameter properties by name."""
    if not isinstance(tools, list):
        raise ValueError(f"tools must be a list, got {type(tools).__name__}")
    params = {}
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or tool.get("type", "function") != "function" \
                or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools take {{'type': 'function', 'function': {{'name': ...}}}}, got {tool!r}")
        parameters = fn.get("parameters") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            raise ValueError(f"tool {fn['name']!r}: parameters must be a JSON schema object with object properties")
        params[fn["name"]] = properties
    return params


def _json_types(schema) -> set[str]:
    """The JSON types a property schema admits; empty when it does not say."""
    if not isinstance(schema, dict):
        return set()
    kind = schema.get("type")
    kinds = [kind] if isinstance(kind, str) else kind if isinstance(kind, list) else []
    types = {k for k in kinds if isinstance(k, str)}
    for key in ("anyOf", "oneOf"):
        for branch in schema.get(key) if isinstance(schema.get(key), list) else []:
            types |= _json_types(branch)
    for value in schema.get("enum") if isinstance(schema.get("enum"), list) else []:
        types.add({str: "string", bool: "boolean", int: "integer", float: "number"}.get(type(value), "object"))
    return types


def _no_constant(name: str):
    raise ValueError(name)


def _finite(value) -> bool:
    """No NaN or infinity anywhere in ``value`` (1e400 parses to inf)."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(map(_finite, value))
    if isinstance(value, dict):
        return all(map(_finite, value.values()))
    return True


def _arg_value(raw: str, schema):
    """A string value arrives bare and any other as JSON; a value the schema
    allows only as a string stays one ("94110" for a zip code)."""
    types = _json_types(schema)
    if types and types <= {"string", "null"}:
        return raw
    try:
        # NaN and Infinity are not JSON: the arguments string must stay parseable
        value = json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return raw
    if not _finite(value):
        return raw
    if types and _json_type(value) not in types and not (_json_type(value) == "integer" and "number" in types):
        return raw
    return value


def _json_type(value) -> str:
    if value is None:
        return "null"
    return {str: "string", bool: "boolean", int: "integer", float: "number", list: "array"}.get(type(value), "object")


def _partial_tag(text: str, tags) -> int:
    """Length of the longest end of ``text`` that may begin one of ``tags``."""
    start = text.rfind("<", max(0, len(text) - max(map(len, tags))))
    while start >= 0:
        tail = text[start:]
        if any(t.startswith(tail) for t in tags):
            return len(tail)
        start = text.find("<", start + 1)
    return 0


class GlmReplyParser:
    """Splits a streamed GLM reply into OpenAI deltas: ``reasoning_content``,
    ``content`` and one ``tool_calls`` delta per finished call. ``feed`` holds
    back text that may be the start of a tag; ``finish`` flushes the rest.

    As vLLM's ``glm47`` parser: ``<think>`` in the answer reopens the reasoning,
    a stray ``<think>`` in the reasoning and ``</think>`` in the answer drop, a
    call may end the reasoning without ``</think>``, and a call to a function the
    request did not offer stays text.
    Content is stripped at both ends, as the template strips it on the way back.
    """

    def __init__(self, *, thinking: bool, tools: dict[str, dict] | None):
        self._thinking = thinking
        self._state = "reasoning" if thinking else "content"
        self._tools = tools
        self._buf = ""
        self._calls = 0
        self._content_started = False
        self._held_space = ""

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self._calls else "stop"

    def feed(self, text: str) -> list[dict]:
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> list[dict]:
        out = self._drain(final=True)
        self._held_space = ""
        return out

    def message(self, text: str) -> dict:
        """The whole reply as a chat ``message``, from the deltas a stream sends."""
        deltas = self.feed(text) + self.finish()
        content = "".join(d.get("content", "") for d in deltas)
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        calls = [{k: v for k, v in c.items() if k != "index"}
                 for d in deltas for c in d.get("tool_calls", [])]
        message: dict = {"role": "assistant", "content": content or (None if calls else "")}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return message

    def _tags(self) -> tuple[str, ...]:
        calls = (CALL_START,) if self._tools else ()
        if self._state == "reasoning":
            return (THINK_START, THINK_END, *calls)
        return (THINK_START, THINK_END, *calls) if self._thinking else calls

    def _drain(self, final: bool) -> list[dict]:
        out: list[dict] = []
        while self._buf or (final and self._state == "call"):
            if self._state == "call":
                end = self._buf.find(CALL_END)
                if end < 0:
                    if final:  # cut off inside a call: its text is all there is
                        self._state = "content"
                        self._emit(out, CALL_START + self._buf)
                        self._buf = ""
                    break
                body, self._buf = self._buf[:end], self._buf[end + len(CALL_END):]
                self._state = "content"
                call = self._call(body)
                if call is None:
                    self._emit(out, CALL_START + body + CALL_END)
                else:
                    self._held_space = ""
                    out.append({"tool_calls": [call]})
                continue
            tags = self._tags()
            hits = [(i, t) for t in tags if (i := self._buf.find(t)) >= 0]
            if not hits:
                keep = 0 if final or not tags else _partial_tag(self._buf, tags)
                self._emit(out, self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                break
            i, tag = min(hits)
            self._emit(out, self._buf[:i])
            self._buf = self._buf[i + len(tag):]
            if tag == THINK_END and self._state == "reasoning":
                self._state = "content"
            elif tag == THINK_START:
                self._state = "reasoning"
            elif tag == CALL_START:
                self._state = "call"
        return _merged(out)

    def _emit(self, out: list[dict], text: str) -> None:
        if not text:
            return
        if self._state == "reasoning":
            out.append({"reasoning_content": text})
            return
        if not self._content_started:
            text = text.lstrip()
        body = text.rstrip()
        if not body:
            self._held_space += text
            return
        out.append({"content": self._held_space + body})
        self._held_space = text[len(body):]
        self._content_started = True

    def _call(self, body: str) -> dict | None:
        name, sep, rest = body.partition("<arg_key>")
        name = name.strip()
        if name not in self._tools:
            return None
        props = self._tools[name]
        args = {k.strip(): _arg_value(v, props.get(k.strip())) for k, v in _ARG.findall(sep + rest)}
        index, self._calls = self._calls, self._calls + 1
        return {
            "index": index,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }


def _merged(deltas: list[dict]) -> list[dict]:
    out: list[dict] = []
    for d in deltas:
        key = next(iter(d))
        if out and key != "tool_calls" and set(out[-1]) == {key}:
            out[-1] = {key: out[-1][key] + d[key]}
        else:
            out.append(d)
    return out
