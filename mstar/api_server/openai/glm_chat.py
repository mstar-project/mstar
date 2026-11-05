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
import uuid

THINK_START, THINK_END = "<think>", "</think>"
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
ARG_KEY, ARG_KEY_END = "<arg_key>", "</arg_key>"
ARG_VALUE, ARG_VALUE_END = "<arg_value>", "</arg_value>"
# what ends each phase of reading a call (see GlmReplyParser._drain_call)
_CALL_TAGS = {"name": (ARG_KEY, CALL_END), "text": (CALL_END,), "args": (ARG_KEY, CALL_END),
              "key": (ARG_KEY_END, CALL_END), "value": (ARG_VALUE_END, CALL_END)}

# what the templates read; others are dropped, as some would collide with
# apply_chat_template's own arguments
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "clear_thinking")


def template_messages(messages) -> list[dict]:
    """The messages as the templates read them: a call's ``arguments`` as the
    object the template iterates (clients send a JSON string), no null fields
    (GLM-5.2's template prints ``None``), and ``developer`` turns as ``system``."""
    out = []
    for m in messages:
        msg = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        msg = {k: v for k, v in msg.items() if v is not None}
        if msg.get("role") == "developer":
            msg["role"] = "system"
        if msg.get("role") not in ("system", "user", "assistant", "tool"):
            # the templates skip any other role: the model would answer without that turn
            raise ValueError(f"role {msg.get('role')!r} is not supported; use system, user, assistant or tool")
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
    if "integer" in types and isinstance(value, float) and value.is_integer():
        value = int(value)  # an integer the model wrote as 3.0 is still one
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
    ``content`` and ``tool_calls``. ``feed`` holds back text that may be the
    start of a tag; ``finish`` flushes the rest.

    As vLLM's ``glm47`` parser: ``<think>`` in the answer reopens the reasoning,
    a stray ``<think>`` in the reasoning and ``</think>`` in the answer drop, a
    call may end the reasoning without ``</think>``, and a call to a function the
    request did not offer stays text.
    Content is stripped at both ends, as the template strips it on the way back.
    ``forced_call`` says the prompt opened a call, and named it unless empty.
    ``single_call`` ends the reply at the end of its first call: ``done``
    turns true, and what follows drops.

    A call streams as it is written: its id and name once the name ends, then
    its ``arguments`` in pieces that add up to their ``json.dumps`` in any
    chunking. A value the schema allows only as a string streams as it comes;
    any other goes whole at ``</arg_value>``. A key given twice is in the JSON
    twice. A reply cut off inside a call keeps the call as far as it got, with
    no ``tool_calls`` finish; cut off before the name ends, the call is text.
    """

    def __init__(self, *, thinking: bool, tools: dict[str, dict] | None, forced_call: str | None = None,
                 single_call: bool = False):
        self._thinking = thinking
        self._single_call = single_call
        self.done = False
        self._state = "reasoning" if thinking else "content"
        self._tools = tools
        self._buf = ""
        # In a call: "name" until it ends, then "args" (between arguments), "key",
        # "open" (past </arg_key>, before <arg_value>) and "value"; "text" for a
        # call to a function not offered.
        self._phase = "name"
        # the call's text up to the last feed, kept while it may still turn out text
        self._raw: list[str] = []
        self._calls = 0
        self._cut = False
        if forced_call is not None:
            self._state, self._raw = "call", [forced_call]
        self._props: dict = {}
        self._nargs = 0
        self._key: list[str] = []
        self._string = False
        self._value: list[str] = []
        self._content_started = False
        self._held_space = ""

    @property
    def reasoning(self) -> bool:
        """Whether the text read so far leaves the reply inside its reasoning."""
        return self._state == "reasoning"

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self._calls and not self._cut else "stop"

    def feed(self, text: str) -> list[dict]:
        if self.done:
            return []
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
        calls: dict[int, dict] = {}
        for d in deltas:
            for c in d.get("tool_calls", []):
                if "id" in c:
                    calls[c["index"]] = {"id": c["id"], "type": "function", "function": dict(c["function"])}
                else:
                    calls[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
        message: dict = {"role": "assistant", "content": content or (None if calls else "")}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = list(calls.values())
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
                if not self._drain_call(out, final):
                    break
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
                self._state, self._phase = "call", "name"
        return _merged(out)

    def _drain_call(self, out: list[dict], final: bool) -> bool:
        """Reads what it can of the call; False once it needs more text, or at
        the end of the reply. What is set aside or sent is never searched
        again, so a long call is read once."""
        buf, phase = self._buf, self._phase
        if phase == "open":
            rest = buf.lstrip()
            if rest.startswith(ARG_VALUE):
                self._buf = rest[len(ARG_VALUE):]
                self._open_value(out)
                return True
            self._buf = rest
            if ARG_VALUE.startswith(rest) and not final:
                return False
            self._phase = "args"  # no value follows: the key drops
            return True
        tags = _CALL_TAGS[phase]
        hits = [(i, t) for t in tags if (i := buf.find(t)) >= 0]
        if not hits:
            if final:
                self._end_cut_off(out)
                return False
            keep = _partial_tag(buf, tags)
            self._buf = buf[len(buf) - keep:]
            self._take(out, buf[:len(buf) - keep])
            return False
        i, tag = min(hits)
        self._buf = buf[i + len(tag):]
        if phase == "name":
            name = ("".join(self._raw) + buf[:i]).strip()
            if name not in (self._tools or {}):
                self._raw.append(buf[:i])
                self._buf = buf[i:]  # the call's text, tags and all
                self._phase = "text"
                return True
            self._start_call(out, name)
        else:
            self._take(out, buf[:i])
        if tag == ARG_KEY:
            self._phase, self._key = "key", []
        elif tag == ARG_KEY_END:
            self._phase = "open"
        elif tag == ARG_VALUE_END:
            self._close_value(out)
            self._phase = "args"
        elif phase == "text":
            self._as_text(out, CALL_START + "".join(self._raw) + CALL_END)
        else:
            if phase == "value":
                self._close_value(out)  # the call closed it
            self._args(out, "}" if self._nargs else "{}")
            self._state = "content"
            if self._single_call:
                self.done, self._buf = True, ""
                return False
        return True

    def _take(self, out: list[dict], text: str) -> None:
        """Text read in the current phase of a call."""
        if self._phase in ("name", "text"):
            self._raw.append(text)
        elif self._phase == "key":
            self._key.append(text)
        elif self._phase == "value":
            if self._string:
                self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
            else:
                self._value.append(text)
        # between arguments, text drops

    def _start_call(self, out: list[dict], name: str) -> None:
        self._raw, self._props, self._nargs = [], self._tools[name] or {}, 0
        self._held_space = ""
        index, self._calls = self._calls, self._calls + 1
        out.append({"tool_calls": [{
            "index": index,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": ""},
        }]})
        self._phase = "args"

    def _open_value(self, out: list[dict]) -> None:
        key = "".join(self._key).strip()
        types = _json_types(self._props.get(key))
        self._string = bool(types) and types <= {"string", "null"}
        self._value = []
        opening = ("{" if not self._nargs else ", ") + json.dumps(key, ensure_ascii=False) + ": "
        self._args(out, opening + ('"' if self._string else ""))
        self._nargs += 1
        self._phase = "value"

    def _close_value(self, out: list[dict]) -> None:
        if self._string:
            self._args(out, '"')
            return
        key = "".join(self._key).strip()
        self._args(out, json.dumps(_arg_value("".join(self._value), self._props.get(key)), ensure_ascii=False))

    def _args(self, out: list[dict], text: str) -> None:
        if text:
            out.append({"tool_calls": [{"index": self._calls - 1, "function": {"arguments": text}}]})

    def _end_cut_off(self, out: list[dict]) -> None:
        """The reply ended inside the call."""
        if self._phase in ("name", "text"):  # its text is all there is
            text, self._buf = CALL_START + "".join(self._raw) + self._buf, ""
            self._as_text(out, text)
            return
        if self._phase == "value" and self._string:
            self._take(out, self._buf)
        self._buf, self._cut, self._state = "", True, "content"

    def _as_text(self, out: list[dict], text: str) -> None:
        self._raw, self._state = [], "content"
        self._emit(out, text)

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


def _merged(deltas: list[dict]) -> list[dict]:
    """Adjacent deltas of one kind as one: text joins, and a call's argument
    pieces join onto the delta before them."""
    out: list[dict] = []
    for d in deltas:
        key = next(iter(d))
        prev = out[-1] if out else None
        if prev is None or set(prev) != {key}:
            out.append(d)
        elif key != "tool_calls":
            out[-1] = {key: prev[key] + d[key]}
        elif "id" in d["tool_calls"][0] or prev["tool_calls"][0]["index"] != d["tool_calls"][0]["index"]:
            out.append(d)
        else:
            (call,) = prev["tool_calls"]
            fn = call["function"]
            arguments = fn["arguments"] + d["tool_calls"][0]["function"]["arguments"]
            out[-1] = {"tool_calls": [{**call, "function": {**fn, "arguments": arguments}}]}
    return out
