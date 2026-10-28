"""Tool calls in a chat template's own markup, and the reasoning around them.

A template family writes a call in markup of its own (:class:`CallFormat`),
named as vLLM's ``--tool-call-parser`` names it:

* ``glm47`` (GLM-4.7 to GLM-5.3):
  ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>``;
* ``qwen3_coder`` (Qwen3-Coder, Qwen3-Coder-Next, Qwen3.5 to Qwen3.8):
  ``<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n...</function>\\n</tool_call>``.

Both write a string value bare and any other as JSON, which some Qwen
templates spell as Python (``True``, ``False``, ``None``), so one parser
reads every family (:class:`ReplyParser`). An adapter names its family's
format in a :class:`ChatTools`, which takes the request's messages, tools and
``tool_choice`` to the model's chat template and makes the parser.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass

THINK_START, THINK_END = "<think>", "</think>"


@dataclass(frozen=True)
class CallFormat:
    """How a template family writes one call: ``call_start``, then
    ``name_start NAME name_end``, then per argument ``key_start KEY key_end
    value_start VALUE value_end``, then ``body_end`` and ``call_end``.

    With no ``name_end`` the name runs to the first ``key_start`` or the
    call's end; with no ``value_start`` the value follows the key's end.
    ``wrapped_values``: the template writes a newline after the key and one
    before ``value_end``, which are not the value's. ``python_literals``:
    ``True``, ``False`` and ``None`` are the template's spelling of a JSON
    ``true``, ``false`` and ``null`` (Jinja's ``string`` filter).
    ``forced`` and ``forced_any`` are what a prompt ends with to open the
    reply inside a call to a named function (``{name}``), or to one the
    model names. ``bare_start`` opens a call the model wrote without
    ``call_start``: Qwen3-Coder writes ``<function=...>`` straight after some
    answer text at times, still closing it with ``</tool_call>`` (vLLM's
    ``qwen3_coder`` reads it as a call too).
    """

    name: str
    call_start: str
    call_end: str
    key_start: str
    key_end: str
    value_end: str
    forced: str
    forced_any: str
    name_start: str = ""
    name_end: str = ""
    value_start: str = ""
    body_end: str = ""
    wrapped_values: bool = False
    python_literals: bool = False
    bare_start: str = ""

    def forced_prefix(self, name: str) -> str:
        return self.forced.format(name=name) if name else self.forced_any


GLM47 = CallFormat(
    "glm47", call_start="<tool_call>", call_end="</tool_call>", key_start="<arg_key>", key_end="</arg_key>",
    value_start="<arg_value>", value_end="</arg_value>", forced="<tool_call>{name}", forced_any="<tool_call>",
)
QWEN3_CODER = CallFormat(
    "qwen3_coder", call_start="<tool_call>", call_end="</tool_call>", name_start="<function=", name_end=">",
    key_start="<parameter=", key_end=">", value_end="</parameter>", body_end="</function>",
    wrapped_values=True, python_literals=True, bare_start="<function=",
    # the template's layout up to the arguments: the name is closed, so the call is the named one's
    forced="<tool_call>\n<function={name}>\n", forced_any="<tool_call>\n",
)
CALL_FORMATS = {f.name: f for f in (GLM47, QWEN3_CODER)}

def _phase_tags(f: CallFormat) -> dict[str, tuple[str, ...]]:
    """What each phase of reading a call ends at (see ``ReplyParser._drain_call``)."""
    return {
        "head": (f.name_start, f.call_end),
        "name": (f.name_end or f.key_start, f.call_end),
        "args": tuple(t for t in (f.key_start, f.body_end, f.call_end) if t),
        "key": (f.key_end, f.call_end),
        "value": (f.value_end, f.call_end),
        "tail": (f.call_end,),
        "text": (f.call_end,),
    }


# ---------------------------------------------------------------------------
# the request: messages and tools as a template reads them
# ---------------------------------------------------------------------------

# a content part's modality, as flatten_messages reads it: the key holding its reference
_MEDIA_PARTS = {"image_url": ("image", "image_url"), "video_url": ("video", "video_url"),
                "audio_url": ("audio", "audio_url"), "input_audio": ("audio", "input_audio")}


def template_messages(messages, *, flatten_content: bool = False) -> list[dict]:
    """The messages with their roles, tool calls and tool results. The template
    iterates an assistant call's ``arguments``, which OpenAI clients send as a
    JSON string. A null field is left out: GLM-5.2's template prints ``None``.
    ``developer`` turns are ``system`` turns; the templates know only the latter.

    ``flatten_content`` is for templates that concatenate ``content``: it
    becomes a string, the text parts a line each, or with an attachment a
    list of text items and empty ``{"type": modality, modality: ""}`` items,
    where the template writes the attachment's placeholder (as
    ``messages_from_parts`` writes them); a missing one is ``""``.
    """
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
        if flatten_content:
            msg["content"] = _flat_content(msg.get("content"))
        out.append(msg)
    return out


def _flat_content(content):
    if content is None or isinstance(content, str):
        return content or ""
    items: list = []
    for part in content if isinstance(content, list) else []:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            # flatten_messages joins a message's text parts a line apart, and skips empty ones
            if part.get("text"):
                if items and isinstance(items[-1], str):
                    items[-1] += "\n" + part["text"]
                else:
                    items.append(part["text"])
        elif kind in _MEDIA_PARTS:
            modality, key = _MEDIA_PARTS[kind]
            ref = part.get(key) or {}
            if ref.get("data" if kind == "input_audio" else "url"):
                items.append({"type": modality, modality: ""})
    if all(isinstance(x, str) for x in items):
        return "\n".join(items)
    return [{"type": "text", "text": x} if isinstance(x, str) else x for x in items]


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


@dataclass(frozen=True)
class ChatTools:
    """Tool calls for a chat adapter whose model renders the request's own
    messages through its chat template: what reaches ``process_prompt`` and
    the parser of the reply.

    ``thinking`` is ``"never"`` (no reasoning block), ``"optional"`` (on
    unless ``enable_thinking`` is false) or ``"always"`` (turning it off is a
    400). ``template_kwargs`` are the template's switches: taken flat or from
    ``chat_template_kwargs``, others dropped, as vLLM drops the ones a
    template does not read. ``efforts`` are the template's ``reasoning_effort``
    levels, least first (``/v1/responses`` maps ``reasoning.effort`` onto them).

    ``tool_choice`` is ``"auto"`` (default), ``"none"``, which leaves the tools
    out of the prompt, ``"required"`` or a named function. With no grammar, a
    forced call is the prompt's: it ends inside the call
    (``CallFormat.forced_prefix``), so the reply opens there. ``"required"``
    over one tool names it; over several the model picks, and a name it makes
    up stays text. The call opens the reply, so a forced call has no
    reasoning: an optional thinker's prompt ends as with thinking off, an
    always-thinking one's closes the empty reasoning, and an explicit
    ``enable_thinking`` with it is a 400. ``parallel_tool_calls: false`` ends
    the reply, and the request, when its first call ends.
    """

    call_format: CallFormat
    thinking: str
    template_kwargs: tuple[str, ...] = ()
    efforts: tuple[str, ...] = ()
    flatten_content: bool = False

    def __post_init__(self):
        if self.thinking not in ("never", "optional", "always"):
            raise ValueError(f"thinking is never, optional or always; got {self.thinking!r}")

    def apply(self, req, mk: dict) -> None:
        """Puts what the template reads into ``mk``: ``messages``, ``tools``,
        ``chat_template_kwargs`` and, for a forced call, ``reply_prefix``."""
        tools, template_kwargs, forced = self.options(mk)
        thinking = template_kwargs.get("enable_thinking")
        if self.thinking == "always" and thinking is not None and not thinking:
            # the template has no switch, so off would still reason
            raise ValueError("this model always reasons; enable_thinking cannot turn it off")
        mk["messages"] = template_messages(req.messages, flatten_content=self.flatten_content)
        if tools:
            mk["tools"] = tools
        if forced is not None:
            # an always-thinking prompt ends inside <think>: an empty reasoning, as the
            # template renders a turn without one
            mk["reply_prefix"] = (THINK_END if self.thinking == "always" else "") + \
                self.call_format.forced_prefix(forced)
        mk["chat_template_kwargs"] = template_kwargs

    def parser(self, extra: dict) -> ReplyParser | None:
        """The parser of the reply to a request whose extra fields are ``extra``,
        None when the reply has no calls or reasoning to split out."""
        mk = dict(extra)
        single_call = mk.get("parallel_tool_calls") is False
        tools, template_kwargs, forced = self.options(mk)
        # truthiness, as the templates read it
        thinking = self.thinking == "always" or (
            self.thinking == "optional" and bool(template_kwargs.get("enable_thinking", True)))
        params = tool_parameters(tools) if tools else None
        if params is None and not thinking:
            return None  # nothing in the reply to split: it is all answer, as without a parser
        if forced:
            # the prompt opened a call to this function: a name the model goes on to write
            # (get_weather -> get_weather_hourly) is not another tool's call
            params = {forced: params[forced]}
        return ReplyParser(self.call_format, thinking=thinking, tools=params, forced_call=forced,
                           single_call=single_call)

    def options(self, mk: dict) -> tuple[list | None, dict, str | None]:
        """Takes ``tools``, ``tool_choice`` and the template's keys out of ``mk``.
        The third is the call a forced ``tool_choice`` opens the reply with:
        the function's name, or ``""`` for the model to name one."""
        if "reply_prefix" in mk:
            raise ValueError("reply_prefix is not a request field; tool_choice sets it")
        # the old names, which no client needs any more: refused where they ask for a
        # call, rather than answered with text
        functions, function_call = mk.pop("functions", None), mk.pop("function_call", None)
        if (functions not in (None, []) and function_call != "none") or function_call not in (None, "none", "auto"):
            raise ValueError("functions and function_call are not served; send tools and tool_choice")
        # false ends the reply at its first call (the parser's single_call); read, not sent on
        parallel = mk.pop("parallel_tool_calls", None)
        if parallel is not None and not isinstance(parallel, bool):
            raise ValueError(f"parallel_tool_calls must be a boolean, got {parallel!r}")
        tools = mk.pop("tools", None) or None
        choice = mk.pop("tool_choice", None)
        names = list(tool_parameters(tools)) if tools is not None else []
        template_kwargs = mk.pop("chat_template_kwargs", None) or {}
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        template_kwargs = {k: v for k, v in template_kwargs.items() if k in self.template_kwargs}
        for key in self.template_kwargs:
            if key in mk:
                template_kwargs.setdefault(key, mk.pop(key))
        forced = None
        if choice == "none":
            tools = None
        elif choice == "required" or isinstance(choice, dict):
            if not names:
                raise ValueError(f"tool_choice {choice!r} needs tools")
            if choice == "required":
                forced = names[0] if len(names) == 1 else ""
            else:
                fn = choice.get("function")
                forced = fn.get("name") if choice.get("type") == "function" and isinstance(fn, dict) else None
                if forced not in names:
                    raise ValueError(f"tool_choice {choice!r} names no function in tools {names}")
            if self.thinking != "never" and template_kwargs.get("enable_thinking"):
                raise ValueError("a forced tool call opens the reply, which leaves no room to reason; "
                                 "leave enable_thinking unset, or use tool_choice 'auto'")
            if self.thinking == "optional":
                # the prompt ends as a turn that does not reason
                template_kwargs["enable_thinking"] = False
        elif choice not in (None, "auto"):
            raise ValueError(f"tool_choice takes 'auto', 'none', 'required' or "
                             f"{{'type': 'function', 'function': {{'name': ...}}}}; got {choice!r}")
        return tools, template_kwargs, forced


# ---------------------------------------------------------------------------
# argument values
# ---------------------------------------------------------------------------

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


_PYTHON_LITERALS = {"True": "true", "False": "false", "None": "null"}


def arg_value(raw: str, schema, *, python_literals: bool = False):
    """A string value arrives bare and any other as JSON; a value the schema
    allows only as a string stays one ("94110" for a zip code).
    ``python_literals`` reads ``True``, ``False`` and ``None`` as JSON's."""
    types = _json_types(schema)
    if types and types <= {"string", "null"}:
        return raw
    text = _PYTHON_LITERALS.get(raw.strip(), raw) if python_literals else raw
    try:
        # NaN and Infinity are not JSON: the arguments string must stay parseable
        value = json.loads(text, parse_constant=_no_constant)
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


# ---------------------------------------------------------------------------
# the reply
# ---------------------------------------------------------------------------

def _partial_tag(text: str, tags) -> int:
    """Length of the longest end of ``text`` that may begin one of ``tags``."""
    start = text.rfind("<", max(0, len(text) - max(map(len, tags))))
    while start >= 0:
        tail = text[start:]
        if any(t.startswith(tail) for t in tags):
            return len(tail)
        start = text.find("<", start + 1)
    return 0


class ReplyParser:
    """Splits a streamed reply into OpenAI deltas: ``reasoning_content``,
    ``content`` and ``tool_calls``, the calls in ``call_format``'s markup.
    ``feed`` holds back text that may be the start of a tag; ``finish``
    flushes the rest.

    As vLLM's parsers: ``<think>`` in the answer reopens the reasoning, a
    stray ``<think>`` in the reasoning and ``</think>`` in the answer drop, a
    call may end the reasoning without ``</think>``, and a call to a function
    the request did not offer stays text. Content is stripped at both ends,
    as the templates strip it on the way back.
    ``forced_call`` says the prompt opened a call (a forced ``tool_choice``,
    ``CallFormat.forced_prefix``): the reply starts inside it.
    ``single_call`` (``parallel_tool_calls: false``) ends the reply at the end
    of its first call: ``done`` turns true, and what follows drops.

    A call streams as it is written: its id and name once the name ends, then
    its ``arguments`` JSON in pieces: ``{"key": `` as a value opens, a value
    the schema allows only as a string JSON-escaped as it comes, any other
    whole at its end, and ``}`` at the call's end. The pieces add up to the
    ``json.dumps`` of the arguments, in any chunking. A value ends only at
    the format's ``value_end`` (or the call's ``call_end``), so markup inside
    it (``<div>``, ``</function>``) is the value's. Text between arguments
    drops, and so does a key with no value; a key given twice is in the JSON
    twice. A reply cut off inside a call keeps the call as far as it got (its
    ``arguments`` not JSON), and the parser then reports no ``tool_calls``
    finish; cut off before the name ends, the call is text, as is a call to a
    function not offered.
    """

    def __init__(self, call_format: CallFormat, *, thinking: bool, tools: dict[str, dict] | None,
                 forced_call: str | None = None, single_call: bool = False):
        self._format = call_format
        self._ends = _phase_tags(call_format)
        self._thinking = thinking
        self._single_call = single_call
        self.done = False
        self._state = "reasoning" if thinking else "content"
        self._tools = tools
        self._buf = ""
        # In a call: "head" before the name (a format with a name_start), "name"
        # until it ends, then "args" (between arguments), "key", "open" (past the
        # key, before a value_start), "value" and "tail" (past the body_end, before
        # the call_end); "text" for a call to a function not offered.
        self._phase = "name"
        # the call's text up to the last feed, kept while it may still turn out text
        self._raw: list[str] = []
        self._name: list[str] = []
        self._calls = 0
        self._cut = False
        self._props: dict = {}
        self._nargs = 0
        self._key: list[str] = []
        self._string = False
        self._value: list[str] = []
        # a wrapped value's opening newline is still to come
        self._lead = False
        # the call opened at the format's bare_start, without call_start
        self._bare = False
        self._content_started = False
        self._held_space = ""
        self._opening: list[dict] = []
        if forced_call is not None:
            # the prompt's end, read as the reply's start: what it sends goes out with the first feed
            self._opening = self.feed(call_format.forced_prefix(forced_call))

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
        calls = (self._format.call_start,) if self._tools else ()
        if self._state == "reasoning":
            return (THINK_START, THINK_END, *calls)
        if self._tools and self._format.bare_start:
            calls = (*calls, self._format.bare_start)
        return (THINK_START, THINK_END, *calls) if self._thinking else calls

    def _drain(self, final: bool) -> list[dict]:
        out, self._opening = self._opening, []
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
            elif tag == self._format.call_start:
                self._state, self._raw, self._name, self._bare = "call", [], [], False
                self._phase = "head" if self._format.name_start else "name"
            elif tag == self._format.bare_start:
                self._state, self._raw, self._name, self._bare = "call", [tag], [], True
                self._phase = "name"
        return _merged(out)

    def _drain_call(self, out: list[dict], final: bool) -> bool:
        """Reads what it can of the call; False once it needs more text, or at
        the end of the reply. What is set aside or sent is never searched
        again, so a long call is read once."""
        f, buf, phase = self._format, self._buf, self._phase
        if phase == "tail" and self._bare:
            # a bare call may still close with call_end; anything else is answer again
            rest = buf.lstrip()
            if rest.startswith(f.call_end):
                self._buf, self._state = rest[len(f.call_end):], "content"
            elif rest and f.call_end.startswith(rest) and not final or not rest and not final:
                return False
            else:
                self._state = "content"
            return True
        if phase == "open":
            rest = buf.lstrip()
            if rest.startswith(f.value_start):
                self._buf = rest[len(f.value_start):]
                self._open_value(out)
                return True
            self._buf = rest
            if f.value_start.startswith(rest) and not final:
                return False
            self._phase = "args"  # no value follows: the key drops
            return True
        tags = self._ends[phase]
        hits = [(i, t) for t in tags if (i := buf.find(t)) >= 0]
        if not hits:
            if final:
                self._end_cut_off(out)
                return False
            keep = _partial_tag(buf, tags)
            if phase == "value" and f.wrapped_values and buf[:len(buf) - keep].endswith("\n"):
                keep += 1  # it may be the newline the template writes before value_end
            self._buf = buf[len(buf) - keep:]
            self._take(out, buf[:len(buf) - keep])
            return False
        i, tag = min(hits)
        self._buf = buf[i + len(tag):]
        text = buf[:i]
        if phase == "name":
            name = ("".join(self._name) + text).strip()
            if name not in (self._tools or {}):
                self._raw.append(text)
                self._buf = buf[i:]  # the call's text, tags and all
                if self._bare:  # no call_end to read to: the answer goes on from here
                    self._as_text(out, "".join(self._raw))
                    return True
                self._phase = "text"
                return True
            self._start_call(out, name)
        else:
            if phase == "value" and f.wrapped_values and text.endswith("\n"):
                text = text[:-1]
            self._take(out, text)
        if tag == f.call_end:
            return self._close_call(out, phase)
        if phase == "head":  # name_start
            self._raw.append(tag)
            self._phase = "name"
        elif tag == f.key_start:
            self._phase, self._key = "key", []
        elif phase == "key":  # key_end
            if f.value_start:
                self._phase = "open"
            else:
                self._open_value(out)
        elif phase == "value":  # value_end
            self._close_value(out)
            self._phase = "args"
        elif tag == f.body_end:
            # the call's arguments are whole; what follows up to call_end drops
            return self._end_arguments(out, "tail")
        return True

    def _close_call(self, out: list[dict], phase: str) -> bool:
        """``call_end`` read in ``phase``."""
        if phase in ("head", "text"):
            self._as_text(out, self._format.call_start + "".join(self._raw) + self._format.call_end)
            return True
        if phase == "tail":
            self._state = "content"
            return True
        if phase == "value":
            self._close_value(out)  # the call closed it
        return self._end_arguments(out, None)

    def _end_arguments(self, out: list[dict], then: str | None) -> bool:
        """Closes the call's JSON; ``then`` is the phase that reads on to
        ``call_end``, or None when the call has ended. False once the reply
        is done (``single_call``)."""
        self._args(out, "}" if self._nargs else "{}")
        if then is None or self._single_call:
            self._state = "content"
        else:
            self._phase = then
        if self._single_call:
            self.done, self._buf = True, ""
            return False
        return True

    def _take(self, out: list[dict], text: str) -> None:
        """Text read in the current phase of a call."""
        if self._phase in ("head", "text"):
            self._raw.append(text)
        elif self._phase == "name":
            self._raw.append(text)
            self._name.append(text)
        elif self._phase == "key":
            self._key.append(text)
        elif self._phase == "value":
            if self._lead and text:
                self._lead = False
                if text[0] == "\n":
                    text = text[1:]
            if self._string:
                self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
            else:
                self._value.append(text)
        # between arguments and past the body, text drops

    def _start_call(self, out: list[dict], name: str) -> None:
        self._raw, self._name, self._props, self._nargs = [], [], self._tools[name] or {}, 0
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
        self._lead = self._format.wrapped_values
        opening = ("{" if not self._nargs else ", ") + json.dumps(key, ensure_ascii=False) + ": "
        self._args(out, opening + ('"' if self._string else ""))
        self._nargs += 1
        self._phase = "value"

    def _close_value(self, out: list[dict]) -> None:
        if self._string:
            self._args(out, '"')
            return
        key = "".join(self._key).strip()
        raw = "".join(self._value)
        value = arg_value(raw, self._props.get(key), python_literals=self._format.python_literals)
        self._args(out, json.dumps(value, ensure_ascii=False))

    def _args(self, out: list[dict], text: str) -> None:
        if text:
            out.append({"tool_calls": [{"index": self._calls - 1, "function": {"arguments": text}}]})

    def _end_cut_off(self, out: list[dict]) -> None:
        """The reply ended inside the call."""
        if self._phase in ("head", "name", "text"):  # its text is all there is
            opening = "" if self._bare else self._format.call_start
            text, self._buf = opening + "".join(self._raw) + self._buf, ""
            self._as_text(out, text)
            return
        if self._phase == "tail":  # the arguments were whole
            self._buf, self._state = "", "content"
            return
        if self._phase == "value" and self._string:
            self._take(out, self._buf)
        self._buf, self._cut, self._state = "", True, "content"

    def _as_text(self, out: list[dict], text: str) -> None:
        self._raw, self._name, self._state = [], [], "content"
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
