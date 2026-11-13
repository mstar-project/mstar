"""A chat reply split into reasoning, content and tool calls, with the calls'
markup as data.

A template family writes a call in markup of its own (:class:`CallFormat`),
named as vLLM's ``--tool-call-parser`` names it:

* ``glm47`` (GLM-4.7 to GLM-5.3):
  ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>``;
* ``deepseek_v41`` (DeepSeek-V4.1's DSML): a ``<｜DSML｜ calls>`` block of
  ``<｜DSML｜ invoke name="NAME">`` calls, each argument a
  ``<｜DSML｜ parameter name="K" string="true">V</｜DSML｜ parameter>``.

Each writes a string value bare and any other as JSON, so one state machine
reads them all (:class:`ReplyParser`). It is ``glm_chat.GlmReplyParser`` with
the tags taken from the format: GLM still serves through its own, and
``test_chat_reply_corpus`` holds the two to the same deltas on every ``glm47``
reply.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

# what a value is typed by, and how deltas join: the same for every format
from mstar.api_server.openai.glm_chat import _arg_value, _json_types, _merged, _partial_tag

THINK_START, THINK_END = "<think>", "</think>"


class ValueType(Enum):
    """What a value's opening says the value is (``CallFormat.value_openings``)."""

    TEXT = "text"  # the text as written
    JSON = "json"  # a JSON value; one that is not JSON stays the text
    SCHEMA = "schema"  # as the tool's schema says


class ValueOpening(NamedTuple):
    tag: str
    type: ValueType


@dataclass(frozen=True)
class CallFormat:
    """How a template family writes one call: ``call_start NAME``, then per
    argument ``key_start KEY key_end value_start VALUE value_end``, then
    ``call_end``.

    With no ``name_end`` the name runs to the first ``key_start`` or the
    call's end. ``value_openings``, in place of ``value_start``, are the tags a
    value may follow its key with, each saying its type (DeepSeek's
    ``string="true"``). ``section_start`` and ``section_end`` are a block the
    calls sit in: not text, and its start ends the reasoning, as a call does.

    ``forced`` and ``forced_any`` are what a prompt ends with to open the
    reply inside a call to a named function (``{name}``), or to one the model
    names; ``forced_empty`` is ``forced`` for a function that takes no
    arguments, where the template lays an empty call out differently.
    """

    name: str
    call_start: str
    call_end: str
    key_start: str
    key_end: str
    value_end: str
    forced: str
    forced_any: str
    name_end: str = ""
    value_start: str = ""
    value_openings: tuple[ValueOpening, ...] = ()
    section_start: str = ""
    section_end: str = ""
    forced_empty: str = ""

    def forced_prefix(self, name: str, takes_arguments: bool = True) -> str:
        if not name:
            return self.forced_any
        return (self.forced if takes_arguments or not self.forced_empty else self.forced_empty).format(name=name)


GLM47 = CallFormat(
    "glm47", call_start="<tool_call>", call_end="</tool_call>", key_start="<arg_key>", key_end="</arg_key>",
    value_start="<arg_value>", value_end="</arg_value>", forced="<tool_call>{name}", forced_any="<tool_call>",
)
DEEPSEEK_V41 = CallFormat(
    # a name and a key end at their attribute's closing quote; the ">" after a name is text
    # between arguments, which drops
    "deepseek_v41", call_start='<｜DSML｜ invoke name="', name_end='"', call_end="</｜DSML｜ invoke>",
    key_start='<｜DSML｜ parameter name="', key_end='"', value_end="</｜DSML｜ parameter>",
    value_openings=(ValueOpening('string="true">', ValueType.TEXT), ValueOpening('string="false">', ValueType.JSON),
                    ValueOpening(">", ValueType.SCHEMA)),
    section_start="<｜DSML｜ calls>", section_end="</｜DSML｜ calls>",
    # the encoder's layout up to the first parameter, after the prompt's </think>. Each ends on
    # a token boundary of the turn the encoder writes: '">\n' before a parameter and '">\n\n'
    # in a call without any are single tokens
    forced='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n',
    forced_empty='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n\n',
    forced_any='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="',
)
CALL_FORMATS = {f.name: f for f in (GLM47, DEEPSEEK_V41)}


class ReplyParser:
    """Splits a streamed reply into OpenAI deltas: ``reasoning_content``,
    ``content`` and ``tool_calls``, the calls in ``call_format``'s markup.
    ``feed`` holds back text that may be the start of a tag; ``finish``
    flushes the rest.

    As vLLM's parsers: ``<think>`` in the answer reopens the reasoning, a
    stray ``<think>`` in the reasoning and ``</think>`` in the answer drop, a
    call may end the reasoning without ``</think>``, and a call to a function
    the request did not offer stays text. Content is stripped at both ends.
    ``reply_prefix`` is what the prompt already holds of the reply (a forced
    call's opening, ``CallFormat.forced_prefix``): read first, as the reply's
    start. ``single_call`` ends the reply at the end of its first call:
    ``done`` turns true, and what follows drops.

    A call streams as it is written: its id and name once the name ends, then
    its ``arguments`` in pieces that add up to their ``json.dumps`` in any
    chunking. A value that is a string (by its opening, else by the schema)
    streams as it comes; any other goes whole at its end. A value ends only at
    ``value_end`` or the call's end, so markup inside it is the value's. A key
    given twice is in the JSON twice. A reply cut off inside a call keeps the
    call as far as it got, with no ``tool_calls`` finish; cut off before the
    name ends, the call is text.
    """

    def __init__(self, call_format: CallFormat, *, thinking: bool, tools: dict[str, dict] | None,
                 reply_prefix: str = "", single_call: bool = False):
        f = self._format = call_format
        self._single_call = single_call
        self.done = False
        self._state = "reasoning" if thinking else "content"
        self._tools = tools
        self._buf = ""
        # the tags read outside a call, by state: a section's end only in the answer
        think = (THINK_START, THINK_END)
        calls = tuple(t for t in (f.call_start, f.section_start) if t) if tools else ()
        answer = calls + ((f.section_end,) if calls and f.section_end else ())
        self._tags = {"reasoning": think + calls, "content": think + answer if thinking else answer}
        # In a call: "name" until it ends, then "args" (between arguments), "key",
        # "open" (past the key, before the value's opening) and "value"; "text" for a
        # call to a function not offered. What ends each:
        self._ends = {"name": (f.name_end or f.key_start, f.call_end), "text": (f.call_end,),
                      "args": (f.key_start, f.call_end), "key": (f.key_end, f.call_end),
                      "value": (f.value_end, f.call_end)}
        self._openings = f.value_openings or (ValueOpening(f.value_start, ValueType.SCHEMA),)
        self._phase = "name"
        # the call's text up to the last feed, kept while it may still turn out text
        self._raw: list[str] = []
        self._calls = 0
        self._cut = False
        self._props: dict = {}
        self._nargs = 0
        self._key: list[str] = []
        self._string = False
        # the value's opening said it is JSON: no schema types it
        self._json = False
        self._value: list[str] = []
        self._content_started = False
        self._held_space = ""
        # what the prefix sends goes out with the first feed
        self._opening: list[dict] = []
        if reply_prefix:
            self._opening = self.feed(reply_prefix)

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

    def _drain(self, final: bool) -> list[dict]:
        f = self._format
        out, self._opening = self._opening, []
        while self._buf or (final and self._state == "call"):
            if self._state == "call":
                if not self._drain_call(out, final):
                    break
                continue
            tags = self._tags[self._state]
            hits = [(i, t) for t in tags if (i := self._buf.find(t)) >= 0]
            if not hits:
                keep = 0 if final or not tags else _partial_tag(self._buf, tags)
                self._emit(out, self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                break
            i, tag = min(hits)
            self._emit(out, self._buf[:i])
            self._buf = self._buf[i + len(tag):]
            if tag == THINK_END:
                self._state = "content"
            elif tag == THINK_START:
                self._state = "reasoning"
            elif tag == f.call_start:
                self._state, self._phase = "call", "name"
            else:  # the section's tags
                self._state = "content"
        return _merged(out)

    def _drain_call(self, out: list[dict], final: bool) -> bool:
        """Reads what it can of the call; False once it needs more text, or at
        the end of the reply. What is set aside or sent is never searched
        again, so a long call is read once."""
        f, buf, phase = self._format, self._buf, self._phase
        if phase == "open":
            rest = buf.lstrip()
            for opening in self._openings:
                if rest.startswith(opening.tag):
                    self._buf = rest[len(opening.tag):]
                    self._open_value(out, opening.type)
                    return True
            self._buf = rest
            if not final and any(o.tag.startswith(rest) for o in self._openings):
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
        if tag == f.call_end:
            if phase == "text":
                self._as_text(out, f.call_start + "".join(self._raw) + f.call_end)
                return True
            if phase == "value":
                self._close_value(out)  # the call closed it
            self._args(out, "}" if self._nargs else "{}")
            self._state = "content"
            if self._single_call:
                self.done, self._buf = True, ""
                return False
        elif tag == f.key_start:
            self._phase, self._key = "key", []
        elif phase == "key":
            self._phase = "open"
        elif phase == "value":
            self._close_value(out)
            self._phase = "args"
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

    def _open_value(self, out: list[dict], kind: ValueType) -> None:
        key = "".join(self._key).strip()
        if kind is ValueType.SCHEMA:
            types = _json_types(self._props.get(key))
            self._string = bool(types) and types <= {"string", "null"}
        else:
            self._string = kind is ValueType.TEXT
        self._json = kind is ValueType.JSON
        self._value = []
        opening = ("{" if not self._nargs else ", ") + json.dumps(key, ensure_ascii=False) + ": "
        self._args(out, opening + ('"' if self._string else ""))
        self._nargs += 1
        self._phase = "value"

    def _close_value(self, out: list[dict]) -> None:
        if self._string:
            self._args(out, '"')
            return
        schema = None if self._json else self._props.get("".join(self._key).strip())
        self._args(out, json.dumps(_arg_value("".join(self._value), schema), ensure_ascii=False))

    def _args(self, out: list[dict], text: str) -> None:
        if text:
            out.append({"tool_calls": [{"index": self._calls - 1, "function": {"arguments": text}}]})

    def _end_cut_off(self, out: list[dict]) -> None:
        """The reply ended inside the call."""
        if self._phase in ("name", "text"):  # its text is all there is
            text, self._buf = self._format.call_start + "".join(self._raw) + self._buf, ""
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
