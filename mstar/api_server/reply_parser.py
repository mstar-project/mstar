"""A chat reply's text, split as it streams into its reasoning, its answer and its tool calls.

One parser per chat request, fed each text chunk in order on the API server's message thread
(`APIServer`), so both frontends get the same pieces: the answer as the chunk's text, the
reasoning under ``metadata["reasoning"]``, each finished call under ``metadata["tool_calls"]``.
A model that reasons declares where its reasoning ends (`TextOutput.reasoning_end`); the data
worker, which built the prompt, says whether the reply starts inside the reasoning. A model that
calls tools declares its call format (`Model.TOOL_FORMAT`, a key of `TOOL_FORMATS`).
"""

from __future__ import annotations

import json
import re
import uuid

from mstar.utils.utf8_stream import Utf8Stream

# where a chunk carries the reasoning text, and the finished tool calls, the parser split off it
REASONING = "reasoning"
TOOL_CALLS = "tool_calls"
# on a request's first text chunk, from the data worker: whether the reply starts inside its reasoning
REPLY_IN_REASONING = "reply_in_reasoning"


class ToolCallFormat:
    """How a model writes a tool call: the text that opens one (``begin``) and closes it (``end``),
    which `ReplyParser` finds as the reply streams, and `parse`, which reads the text between them.
    One subclass per format; the markers are the ones xgrammar's structural tag for the model's
    format holds a forced call to (`grammar.tool_call_grammar`)."""

    begin: str
    end: str

    def parse(self, block: str, schemas: dict[str, dict]) -> tuple[str, dict] | None:
        """A closed call's name and arguments, from the text between its markers; ``schemas`` maps
        each tool's name to its parameters' schemas, by name. None for text that isn't a call."""
        raise NotImplementedError


class QwenXmlFormat(ToolCallFormat):
    """Qwen3.5's (xgrammar's style ``qwen_xml``): the function's name, then one
    ``<parameter=NAME>`` block per argument, its value as text, typed by the tool's schema."""

    begin = "<tool_call>\n<function="
    name_end = ">\n"
    end = "</function>\n</tool_call>"
    _param = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)

    def parse(self, block: str, schemas: dict[str, dict]) -> tuple[str, dict] | None:
        name, sep, params = block.partition(self.name_end)
        if not sep or not name or "\n" in name:
            return None
        types = schemas.get(name, {})
        return name, {key: _typed(value, types.get(key)) for key, value in self._param.findall(params)}


# `Model.TOOL_FORMAT` -> its format
TOOL_FORMATS: dict[str, ToolCallFormat] = {
    # Qwen3.5's chat template and xgrammar's "qwen_3_5" tag
    "qwen_3_5": QwenXmlFormat(),
}


class ReplyParser:
    """``feed`` takes a text chunk's bytes and returns ``(reasoning, answer, calls)``: the text certain
    so far on each side, and the calls that closed. Text that could be the start of a marker (the
    reasoning's end, a call's begin), and newlines that may come just before one, are held until
    known; the newlines that open the reasoning, the answer and the text after a call are dropped,
    as the chat template does when it renders a reply back. Calls are found only after the
    reasoning (as vLLM finds them), and only with ``tools``; a call's parameters are typed by its
    tool's schema. `finish` returns what is held when the reply ends: a reply cut inside its
    reasoning keeps it as reasoning, and a call that never closed comes out as text, verbatim
    (no repairs). Outside the reasoning, with no tools, a chunk passes through as text."""

    __slots__ = ("_end", "_in_reasoning", "_held", "_opening", "_thinking", "_utf8", "_format", "_types",
                 "_in_call", "calls")

    def __init__(self, reasoning_end: str | None, starts_in_reasoning: bool,
                 tool_format: ToolCallFormat | None = None, tools: list | None = None):
        self._end = reasoning_end
        self._in_reasoning = starts_in_reasoning and reasoning_end is not None
        self._held = ""
        # dropping the newlines that open the answer, right after the reasoning or a call
        self._opening = False
        # dropping the newlines that open the reasoning
        self._thinking = self._in_reasoning
        self._utf8 = Utf8Stream()
        self._format = tool_format if tools else None
        # tool name -> parameter name -> its schema (a client's malformed one counts as none)
        self._types = {t["function"]["name"]: _properties(t["function"]) for t in tools or ()}
        self._in_call = False
        # how many calls have closed so far: the next one's index
        self.calls = 0

    def feed(self, data: bytes) -> tuple[str, str, list[dict]]:
        text = self._utf8.feed(data)
        if self._in_reasoning:
            # most of a reasoning block: text that can't hold or begin its end marker, and doesn't
            # end on a newline that may come before one, is reasoning as it is
            if not self._held and not self._thinking and self._end[0] not in text and text[-1:] != "\n":
                return text, "", []
        elif not self._opening and self._format is None:
            return "", text, []
        return self._split(text)

    @property
    def passes_through(self) -> bool:
        """Whether every chunk from here on is answer text as it came: past the reasoning and the
        newlines that open the answer, with no tools, and holding nothing (a marker's start, or a
        character's first bytes). The caller can stop feeding the parser then."""
        return not (self._in_reasoning or self._opening or self._format or self._held or self._utf8.pending)

    def finish(self) -> tuple[str, str, list[dict]]:
        text = self._utf8.flush()
        reasoning, answer, calls = self._split(text) if text else ("", "", [])
        if self._in_reasoning:
            reasoning, self._held = reasoning + self._held, ""
        elif self._in_call:
            answer, self._held, self._in_call = answer + self._format.begin + self._held, "", False
        else:
            answer, self._held = answer + self._held, ""
        return reasoning, answer, calls

    def _split(self, text: str) -> tuple[str, str, list[dict]]:
        reasoning = ""
        if self._in_reasoning:
            buf = self._held + text
            if self._thinking:
                buf = buf.lstrip("\n")
                self._thinking = not buf
            at = buf.find(self._end)
            if at < 0:
                self._held, out = _hold(buf, self._end)
                return out, "", []
            self._in_reasoning, self._held, self._opening = False, "", True
            reasoning, text = buf[:at].rstrip("\n"), buf[at + len(self._end):]
        answer, calls = self._answer(text)
        return reasoning, answer, calls

    def _answer(self, text: str) -> tuple[str, list[dict]]:
        fmt = self._format
        if fmt is None:
            return self._open(text), []
        buf = self._held + text
        self._held = ""
        out: list[str] = []
        calls: list[dict] = []
        while True:
            if self._in_call:
                at = buf.find(fmt.end)
                if at < 0:
                    self._held = buf
                    break
                block, buf = buf[:at], buf[at + len(fmt.end):]
                call = self._call(block)
                if call is None:
                    out.append(fmt.begin + block + fmt.end)  # not a call after all: as written
                else:
                    calls.append(call)
                self._in_call, self._opening = False, True
                continue
            buf = self._open(buf)
            at = buf.find(fmt.begin)
            if at < 0:
                self._held, rest = _hold(buf, fmt.begin)
                out.append(rest)
                break
            # the template writes the answer's text, a blank line, then the call
            out.append(buf[:at].rstrip("\n"))
            buf, self._in_call = buf[at + len(fmt.begin):], True
        return "".join(out), calls

    def _call(self, block: str) -> dict | None:
        """A closed call's block, as an OpenAI tool call; None for one the format doesn't read."""
        parsed = self._format.parse(block, self._types)
        if parsed is None:
            return None
        name, args = parsed
        index, self.calls = self.calls, self.calls + 1
        return {"index": index, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}

    def _open(self, text: str) -> str:
        if self._opening:
            text = text.lstrip("\n")
            self._opening = not text
        return text


def _hold(text: str, marker: str) -> tuple[str, str]:
    """``(held, out)``: a tail of ``text`` that may begin ``marker``, and the newlines just before
    it, held back; the rest goes out."""
    cut = len(text) - _marker_prefix(text, marker)
    while cut > 0 and text[cut - 1] == "\n":
        cut -= 1
    return text[cut:], text[:cut]


def _marker_prefix(text: str, marker: str) -> int:
    """The length of the longest tail of ``text`` that is a proper prefix of ``marker``: such a tail
    starts with the marker's first character, so only those starts are tried (usually none)."""
    start = text.find(marker[0], max(0, len(text) - len(marker) + 1))
    while start >= 0:
        if marker.startswith(text[start:]):
            return len(text) - start
        start = text.find(marker[0], start + 1)
    return 0


def _typed(raw: str, schema: dict | None):
    """A parameter's text as its schema's type: a string stays as written (``94110`` included); a
    number, boolean, object, array or null is parsed, Python's ``True`` / ``False`` too (Qwen's
    template prints booleans that way); text that doesn't parse, or parses only as Python's
    ``NaN`` / ``Infinity``, stays a string. No schema: parsed if it parses."""
    kind = schema.get("type") if isinstance(schema, dict) else None
    kinds = kind if isinstance(kind, list) else [kind]
    if kinds == ["string"]:
        return raw
    if raw in ("True", "False") and ("boolean" in kinds or kind is None):
        return raw == "True"
    try:
        return json.loads(raw, parse_constant=_not_json)
    except ValueError:
        return raw


def _not_json(constant: str):
    """``NaN`` and ``Infinity``, which Python's JSON reads but strict JSON doesn't: left as text."""
    raise ValueError(constant)


def _properties(function: dict) -> dict:
    """A tool's parameter schemas by name; empty when its ``parameters`` or their ``properties``
    aren't objects."""
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    return properties if isinstance(properties, dict) else {}
