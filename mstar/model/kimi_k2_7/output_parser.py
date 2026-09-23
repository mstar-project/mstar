"""Kimi-K2.7 streaming output parser.

Turns the model's raw decoded text (with special tokens preserved) into
OpenAI chat delta dicts: ``reasoning_content``, ``content``, ``tool_calls``.

Kimi's native tool-call layout::

    <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0\
<|tool_call_argument_begin|>{"city": "Tokyo"}<|tool_call_end|><|tool_calls_section_end|>
"""
from __future__ import annotations

import re

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_SECTION_START = "<|tool_calls_section_begin|>"
TOOL_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_START = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_ARG_START = "<|tool_call_argument_begin|>"

_ALL_MARKERS = (
    THINK_START, THINK_END, TOOL_SECTION_START, TOOL_SECTION_END,
    TOOL_CALL_START, TOOL_CALL_END, TOOL_ARG_START,
)

_TOOL_ID_RE = re.compile(r"(?P<id>.+:\d+)")

# States
_REASONING = "REASONING"
_CONTENT = "CONTENT"
_TOOL_PREAMBLE = "TOOL_PREAMBLE"  # between tool calls / before the first one
_TOOL_HEADER = "TOOL_HEADER"      # reading id:index before the arg marker
_TOOL_ARGS = "TOOL_ARGS"          # reading raw argument text

_ACTIVE_MARKERS = {
    _REASONING: (THINK_START, THINK_END, TOOL_SECTION_START),
    _CONTENT: (THINK_END, TOOL_SECTION_START),
    _TOOL_PREAMBLE: (TOOL_CALL_START, TOOL_SECTION_END),
    _TOOL_HEADER: (TOOL_ARG_START,),
    _TOOL_ARGS: (TOOL_CALL_END, TOOL_SECTION_END),
}


def _split_trailing_ws(pending: str, chunk: str) -> tuple[str, str]:
    """Combine ``pending`` + ``chunk``, holding back trailing whitespace until it resolves."""
    combined = pending + chunk
    cut = len(combined.rstrip())
    return combined[:cut], combined[cut:]


class KimiOutputParser:
    """Feed decoded text incrementally; get OpenAI chat delta dicts back."""

    def __init__(self, thinking: bool = True):
        self.thinking = thinking
        self._state = _REASONING if thinking else _CONTENT
        self._buf = ""  # tail that might be a marker prefix split across feed()

        self._reasoning_pending = ""

        self._content_pending = ""
        self._content_seen_nonws = False

        self._tool_calls: list[dict] = []  # only successfully-parsed headers
        self._had_tool_call = False

        self._cur_header = ""
        self._cur_index: int | None = None
        self._cur_args_pending = ""
        self._cur_args_started = False
        self._cur_args_emitted = False

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self._had_tool_call else "stop"

    def feed(self, text: str) -> list[dict]:
        data = self._buf + text
        self._buf = ""
        deltas: list[dict] = []
        while data:
            markers = _ACTIVE_MARKERS[self._state]
            idx, marker = None, None
            for m in markers:
                i = data.find(m)
                if i != -1 and (idx is None or i < idx):
                    idx, marker = i, m
            if idx is None:
                hold = self._max_marker_prefix_hold(data, markers)
                emit_upto = len(data) - hold
                if emit_upto > 0:
                    deltas.extend(self._consume_text(data[:emit_upto]))
                self._buf = data[emit_upto:]
                break
            if idx > 0:
                deltas.extend(self._consume_text(data[:idx]))
            deltas.extend(self._consume_marker(marker))
            data = data[idx + len(marker):]
        return deltas

    def finish(self) -> list[dict]:
        deltas: list[dict] = []
        if self._buf:
            deltas.extend(self._consume_text(self._buf))
            self._buf = ""
        if self._state == _REASONING and self._reasoning_pending:
            # Reasoning never ended: flush what was held back verbatim.
            deltas.append({"reasoning_content": self._reasoning_pending})
            self._reasoning_pending = ""
        elif self._state == _CONTENT and not self._content_seen_nonws and self._content_pending:
            # No tool section started: flush pending content as a normal reply.
            deltas.append({"content": self._content_pending})
            self._content_pending = ""
        elif self._state == _TOOL_ARGS:
            deltas.extend(self._finalize_args())
        return deltas

    @staticmethod
    def _max_marker_prefix_hold(data: str, markers: tuple[str, ...]) -> int:
        """Longest suffix of ``data`` that is a proper prefix of one of ``markers``."""
        hold = 0
        for m in markers:
            for k in range(min(len(m) - 1, len(data)), 0, -1):
                if data.endswith(m[:k]):
                    hold = max(hold, k)
                    break
        return hold

    def _consume_text(self, text: str) -> list[dict]:
        if self._state == _REASONING:
            emit, self._reasoning_pending = _split_trailing_ws(self._reasoning_pending, text)
            return [{"reasoning_content": emit}] if emit else []

        if self._state == _CONTENT:
            if self._content_seen_nonws:
                return [{"content": text}] if text else []
            combined = self._content_pending + text
            if combined.strip() == "":
                self._content_pending = combined
                return []
            self._content_seen_nonws = True
            self._content_pending = ""
            return [{"content": combined}]

        if self._state == _TOOL_HEADER:
            self._cur_header += text
            return []

        if self._state == _TOOL_ARGS:
            if self._cur_index is None:
                return []  # a dropped (bad-header) call: consume silently
            if not self._cur_args_started:
                text = text.lstrip()
                if not text:
                    return []
                self._cur_args_started = True
            emit, self._cur_args_pending = _split_trailing_ws(self._cur_args_pending, text)
            if not emit:
                return []
            self._cur_args_emitted = True
            return [{"tool_calls": [{"index": self._cur_index, "function": {"arguments": emit}}]}]

        return []  # _TOOL_PREAMBLE: preamble / between-calls text is dropped

    def _consume_marker(self, marker: str) -> list[dict]:
        if self._state == _REASONING:
            if marker == THINK_END:
                self._reasoning_pending = ""
                self._state = _CONTENT
            elif marker == TOOL_SECTION_START:
                self._reasoning_pending = ""
                self._state = _TOOL_PREAMBLE
            return []  # THINK_START (or a stray) is swallowed

        if self._state == _CONTENT:
            if marker == TOOL_SECTION_START:
                if not self._content_seen_nonws:
                    self._content_pending = ""  # whitespace-only: dropped
                self._state = _TOOL_PREAMBLE
            return []  # a stray THINK_END is swallowed

        if self._state == _TOOL_PREAMBLE:
            if marker == TOOL_CALL_START:
                self._cur_header = ""
                self._state = _TOOL_HEADER
            # TOOL_SECTION_END self-loops so trailing text after it is suppressed.
            return []

        if self._state == _TOOL_HEADER and marker == TOOL_ARG_START:
            deltas = self._finalize_header()
            self._state = _TOOL_ARGS
            return deltas

        if self._state == _TOOL_ARGS and marker in (TOOL_CALL_END, TOOL_SECTION_END):
            deltas = self._finalize_args()
            self._state = _TOOL_PREAMBLE
            return deltas

        return []

    def _finalize_header(self) -> list[dict]:
        header = self._cur_header.strip()
        self._cur_args_pending = ""
        self._cur_args_started = False
        self._cur_args_emitted = False
        match = _TOOL_ID_RE.match(header)
        if not match:
            self._cur_index = None
            return []
        tool_id = match.group("id").strip()
        name = tool_id.split(":")[0].removeprefix("functions.")
        self._cur_index = len(self._tool_calls)
        self._tool_calls.append({"id": tool_id, "name": name})
        self._had_tool_call = True
        return [{
            "tool_calls": [{
                "index": self._cur_index, "id": tool_id, "type": "function",
                "function": {"name": name, "arguments": ""},
            }]
        }]

    def _finalize_args(self) -> list[dict]:
        if self._cur_index is None:
            return []
        index, self._cur_index = self._cur_index, None
        if not self._cur_args_emitted:
            return [{"tool_calls": [{"index": index, "function": {"arguments": "{}"}}]}]
        return []


def _merge_deltas(deltas: list[dict]) -> dict:
    """Fold a delta list (as yielded by feed()/finish()) into one message."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    for delta in deltas:
        if "reasoning_content" in delta:
            reasoning_parts.append(delta["reasoning_content"])
        if "content" in delta:
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls", []):
            slot = tool_calls.setdefault(tc["index"], {
                "id": None, "type": "function",
                "function": {"name": None, "arguments": ""},
            })
            if "id" in tc:
                slot["id"] = tc["id"]
            if "type" in tc:
                slot["type"] = tc["type"]
            fn = tc.get("function", {})
            if "name" in fn:
                slot["function"]["name"] = fn["name"]
            if "arguments" in fn:
                slot["function"]["arguments"] += fn["arguments"]

    return {
        "reasoning_content": "".join(reasoning_parts) or None,
        "content": "".join(content_parts) or None,
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)] or None,
    }


def parse_full(text: str, thinking: bool = True) -> dict:
    """Non-streaming convenience: parse the whole text in one call."""
    parser = KimiOutputParser(thinking=thinking)
    deltas = parser.feed(text) + parser.finish()
    return _merge_deltas(deltas)
