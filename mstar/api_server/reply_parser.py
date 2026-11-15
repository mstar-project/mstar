"""A chat reply's text, split as it streams into its reasoning and its answer.

One parser per chat request, fed each text chunk in order on the API server's message thread
(`APIServer`), so both frontends get the same pieces: the answer as the chunk's text, the
reasoning under ``metadata["reasoning"]``. A model that reasons declares where its reasoning
ends (`TextOutput.reasoning_end`); the data worker, which built the prompt, says whether the reply
starts inside the reasoning (a prompt that opened a reasoning block and left it open).
"""

from __future__ import annotations

from mstar.utils.utf8_stream import Utf8Stream

# where a chunk carries the reasoning text the parser split off it
REASONING = "reasoning"
# on a request's first text chunk, from the data worker: whether the reply starts inside its reasoning
REPLY_IN_REASONING = "reply_in_reasoning"


class ReplyParser:
    """``feed`` takes a text chunk's bytes and returns ``(reasoning, answer)``: the text certain so
    far on each side. While in the reasoning, text that could be the start of the end marker, and
    newlines that may come just before it, are held until known; the newlines that open the
    reasoning, and the answer after the marker, are dropped, as the chat template does when it
    renders a reply back.
    `finish` returns what is held when the reply ends (a reply cut inside its reasoning keeps it
    as reasoning). Outside the reasoning a chunk passes through as text."""

    __slots__ = ("_end", "_in_reasoning", "_held", "_opening", "_thinking", "_utf8")

    def __init__(self, reasoning_end: str, starts_in_reasoning: bool):
        self._end = reasoning_end
        self._in_reasoning = starts_in_reasoning
        self._held = ""
        # dropping the newlines that open the answer, right after the end marker
        self._opening = False
        # dropping the newlines that open the reasoning
        self._thinking = starts_in_reasoning
        self._utf8 = Utf8Stream()

    def feed(self, data: bytes) -> tuple[str, str]:
        text = self._utf8.feed(data)
        if not self._in_reasoning and not self._opening:
            return "", text
        return self._split(text)

    def finish(self) -> tuple[str, str]:
        text = self._utf8.flush()
        reasoning, answer = self._split(text) if text else ("", "")
        if self._in_reasoning:
            reasoning, self._held = reasoning + self._held, ""
        return reasoning, answer

    def _split(self, text: str) -> tuple[str, str]:
        if not self._in_reasoning:
            return "", self._open(text)
        buf = self._held + text
        if self._thinking:
            buf = buf.lstrip("\n")
            self._thinking = not buf
        at = buf.find(self._end)
        if at >= 0:
            self._in_reasoning, self._held, self._opening = False, "", True
            return buf[:at].rstrip("\n"), self._open(buf[at + len(self._end):])
        # hold a tail that may begin the marker, and the newlines just before it
        keep = _marker_prefix(buf, self._end)
        cut = len(buf) - keep
        while cut > 0 and buf[cut - 1] == "\n":
            cut -= 1
        self._held = buf[cut:]
        return buf[:cut], ""

    def _open(self, text: str) -> str:
        if self._opening:
            text = text.lstrip("\n")
            self._opening = not text
        return text


def _marker_prefix(text: str, marker: str) -> int:
    """The length of the longest tail of ``text`` that is a proper prefix of ``marker``."""
    for n in range(min(len(text), len(marker) - 1), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0
