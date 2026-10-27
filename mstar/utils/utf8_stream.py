"""Decode a model's text chunks without splitting characters.

Models with a byte-level detokenizer (Whisper, Cosmos3) send each token's raw
bytes as its own text chunk, so one character can span two chunks, and
decoding each chunk on its own turns both halves into U+FFFD. Stdlib only:
the Python SDK imports this too. The Rust frontend's twin is
``rust/server/src/utf8.rs``.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable


class Utf8Stream:
    """One text stream's incremental decoder: ``feed`` returns the complete
    characters so far and holds back an incomplete tail, ``flush`` releases
    that tail when the stream ends (as U+FFFD if it never completed)."""

    __slots__ = ("_decoder",)

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, data: bytes) -> str:
        return self._decoder.decode(data)

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)


def decode_joined(parts: Iterable[bytes]) -> str:
    """A whole response's text chunks as one string: join, then decode once."""
    return b"".join(parts).decode("utf-8", "replace")
