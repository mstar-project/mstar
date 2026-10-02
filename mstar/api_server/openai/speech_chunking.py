"""Sentence chunking for ``/v1/audio/speech``.

Long inputs are split into sentence groups that are synthesized as separate
requests and played back in order. This keeps every autoregressive
text-to-speech request inside the length its model was trained for, lets the
engine batch the pieces of one long input like independent requests, and
starts the second piece while the first is still streaming (see
``serving_speech``). Splitting is purely textual and model-agnostic.
"""

from __future__ import annotations

import re

# One sentence: a lazy body up to a Latin terminator (with any closing quotes
# or brackets) that precedes whitespace or the end, a CJK terminator, a
# paragraph break, or the end of the text.
_SENTENCE = re.compile(
    r""".+?(?:
        [.!?;]["'”’)\]]*(?=\s|\Z)
      | [。！？；]
      | (?=\n[ \t]*\n)
      | \Z
    )""",
    re.VERBOSE | re.DOTALL,
)
# Soft break points inside an over-long sentence, most preferred first.
_SOFT_BREAKS = (
    re.compile(r"(?<=[,;:，；：])\s*"),
    re.compile(r"\s+"),
)


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    """Split one over-long sentence at clause boundaries, then at spaces."""
    pieces = [sentence]
    for pattern in _SOFT_BREAKS:
        wrapped: list[str] = []
        for piece in pieces:
            if len(piece) <= max_chars:
                wrapped.append(piece)
                continue
            current = ""
            for part in (p.strip() for p in pattern.split(piece)):
                if not part:
                    continue
                candidate = f"{current} {part}" if current else part
                if current and len(candidate) > max_chars:
                    wrapped.append(current)
                    current = part
                else:
                    current = candidate
            if current:
                wrapped.append(current)
        pieces = wrapped
    return pieces


def split_sentences(text: str, max_chars: int = 400, min_chars: int = 24) -> list[str]:
    """Group ``text`` into sentence chunks of at most ``max_chars`` characters.

    Sentences are never cut unless one alone exceeds ``max_chars`` (then it is
    wrapped at clause boundaries, or spaces as a last resort). A trailing
    fragment shorter than ``min_chars`` is merged into its predecessor so the
    model is not asked to voice a lone "Okay." Returns ``[text]`` when nothing
    needs splitting and ``[]`` for blank input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences: list[str] = []
    for match in _SENTENCE.finditer(text):
        piece = " ".join(match.group(0).split())
        if not piece:
            continue
        sentences.extend(_hard_wrap(piece, max_chars) if len(piece) > max_chars else [piece])

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)

    if (
        len(chunks) > 1
        and len(chunks[-1]) < min_chars
        and len(chunks[-2]) + 1 + len(chunks[-1]) <= max_chars * 5 // 4
    ):
        chunks[-2:] = [f"{chunks[-2]} {chunks[-1]}"]
    return chunks
