# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the text normalization of
# ``CosyVoiceFrontEnd.text_normalize`` without ttsfrd or wetext, its pieces
# joined back into one text (the API server splits long text), a decimal point
# in Chinese text kept, and superscript digits left to the Chinese rule.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/cli/frontend.py and
# cosyvoice/utils/frontend_utils.py, at 074ca6d) were released under
# Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""CosyVoice3's text frontend: what the reference does to the text before
tokenizing it.

Without Chinese characters, every run of digits is spelled out with
``inflect`` ("Flight 247" -> "Flight two hundred and forty-seven"). With them,
the reference's blank, bracket and punctuation rules apply and digits stay as
written. Either way the text ends on a sentence stop, as the reference's
sentence splitter leaves it. Text carrying a ``<|...|>`` tag is left alone.

The reference runs ``wetext`` (or ``ttsfrd``) before these rules when it can
build one, and falls back to them alone when it cannot; this is that fallback.
Two deliberate differences from it: a ``.`` between two digits in Chinese text
stays a decimal point (the fallback turns "12.5" into "12。5", a sentence break
wetext never leaves it to see), and only decimal digits are spelled out
(``inflect`` reads a superscript such as "²" as "zero").
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
# every "." except one between two digits
_STOP_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_ZH_TRAILING_COMMAS = re.compile(r"[，,、]+$")
_STOPS = {"zh": "。？！；：、.?!;", "en": ".?!;:"}
_CLOSING_QUOTES = ('"', "”")


def contains_chinese(text: str) -> bool:
    return bool(_CHINESE.search(text))


def spell_out_number(text: str, inflect_engine) -> str:
    """Every run of decimal digits as ``inflect`` words; a run too long for
    ``inflect`` stays as written."""

    def words(digits: str) -> str:
        try:
            return inflect_engine.number_to_words(digits)
        except Exception:  # noqa: BLE001 (inflect's NumOutOfRangeError and kin)
            return digits

    out: list[str] = []
    start = None
    for i, c in enumerate(text):
        if c.isdecimal():
            if start is None:
                start = i
            continue
        if start is not None:
            out.append(words(text[start:i]))
            start = None
        out.append(c)
    if start is not None:
        out.append(words(text[start:]))
    return "".join(out)


def replace_blank(text: str) -> str:
    """Drop every space that is not between two non-space ASCII characters."""
    out: list[str] = []
    for i, c in enumerate(text):
        if c != " ":
            out.append(c)
            continue
        before = text[i - 1] if i > 0 else " "
        after = text[i + 1] if i + 1 < len(text) else " "
        if before.isascii() and before != " " and after.isascii() and after != " ":
            out.append(c)
    return "".join(out)


def replace_corner_mark(text: str) -> str:
    return text.replace("²", "平方").replace("³", "立方")


def remove_bracket(text: str) -> str:
    for mark in ("（", "）", "【", "】", "`"):
        text = text.replace(mark, "")
    return text.replace("——", " ")


def close_sentences(text: str, lang: str) -> str:
    """The reference's sentence splitter (``split_paragraph``) with its pieces
    joined back: a stop is added at the end if missing, a stop that ends an
    empty sentence (the second and third dots of "...") is dropped, and a
    closing quote stays with its sentence. Returns "" when no sentence has a
    character other than punctuation or symbols."""
    stops = _STOPS[lang]
    if text[-1] not in stops:
        text += "。" if lang == "zh" else "."
    sentences: list[str] = []
    start = 0
    for i, c in enumerate(text):
        if c not in stops:
            continue
        if i > start:
            sentences.append(text[start:i] + c)
        if i + 1 < len(text) and text[i + 1] in _CLOSING_QUOTES:
            if sentences:
                sentences[-1] += text[i + 1]
            start = i + 2
        else:
            start = i + 1
    joined = "".join(sentences)
    if all(unicodedata.category(c)[0] in "PS" for c in joined):
        return ""
    return joined


class CosyVoice3TextFrontend:
    """``normalize(text)``: the text the reference would tokenize."""

    def __init__(self) -> None:
        self._inflect = None
        self._inflect_missing = False

    def warmup(self) -> None:
        self._inflect_engine()

    def _inflect_engine(self):
        if self._inflect is None and not self._inflect_missing:
            try:
                import inflect
            except ImportError:
                self._inflect_missing = True
                logger.warning(
                    "CosyVoice3 reads digits as written: `inflect` is not installed "
                    "(pip install 'mstar-ai[cosyvoice3]')"
                )
                return None
            self._inflect = inflect.engine()
        return self._inflect

    def normalize(self, text: str) -> str:
        if "<|" in text and "|>" in text:
            return text
        stripped = text.strip()
        if not stripped:
            return text
        if contains_chinese(stripped):
            out = stripped.replace("\n", "")
            out = replace_blank(out)
            out = replace_corner_mark(out)
            out = _STOP_DOT.sub("。", out)
            out = out.replace(" - ", "，")
            out = remove_bracket(out)
            out = _ZH_TRAILING_COMMAS.sub("。", out)
            lang = "zh"
        else:
            engine = self._inflect_engine() if any(c.isdecimal() for c in stripped) else None
            out = spell_out_number(stripped, engine) if engine is not None else stripped
            lang = "en"
        # nothing left to speak: the reference would synthesize nothing; keep the text
        return close_sentences(out, lang) or stripped
