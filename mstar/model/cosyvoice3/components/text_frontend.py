# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the text normalization of
# ``CosyVoiceFrontEnd.text_normalize``, with wetext optional and rules of its
# own for English when wetext is not there, its pieces joined back into one
# text (the API server splits long text), and a decimal point in Chinese text
# kept.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/cli/frontend.py and
# cosyvoice/utils/frontend_utils.py, at 074ca6d) were released under
# Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""CosyVoice3's text frontend: what the reference does to the text before
tokenizing it.

The reference runs ``wetext`` (WeTextProcessing's grammars) over the text, then
for English spells out the digits left, and for text with Chinese applies its
blank, bracket and punctuation rules. Here ``wetext`` is optional (the
``cosyvoice3_wetext`` extra) and English numbers, money, dates and times are
read by ``english_normalizer``'s rules when it is not loaded, or after it for
what it leaves. Text with Chinese keeps its digits without ``wetext``, as the
reference's fallback does. Either way the text ends on a sentence stop, as the
reference's sentence splitter leaves it. Text carrying a ``<|...|>`` tag is
left alone. The reference decides Chinese or English once for the whole text,
so the pieces of a request with Chinese in it all take the Chinese rules (an
English piece keeps its digits), which ``normalize(chinese=True)`` follows.

Differences from the reference: text without Chinese whose letters are mostly
not Latin (kana, Hangul, Cyrillic) is left as written, digits included, for
the LM to read; the reference reads its digits as English words, which made
kana and Russian number text worse in served tests. ``wetext`` grammars are read from the local
ModelScope cache when present (``wetext`` itself asks ModelScope on every
start and fails when rate-limited), a text that ``wetext`` raises on or that is
longer than ``WETEXT_MAX_CHARS`` goes to the rules alone (``wetext`` slows down
faster than linearly: about 0.8 s for 5,000 characters), and a ``.`` between
two digits in Chinese text stays a decimal point (the fallback turns "12.5"
into "12。5", a sentence break ``wetext`` never leaves it to see).
"""

from __future__ import annotations

import logging
import re
import unicodedata

from mstar.model.cosyvoice3.components.english_normalizer import normalize_english

logger = logging.getLogger(__name__)

# longest text handed to wetext; the API server's pieces are at most 320 characters
WETEXT_MAX_CHARS = 1000
WETEXT_REPO = "pengzhendong/wetext"

_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
# every "." except one between two digits
_STOP_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_ZH_TRAILING_COMMAS = "，,、"
_STOPS = {"zh": "。？！；：、.?!;", "en": ".?!;:"}
_CLOSING_QUOTES = ('"', "”")


def contains_chinese(text: str) -> bool:
    return bool(_CHINESE.search(text))


def latin_script(text: str) -> bool:
    """Whether most of ``text``'s letters are Latin; digits, punctuation,
    spaces and marks do not count, so text without letters counts as Latin
    and a tie does not."""
    latin = other = 0
    for c in text:
        if c.isalpha():
            if c < "\x80" or unicodedata.name(c, "").startswith("LATIN"):
                latin += 1
            else:
                other += 1
    return latin > other or other == 0


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
    character other than punctuation, symbols or blanks (and for "", which
    a text of only brackets becomes)."""
    stops = _STOPS[lang]
    if not text:
        return ""
    if text[-1] not in stops:
        text += "。" if lang == "zh" else "."
    # fragments, joined once at the end: appending each closing quote to the
    # previous sentence's string copies it, quadratic on a run of '."'
    sentences: list[list[str]] = []
    start = 0
    for i, c in enumerate(text):
        if c not in stops:
            continue
        if i > start:
            sentences.append([text[start:i], c])
        if i + 1 < len(text) and text[i + 1] in _CLOSING_QUOTES:
            if sentences:
                sentences[-1].append(text[i + 1])
            start = i + 2
        else:
            start = i + 1
    joined = "".join(part for sentence in sentences for part in sentence)
    if all(unicodedata.category(c)[0] in "PSZ" for c in joined):
        return ""
    return joined


def load_wetext() -> dict:
    """``wetext``'s English and Chinese normalizers by language, or ``{}`` when
    ``wetext`` is not installed or its grammars cannot be had."""
    try:
        from modelscope import snapshot_download
        from wetext import Normalizer
    except ImportError:
        return {}
    try:
        try:
            repo = snapshot_download(WETEXT_REPO, local_files_only=True)
        except Exception:  # noqa: BLE001 (not cached yet)
            repo = snapshot_download(WETEXT_REPO)
        # the reference's Normalizer(remove_erhua=False) and Normalizer(), per language
        return {
            lang: Normalizer(f"{repo}/{lang}/tn/tagger.fst", f"{repo}/{lang}/tn/verbalizer.fst", lang=lang)
            for lang in ("en", "zh")
        }
    except Exception:  # noqa: BLE001 (network, rate limits, a partial download)
        logger.warning("CosyVoice3 reads numbers with its own rules: wetext could not load its grammars", exc_info=True)
        return {}


class CosyVoice3TextFrontend:
    """``normalize(text)``: the text the reference would tokenize.

    ``warmup()`` loads ``wetext`` if it is installed; until then, and without
    it, the rules alone run. Loading never happens in ``normalize``, so a
    request never waits on ModelScope.
    """

    def __init__(self) -> None:
        self._wetext: dict = {}
        self._warm = False

    def warmup(self) -> None:
        if not self._warm:
            self._warm = True
            self._wetext = load_wetext()
            if self._wetext:
                logger.info("CosyVoice3 normalizes text with wetext")

    def _wetext_normalize(self, text: str, lang: str) -> str:
        normalizer = self._wetext.get(lang)
        if normalizer is None or len(text) > WETEXT_MAX_CHARS:
            return text
        try:
            return normalizer.normalize(text) or text
        except Exception:  # noqa: BLE001 (wetext asserts on some inputs, e.g. "Item #5.")
            logger.debug("wetext failed on a text; the rules read it", exc_info=True)
            return text

    def normalize(self, text: str, chinese: bool = False) -> str:
        """The text the reference would tokenize. ``chinese`` says the whole
        request has Chinese in it, so a piece without any still takes the
        Chinese rules: the reference decides once per text, not per piece."""
        if "<|" in text and "|>" in text:
            return text
        stripped = text.strip()
        if not stripped:
            return text
        if chinese or contains_chinese(stripped):
            out = self._wetext_normalize(stripped, "zh")
            out = out.replace("\n", "")
            out = replace_blank(out)
            out = replace_corner_mark(out)
            out = _STOP_DOT.sub("。", out)
            out = out.replace(" - ", "，")
            out = remove_bracket(out)
            # rstrip, not a "[，,、]+$" regex, which is quadratic in a run of commas
            if out.endswith(tuple(_ZH_TRAILING_COMMAS)):
                out = out.rstrip(_ZH_TRAILING_COMMAS) + "。"
            lang = "zh"
        elif latin_script(stripped):
            out = normalize_english(self._wetext_normalize(stripped, "en"))
            lang = "en"
        else:
            return text
        # nothing left to speak: the reference would synthesize nothing; keep the text
        return close_sentences(out, lang) or stripped
