"""Word and character error rates with named, recorded text normalizers.

WER here is the corpus rate (total edits over total reference words), which
is what jiwer, Seed-TTS-eval and the Open ASR Leaderboard report. Every result
records the normalizer that actually ran, so two runs are only compared when
they normalized the same way.

Normalizers:

* ``basic``: NFKC, lower case, punctuation and symbols become spaces,
  whitespace collapsed. Dependency free; the CPU tests use it.
* ``seedtts``: the Seed-TTS-eval recipe (punctuation removed except the
  apostrophe, lower case).
* ``whisper``: Whisper's own convention, chosen per utterance from its
  language: ``whisper_en`` for English (and for an utterance with no
  language), ``whisper_basic`` for every other language.
* ``whisper_en``: Whisper's ``EnglishTextNormalizer`` (numbers, spellings,
  contractions) from ``transformers``. English only: it decomposes accented
  and combining characters and drops the marks, so Korean would be scored
  over jamo and Hindi vowel signs would split words; an utterance in another
  language is refused. Both ``whisper`` and ``whisper_en`` fall back to
  ``whisper_basic`` when the spelling table cannot be loaded, and say so in
  the returned name.
* ``whisper_basic``: Whisper's ``BasicTextNormalizer`` (language agnostic).

Both Whisper normalizers delete text inside brackets and parentheses, which
they treat as non-speech annotations. Here the bracket characters are removed
first and their content kept: a TTS input's parenthetical is read aloud, so
dropping it from the reference would score a correct reading as insertions.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Languages scored by characters rather than words (no whitespace word boundaries).
CHAR_LANGS = frozenset({"zh", "ja", "ko", "th", "yue", "cmn", "my", "lo", "km"})
ENGLISH = frozenset({"en", "eng", "english"})


def _base_lang(lang: str) -> str:
    return lang.split("-", maxsplit=1)[0].split("_", maxsplit=1)[0].lower()


def is_english(lang: str | None) -> bool:
    """English, or no language given (scored as English, the default judge language)."""
    return lang is None or _base_lang(lang) in ENGLISH


@dataclass(frozen=True)
class ByLanguage:
    """A normalizer chosen per utterance: ``english`` for English ones,
    ``other`` for the rest, or a refusal when ``other`` is None."""

    name: str
    english: Callable[[str], str]
    other: Callable[[str], str] | None

    def for_lang(self, lang: str | None) -> Callable[[str], str]:
        if is_english(lang):
            return self.english
        if self.other is None:
            raise ValueError(
                f"normalizer {self.name!r} is for English only, but an item's language is {lang!r}; "
                "use 'whisper' (English normalizer for English, the basic one for the rest) or 'whisper_basic'"
            )
        return self.other

    def __call__(self, text: str) -> str:
        return self.english(text)


Normalizer = Callable[[str], str] | ByLanguage


def check_languages(normalize: Normalizer, langs: Sequence[str | None]) -> None:
    """Raise before a run if ``normalize`` refuses any of ``langs``."""
    if isinstance(normalize, ByLanguage):
        for lang in set(langs):
            normalize.for_lang(lang)


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Errors per reference token; None when the reference normalizes to nothing (its
        insertions still count in the corpus rate)."""
        if self.reference_len == 0:
            return None
        return self.errors / self.reference_len

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_len + other.reference_len,
        )


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein alignment counts (S, D, I) of ``hypothesis`` against ``reference``."""
    n, m = len(reference), len(hypothesis)
    # Each cell holds (cost, S, D, I); ties prefer substitution, then deletion,
    # then insertion, the order jiwer/rapidfuzz report.
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        ref_tok = reference[i - 1]
        for j in range(1, m + 1):
            if ref_tok == hypothesis[j - 1]:
                cur[j] = prev[j - 1]
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(sub[0], dele[0], ins[0])
            if sub[0] == best:
                cur[j] = (best + 1, sub[1] + 1, sub[2], sub[3])
            elif dele[0] == best:
                cur[j] = (best + 1, dele[1], dele[2] + 1, dele[3])
            else:
                cur[j] = (best + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    _, s, d, ins = prev[m]
    return EditCounts(s, d, ins, n)


def _basic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    out = [" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text]
    return " ".join("".join(out).split())


_SEEDTTS_DROP = str.maketrans("", "", string.punctuation.replace("'", "") + "，。！？、；：“”‘’（）《》【】…\u2014·")


def _seedtts(text: str) -> str:
    return " ".join(text.translate(_SEEDTTS_DROP).lower().split())


_BRACKETS = re.compile(r"[()\[\]{}<>]")


def _keep_bracketed(fn: Callable[[str], str]) -> Callable[[str], str]:
    return lambda text: fn(_BRACKETS.sub(" ", text))


def _whisper_basic() -> Callable[[str], str]:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    return _keep_bracketed(BasicTextNormalizer())


def _whisper_en() -> Callable[[str], str]:
    import json

    from huggingface_hub import hf_hub_download
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    path = hf_hub_download("openai/whisper-large-v3", "normalizer.json")
    with open(path, encoding="utf-8") as f:
        return _keep_bracketed(EnglishTextNormalizer(json.load(f)))


NORMALIZERS = ("whisper", "whisper_en", "whisper_basic", "seedtts", "basic")


def get_normalizer(name: str) -> tuple[str, Normalizer]:
    """``(name_actually_used, normalizer)`` for a normalizer name."""
    if name == "basic":
        return "basic", _basic
    if name == "seedtts":
        return "seedtts", _seedtts
    if name == "whisper_basic":
        return "whisper_basic", _whisper_basic()
    if name in ("whisper", "whisper_en"):
        try:
            english = _whisper_en()
        except Exception:  # noqa: BLE001 - offline or no transformers: say which one ran
            try:
                return "whisper_basic(fallback)", _whisper_basic()
            except Exception:  # noqa: BLE001
                return "basic(fallback)", _basic
        other = _whisper_basic() if name == "whisper" else None
        return name, ByLanguage(name, english, other)
    raise ValueError(f"unknown normalizer {name!r} ({', '.join(NORMALIZERS)})")


def tokenize(text: str, lang: str | None) -> list[str]:
    """Words, or characters (spaces dropped) for languages in ``CHAR_LANGS``."""
    if lang and _base_lang(lang) in CHAR_LANGS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def error_rate(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize: Normalizer = _basic,
    langs: Sequence[str | None] | None = None,
    unit: str = "auto",
) -> tuple[float | None, list[EditCounts]]:
    """Corpus error rate and per-utterance counts.

    ``unit`` is ``word``, ``char`` or ``auto`` (characters for ``CHAR_LANGS``,
    words otherwise, decided per utterance by ``langs``). A ``ByLanguage``
    normalizer is also picked per utterance by ``langs``.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    langs = list(langs) if langs is not None else [None] * len(references)
    per: list[EditCounts] = []
    for ref, hyp, lang in zip(references, hypotheses, langs, strict=True):
        norm = normalize.for_lang(lang) if isinstance(normalize, ByLanguage) else normalize
        r, h = norm(ref or ""), norm(hyp or "")
        if unit == "char":
            rt, ht = [c for c in r if not c.isspace()], [c for c in h if not c.isspace()]
        elif unit == "word":
            rt, ht = r.split(), h.split()
        else:
            rt, ht = tokenize(r, lang), tokenize(h, lang)
        per.append(edit_counts(rt, ht))
    total = sum(per, EditCounts())
    return total.rate, per
