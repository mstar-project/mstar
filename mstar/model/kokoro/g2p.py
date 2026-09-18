"""Grapheme-to-phoneme front end and sentence chunking for Kokoro.

Runs on the CPU in the API data worker (``process_prompt``). English uses
``misaki``'s lexicon G2P with, when the optional ``misaki[en]`` extra (which
bundles espeak-ng) is installed, its espeak fallback for out-of-dictionary
words; Japanese and Mandarin use ``misaki[ja]`` / ``misaki[zh]``; the other
languages Kokoro ships voices for go through espeak-ng.

Text is cut into chunks at sentence boundaries. A chunk never exceeds the
model's 510-phoneme window; sentences are packed up to a target size so the
first chunk is small (time-to-first-audio) while later ones amortize per-step
cost.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Kokoro voice-name prefixes -> espeak / misaki language, as the reference pipeline maps them.
LANG_ALIASES = {
    "en-us": "a", "en-gb": "b", "es": "e", "fr-fr": "f", "hi": "h", "it": "i", "pt-br": "p", "ja": "j", "zh": "z",
}
LANG_NAMES = {
    "a": "American English", "b": "British English", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br",
    "j": "Japanese", "z": "Mandarin Chinese",
}
ESPEAK_LANGS = {"e", "f", "h", "i", "p"}

SENTENCE_END = set("!.?…")
# Kokoro-FastAPI / KPipeline cut long sentences at these, in order of preference.
CLAUSE_WATERFALL = (":;", ",—")
CLOSERS = {")", "”"}
PARAGRAPH_SPLIT = re.compile(r"\n+")
TEXT_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def normalize_lang_code(lang: str) -> str:
    lang = lang.lower()
    lang = LANG_ALIASES.get(lang, lang)
    if lang not in LANG_NAMES:
        raise ValueError(f"Unsupported Kokoro language {lang!r}; supported: {sorted(LANG_NAMES)}")
    return lang


@dataclass(frozen=True)
class Chunk:
    """One synthesis unit: the source text and its phoneme string."""

    text: str
    phonemes: str


def _tokens_phonemes(tokens) -> str:
    return "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in tokens).strip()


def _tokens_text(tokens) -> str:
    return "".join(t.text + t.whitespace for t in tokens).strip()


def split_sentences(tokens) -> list[list]:
    """Group misaki tokens into sentences, keeping closing quotes/brackets
    with the sentence they end."""
    sentences: list[list] = []
    current: list = []
    for i, token in enumerate(tokens):
        current.append(token)
        ends = token.phonemes in SENTENCE_END
        if ends and i + 1 < len(tokens) and tokens[i + 1].phonemes in CLOSERS:
            continue
        closes = token.phonemes in CLOSERS and len(current) > 1 and current[-2].phonemes in SENTENCE_END
        if ends or closes:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _split_long(tokens: list, hard_max: int) -> list[list]:
    """Cut a sentence longer than ``hard_max`` phonemes: at the last clause
    mark that fits, else at the last whitespace, else hard."""
    if len(_tokens_phonemes(tokens)) <= hard_max:
        return [tokens]
    for marks in CLAUSE_WATERFALL:
        cut = None
        for i in range(len(tokens) - 1, 0, -1):
            if tokens[i].phonemes in marks and len(_tokens_phonemes(tokens[: i + 1])) <= hard_max:
                cut = i + 1
                break
        if cut is not None:
            return [tokens[:cut]] + _split_long(tokens[cut:], hard_max)
    cut = None
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i - 1].whitespace and len(_tokens_phonemes(tokens[:i])) <= hard_max:
            cut = i
            break
    if cut is None:
        cut = 1
    return [tokens[:cut]] + _split_long(tokens[cut:], hard_max)


def pack_pieces(pieces: list[list], target: int, first_target: int | None = None) -> list[list]:
    """Greedily merge consecutive pieces while the phoneme count stays within
    ``target`` (``first_target`` for the first chunk); a single piece may
    exceed it."""
    chunks: list[list] = []
    current: list = []
    limit = target if first_target is None else first_target
    for piece in pieces:
        if current and len(_tokens_phonemes(current + piece)) > limit:
            chunks.append(current)
            current = []
            limit = target
        current = current + piece
    if current:
        chunks.append(current)
    return chunks


def chunk_tokens(tokens, target: int, hard_max: int, first_target: int | None = None) -> list[Chunk]:
    pieces: list[list] = []
    for sentence in split_sentences(tokens):
        pieces.extend(_split_long(sentence, hard_max))
    chunks = []
    for group in pack_pieces(pieces, target, first_target):
        phonemes = _tokens_phonemes(group)
        if phonemes:
            chunks.append(Chunk(text=_tokens_text(group), phonemes=phonemes))
    return chunks


def chunk_phoneme_strings(
    pairs: list[tuple[str, str]], target: int, hard_max: int, first_target: int | None = None
) -> list[Chunk]:
    """Pack (text, phonemes) sentences produced by a string-level G2P."""
    chunks: list[Chunk] = []
    cur_text: list[str] = []
    cur_ps: list[str] = []
    limit = target if first_target is None else first_target
    for text, ps in pairs:
        ps = ps.strip()
        if not ps:
            continue
        if len(ps) > hard_max:
            logger.warning("Truncating a %d-phoneme sentence to %d", len(ps), hard_max)
            ps = ps[:hard_max]
        if cur_ps and len(" ".join(cur_ps + [ps])) > limit:
            chunks.append(Chunk(" ".join(cur_text), " ".join(cur_ps)))
            cur_text, cur_ps = [], []
            limit = target
        cur_text.append(text.strip())
        cur_ps.append(ps)
    if cur_ps:
        chunks.append(Chunk(" ".join(cur_text), " ".join(cur_ps)))
    return chunks


class G2PFrontend:
    """Lazily constructed per-language phonemizers plus chunking."""

    def __init__(
        self, chunk_target: int, max_phonemes: int, espeak_fallback: bool = True, first_chunk_target: int | None = None
    ):
        self.chunk_target = chunk_target
        self.first_chunk_target = first_chunk_target
        self.max_phonemes = max_phonemes
        self.espeak_fallback = espeak_fallback
        self._backends: dict[str, object] = {}

    def _english_fallback(self, british: bool):
        if not self.espeak_fallback:
            return None
        try:
            from misaki import espeak

            return espeak.EspeakFallback(british=british)
        except Exception as exc:  # noqa: BLE001 — optional dependency
            logger.warning("espeak fallback unavailable (%s); out-of-dictionary words will be skipped", exc)
            return None

    def backend(self, lang: str):
        backend = self._backends.get(lang)
        if backend is not None:
            return backend
        try:
            if lang in ("a", "b"):
                from misaki import en

                british = lang == "b"
                backend = en.G2P(trf=False, british=british, fallback=self._english_fallback(british), unk="")
            elif lang == "j":
                from misaki import ja

                backend = ja.JAG2P()
            elif lang == "z":
                from misaki import zh

                backend = zh.ZHG2P(version=None)
            else:
                from misaki import espeak

                backend = espeak.EspeakG2P(language=LANG_NAMES[lang])
        except ImportError as exc:
            extra = {"j": "misaki[ja]", "z": "misaki[zh]"}.get(lang, "misaki[en]")
            raise ImportError(f"Kokoro G2P for language {lang!r} needs `pip install '{extra}'`: {exc}") from exc
        except OSError as exc:
            # spaCy raises OSError when its tagger model is not installed; misaki
            # only downloads it when it has network access.
            raise ImportError(
                "Kokoro English G2P needs the spaCy tagger: `python -m spacy download en_core_web_sm` "
                f"({exc})"
            ) from exc
        self._backends[lang] = backend
        return backend

    def chunk(self, text: str, lang: str) -> list[Chunk]:
        """Phonemize ``text`` and cut it into synthesis chunks."""
        lang = normalize_lang_code(lang)
        backend = self.backend(lang)
        chunks: list[Chunk] = []
        for paragraph in PARAGRAPH_SPLIT.split(text.strip()):
            if not paragraph.strip():
                continue
            if lang in ("a", "b"):
                _, tokens = backend(paragraph)
                chunks.extend(chunk_tokens(tokens, self.chunk_target, self.max_phonemes, self.first_chunk_target))
            else:
                sentences = [s for s in TEXT_SENTENCE_SPLIT.split(paragraph) if s.strip()]
                pairs = [(s, backend(s)[0]) for s in sentences]
                chunks.extend(
                    chunk_phoneme_strings(pairs, self.chunk_target, self.max_phonemes, self.first_chunk_target)
                )
        return chunks

    def chunk_phonemes(self, phonemes: str) -> list[Chunk]:
        """Caller-supplied phonemes bypass G2P; still bounded by the window."""
        phonemes = phonemes.strip()
        if not phonemes:
            return []
        if len(phonemes) > self.max_phonemes:
            raise ValueError(f"Phoneme string has {len(phonemes)} characters; the model window is {self.max_phonemes}")
        return [Chunk(text="", phonemes=phonemes)]
