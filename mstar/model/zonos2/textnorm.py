"""Written-to-spoken text normalization for Zonos2 ("$5.32" -> "five dollars ...").

A port of the reference's ``tokenizer/textnorm.py`` onto upstream
``nemo_text_processing`` (the ``zonos2-norm`` extra) rather than a vendored
copy. Like the reference, normalization never fails a request: without the
package, for an unsupported language, or on a normalizer error, the text passes
through unchanged.

Building a language's grammars takes tens of seconds the first time; NeMo then
caches them as ``.far`` files under ``cache_root`` and later loads take well
under a second. Grammars are built only by :meth:`TextNormalizer.start_build`,
on a background thread; text in a language that is not built yet, or not
listed, passes through, so a request never waits behind a build.

NeMo's cost grows faster than linearly with input length, so text is normalized
a sentence at a time, and once the time budget is spent the rest passes through.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterable
from functools import lru_cache

logger = logging.getLogger(__name__)

# Request language codes (the reference's) -> NeMo language packages.
SERVER_TO_NEMO_LANG: dict[str, str] = {
    "en_us": "en",
    "en_gb": "en",
    "fr_fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt_br": "pt",
    "ja": "ja",
    "cmn": "zh",
    "ko": "ko",
}

# NeMo's own tests run Korean lower-cased; every other language is cased.
_LOWER_CASED_LANGS = {"ko"}

# A digit right before sentence punctuation trips several NeMo grammars (the
# reference found pt raising and de reading dates digit by digit). Space the
# punctuation off first; the post-processing re-attaches it.
_DIGIT_PUNCT_RE = re.compile(r"(\d)([.!?,;:])(?=\s|$)")
_SPACE_PUNCT_RE = re.compile(r" +([.!?,;:])(?=\s|$)")

# Moses punctuation post-processing is right for these; for the European
# languages it glues currency symbols onto the next word, so skip it there.
_MOSES_LANGS = {"en", "zh", "ja", "ko"}

# The longest piece one NeMo call gets; longer sentences are cut at whitespace.
MAX_CHUNK_BYTES = 1000
# NeMo's sentence splitter needs whitespace, so CJK terminators get their own split.
_CJK_SENTENCE_RE = re.compile(r"(?<=[。！？])")


def _warnings_only(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING


@lru_cache(maxsize=1)
def available() -> bool:
    """Whether the ``zonos2-norm`` extra is installed."""
    try:
        import nemo_text_processing.text_normalization.normalize  # noqa: F401
    except ImportError:
        return False
    return True


def _split_long(piece: str) -> list[str]:
    """Cut ``piece`` into parts of at most ``MAX_CHUNK_BYTES``, at a space when there is one."""
    parts = []
    while len(piece.encode("utf-8")) > MAX_CHUNK_BYTES:
        head = piece.encode("utf-8")[:MAX_CHUNK_BYTES].decode("utf-8", errors="ignore")
        cut = head.rfind(" ")
        cut = cut if cut > 0 else len(head)
        parts.append(piece[:cut])
        piece = piece[cut:].lstrip()
    if piece:
        parts.append(piece)
    return parts


class TextNormalizer:
    """Per-language NeMo normalizers, safe to call from several threads.

    NeMo's ``Normalizer`` shares mutable parser state, so each language's
    construction and calls are serialized. ``time_budget_s`` caps the NeMo time
    of one :meth:`normalize` call, checked between sentences; ``None`` is no cap.
    """

    def __init__(self, cache_root: str, time_budget_s: float | None = None):
        self.cache_root = cache_root
        self.time_budget_s = time_budget_s
        self._normalizers: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._warned_missing = False
        self._warned_unbuilt: set[str] = set()
        self._building: set[str] = set()

    def _lock(self, lang: str) -> threading.Lock:
        with self._global_lock:
            return self._locks.setdefault(lang, threading.Lock())

    def _get(self, lang: str):
        # Called with the language's lock held.
        if lang not in self._normalizers:
            from nemo_text_processing.text_normalization.normalize import Normalizer

            # NeMo logs at INFO on every call and resets its level to INFO in
            # normalize(), so drop those records with a filter instead.
            nemo_logger = logging.getLogger("NeMo-text-processing")
            if _warnings_only not in nemo_logger.filters:
                nemo_logger.addFilter(_warnings_only)
            case = "lower_cased" if lang in _LOWER_CASED_LANGS else "cased"
            # One directory per (lang, case): NeMo's .far names collide across languages.
            cache_dir = os.path.join(self.cache_root, f"{lang}_{case}")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(
                "Zonos2: building the '%s' text normalizer (tens of seconds the first "
                "time; cached under %s)", lang, cache_dir,
            )
            self._normalizers[lang] = Normalizer(
                input_case=case, lang=lang, cache_dir=cache_dir, overwrite_cache=False,
            )
        return self._normalizers[lang]

    def start_build(self, languages: Iterable[str]) -> threading.Thread | None:
        """Build these request languages' grammars on a daemon thread; return it.

        Until a language is built, its text passes through unchanged.
        """
        if not available():
            self._warn_missing()
            return None
        langs = []
        for code in languages:
            lang = SERVER_TO_NEMO_LANG.get(code)
            if lang is None:
                logger.warning(
                    "Zonos2: no text normalizer for language %r; known: %s",
                    code, ", ".join(SERVER_TO_NEMO_LANG),
                )
            elif lang not in langs:
                langs.append(lang)
        self._building.update(langs)
        thread = threading.Thread(
            target=self._build, args=(langs,), name="zonos2-textnorm-build", daemon=True,
        )
        thread.start()
        return thread

    def _build(self, langs: list[str]) -> None:
        for lang in langs:
            try:
                with self._lock(lang):
                    self._get(lang)
            except Exception:  # noqa: BLE001 - a bad grammar must not stop the server
                logger.exception("Zonos2: building the '%s' text normalizer failed", lang)
            finally:
                self._building.discard(lang)

    def _warn_missing(self) -> None:
        if not self._warned_missing:
            logger.warning(
                "Zonos2: text normalization is on but nemo_text_processing is not "
                "installed (the zonos2-norm extra); using raw text"
            )
            self._warned_missing = True

    def normalize(self, text: str, language: str) -> str:
        """Return ``text`` in spoken form, or unchanged if it cannot be normalized."""
        lang = SERVER_TO_NEMO_LANG.get(language)
        if lang is None or not text.strip():
            return text
        if not available():
            self._warn_missing()
            return text
        if lang not in self._normalizers:
            if lang in self._building:
                logger.debug("Zonos2: '%s' grammars still building; using raw text", lang)
                return text
            if lang not in self._warned_unbuilt:
                self._warned_unbuilt.add(lang)
                logger.warning(
                    "Zonos2: no '%s' text normalizer is built, so %r text is spoken as "
                    "written; add it to text_normalization_languages",
                    lang, language,
                )
            return text
        try:
            with self._lock(lang):
                return self._normalize_chunks(text, lang)
        except Exception:  # noqa: BLE001 - normalization must never fail a request
            logger.exception("Zonos2: text normalization failed (lang=%s); using raw text", lang)
            return text

    def _normalize_chunks(self, text: str, lang: str) -> str:
        # Called with the language's lock held.
        normalizer = self._normalizers[lang]
        chunks = [
            part
            for sentence in normalizer.split_text_into_sentences(text.strip())
            for cjk in _CJK_SENTENCE_RE.split(sentence)
            for part in _split_long(cjk.strip())
        ]
        deadline = None if self.time_budget_s is None else time.monotonic() + self.time_budget_s
        out = []
        for i, chunk in enumerate(chunks):
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    "Zonos2: text normalization ran past its %.1f s budget; the last %d "
                    "of %d sentences are spoken as written",
                    self.time_budget_s, len(chunks) - i, len(chunks),
                )
                out.extend(chunks[i:])
                break
            out.append(self._normalize_one(normalizer, chunk, lang))
        return " ".join(out)

    @staticmethod
    def _normalize_one(normalizer, text: str, lang: str) -> str:
        spaced = _DIGIT_PUNCT_RE.sub(r"\1 \2", text)
        result = normalizer.normalize(spaced, punct_post_process=lang in _MOSES_LANGS)
        if not isinstance(result, str) or not result.strip():
            return text
        return _SPACE_PUNCT_RE.sub(r"\1", result)
