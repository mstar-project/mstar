"""Written-to-spoken text normalization for Zonos2 ("$5.32" -> "five dollars ...").

A port of the reference's ``tokenizer/textnorm.py`` onto upstream
``nemo_text_processing`` (the ``zonos2-norm`` extra) rather than a vendored
copy. Like the reference, normalization never fails a request: without the
package, for an unsupported language, or on a normalizer error, the text passes
through unchanged.

Building a language's grammars takes tens of seconds the first time; NeMo then
caches them as ``.far`` files under ``cache_root`` and later loads take well
under a second.
"""
from __future__ import annotations

import logging
import os
import re
import threading

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


def _warnings_only(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING


def available() -> bool:
    """Whether the ``zonos2-norm`` extra is installed."""
    try:
        import nemo_text_processing.text_normalization.normalize  # noqa: F401
    except ImportError:
        return False
    return True


class TextNormalizer:
    """Lazy per-language NeMo normalizers, safe to call from several threads.

    NeMo's ``Normalizer`` shares mutable parser state, so each language's
    construction and calls are serialized.
    """

    def __init__(self, cache_root: str):
        self.cache_root = cache_root
        self._normalizers: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._warned_missing = False

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

    def normalize(self, text: str, language: str) -> str:
        """Return ``text`` in spoken form, or unchanged if it cannot be normalized."""
        lang = SERVER_TO_NEMO_LANG.get(language)
        if lang is None or not text.strip():
            return text
        if not available():
            if not self._warned_missing:
                logger.warning(
                    "Zonos2: text normalization is on but nemo_text_processing is not "
                    "installed (the zonos2-norm extra); using raw text"
                )
                self._warned_missing = True
            return text
        spaced = _DIGIT_PUNCT_RE.sub(r"\1 \2", text)
        try:
            with self._lock(lang):
                result = self._get(lang).normalize(
                    spaced, punct_post_process=lang in _MOSES_LANGS,
                )
        except Exception:  # noqa: BLE001 - normalization must never fail a request
            logger.exception("Zonos2: text normalization failed (lang=%s); using raw text", lang)
            return text
        if not isinstance(result, str) or not result.strip():
            return text
        return _SPACE_PUNCT_RE.sub(r"\1", result)
