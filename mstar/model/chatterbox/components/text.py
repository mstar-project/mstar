"""Text normalisation and tokenisation for Chatterbox.

Three front ends ship with the checkpoints (reference: ``chatterbox/tts.py``,
``chatterbox/tts_turbo.py``, ``chatterbox/mtl_tts.py``,
``chatterbox/models/tokenizers/tokenizer.py``):

* Chatterbox uses a small BPE vocabulary (``tokenizer.json``, 704 entries)
  with ``[SPACE]`` standing in for blanks and the text wrapped in
  ``[START]`` / ``[STOP]`` (ids 255 and 0).
* Chatterbox Multilingual uses a 2454-entry grapheme BPE vocabulary
  (``grapheme_mtl_merged_expanded_v1.json``) with the same special tokens,
  a ``[<lang>]`` token in front of the text and per-language preprocessing:
  lower-casing and NFKD everywhere, Hangul syllables decomposed into jamo,
  Chinese characters spelled as Cangjie codes (``Cangjie5_TC.json``), and,
  through the optional packages the reference uses, kanji read as hiragana
  (``pykakasi``), Hebrew diacritics (``dicta_onnx``) and Russian stress
  marks (``russian_text_stresser``). Missing packages skip their step with a
  warning, as the reference does.
* Chatterbox-Turbo uses a GPT-2 tokenizer extended with paralinguistic tags
  such as ``[laugh]`` (50276 entries) and no wrapping tokens.

All normalise punctuation first. The ``punc_norm`` tables differ slightly
between the reference front ends and are kept apart here for parity.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SPACE = "[SPACE]"
_START = "[START]"
_STOP = "[STOP]"

# The multilingual checkpoint's languages (reference ``mtl_tts.SUPPORTED_LANGUAGES``).
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

EMPTY_TEXT_FALLBACK = "You need to add some text for me to talk."

_PUNC_REPLACEMENTS = [
    ("...", ", "),
    ("…", ", "),   # …
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),    # —
    ("–", "-"),    # –
    (" ,", ","),
    ("“", '"'),    # “
    ("”", '"'),    # ”
    ("‘", "'"),    # ‘
    ("’", "'"),    # ’
]

# Turbo keeps ellipses, spaced dashes and semicolons as they are.
_PUNC_REPLACEMENTS_TURBO = [
    pair for pair in _PUNC_REPLACEMENTS if pair[0] not in ("...", " - ", ";")
]

_SENTENCE_ENDERS = (".", "!", "?", "-", ",")
# The multilingual front end also accepts the CJK enders.
_SENTENCE_ENDERS_MULTILINGUAL = _SENTENCE_ENDERS + ("、", "，", "。", "？", "！")


def punc_norm(text: str, *, turbo: bool = False, multilingual: bool = False) -> str:
    """Capitalise, collapse whitespace, replace unusual punctuation and make
    sure the text ends with a sentence ender (reference ``punc_norm``)."""
    if len(text) == 0:
        return EMPTY_TEXT_FALLBACK
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (_PUNC_REPLACEMENTS_TURBO if turbo else _PUNC_REPLACEMENTS):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith(_SENTENCE_ENDERS_MULTILINGUAL if multilingual else _SENTENCE_ENDERS):
        text += "."
    return text


class ChatterboxTextTokenizer:
    """The 704-token BPE front end of Chatterbox (English)."""

    def __init__(self, vocab_file: str | Path, start_token: int, stop_token: int):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(vocab_file))
        vocab = self.tokenizer.get_vocab()
        if _START not in vocab or _STOP not in vocab:
            raise ValueError(f"{vocab_file} lacks the {_START}/{_STOP} tokens")
        self.start_token = start_token
        self.stop_token = stop_token

    def encode(self, text: str) -> list[int]:
        """Token ids without the wrapping start/stop tokens."""
        return self.tokenizer.encode(text.replace(" ", _SPACE)).ids

    def __call__(self, text: str) -> torch.Tensor:
        """``[START] + ids + [STOP]`` after punctuation normalisation."""
        ids = self.encode(punc_norm(text))
        return torch.tensor(
            [self.start_token, *ids, self.stop_token], dtype=torch.long
        )

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return text.replace(" ", "").replace(_SPACE, " ").replace(_STOP, "")


class TurboTextTokenizer:
    """GPT-2 BPE plus paralinguistic tags, as shipped with Chatterbox-Turbo."""

    def __init__(self, snapshot_dir: str | Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(punc_norm(text, turbo=True))["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Multilingual front end
# ---------------------------------------------------------------------------

_warned: set[str] = set()


def _skip(step: str, reason: str) -> None:
    if step not in _warned:
        _warned.add(step)
        logger.warning("%s: %s; the text goes through unchanged", step, reason)


def decompose_hangul(text: str) -> str:
    """Korean: split every Hangul syllable into its jamo (reference ``korean_normalize``)."""
    out = []
    for char in text:
        if not ("\uac00" <= char <= "\ud7af"):
            out.append(char)
            continue
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ""
        out.append(initial + medial + final)
    return "".join(out).strip()


def _is_kanji(c: str) -> bool:
    return 19968 <= ord(c) <= 40959


def _is_katakana(c: str) -> bool:
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese: read kanji as hiragana with ``pykakasi``, keep katakana, then
    NFKD (reference ``hiragana_normalize``). Without the package the text is
    left as it is."""
    try:
        import pykakasi
    except ImportError:
        _skip("Japanese kana normalisation", "pykakasi is not installed")
        return text
    out = []
    for piece in pykakasi.kakasi().convert(text):
        original, hira = piece["orig"], piece["hira"]
        if any(_is_kanji(c) for c in original):
            if hira and hira[0] in ("は", "へ"):
                hira = " " + hira
            out.append(hira)
        elif original and all(_is_katakana(c) for c in original):
            out.append(original)
        else:
            out.append(original)
    return unicodedata.normalize("NFKD", "".join(out))


def add_hebrew_diacritics(text: str) -> str:
    try:
        from dicta_onnx import Dicta
    except ImportError:
        _skip("Hebrew diacritics", "dicta_onnx is not installed")
        return text
    try:
        return Dicta().add_diacritics(text)
    except Exception as exc:  # the package's own failures are non-fatal in the reference too
        _skip("Hebrew diacritics", f"dicta_onnx failed: {exc}")
        return text


def add_russian_stress(text: str) -> str:
    try:
        from russian_text_stresser.text_stresser import RussianTextStresser
    except ImportError:
        _skip("Russian stress marks", "russian_text_stresser is not installed")
        return text
    try:
        return RussianTextStresser().stress_text(text)
    except Exception as exc:
        _skip("Russian stress marks", f"russian_text_stresser failed: {exc}")
        return text


class CangjieConverter:
    """Chinese: spell each character as ``[cj_<code letters>][cj_.]`` tokens
    from the ``Cangjie5_TC.json`` table (reference ``ChineseCangjieConverter``);
    characters sharing a code carry their index in that code's list. Words are
    first separated by a blank with ``spacy_pkuseg`` when it is installed."""

    def __init__(self, mapping_file: str | Path):
        self.word2cj: dict[str, str] = {}
        self.cj2word: dict[str, list[str]] = {}
        with open(mapping_file, encoding="utf-8") as fp:
            for entry in json.load(fp):
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                self.cj2word.setdefault(code, []).append(word)
        try:
            from spacy_pkuseg import pkuseg

            self.segmenter = pkuseg()
        except ImportError:
            _skip("Chinese word segmentation", "spacy_pkuseg is not installed")
            self.segmenter = None

    def encode_glyph(self, glyph: str) -> str | None:
        code = self.word2cj.get(glyph)
        if code is None:
            return None
        index = self.cj2word[code].index(glyph)
        return code + (str(index) if index > 0 else "")

    def __call__(self, text: str) -> str:
        if self.segmenter is not None:
            text = " ".join(self.segmenter.cut(text))
        out = []
        for char in text:
            code = self.encode_glyph(char) if unicodedata.category(char) == "Lo" else None
            if code is None:
                out.append(char)
            else:
                out.append("".join(f"[cj_{c}]" for c in code) + "[cj_.]")
        return "".join(out)


class MultilingualTextTokenizer:
    """The 2454-token grapheme front end of Chatterbox Multilingual."""

    def __init__(
        self, vocab_file: str | Path, cangjie_file: str | Path | None,
        start_token: int = 255, stop_token: int = 0,
    ):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(vocab_file))
        vocab = self.tokenizer.get_vocab()
        if _START not in vocab or _STOP not in vocab:
            raise ValueError(f"{vocab_file} lacks the {_START}/{_STOP} tokens")
        self.start_token = start_token
        self.stop_token = stop_token
        self.cangjie = CangjieConverter(cangjie_file) if cangjie_file else None

    @staticmethod
    def check_language(language_id: str | None) -> str | None:
        if language_id is None:
            return None
        lang = str(language_id).lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language_id {language_id!r}; supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )
        return lang

    def preprocess(self, text: str, language_id: str | None) -> str:
        """Lower-case, NFKD, the language's own step, then the language token
        (reference ``MTLTokenizer.encode`` up to the vocabulary lookup)."""
        text = unicodedata.normalize("NFKD", text.lower())
        if language_id == "zh":
            if self.cangjie is None:
                _skip("Chinese Cangjie spelling", "no Cangjie table was given")
            else:
                text = self.cangjie(text)
        elif language_id == "ja":
            text = hiragana_normalize(text)
        elif language_id == "he":
            text = add_hebrew_diacritics(text)
        elif language_id == "ko":
            text = decompose_hangul(text)
        elif language_id == "ru":
            text = add_russian_stress(text)
        if language_id:
            text = f"[{language_id}]{text}"
        return text

    def encode(self, text: str, language_id: str | None = None) -> list[int]:
        """Token ids without the wrapping start/stop tokens."""
        text = self.preprocess(text, self.check_language(language_id))
        return self.tokenizer.encode(text.replace(" ", _SPACE)).ids

    def __call__(self, text: str, language_id: str | None = None) -> torch.Tensor:
        """``[START] + ids + [STOP]`` after punctuation normalisation."""
        ids = self.encode(punc_norm(text, multilingual=True), language_id)
        return torch.tensor([self.start_token, *ids, self.stop_token], dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return text.replace(" ", "").replace(_SPACE, " ").replace(_STOP, "").replace("[UNK]", "")
