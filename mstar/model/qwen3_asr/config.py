"""Config for Qwen3-ASR (Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B).

Qwen3-ASR = AuT audio encoder (``thinker.audio_tower``) + dense Qwen3 text
decoder (``thinker.model`` / ``thinker.lm_head``). Everything is read from
the checkpoint's ``config.json`` (``thinker_config.audio_config`` /
``text_config`` and the audio token ids) and ``preprocessor_config.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.aut_encoder import AuTEncoderConfig
from mstar.model.components.qwen3_lm import Qwen3LMConfig

# ---------------------------------------------------------------------------
# Resource keys and graph names
# ---------------------------------------------------------------------------
# audio_encoder node: cacheless windowed attention
AUT_ATTN = "aut_attn"
# LLM node
KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"

ENCODER_NODE = "audio_encoder"
LLM_NODE = "LLM"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

# The model writes ``language {Name}<asr_text>{transcript}``; a forced
# language is pre-filled into the assistant turn in that same form.
LANGUAGE_PREFIX = "language "
ASR_TEXT_TAG = "<asr_text>"

# ISO-639-1/-3 codes (what OpenAI clients send) -> the canonical names the
# model was trained on (``support_languages`` in config.json).
LANGUAGE_CODES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "tl": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian",
    "mk": "Macedonian",
}


@dataclass
class Qwen3ASRModelConfig:
    audio: AuTEncoderConfig = field(default_factory=AuTEncoderConfig)
    text: Qwen3LMConfig = field(default_factory=Qwen3LMConfig)

    # special tokens (thinker_config)
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    audio_token_id: int = 151676
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    eos_token_id: int = 151643  # <|endoftext|>, also the pad token

    support_languages: list[str] = field(default_factory=list)

    # log-mel front end (preprocessor_config.json)
    num_mel_bins: int = 128
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    # one request holds at most this much audio (the reference SDK's
    # MAX_ASR_INPUT_SECONDS); longer files are split by the caller
    max_audio_seconds: float = 1200.0
    # the reference decodes at most this many tokens per request
    max_new_tokens: int = 4096

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset({self.eos_token_id, self.im_end_token_id})

    @property
    def max_audio_samples(self) -> int:
        return int(self.max_audio_seconds * self.sampling_rate)

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.hop_length

    def num_audio_tokens(self, num_samples: int) -> int:
        """LLM tokens one clip occupies (about 13 per second)."""
        return self.audio.tokens_for_frames(self.num_frames(num_samples))

    def language_name(self, language: str | None) -> str | None:
        """Canonical model language for an ISO code or a (case-insensitive)
        name; ``None`` stays ``None`` (the model then reports the language)."""
        if language is None or not str(language).strip():
            return None
        key = str(language).strip()
        name = LANGUAGE_CODES.get(key.lower(), key)
        for supported in self.support_languages or LANGUAGE_CODES.values():
            if supported.lower() == name.lower():
                return supported
        raise ValueError(
            f"Unsupported Qwen3-ASR language {language!r}; "
            f"supported: {sorted(set(self.support_languages or LANGUAGE_CODES.values()))}"
        )

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "Qwen3ASRModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)
        thinker = hf.get("thinker_config", hf)
        values: dict = {
            "audio": AuTEncoderConfig.from_hf(thinker["audio_config"]),
            "text": Qwen3LMConfig.from_hf(thinker["text_config"]),
            "support_languages": list(hf.get("support_languages", [])),
        }
        for key in ("audio_start_token_id", "audio_end_token_id", "audio_token_id"):
            if key in thinker:
                values[key] = thinker[key]
        feat_path = local_dir / "preprocessor_config.json"
        if feat_path.exists():
            with open(feat_path) as f:
                feat = json.load(f)
            values["num_mel_bins"] = feat.get("feature_size", 128)
            for key in ("sampling_rate", "hop_length", "n_fft"):
                if key in feat:
                    values[key] = feat[key]
        return cls(**values)
