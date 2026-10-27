"""Config for NVIDIA Canary encoder-decoder speech models (canary-1b-v2).

Dimensions come from the HF checkpoint's ``config.json``; the task tokens of
the decoder prompt come from its ``tokenizer.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.utils import GRANARY_LANGUAGES, base_language

# decoder node resources
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
# no RoPE: declared for the position counter that drives the sinusoid lookup
POS = "positions"
# the encoder output, written once at prefill, in a cache of its own so the
# self-attention resource never plans over it
CROSS_KV_CACHE = "cross_kv_cache"
CROSS_ATTN = "cross_attn"
CONTEXT_LABEL = "main"

# the HF feature extractor's settings that NemoLogMel implements
MEL_SETTINGS = dict(sampling_rate=16_000, n_fft=512, win_length=400, hop_length=160, preemphasis=0.97)

# the decoder prompt is 9 tokens for every task (``prompt_ids``)
PROMPT_LENGTH = 9

# Languages canary-1b-v2 transcribes, and translates to or from English.
LANGUAGES = GRANARY_LANGUAGES


def check_task(source_lang: str, target_lang: str) -> tuple[str, str]:
    """Normalize a (speech language, output language) pair, or raise
    ValueError naming what Canary cannot do."""
    source_lang, target_lang = base_language(source_lang), base_language(target_lang)
    for lang in (source_lang, target_lang):
        if lang not in LANGUAGES:
            raise ValueError(f"Canary does not support language {lang!r}; supported: {', '.join(LANGUAGES)}")
    if source_lang != target_lang and "en" not in (source_lang, target_lang):
        raise ValueError(f"Canary translates to or from English only; got {source_lang!r} -> {target_lang!r}")
    return source_lang, target_lang


@dataclass
class CanaryModelConfig:
    encoder: FastConformerConfig
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 4096
    vocab_size: int = 16384
    max_position_embeddings: int = 1024
    eos_token_id: int = 3
    # token id by content, for the prompt's task tokens
    special_ids: dict[str, int] = field(default_factory=dict)
    sampling_rate: int = 16_000

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> CanaryModelConfig:
        local_dir = Path(local_dir)
        hf = json.loads((local_dir / "config.json").read_text())
        dec = hf["decoder_config"]
        tok = json.loads((local_dir / "tokenizer.json").read_text())
        mel = json.loads((local_dir / "processor_config.json").read_text())["feature_extractor"]
        expected = {**MEL_SETTINGS, "feature_size": hf["encoder_config"]["num_mel_bins"]}
        unsupported = {k: mel.get(k) for k, v in expected.items() if mel.get(k) != v}
        if unsupported:
            raise ValueError(f"unsupported mel front end settings {unsupported}; expected {expected}")
        if dec.get("hidden_act", "relu") != "relu":
            raise ValueError(f"unsupported Canary decoder activation {dec['hidden_act']!r}")
        return cls(
            encoder=FastConformerConfig.from_hf(hf["encoder_config"]),
            hidden_size=dec["hidden_size"],
            num_layers=dec["num_hidden_layers"],
            num_heads=dec["num_attention_heads"],
            intermediate_size=dec["intermediate_size"],
            vocab_size=dec["vocab_size"],
            max_position_embeddings=dec["max_position_embeddings"],
            eos_token_id=hf["eos_token_id"],
            special_ids={t["content"]: t["id"] for t in tok["added_tokens"]},
        )

    def prompt_ids(
        self, source_lang: str = "en", target_lang: str | None = None, punctuation: bool = True,
    ) -> list[int]:
        """The decoder prompt the checkpoint's chat template renders:
        ``<|startofcontext|><|startoftranscript|><|emo:undefined|><|src|><|tgt|>``
        ``<|pnc|><|noitn|><|notimestamp|><|nodiarize|>``."""
        source_lang, target_lang = check_task(source_lang, target_lang or source_lang)
        tokens = [
            "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
            f"<|{source_lang}|>", f"<|{target_lang}|>", "<|pnc|>" if punctuation else "<|nopnc|>",
            "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
        ]
        return [self.special_ids[t] for t in tokens]
