"""Config for Whisper encoder-decoder ASR models (large-v3 and friends).

Values are read from the HF checkpoint's ``config.json`` +
``generation_config.json`` so the same class serves any Whisper size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Resource label constants (decoder node)
# ---------------------------------------------------------------------------
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
# Positions: Whisper has no RoPE. The resource is declared for its
# per-(request, label) position counter, which drives the learned absolute
# ``embed_positions`` lookup; the attention layers never call it.
POS = "positions"

# The encoder context lives in its own KV cache so the self-attention
# resource, which plans a wrapper per label of the cache it names, never sees
# it. Written once at prefill, read (zero-span) by every later step.
CROSS_KV_CACHE = "cross_kv_cache"
CROSS_ATTN = "cross_attn"
CONTEXT_LABEL = "main"


@dataclass
class WhisperModelConfig:
    # transformer dims (large-v3 defaults)
    d_model: int = 1280
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    num_mel_bins: int = 128
    vocab_size: int = 51866
    max_target_positions: int = 448
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    scale_embedding: bool = False

    # special tokens
    decoder_start_token_id: int = 50258
    eos_token_id: int = 50257
    no_timestamps_token_id: int = 50364

    # generation_config maps: "<|en|>" -> 50259, "transcribe" -> 50360
    lang_to_id: dict[str, int] = field(default_factory=dict)
    task_to_id: dict[str, int] = field(default_factory=dict)

    # Logit suppression (HF generate parity): tokens never sampled, and
    # tokens additionally blocked for the first generated token.
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.decoder_attention_heads

    # generation_config.json keys (not in config.json); the rest of the
    # dataclass fields map 1:1 to config.json keys by name.
    _GEN_KEYS = (
        "lang_to_id", "task_to_id", "no_timestamps_token_id",
        "suppress_tokens", "begin_suppress_tokens",
    )

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "WhisperModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)

        gen: dict = {}
        gen_path = local_dir / "generation_config.json"
        if gen_path.exists():
            with open(gen_path) as f:
                gen = json.load(f)

        # Every field is named to match its source key; pull from
        # generation_config.json for the _GEN_KEYS, else config.json. Fields
        # absent from both keep their dataclass default.
        names = {f.name for f in fields(cls)}
        values = {
            name: (gen if name in cls._GEN_KEYS else hf)[name]
            for name in names
            if name in (gen if name in cls._GEN_KEYS else hf)
        }
        return cls(**values)

    def decoder_prompt_ids(self, language: str = "en", task: str = "transcribe") -> list[int]:
        """``<|startoftranscript|><|{lang}|><|{task}|><|notimestamps|>``."""
        lang_token = f"<|{language}|>"
        if lang_token not in self.lang_to_id:
            raise ValueError(
                f"Unknown Whisper language {language!r}; "
                f"available: {sorted(t.strip('<|>') for t in self.lang_to_id)}"
            )
        if task not in self.task_to_id:
            raise ValueError(
                f"Unknown Whisper task {task!r}; available: {list(self.task_to_id)}"
            )
        return [
            self.decoder_start_token_id,
            self.lang_to_id[lang_token],
            self.task_to_id[task],
            self.no_timestamps_token_id,
        ]
