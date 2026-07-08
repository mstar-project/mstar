"""Config for Whisper encoder-decoder ASR models (large-v3 and friends).

Values are read from the HF checkpoint's ``config.json`` +
``generation_config.json`` so the same class serves any Whisper size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "WhisperModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)

        cfg = cls(
            d_model=hf["d_model"],
            decoder_layers=hf["decoder_layers"],
            decoder_attention_heads=hf["decoder_attention_heads"],
            decoder_ffn_dim=hf["decoder_ffn_dim"],
            encoder_layers=hf["encoder_layers"],
            num_mel_bins=hf["num_mel_bins"],
            vocab_size=hf["vocab_size"],
            max_target_positions=hf["max_target_positions"],
            max_source_positions=hf["max_source_positions"],
            activation_function=hf.get("activation_function", "gelu"),
            scale_embedding=hf.get("scale_embedding", False),
            decoder_start_token_id=hf["decoder_start_token_id"],
            eos_token_id=hf["eos_token_id"],
        )

        gen_path = local_dir / "generation_config.json"
        if gen_path.exists():
            with open(gen_path) as f:
                gen = json.load(f)
            cfg.lang_to_id = gen.get("lang_to_id", {})
            cfg.task_to_id = gen.get("task_to_id", {})
            cfg.no_timestamps_token_id = gen.get(
                "no_timestamps_token_id", cfg.no_timestamps_token_id
            )
            cfg.suppress_tokens = gen.get("suppress_tokens", []) or []
            cfg.begin_suppress_tokens = gen.get("begin_suppress_tokens", []) or []
        return cfg

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
