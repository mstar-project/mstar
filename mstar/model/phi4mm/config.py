"""Phi-4-multimodal's configs and keys, read from the checkpoint's ``config.json``.

The LLM is Phi-4-mini (``Phi4MMConfig``); the vision tower and the audio conformer
keep their own configs beside their modules (``components/vision.py``,
``components/audio.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mstar.engine.resources.position.config import LongRoPE
from mstar.model.keys import KV_CACHE, ROPE, VISION_ATTN  # noqa: F401  (the scaffold's keys)

# the LLM's attention; KV, positions, sampler and the vision tower's attention take
# the scaffold's keys
ATTN = "attn"

# the pads the processor writes for each attachment, expanded to one per embedding row
IMAGE_PAD_ID = 200010  # <|endoftext10|>
AUDIO_PAD_ID = 200011  # <|endoftext11|>

# the modality LoRAs' adapter names, as the serving config's `resources: lora:
# adapters:` names them
VISION_ADAPTER = "vision"
SPEECH_ADAPTER = "speech"


def read_hf_config(local_dir: str | Path) -> dict:
    return json.loads((Path(local_dir) / "config.json").read_text())


@dataclass
class Phi4MMConfig:
    """Phi-4-mini, the LLM node."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    longrope: LongRoPE
    tie_word_embeddings: bool
    # generation_config.json's eos ids, unioned with the tokenizer's by the model
    stop_token_ids: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_hf(cls, hf: dict) -> "Phi4MMConfig":
        return cls(
            hidden_size=hf["hidden_size"],
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            vocab_size=hf["vocab_size"],
            rms_norm_eps=hf["rms_norm_eps"],
            rope_theta=hf["rope_theta"],
            partial_rotary_factor=hf.get("partial_rotary_factor", 1.0),
            max_position_embeddings=hf["max_position_embeddings"],
            longrope=LongRoPE.from_hf(hf),
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
        )
