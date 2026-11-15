"""Ripple's config, read off the HF ``config.json``."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from mstar.model import keys

KV_CACHE = keys.KV_CACHE
ROPE = keys.ROPE
SAMPLER = keys.SAMPLER
# one attention resource per window over the one cache: the sliding layers and the full (NoPE) ones
ATTN_SLIDING = "attn_sliding"
ATTN_FULL = "attn_full"

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# Would load fine and compute something else: the layer here is built for these values.
_PINNED = {
    "model_type": "ricursive",
    "hidden_act": "silu",
    "use_valres": True,
    "use_gated_attention": True,
    "gate_full_precision": True,
    "tie_word_embeddings": False,
}


@dataclass(frozen=True)
class RippleConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    sliding_window: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    embed_scale: float
    ffn_clamp_threshold: float
    stop_token_ids: tuple[int, ...]

    @property
    def is_sliding(self) -> tuple[bool, ...]:
        return tuple(kind == SLIDING_ATTENTION for kind in self.layer_types)

    @classmethod
    def from_dict(cls, data: dict) -> RippleConfig:
        for name, supported in _PINNED.items():
            if data.get(name) != supported:
                raise ValueError(f"Ripple's {name} must be {supported!r}; got {data.get(name)!r}")
        if data.get("quantization_config"):
            raise ValueError("only the BF16 Ripple checkpoint is served; this one is quantized")
        for name in ("embed_scale", "ffn_clamp_threshold"):
            if not data.get(name):
                raise ValueError(f"Ripple needs {name}; got {data.get(name)!r}")
        if set(data["layer_types"]) != {SLIDING_ATTENTION, FULL_ATTENTION}:
            raise ValueError(f"Ripple interleaves sliding and full layers; got {sorted(set(data['layer_types']))}")
        eos = data["eos_token_id"]
        derived = {
            "layer_types": tuple(data["layer_types"]),
            "stop_token_ids": tuple(eos) if isinstance(eos, list) else (eos,),
        }
        return cls(**{f.name: derived.get(f.name, data.get(f.name)) for f in fields(cls)})

    @classmethod
    def from_hf(cls, path: str | Path) -> RippleConfig:
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))
