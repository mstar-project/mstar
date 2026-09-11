"""Qwen3.5's text-stack config, read off the HF ``config.json``.

Text tower only; the vision config is a follow-up. Field names mirror HF's so
``from_hf`` can splat the checkpoint's own dict and a reader can diff the two.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# resource keys this model declares
KV_CACHE = "kv_cache"
ATTN = "attn"
GDN_STATE = "gdn_state"
LINEAR_ATTN = "linear_attn"
ROPE = "rope"
SAMPLER = "sampler"


@dataclass
class Qwen3_5Config:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    # per layer, LINEAR_ATTENTION or FULL_ATTENTION
    layer_types: list[str]

    # full-attention layers
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # gated delta net layers
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int

    attn_output_gate: bool = False
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000.0
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    partial_rotary_factor: float = 1.0
    mtp_num_hidden_layers: int = 0
    mamba_ssm_dtype: str = "float32"

    @classmethod
    def from_hf(cls, path: str | Path) -> "Qwen3_5Config":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())
        tc = raw.get("text_config", raw)
        # rope lives one level down; flatten it so the names line up
        merged = {**tc, **tc.get("rope_parameters", {})}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in known})

    # Derived

    @property
    def linear_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION]

    @property
    def full_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION]

    @property
    def rotary_dim(self) -> int:
        """Only part of each head is rotated; mrope_section covers half of it."""
        return int(self.head_dim * self.partial_rotary_factor)

    def resource_layer_index(self, layer_idx: int) -> int:
        """Where a stack layer sits among *its own* resource's layers.

        The KV cache is sized by the full-attention layers and the recurrent
        pool by the linear ones, so neither is indexed by stack position.
        Handing either the stack index reads another layer's state.
        """
        same = (
            self.linear_layer_indices
            if self.layer_types[layer_idx] == LINEAR_ATTENTION
            else self.full_layer_indices
        )
        return same.index(layer_idx)
