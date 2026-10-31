"""Qwen3.5's text and vision configs, read off the HF ``config.json``.

Field names mirror HF's so ``from_hf`` can splat the checkpoint's own dict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from mstar.model.components.qwen_vision import QwenVisionConfig

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# text prefill and decode rows batched in one step (Model.get_combined_graph_walks)
LLM_MIXED = "mixed"
# Prompt rows a mixed step may carry: its GDN runs them on the chunked kernel,
# which gathers and scatters each one's whole state (`GDNMixedWrapper`), and
# the decode rows on the decode kernel.
MIXED_MAX_PREFILL_ROWS = 4

# resource keys this model declares
KV_CACHE = "kv_cache"
ATTN = "attn"
GDN_STATE = "gdn_state"
LINEAR_ATTN = "linear_attn"
ROPE = "rope"
SAMPLER = "sampler"
# the ViT tower's cacheless attention, on the `vision_encoder` node
VISION_ATTN = "vision_attn"


# The ViT tower's config is the shared Qwen one (read off ``config.json``'s
# ``vision_config``); Qwen3.5 refuses deepstack in ``Qwen3_5VisionModel``.
Qwen3_5VisionConfig = QwenVisionConfig


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
    eos_token_id: int | list[int] | None = None
    # multimodal sentinels, at the root of `config.json`, not `text_config`
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    rope_theta: float = 10_000.0
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    partial_rotary_factor: float = 1.0
    mtp_num_hidden_layers: int = 0
    mamba_ssm_dtype: str = "float32"
    # Filled from the tokenizer at model construction; see `stop_token_ids`.
    extra_stop_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_hf(cls, path: str | Path) -> "Qwen3_5Config":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())
        tc = raw.get("text_config", raw)
        # root first so `text_config` wins shared names; rope one level down
        merged = {**raw, **tc, **tc.get("rope_parameters", {})}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in known})

    # Derived

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """What ``config.json`` names; one id or several."""
        if self.eos_token_id is None:
            return frozenset()
        if isinstance(self.eos_token_id, int):
            return frozenset({self.eos_token_id})
        return frozenset(self.eos_token_id)

    @property
    def stop_token_ids(self) -> frozenset[int]:
        """Every id that ends generation — wider than ``eos_token_ids``.

        Chat turns end on ``<|im_end|>``, but ``config.json`` names only
        ``<|endoftext|>``, so the model adds the tokenizer's eos via
        ``extra_stop_token_ids``; without it every reply runs to the budget.
        """
        return self.eos_token_ids | frozenset(self.extra_stop_token_ids)

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

        The KV cache holds only full-attention layers and the recurrent pool
        only linear ones; a stack index would read another layer's state.
        """
        same = (
            self.linear_layer_indices
            if self.layer_types[layer_idx] == LINEAR_ATTENTION
            else self.full_layer_indices
        )
        return same.index(layer_idx)
