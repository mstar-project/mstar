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
class Qwen3_5VisionConfig:
    """The ViT tower's own config, read off ``config.json``'s ``vision_config``."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int

    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10_000.0
    # Dead config surface, kept only so a checkpoint that revives it fails
    # loudly: `[]` on all ten released sizes, and transformers' own
    # `modeling_qwen3_5` never reads the field. Qwen3-VL is where it is live
    # (default `[8, 16, 24]`), feeding intermediate blocks back into the LLM.
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """The learned position table is a square grid."""
        return int(self.num_position_embeddings**0.5)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @classmethod
    def from_hf(cls, path: str | Path) -> "Qwen3_5VisionConfig":
        config = cls.from_hf_or_none(path)
        if config is None:
            raise ValueError(f"{path} has no `vision_config`")
        return config

    @classmethod
    def from_hf_or_none(cls, path: str | Path) -> "Qwen3_5VisionConfig | None":
        """None when the checkpoint is text-only."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text()).get("vision_config")
        if raw is None:
            return None
        merged = {**raw, **raw.get("rope_parameters", {})}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in known})


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
    # Multimodal sentinels. These sit at the *root* of `config.json`, not under
    # `text_config`, because they are what joins the two towers.
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
        # Root first so `text_config` wins any name they share, and rope from
        # one level down so its names line up.
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

        The released checkpoints are chat models whose assistant turns end on
        ``<|im_end|>``, but ``config.json`` names only ``<|endoftext|>``.
        Stopping on the config's alone runs every reply to the token budget.
        The model fills ``extra_stop_token_ids`` from the tokenizer, which
        knows the real one.
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
