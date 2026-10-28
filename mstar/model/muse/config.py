"""Muse Glimmer's text and vision configs, read off the HF ``config.json``.

Field names mirror HF's so ``from_hf`` can splat the checkpoint's own dicts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

from mstar.model import keys

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
WINDOW_ATTENTION = "window_attention"

# resource keys this model declares; the shared scaffolds read `keys`' ones
KV_CACHE = keys.KV_CACHE
# one attention resource per window over the one cache: the 3-in-4 sliding
# layers and the full (NoPE) layers plan different FlashInfer kernels
ATTN_SLIDING = "attn_sliding"
ATTN_FULL = "attn_full"
ROPE = keys.ROPE
SAMPLER = keys.SAMPLER
# the ViT tower's cacheless attention, on the `vision_encoder` node
VISION_ATTN = keys.VISION_ATTN
# the tower's two ragged layouts: 32x32-patch windows, and whole frames
VISION_WINDOW = "window"
VISION_FULL = "full"


def _read_config(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


@dataclass
class MuseGlimmerVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    patch_temporal: int
    merge_size: int
    pos_emb_height: int
    pos_emb_width: int
    layer_types: list[str]
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    rope_theta: float = 10_000.0
    # from the root of config.json: the adapter in front of the LLM
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    text_hidden_size: int = 6656

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        return self.patch_temporal * 3 * self.patch_size ** 2

    @property
    def window_patches(self) -> int:
        """A window is one learned-table's worth of patches on a side."""
        return self.pos_emb_height

    @classmethod
    def from_hf(cls, path: str | Path) -> MuseGlimmerVisionConfig:
        raw = _read_config(path)
        vc = raw["vision_config"]
        merged = {
            **vc, **vc.get("rope_parameters", {}),
            "out_hidden_size": raw.get("out_hidden_size", 6144),
            "projector_hidden_size": raw.get("projector_hidden_size", 4096),
            "projector_hidden_act": raw.get("projector_hidden_act", "gelu"),
            "text_hidden_size": raw["text_config"]["hidden_size"],
        }
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in merged.items() if k in known})
        if config.pos_emb_height != config.pos_emb_width:
            raise NotImplementedError("Muse Glimmer's vision tower assumes a square position table")
        return config


@dataclass
class MuseGlimmerConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: list[str]
    # 0 for a NoPE layer (the full-attention ones)
    layer_rope_theta: list[float]
    sliding_window: int
    max_position_embeddings: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    rope_theta: float = 500_000.0
    tie_word_embeddings: bool = False
    # root of config.json
    image_token_id: int = 200092
    video_token_id: int = 200091
    eos_token_id: int | list[int] | None = None
    # generation_config.json's eos list; see `from_hf`
    generation_eos_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_hf(cls, path: str | Path) -> MuseGlimmerConfig:
        raw = _read_config(path)
        tc = raw["text_config"]
        merged = {
            **tc, **tc.get("rope_parameters", {}),
            "image_token_id": raw.get("image_token_id", 200092),
            "video_token_id": raw.get("video_token_id", 200091),
        }
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in merged.items() if k in known})
        gen = Path(path) / "generation_config.json" if Path(path).is_dir() else None
        if gen is not None and gen.exists():
            g = json.loads(gen.read_text())
            eos = g.get("eos_token_id")
            config.generation_eos_token_ids = tuple(
                eos if isinstance(eos, list) else [eos] if eos is not None else []
            )
        return config

    @cached_property
    def stop_token_ids(self) -> frozenset[int]:
        """``<|end_of_text|>`` and ``<|eot|>``.

        ``<|eom|>`` is not one: it closes the reasoning channel and all but the
        last of parallel tool calls, and the model goes on after it.
        """
        ids = set(self.generation_eos_token_ids)
        if isinstance(self.eos_token_id, int):
            ids.add(self.eos_token_id)
        elif self.eos_token_id:
            ids.update(self.eos_token_id)
        return frozenset(ids)

    @property
    def sliding_layers(self) -> list[bool]:
        return [t == SLIDING_ATTENTION for t in self.layer_types]

    @property
    def attn_scale(self) -> float:
        """The query multiplier after QK-norm, folded into the softmax scale."""
        return self.qk_scale_factor / self.head_dim ** 0.5
