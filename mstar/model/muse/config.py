"""Muse Glimmer's text and vision configs, read off the HF ``config.json``.

Field names mirror HF's so ``from_hf`` can splat the checkpoint's own dicts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

from mstar.model import keys

logger = logging.getLogger(__name__)

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


# a mixture-of-experts text config, which this dense port would load wrong
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size")


def _read_config(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


# keys of a config section that this port leaves unread on purpose: bookkeeping, settings
# for training, or what `from_hf` checks itself
_UNREAD = frozenset({
    "model_type", "architectures", "dtype", "torch_dtype", "transformers_version",
    "initializer_range", "use_cache", "bos_token_id", "pad_token_id", "attention_dropout",
    "attention_bias", "hidden_activation", "hidden_act", "rope_parameters", "rope_scaling",
    "rope_type", "max_position_embeddings", "_name_or_path",
})


def _known(cls, merged: dict, section: str) -> dict:
    """``merged``'s fields of ``cls``, naming in a warning the keys it drops."""
    known = {f.name for f in fields(cls)}
    dropped = sorted(set(merged) - known - _UNREAD)
    if dropped:
        logger.warning("Muse Glimmer's %s has keys this port does not read: %s", section, dropped)
    return {k: v for k, v in merged.items() if k in known}


def _plain_rope(section: dict, where: str) -> dict:
    """``section``'s ``rope_parameters``, refusing a rope type or scaling this
    port does not build (it reads only ``rope_theta``)."""
    rope = section.get("rope_parameters") or {}
    if (
        section.get("rope_scaling")
        or rope.get("rope_type", "default") != "default"
        or set(rope) - {"rope_type", "rope_theta"}
    ):
        raise ValueError(
            f"the {where} rope is a default one here, without scaling; "
            f"the config's is {section.get('rope_scaling') or rope}"
        )
    return rope


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
    # from the root of config.json: the adapter in front of the LLM
    out_hidden_size: int
    projector_hidden_size: int
    text_hidden_size: int
    # the text stack's, for the weightless norm after the projection
    text_rms_norm_eps: float = 1e-5
    projector_hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    rope_theta: float = 10_000.0

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
            **vc, **_plain_rope(vc, "vision tower's"),
            "out_hidden_size": raw["out_hidden_size"],
            "projector_hidden_size": raw["projector_hidden_size"],
            "projector_hidden_act": raw.get("projector_hidden_act", "gelu"),
            "text_hidden_size": raw["text_config"]["hidden_size"],
        }
        if "rms_norm_eps" in raw["text_config"]:
            merged["text_rms_norm_eps"] = raw["text_config"]["rms_norm_eps"]
        config = cls(**_known(cls, merged, "vision_config"))
        if config.pos_emb_height != config.pos_emb_width:
            raise ValueError("Muse Glimmer's vision tower assumes a square position table")
        # any other kind would run as a full-frame layer
        kinds = set(config.layer_types) - {WINDOW_ATTENTION, FULL_ATTENTION}
        if kinds:
            raise ValueError(
                f"the vision tower runs window and full attention layers; the config's are also {sorted(kinds)}"
            )
        if config.projector_hidden_act != "gelu":
            raise ValueError(
                f"the vision adapter's activation is gelu here, not {config.projector_hidden_act}"
            )
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
    # root of config.json
    image_token_id: int
    # the reference config's defaults
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    rope_theta: float = 500_000.0
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = None
    # generation_config.json's eos list; see `from_hf`
    generation_eos_token_ids: tuple[int, ...] = ()
    # the tokenizer's, set by the model: a stop whether or not generation_config.json names it
    eot_token_id: int | None = None

    @classmethod
    def from_hf(cls, path: str | Path) -> MuseGlimmerConfig:
        raw = _read_config(path)
        tc = raw["text_config"]
        merged = {
            **tc, **_plain_rope(tc, "text stack's"),
            "image_token_id": raw["image_token_id"],
        }
        moe = sorted(k for k in _MOE_KEYS if tc.get(k))
        if moe:
            raise ValueError(f"this port is the dense Muse Glimmer; the config has MoE keys {moe}")
        # the 30B's config names it hidden_activation
        act = tc.get("hidden_activation", tc.get("hidden_act", "silu"))
        if act != "silu":
            raise ValueError(f"the MLP's activation is silu here, not {act}")
        if tc.get("attention_bias"):
            raise ValueError("the attention projections have no bias here")
        config = cls(**_known(cls, merged, "text_config"))
        config._check_rope()
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
        if self.eot_token_id is not None:
            ids.add(self.eot_token_id)
        if isinstance(self.eos_token_id, int):
            ids.add(self.eos_token_id)
        elif self.eos_token_id:
            ids.update(self.eos_token_id)
        return frozenset(ids)

    def _check_rope(self) -> None:
        """Layers of both kinds, and exactly the sliding ones rotate, all at
        ``rope_theta``: the one position resource the model declares."""
        kinds = sorted(set(self.layer_types))
        if kinds != sorted({SLIDING_ATTENTION, FULL_ATTENTION}):
            raise ValueError(
                f"this port runs sliding and full attention layers, both; the config's are {kinds}"
            )
        for i, (kind, theta) in enumerate(zip(self.layer_types, self.layer_rope_theta, strict=True)):
            if (kind == SLIDING_ATTENTION) != bool(theta) or (theta and theta != self.rope_theta):
                raise ValueError(
                    f"layer {i} is {kind} with rope theta {theta}; this port rotates exactly "
                    f"the sliding layers, at rope_theta {self.rope_theta}"
                )

    @property
    def sliding_layers(self) -> list[bool]:
        return [t == SLIDING_ATTENTION for t in self.layer_types]

    @property
    def attn_scale(self) -> float:
        """The query multiplier after QK-norm, folded into the softmax scale."""
        return self.qk_scale_factor / self.head_dim ** 0.5
