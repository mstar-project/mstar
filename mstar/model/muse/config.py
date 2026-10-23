"""The Muse family's text and vision configs, read off the HF ``config.json``:
Muse Glimmer's (dense) and Muse Spark Open's (mixture of experts).

Field names mirror HF's so ``from_hf`` can splat the checkpoint's own dicts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import ClassVar

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
# Muse Spark: the audio tower's convs' left context, carried between chunks
AUDIO_CONV = "audio_conv"


# a mixture-of-experts text config, which this dense port would load wrong
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size")


def _read_config(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


def _plain_rope(section: dict, where: str, rope_types: tuple[str, ...] = ("default",)) -> dict:
    """``section``'s ``rope_parameters``, refusing a rope type or scaling this
    port does not build (it reads only ``rope_theta``)."""
    rope = section.get("rope_parameters") or {}
    if (
        section.get("rope_scaling")
        or rope.get("rope_type", "default") not in rope_types
        or set(rope) - {"rope_type", "rope_theta"}
    ):
        raise ValueError(
            f"the {where} rope is a {' or '.join(rope_types)} one here, without scaling; "
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
    projector_hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    rope_theta: float = 10_000.0
    # Muse Spark's tower reads its position table through a (t, w, h) grid,
    # the reference's `meshgrid(..., indexing="xy")`; Glimmer's through (t, h, w).
    # They differ on non-square grids only.
    pos_table_xy: bool = False

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
            # the tower's 2D rope, which Muse Spark's config calls axial
            **vc, **_plain_rope(vc, "vision tower's", ("default", "axial")),
            "out_hidden_size": raw["out_hidden_size"],
            "projector_hidden_size": raw["projector_hidden_size"],
            "projector_hidden_act": raw.get("projector_hidden_act", "gelu"),
            "text_hidden_size": raw["text_config"]["hidden_size"],
            "pos_table_xy": vc.get("model_type") == "muse_spark_vision",
        }
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in merged.items() if k in known})
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
    # None: off (a plain softmax scale, unscaled logits, no softcap)
    qk_scale_factor: float | None = None
    output_multiplier: float | None = None
    final_logit_softcapping: float | None = None
    rope_theta: float = 500_000.0
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = None
    # generation_config.json's eos list; see `from_hf`
    generation_eos_token_ids: tuple[int, ...] = ()
    # the fields read from config.json's root rather than its text_config; required
    ROOT_FIELDS: ClassVar[tuple[str, ...]] = ("image_token_id",)

    @classmethod
    def _check_text_config(cls, raw: dict) -> None:
        moe = sorted(k for k in _MOE_KEYS if raw["text_config"].get(k))
        if moe:
            raise ValueError(f"this port is the dense Muse Glimmer; the config has MoE keys {moe}")

    @classmethod
    def from_hf(cls, path: str | Path) -> MuseGlimmerConfig:
        raw = _read_config(path)
        tc = raw["text_config"]
        merged = {
            **tc, **_plain_rope(tc, "text stack's"),
            **{k: raw[k] for k in cls.ROOT_FIELDS},
        }
        cls._check_text_config(raw)
        # the 30B's config names it hidden_activation
        act = tc.get("hidden_activation", tc.get("hidden_act", "silu"))
        if act != "silu":
            raise ValueError(f"the MLP's activation is silu here, not {act}")
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in merged.items() if k in known})
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
        return (self.qk_scale_factor or 1.0) / self.head_dim ** 0.5


@dataclass
class MuseSparkConfig(MuseGlimmerConfig):
    """Muse Spark Open's text stack: Glimmer's attention (with an output
    norm), residual gates, and a mixture of experts in a narrower latent."""

    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    # the latent the experts run in, between `pre_expert_proj` and `post_expert_proj`
    moe_hidden_size: int = 4096
    moe_intermediate_size: int = 4096
    routed_scaling_factor: float = 1.0
    residual_gate_temperature: float = 0.3
    audio_token_id: int | None = None
    ROOT_FIELDS: ClassVar[tuple[str, ...]] = ("image_token_id", "audio_token_id")

    @classmethod
    def _check_text_config(cls, raw: dict) -> None:
        tc = raw["text_config"]
        if raw.get("model_type") != "muse_spark":
            raise ValueError(f"not a Muse Spark config: model_type {raw.get('model_type')!r}")
        dense = [i for i, t in enumerate(tc.get("mlp_layer_types") or []) if t != "sparse"]
        if dense:
            raise ValueError(f"this port runs every layer as experts; layers {dense} are dense")
        if tc.get("num_shared_experts") or tc.get("n_group", 1) != 1 or tc.get("topk_group", 1) != 1:
            raise ValueError("this port's router has no shared experts and no expert groups")
        if tc.get("output_multiplier") is None:
            raise ValueError("Muse Spark's text_config needs output_multiplier: it scales the router's logits")



# audio_config keys that do not change the tower's computation
_AUDIO_CONFIG_UNREAD = {"initializer_range", "model_type"}
# What `_audio_intake` builds in: mono at 24 kHz, zero-padded on the right to
# whole tokens, with no normalization. `samples_per_token` is the tower's.
_FEATURE_EXTRACTOR = {
    "feature_extractor_type": "MuseSparkFeatureExtractor", "feature_size": 1, "sampling_rate": 24_000,
    "padding_side": "right", "padding_value": 0.0, "return_attention_mask": True,
}


@dataclass
class MuseSparkAudioConfig:
    """Muse Spark's waveform audio tower: a causal convolutional pyramid, one
    token per ``prod(patch_sizes)`` samples."""

    patch_sizes: list[int]
    hidden_sizes: list[int]
    depths: list[int]
    out_hidden_size: int
    conv_kernel_size: int = 7
    mlp_ratio: float = 4.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "gelu"
    # the text embedding's weightless norm, which the clips' tokens go through too
    text_rms_norm_eps: float = 1e-5
    sampling_rate: int = _FEATURE_EXTRACTOR["sampling_rate"]

    @property
    def samples_per_token(self) -> int:
        return math.prod(self.patch_sizes)

    @classmethod
    def from_hf(cls, path: str | Path) -> MuseSparkAudioConfig:
        """Refuses an ``audio_config`` key the tower does not build, and a
        ``processor_config.json`` feature extractor intake would misread."""
        raw = _read_config(path)
        ac = raw.get("audio_config")
        if ac is None:
            raise ValueError(
                "Muse Spark's config.json has no audio_config: this port builds the audio tower, so it needs one"
            )
        tower = {f.name for f in fields(cls)} - {"text_rms_norm_eps", "sampling_rate"}
        unknown = sorted(ac.keys() - tower - _AUDIO_CONFIG_UNREAD - {"num_hidden_layers"})
        if unknown:
            raise ValueError(f"audio_config's {unknown} are not keys this port reads")
        config = cls(
            **{k: v for k, v in ac.items() if k in tower},
            text_rms_norm_eps=raw["text_config"]["rms_norm_eps"],
        )
        if not len(config.patch_sizes) == len(config.hidden_sizes) == len(config.depths):
            raise ValueError("audio_config: patch_sizes, hidden_sizes and depths need one entry per stage")
        if ac.get("num_hidden_layers", len(config.depths)) != len(config.depths):
            raise ValueError("audio_config's num_hidden_layers is its stages here, one per entry of depths")
        processor = Path(path) / "processor_config.json" if Path(path).is_dir() else None
        if processor is not None and processor.exists():
            ours = {**_FEATURE_EXTRACTOR, "samples_per_token": config.samples_per_token}
            for key, value in json.loads(processor.read_text()).get("feature_extractor", {}).items():
                if key not in ours:
                    raise ValueError(f"feature_extractor's {key} is not one this port reads")
                if value != ours[key]:
                    raise ValueError(f"feature_extractor's {key} is {value!r}, this port's {ours[key]!r}")
        if config.hidden_act != "gelu":
            raise ValueError(f"the audio tower's activation is gelu here, not {config.hidden_act}")
        if config.out_hidden_size != raw["text_config"]["hidden_size"]:
            raise ValueError("the audio tower's output feeds the text embeddings: out_hidden_size != hidden_size")
        return config
