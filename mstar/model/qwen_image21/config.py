"""Qwen-Image-2.1's configuration, read from the checkpoint's component configs; what the
reference keeps in pipeline code (the prompt templates, the 16x latent grid) is below."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.flow_match import FlowMatchConfig, ShiftMode
from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig
from mstar.model.components.qwen_vision import QwenVisionConfig

QWEN_IMAGE21 = "Qwen/Qwen-Image-2.1"

# Graph names shared by the model file and the submodules.
TEXT_ENCODER_NODE = "text_encoder"
VAE_ENCODER_NODE = "vae_encoder"
DIT_NODE = "dit"
VAE_DECODER_NODE = "vae_decoder"
ENCODE_WALK = "encode"
ENCODE_IMAGES_WALK = "encode_images"
PREFIX_TEXT_WALK = "prefix_text"
PREFIX_IMAGE_WALK = "prefix_image"
IMAGE_GEN_WALK = "image_gen"
DENOISE_LOOP = "denoise_loop"
# Resource keys and KV labels.
KV = "kv"
ATTN = "attn"
# the denoise target's attention when its steps run eager: one dense varlen pass over [gathered prefix | the
# step's own K/V] (AttnBackend.DENSE), not a K/V write into the pages and a paged kernel every step
ATTN_GEN = "attn_gen"
MAIN = "main"
UNCOND = "uncond"

# The VAE compresses 16x and the transformer reads latents unpatched; sizes snap to 32 px.
VAE_SCALE = 16
SIZE_MULTIPLE = 32
MAX_REFERENCE_IMAGES = 10
DEFAULT_OUTPUT_RESOLUTION = 1024
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def _fields(cls, cfg: dict) -> dict:
    names = cls.__dataclass_fields__
    return {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _image_processor_config(directory: Path) -> dict:
    """The image processor's settings: ``preprocessor_config.json``, or the ``image_processor`` entry of
    ``processor_config.json`` where a checkpoint saves its processor whole (the Turbo repo does)."""
    if (directory / "preprocessor_config.json").exists():
        return _read(directory / "preprocessor_config.json")
    return _read(directory / "processor_config.json")["image_processor"]


@dataclass(frozen=True)
class QwenImage21TransformerConfig:
    """``transformer/config.json`` (``QwenImage21Transformer2DModel``)."""

    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "QwenImage21TransformerConfig":
        out = _fields(cls, cfg)
        if out.get("out_channels") is None:
            out["out_channels"] = out.get("in_channels", cls.in_channels)
        return cls(**out)


def _default_text() -> QwenEncoderConfig:
    # Qwen3-VL-8B's language model, tapped at its last layer before the final norm (untied). q/k/v run
    # as three GEMMs, as transformers does: fused, its 1024-wide k/v round differently in bf16.
    return QwenEncoderConfig(
        vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, rope_theta=5_000_000.0, hidden_state_layers=(36,),
        tie_last_hidden_states=False, max_sequence_length=262_144, fused_qkv=False, image_token_id=151655,
        mrope_section=(24, 20, 20),
    )


def _default_vision() -> QwenVisionConfig:
    return QwenVisionConfig(
        depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, in_channels=3, patch_size=16,
        temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=2304, out_hidden_size=4096,
        deepstack_visual_indexes=(8, 16, 24),
    )


def _default_scheduler() -> FlowMatchConfig:
    return FlowMatchConfig(
        shift=1.0, shift_mode=ShiftMode.DYNAMIC, base_shift=0.5, max_shift=0.9, base_image_seq_len=256,
        max_image_seq_len=8192, shift_terminal=0.02,
    )


@dataclass(frozen=True)
class QwenImage21Config:
    transformer: QwenImage21TransformerConfig = field(default_factory=QwenImage21TransformerConfig)
    text: QwenEncoderConfig = field(default_factory=_default_text)
    vision: QwenVisionConfig = field(default_factory=_default_vision)
    scheduler: FlowMatchConfig = field(default_factory=_default_scheduler)
    # A distilled checkpoint's fixed grid (``model_index.json``'s ``sample_sigmas``); its length is
    # the step count, and the scheduler applies nothing to it (shift 1, no dynamic shift).
    sample_sigmas: tuple[float, ...] | None = None
    default_num_inference_steps: int = 40
    # Qwen3-VL's image processor bounds (``size``), in pixels
    min_pixels: int = 65536
    max_pixels: int = 16_777_216

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "QwenImage21Config":
        snapshot = Path(snapshot)
        text_cfg = _read(snapshot / "text_encoder" / "config.json")
        layers = int(text_cfg.get("text_config", text_cfg)["num_hidden_layers"])
        processor = _image_processor_config(snapshot / "processor")
        sigmas = _read(snapshot / "model_index.json").get("sample_sigmas")
        scheduler_cfg = _read(snapshot / "scheduler" / "scheduler_config.json")
        if sigmas is not None and (scheduler_cfg.get("use_dynamic_shifting") or float(scheduler_cfg.get("shift", 1.0))
                                   != 1.0 or scheduler_cfg.get("shift_terminal")):
            raise ValueError("a fixed sample_sigmas grid with a shifting scheduler is not supported")
        return cls(
            transformer=QwenImage21TransformerConfig.from_dict(_read(snapshot / "transformer" / "config.json")),
            text=QwenEncoderConfig.from_qwen3_vl(
                text_cfg, hidden_state_layers=(layers,), tie_last_hidden_states=False,
                max_sequence_length=int(text_cfg["text_config"].get("max_position_embeddings", 262_144)),
                fused_qkv=False,
            ),
            vision=QwenVisionConfig.from_hf(snapshot / "text_encoder"),
            scheduler=FlowMatchConfig.from_scheduler_config(scheduler_cfg),
            sample_sigmas=None if sigmas is None else tuple(float(s) for s in sigmas),
            default_num_inference_steps=40 if sigmas is None else len(sigmas),
            min_pixels=int(processor.get("size", {}).get("shortest_edge", 65536)),
            max_pixels=int(processor.get("size", {}).get("longest_edge", 16_777_216)),
        )
