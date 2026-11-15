"""MiniMax-H3-Image's configuration, read from the checkpoint's root components; what the
reference keeps in pipeline code (the canvas rule, the reference sizes, the label, the
references' noise level) is below, named for diffusers' ``MiniMaxH3ImageModularPipeline``
properties and config entries.

It is MiniMax-H3's transformer without audio and with one output head per sampling step,
the same Qwen3-VL-32B conditioner, and an image VAE: H3's video encoder run on one frame
and a 2D ViT decoder.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.qwen3.encoder import Qwen3EncoderConfig
from mstar.model.minimax_h3.config import (
    TEXT_ENCODER_LAYER,
    TEXT_MAX_POSITIONS,
    H3Config,
    H3TransformerConfig,
    H3VideoEncoderConfig,
    H3VideoVAEConfig,
    _fields,
)

H3_IMAGE_REPO = "MiniMaxAI/MiniMax-H3-Image"
# The repo also holds diffusers copies of the transformer (66 GB) and the VAE; the native
# port loads the original-format ones and reads only the diffusers VAE's config.
SNAPSHOT_PATTERNS = [
    "model_index.json", "modular_model_index.json", "pdd_config.json", "pdd_fused_heads.safetensors",
    "transformer/*", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "vae_diffusers/config.json",
]
# One video head per sampling step (``video_out.weight [steps, 96, hidden]``), distilled with
# the transformer on the fixed grid of ``pdd_config.json``
OUTPUT_HEADS = "pdd_fused_heads.safetensors"

CANVAS_MULTIPLE = 32               # the VAE's 16 times the transformer's 2-wide patch
DEFAULT_SIZE = 2048
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MIN_GENERATION_AREA_RANGE = (1024, 4096 * 4096)
REFERENCE_SHORT_EDGE = (1024, 2048)  # the VAE's copy of a reference
VISION_SHORT_EDGE = (1024, 1024)     # the text encoder's copy
MAX_REFERENCES = 16
MAX_TEXT_TOKENS = 4096               # the prompt and each label are cut to this many tokens
PICTURE_LABEL = "Picture {}: "
# the references are noised to t = 0.999 and held there for every step
REFERENCE_NOISE_AUG = 0.999


# the released grid (pdd_config.json)
SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


@dataclass(frozen=True)
class H3ImageConfig:
    """Defaults are the released checkpoint's; ``from_snapshot`` reads them."""

    transformer: H3TransformerConfig = field(
        default_factory=lambda: H3TransformerConfig(audio_enabled=False, num_output_heads=len(SIGMAS) - 1))
    text: Qwen3EncoderConfig = field(default_factory=lambda: H3Config().text)
    vae: H3VideoVAEConfig = field(default_factory=lambda: H3VideoVAEConfig(
        clip_length=1, token_drop=0, decoder_layers=24, decoder_heads=16, decoder_head_dim=64,
        decoder_rope_dim_ratio=0.5, decoder_2d=True))
    vae_encoder: H3VideoEncoderConfig = field(default_factory=H3VideoEncoderConfig)
    # the fixed sampling grid, terminal 0 included (pdd_config.json)
    sigmas: tuple[float, ...] = SIGMAS
    image_token_id: int = 151655
    mrope_section: tuple[int, ...] = (24, 20, 20)
    text_max_positions: int = TEXT_MAX_POSITIONS

    @property
    def num_steps(self) -> int:
        return len(self.sigmas) - 1

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "H3ImageConfig":
        root = Path(snapshot)
        transformer = H3TransformerConfig(**_fields(H3TransformerConfig, json.loads(
            (root / "transformer" / "config.json").read_text())))
        pdd = json.loads((root / "pdd_config.json").read_text())
        sigmas = tuple(float(s) for s in pdd["sigmas"])
        if len(sigmas) - 1 != int(pdd["num_inference_steps"]) or sigmas[-1] != 0.0:
            raise ValueError(f"{root}/pdd_config.json: expected num_inference_steps + 1 sigmas ending at 0, got {pdd}")
        transformer = dataclasses.replace(transformer, num_output_heads=len(sigmas) - 1)
        transformer.validate()
        text_encoder = json.loads((root / "text_encoder" / "config.json").read_text())
        rope = text_encoder["text_config"].get("rope_scaling") or text_encoder["text_config"].get("rope_parameters", {})
        max_positions = int(text_encoder["text_config"].get("max_position_embeddings", TEXT_MAX_POSITIONS))
        text = Qwen3EncoderConfig.from_qwen3_vl(
            text_encoder, hidden_state_layers=(TEXT_ENCODER_LAYER,), fused_qkv=False, max_sequence_length=max_positions,
        )
        vae, vae_encoder = image_vae_configs(root)
        return cls(
            transformer=transformer, text=text, vae=vae, vae_encoder=vae_encoder, sigmas=sigmas,
            image_token_id=int(text_encoder["image_token_id"]),
            mrope_section=tuple(int(n) for n in rope.get("mrope_section", (24, 20, 20))),
            text_max_positions=max_positions,
        )


def image_vae_configs(root: Path) -> tuple[H3VideoVAEConfig, H3VideoEncoderConfig]:
    """``vae/config.json`` (the original-format checkpoint's decoder and tiles) and the encoder
    geometry, which only the diffusers copy's config spells out."""
    top = json.loads((root / "vae" / "config.json").read_text())
    geometry = json.loads((root / "vae_diffusers" / "config.json").read_text())
    vit = top["vit_decoder_kwargs"]
    if not top.get("vit_decoder_2d"):
        raise ValueError(f"{root}/vae: expected the 2D ViT decoder (vit_decoder_2d)")
    spatial, temporal = geometry["spatial_downsample_factors"], geometry["temporal_downsample_factors"]
    base = int(geometry["block_out_channels"][0])
    vae = H3VideoVAEConfig(
        latent_channels=int(top["latent_channels"]), clip_length=1, token_drop=0,
        spatial_compression=math.prod(spatial), temporal_compression=math.prod(temporal),
        latents_mean=tuple(float(x) for x in top["latents_mean"]),
        latents_std=tuple(float(x) for x in top["latents_std"]),
        tile_size=int(top.get("vae_tile_size", 256)), tile_overlap_min=int(top.get("vae_tile_overlap_min", 64)),
        decoder_layers=int(vit["num_layers"]), decoder_heads=int(vit["heads"]), decoder_head_dim=int(vit["dim_head"]),
        decoder_rope_theta=float(vit["rope_theta"]), decoder_rope_dim_ratio=float(vit["rope_dim_ratio"]),
        decoder_2d=True,
    )
    encoder = H3VideoEncoderConfig(
        in_channels=int(geometry["in_channels"]), base_channels=base,
        channel_mult=tuple(int(c) // base for c in geometry["block_out_channels"]),
        num_res_blocks=int(geometry["layers_per_block"]), spatial_down=tuple(int(s) for s in spatial),
        temporal_down=tuple(int(t) for t in temporal), latent_channels=int(geometry["latent_channels"]),
        padding_mode=str(geometry["spatial_padding_mode"]), norm_groups=int(geometry["norm_num_groups"]),
        norm_eps=float(geometry["norm_eps"]),
    )
    return vae, encoder


# --------------------------------------------------------------------------- geometry
# Ports of diffusers' modular_pipelines/minimax_h3_image helpers.

def canvas_size(height: int, width: int, min_generation_area: int, multiple: int = CANVAS_MULTIPLE
                ) -> tuple[int, int]:
    """``get_image_canvas_size``: an image below ``min_generation_area`` pixels is generated on
    a canvas scaled up to that area at the same aspect ratio; both sides round to ``multiple``."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (max(multiple, round(height * scale / multiple) * multiple),
            max(multiple, round(width * scale / multiple) * multiple))


def bound_short_edge(width: int, height: int, min_short_edge: int, max_short_edge: int,
                     multiple: int = CANVAS_MULTIPLE) -> tuple[int, int]:
    """``bound_short_edge``: the ``(width, height)`` a reference is resized to for one of its
    consumers; a short edge within the bounds keeps the original size."""
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))
