"""The original-format checkpoint into the native modules. The transformer, its text
conditioner, the Qwen3-VL text encoder and its vision tower load through MiniMax-H3's
builders (the same formats, at the snapshot's root), the transformer with the per-step
video heads of ``pdd_fused_heads.safetensors``; the VAE (``vae/``) loads here. Every
builder raises unless each kept key reaches a parameter and each parameter is reached.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.model.components.diffusion.weight_loading import iter_diffusers_component, load_native
from mstar.model.minimax_h3 import weight_loader as h3
from mstar.model.minimax_h3.components.video_encoder import H3KeyframeEncoder
from mstar.model.minimax_h3_image.components.image_vae import H3ImageDecoder
from mstar.model.minimax_h3_image.config import OUTPUT_HEADS, H3ImageConfig

ROOT = ""  # every component sits at the snapshot's root


def build_transformer(config: H3ImageConfig, snapshot: Path, device, comm_group=None,
                      dtype: torch.dtype = torch.bfloat16):
    return h3.build_transformer(config, snapshot, device, comm_group=comm_group, dtype=dtype, partition=ROOT,
                                output_heads=OUTPUT_HEADS)


def build_text_conditioner(config: H3ImageConfig, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    return h3.build_text_conditioner(config, snapshot, device, dtype=dtype, partition=ROOT)


def build_text_encoder(config: H3ImageConfig, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    return h3.build_text_encoder(config, snapshot, device, dtype=dtype, partition=ROOT)


def build_vision_tower(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    return h3.build_vision_tower(snapshot, device, dtype=dtype, partition=ROOT)


def vae_weights(snapshot: Path, device, skip):
    """``vae/``'s single file (named as diffusers names it, with the original keys), minus ``skip``."""
    return ((name, tensor) for name, tensor in iter_diffusers_component(Path(snapshot) / "vae", device)
            if not skip(name))


def build_image_decoder(config: H3ImageConfig, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                        ) -> H3ImageDecoder:
    with torch.device("meta"):
        vae = H3ImageDecoder(config.vae)
    vae.to(dtype)
    vae.keep_fp32()
    vae.to_empty(device=device)

    def skip(name: str) -> bool:
        return name.startswith(("encoder.", "quant_conv.")) or name == "decoder.mask_token"

    return load_native(
        vae, vae_weights(snapshot, device, skip), lambda n: n,
        "MiniMax-H3-Image VAE decoder", skip=skip,
    )


def build_image_encoder(config: H3ImageConfig, snapshot: Path, device) -> H3KeyframeEncoder:
    """The VAE's encoder half, fp32 as the reference pins it: H3's video encoder on one frame."""
    with torch.device("meta"):
        encoder = H3KeyframeEncoder(config.vae_encoder, config.vae)
    encoder.to_empty(device=device)

    def skip(name: str) -> bool:
        return not name.startswith(("encoder.", "quant_conv."))

    return load_native(
        encoder, vae_weights(snapshot, device, skip), lambda n: n,
        "MiniMax-H3-Image VAE encoder", skip=skip,
    )
