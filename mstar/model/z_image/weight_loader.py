"""Streaming checkpoint -> native module loading for Z-Image-Turbo.

Same pattern as the klein loader (meta build, cast, ``to_empty``, stream the shards,
completeness check). The transformer checkpoint is stored in fp32 and cast to the serving
dtype on copy — the same round-to-nearest cast ``from_pretrained(torch_dtype=bfloat16)``
performs, so the loaded bits match the reference module parameter for parameter.

Key remaps (checkpoint -> native):

    t_embedder.mlp.{0,2}                        -> time_in / time_out
    all_x_embedder.2-1                           -> x_embedder
    cap_embedder.{0,1}                           -> cap_norm / cap_proj
    {layers,noise_refiner,context_refiner}.N.attention.{to_q,to_k,to_v} -> ....attn.qkv (shards q,k,v)
    ....attention.{norm_q,norm_k,to_out.0}      -> ....attn.{q_norm,k_norm,out}
    ....feed_forward.{w1,w3}                     -> ....ff.w13 (shards w1,w3);  feed_forward.w2 -> ff.w2
    ....{attention_norm1,attention_norm2}       -> ....{attn_norm1,attn_norm2}      (ffn norms keep their names)
    ....adaLN_modulation.0                       -> ....adaln
    all_final_layer.2-1.adaLN_modulation.1       -> final_mod;   all_final_layer.2-1.linear -> final_proj
    x_pad_token / cap_pad_token                  unchanged
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from mstar.model.components.diffusion.autoencoder_kl import AutoencoderKL, remap_autoencoder_kl_key
from mstar.model.components.diffusion.text_encoder import Qwen3HiddenStateEncoder
from mstar.model.flux2_klein.weight_loader import (
    _QKV_RULES_LM,
    _materialize,
    iter_diffusers_component,
    iter_transformers_component,
    load_native,
    make_text_encoder,
    remap_text_encoder_key,
    text_encoder_skip,
)
from mstar.model.loader.base import StackedParamRule
from mstar.model.z_image.components.transformer import ZImageDiT
from mstar.model.z_image.config import ZImageConfig

_STACKED_RULES = [
    StackedParamRule(".attn.qkv", ".attention.to_q", "q"),
    StackedParamRule(".attn.qkv", ".attention.to_k", "k"),
    StackedParamRule(".attn.qkv", ".attention.to_v", "v"),
    StackedParamRule(".ff.w13", ".feed_forward.w1", "w1"),
    StackedParamRule(".ff.w13", ".feed_forward.w3", "w3"),
]

_TOP_LEVEL = {
    "t_embedder.mlp.0.": "time_in.",
    "t_embedder.mlp.2.": "time_out.",
    "all_x_embedder.2-1.": "x_embedder.",
    "cap_embedder.0.": "cap_norm.",
    "cap_embedder.1.": "cap_proj.",
    "all_final_layer.2-1.adaLN_modulation.1.": "final_mod.",
    "all_final_layer.2-1.linear.": "final_proj.",
}
_BLOCK_MAP = {
    ".attention.norm_q.": ".attn.q_norm.",
    ".attention.norm_k.": ".attn.k_norm.",
    ".attention.to_out.0.": ".attn.out.",
    ".feed_forward.w2.": ".ff.w2.",
    ".attention_norm1.": ".attn_norm1.",
    ".attention_norm2.": ".attn_norm2.",
    ".adaLN_modulation.0.": ".adaln.",
}
_BLOCK_RE = re.compile(r"^(layers|noise_refiner|context_refiner)\.\d+\.")


def remap_transformer_key(name: str) -> str:
    for src, dst in _TOP_LEVEL.items():
        if name.startswith(src):
            return dst + name[len(src):]
    if _BLOCK_RE.match(name):
        for src, dst in _BLOCK_MAP.items():
            name = name.replace(src, dst)
    return name


def build_transformer(config: ZImageConfig, snapshot: Path, device, dtype=torch.bfloat16) -> ZImageDiT:
    with torch.device("meta"):
        dit = ZImageDiT(config.transformer)
    _materialize(dit, dtype, device)
    return load_native(
        dit, iter_diffusers_component(snapshot / "transformer", device), remap_transformer_key,
        "Z-Image transformer", stacked_params=_STACKED_RULES,
    )


def build_vae(config: ZImageConfig, snapshot: Path, device, dtype=torch.bfloat16) -> AutoencoderKL:
    with torch.device("meta"):
        vae = AutoencoderKL(config.vae)
    _materialize(vae, dtype, device)
    return load_native(vae, iter_diffusers_component(snapshot / "vae", device), remap_autoencoder_kl_key, "Z-Image VAE")


def build_text_encoder(config: ZImageConfig, snapshot: Path, device, dtype=torch.bfloat16) -> Qwen3HiddenStateEncoder:
    with torch.device("meta"):
        encoder = make_text_encoder(config.text_encoder)
    _materialize(encoder, dtype, device)
    return load_native(
        encoder, iter_transformers_component(snapshot / "text_encoder", device), remap_text_encoder_key,
        "Qwen3 caption encoder", stacked_params=_QKV_RULES_LM, skip=text_encoder_skip(config.text_encoder),
    )
