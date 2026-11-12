"""A tiny random MiniMax-H3 transformer checkpoint (original FL2VA format, or diffusers' layout)
and the config files of a snapshot around it, for tests that need no diffusers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mstar.model.minimax_h3.components.transformer import rope_inv_freq
from mstar.model.minimax_h3.config import PARTITION, H3TransformerConfig

CFG = H3TransformerConfig(
    hidden_size=48, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
    ffn_hidden_size=64, latents_dim=4, audio_latents_dim=8, text_dim=32, time_embed_hidden_size=32,
    time_embed_dim=16, adaln_out_features=6 * 48 * 3, final_adaln_out_features=2 * 48, rope_inv_freq_len=4,
)


def random_checkpoint(cfg: H3TransformerConfig) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def t(*shape, scale=0.2):
        return torch.randn(*shape, generator=g) * scale

    h, inner, ffn, hd = cfg.hidden_size, cfg.inner_dim, cfg.ffn_hidden_size, cfg.attention_head_dim

    def block(prefix: str, adaln: bool) -> dict:
        out = {
            f"{prefix}.norm1.weight": 1 + t(h), f"{prefix}.norm2.weight": 1 + t(h),
            f"{prefix}.attn.qkv_proj.weight": t(3 * inner, h), f"{prefix}.attn.q_norm.weight": 1 + t(hd),
            f"{prefix}.attn.k_norm.weight": 1 + t(hd), f"{prefix}.attn.out_proj.weight": t(h, inner),
            f"{prefix}.mlp.fc1.weight": t(2 * ffn, h), f"{prefix}.mlp.fc2.weight": t(h, ffn),
        }
        if adaln:
            out[f"{prefix}.adaln_proj.linear.weight"] = t(cfg.adaln_out_features, cfg.time_embed_dim)
            out[f"{prefix}.adaln_proj.linear.bias"] = t(cfg.adaln_out_features)
        return out

    state = {
        "video_patch_proj.weight": t(h, cfg.patch_dim), "video_patch_proj.bias": t(h),
        "audio_patch_proj.weight": t(h, cfg.audio_latents_dim), "audio_patch_proj.bias": t(h),
        "condition_proj.weight": t(h, cfg.text_dim), "condition_proj.bias": t(h),
        "time_embedder.proj_in.weight": t(cfg.time_embed_hidden_size, cfg.timestep_input_dim),
        "time_embedder.proj_in.bias": t(cfg.time_embed_hidden_size),
        "time_embedder.proj_out.weight": t(cfg.time_embed_dim, cfg.time_embed_hidden_size),
        "time_embedder.proj_out.bias": t(cfg.time_embed_dim),
        "rope.inv_freq": rope_inv_freq(cfg),
        "token_refiner.final_norm.weight": 1 + t(h),
        "final_layer.norm.weight": 1 + t(h),
        "final_layer.adaln_proj.linear.weight": t(cfg.final_adaln_out_features, cfg.time_embed_dim),
        "final_layer.adaln_proj.linear.bias": t(cfg.final_adaln_out_features),
        "final_layer.video_out.weight": t(cfg.patch_dim, h), "final_layer.video_out.bias": t(cfg.patch_dim),
        "final_layer.audio_out.weight": t(cfg.audio_latents_dim, h),
        "final_layer.audio_out.bias": t(cfg.audio_latents_dim),
    }
    for i in range(cfg.num_layers):
        state.update(block(f"blocks.{i}", adaln=True))
    for i in range(cfg.token_refiner_num_layers):
        state.update(block(f"token_refiner.blocks.{i}", adaln=False))
    return state


# diffusers' convert_minimax_h3_transformer_checkpoint_to_diffusers, restated
_TO_DIFFUSERS = {
    "token_refiner.blocks.": "token_refiner.refiner_blocks.", "time_embedder.proj_in.": "time_embedder.linear_1.",
    "time_embedder.proj_out.": "time_embedder.linear_2.", "video_patch_proj.": "proj_in.",
    "audio_patch_proj.": "audio_proj_in.", "condition_proj.": "context_embedder.",
    "final_layer.norm.": "norm_out.norm.", "final_layer.adaln_proj.linear.": "norm_out.linear.",
    "final_layer.video_out.": "proj_out.",
    "final_layer.audio_out.": "audio_proj_out.", ".attn.q_norm.": ".attn.norm_q.", ".attn.k_norm.": ".attn.norm_k.",
    ".attn.out_proj.": ".attn.to_out.0.", ".mlp.fc1.": ".ff.net.0.proj.", ".mlp.fc2.": ".ff.net.2.",
}


def to_diffusers(state: dict[str, torch.Tensor], cfg: H3TransformerConfig, vsa_gates: bool = True
                 ) -> dict[str, torch.Tensor]:
    """``state`` (FL2VA's layout) as a diffusers ``MiniMaxH3Transformer3DModel`` checkpoint: renamed,
    q/k/v unfused from their per-head interleave, the SwiGLU input reordered to ``[value; gate]``,
    no ``rope.inv_freq``; with ``vsa_gates``, a VSA compression gate per block, as FastH3 ships."""
    out = {}
    for name, tensor in state.items():
        if name == "rope.inv_freq":
            continue
        if name.startswith("blocks."):
            name = "transformer_blocks." + name[len("blocks."):]
        for native, diffusers in _TO_DIFFUSERS.items():
            name = name.replace(native, diffusers)
        if name.endswith(".attn.qkv_proj.weight"):
            q, k, v = tensor.unflatten(0, (cfg.num_attention_heads, 3, cfg.attention_head_dim)).unbind(1)
            for part, rows in (("to_q", q), ("to_k", k), ("to_v", v)):
                out[name.replace("qkv_proj", part)] = rows.flatten(0, 1).contiguous()
        elif name.endswith(".ff.net.0.proj.weight"):
            gate, value = tensor.chunk(2, dim=0)
            out[name] = torch.cat([value, gate])
        else:
            out[name] = tensor
    if vsa_gates:
        for i in range(cfg.num_layers):
            out[f"transformer_blocks.{i}.attn.to_gate_compress.weight"] = torch.full((cfg.inner_dim, cfg.hidden_size),
                                                                                     float("nan"))
    return out


def write_snapshot_configs(root: Path, cfg: H3TransformerConfig) -> None:
    """The json files ``H3Config.from_snapshot`` reads, with ``cfg`` as the transformer's
    (plus a key the reader must ignore) and the released checkpoint's other values."""
    part = Path(root) / PARTITION

    def write(rel: str, data: dict) -> None:
        path = part / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    transformer = {name: list(value) if isinstance(value, tuple) else value
                   for name, value in vars(cfg).items()}
    write("transformer/config.json", {"_class_name": "MiniMaxH3DiTModel", **transformer})
    write("text_encoder/config.json", {"image_token_id": 151655, "text_config": {
        "model_type": "qwen3_vl_text", "hidden_size": 5120, "intermediate_size": 25600, "num_hidden_layers": 64,
        "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-6,
        "vocab_size": 151936, "rope_theta": 5000000,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
    }})
    write("model_index.json", {"_minimax_h3": {"sigma_shift_scales": {"video": 11.0, "audio": 2.5}}})
    write("video_vae/config.json", {
        "source_path": "source", "latent_channels": 24, "vae_clip_length": 17, "vae_token_drop": 3,
        "vae_tile_size": 256, "vae_tile_overlap_min": 64, "latents_mean": [0.5] * 24, "latents_std": [2.0] * 24,
    })
    write("video_vae/source/config.json", {
        "vae_ratio": 16, "vae_ratio_t": 4, "in_channels": 3, "ch": 128, "ch_mult": [1, 2, 2, 4, 4, 8],
        "num_res_blocks": 2, "space_down": [2, 2, 2, 2, 1, 1], "time_down": [1, 2, 2, 1, 1, 1], "embed_dim": 24,
        "padding_mode": "reflect", "use_3d_conv": True, "causal_encoder": True, "use_t_isolated_gn": True,
        "vit_decoder_kwargs": {"num_layers": 36, "heads": 32, "dim_head": 64, "rope_theta": 100.0,
                               "rope_dim_ratio": 0.75},
    })
    write("audio_vae/config.json", {
        "sample_rate": 32000, "latent_channels": 32, "output_channel": 2, "source_metadata_path": "metadata.json",
        "latents_mean": [0.0] * 32, "latents_std": [1.0] * 32,
    })
    write("audio_vae/metadata.json", {"metadata": {"kwargs": {
        "latent_dim": 2048, "decoder_dim": 1024, "decoder_type": "bigvgan", "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
    }}})
