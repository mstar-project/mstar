"""Configuration for Z-Image-Turbo, read from the checkpoint's config.json files.

``ZImageConfig.from_snapshot`` reads ``transformer/config.json``, ``vae/config.json``
(a FLUX.1-style ``AutoencoderKL`` with ``scaling_factor`` / ``shift_factor``),
``text_encoder/config.json`` (Qwen3-4B) and ``scheduler/scheduler_config.json`` (linear
shift 3.0, no dynamic shifting). The pipeline-level recipe that diffusers keeps in code
— ``hidden_states[-2]`` of the caption encoder, ``enable_thinking=True`` in the chat
template, 512-token truncation, sequences padded to multiples of 32 with learned pad
tokens, 8 steps and no guidance for Turbo — is spelled out as defaults here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.autoencoder_kl import AutoencoderKLConfig
from mstar.model.components.diffusion.flow_match import FlowMatchConfig
from mstar.model.flux2_klein.config import Qwen3EncoderConfig, resolve_snapshot_dir

DIT_ATTN = "dit_attn"
DENOISE_LOOP = "denoise_loop"
Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"

# Every token sequence (image patches, caption) is padded to a multiple of this with a
# learned pad token (diffusers ``SEQ_MULTI_OF``).
SEQ_MULTIPLE = 32
# Width of the adaLN conditioning vector (``ADALN_EMBED_DIM``); the timestep MLP's output.
ADALN_DIM = 256


@dataclass(frozen=True)
class ZImageTransformerConfig:
    """``transformer/config.json`` of ``ZImageTransformer2DModel``."""

    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    in_channels: int = 16
    patch_size: int = 2
    f_patch_size: int = 1
    cap_feat_dim: int = 2560
    norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1536, 512, 512)
    # timestep MLP hidden width (diffusers ``TimestepEmbedder(mid_size=1024)``)
    t_mid_size: int = 1024

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(self.dim / 3 * 8)

    @property
    def patch_dim(self) -> int:
        return self.f_patch_size * self.patch_size * self.patch_size * self.in_channels

    @property
    def adaln_dim(self) -> int:
        return min(self.dim, ADALN_DIM)

    @classmethod
    def from_dict(cls, cfg: dict) -> "ZImageTransformerConfig":
        if cfg.get("siglip_feat_dim") is not None:
            raise NotImplementedError("Z-Image Omni (SigLIP conditioning) is not supported")
        if int(cfg.get("n_kv_heads", cfg["n_heads"])) != int(cfg["n_heads"]):
            raise NotImplementedError("Z-Image port implements n_kv_heads == n_heads")
        patch_sizes, f_patch_sizes = cfg.get("all_patch_size", [2]), cfg.get("all_f_patch_size", [1])
        if len(patch_sizes) != 1 or len(f_patch_sizes) != 1:
            raise NotImplementedError("Z-Image port implements a single patch size")
        return cls(
            dim=int(cfg["dim"]), n_layers=int(cfg["n_layers"]), n_refiner_layers=int(cfg["n_refiner_layers"]),
            n_heads=int(cfg["n_heads"]), in_channels=int(cfg["in_channels"]),
            patch_size=int(patch_sizes[0]), f_patch_size=int(f_patch_sizes[0]),
            cap_feat_dim=int(cfg["cap_feat_dim"]), norm_eps=float(cfg.get("norm_eps", 1e-5)),
            qk_norm=bool(cfg.get("qk_norm", True)), rope_theta=float(cfg.get("rope_theta", 256.0)),
            t_scale=float(cfg.get("t_scale", 1000.0)),
            axes_dims=tuple(int(d) for d in cfg["axes_dims"]), axes_lens=tuple(int(n) for n in cfg["axes_lens"]),
        )


@dataclass
class ZImageConfig:
    transformer: ZImageTransformerConfig = field(default_factory=ZImageTransformerConfig)
    vae: AutoencoderKLConfig = field(default_factory=lambda: AutoencoderKLConfig(
        latent_channels=16, scaling_factor=0.3611, shift_factor=0.1159,
    ))
    text_encoder: Qwen3EncoderConfig = field(default_factory=lambda: Qwen3EncoderConfig(
        # hidden_states[-2] of the 36-layer Qwen3-4B == output of layer 35
        hidden_state_layers=(35,), max_sequence_length=512,
    ))
    scheduler: FlowMatchConfig = field(default_factory=lambda: FlowMatchConfig.from_scheduler_config(
        {"num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": False},
    ))

    default_height: int = 1024
    default_width: int = 1024
    # Turbo: 8 function evaluations, no classifier-free guidance.
    default_num_inference_steps: int = 8
    default_guidance_scale: float = 0.0
    max_denoise_steps: int = 50

    @property
    def spatial_alignment(self) -> int:
        """Pixel multiple of a valid size: VAE stride x DiT patch (16)."""
        return self.vae.spatial_compression * self.transformer.patch_size

    def latent_grid(self, height: int, width: int) -> tuple[int, int]:
        """Token grid ``(h, w)``: one token per 16x16 pixels."""
        return height // self.spatial_alignment, width // self.spatial_alignment

    @classmethod
    def from_snapshot(cls, snapshot: str | Path) -> "ZImageConfig":
        snapshot = Path(snapshot)
        with open(snapshot / "model_index.json") as f:
            index = json.load(f)
        if index.get("_class_name") != "ZImagePipeline":
            raise ValueError(f"{snapshot} is not a Z-Image pipeline snapshot ({index.get('_class_name')!r})")
        with open(snapshot / "transformer" / "config.json") as f:
            transformer = ZImageTransformerConfig.from_dict(json.load(f))
        with open(snapshot / "vae" / "config.json") as f:
            vae = AutoencoderKLConfig.from_dict(json.load(f))
        with open(snapshot / "text_encoder" / "config.json") as f:
            text_cfg = json.load(f)
        text_encoder = Qwen3EncoderConfig.from_dict(
            text_cfg, hidden_state_layers=(int(text_cfg["num_hidden_layers"]) - 1,), max_sequence_length=512,
        )
        with open(snapshot / "scheduler" / "scheduler_config.json") as f:
            scheduler = FlowMatchConfig.from_scheduler_config(json.load(f))
        if transformer.cap_feat_dim != text_encoder.hidden_size:
            raise ValueError(f"cap_feat_dim {transformer.cap_feat_dim} != Qwen3 hidden size {text_encoder.hidden_size}")
        if vae.scaling_factor is None or vae.shift_factor is None:
            raise ValueError("Z-Image's VAE config must carry scaling_factor and shift_factor")
        return cls(transformer=transformer, vae=vae, text_encoder=text_encoder, scheduler=scheduler)


__all__ = ["ZImageConfig", "ZImageTransformerConfig", "resolve_snapshot_dir"]
