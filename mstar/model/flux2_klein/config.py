"""Configuration for FLUX.2 [klein] (4B / 9B): read from the checkpoint's config.json files.

Nothing architectural is hardcoded here: ``Flux2KleinConfig.from_snapshot`` reads
``transformer/config.json``, ``vae/config.json``, ``text_encoder/config.json``
and ``scheduler/scheduler_config.json`` from the (locally cached) Hugging Face
snapshot, so the 4B and 9B checkpoints — and any future klein-family variant with
the same component classes — load from the same code. The pipeline-level facts
that diffusers keeps in code rather than in a config (text tap layers, the 512
token prompt length, reference-image time offsets) are the defaults below, each
named for the pipeline constant it mirrors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.autoencoder_kl import AutoencoderKLConfig
from mstar.model.components.diffusion.flow_match import FlowMatchConfig
from mstar.model.components.diffusion.qwen3.encoder import Qwen3EncoderConfig

# Resource key the dit node declares its ragged joint attention under.
DIT_ATTN = "dit_attn"
# Loop name the model file and the dit submodule's check_stop agree on.
DENOISE_LOOP = "denoise_loop"

# Registry defaults (HF ids).
FLUX2_KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_9B = "black-forest-labs/FLUX.2-klein-9B"


@dataclass(frozen=True)
class Flux2TransformerConfig:
    """``transformer/config.json`` of ``Flux2Transformer2DModel``."""

    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 5               # double-stream blocks
    num_single_layers: int = 20       # single-stream blocks
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 7680
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: float = 2000.0
    eps: float = 1e-6
    guidance_embeds: bool = False
    patch_size: int = 1

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @classmethod
    def from_dict(cls, cfg: dict) -> "Flux2TransformerConfig":
        return cls(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg.get("out_channels") or cfg["in_channels"]),
            num_layers=int(cfg["num_layers"]),
            num_single_layers=int(cfg["num_single_layers"]),
            attention_head_dim=int(cfg["attention_head_dim"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            joint_attention_dim=int(cfg["joint_attention_dim"]),
            timestep_guidance_channels=int(cfg.get("timestep_guidance_channels", 256)),
            mlp_ratio=float(cfg.get("mlp_ratio", 3.0)),
            axes_dims_rope=tuple(int(d) for d in cfg.get("axes_dims_rope", (32, 32, 32, 32))),
            rope_theta=float(cfg.get("rope_theta", 2000)),
            eps=float(cfg.get("eps", 1e-6)),
            guidance_embeds=bool(cfg.get("guidance_embeds", False)),
            patch_size=int(cfg.get("patch_size", 1)),
        )


@dataclass(frozen=True)
class Flux2VaeConfig:
    """``vae/config.json`` of ``AutoencoderKLFlux2``: the shared KL autoencoder config plus the
    FLUX.2 specifics (BatchNorm latent statistics over 2x2 patches)."""

    autoencoder: AutoencoderKLConfig = field(default_factory=lambda: AutoencoderKLConfig(
        latent_channels=32, use_quant_conv=True, use_post_quant_conv=True,
    ))
    batch_norm_eps: float = 1e-4
    patch_size: tuple[int, int] = (2, 2)

    @property
    def latent_channels(self) -> int:
        return self.autoencoder.latent_channels

    @property
    def spatial_compression(self) -> int:
        """Pixels per latent cell along each axis (8)."""
        return self.autoencoder.spatial_compression

    @property
    def patched_latent_channels(self) -> int:
        return self.latent_channels * self.patch_size[0] * self.patch_size[1]

    @classmethod
    def from_dict(cls, cfg: dict) -> "Flux2VaeConfig":
        return cls(
            autoencoder=AutoencoderKLConfig.from_dict(cfg),
            batch_norm_eps=float(cfg.get("batch_norm_eps", 1e-4)),
            patch_size=tuple(int(p) for p in cfg.get("patch_size", (2, 2))),
        )



@dataclass
class Flux2KleinConfig:
    """Everything the model needs, resolved from one checkpoint snapshot."""

    transformer: Flux2TransformerConfig = field(default_factory=Flux2TransformerConfig)
    vae: Flux2VaeConfig = field(default_factory=Flux2VaeConfig)
    text_encoder: Qwen3EncoderConfig = field(default_factory=Qwen3EncoderConfig)
    scheduler: FlowMatchConfig = field(default_factory=FlowMatchConfig)

    # Generation defaults; every one is overridable per request through model_kwargs.
    default_height: int = 1024
    default_width: int = 1024
    default_num_inference_steps: int = 4
    # Distilled: no classifier-free guidance (the pipeline ignores guidance_scale).
    # Kept as a knob for a -base checkpoint, but CFG is NOT implemented yet --
    # ``denoise`` runs a single conditional pass and ``guidance_scale`` is only
    # logged -- so ``from_snapshot`` refuses a non-distilled snapshot rather than
    # serving it silently unguided. Flip this back to a real knob when CFG lands.
    is_distilled: bool = True
    default_guidance_scale: float = 1.0
    # Ceiling on the denoise Loop; a request's step count is clamped to it.
    max_denoise_steps: int = 50
    # Reference (edit) images: the pipeline's T-coordinate spacing and the
    # area cap it resizes to before encoding.
    ref_image_time_scale: int = 10
    ref_image_max_area: int = 1024 * 1024
    max_ref_images: int = 4

    @property
    def spatial_alignment(self) -> int:
        """Pixel multiple a request's height/width must satisfy: VAE stride x 2x2 patch = 16."""
        return self.vae.spatial_compression * self.vae.patch_size[0]

    def latent_grid(self, height: int, width: int) -> tuple[int, int]:
        """Token grid ``(h, w)`` for a pixel size — one token per 16x16 pixel patch."""
        return height // self.spatial_alignment, width // self.spatial_alignment

    @classmethod
    def from_snapshot(cls, snapshot: str | Path) -> "Flux2KleinConfig":
        snapshot = Path(snapshot)
        with open(snapshot / "model_index.json") as f:
            index = json.load(f)
        # klein only: the non-distilled Flux2Pipeline needs CFG, which `denoise` does not do.
        if index.get("_class_name") != "Flux2KleinPipeline":
            raise ValueError(f"{snapshot} is not a FLUX.2 [klein] pipeline snapshot ({index.get('_class_name')!r})")
        # Before parsing the rest: nothing downstream runs CFG. `denoise` does one
        # conditional pass and `guidance_scale` is only logged, so a non-distilled
        # checkpoint would load and serve silently unguided images at 50 steps.
        # `is_distilled` is absent from some snapshots, so the DEFAULT decides --
        # hence True; defaulting to False made the unguided path the common one.
        if not bool(index.get("is_distilled", True)):
            raise NotImplementedError(
                f"{snapshot} is a non-distilled FLUX.2 checkpoint, which needs "
                "classifier-free guidance; mstar runs a single conditional pass, so it "
                "would serve unguided images. Only the step-distilled klein checkpoints "
                "are supported."
            )
        with open(snapshot / "transformer" / "config.json") as f:
            transformer = Flux2TransformerConfig.from_dict(json.load(f))
        # Same reason as the is_distilled guard, but it needs the transformer config:
        # a guidance embedder the forward never feeds.
        if transformer.guidance_embeds:
            raise NotImplementedError(
                f"{snapshot} sets guidance_embeds; mstar does not feed a guidance "
                "embedding to the transformer, so the served images would ignore it."
            )
        with open(snapshot / "vae" / "config.json") as f:
            vae = Flux2VaeConfig.from_dict(json.load(f))
        with open(snapshot / "text_encoder" / "config.json") as f:
            text_encoder = Qwen3EncoderConfig.from_dict(json.load(f))
        with open(snapshot / "scheduler" / "scheduler_config.json") as f:
            scheduler = FlowMatchConfig.from_scheduler_config(json.load(f), empirical_mu=True)
        if transformer.joint_attention_dim != text_encoder.output_dim:
            raise ValueError(
                f"transformer joint_attention_dim {transformer.joint_attention_dim} != "
                f"{len(text_encoder.hidden_state_layers)} tapped Qwen3 layers x {text_encoder.hidden_size}"
            )
        if transformer.in_channels != vae.patched_latent_channels:
            raise ValueError(
                f"transformer in_channels {transformer.in_channels} != VAE patched latent channels "
                f"{vae.patched_latent_channels}"
            )
        # Past the guard every loadable snapshot is distilled, so the defaults are
        # the distilled ones; the non-distilled branch comes back with CFG.
        return cls(
            transformer=transformer, vae=vae, text_encoder=text_encoder, scheduler=scheduler,
            is_distilled=True, default_num_inference_steps=4, default_guidance_scale=1.0,
        )
