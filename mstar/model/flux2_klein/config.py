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

from mstar.model.components.diffusion.flow_match import FlowMatchConfig

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
    """``vae/config.json`` of ``AutoencoderKLFlux2`` (an SD-style KL autoencoder
    whose BatchNorm running statistics normalize the 2x2-patchified latents)."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    batch_norm_eps: float = 1e-4
    patch_size: tuple[int, int] = (2, 2)
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True

    @property
    def spatial_compression(self) -> int:
        """Pixels per latent cell along each axis (2 ** (#blocks - 1) == 8)."""
        return 2 ** (len(self.block_out_channels) - 1)

    @property
    def patched_latent_channels(self) -> int:
        return self.latent_channels * self.patch_size[0] * self.patch_size[1]

    @classmethod
    def from_dict(cls, cfg: dict) -> "Flux2VaeConfig":
        if cfg.get("act_fn", "silu") != "silu":
            raise NotImplementedError(f"FLUX.2 VAE port implements act_fn='silu', got {cfg['act_fn']!r}")
        if any(t != "DownEncoderBlock2D" for t in cfg.get("down_block_types", ())) or any(
            t != "UpDecoderBlock2D" for t in cfg.get("up_block_types", ())
        ):
            raise NotImplementedError("FLUX.2 VAE port implements DownEncoderBlock2D / UpDecoderBlock2D only")
        return cls(
            in_channels=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 3)),
            latent_channels=int(cfg["latent_channels"]),
            block_out_channels=tuple(int(c) for c in cfg["block_out_channels"]),
            layers_per_block=int(cfg.get("layers_per_block", 2)),
            norm_num_groups=int(cfg.get("norm_num_groups", 32)),
            batch_norm_eps=float(cfg.get("batch_norm_eps", 1e-4)),
            patch_size=tuple(int(p) for p in cfg.get("patch_size", (2, 2))),
            use_quant_conv=bool(cfg.get("use_quant_conv", True)),
            use_post_quant_conv=bool(cfg.get("use_post_quant_conv", True)),
            mid_block_add_attention=bool(cfg.get("mid_block_add_attention", True)),
        )


@dataclass(frozen=True)
class Qwen3EncoderConfig:
    """``text_encoder/config.json`` of the Qwen3 LM the prompt runs through,
    plus the pipeline's tapping recipe (``text_encoder_out_layers``,
    ``max_sequence_length``)."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    pad_token_id: int = 151643
    # Pipeline recipe (Flux2KleinPipeline.encode_prompt defaults).
    hidden_state_layers: tuple[int, ...] = (9, 18, 27)
    max_sequence_length: int = 512

    @property
    def num_layers_needed(self) -> int:
        """Decoder layers that must run to produce the deepest tap
        (``hidden_states[k]`` is the output of layer ``k``, 1-indexed)."""
        return max(self.hidden_state_layers)

    @property
    def output_dim(self) -> int:
        return self.hidden_size * len(self.hidden_state_layers)

    @classmethod
    def from_dict(cls, cfg: dict, **overrides) -> "Qwen3EncoderConfig":
        if cfg.get("model_type") not in (None, "qwen3"):
            raise NotImplementedError(f"text encoder model_type {cfg.get('model_type')!r} is not Qwen3")
        if cfg.get("rope_scaling"):
            raise NotImplementedError("Qwen3 text encoder with rope_scaling is not supported")
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            pad_token_id=int(cfg.get("pad_token_id") or 151643),
            **overrides,
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
    # Distilled: no classifier-free guidance (the pipeline ignores guidance_scale
    # when is_distilled). Kept as a knob so a -base checkpoint can turn it on.
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
        if index.get("_class_name") not in ("Flux2KleinPipeline", "Flux2Pipeline"):
            raise ValueError(f"{snapshot} is not a FLUX.2 pipeline snapshot ({index.get('_class_name')!r})")
        with open(snapshot / "transformer" / "config.json") as f:
            transformer = Flux2TransformerConfig.from_dict(json.load(f))
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
        is_distilled = bool(index.get("is_distilled", False))
        return cls(
            transformer=transformer, vae=vae, text_encoder=text_encoder, scheduler=scheduler,
            is_distilled=is_distilled,
            default_num_inference_steps=4 if is_distilled else 50,
            default_guidance_scale=1.0 if is_distilled else 4.0,
        )


def resolve_snapshot_dir(model_path_hf: str, cache_dir: str | None = None) -> Path:
    """Local snapshot directory of a pipeline repo: a local path is used as-is,
    a hub id resolves through the HF cache (configs + weights of every
    component, offline when ``HF_HUB_OFFLINE`` is set)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path_hf, cache_dir=cache_dir))
