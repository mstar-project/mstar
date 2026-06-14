"""Configuration for the Cosmos3 omni generator.

A single ``Cosmos3Config`` describes every Cosmos3 checkpoint (Nano, Super,
Policy-DROID, and the Super task variants). The checkpoints share one
architecture; they differ only in the transformer dimensions
(``num_hidden_layers`` / ``hidden_size`` / ``num_attention_heads`` /
``intermediate_size``) and two capability flags (``sound_gen``,
``action_gen``).

Values load from a local HF checkpoint directory laid out the diffusers way::

    <ckpt>/transformer/config.json   -> the DiT (dual-pathway MoT) dimensions
    <ckpt>/vae/config.json           -> AutoencoderKLWan factors + latent stats
    <ckpt>/scheduler/scheduler_config.json -> UniPC flow scheduler settings

Dataclass defaults mirror Cosmos3-Nano so a bare ``Cosmos3Config()`` is a
valid Nano config without any file present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _filtered(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Keep only the dict entries that name a field on the dataclass ``cls``."""
    names = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Cosmos3VAEConfig:
    """The Wan2.2-TI2V-5B VAE (``AutoencoderKLWan``) parameters we need at the
    serving layer. The full VAE module loads from the ``vae/`` subfolder via
    diffusers; here we only track the latent geometry and the per-channel
    normalization statistics the pipeline applies to/from latent space.
    """

    z_dim: int = 48
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    # Per-channel latent normalization (length == z_dim). The pipeline maps
    # raw VAE latents x -> (x - mean) / std before denoising and inverts it
    # before decode.
    latents_mean: list[float] = field(default_factory=list)
    latents_std: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3VAEConfig":
        return cls(**_filtered(cls, d))


@dataclass
class Cosmos3SchedulerConfig:
    """UniPC multistep flow scheduler settings (``scheduler/scheduler_config``).

    The denoise loop drives a diffusers ``UniPCMultistepScheduler`` configured
    from these fields; we do not re-implement the bh2 corrector.
    """

    scheduler_type: str = "unipc"
    prediction_type: str = "flow_prediction"
    predict_x0: bool = True
    solver_order: int = 2
    solver_type: str = "bh2"
    use_flow_sigmas: bool = True
    use_karras_sigmas: bool = True
    final_sigmas_type: str = "zero"
    num_train_timesteps: int = 1000
    flow_shift: float = 1.0
    sigma_min: float = 0.147
    sigma_max: float = 200.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3SchedulerConfig":
        # diffusers stores the flow shift under "flow_shift"; keep the rest by name.
        return cls(**_filtered(cls, d))


@dataclass
class Cosmos3Config:
    """Cosmos3 generator configuration (one architecture, swappable weights)."""

    # ----- dual-pathway MoT transformer (the DiT) -----
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    max_position_embeddings: int = 262144

    # ----- 3D interleaved mRoPE -----
    rope_theta: float = 5_000_000.0
    rope_axes_dim: tuple[int, int, int] = (24, 20, 20)  # rope_scaling.mrope_section
    mrope_interleaved: bool = True
    unified_3d_mrope_temporal_modality_margin: int = 15000
    unified_3d_mrope_reset_spatial_ids: bool = True
    base_fps: int = 24
    enable_fps_modulation: bool = True

    # ----- latent geometry / patchify -----
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192  # latent_patch_size**2 * latent_channel
    timestep_scale: float = 0.001

    # ----- attention / norm style -----
    joint_attn_implementation: str = "two_way"  # GEN attends [UND|GEN]; UND causal, UND-only
    qk_norm_for_diffusion: bool = True
    qk_norm_for_text: bool = True
    use_moe: bool = True  # MoT two-FFN split (mlp / mlp_moe_gen), NOT sparse experts

    # ----- capability flags + modality heads -----
    action_gen: bool = True
    max_action_dim: int = 64
    num_embodiment_domains: int = 32
    sound_gen: bool = True
    sound_dim: int | None = 64
    sound_latent_fps: float = 25.0
    temporal_compression_factor_sound: int = 1
    video_temporal_causal: bool = False
    freeze_und: bool = False

    # ----- default sampling (overridable per request / yaml) -----
    # Number of denoise model evaluations. The per-mode cookbook defaults are
    # t2i 50, t2v/i2v 35, action fd/id 30, DROID policy ~4. ``num_inference_steps``
    # is the image default; ``num_inference_steps_video`` is the video default.
    # A request may override either; the value is clamped to ``max_inference_steps``.
    num_inference_steps: int = 50
    num_inference_steps_video: int = 35
    # Upper bound on the denoise loop's iteration count. The loop is built with
    # this many iterations and each request stops early at its own step count, so
    # one graph serves any per-request step count up to this cap.
    max_inference_steps: int = 100
    # Default frames-per-second for video generation + mp4 playback (overridable
    # per request via ``fps``).
    fps: float = 24.0
    # Default frame count for a video request that doesn't specify ``num_frames``
    # (the Wan VAE downsamples time by 4, so latent frames = 1 + (n - 1) // 4).
    num_frames_video: int = 17

    # ----- sub-configs -----
    vae: Cosmos3VAEConfig = field(default_factory=Cosmos3VAEConfig)
    scheduler: Cosmos3SchedulerConfig = field(default_factory=Cosmos3SchedulerConfig)

    # ----- provenance -----
    local_dir: str = ""

    @classmethod
    def from_transformer_dict(cls, d: dict[str, Any]) -> "Cosmos3Config":
        """Build from a diffusers ``transformer/config.json`` dict alone.

        Sub-configs are left at their defaults; use ``from_pretrained`` to also
        populate VAE/scheduler from their sibling folders.
        """
        kwargs = _filtered(cls, d)
        rope = d.get("rope_scaling") or {}
        if "mrope_section" in rope:
            kwargs["rope_axes_dim"] = tuple(rope["mrope_section"])
        if "mrope_interleaved" in rope:
            kwargs["mrope_interleaved"] = bool(rope["mrope_interleaved"])
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "Cosmos3Config":
        """Load from a diffusers-layout checkpoint directory."""
        root = Path(local_dir)
        tcfg_path = root / "transformer" / "config.json"
        if not tcfg_path.exists():
            raise FileNotFoundError(f"transformer/config.json not found under {root}")
        with open(tcfg_path) as f:
            cfg = cls.from_transformer_dict(json.load(f))
        cfg.local_dir = str(root)

        vae_path = root / "vae" / "config.json"
        if vae_path.exists():
            with open(vae_path) as f:
                cfg.vae = Cosmos3VAEConfig.from_dict(json.load(f))

        sched_path = root / "scheduler" / "scheduler_config.json"
        if sched_path.exists():
            with open(sched_path) as f:
                cfg.scheduler = Cosmos3SchedulerConfig.from_dict(json.load(f))

        return cfg
