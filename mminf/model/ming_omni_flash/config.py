"""Configuration dataclass for Ming-flash-omni-2.0.

Mirrors mminf's qwen3_omni pattern (pure ``@dataclass`` tree,
``from_pretrained(local_dir)``, convenience ``@property``s) so the rest of
the framework can read dims off the loaded config without going through
``transformers.PretrainedConfig`` machinery.

The released checkpoint (``inclusionAI/Ming-flash-omni-2.0``) does NOT match
upstream vllm-omni's flat ``MingFlashOmniConfig`` nesting. On disk only the
``BailingMM2Config`` shape lives at ``config.json``::

    config.json                     # thinker: audio_config + llm_config + vision_config + scalars
    talker/config.json              # talker top-level (BailingTalker2)
    talker/llm/config.json          # talker LLM backbone (Qwen2)
    talker/vae/config.json          # talker AudioVAE
    transformer/config.json         # image-gen DiT (ZImageTransformer2DModel)
    vae/config.json                 # image-gen VAE
    scheduler/scheduler_config.json # image-gen diffusion scheduler
    byt5/google__byt5-smal/config.json   # image-gen text encoder
    connector/config.json           # image-gen connector
    mlp/config.json                 # image-gen projector

This loader follows the on-disk layout: it parses ``config.json`` for the
thinker path and lazy-loads talker / image-gen from sibling subdirs when
those exist. Talker and image-gen are SKELETON dataclasses today — exhaustive
field semantics land with the talker port (step 6 of PORTING_NOTES.md) and
the image-gen port (step 9).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thinker LLM (Ling-2.0 sparse MoE — model_type "bailing_moe_v2")
# ---------------------------------------------------------------------------

@dataclass
class ThinkerLLMConfig:
    """Ling-2.0 sparse-MoE thinker (BailingMoeV2).

    Field set is the union of what upstream
    ``vllm_omni/transformers_utils/configs/ming_flash_omni.py:BailingMoeV2Config``
    declares and what the released ``llm_config`` actually populates.
    Defaults reflect the released ckpt, not the upstream class defaults
    (which were trained for a smaller config).
    """

    # Dims
    vocab_size: int = 157184
    hidden_size: int = 4096
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int | None = None  # computed in __post_init__

    # Norm / activation
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    tie_word_embeddings: bool = False

    # Position / RoPE
    max_position_embeddings: int = 32768
    rope_theta: float = 2_400_000.0
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float = 0.5

    # MoE
    num_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1024
    first_k_dense_replace: int = 1
    router_type: str = "MultiRouter"
    n_group: int = 8
    topk_group: int = 4
    moe_router_topk_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    use_expert_bias: bool = True
    output_router_logits: bool = False

    # Misc
    pad_token_id: int = 156892
    eos_token_id: int = 156895
    use_interleaved_frame_timestamp: bool = True

    # Multimodal token IDs (used by the prefill processor / chat template).
    # Defaults mirror the actual tokenizer (`tokenizer.json` added_tokens at
    # the released ckpt; cross-checked against Jonathan1909's patched config
    # and vllm-omni's BailingMoeV2Config defaults). Two gotchas the on-disk
    # `config.json` of `inclusionAI/Ming-flash-omni-2.0` introduces:
    #   * `video_start_token` is mislabeled as 157159 (= </image>) in the
    #     ckpt config; the real `<video>` token is 157160. Jonathan1909's
    #     patched config corrects this. `__post_init__` warns loudly if a
    #     load picks up the bogus value.
    #   * `audio_*` / `*_end` / `tokens_per_second` are not in the on-disk
    #     llm_config at all; they're tokenizer-derived constants and are
    #     hardcoded in vllm-omni. We mirror those defaults here so
    #     vision/audio masking + MRoPE temporal-position math can read them
    #     directly off `ThinkerLLMConfig`.
    image_patch_token: int = 157157
    video_patch_token: int = 157175
    audio_patch_token: int = 157168
    image_start_token: int = 157158
    video_start_token: int = 157160
    audio_start_token: int = 157169
    image_end_token: int = 157159
    video_end_token: int = 157161
    audio_end_token: int = 157170
    tokens_per_second: int = 2

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # Released ckpt has hidden_size=4096, num_attention_heads=32 → head_dim=128.
        # Mirror qwen3_omni's loud-on-mismatch warning (config.py:46-64) so a
        # silently-wrong head_dim doesn't break MRoPE downstream.
        if self.head_dim * self.num_attention_heads != self.hidden_size and self.head_dim != 128:
            logger.warning(
                "ThinkerLLMConfig: unusual head_dim=%d "
                "(hidden_size=%d, num_attention_heads=%d). "
                "Expected head_dim=128 for Ming-flash-omni-2.0. "
                "Verify the checkpoint config.json contains 'head_dim': 128 "
                "under llm_config.",
                self.head_dim, self.hidden_size, self.num_attention_heads,
            )
        # The inclusionAI ckpt's llm_config.video_start_token is mislabeled
        # (157159 = </image> per tokenizer; the real <video> token is 157160).
        # If we picked up the bogus value, repair it and warn loudly — vision
        # masking would otherwise key on </image> for video-start markers.
        if self.video_start_token == 157159 and self.image_end_token == 157159:
            logger.warning(
                "ThinkerLLMConfig: ckpt-supplied video_start_token=157159 "
                "matches image_end_token (= </image> per tokenizer). The "
                "released inclusionAI/Ming-flash-omni-2.0 config.json "
                "mislabels this field; correcting to 157160 (= <video>). "
                "If this is intentional, set video_start_token explicitly "
                "after construction."
            )
            self.video_start_token = 157160

    @property
    def mrope_section(self) -> list[int]:
        """MRoPE section split. Upstream default [8, 12, 12] sums to 32 — the
        number of rotary dims (head_dim=128 * partial_rotary_factor=0.5)."""
        return (self.rope_scaling or {}).get("mrope_section", [8, 12, 12])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThinkerLLMConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Vision encoder (Qwen3-MoE ViT — model_type "qwen3_moe_vit")
# ---------------------------------------------------------------------------

@dataclass
class VisionEncoderConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisionEncoderConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in fnames}
        # HF stores tuple fields as lists; coerce.
        if "deepstack_visual_indexes" in filtered and isinstance(
            filtered["deepstack_visual_indexes"], list
        ):
            filtered["deepstack_visual_indexes"] = tuple(
                filtered["deepstack_visual_indexes"]
            )
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Audio encoder (Whisper-style, with Ming-side knobs)
# ---------------------------------------------------------------------------

@dataclass
class AudioEncoderConfig:
    """Whisper encoder.

    On disk the outer ``audio_config`` carries Ming-side knobs (downsample
    kernel + stride for the post-encoder convolution, ``norm_query_embeds``)
    while the actual Whisper dims sit nested under
    ``audio_config.whisper_encoder_config`` as ``{n_ctx, n_head, n_layer,
    n_mels, n_state}``. We keep the same nesting and expose convenience
    properties so callers can read ``d_model`` / ``encoder_layers`` /
    ``encoder_attention_heads`` without traversing the dict.
    """

    ds_kernel_size: int = 3
    ds_stride: int = 2
    norm_query_embeds: bool = True
    whisper_encoder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "n_ctx": 15000, "n_head": 20, "n_layer": 32, "n_mels": 128, "n_state": 1280,
        }
    )

    @property
    def d_model(self) -> int:
        return int(self.whisper_encoder_config["n_state"])

    @property
    def encoder_layers(self) -> int:
        return int(self.whisper_encoder_config["n_layer"])

    @property
    def encoder_attention_heads(self) -> int:
        return int(self.whisper_encoder_config["n_head"])

    @property
    def n_mels(self) -> int:
        return int(self.whisper_encoder_config["n_mels"])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AudioEncoderConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Talker — CFM (continuous flow matching) head + Qwen2 LLM backbone
# ---------------------------------------------------------------------------
#
# Step 6a (this commit) lifts the on-disk talker config tree into proper
# typed dataclasses so the downstream modeling code (CFM head, DiT
# blocks, Aggregator, Qwen2 backbone) can read dims off `config.talker`
# directly. The on-disk layout is::
#
#   talker/config.json        — top-level BailingTalker2 config (CFM steps,
#                                patch sizes, flowmodel + aggregator blocks)
#   talker/llm/config.json    — Qwen2 LLM backbone (896-dim, 24 layers)
#   talker/vae/config.json    — AudioVAE (44.1 kHz output, latent_dim=64)
#
# The flowmodel / aggregator blocks share the same DiTBlockConfig shape
# (depth, hidden_size, num_heads, mlp_ratio, dropout, in_channels) so we
# reuse one dataclass for both.


@dataclass
class TalkerLLMConfig:
    """Qwen2 LLM backbone used inside the talker.

    Distinct from `ThinkerLLMConfig` — different vocab, smaller dims, no
    MoE, GQA with sliding-window attention. Field set mirrors the
    populated keys in `talker/llm/config.json` on the released ckpt.
    """

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2

    # Norm / activation
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # Position / RoPE
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0

    # Sliding window attention. On the released ckpt the talker LLM has
    # use_sliding_window=False but a non-trivial sliding_window value;
    # we expose both so the eventual attention impl can branch correctly.
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 21

    # Misc
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    attention_dropout: float = 0.0
    use_cache: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TalkerLLMConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


@dataclass
class DiTBlockConfig:
    """Shared shape for the talker's `flowmodel` and `aggregator` DiT stacks.

    Both blocks live under the talker top-level config with identical
    fields (only the dropout value differs — flowmodel=0, aggregator=0.1).
    The block builds out as Attention + RMSNorm + FeedForward(GeGLU);
    individual layers are stacked `depth` times.
    """

    attn_backend: str = "torch"
    attn_mask_enabled: bool = False
    depth: int = 8
    dropout: float = 0.0
    hidden_size: int = 1024
    in_channels: int = 64
    mlp_ratio: int = 4
    num_heads: int = 16
    pe_attn_head: Any | None = None
    qk_norm: Any | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.mlp_ratio

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiTBlockConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


@dataclass
class AudioVAEConfig:
    """Audio VAE: encoder (waveform → latent) + decoder (latent → ISTFT mag/phase).

    Both encoder and decoder are Qwen2-LLM-shaped sliding-window attention
    backbones (`enc_kwargs.backbone` / `dec_kwargs.backbone` on disk —
    nearly identical to `TalkerLLMConfig` but with vocab_size=1 since they
    don't tokenize text). The encoder takes waveform features and outputs
    `latent_dim` channels; the decoder consumes the latent and projects to
    `output_dim` STFT bins (882 on the released ckpt) for an ISTFT head.

    Discriminator + loss-weight fields (`hifi_gan_disc_kwargs`,
    `lambda_*`) are training-time only and stored as raw dicts here —
    inference doesn't read them.
    """

    sample_rate: int = 44100
    patch_size: int = 4
    init_method: str = "kaiming"

    # Encoder / decoder dims pulled out of enc_kwargs / dec_kwargs.
    latent_dim: int = 64
    encoder_input_dim: int = 80    # mel bins (default; overridden below)
    encoder_hop_size: int = 320    # frames-per-latent (default)
    decoder_output_dim: int = 882  # STFT bins fed to ISTFTHead

    # The full Qwen2-shaped backbones for enc/dec are kept as raw dicts
    # here; the modeling code (step 6d) will lift the relevant fields
    # into its own block-builder.
    enc_backbone: dict[str, Any] = field(default_factory=dict)
    dec_backbone: dict[str, Any] = field(default_factory=dict)

    # Training-time only; not consumed by inference. Stored verbatim so a
    # full round-trip remains possible.
    hifi_gan_disc_kwargs: dict[str, Any] = field(default_factory=dict)
    spec_disc_kwargs: dict[str, Any] = field(default_factory=dict)
    semantic_module_kwargs: dict[str, Any] | None = None
    lambda_adv: float = 1.0
    lambda_disc: float = 1.0
    lambda_feat_match_loss: float = 1.0
    lambda_mel_loss: float = 1.0
    lambda_semantic: float = 2.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AudioVAEConfig:
        enc_kwargs = d.get("enc_kwargs") or {}
        dec_kwargs = d.get("dec_kwargs") or {}
        return cls(
            sample_rate=int(d.get("sample_rate", 44100)),
            patch_size=int(d.get("patch_size", 4)),
            init_method=str(d.get("init_method", "kaiming")),
            latent_dim=int(enc_kwargs.get("latent_dim", dec_kwargs.get("latent_dim", 64))),
            encoder_input_dim=int(enc_kwargs.get("input_dim", 80)),
            encoder_hop_size=int(enc_kwargs.get("hop_size", 320)),
            decoder_output_dim=int(dec_kwargs.get("output_dim", 882)),
            enc_backbone=dict(enc_kwargs.get("backbone", {})),
            dec_backbone=dict(dec_kwargs.get("backbone", {})),
            hifi_gan_disc_kwargs=dict(d.get("hifi_gan_disc_kwargs") or {}),
            spec_disc_kwargs=dict(d.get("spec_disc_kwargs") or {}),
            semantic_module_kwargs=d.get("semantic_module_kwargs"),
            lambda_adv=float(d.get("lambda_adv", 1.0)),
            lambda_disc=float(d.get("lambda_disc", 1.0)),
            lambda_feat_match_loss=float(d.get("lambda_feat_match_loss", 1.0)),
            lambda_mel_loss=float(d.get("lambda_mel_loss", 1.0)),
            lambda_semantic=float(d.get("lambda_semantic", 2.0)),
        )


@dataclass
class TalkerConfig:
    """Ming-flash-omni-2.0 talker (BailingTalker2) — Qwen2 LLM + CFM head.

    Sub-configs:
      * ``llm`` — Qwen2-based AR LLM that consumes thinker hidden states
        + voice prompt and emits CFM-driving latents.
      * ``flowmodel`` / ``aggregator`` — DiT stacks (CFM + condition
        aggregator); identical shape, different dropout.
      * ``vae`` — AudioVAE for waveform synthesis from CFM-generated
        latents.

    Inference-time scalars (`steps`, `cfg_strength`, `patch_size`,
    `history_patch_size`) drive the CFM sampling loop.
    """

    # Top-level scalars from talker/config.json
    steps: int = 10
    patch_size: int = 4
    history_patch_size: int = 32
    cfg_strength: float = 2.0

    # Typed sub-configs (replaces step-1's raw-dict skeletons).
    llm: TalkerLLMConfig = field(default_factory=TalkerLLMConfig)
    flowmodel: DiTBlockConfig = field(default_factory=DiTBlockConfig)
    aggregator: DiTBlockConfig = field(default_factory=DiTBlockConfig)
    vae: AudioVAEConfig = field(default_factory=AudioVAEConfig)

    # Convenience accessors used by Model.get_output_sample_rate (kept
    # for backward compat with code that previously read `vae_sample_rate`
    # / `vae_patch_size` directly off this dataclass).
    @property
    def vae_sample_rate(self) -> int:
        return self.vae.sample_rate

    @property
    def vae_patch_size(self) -> int:
        return self.vae.patch_size

    @classmethod
    def from_subdir(cls, talker_dir: str | os.PathLike[str]) -> TalkerConfig | None:
        """Load from ``<local_dir>/talker/``; return None if the subdir is absent."""
        talker_dir = Path(talker_dir)
        cfg_path = talker_dir / "config.json"
        if not cfg_path.exists():
            return None

        with open(cfg_path) as f:
            raw = json.load(f)

        # Top-level scalars
        steps = int(raw.get("steps", 10))
        patch_size = int(raw.get("patch_size", 4))
        history_patch_size = int(raw.get("history_patch_size", 32))
        cfg_strength = float(raw.get("cfg_strength", 2.0))

        # flowmodel + aggregator sub-blocks
        flowmodel = DiTBlockConfig.from_dict(raw.get("flowmodel", {}))
        aggregator = DiTBlockConfig.from_dict(raw.get("aggregator", {}))

        # LLM sub-config
        llm = TalkerLLMConfig()
        llm_path = talker_dir / "llm" / "config.json"
        if llm_path.exists():
            with open(llm_path) as f:
                llm = TalkerLLMConfig.from_dict(json.load(f))

        # VAE sub-config
        vae = AudioVAEConfig()
        vae_path = talker_dir / "vae" / "config.json"
        if vae_path.exists():
            with open(vae_path) as f:
                vae = AudioVAEConfig.from_dict(json.load(f))

        return cls(
            steps=steps,
            patch_size=patch_size,
            history_patch_size=history_patch_size,
            cfg_strength=cfg_strength,
            llm=llm,
            flowmodel=flowmodel,
            aggregator=aggregator,
            vae=vae,
        )


# ---------------------------------------------------------------------------
# Image generation (SKELETON — step 9 will fill in)
# ---------------------------------------------------------------------------

@dataclass
class ImageGenConfig:
    """Ming-flash-omni-2.0 image-generation pipeline (ZImage DiT + ByT5).

    SKELETON. On the released ckpt the imagegen components live in sibling
    subdirs: ``transformer/`` (DiT), ``vae/`` (AutoencoderKL),
    ``scheduler/`` (FlowMatchEulerDiscreteScheduler), ``byt5/`` (text
    encoder), ``connector/`` (Qwen2-based connector), ``mlp/`` (projector
    with ``img_gen_scales``, ``diffusion_c_input_dim``). Exhaustive porting
    happens at step 9.
    """

    # Subfolder names (mirror upstream MingImageGenConfig)
    transformer_subfolder: str = "transformer"
    vae_subfolder: str = "vae"
    scheduler_subfolder: str = "scheduler"
    byt5_subfolder: str = "byt5"
    connector_subfolder: str = "connector"
    mlp_subfolder: str = "mlp"

    # From mlp/config.json
    img_gen_scales: list[int] = field(default_factory=lambda: [16])
    diffusion_c_input_dim: int = 2560
    text_encoder_norm: bool = True

    # Defaults for image-gen sampling (match upstream MingImageGenConfig)
    num_inference_steps: int = 30
    guidance_scale: float = 2.0
    default_height: int = 1024
    default_width: int = 1024

    @property
    def num_query_tokens(self) -> int:
        """Total learnable query tokens appended to the thinker for image-gen.

        ``img_gen_scales=[16]`` ⇒ 256. Matches upstream
        ``MingImageGenConfig.num_query_tokens`` and
        ``vllm_omni/.../ming_flash_omni/prompt_utils.py:DEFAULT_NUM_QUERY_TOKENS``.
        """
        return sum(s * s for s in self.img_gen_scales)

    @classmethod
    def from_subdirs(cls, local_dir: str | os.PathLike[str]) -> ImageGenConfig | None:
        """Load from sibling subdirs; return None if none of the imagegen
        subdirs exist (e.g. a thinker-only checkpoint)."""
        local_dir = Path(local_dir)
        # Use the DiT transformer config presence as the load gate — that's
        # the most expensive component and would fail loudly later anyway.
        if not (local_dir / "transformer" / "config.json").exists():
            return None

        instance = cls()

        # mlp/config.json overrides the imagegen knobs we expose at the top
        # level (img_gen_scales, diffusion_c_input_dim, text_encoder_norm).
        mlp_path = local_dir / instance.mlp_subfolder / "config.json"
        if mlp_path.exists():
            with open(mlp_path) as f:
                mlp_raw = json.load(f)
            if "img_gen_scales" in mlp_raw:
                instance.img_gen_scales = list(mlp_raw["img_gen_scales"])
            if "diffusion_c_input_dim" in mlp_raw:
                instance.diffusion_c_input_dim = int(mlp_raw["diffusion_c_input_dim"])
            if "text_encoder_norm" in mlp_raw:
                instance.text_encoder_norm = bool(mlp_raw["text_encoder_norm"])

        return instance


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class MingFlashOmniModelConfig:
    """Unified config for Ming-flash-omni-2.0 loaded from a local HF checkpoint."""

    local_dir: str = ""

    # Top-level scalar from config.json (cross-modal connector MLP depth)
    mlp_depth: int = 2

    # Sub-configs
    thinker_llm: ThinkerLLMConfig = field(default_factory=ThinkerLLMConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    talker: TalkerConfig | None = None
    image_gen: ImageGenConfig | None = None

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        llm = self.thinker_llm
        assert llm.head_dim is not None  # set in ThinkerLLMConfig.__post_init__

        # head_dim consistency. We tolerate the upstream-default mismatch
        # (head_dim=128 paired with hidden_size//num_heads) because Ming
        # explicitly overrides it; only fail when nothing matches.
        if llm.head_dim * llm.num_attention_heads != llm.hidden_size and llm.head_dim != 128:
            raise ValueError(
                f"ThinkerLLMConfig: head_dim={llm.head_dim} inconsistent with "
                f"hidden_size={llm.hidden_size} / num_attention_heads={llm.num_attention_heads}"
            )

        # MRoPE / partial-rotary invariant. The rotary subset of each head is
        # ``head_dim * partial_rotary_factor`` dims, which come in (cos, sin)
        # pairs — so ``mrope_section`` partitions half of that (the dims that
        # one of cos/sin owns) across the time / height / width axes. The
        # same arithmetic governs Qwen3-Omni (head_dim=128, partial=1.0 →
        # sum([16,24,24])=64=128/2) and Ming-flash-omni (head_dim=128,
        # partial=0.5 → sum([8,12,12])=32=64/2).
        rotary_pair_dims = int(llm.head_dim * llm.partial_rotary_factor) // 2
        section_sum = sum(llm.mrope_section)
        if section_sum != rotary_pair_dims:
            raise ValueError(
                f"MRoPE section {llm.mrope_section} sums to {section_sum} but "
                f"(head_dim={llm.head_dim} * partial_rotary_factor="
                f"{llm.partial_rotary_factor}) / 2 = {rotary_pair_dims}. "
                f"Section must partition the cos/sin half of the rotary dims."
            )

        # Multimodal token IDs must be within vocab.
        for name in (
            "image_patch_token", "video_patch_token", "audio_patch_token",
            "image_start_token", "video_start_token", "audio_start_token",
            "image_end_token", "video_end_token", "audio_end_token",
        ):
            v = getattr(llm, name)
            if not (0 <= v < llm.vocab_size):
                raise ValueError(
                    f"ThinkerLLMConfig.{name}={v} is out of range for "
                    f"vocab_size={llm.vocab_size}"
                )

    # ------------------------------------------------------------------
    # Convenience accessors (downstream code reads these — keep stable)
    # ------------------------------------------------------------------

    @property
    def thinker_hidden_size(self) -> int:
        return self.thinker_llm.hidden_size

    @property
    def thinker_num_layers(self) -> int:
        return self.thinker_llm.num_hidden_layers

    @property
    def thinker_head_dim(self) -> int:
        assert self.thinker_llm.head_dim is not None
        return self.thinker_llm.head_dim

    @property
    def thinker_num_kv_heads(self) -> int:
        return self.thinker_llm.num_key_value_heads

    @property
    def vocab_size(self) -> int:
        return self.thinker_llm.vocab_size

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, local_dir: str | os.PathLike[str]) -> MingFlashOmniModelConfig:
        """Load configuration from a local HF checkpoint directory.

        Reads ``config.json`` for the thinker path. Lazy-loads ``talker/`` and
        the imagegen subdir family if present — a thinker-only snapshot will
        leave those as None.
        """
        local_dir = str(local_dir)
        config_path = Path(local_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {local_dir}")

        with open(config_path) as f:
            raw: dict[str, Any] = json.load(f)

        thinker_llm = ThinkerLLMConfig.from_dict(raw.get("llm_config", {}))
        vision = VisionEncoderConfig.from_dict(raw.get("vision_config", {}))
        audio_encoder = AudioEncoderConfig.from_dict(raw.get("audio_config", {}))
        mlp_depth = int(raw.get("mlp_depth", 2))

        talker = TalkerConfig.from_subdir(Path(local_dir) / "talker")
        image_gen = ImageGenConfig.from_subdirs(local_dir)

        return cls(
            local_dir=local_dir,
            mlp_depth=mlp_depth,
            thinker_llm=thinker_llm,
            vision=vision,
            audio_encoder=audio_encoder,
            talker=talker,
            image_gen=image_gen,
        )
