"""Configuration for LTX-2.5, read from the diffusers snapshot's component configs.

Architecture comes from ``transformer/config.json``, ``connectors/config.json``,
``text_encoder/config.json`` and the VAE / vocoder configs, so nothing structural is
hardcoded here. What diffusers keeps in pipeline code rather than in a config (the
1024-token prompt length, the distilled sigma schedules, the request defaults of the
model card's quick start) is below, each named for the constant it mirrors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LTX25_REPO = "Lightricks/LTX-2.5-Diffusers"

# The components this model serves. The repo also ships ``transformer_full/`` (the SFT
# DiT, 38 GB), a prompt enhancer and a diffusion decoder, none of which the distilled
# recipe loads.
SNAPSHOT_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "connectors/config.json",
    "connectors/*-of-*.safetensors",
    "connectors/*.index.json",
    "transformer/config.json",
    "transformer/*-of-00004.safetensors",
    "transformer/*.index.json",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "latent_upsampler/*",
]

# Graph names shared by the model file and the submodules.
TEXT_ENCODER_NODE = "text_encoder"
DIT_NODE = "dit"
UPSAMPLER_NODE = "latent_upsampler"
VAE_DECODER_NODE = "vae_decoder"
AUDIO_DECODER_NODE = "audio_decoder"
# One loop per denoising stage; the two-stage recipe runs the dit node in both.
DENOISE_LOOP = "denoise_loop"
REFINE_LOOP = "refine_loop"
# Resource key of the dit's ragged self-attention.
DIT_ATTN = "dit_attn"

# diffusers.pipelines.ltx2.utils
DISTILLED_SIGMA_VALUES = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
STAGE_2_DISTILLED_SIGMA_VALUES = (0.909375, 0.725, 0.421875)
# LTX2Pipeline.encode_prompt(max_sequence_length=1024)
TEXT_MAX_SEQ_LEN = 1024


def _from_checkpoint(cls, cfg: dict):
    """``cls`` from a checkpoint config whose keys are ``cls``'s field names: the keys
    the port doesn't read are dropped, JSON lists become tuples."""
    names = cls.__dataclass_fields__
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names})


@dataclass(frozen=True)
class LTX2TransformerConfig:
    """``transformer/config.json`` of ``LTX2VideoTransformer3DModel``; the defaults are
    LTX-2.5's.

    Only the LTX-2.5 configuration is implemented; ``validate`` rejects flags this port
    does not carry (an LTX-2.0 caption projection, interleaved RoPE, ...).
    """

    in_channels: int = 128
    out_channels: int = 128
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    cross_attention_dim: int = 4096
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    num_layers: int = 48
    norm_eps: float = 1e-6
    ff_bias: bool = False
    audio_ff_bias: bool = True
    attention_bias: bool = True
    attention_out_bias: bool = True
    gated_attn: bool = True
    audio_gated_attn: bool = True
    cross_attn_mod: bool = True
    audio_cross_attn_mod: bool = True
    use_prompt_adaln_single: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    rope_double_precision: bool = True
    causal_offset: int = 1
    pos_embed_max_pos: int = 20
    audio_pos_embed_max_pos: int = 20
    base_height: int = 2048
    base_width: int = 2048
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    audio_sampling_rate: int = 16000
    audio_hop_length: int = 160
    audio_scale_factor: int = 4
    timestep_scale_multiplier: int = 1000
    cross_attn_timestep_scale_multiplier: int = 1000
    patch_size: int = 1
    patch_size_t: int = 1
    audio_patch_size: int = 1
    audio_patch_size_t: int = 1
    use_prompt_embeddings: bool = False
    qk_norm: str = "rms_norm_across_heads"
    activation_fn: str = "gelu-approximate"
    norm_elementwise_affine: bool = False

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "LTX2TransformerConfig":
        out = _from_checkpoint(cls, cfg)
        out.validate()
        return out

    def validate(self) -> None:
        expected = {
            "rope_type": "split", "qk_norm": "rms_norm_across_heads", "activation_fn": "gelu-approximate",
            "use_prompt_embeddings": False, "norm_elementwise_affine": False, "patch_size": 1,
            "patch_size_t": 1, "audio_patch_size": 1, "audio_patch_size_t": 1,
            "cross_attn_mod": True, "audio_cross_attn_mod": True,
        }
        wrong = {k: getattr(self, k) for k, v in expected.items() if getattr(self, k) != v}
        if wrong:
            raise NotImplementedError(
                f"LTX-2.5 port implements {expected}; this checkpoint has {wrong}"
            )


def _gemma4_layer_types(num_layers: int = 48, every: int = 6) -> tuple[str, ...]:
    """Gemma-4's pattern: every ``every``-th layer global, the rest sliding."""
    return tuple("full_attention" if (i + 1) % every == 0 else "sliding_attention" for i in range(num_layers))


@dataclass(frozen=True)
class GemmaTextConfig:
    """``text_encoder/config.json``'s ``text_config`` (Gemma-4 unified), text tower only;
    the defaults are LTX-2.5's Gemma-4 12B."""

    hidden_size: int = 3840
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int = 1
    intermediate_size: int = 15360
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    layer_types: tuple[str, ...] = field(default_factory=_gemma4_layer_types)
    sliding_window: int = 1024
    attention_k_eq_v: bool = True
    rope_parameters: dict = field(default_factory=lambda: {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
    })
    pad_token_id: int = 0

    @property
    def sliding_rope_theta(self) -> float:
        return self.rope_parameters["sliding_attention"]["rope_theta"]

    @property
    def global_rope_theta(self) -> float:
        return self.rope_parameters["full_attention"]["rope_theta"]

    @property
    def global_partial_rotary_factor(self) -> float:
        return self.rope_parameters["full_attention"].get("partial_rotary_factor", 1.0)

    @classmethod
    def from_dict(cls, cfg: dict) -> "GemmaTextConfig":
        tc = cfg.get("text_config", cfg)
        rope = tc["rope_parameters"]
        if rope["sliding_attention"]["rope_type"] != "default" or rope["full_attention"]["rope_type"] != "proportional":
            raise NotImplementedError(f"Gemma-4 port implements default/proportional RoPE; got {rope}")
        for flag, ok in (("num_kv_shared_layers", 0), ("hidden_size_per_layer_input", 0),
                         ("enable_moe_block", False), ("use_double_wide_mlp", False), ("attention_bias", False)):
            if tc.get(flag, ok) != ok:
                raise NotImplementedError(f"Gemma-4 port does not implement {flag}={tc[flag]!r}")
        if tc["hidden_activation"] != "gelu_pytorch_tanh":
            raise NotImplementedError(f"Gemma-4 port implements gelu_pytorch_tanh; got {tc['hidden_activation']}")
        return _from_checkpoint(cls, tc)


@dataclass(frozen=True)
class ConnectorConfig:
    """``connectors/config.json`` of ``LTX2TextConnectors`` (LTX-2.3+ per-modality
    projections); the defaults are LTX-2.5's."""

    caption_channels: int = 3840
    text_proj_in_factor: int = 49
    video_connector_num_attention_heads: int = 32
    video_connector_attention_head_dim: int = 128
    video_connector_num_layers: int = 8
    video_connector_num_learnable_registers: int = 128
    video_gated_attn: bool = True
    audio_connector_num_attention_heads: int = 32
    audio_connector_attention_head_dim: int = 64
    audio_connector_num_layers: int = 8
    audio_connector_num_learnable_registers: int = 128
    audio_gated_attn: bool = True
    connector_rope_base_seq_len: int = 4096
    rope_theta: float = 10000.0
    video_hidden_dim: int = 4096
    audio_hidden_dim: int = 2048
    proj_bias: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "ConnectorConfig":
        if not cfg.get("per_modality_projections", False) or cfg.get("rope_type") != "split":
            raise NotImplementedError("LTX-2.5 connector port implements per-modality projections with split RoPE")
        if cfg.get("causal_temporal_positioning", False):
            raise NotImplementedError("causal_temporal_positioning is not implemented")
        return _from_checkpoint(cls, cfg)


@dataclass(frozen=True)
class LatentGeometry:
    """Pixel <-> latent arithmetic of the video and audio VAEs."""

    spatial_compression: int = 32
    temporal_compression: int = 8
    latent_channels: int = 128
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_temporal_compression: int = 4
    audio_mel_bins: int = 64
    audio_mel_compression: int = 4
    audio_latent_channels: int = 8
    output_sample_rate: int = 48000
    output_audio_channels: int = 2

    @property
    def audio_latents_per_second(self) -> float:
        return self.audio_sample_rate / self.audio_hop_length / float(self.audio_temporal_compression)

    @property
    def audio_latent_mel_bins(self) -> int:
        return self.audio_mel_bins // self.audio_mel_compression

    @property
    def audio_token_channels(self) -> int:
        return self.audio_latent_channels * self.audio_latent_mel_bins

    def latent_frames(self, num_frames: int) -> int:
        return (num_frames - 1) // self.temporal_compression + 1

    def audio_frames(self, num_frames: int, fps: float) -> int:
        # LTX2Pipeline.__call__: round(duration_s * audio_latents_per_second)
        return round(num_frames / fps * self.audio_latents_per_second)


@dataclass(frozen=True)
class LTX25Config:
    """The whole model's config. The defaults are LTX-2.5's, which is what dummy-mode
    tests build; serving reads the checkpoint (``from_snapshot``)."""

    transformer: LTX2TransformerConfig = field(default_factory=LTX2TransformerConfig)
    text: GemmaTextConfig = field(default_factory=GemmaTextConfig)
    connectors: ConnectorConfig = field(default_factory=ConnectorConfig)
    geometry: LatentGeometry = field(default_factory=LatentGeometry)

    # Request defaults: the model card's quick start (single stage) and two-stage recipe.
    default_height: int = 544
    default_width: int = 960
    default_num_frames: int = 121
    default_fps: float = 24.0
    text_max_seq_len: int = TEXT_MAX_SEQ_LEN

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "LTX25Config":
        def read(rel: str) -> dict:
            with open(snapshot / rel) as f:
                return json.load(f)

        vae, audio_vae, vocoder = read("vae/config.json"), read("audio_vae/config.json"), read("vocoder/config.json")
        geometry = LatentGeometry(
            spatial_compression=int(vae["spatial_compression_ratio"]),
            temporal_compression=int(vae["temporal_compression_ratio"]),
            latent_channels=int(vae["latent_channels"]),
            audio_sample_rate=int(audio_vae["sample_rate"]),
            audio_hop_length=int(audio_vae["mel_hop_length"]),
            # AutoencoderKLLTX2Audio downsamples once per ch_mult level past the first
            audio_temporal_compression=2 ** (len(audio_vae["ch_mult"]) - 1),
            audio_mel_bins=int(audio_vae["mel_bins"]),
            audio_mel_compression=2 ** (len(audio_vae["ch_mult"]) - 1),
            audio_latent_channels=int(audio_vae["latent_channels"]),
            output_sample_rate=int(vocoder["output_sampling_rate"]),
            output_audio_channels=int(vocoder.get("bwe_out_channels", vocoder["out_channels"])),
        )
        return cls(
            transformer=LTX2TransformerConfig.from_dict(read("transformer/config.json")),
            text=GemmaTextConfig.from_dict(read("text_encoder/config.json")),
            connectors=ConnectorConfig.from_dict(read("connectors/config.json")),
            geometry=geometry,
        )
