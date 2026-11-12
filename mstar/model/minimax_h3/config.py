"""MiniMax-H3's configuration, read from the checkpoint's ``FL2VA/`` components; what the
reference keeps in pipeline code (modality tags, the 24 fps clock, the canvas rule, frame
alignment, the keyframes' noise level) is below, named for diffusers' constants.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig

H3_REPO = "MiniMaxAI/MiniMax-H3"
PARTITION = "FL2VA"
# The repo also holds Ref2VA/ (144 GB) and a diffusers copy at the root; FL2VA serves alone.
SNAPSHOT_PATTERNS = ["model_index.json", f"{PARTITION}/*"]

# Graph names shared by the model file and the submodules.
TEXT_ENCODER_NODE = "text_encoder"
DIT_NODE = "dit"
VAE_DECODER_NODE = "vae_decoder"
VAE_ENCODER_NODE = "vae_encoder"  # FL2VA keyframes
AUDIO_DECODER_NODE = "audio_decoder"
DENOISE_LOOP = "denoise_loop"
# The DiT's packed rows and the token refiner's prompt rows, and the resource keys of the
# attentions over them.
DIT_ROWS, PROMPT_ROWS = "dit_rows", "prompt_rows"
DIT_ATTN = "dit_attn"
# on SP ranks, splits the dit's packed rows over them
DIT_SHARD = "dit_shard"
# on SP ranks, splits the video decoder's clips over them
VAE_SHARD = "vae_shard"
REFINER_ATTN = "refiner_attn"

# Per-row modality tags. They index the AdaLN tables, so the values are a checkpoint
# contract (MINIMAX_H3_{VIDEO,TEXT,AUDIO}_TAG).
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
NUM_MODALITIES = 3

FPS = 24                       # MINIMAX_H3_FPS
AUDIO_LATENTS_PER_SECOND = 40  # 32 kHz audio hopped by 800
AUDIO_CHANNELS = 2             # stereo, packed channel-major
MIN_DURATION, MAX_DURATION = 5.0, 15.0
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 1 / 4, 4
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
DEFAULT_NUM_FRAMES = 124       # 17 * 7 + 5, 5.17 s
DEFAULT_NUM_INFERENCE_STEPS = 50
# hidden_states[50] of the Qwen3-VL conditioner (the last layer is post-norm)
TEXT_ENCODER_LAYER = 50
TEXT_MAX_POSITIONS = 262_144  # Qwen3-VL's max_position_embeddings
# A visual conditioning anchor is held at t = 0.999 (just short of clean), and its
# posterior is sampled under its own seed, rounded to fp16.
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def _fields(cls, cfg: dict) -> dict:
    names = cls.__dataclass_fields__
    return {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names}


@dataclass(frozen=True)
class H3TransformerConfig:
    """``FL2VA/transformer/config.json`` (``MiniMaxH3DiTModel``)."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 96768
    final_adaln_out_features: int = 10752
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # not in the config: diffusers' MiniMaxH3RotaryPosEmbed default, and the stored
    # ``rope.inv_freq`` agrees with it
    rope_theta: float = 10000.0

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def patch_dim(self) -> int:
        return self.latents_dim * math.prod(self.patch_size)

    @property
    def rotary_dim(self) -> int:
        """Rotated channels of every head: three axes of ``rope_inv_freq_len`` angles, doubled."""
        return 2 * 3 * self.rope_inv_freq_len

    def validate(self) -> None:
        if self.adaln_out_features != 6 * self.hidden_size * NUM_MODALITIES:
            raise ValueError(f"adaln_out_features {self.adaln_out_features} is not 6 x hidden x 3 modalities")
        if self.final_adaln_out_features != 2 * self.hidden_size:
            raise ValueError(f"final_adaln_out_features {self.final_adaln_out_features} is not 2 x hidden")
        if self.patch_size[0] != 1:
            raise NotImplementedError(f"temporal patch {self.patch_size[0]}: only 1 is implemented")


@dataclass(frozen=True)
class H3VideoVAEConfig:
    """``FL2VA/video_vae/config.json`` and ``source/config.json``."""

    latent_channels: int = 24
    clip_length: int = 17       # pixel frames encoded per chunk
    token_drop: int = 3         # trailing latent frames dropped per chunk
    spatial_compression: int = 16
    temporal_compression: int = 4
    latents_mean: tuple[float, ...] = (0.0,) * 24
    latents_std: tuple[float, ...] = (1.0,) * 24
    tile_size: int = 256
    tile_overlap_min: int = 64
    # ViT decoder
    decoder_layers: int = 36
    decoder_heads: int = 32
    decoder_head_dim: int = 64
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75

    @property
    def latents_per_chunk(self) -> int:
        """Latent frames a chunk keeps: 5 for 17 frames at /4 (``tokens_chunk_size``)."""
        return math.ceil(self.clip_length / self.temporal_compression)

    @classmethod
    def from_dirs(cls, video_vae: Path) -> "H3VideoVAEConfig":
        top = json.loads((video_vae / "config.json").read_text())
        src = json.loads((video_vae / top.get("source_path", "source") / "config.json").read_text())
        vit = src.get("vit_decoder_kwargs", {})
        return cls(
            latent_channels=int(top["latent_channels"]),
            clip_length=int(top["vae_clip_length"]),
            token_drop=int(top["vae_token_drop"]),
            spatial_compression=int(src["vae_ratio"]),
            temporal_compression=int(src["vae_ratio_t"]),
            latents_mean=tuple(float(x) for x in top["latents_mean"]),
            latents_std=tuple(float(x) for x in top["latents_std"]),
            tile_size=int(top.get("vae_tile_size", 256)),
            tile_overlap_min=int(top.get("vae_tile_overlap_min", 64)),
            decoder_layers=int(vit.get("num_layers", 36)),
            decoder_heads=int(vit.get("heads", 32)),
            decoder_head_dim=int(vit.get("dim_head", 64)),
            decoder_rope_theta=float(vit.get("rope_theta", 100.0)),
            decoder_rope_dim_ratio=float(vit.get("rope_dim_ratio", 0.75)),
        )


@dataclass(frozen=True)
class H3VideoEncoderConfig:
    """``FL2VA/video_vae/source/config.json``, the encoder's half."""

    in_channels: int = 3
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8)
    num_res_blocks: int = 2
    spatial_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    latent_channels: int = 24
    padding_mode: str = "reflect"
    norm_groups: int = 32
    norm_eps: float = 1e-6

    @property
    def block_channels(self) -> tuple[int, ...]:
        return tuple(self.base_channels * m for m in self.channel_mult)

    @classmethod
    def from_dir(cls, video_vae: Path) -> "H3VideoEncoderConfig":
        top = json.loads((video_vae / "config.json").read_text())
        src = json.loads((video_vae / top.get("source_path", "source") / "config.json").read_text())
        if not (src.get("use_3d_conv") and src.get("causal_encoder") and src.get("use_t_isolated_gn")):
            raise ValueError(f"{video_vae}: only the causal 3D encoder with per-frame GroupNorm is supported")
        return cls(
            in_channels=int(src["in_channels"]), base_channels=int(src["ch"]),
            channel_mult=tuple(int(m) for m in src["ch_mult"]), num_res_blocks=int(src["num_res_blocks"]),
            spatial_down=tuple(int(s) for s in src["space_down"]),
            temporal_down=tuple(int(t) for t in src["time_down"]),
            latent_channels=int(src["embed_dim"]), padding_mode=str(src.get("padding_mode", "reflect")),
        )


@dataclass(frozen=True)
class H3AudioVAEConfig:
    """``FL2VA/audio_vae/config.json`` and ``metadata.json``; the decoder half only.

    The BigVGAN's upsampling plan is not in the checkpoint's kwargs: the reference picks
    it by sample rate (``DacAudioVAE`` at 32 kHz: rates 5,5,2,2,2,2,2, kernels 9,9,4,...)
    and ignores the ``decoder_rates`` kwarg, which ``from_dir`` checks agrees.
    """

    sample_rate: int = 32000
    latent_channels: int = 32
    output_channels: int = 2
    latents_mean: tuple[float, ...] = (0.0,) * 32
    latents_std: tuple[float, ...] = (1.0,) * 32
    latent_dim: int = 2048
    decoder_dim: int = 1024
    upsample_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5),) * 3

    @classmethod
    def from_dir(cls, audio_vae: Path) -> "H3AudioVAEConfig":
        top = json.loads((audio_vae / "config.json").read_text())
        meta = json.loads((audio_vae / top.get("source_metadata_path", "metadata.json")).read_text())
        kwargs = meta["metadata"]["kwargs"]
        config = cls(
            sample_rate=int(top["sample_rate"]),
            latent_channels=int(top["latent_channels"]),
            output_channels=int(top.get("output_channel", 2)),
            latents_mean=tuple(float(x) for x in top["latents_mean"]),
            latents_std=tuple(float(x) for x in top["latents_std"]),
            latent_dim=int(kwargs["latent_dim"]),
            decoder_dim=int(kwargs["decoder_dim"]),
        )
        if (kwargs.get("decoder_type", "bigvgan") != "bigvgan" or config.sample_rate != 32000
                or tuple(kwargs.get("decoder_rates", config.upsample_rates)) != config.upsample_rates):
            raise ValueError(f"{audio_vae}: only the 32 kHz BigVGAN decoder is supported, got {kwargs}")
        return config


@dataclass
class H3Config:
    transformer: H3TransformerConfig = field(default_factory=H3TransformerConfig)
    text: QwenEncoderConfig = field(
        default_factory=lambda: QwenEncoderConfig(
            hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_attention_heads=64,
            num_key_value_heads=8, rope_theta=5_000_000.0, hidden_state_layers=(TEXT_ENCODER_LAYER,),
            max_sequence_length=TEXT_MAX_POSITIONS, fused_qkv=False,
            # Qwen3-VL's image pad and its interleaved MRoPE split (FL2VA keyframes)
            image_token_id=151655, mrope_section=(24, 20, 20),
        )
    )
    video_vae: H3VideoVAEConfig = field(default_factory=H3VideoVAEConfig)
    video_encoder: H3VideoEncoderConfig = field(default_factory=H3VideoEncoderConfig)
    audio_vae: H3AudioVAEConfig = field(default_factory=H3AudioVAEConfig)
    # model_index.json "_minimax_h3.sigma_shift_scales"
    video_shift: float = 12.0
    audio_shift: float = 3.0
    # the text encoder's positions: the longest prompt the reference can encode
    text_max_positions: int = TEXT_MAX_POSITIONS

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "H3Config":
        root = Path(snapshot) / PARTITION
        transformer = H3TransformerConfig(**_fields(H3TransformerConfig, json.loads(
            (root / "transformer" / "config.json").read_text())))
        transformer.validate()
        text_encoder = json.loads((root / "text_encoder" / "config.json").read_text())
        max_positions = int(text_encoder["text_config"].get("max_position_embeddings", TEXT_MAX_POSITIONS))
        text = QwenEncoderConfig.from_qwen3_vl(
            text_encoder,
            # three q/k/v GEMMs, as transformers runs them: fused, cuBLAS picks another
            # kernel for the 1024-wide k/v and they drift from the reference's last bit
            hidden_state_layers=(TEXT_ENCODER_LAYER,), fused_qkv=False,
            # no pad-to-length recipe: a prompt runs at its own length, up to the positions
            max_sequence_length=max_positions,
        )
        index = json.loads((root / "model_index.json").read_text()).get("_minimax_h3", {})
        shifts = index.get("sigma_shift_scales", {})
        return cls(
            transformer=transformer,
            text=text,
            video_vae=H3VideoVAEConfig.from_dirs(root / "video_vae"),
            video_encoder=H3VideoEncoderConfig.from_dir(root / "video_vae"),
            audio_vae=H3AudioVAEConfig.from_dir(root / "audio_vae"),
            video_shift=float(shifts.get("video", 12.0)),
            audio_shift=float(shifts.get("audio", 3.0)),
            text_max_positions=max_positions,
        )


# --------------------------------------------------------------------------- geometry
# Ports of diffusers' modular_pipelines/minimax_h3/modular_pipeline.py helpers.

def resolve_canvas_size(
    aspect_width: float, aspect_height: float, multiple: int = 32,
    short_edge: int = CANVAS_SHORT_EDGE, max_pixels: int = CANVAS_MAX_PIXELS,
) -> tuple[int, int]:
    """``(height, width)`` of the canvas for an aspect ratio: short edge first, the area
    capped, both rounded to ``multiple`` (``resolve_canvas_size``)."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"the aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}")
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """Snap up to the next ``17 n + 5`` the video VAE can encode (``align_num_frames``)."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return num_frames + (latents_per_chunk - num_frames) % frames_per_chunk


def video_latent_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """``17 n + 5`` pixel frames -> ``5 n + 2`` latent frames."""
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(f"num_frames must be {frames_per_chunk} n + {latents_per_chunk}, got {num_frames}")
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents per channel covering ``num_frames`` video frames."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))
