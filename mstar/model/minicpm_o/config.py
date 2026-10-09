"""MiniCPM-o 4.5's checkpoint config, split per node, and its resource keys.

The checkpoint is one ``config.json`` holding the Qwen3-8B LLM at the top
level and the vision (SigLIP navit), audio (Whisper-medium encoder) and TTS
(Llama) towers as sub-configs. Token ids are read off the tokenizer, not this
file: the two disagree on what ends a turn (see ``MiniCPMOConfig.stop_token_ids``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.qwen3_lm import Qwen3LMConfig

# resource keys
LLM_KV = "llm_kv"
LLM_ATTN = "llm_attn"
LLM_POS = "llm_pos"
LLM_SAMPLER = "llm_sampler"
VISION_ATTN = "vision_attn"
RESAMPLER_ATTN = "resampler_attn"
AUDIO_ATTN = "audio_attn"

# ragged span labels
PATCHES = "patches"
QUERIES = "queries"


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    image_size: int = 980
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size


@dataclass
class ResamplerConfig:
    num_queries: int = 64
    # the LLM's width; the resampler attends in it
    embed_dim: int = 4096
    kv_dim: int = 1152

    @property
    def num_heads(self) -> int:
        return self.embed_dim // 128

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class AudioConfig:
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    # post-conv frames (50 a second) per attention block
    chunk_frames: int = 50
    # encoder frames averaged into one LLM token
    pool_step: int = 5
    # the LLM's width
    output_dim: int = 4096

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def projector_dim(self) -> int:
        # upstream sizes the projector's input off the FFN width
        return self.encoder_ffn_dim // 4

    def conv_frames(self, mel_frames: int) -> int:
        """Encoder frames after the stride-2 ``conv2``."""
        return (mel_frames - 1) // 2 + 1

    def pooled_tokens(self, mel_frames: int) -> int:
        """LLM tokens one mel span becomes; ``AvgPool1d`` drops the tail."""
        return (self.conv_frames(mel_frames) - self.pool_step) // self.pool_step + 1


@dataclass
class MiniCPMOConfig:
    llm: Qwen3LMConfig
    vision: VisionConfig
    resampler: ResamplerConfig
    audio: AudioConfig
    # tokens that end a reply; filled from the tokenizer by the model
    stop_token_ids: tuple[int, ...] = ()
    tts_config: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, local_dir: str) -> "MiniCPMOConfig":
        raw = json.loads((Path(local_dir) / "config.json").read_text())
        if str(raw.get("version")) != "4.5":
            raise ValueError(
                f"{local_dir} is MiniCPM-o {raw.get('version')!r}; this port is 4.5 only"
            )
        vision_raw = raw["vision_config"]
        vision = VisionConfig(**{
            k: vision_raw[k] for k in VisionConfig.__dataclass_fields__ if k in vision_raw
        })
        if raw.get("drop_vision_last_layer", False):
            vision.num_hidden_layers -= 1
        audio_raw = raw["audio_config"]
        audio = AudioConfig(
            d_model=audio_raw["d_model"],
            encoder_layers=audio_raw["encoder_layers"],
            encoder_attention_heads=audio_raw["encoder_attention_heads"],
            encoder_ffn_dim=audio_raw["encoder_ffn_dim"],
            num_mel_bins=audio_raw["num_mel_bins"],
            max_source_positions=audio_raw["max_source_positions"],
            activation_function=audio_raw["activation_function"],
            chunk_frames=int(raw["audio_chunk_length"] * 50),
            pool_step=raw["audio_pool_step"],
            output_dim=raw["hidden_size"],
        )
        return cls(
            llm=Qwen3LMConfig.from_hf(raw),
            vision=vision,
            resampler=ResamplerConfig(
                num_queries=raw["query_num"],
                embed_dim=raw["hidden_size"],
                kv_dim=vision.hidden_size,
            ),
            audio=audio,
            tts_config=raw.get("tts_config", {}),
        )
