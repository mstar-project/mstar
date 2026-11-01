"""Checkpoint-backed configuration for VoxCPM2 (``openbmb/VoxCPM2``).

``config.json`` describes every component: the MiniCPM-4 text-semantic LM, the
residual acoustic LM, the local encoder and local DiT (both MiniCPM-4 stacks
read bidirectionally), and the AudioVAE V2 (16 kHz in, 48 kHz out). The
dataclasses below mirror that file; anything the checkpoint does not carry is
a generation default of the reference ``VoxCPM.generate``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Talker resources: the base LM's paged KV + attention + RoPE positions, and
# the residual LM's paged KV + attention (it has no RoPE).
BASE_KV = "base_kv"
BASE_ATTN = "base_attn"
BASE_POS = "base_pos"
RES_KV = "res_kv"
RES_ATTN = "res_attn"

# Special token ids of the VoxCPM2 tokenizer (``VoxCPM2Model.__init__``).
AUDIO_START_TOKEN = 101
AUDIO_END_TOKEN = 102
REF_AUDIO_START_TOKEN = 103
REF_AUDIO_END_TOKEN = 104

# Prompt modes, as ``VoxCPM2Model._generate_with_prompt_cache`` names them.
MODE_ZERO_SHOT = 0
MODE_REFERENCE = 1  # ref audio isolated between ref_start/ref_end tokens
MODE_REF_CONTINUATION = 2  # the reference prefix, then text + its audio to continue

# Request facts settled while tokenizing, which the Talker partition's nodes
# read from ``step_metadata`` rather than off a device tensor. The conductor
# merges step metadata into the partition's kwargs, hence model-prefixed names.
# The patch budget also lets the conductor end a one-patch request after
# prefill, where the loop that would stop it is absent.
PATCH_BUDGET_KEY = "talker_patch_budget"
PROMPT_MODE_KEY = "voxcpm2_prompt_mode"
MIN_LEN_KEY = "voxcpm2_min_len"
# content hash of the reference clip, the RefEncoder's memo key with the mode
REF_DIGEST_KEY = "voxcpm2_ref_digest"
TALKER_STEP_METADATA = (PATCH_BUDGET_KEY, PROMPT_MODE_KEY, MIN_LEN_KEY, REF_DIGEST_KEY)


@dataclass
class MiniCPM4Config:
    """One MiniCPM-4 stack; the residual LM, encoder and DiT override sizes."""

    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    vocab_size: int = 73448
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    rope_short_factor: tuple[float, ...] = ()
    rope_long_factor: tuple[float, ...] = ()
    rope_original_max_position_embeddings: int = 32768
    no_rope: bool = False
    use_mup: bool = False

    def resized(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int | None,
    ) -> MiniCPM4Config:
        """A stack sharing this one's RoPE, norms and KV head count (the
        reference deep-copies ``lm_config`` and overrides the sizes)."""
        return MiniCPM4Config(
            **{
                **self.__dict__,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "head_dim": head_dim or hidden_size // num_attention_heads,
                "vocab_size": 0,
            }
        )

    @property
    def rope_factors(self) -> tuple[float, ...]:
        """LongRoPE per-dimension divisors, picked as ``MiniCPMLongRoPE`` does
        for its cache length (``max_position_embeddings``)."""
        if self.max_position_embeddings > self.rope_original_max_position_embeddings:
            return self.rope_long_factor
        return self.rope_short_factor

    @property
    def rope_attention_scale(self) -> float:
        scale = self.max_position_embeddings / self.rope_original_max_position_embeddings
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.rope_original_max_position_embeddings))


@dataclass
class AudioVAEConfig:
    encoder_dim: int = 128
    encoder_rates: tuple[int, ...] = (2, 5, 8, 8)
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: tuple[int, ...] = (8, 6, 5, 2, 2, 2)
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    use_noise_block: bool = False
    sr_bin_boundaries: tuple[int, ...] | None = (20000, 30000, 40000)
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False

    @property
    def encode_hop(self) -> int:
        """16 kHz input samples per latent frame (640: 25 Hz)."""
        return math.prod(self.encoder_rates)

    @property
    def decode_hop(self) -> int:
        """48 kHz output samples per latent frame (1920: 25 Hz)."""
        return math.prod(self.decoder_rates)


@dataclass
class VoxCPM2Config:
    lm: MiniCPM4Config = field(default_factory=MiniCPM4Config)
    residual: MiniCPM4Config = field(default_factory=MiniCPM4Config)
    encoder: MiniCPM4Config = field(default_factory=MiniCPM4Config)
    dit: MiniCPM4Config = field(default_factory=MiniCPM4Config)
    audio_vae: AudioVAEConfig = field(default_factory=AudioVAEConfig)
    patch_size: int = 4
    feat_dim: int = 64
    fsq_latent_dim: int = 512
    fsq_scale: int = 9
    dit_mean_mode: bool = False
    max_length: int = 8192

    # Generation defaults of the reference ``VoxCPM.generate``.
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    min_len: int = 2
    max_len: int = 4096
    # Each request stops after ``target_text_tokens * ratio + 10`` patches at
    # most (``_generate_with_prompt_cache``'s ``retry_badcase_ratio_threshold``).
    max_len_ratio: float = 6.0
    # Continuation prompts lead the VAE stream with this many prompt patches
    # (``streaming_prefix_len - 1``), as the reference's decode does.
    prompt_context_patches: int = 3

    @property
    def patch_samples(self) -> int:
        """48 kHz samples one LM step (one patch) decodes to: 7680, 160 ms."""
        return self.patch_size * self.audio_vae.decode_hop

    @property
    def ref_patch_samples(self) -> int:
        """16 kHz samples of reference audio per patch: 2560."""
        return self.patch_size * self.audio_vae.encode_hop

    @property
    def output_sample_rate(self) -> int:
        return self.audio_vae.out_sample_rate

    def max_patches(self, target_text_tokens: int, max_len: int | None = None) -> int:
        cap = self.max_len if max_len is None else max_len
        return max(1, min(int(target_text_tokens * self.max_len_ratio + 10), cap))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoxCPM2Config:
        arch = data.get("architecture", "voxcpm2")
        if arch != "voxcpm2":
            raise ValueError(f"VoxCPM2 port loads 'voxcpm2' checkpoints, got architecture {arch!r}")
        lm_raw = data["lm_config"]
        if lm_raw.get("use_mup", False):
            raise ValueError("VoxCPM2 port does not implement muP scaling (use_mup=true)")
        rope = lm_raw.get("rope_scaling") or {}
        lm = MiniCPM4Config(
            hidden_size=lm_raw["hidden_size"],
            intermediate_size=lm_raw["intermediate_size"],
            num_hidden_layers=lm_raw["num_hidden_layers"],
            num_attention_heads=lm_raw["num_attention_heads"],
            num_key_value_heads=lm_raw["num_key_value_heads"],
            head_dim=lm_raw.get("kv_channels") or lm_raw["hidden_size"] // lm_raw["num_attention_heads"],
            rms_norm_eps=lm_raw["rms_norm_eps"],
            vocab_size=lm_raw["vocab_size"],
            rope_theta=lm_raw.get("rope_theta", 10000.0),
            max_position_embeddings=lm_raw["max_position_embeddings"],
            rope_short_factor=tuple(rope.get("short_factor", ())),
            rope_long_factor=tuple(rope.get("long_factor", ())),
            rope_original_max_position_embeddings=rope.get(
                "original_max_position_embeddings", lm_raw["max_position_embeddings"]
            ),
            no_rope=lm_raw.get("no_rope", False),
        )
        residual = MiniCPM4Config(
            **{
                **lm.__dict__,
                "num_hidden_layers": data.get("residual_lm_num_layers", 8),
                "no_rope": data.get("residual_lm_no_rope", False),
                "vocab_size": 0,
            }
        )

        def local(raw: dict[str, Any]) -> MiniCPM4Config:
            return lm.resized(
                hidden_size=raw.get("hidden_dim", 1024),
                intermediate_size=raw.get("ffn_dim", 4096),
                num_attention_heads=raw.get("num_heads", 16),
                num_hidden_layers=raw.get("num_layers", 4),
                head_dim=raw.get("kv_channels"),
            )

        dit_raw = data.get("dit_config", {})
        vae_raw = dict(data.get("audio_vae_config") or {})
        for key in ("encoder_rates", "decoder_rates", "sr_bin_boundaries"):
            if vae_raw.get(key) is not None:
                vae_raw[key] = tuple(vae_raw[key])
        return cls(
            lm=lm,
            residual=residual,
            encoder=local(data.get("encoder_config", {})),
            dit=local(dit_raw),
            audio_vae=AudioVAEConfig(**vae_raw),
            patch_size=data.get("patch_size", 4),
            feat_dim=data.get("feat_dim", 64),
            fsq_latent_dim=data.get("scalar_quantization_latent_dim", 512),
            fsq_scale=data.get("scalar_quantization_scale", 9),
            dit_mean_mode=bool(dit_raw.get("mean_mode", dit_raw.get("dit_mean_mode", False))),
            max_length=data.get("max_length", 8192),
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> VoxCPM2Config:
        with (Path(path) / "config.json").open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
