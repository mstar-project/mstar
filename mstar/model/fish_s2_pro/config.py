"""Checkpoint-backed configuration for Fish Audio S2-Pro (``fishaudio/s2-pro``).

S2-Pro is a dual-autoregressive TTS model:

* the Slow AR (``text_config``) is a Qwen3-4B-shaped decoder over a Qwen3
  vocabulary extended with 4,096 ``<|semantic:i|>`` tokens. It predicts one
  semantic code per 21.5 Hz frame;
* the Fast AR (``audio_decoder_config``) is a 4-layer decoder that walks the
  9 residual codebooks of that frame from the Slow AR hidden state;
* the codec (``codec.pth``) is a causal "modded DAC" at 44.1 kHz with a hop
  of 2,048 samples and 10 codebooks (1 semantic + 9 residual).

``config.json`` uses Fish field names (``dim``, ``n_head``, ``n_local_heads``,
``rope_base`` ...); the dataclasses below keep those names. The codec has no
JSON config in the repository; ``FishDACConfig`` carries the architecture of
``fish_speech/configs/modded_dac_vq.yaml`` from the Apache-2.0
``fish-speech`` 0.1.0 PyPI release, which matches ``codec.pth`` (its RoPE
tables have the 4,096 / 16,384 positions set there).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resource label constants (AR node)
# ---------------------------------------------------------------------------
AR_KV = "slow_ar_kv"
AR_ATTN = "slow_ar_attn"
AR_POS = "slow_ar_pos"
AR_SAMPLER = "semantic_sampler"

# Step-metadata keys ``process_prompt`` settles on the host for the prefill:
# ``[prefix_len, ref_frames]``, and a clone's reference-clip digest.
PROMPT_LAYOUT_KEY = "prompt_layout"
REF_CLIP_KEY = "ref_clip_key"

# Fixed prompt text of Fish Audio's published S2-Pro serving recipe, which
# tokenizes the segments one by one exactly as here (the fish-speech 0.1.0
# package has no S2-Pro prompt).
# test_prompt_layout_matches_the_reference_segments pins the layout.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VOICE_TOKEN = "<|voice|>"
CLONE_SYSTEM_TEXT = "convert the provided text to speech reference to the following:\n\nText:\n"
CLONE_SPEECH_TEXT = "\n\nSpeech:\n"
# Control tokens a request's text may carry (phoneme spans); every other
# special token would let the text rewrite the chat structure.
ALLOWED_TEXT_SPECIAL_TOKENS = ("<|phoneme_start|>", "<|phoneme_end|>")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _pick(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    return {name: data[name] for name in cls.__dataclass_fields__ if name in data}


@dataclass
class FishSlowARConfig:
    """``text_config``: the Qwen3-4B-shaped time-axis decoder."""

    dim: int = 2560
    n_layer: int = 36
    n_head: int = 32
    n_local_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0
    max_seq_len: int = 32768
    vocab_size: int = 155776
    tie_word_embeddings: bool = True
    attention_qk_norm: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FishSlowARConfig":
        return cls(**_pick(cls, data))

    def qwen3_lm_config(self):
        """The same decoder in ``Qwen3DenseLM`` terms."""
        from mstar.model.components.qwen3_lm import Qwen3LMConfig

        return Qwen3LMConfig(
            hidden_size=self.dim,
            num_hidden_layers=self.n_layer,
            num_attention_heads=self.n_head,
            num_key_value_heads=self.n_local_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.norm_eps,
            rope_theta=self.rope_base,
            max_position_embeddings=self.max_seq_len,
            tie_word_embeddings=self.tie_word_embeddings,
        )


@dataclass
class FishFastARConfig:
    """``audio_decoder_config``: the depth-axis residual-codebook decoder."""

    dim: int = 2560
    text_dim: int = 2560
    n_layer: int = 4
    n_head: int = 32
    n_local_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0
    max_seq_len: int = 11
    vocab_size: int = 4096
    num_codebooks: int = 10
    attention_qk_norm: bool = False
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FishFastARConfig":
        return cls(**_pick(cls, data))


@dataclass
class FishDACConfig:
    """Modded DAC codec architecture plus M* streaming controls.

    The architecture fields mirror ``modded_dac_vq.yaml``. The stream policy
    gives every codec window enough already-decoded frames in front of its
    new ones that the new frames decode exactly as in one decode of the
    whole stream: the quantizer's post transformer looks back
    ``post_layers * (post_window - 1)`` frames and the causal conv stack
    after it ``conv_context_frames`` more.
    """

    sample_rate: int = 44100
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 8, 8)
    encoder_transformer_layers: tuple[int, ...] = (0, 0, 0, 4)
    encoder_window: int = 512
    encoder_block_size: int = 16384
    decoder_dim: int = 1536
    decoder_rates: tuple[int, ...] = (8, 8, 4, 2)
    latent_dim: int = 1024
    downsample_factor: tuple[int, ...] = (2, 2)
    semantic_codebook_size: int = 4096
    codebook_size: int = 1024
    codebook_dim: int = 8
    n_residual_codebooks: int = 9
    # Quantizer pre/post WindowLimitedTransformer (one shared config).
    post_layers: int = 8
    post_heads: int = 16
    post_head_dim: int = 64
    post_intermediate_size: int = 3072
    post_window: int = 128
    post_block_size: int = 4096
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    # Streaming (codec windows). The first windows are small so the first
    # audio leaves a few AR steps after prefill; then ``chunk_frames`` new
    # frames per window.
    chunk_schedule: tuple[int, ...] = (4, 8, 16)
    chunk_frames: int = 32
    # None: the exact receptive field (transformer + conv stack).
    left_context_frames: int | None = None
    conv_context_frames: int = 10

    def __post_init__(self) -> None:
        for name in (
            "encoder_rates",
            "encoder_transformer_layers",
            "decoder_rates",
            "downsample_factor",
            "chunk_schedule",
        ):
            setattr(self, name, tuple(int(v) for v in getattr(self, name)))
        if self.left_context_frames is None:
            self.left_context_frames = self.transformer_context_frames + self.conv_context_frames
        if self.chunk_frames <= 0 or any(size <= 0 for size in self.chunk_schedule):
            raise ValueError("Fish codec chunk sizes must be positive")
        if self.left_context_frames < 0 or self.conv_context_frames < 0:
            raise ValueError("Fish codec context frames must be non-negative")

    @property
    def num_codebooks(self) -> int:
        return 1 + self.n_residual_codebooks

    @property
    def hop_length(self) -> int:
        """Audio samples per codec frame (2,048: 512 in the conv stack x 4 in the quantizer)."""
        return math.prod(self.encoder_rates) * math.prod(self.downsample_factor)

    @property
    def transformer_context_frames(self) -> int:
        """Frames the quantizer's post transformer looks back over (8 x 127)."""
        return self.post_layers * (self.post_window - 1)

    @property
    def max_frames(self) -> int:
        """Longest stream the post transformer's RoPE table covers."""
        return self.post_block_size

    @property
    def exact_streaming(self) -> bool:
        return self.left_context_frames >= self.transformer_context_frames + self.conv_context_frames

    def frames_for_samples(self, num_samples: int) -> int:
        """Codec frames the encoder emits for ``num_samples`` of 44.1 kHz audio."""
        return -(-int(num_samples) // self.hop_length)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FishDACConfig":
        return cls(**_pick(cls, data))


@dataclass
class FishGenerationConfig:
    """Request sampling defaults.

    Temperature, top-p and top-k are the defaults of Fish Audio's published
    S2-Pro serving recipe (test_sampling_defaults_and_greedy pins them).
    The residual codebooks are argmax, as there. That recipe's 16-token
    repetition-penalty window and repetition-aware sampling are not ported:
    M*'s penalty counts the whole generation, so it defaults to off.
    """

    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FishGenerationConfig":
        return cls(**_pick(cls, data))


@dataclass
class FishS2ProConfig:
    """Top-level S2-Pro metadata composed from ``config.json`` and the codec."""

    model_type: str = "fish_qwen3_omni"
    semantic_start_token_id: int = 151678
    semantic_end_token_id: int = 155773
    eos_token_id: int = 151645
    slow_ar: FishSlowARConfig = field(default_factory=FishSlowARConfig)
    fast_ar: FishFastARConfig = field(default_factory=FishFastARConfig)
    codec: FishDACConfig = field(default_factory=FishDACConfig)
    generation: FishGenerationConfig = field(default_factory=FishGenerationConfig)
    # Accepted reference-clip duration for voice cloning.
    reference_min_seconds: float = 0.5
    reference_max_seconds: float = 30.0

    SUPPORTED_MODEL_TYPES = ("fish_qwen3_omni",)

    def __post_init__(self) -> None:
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported Fish checkpoint model_type {self.model_type!r}; "
                f"supported: {', '.join(self.SUPPORTED_MODEL_TYPES)}"
            )
        if self.num_semantic_codes != self.fast_ar.vocab_size:
            raise ValueError(
                f"Fish semantic token range holds {self.num_semantic_codes} codes but the "
                f"Fast AR vocabulary has {self.fast_ar.vocab_size}"
            )
        if self.num_semantic_codes != self.codec.semantic_codebook_size:
            raise ValueError(
                f"Fish semantic token range holds {self.num_semantic_codes} codes but the "
                f"codec's semantic codebook has {self.codec.semantic_codebook_size}"
            )
        if self.fast_ar.num_codebooks != self.codec.num_codebooks:
            raise ValueError(
                f"Fish Fast AR predicts {self.fast_ar.num_codebooks} codebooks but the codec "
                f"decodes {self.codec.num_codebooks}"
            )
        if self.fast_ar.text_dim != self.slow_ar.dim:
            raise ValueError("Fish Fast AR text_dim must equal the Slow AR width")
        if self.generation.max_new_tokens > self.codec.max_frames:
            raise ValueError(
                f"max_new_tokens {self.generation.max_new_tokens} exceeds the codec's "
                f"{self.codec.max_frames}-frame RoPE table"
            )

    @property
    def num_semantic_codes(self) -> int:
        return self.semantic_end_token_id - self.semantic_start_token_id + 1

    @property
    def eos_code(self) -> int:
        """Index of ``<|im_end|>`` in the semantic head (after the 4,096 codes)."""
        return self.num_semantic_codes

    @property
    def semantic_head_size(self) -> int:
        """Semantic codes plus ``<|im_end|>``: the only tokens the Slow AR may emit."""
        return self.num_semantic_codes + 1

    @property
    def num_codebooks(self) -> int:
        return self.fast_ar.num_codebooks

    @property
    def vq_scale(self) -> float:
        """Scale of a frame's input embedding: token + 10 codebook embeddings."""
        return 1.0 / math.sqrt(self.num_codebooks + 1)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        codec_overrides: dict[str, Any] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> "FishS2ProConfig":
        data = _read_json(Path(model_dir) / "config.json")
        values = {
            name: data[name]
            for name in ("model_type", "semantic_start_token_id", "semantic_end_token_id", "eos_token_id")
            if name in data
        }
        values.update(
            slow_ar=FishSlowARConfig.from_dict(data.get("text_config", {})),
            fast_ar=FishFastARConfig.from_dict(data.get("audio_decoder_config", {})),
            codec=FishDACConfig.from_dict(dict(codec_overrides or {})),
            generation=FishGenerationConfig.from_dict(dict(generation_overrides or {})),
        )
        return cls(**values)
