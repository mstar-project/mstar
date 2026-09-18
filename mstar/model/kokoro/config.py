"""Kokoro-82M configuration, built from the checkpoint's ``config.json``.

The upstream ``kokoro`` package hard-codes a handful of architecture constants
outside its config (the 24 kHz sample rate and the harmonic source parameters
of the iSTFTNet generator, the 1024-wide decoder trunk, the 64-channel ASR
residual). They are fields here so nothing else in the package carries a
magic number.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Graph names shared by the model, the submodule and the tests.
SYNTH_NODE = "kokoro"
SYNTH_WALK = "synth"
CHUNK_LOOP = "chunk_loop"

# Names of the tensors ``process_prompt`` produces for one request.
PHONEME_IDS = "phoneme_ids"
PHONEME_LENS = "phoneme_lens"
REF_STYLE = "ref_style"
SPEED = "speed"
AUDIO_CHUNK = "audio_chunk"

# Token id used for both the <bos> and the <eos> the model expects around a
# phoneme sequence, and for padding (masked out by the lengths).
BOUNDARY_TOKEN_ID = 0


def _from_dict(cls, data: dict[str, Any]):
    return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


@dataclass
class KokoroBertConfig:
    """PL-BERT: an ALBERT encoder over phoneme tokens (``plbert`` in config.json).

    ``embedding_size`` and ``layer_norm_eps`` are ALBERT defaults the checkpoint
    relies on without stating them.
    """

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12
    dropout: float = 0.1
    embedding_size: int = 128
    layer_norm_eps: float = 1e-12

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class KokoroISTFTNetConfig:
    """The iSTFTNet generator (``istftnet`` in config.json)."""

    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [20, 12])
    upsample_rates: list[int] = field(default_factory=lambda: [10, 6])
    gen_istft_hop_size: int = 5
    gen_istft_n_fft: int = 20
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_initial_channel: int = 512

    @property
    def upsample_factor(self) -> int:
        """Waveform samples per generator input frame (60 * 5 = 300)."""
        return math.prod(self.upsample_rates) * self.gen_istft_hop_size


@dataclass
class KokoroModelConfig:
    """Everything the M* Kokoro integration reads.

    The top block mirrors ``config.json``; the serving block has M*-side
    defaults that a deployment can change through model kwargs.
    """

    # --- config.json ------------------------------------------------------
    n_token: int = 178
    hidden_dim: int = 512
    style_dim: int = 128
    n_layer: int = 3
    max_dur: int = 50
    dropout: float = 0.2
    n_mels: int = 80
    text_encoder_kernel_size: int = 5
    dim_in: int = 64
    max_conv_dim: int = 512
    multispeaker: bool = True
    vocab: dict[str, int] = field(default_factory=dict)
    plbert: KokoroBertConfig = field(default_factory=KokoroBertConfig)
    istftnet: KokoroISTFTNetConfig = field(default_factory=KokoroISTFTNetConfig)

    # --- architecture constants the upstream code hard-codes ---------------
    sample_rate: int = 24000
    decoder_hidden: int = 1024          # width of the decoder trunk
    asr_res_dim: int = 64               # channels of the ASR residual fed to every decode block
    source_harmonics: int = 8           # harmonics above F0 in the sine source
    source_sine_amp: float = 0.1
    source_noise_std: float = 0.003
    source_voiced_threshold: float = 10.0

    # --- checkpoint layout -------------------------------------------------
    weights_file: str = "kokoro-v1_0.pth"
    voices_dir: str = "voices"

    # --- serving defaults -------------------------------------------------
    default_voice: str = "af_heart"
    default_speed: float = 1.0
    min_speed: float = 0.25
    max_speed: float = 4.0
    # Sentences are packed into chunks of about this many phonemes. The first
    # chunk is kept short because its synthesis time is the time to first
    # audio; later chunks are larger to amortize per-step overhead.
    first_chunk_target_phonemes: int = 80
    chunk_target_phonemes: int = 200
    # Upper bound on chunks per request (the loop's ``max_iters``).
    max_chunks: int = 512

    # ---------------------------------------------------------------------
    @property
    def context_length(self) -> int:
        return self.plbert.max_position_embeddings

    @property
    def max_phonemes(self) -> int:
        """Phonemes per chunk, leaving room for the <bos>/<eos> pair."""
        return self.context_length - 2

    @property
    def samples_per_frame(self) -> int:
        """Waveform samples per predicted duration unit: the prosody
        predictor doubles the frame rate, the generator upsamples by 300."""
        return 2 * self.istftnet.upsample_factor

    @property
    def style_pack_rows(self) -> int:
        """A voice pack holds one style vector per phoneme count."""
        return self.max_phonemes

    @property
    def style_vector_dim(self) -> int:
        """A pack row is [decoder style, predictor style]."""
        return 2 * self.style_dim

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KokoroModelConfig:
        data = dict(data)
        plbert = KokoroBertConfig(**data.pop("plbert", {}))
        istftnet = KokoroISTFTNetConfig(**data.pop("istftnet", {}))
        cfg = _from_dict(cls, data)
        cfg.plbert = plbert
        cfg.istftnet = istftnet
        return cfg

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> KokoroModelConfig:
        with (Path(local_dir) / "config.json").open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
