"""Config for Higgs-Audio STT (bosonai/higgs-audio-v3-stt).

Higgs-Audio STT = Whisper-style audio encoder ("audio_tower") + MLP
projector ("audio_encoder_proj") + dense Qwen3 text LLM. Values are read
from the checkpoint's ``config.json`` (top-level audio fields +
``text_config``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HiggsAudioModelConfig:
    # Qwen3 text LLM (higgs-audio-v3-stt defaults: Qwen3-1.7B)
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768

    # audio encoder / projector
    audio_d_model: int = 1280
    audio_encoder_layers: int = 32
    audio_encoder_attention_heads: int = 20
    audio_encoder_ffn_dim: int = 5120
    audio_num_mel_bins: int = 128
    audio_max_source_positions: int = 1500
    projector_temporal_downsample: int = 2

    # special tokens
    audio_in_token_idx: int = 151672   # <|AUDIO|>
    audio_bos_token_id: int = 151669   # <|audio_bos|>
    audio_eos_token_id: int = 151670   # <|audio_eos|>
    im_end_token_id: int = 151645      # <|im_end|>
    eos_token_id: int = 151643         # <|endoftext|>

    # audio front-end
    chunk_size_seconds: float = 4.0
    sampling_rate: int = 16000

    # transcription prompt used by the reference transcribe.py
    default_prompt: str = (
        "Transcribe the speech. Output only the spoken words in lowercase "
        "with no punctuation."
    )

    stop_token_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.stop_token_ids:
            self.stop_token_ids = frozenset({self.eos_token_id, self.im_end_token_id})

    @property
    def chunk_size_samples(self) -> int:
        return int(self.chunk_size_seconds * self.sampling_rate)

    def encoder_output_length(self, mel_len: int) -> int:
        """Valid audio-embedding count for ``mel_len`` mel frames:
        conv2 (stride 2) -> avg_pool (stride 2) -> projector conv (stride 2)."""
        after_conv = (mel_len - 1) // 2 + 1
        after_pool = (after_conv - 2) // 2 + 1
        return (after_pool - 1) // self.projector_temporal_downsample + 1

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "HiggsAudioModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)
        tc = hf["text_config"]
        enc = hf["audio_encoder_config"]

        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            rope_theta=tc.get("rope_theta", 1_000_000.0),
            max_position_embeddings=tc.get("max_position_embeddings", 32768),
            audio_d_model=enc["d_model"],
            audio_encoder_layers=enc["encoder_layers"],
            audio_encoder_attention_heads=enc["encoder_attention_heads"],
            audio_encoder_ffn_dim=enc["encoder_ffn_dim"],
            audio_num_mel_bins=enc["num_mel_bins"],
            audio_max_source_positions=enc["max_source_positions"],
            projector_temporal_downsample=hf.get("projector_temporal_downsample", 2),
            audio_in_token_idx=hf["audio_in_token_idx"],
            audio_bos_token_id=hf.get("audio_bos_token_id", 151669),
            audio_eos_token_id=hf.get("audio_eos_token_id", 151670),
            eos_token_id=hf.get("pad_token_id", 151643),
            chunk_size_seconds=hf.get("chunk_size_seconds", 4.0),
        )
