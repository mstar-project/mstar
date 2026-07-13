"""Config for Higgs-Audio STT (bosonai/higgs-audio-v3-stt).

Higgs-Audio STT = Whisper-style audio encoder ("audio_tower") + MLP
projector ("audio_encoder_proj") + dense Qwen3 text LLM. Values are read
from the checkpoint's ``config.json`` (top-level audio fields +
``text_config``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
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

    # audio_<X> fields sourced from audio_encoder_config[<X>].
    _ENCODER_FIELDS = {
        "audio_d_model": "d_model",
        "audio_encoder_layers": "encoder_layers",
        "audio_encoder_attention_heads": "encoder_attention_heads",
        "audio_encoder_ffn_dim": "encoder_ffn_dim",
        "audio_num_mel_bins": "num_mel_bins",
        "audio_max_source_positions": "max_source_positions",
    }
    # Fields sourced from the top level of config.json by the same name.
    _TOPLEVEL_FIELDS = (
        "projector_temporal_downsample", "audio_in_token_idx",
        "audio_bos_token_id", "audio_eos_token_id", "chunk_size_seconds",
    )

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "HiggsAudioModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)
        tc = hf["text_config"]
        enc = hf["audio_encoder_config"]
        field_names = {f.name for f in fields(cls)}

        values: dict = {}
        # 1. text LLM: fields whose name matches a text_config key.
        values.update({k: tc[k] for k in field_names if k in tc})
        values["head_dim"] = tc.get(
            "head_dim", tc["hidden_size"] // tc["num_attention_heads"]
        )
        # 2. audio encoder: audio_<X> <- audio_encoder_config[<X>].
        values.update({f: enc[k] for f, k in cls._ENCODER_FIELDS.items() if k in enc})
        # 3. top-level config.json (+ the pad_token_id -> eos_token_id rename).
        values.update({k: hf[k] for k in cls._TOPLEVEL_FIELDS if k in hf})
        if "pad_token_id" in hf:
            values["eos_token_id"] = hf["pad_token_id"]

        return cls(**values)
