"""Checkpoint-backed configuration for Qwen3-TTS 12 Hz CustomVoice.

Qwen publishes configuration across three files rather than one monolithic
object:

* ``config.json``: Talker architecture, special IDs, speakers, and languages
* ``generation_config.json``: main Talker and residual sampling defaults
* ``speech_tokenizer/config.json``: neural audio decoder architecture/rates

The dataclasses below preserve that ownership while exposing one
``Qwen3TTSModelConfig`` to the M* model and submodules.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    """Read optional checkpoint metadata, leaving dataclass defaults intact."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Qwen3TTSCodePredictorConfig:
    """Depth-wise transformer configuration for residual codec groups 1-15."""

    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 3072
    head_dim: int = 128
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 2048
    num_code_groups: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TTSCodePredictorConfig":
        return cls(**{
            name: data[name]
            for name in cls.__dataclass_fields__
            if name in data
        })


def _default_speaker_ids() -> dict[str, int]:
    return {
        "serena": 3066,
        "vivian": 3065,
        "uncle_fu": 3010,
        "ryan": 3061,
        "aiden": 2861,
        "ono_anna": 2873,
        "sohee": 2864,
        "eric": 2875,
        "dylan": 2878,
    }


def _default_speaker_dialects() -> dict[str, str | bool]:
    return {
        "serena": False,
        "vivian": False,
        "uncle_fu": False,
        "ryan": False,
        "aiden": False,
        "ono_anna": False,
        "sohee": False,
        "eric": "sichuan_dialect",
        "dylan": "beijing_dialect",
    }


def _default_language_ids() -> dict[str, int]:
    return {
        "chinese": 2055,
        "english": 2050,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
        "beijing_dialect": 2074,
        "sichuan_dialect": 2062,
    }


@dataclass
class Qwen3TTSTalkerConfig:
    """Autoregressive 12 Hz Talker architecture and conditioning IDs."""

    # Transformer geometry for the time-axis Talker.
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 3072
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 3072
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    num_code_groups: int = 16
    position_id_per_seconds: int = 13

    # Codec-side control tokens used to build the mixed prefill sequence.
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_language_id: dict[str, int] = field(default_factory=_default_language_ids)
    spk_id: dict[str, int] = field(default_factory=_default_speaker_ids)
    spk_is_dialect: dict[str, str | bool] = field(default_factory=_default_speaker_dialects)
    code_predictor: Qwen3TTSCodePredictorConfig = field(
        default_factory=Qwen3TTSCodePredictorConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TTSTalkerConfig":
        values = {
            name: data[name]
            for name in cls.__dataclass_fields__
            if name in data and name != "code_predictor"
        }
        values["code_predictor"] = Qwen3TTSCodePredictorConfig.from_dict(
            data.get("code_predictor_config", {})
        )
        return cls(**values)


@dataclass
class Qwen3TTSCodecConfig:
    """Official speech-tokenizer decoder plus M* streaming chunk controls."""

    # Values forwarded to Qwen3TTSTokenizerV2DecoderConfig.
    num_quantizers: int = 16
    codebook_size: int = 2048
    codebook_dim: int = 512
    latent_dim: int = 1024
    hidden_size: int = 512
    intermediate_size: int = 1024
    head_dim: int = 64
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    max_position_embeddings: int = 8000
    sliding_window: int = 72
    decoder_dim: int = 1536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: tuple[int, ...] = (2, 2)

    # Runtime/output metadata from the speech tokenizer's top-level config.
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920

    # M* stream policy: 300 new 12 Hz frames with 25 frames of overlap.
    chunk_frames: int = 300
    left_context_frames: int = 25

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TTSCodecConfig":
        decoder = data.get("decoder_config", {})
        values = {
            name: decoder[name]
            for name in cls.__dataclass_fields__
            if name in decoder
        }
        values.update({
            name: data[name]
            for name in (
                "input_sample_rate",
                "output_sample_rate",
                "decode_upsample_rate",
            )
            if name in data
        })
        return cls(**values)

    def decoder_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by the official 12 Hz decoder config."""
        excluded = {
            "input_sample_rate",
            "output_sample_rate",
            "decode_upsample_rate",
            "chunk_frames",
            "left_context_frames",
        }
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in excluded
        }


@dataclass
class Qwen3TTSGenerationConfig:
    """Sampling defaults for the two codec-generation levels.

    Unprefixed fields control codec group 0 through M*'s engine sampler.
    ``subtalker_*`` fields control residual groups 1-15 in CodePredictor.
    """

    min_new_tokens: int = 2
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    max_new_tokens: int = 8192

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TTSGenerationConfig":
        return cls(**{
            name: data[name]
            for name in cls.__dataclass_fields__
            if name in data
        })


@dataclass
class Qwen3TTSModelConfig:
    """Top-level model metadata composed from all checkpoint config files."""

    model_type: str = "qwen3_tts"
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"
    tts_model_size: str = "0b6"
    tts_model_type: str = "custom_voice"

    assistant_token_id: int = 77091
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673

    default_speaker: str = "vivian"
    default_language: str = "auto"
    talker: Qwen3TTSTalkerConfig = field(default_factory=Qwen3TTSTalkerConfig)
    codec: Qwen3TTSCodecConfig = field(default_factory=Qwen3TTSCodecConfig)
    generation: Qwen3TTSGenerationConfig = field(
        default_factory=Qwen3TTSGenerationConfig
    )

    @property
    def code_predictor(self) -> Qwen3TTSCodePredictorConfig:
        return self.talker.code_predictor

    @property
    def num_code_groups(self) -> int:
        return self.talker.num_code_groups

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3TTSModelConfig":
        """Compose Talker, Codec, and generation configs from a local snapshot."""
        root = Path(model_dir)
        model_data = _read_json(root / "config.json")
        generation_data = _read_json(root / "generation_config.json")
        codec_data = _read_json(root / "speech_tokenizer" / "config.json")

        values = {
            name: model_data[name]
            for name in (
                "model_type",
                "tokenizer_type",
                "tts_model_size",
                "tts_model_type",
                "assistant_token_id",
                "im_start_token_id",
                "im_end_token_id",
                "tts_pad_token_id",
                "tts_bos_token_id",
                "tts_eos_token_id",
            )
            if name in model_data
        }
        values.update(
            talker=Qwen3TTSTalkerConfig.from_dict(
                model_data.get("talker_config", {})
            ),
            codec=Qwen3TTSCodecConfig.from_dict(codec_data),
            generation=Qwen3TTSGenerationConfig.from_dict(generation_data),
        )
        return cls(**values)
