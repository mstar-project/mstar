"""Checkpoint-backed configuration for the Qwen3-TTS 12 Hz family.

One dataclass tree serves every published 12 Hz checkpoint: the 0.6B and
1.7B CustomVoice models (built-in speakers, optional style instruction on
1.7B), VoiceDesign (voice described by an instruction) and Base (voice
cloned from reference audio through an ECAPA-TDNN speaker encoder). Which
paths a checkpoint supports is read from ``config.json``, never hard-coded.

Qwen publishes configuration across three files rather than one monolithic
object:

* ``config.json``: Talker architecture, special IDs, speakers, languages and
  (Base only) the speaker encoder
* ``generation_config.json``: main Talker and residual sampling defaults
* ``speech_tokenizer/config.json``: neural audio decoder architecture/rates

The dataclasses below preserve that ownership while exposing one
``Qwen3TTSModelConfig`` to the M* model and submodules.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resource label constants (Talker node)
# ---------------------------------------------------------------------------
TALKER_KV = "talker_kv"
TALKER_ATTN = "talker_attn"
TALKER_POS = "talker_pos"
TALKER_SAMPLER = "talker_sampler"
CODE_PRED_SAMPLER = "code_predictor"

# ---------------------------------------------------------------------------
# Fixed ChatML wrapper of the assistant turn, as the Qwen2 tokenizer emits it:
# ``<|im_start|>assistant\n`` ... ``<|im_end|>\n<|im_start|>assistant\n``.
# The reference slices the text span as ``input_id[:, 3:-5]``; the API side
# uses these to size the span and the Talker verifies them before prefill.
# ---------------------------------------------------------------------------
CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (151644, 77091, 198)
CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (151645, 198, 151644, 77091, 198)


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
class Qwen3TTSSpeakerEncoderConfig:
    """ECAPA-TDNN speaker encoder shipped with the Base checkpoint.

    Defaults mirror ``Qwen3TTSSpeakerEncoderConfig`` in the reference
    implementation; ``config.json`` only pins ``enc_dim`` (the Talker hidden
    size the x-vector is added into) and the sample rate of the reference
    audio.
    """

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    enc_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    # Mel front end used by the reference ``extract_speaker_embedding``.
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 12000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TTSSpeakerEncoderConfig":
        values = {
            name: data[name]
            for name in cls.__dataclass_fields__
            if name in data
        }
        for name in ("enc_channels", "enc_kernel_sizes", "enc_dilations"):
            if name in values:
                values[name] = tuple(values[name])
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
    encode_downsample_rate: int = 1920
    encoder_valid_num_quantizers: int = 16
    # Raw Mimi encoder configuration (``encoder_config`` in the speech
    # tokenizer's config.json); it is handed verbatim to the encoder that
    # turns reference audio into codec frames for voice cloning.
    encoder_config: dict[str, Any] = field(default_factory=dict)

    # M* stream policy: the codec pops a ramp of small chunks first (the first
    # frame alone, so first audio leaves one Talker step after prefill; the
    # decoder is causal, so a frame's audio does not depend on how it was
    # chunked), then ``chunk_frames`` new frames per call, each preceded by up
    # to ``left_context_frames`` already decoded frames so the causal decoder
    # warms up (the reference's own ``chunked_decode`` uses 25 frames of left
    # context).
    chunk_schedule: tuple[int, ...] = (1, 3, 8, 16)
    chunk_frames: int = 25
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
                "encode_downsample_rate",
                "encoder_valid_num_quantizers",
                "encoder_config",
            )
            if name in data
        })
        return cls(**values)

    def codec_windows(self) -> list[int]:
        """Distinct window sizes (context + new frames) the chunk schedule produces.

        These are the shapes the codec captures CUDA graphs for; a terminal
        flush shorter than a window is padded up to the next one.
        """
        windows = set()
        delivered = 0
        for size in (*self.chunk_schedule, self.chunk_frames):
            windows.add(min(self.left_context_frames, delivered) + size)
            delivered += size
        return sorted(windows)

    def frames_for_samples(self, num_samples: int) -> int:
        """Codec frames the encoder emits for ``num_samples`` of input audio."""
        return -(-int(num_samples) // self.encode_downsample_rate)

    def decoder_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by the official 12 Hz decoder config."""
        excluded = {
            "input_sample_rate",
            "output_sample_rate",
            "decode_upsample_rate",
            "encode_downsample_rate",
            "encoder_valid_num_quantizers",
            "encoder_config",
            "chunk_schedule",
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

    default_language: str = "auto"
    talker: Qwen3TTSTalkerConfig = field(default_factory=Qwen3TTSTalkerConfig)
    codec: Qwen3TTSCodecConfig = field(default_factory=Qwen3TTSCodecConfig)
    generation: Qwen3TTSGenerationConfig = field(
        default_factory=Qwen3TTSGenerationConfig
    )
    # Present only on Base checkpoints (``speaker_encoder_config`` in
    # config.json); CustomVoice and VoiceDesign carry no speaker encoder.
    speaker_encoder: Qwen3TTSSpeakerEncoderConfig | None = None

    SUPPORTED_MODEL_TYPES = ("custom_voice", "voice_design", "base")

    def __post_init__(self) -> None:
        if self.tts_model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported Qwen3-TTS tts_model_type {self.tts_model_type!r}; "
                f"supported: {', '.join(self.SUPPORTED_MODEL_TYPES)}"
            )
        if self.tts_model_type == "base" and self.speaker_encoder is None:
            self.speaker_encoder = Qwen3TTSSpeakerEncoderConfig(
                enc_dim=self.talker.hidden_size
            )

    @property
    def code_predictor(self) -> Qwen3TTSCodePredictorConfig:
        return self.talker.code_predictor

    @property
    def num_code_groups(self) -> int:
        return self.talker.num_code_groups

    # -- Variant capabilities (all derived from the checkpoint metadata) ----

    @property
    def is_custom_voice(self) -> bool:
        return self.tts_model_type == "custom_voice"

    @property
    def is_voice_design(self) -> bool:
        return self.tts_model_type == "voice_design"

    @property
    def is_base(self) -> bool:
        return self.tts_model_type == "base"

    @property
    def has_builtin_speakers(self) -> bool:
        """CustomVoice ships named speakers; VoiceDesign and Base do not."""
        return bool(self.talker.spk_id)

    @property
    def default_speaker(self) -> str | None:
        """Speaker used when a request names none (``None`` = no speaker tag)."""
        if not self.has_builtin_speakers:
            return None
        return "vivian" if "vivian" in self.talker.spk_id else sorted(self.talker.spk_id)[0]

    @property
    def supports_instruct(self) -> bool:
        """Instruction text (style or voice description) in the prefill.

        VoiceDesign is driven by it; the 1.7B CustomVoice accepts it for
        style/emotion control. The reference silently drops instructions for
        the 0.6B CustomVoice, which was not trained with them, so M* rejects
        them there instead of ignoring the request field.
        """
        if self.is_voice_design:
            return True
        return self.is_custom_voice and self.tts_model_size != "0b6"

    @property
    def requires_instruct(self) -> bool:
        return self.is_voice_design

    @property
    def supports_reference_audio(self) -> bool:
        return self.is_base

    @property
    def default_non_streaming_mode(self) -> bool:
        """Reference/vLLM-Omni/SGLang-Omni default text layout per variant.

        CustomVoice and VoiceDesign place the whole text in the prefill;
        Base (voice clone) feeds text one token per generated frame.
        """
        return not self.is_base

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
        if "speaker_encoder_config" in model_data:
            values["speaker_encoder"] = Qwen3TTSSpeakerEncoderConfig.from_dict(
                model_data["speaker_encoder_config"]
            )
        return cls(**values)
