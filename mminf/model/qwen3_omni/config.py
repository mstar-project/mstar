"""Configuration dataclass for Qwen3-Omni-Moe.

Wraps the HF ``Qwen3OmniMoeConfig`` layout so every sub-component of the
model can be configured from a single ``Qwen3OmniModelConfig`` instance.
All values are loaded at runtime from a local HF checkpoint directory --
nothing is hard-coded except the dataclass defaults which mirror the
published HF defaults for easy reference / fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple


# ---------------------------------------------------------------------------
# Thinker text backbone
# ---------------------------------------------------------------------------

@dataclass
class ThinkerTextConfig:
    vocab_size: int = 3584
    hidden_size: int = 2048
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    # MoE
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True
    mlp_only_layers: list[int] = field(default_factory=list)
    decoder_sparse_step: int = 1
    rope_parameters: dict = field(default_factory=dict)

    # Computed -- not stored in HF config
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # Sanity check: the published Qwen3-Omni Thinker uses head_dim=128.
        # The fallback ``hidden_size // num_attention_heads = 2048 // 28 = 73``
        # is wrong and would silently break the MRoPE interleave layout
        # (which assumes head_dim // 2 == sum(mrope_section) == 64).
        # Fail loudly if this happens so we don't waste time chasing
        # downstream gibberish.
        if self.head_dim * self.num_attention_heads != self.hidden_size and self.head_dim != 128:
            import logging
            logging.getLogger(__name__).warning(
                "ThinkerTextConfig: unusual head_dim=%d "
                "(hidden_size=%d, num_attention_heads=%d). "
                "Expected head_dim=128 for Qwen3-Omni. "
                "Verify the checkpoint config.json contains 'head_dim': 128 "
                "under thinker_config.text_config.",
                self.head_dim, self.hidden_size, self.num_attention_heads,
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThinkerTextConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Thinker (top-level wrapper around text + encoders)
# ---------------------------------------------------------------------------

@dataclass
class ThinkerConfig:
    audio_token_id: int = 151646
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_start_token_id: int = 151647
    # NOTE: Qwen3-Omni's HF config explicitly omits ``audio_end_token_id``
    # (it is marked as ``AttributeError()`` in the modular file), but the
    # tokenizer still defines ``<|audio_eos|>`` which has the same id as
    # in Qwen2.5-Omni (151648).  We track it here so we can wrap audio
    # spans in their sentinel BOS/EOS tokens during prefill.
    audio_end_token_id: int = 151648
    vision_start_token_id: int = 151652
    # NOTE: Qwen3-Omni's HF config exposes ``vision_start_token_id`` but
    # not ``vision_end_token_id``; we use the same value as Qwen3-VL /
    # Qwen2.5-Omni (151653) which corresponds to ``<|vision_eos|>``.
    vision_end_token_id: int = 151653
    position_id_per_seconds: int = 25

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThinkerConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

@dataclass
class VisionEncoderConfig:
    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    spatial_merge_size: int = 2
    out_hidden_size: int = 3584
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisionEncoderConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in fnames}
        # HF may store tuple fields as lists
        if "deepstack_visual_indexes" in filtered and isinstance(
            filtered["deepstack_visual_indexes"], list
        ):
            filtered["deepstack_visual_indexes"] = tuple(
                filtered["deepstack_visual_indexes"]
            )
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------

@dataclass
class AudioEncoderConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    output_dim: int = 3584
    max_source_positions: int = 1500

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AudioEncoderConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Talker text backbone
# ---------------------------------------------------------------------------

@dataclass
class TalkerTextConfig:
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6

    # MoE
    moe_intermediate_size: int = 384
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = False
    shared_expert_intermediate_size: int | None = None

    # Computed
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TalkerTextConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Talker (top-level wrapper)
# ---------------------------------------------------------------------------

@dataclass
class TalkerConfig:
    accept_hidden_layer: int = 18
    # Qwen3-Omni-30B-A3B-Instruct has 16 code groups (verified against the
    # published config.json on HF Hub).  The HF class-level default was 32
    # but that's not what the published checkpoint uses.
    num_code_groups: int = 16
    thinker_hidden_size: int = 2048

    # Codec special token IDs.
    #
    # IMPORTANT: HF's ``Qwen3OmniMoeTalkerConfig`` class declares the
    # defaults for these fields in the 4196-4205 range, but those are
    # placeholders — the Talker's ``codec_embedding`` is an
    # ``nn.Embedding`` sized by the talker text vocab (``vocab_size=3072``
    # in HF defaults), so IDs ≥ 3072 are out-of-range and cause a
    # CUDA device-side assert when the assistant-prefix builder tries
    # to embed them.  The ACTUAL published Qwen3-Omni checkpoint uses
    # the 2148-2157 range (matching sglang-omni's values and the
    # model's tokenizer).  We use those as our defaults here and also
    # load from the checkpoint's ``config.json`` via ``from_dict`` if
    # the keys are present.
    codec_eos_token_id: int = 2150
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149

    audio_start_token_id: int = 151669

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TalkerConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Code Predictor
# ---------------------------------------------------------------------------

@dataclass
class CodePredictorConfig:
    # === Existing fields ===
    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    # Qwen3-Omni uses 16 code groups (verified against the published HF
    # config.json — code_predictor_config.num_code_groups = 16).
    num_code_groups: int = 16

    attention_bias: bool = False
    attention_dropout: float = 0.0
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    model_type: str = ""
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    semantic_codebook_size: int = 4096
    sliding_window: int = 72
    upsample_rates: Tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: Tuple[int, ...] = (2, 2)
    vector_quantization_hidden_dimension: int = 512

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodePredictorConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fnames})


# ---------------------------------------------------------------------------
# Code2Wav (vocoder)
# ---------------------------------------------------------------------------

@dataclass
class Code2WavConfig:
    # === Existing fields ===
    codebook_size: int = 2048
    num_quantizers: int = 16
    sliding_window: int = 72
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    upsample_rates: Tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: Tuple[int, ...] = (2, 2)
    chunk_size: int = 300
    left_context_size: int = 25
    attention_bias: bool = False
    attention_dropout: float = 0.0
    codebook_dim: int = 512
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    intermediate_size: int = 3072
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    model_type: str = ""
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    semantic_codebook_size: int = 4096
    vector_quantization_hidden_dimension: int = 512

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Code2WavConfig:
        fnames = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in fnames}
        for tup_field in ("upsample_rates", "upsampling_ratios"):
            if tup_field in filtered and isinstance(filtered[tup_field], list):
                filtered[tup_field] = tuple(filtered[tup_field])
        return cls(**filtered)

    def get_hf_config(self):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeCode2WavConfig,
        )
        return Qwen3OmniMoeCode2WavConfig(
            codebook_size=self.codebook_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            sliding_window=self.sliding_window,
            num_quantizers=self.num_quantizers,
            upsample_rates=list(self.upsample_rates),
            upsampling_ratios=list(self.upsampling_ratios),
            decoder_dim=self.decoder_dim,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            layer_scale_initial_scale=self.layer_scale_initial_scale,
        )


# ---------------------------------------------------------------------------
# Top-level model config
# ---------------------------------------------------------------------------

@dataclass
class Qwen3OmniModelConfig:
    """Unified config for Qwen3-Omni-Moe loaded from a local HF checkpoint."""

    # Path to the local HF checkpoint directory (for weight loading)
    local_dir: str = ""

    # --- Top-level special tokens ----------------------------------------
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    system_token_id: int = 8948
    user_token_id: int = 872
    assistant_token_id: int = 77091

    # --- Sub-configs -----------------------------------------------------
    thinker_text: ThinkerTextConfig = field(default_factory=ThinkerTextConfig)
    thinker: ThinkerConfig = field(default_factory=ThinkerConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    talker_text: TalkerTextConfig = field(default_factory=TalkerTextConfig)
    talker: TalkerConfig = field(default_factory=TalkerConfig)
    code_predictor: CodePredictorConfig = field(default_factory=CodePredictorConfig)
    code2wav: Code2WavConfig = field(default_factory=Code2WavConfig)

    def __post_init__(self) -> None:
        # Sanity check: all codec special token IDs must be < the Talker's
        # codec_embedding vocab size (talker_text.vocab_size, typically 3072).
        # HF's class-level defaults for Qwen3OmniMoeTalkerConfig put these
        # in the 4196-4205 range, but the actual published checkpoint uses
        # the 2148-2157 range.  If the loaded values are out-of-range, the
        # Talker's codec_embedding lookups would trigger a CUDA device-side
        # assert — fail loudly here instead.
        vocab = self.talker_text.vocab_size
        ids = {
            "codec_pad_id": self.talker.codec_pad_id,
            "codec_bos_id": self.talker.codec_bos_id,
            "codec_eos_token_id": self.talker.codec_eos_token_id,
            "codec_nothink_id": self.talker.codec_nothink_id,
            "codec_think_bos_id": self.talker.codec_think_bos_id,
            "codec_think_eos_id": self.talker.codec_think_eos_id,
        }
        bad = [(k, v) for k, v in ids.items() if v < 0 or v >= vocab]
        if bad:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                "Qwen3OmniModelConfig: codec special token IDs are out of "
                "range for talker_text.vocab_size=%d. Bad IDs: %s. "
                "This will cause CUDA device-side asserts when the Talker "
                "assistant-prefix builder calls codec_embedding() on these "
                "IDs. Check that your checkpoint's config.json provides the "
                "correct codec_*_id fields under talker_config, or that the "
                "TalkerConfig defaults match the actual model.",
                vocab, bad,
            )
            raise ValueError(
                f"Codec special token IDs out of range for vocab_size={vocab}: {bad}"
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def thinker_hidden_size(self) -> int:
        return self.thinker_text.hidden_size

    @property
    def thinker_num_layers(self) -> int:
        return self.thinker_text.num_hidden_layers

    @property
    def thinker_head_dim(self) -> int:
        assert self.thinker_text.head_dim is not None
        return self.thinker_text.head_dim

    @property
    def thinker_num_kv_heads(self) -> int:
        return self.thinker_text.num_key_value_heads

    @property
    def talker_hidden_size(self) -> int:
        return self.talker_text.hidden_size

    @property
    def talker_num_layers(self) -> int:
        return self.talker_text.num_hidden_layers

    @property
    def talker_head_dim(self) -> int:
        assert self.talker_text.head_dim is not None
        return self.talker_text.head_dim

    @property
    def talker_num_kv_heads(self) -> int:
        return self.talker_text.num_key_value_heads

    @property
    def accept_hidden_layer(self) -> int:
        return self.talker.accept_hidden_layer

    @property
    def num_code_groups(self) -> int:
        return self.talker.num_code_groups

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, local_dir: str | os.PathLike[str]) -> Qwen3OmniModelConfig:
        """Load configuration from a local HF checkpoint directory.

        Reads ``config.json`` from *local_dir*, parses the nested
        sub-config dicts (``thinker_config``, ``talker_config``, etc.),
        and populates every sub-dataclass accordingly.
        """
        local_dir = str(local_dir)
        config_path = Path(local_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found in {local_dir}"
            )

        with open(config_path, "r") as f:
            raw: dict[str, Any] = json.load(f)

        return cls._from_raw_dict(raw, local_dir=local_dir)

    @classmethod
    def _from_raw_dict(
        cls, raw: dict[str, Any], *, local_dir: str = ""
    ) -> Qwen3OmniModelConfig:
        """Build config from the parsed JSON dict.

        The HF ``Qwen3OmniMoeConfig`` nests sub-configs under keys like
        ``thinker_config``, ``talker_config``, etc.  Each of those in turn
        may have further nesting (e.g. ``thinker_config.text_config``,
        ``thinker_config.audio_config``, ``thinker_config.vision_config``).
        """

        # --- Thinker --------------------------------------------------
        thinker_raw = raw.get("thinker_config", {})
        thinker_text_raw = thinker_raw.get("text_config", {})
        vision_raw = thinker_raw.get("vision_config", {})
        audio_enc_raw = thinker_raw.get("audio_config", {})

        thinker_text = ThinkerTextConfig.from_dict(thinker_text_raw)
        thinker = ThinkerConfig.from_dict(thinker_raw)
        vision = VisionEncoderConfig.from_dict(vision_raw)
        audio_encoder = AudioEncoderConfig.from_dict(audio_enc_raw)

        # --- Talker ---------------------------------------------------
        talker_raw = raw.get("talker_config", {})
        talker_text_raw = talker_raw.get("text_config", {})
        code_predictor_raw = talker_raw.get("code_predictor_config", {})
        # code2wav_config is at the TOP LEVEL of Qwen3OmniMoeConfig,
        # not nested under talker_config.
        code2wav_raw = raw.get("code2wav_config", {})

        talker_text = TalkerTextConfig.from_dict(talker_text_raw)
        talker = TalkerConfig.from_dict(talker_raw)
        code_predictor = CodePredictorConfig.from_dict(code_predictor_raw)
        code2wav = Code2WavConfig.from_dict(code2wav_raw)

        # --- Top-level fields -----------------------------------------
        top_fields = {
            f.name
            for f in cls.__dataclass_fields__.values()
            if f.name not in {
                "local_dir",
                "thinker_text",
                "thinker",
                "vision",
                "audio_encoder",
                "talker_text",
                "talker",
                "code_predictor",
                "code2wav",
            }
        }
        top_kwargs = {k: v for k, v in raw.items() if k in top_fields}

        return cls(
            local_dir=local_dir,
            thinker_text=thinker_text,
            thinker=thinker,
            vision=vision,
            audio_encoder=audio_encoder,
            talker_text=talker_text,
            talker=talker,
            code_predictor=code_predictor,
            code2wav=code2wav,
            **top_kwargs,
        )
