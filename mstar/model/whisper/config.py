"""Config for Whisper encoder-decoder ASR models (large-v3, large-v3-turbo, ...).

Values are read from the HF checkpoint's ``config.json``,
``generation_config.json`` and ``preprocessor_config.json`` so one class
serves every Whisper size: turbo differs from large-v3 only in
``decoder_layers`` (4 vs 32) and its alignment heads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path

# ---------------------------------------------------------------------------
# Resource label constants (decoder node)
# ---------------------------------------------------------------------------
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
# Positions: Whisper has no RoPE. The resource is declared for its
# per-(request, label) position counter, which drives the learned absolute
# ``embed_positions`` lookup; the attention layers never call it.
POS = "positions"

# The encoder context lives in its own KV cache so the self-attention
# resource, which plans a wrapper per label of the cache it names, never sees
# it. Written once at prefill, read (zero-span) by every later step.
CROSS_KV_CACHE = "cross_kv_cache"
CROSS_ATTN = "cross_attn"
CONTEXT_LABEL = "main"

# Graph node names, shared by the model, the submodules and the configs.
ENCODER_NODE = "audio_encoder"
DECODER_NODE = "decoder"

# Graph walks. ``prefill`` runs the encoder and the whole forced prompt;
# ``detect_language`` runs the encoder and ``<|startoftranscript|>`` alone,
# sampling only among language tokens; ``prefill_prompt`` then appends the
# rest of the forced prompt to that detected token; ``decode`` is the loop.
PREFILL_WALK = "prefill"
DETECT_LANGUAGE_WALK = "detect_language"
PREFILL_PROMPT_WALK = "prefill_prompt"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"


@dataclass
class WhisperModelConfig:
    # transformer dims (large-v3 defaults)
    d_model: int = 1280
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    vocab_size: int = 51866
    max_target_positions: int = 448
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    scale_embedding: bool = False

    # special tokens
    decoder_start_token_id: int = 50258
    eos_token_id: int = 50257
    no_timestamps_token_id: int = 50364
    prev_sot_token_id: int = 50362
    # the first timestamp token may not exceed this index (HF/openai parity)
    max_initial_timestamp_index: int = 50

    # generation_config maps: "<|en|>" -> 50259, "transcribe" -> 50360
    lang_to_id: dict[str, int] = field(default_factory=dict)
    task_to_id: dict[str, int] = field(default_factory=dict)

    # Logit suppression (HF generate parity): tokens never sampled, and
    # tokens additionally blocked for the first generated token.
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)
    # (layer, head) pairs of the cross-attention heads whose weights align
    # text to audio; used for word-level timestamps (DTW).
    alignment_heads: list[list[int]] = field(default_factory=list)

    # log-mel front end (preprocessor_config.json)
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    chunk_length: int = 30

    @property
    def head_dim(self) -> int:
        return self.d_model // self.decoder_attention_heads

    @property
    def encoder_head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def n_samples(self) -> int:
        """Samples in one encoder window (30 s at 16 kHz)."""
        return self.chunk_length * self.sampling_rate

    @property
    def num_frames(self) -> int:
        """Mel frames in one encoder window (3000); the encoder's conv2 halves
        them to ``max_source_positions``."""
        return self.n_samples // self.hop_length

    @property
    def timestamp_begin(self) -> int:
        """``<|0.00|>``; token ``timestamp_begin + i`` means ``0.02 * i`` s."""
        return self.no_timestamps_token_id + 1

    @property
    def timestamp_precision(self) -> float:
        return 0.02

    @property
    def max_prev_tokens(self) -> int:
        """How much ``<|startofprev|>`` context fits before the prompt proper
        (openai-whisper / HF: half the context window, minus the marker)."""
        return self.max_target_positions // 2 - 1

    @cached_property
    def language_token_ids(self) -> list[int]:
        return sorted(self.lang_to_id.values())

    @cached_property
    def _id_to_lang(self) -> dict[int, str]:
        return {tok: name.strip("<|>") for name, tok in self.lang_to_id.items()}

    def language_token(self, language: str) -> int:
        lang_token = f"<|{language}|>"
        if lang_token not in self.lang_to_id:
            raise ValueError(
                f"Unknown Whisper language {language!r}; "
                f"available: {sorted(t.strip('<|>') for t in self.lang_to_id)}"
            )
        return self.lang_to_id[lang_token]

    def language_of(self, token_id: int) -> str | None:
        return self._id_to_lang.get(token_id)

    def task_token(self, task: str) -> int:
        if task not in self.task_to_id:
            raise ValueError(
                f"Unknown Whisper task {task!r}; available: {list(self.task_to_id)}"
            )
        return self.task_to_id[task]

    def is_timestamp(self, token_id: int) -> bool:
        return token_id >= self.timestamp_begin

    def timestamp_seconds(self, token_id: int) -> float:
        return (token_id - self.timestamp_begin) * self.timestamp_precision

    # generation_config.json keys (not in config.json); the rest of the
    # dataclass fields map 1:1 to config.json keys by name.
    _GEN_KEYS = (
        "lang_to_id", "task_to_id", "no_timestamps_token_id", "prev_sot_token_id",
        "max_initial_timestamp_index", "suppress_tokens", "begin_suppress_tokens",
        "alignment_heads",
    )
    # preprocessor_config.json keys
    _FEATURE_KEYS = ("sampling_rate", "hop_length", "n_fft", "chunk_length")

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "WhisperModelConfig":
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            hf = json.load(f)

        gen: dict = {}
        gen_path = local_dir / "generation_config.json"
        if gen_path.exists():
            with open(gen_path) as f:
                gen = json.load(f)
        feat: dict = {}
        feat_path = local_dir / "preprocessor_config.json"
        if feat_path.exists():
            with open(feat_path) as f:
                feat = json.load(f)

        # Every field is named to match its source key; pull from
        # generation_config.json for the _GEN_KEYS, preprocessor_config.json
        # for the _FEATURE_KEYS, else config.json. Fields absent from every
        # source keep their dataclass default.
        def source(name: str) -> dict:
            if name in cls._GEN_KEYS:
                return gen
            if name in cls._FEATURE_KEYS:
                return feat
            return hf

        names = {f.name for f in fields(cls) if not f.name.startswith("_")}
        values = {name: source(name)[name] for name in names if name in source(name)}
        return cls(**values)

    def decoder_prompt_ids(
        self,
        language: str | None = "en",
        task: str = "transcribe",
        timestamps: bool = False,
        prev_tokens: list[int] | None = None,
    ) -> list[int]:
        """The forced decoder prompt.

        ``[<|startofprev|> prev...] <|startoftranscript|> <|lang|> <|task|>
        [<|notimestamps|>]``. With ``language=None`` the prompt stops after
        ``<|startoftranscript|>``: the language token is *sampled* next
        (``detect_language`` walk) and :meth:`prompt_tail_ids` supplies the
        rest. ``prev_tokens`` is the previous window's transcript for
        long-form carry-over, kept to the last :attr:`max_prev_tokens`.
        """
        ids: list[int] = []
        if prev_tokens:
            ids.append(self.prev_sot_token_id)
            ids.extend(prev_tokens[-self.max_prev_tokens:])
        ids.append(self.decoder_start_token_id)
        if language is None:
            return ids
        ids.append(self.language_token(language))
        ids.extend(self.prompt_tail_ids(task=task, timestamps=timestamps))
        return ids

    def prompt_tail_ids(self, task: str = "transcribe", timestamps: bool = False) -> list[int]:
        """What follows the language token: ``<|task|> [<|notimestamps|>]``."""
        ids = [self.task_token(task)]
        if not timestamps:
            ids.append(self.no_timestamps_token_id)
        return ids
