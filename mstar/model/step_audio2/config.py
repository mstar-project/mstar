"""Step-Audio-2-mini's checkpoint config, its token ids and its audio arithmetic.

The checkpoint's ``config.json`` holds a Qwen2.5-7B text decoder (``text_config``) and a
Whisper-large-v3-shaped audio encoder with a strided adapter (``audio_encoder_config``),
read by ``configuration_step_audio_2.py`` and built by ``modeling_step_audio_2.py``'s
``StepAudio2ForCausalLM``. Token ids are fixed by the tokenizer
(``tokenizer_config.json``'s ``added_tokens_decoder``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mstar.model.components.qwen3_lm import Qwen3LMConfig

# -- token ids ----------------------------------------------------------------
ENDOFTEXT = 151643  # <|endoftext|>: text_config.eos_token_id and the pad token
EOT = 151665  # <|EOT|>: ends a turn (the tokenizer's eos_token)
BOT = 151666  # <|BOT|>: starts a turn
AUDIO_START = 151688  # <audio_start>
AUDIO_END = 151689  # <audio_end>
AUDIO_PATCH = 151690  # <audio_patch>: one per audio embedding row
TTS_START = 151693  # <tts_start>: opens a spoken reply
TTS_END = 151694  # <tts_end>: a spoken reply ends here
TTS_PAD = 151695  # <tts_pad>: fills the text slot once a spoken reply's text has ended
SPEECH_BASE = 151696  # <audio_0>: speech id = SPEECH_BASE + code
NUM_SPEECH_CODES = 6561
# text ids are below the first multimodal special token (``stepaudio2.py:55``)
TEXT_VOCAB_END = AUDIO_START
# A reply ends at a turn's end, a spoken reply's end, or the config's eos. The reference
# ``StepAudio2`` stops on EOT alone, its tokenizer's end token (``stepaudio2.py:90-92``), and
# drops that last token (``outputs[0, prompt_ids.shape[-1]:-1]``, ``:54``). Its sampled
# spoken replies end on <tts_end> followed directly by EOT, so stopping on <tts_end> gives
# the same reply and saves the step that would sample the EOT.
STOP_TOKEN_IDS = (EOT, TTS_END, ENDOFTEXT)
# A spoken reply interleaves one text token with this many speech ids.
SPEECH_IDS_PER_TEXT = 4


def is_speech_id(token: int) -> bool:
    """Whether an LLM id is a speech code's (``<audio_0>`` .. ``<audio_6560>``). The
    tokenizer has ids past those too, which the reference drops before its vocoder
    (``examples.py``, ``x < 6561``)."""
    return SPEECH_BASE <= token < SPEECH_BASE + NUM_SPEECH_CODES


# -- resource keys --------------------------------------------------------------
LLM_KV = "llm_kv"
LLM_ATTN = "llm_attn"
LLM_POS = "llm_pos"
LLM_SAMPLER = "llm_sampler"
AUDIO_ATTN = "audio_attn"
# the audio tower's within-step array, which its key-prefix read (AUDIO_ATTN) attends
AUDIO_TOKENS = "audio_tokens"
T2W_STATE = "t2w_state"
# token2wav's attention caches, one resource per ``token2wav.CACHE_FAMILIES`` entry
T2W_KV = {"enc1": "t2w_enc1_kv", "enc2": "t2w_enc2_kv", "dit": "t2w_dit_kv"}

# -- walks ------------------------------------------------------------------------
PREFILL_TEXT = "prefill_text"
PREFILL_AUDIO = "prefill_audio"
PREFILL_WALKS = frozenset({PREFILL_TEXT, PREFILL_AUDIO})
# The LLM's decode loop and its walks: a text reply's, whose token loops back on the
# device, and a spoken reply's, which also streams its speech ids to the vocoder. Their
# rows share a step as DECODE_MIXED.
DECODE_LOOP = "decode_loop"
DECODE = "decode"
DECODE_SPOKEN = "decode_spoken"
DECODE_MIXED = "decode_mixed"
DECODE_WALKS = frozenset({DECODE, DECODE_SPOKEN, DECODE_MIXED})
# the vocoder's walk, in its own partition
T2W_CHUNK = "t2w_chunk"

# -- signals ------------------------------------------------------------------------
# the request's clips' log-mels, end to end along time
AUDIO_MELS = "audio_mels"
AUDIO_EMBEDS = "audio_embeds"
# the spoken walk's stream to the vocoder: the sampled ids that are speech
SPEECH_IDS = "speech_ids"

# -- step_metadata keys ------------------------------------------------------------
# each clip's mel frames, which the audio tower lays its segments out by (read off a
# routed tensor, they would wait on the device)
AUDIO_MEL_FRAMES = "audio_mel_frames"
# a spoken reply's token cap (``get_max_output_tokens``), a default rather than a length
# kwarg, so admission still fits it to the prompt
SPEECH_MAX_TOKENS = "speech_max_tokens"

# The reference's sampling (``examples.py``, ``examples-vllm-stream.py``).
DEFAULT_SAMPLING = dict(temperature=0.7, top_p=0.9, repetition_penalty=1.05)
# the sampler knobs a request may set
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "repetition_penalty", "penalize_prompt", "ignore_eos",
    "return_logprobs", "logprobs_mode",
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# -- audio input --------------------------------------------------------------
SAMPLE_RATE = 16_000
# audio is encoded in windows of 25 s (``stepaudio2.py:113``)
WINDOW_SAMPLES = SAMPLE_RATE * 25
N_FFT = 400
HOP_LENGTH = 160
# samples of zeros appended before the STFT (``utils.py:33``, ``padding=479``)
MEL_PADDING = 479


def mel_frames(num_samples: int) -> int:
    """Log-mel frames of a window of ``num_samples``: the centred STFT's frames over the
    padded window, less the last (``utils.py:43-47``)."""
    return (num_samples + MEL_PADDING) // HOP_LENGTH


def audio_tokens(num_mel_frames: int) -> int:
    """Embedding rows (``<audio_patch>`` tokens) a window of ``num_mel_frames`` becomes:
    ``compute_token_num`` (``utils.py:56-69``). The reference drops the 2 frames its
    padding added, then counts conv2's stride, the 2x pool and the adapter's stride."""
    encoder_frames = (num_mel_frames - 2 + 1) // 2 // 2
    return (encoder_frames - 1) // 2 + 1


@dataclass
class AudioEncoderConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    llm_dim: int = 3584
    kernel_size: int = 3
    adapter_stride: int = 2
    # the adapter's first linear, fixed in the reference (``modeling_step_audio_2.py:279``)
    adapter_hidden: int = 2048

    @property
    def head_dim(self) -> int:
        return self.n_audio_state // self.n_audio_head

    @property
    def ffn_dim(self) -> int:
        # ``ResidualAttentionBlock``'s 4x MLP (``modeling_step_audio_2.py:215``)
        return 4 * self.n_audio_state


@dataclass
class StepAudio2Config:
    llm: Qwen3LMConfig
    audio: AudioEncoderConfig
    stop_token_ids: tuple[int, ...] = STOP_TOKEN_IDS
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_hf(cls, local_dir: str | Path) -> StepAudio2Config:
        raw = json.loads((Path(local_dir) / "config.json").read_text())
        if raw.get("model_type") != "step_audio_2":
            raise ValueError(f"{local_dir} is a {raw.get('model_type')!r} checkpoint, not step_audio_2")
        text = raw["text_config"]
        if text.get("num_attention_groups", text["num_key_value_heads"]) != text["num_key_value_heads"]:
            raise ValueError("num_attention_groups must equal num_key_value_heads (configuration_step_audio_2.py:73)")
        if text.get("rope_scaling"):
            raise ValueError(f"rope_scaling {text['rope_scaling']!r} is not supported; the checkpoint has none")
        llm = Qwen3LMConfig(
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["hidden_size"] // text["num_attention_heads"],
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=text["rope_theta"],
            max_position_embeddings=text["max_position_embeddings"],
            # the reference builds an untied ``lm_head`` (``modeling_step_audio_2.py:330``)
            tie_word_embeddings=False,
            # Qwen2: q/k/v biases, no per-head QK norm
            attention_bias=True,
            qk_norm=False,
        )
        audio_raw = raw["audio_encoder_config"]
        audio = AudioEncoderConfig(**{
            k: audio_raw[k] for k in AudioEncoderConfig.__dataclass_fields__ if k in audio_raw
        })
        if audio.llm_dim != llm.hidden_size:
            raise ValueError(f"the adapter's llm_dim {audio.llm_dim} is not the LLM's width {llm.hidden_size}")
        return cls(llm=llm, audio=audio, raw=raw)
