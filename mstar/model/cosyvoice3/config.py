"""Configuration for Fun-CosyVoice3 (``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``).

The checkpoint ships a hyperpyyaml file (``cosyvoice3.yaml``) that builds the
reference classes directly; there is no ``config.json``. The dataclasses below
write its values down once. Every shape they declare is checked against the
checkpoint when the weights load.

Sources (read, not copied): ``cosyvoice3.yaml`` from the HF snapshot and
``cosyvoice/{llm/llm.py,flow/flow.py,flow/flow_matching.py,flow/DiT/*,
hifigan/generator.py,hifigan/f0_predictor.py,cli/model.py,cli/frontend.py}``
from FunAudioLLM/CosyVoice.

The reference-audio front end is Chatterbox's: the 25 Hz speech tokenizer v3 is
Chatterbox's ``S3Tokenizer`` with twelve blocks instead of six, the speaker
encoder is the same CAMPPlus, and the prompt mel is the same 24 kHz Matcha mel
with ``fmax`` left at Nyquist. Their configs are Chatterbox's dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mstar.model.chatterbox.config import (
    S3GenCFMConfig,
    S3GenHiFTConfig,
    S3GenMelConfig,
    S3GenXVectorConfig,
    S3TokenizerConfig,
)

# Resource keys (LM node)
LM_KV = "lm_kv"
LM_ATTN = "lm_attn"
LM_POS = "lm_pos"
LM_LABEL = "main"

# Node names
SPEAKER_ENCODER_NODE = "speaker_encoder"
LM_NODE = "llm"
FLOW_NODE = "flow"
HIFT_NODE = "hift"

# Sample rates and token geometry fixed by the checkpoint
TOKENIZER_SR = 16_000  # speech tokenizer and CAMPPlus input rate
SAMPLE_RATE = 24_000  # prompt mel and output waveform rate
TOKEN_RATE = 25  # speech tokens per second
SPEECH_VOCAB = 6561  # 3**8 FSQ codes; ids >= this are control tokens
SAMPLES_PER_FRAME = 480  # one 50 Hz mel frame of 24 kHz audio

# Instruction prefix the LM was trained with. Every prompt must contain
# ``<|endofprompt|>``; the reference asserts on token 151646.
SYSTEM_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
END_OF_PROMPT_ID = 151646


@dataclass
class LMConfig:
    """CosyVoice3LM: a Qwen2 0.5B backbone (``CosyVoice-BlankEN``) with its own
    speech-token embedding and head over ``speech_token_size + 200`` ids."""

    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    text_vocab_size: int = 151936
    speech_token_size: int = SPEECH_VOCAB
    num_control_tokens: int = 200

    @property
    def speech_vocab_size(self) -> int:
        return self.speech_token_size + self.num_control_tokens

    @property
    def sos_token(self) -> int:
        return self.speech_token_size

    @property
    def eos_token(self) -> int:
        return self.speech_token_size + 1

    @property
    def task_id_token(self) -> int:
        return self.speech_token_size + 2


@dataclass
class SamplingDefaults:
    """Repetition-aware sampling (``cosyvoice.utils.common.ras_sampling``): a
    nucleus draw (top-p over at most top-k ids); a draw that already appears
    ``>= ras_window * ras_tau`` times in the last ``ras_window`` tokens is
    masked and redrawn from the full distribution."""

    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 25
    ras_window: int = 10
    ras_tau: float = 0.1
    # speech tokens per text token, bounding the decode
    min_token_text_ratio: float = 2.0
    max_token_text_ratio: float = 20.0


@dataclass
class DiTConfig:
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    dim_head: int = 64
    ff_mult: int = 2
    mel_dim: int = 80
    mu_dim: int = 80
    spk_dim: int = 80
    conv_pos_kernel: int = 31
    conv_pos_groups: int = 16
    time_freq_dim: int = 256
    # streaming attention: block-causal over chunks of this many mel frames,
    # counted from the start of the prompt
    static_chunk_size: int = 50


@dataclass
class FlowConfig:
    """CausalMaskedDiffWithDiT: tokens -> ``mu`` (embedding, look-ahead convs,
    x2 repeat) -> mel by a classifier-free-guided Euler solve of the DiT."""

    vocab_size: int = SPEECH_VOCAB
    input_size: int = 80
    output_size: int = 80
    spk_embed_dim: int = 192
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3
    pre_lookahead_channels: int = 1024
    dit: DiTConfig = field(default_factory=DiTConfig)
    cfm: S3GenCFMConfig = field(default_factory=S3GenCFMConfig)
    # The reference fixes the flow noise: ``set_all_random_seed(0)`` then one
    # ``randn(1, 80, 50 * 300)`` field, indexed by frame from the prompt's start.
    noise_seed: int = 0
    noise_frames: int = 50 * 300

    @property
    def chunk_tokens(self) -> int:
        return self.dit.static_chunk_size // self.token_mel_ratio


@dataclass
class CausalHiFTConfig(S3GenHiFTConfig):
    """``CausalHiFTGenerator``: Chatterbox's HiFT hyper-parameters, with causal
    convolutions, a causal F0 predictor (three frames of look-ahead), a
    pre-convolution that looks four frames ahead and fixed excitation noise."""

    source_resblock_kernel_sizes: tuple[int, ...] = (7, 7, 11)
    conv_pre_look_right: int = 4
    f0_look_right: int = 3
    # The reference draws a ``rand(1, 300 s * 24 kHz, 9)`` noise field once and
    # indexes it by output sample; here it is drawn from this seed.
    noise_seed: int = 1986
    noise_seconds: float = 300.0

    @property
    def stream_holdback_frames(self) -> int:
        """Mel frames at the end of a non-final call whose audio is not final yet:
        the F0 predictor's and the pre-convolution's look-ahead, plus the frame
        the inverse STFT cannot finish."""
        return self.f0_look_right + self.conv_pre_look_right + 1


@dataclass
class CosyVoice3Config:
    lm: LMConfig = field(default_factory=LMConfig)
    sampling: SamplingDefaults = field(default_factory=SamplingDefaults)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: CausalHiFTConfig = field(default_factory=CausalHiFTConfig)
    tokenizer: S3TokenizerConfig = field(default_factory=lambda: S3TokenizerConfig(n_layer=12))
    xvector: S3GenXVectorConfig = field(default_factory=S3GenXVectorConfig)
    mel: S3GenMelConfig = field(default_factory=lambda: S3GenMelConfig(fmax=None))

    # Checkpoint files inside the HF snapshot
    llm_weights: str = "llm.pt"
    flow_weights: str = "flow.pt"
    hift_weights: str = "hift.pt"
    speech_tokenizer_onnx: str = "speech_tokenizer_v3.onnx"
    campplus_onnx: str = "campplus.onnx"
    text_tokenizer_dir: str = "CosyVoice-BlankEN"

    # Silent and breath FSQ tokens; a run of more than ``max_silent_tokens`` of
    # them is cut before the flow decoder (reference ``CosyVoice3Model``).
    silent_tokens: tuple[int, ...] = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
    max_silent_tokens: int = 5

    # Serving limits
    max_text_tokens: int = 512  # prompt text + text
    max_reference_seconds: float = 30.0  # the tokenizer's single-pass limit
    max_speech_tokens: int = 2048  # ~82 s of audio per request
    voice_cache_size: int = 64
    max_new_tokens_limit: int | None = None

    # Streaming. The flow decoder re-runs over the prompt and every settled
    # token on each hop and emits whole 25-token attention chunks, so the
    # streamed mel equals a whole-utterance decode in streaming mode. Hops
    # follow the reference: 25 tokens (plus whatever aligns the prompt to a
    # chunk), then x2 up to 100. ``stream_hop_tokens = 0`` decodes once, at the
    # end, with full attention (the reference's ``stream=False``).
    stream_hop_tokens: int = 25
    stream_hop_growth: float = 2.0
    stream_max_hop_tokens: int = 100
    # How often the LM hands tokens to the flow node (tokens per stream chunk).
    # A hop settles 25 tokens or more, so polling more often than every 25
    # wakes the flow node (and its token read-back) mostly to settle nothing.
    stream_poll_tokens: int = 25
    # Mel frames of left context the streaming vocoder re-runs before the new
    # frames (0 = the whole history, the reference's O(N^2) re-vocode).
    hift_context_frames: int = 64
    # Flow estimator precision ("bfloat16", "float32", "float16"); the Euler
    # state stays float32. "float32" is the reference's precision.
    flow_estimator_dtype: str = "bfloat16"
    flow_max_batch_size: int = 8
    hift_max_batch_size: int = 8
