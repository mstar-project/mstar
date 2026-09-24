"""Configuration for Chatterbox and Chatterbox-Turbo.

Resemble publishes no ``config.json`` for either checkpoint: every hyper
parameter lives in the reference package's Python sources (``T3Config``,
``LLAMA_CONFIGS``, the S3Gen constructor arguments) and, for Turbo, in a
stale training yaml. The dataclasses below are that specification, written
down once, with the two shipped variants as constructors. Every shape they
declare is checked against the checkpoint when the weights are loaded.

Sources (read, not copied): ``chatterbox/models/t3/modules/t3_config.py``,
``chatterbox/models/t3/llama_configs.py``, ``chatterbox/tts.py``,
``chatterbox/tts_turbo.py``, ``chatterbox/models/s3gen/s3gen.py``,
``chatterbox/models/voice_encoder/config.py``,
``chatterbox/models/s3tokenizer/s3tokenizer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Resource keys and KV labels (T3 node)
# ---------------------------------------------------------------------------
T3_KV = "t3_kv"
T3_ATTN = "t3_attn"
T3_POS = "t3_pos"
T3_SAMPLER = "t3_sampler"

# The conditional stream, the unconditional (text-less) stream classifier-free
# guidance reads beside it, and the plan label the two are packed under when a
# step runs both through one attention call (label-major: all ``main`` rows,
# then all ``uncond`` rows).
COND_LABEL = "main"
UNCOND_LABEL = "uncond"
CFG_LABEL = "cfg"

# Node names
VOICE_ENCODER_NODE = "voice_encoder"
T3_NODE = "T3"
S3GEN_NODE = "s3gen"

# Sample rates fixed by the checkpoints
S3_SR = 16_000        # voice encoder, S3 tokenizer and x-vector input rate
S3GEN_SR = 24_000     # S3Gen reference mel and output waveform rate
S3_TOKEN_RATE = 25    # S3 speech tokens per second
S3_SPEECH_VOCAB = 6561  # 3**8 FSQ codes; ids >= this are control tokens
S3GEN_SILENCE_TOKEN = 4299


# ---------------------------------------------------------------------------
# T3: text -> speech-token transformer
# ---------------------------------------------------------------------------


@dataclass
class T3BackboneConfig:
    """The causal transformer behind T3.

    ``kind`` selects the block: ``"llama"`` (RMSNorm, SwiGLU, RoPE through the
    position resource) for Chatterbox, ``"gpt2"`` (LayerNorm, GELU-tanh MLP,
    learned absolute positions looked up on the planned position ids) for
    Turbo.
    """

    kind: str = "llama"
    hidden_size: int = 1024
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    intermediate_size: int = 4096
    norm_eps: float = 1e-5
    max_position_embeddings: int = 131072
    # RoPE (llama only)
    rope_theta: float = 500000.0
    rope_scaling: dict = field(default_factory=lambda: {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    })

    @property
    def is_gpt2(self) -> bool:
        return self.kind == "gpt2"

    @classmethod
    def llama_520m(cls) -> "T3BackboneConfig":
        return cls()

    @classmethod
    def gpt2_medium(cls) -> "T3BackboneConfig":
        return cls(
            kind="gpt2",
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=64,
            intermediate_size=4096,
            norm_eps=1e-5,
            max_position_embeddings=8196,
            rope_theta=0.0,
            rope_scaling={},
        )


@dataclass
class T3Config:
    backbone: T3BackboneConfig = field(default_factory=T3BackboneConfig.llama_520m)

    text_vocab_size: int = 704
    speech_vocab_size: int = 8194
    start_text_token: int = 255
    stop_text_token: int = 0
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    max_text_tokens: int = 2048
    max_speech_tokens: int = 4096

    # Learned absolute position tables added to the text and speech embeddings
    # (index within the text segment / within the speech segment). Turbo's
    # GPT-2 backbone carries its own table instead.
    learned_pos_emb: bool = True

    # Conditioning
    speaker_embed_size: int = 256
    speech_cond_prompt_len: int = 150
    use_perceiver_resampler: bool = True
    perceiver_num_queries: int = 32
    perceiver_num_heads: int = 4
    emotion_adv: bool = True
    speech_head_bias: bool = False

    # The reference ``T3.inference`` appends a second start-of-speech
    # embedding (both at speech position 0) to the prefill; ``inference_turbo``
    # does not. Kept as a flag so parity is exact and the quirk is named.
    duplicate_bos_in_prefill: bool = True

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    @property
    def cond_len(self) -> int:
        """Conditioning tokens in front of the text: speaker, prompt (resampled
        or raw), emotion."""
        prompt = (
            self.perceiver_num_queries
            if self.use_perceiver_resampler
            else self.speech_cond_prompt_len
        )
        return 1 + prompt + (1 if self.emotion_adv else 0)

    @property
    def text_pos_table_size(self) -> int:
        return self.max_text_tokens + 2

    @property
    def speech_pos_table_size(self) -> int:
        return self.max_speech_tokens + 2 + 2

    @classmethod
    def english(cls) -> "T3Config":
        return cls()

    @classmethod
    def multilingual(cls) -> "T3Config":
        """The 23-language checkpoint: the English model with a 2454-token
        grapheme text vocabulary (reference ``T3Config.multilingual``)."""
        return cls(text_vocab_size=2454)

    @classmethod
    def turbo(cls) -> "T3Config":
        return cls(
            backbone=T3BackboneConfig.gpt2_medium(),
            text_vocab_size=50276,
            speech_vocab_size=6563,
            learned_pos_emb=False,
            speech_cond_prompt_len=375,
            use_perceiver_resampler=False,
            emotion_adv=False,
            speech_head_bias=True,
            duplicate_bos_in_prefill=False,
        )


# ---------------------------------------------------------------------------
# Voice encoder (speaker embedding), S3 tokenizer, S3Gen
# ---------------------------------------------------------------------------


@dataclass
class VoiceEncoderConfig:
    num_mels: int = 40
    sample_rate: int = S3_SR
    hidden_size: int = 256
    num_layers: int = 3
    speaker_embed_size: int = 256
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
    fmin: float = 0.0
    fmax: float = 8000.0
    mel_power: float = 2.0
    partial_frames: int = 160
    # Partial utterances per second (Resemble's default) and the coverage a
    # trailing partial needs to count.
    partials_rate: float = 1.3
    min_coverage: float = 0.8
    final_relu: bool = True


@dataclass
class S3TokenizerConfig:
    """S3TokenizerV2 (25 Hz): Whisper-style log-mel front end, FSMN attention
    encoder, finite-scalar quantizer."""

    n_mels: int = 128
    n_fft: int = 400
    hop_size: int = 160
    sample_rate: int = S3_SR
    n_state: int = 1280
    n_head: int = 20
    n_layer: int = 6
    fsmn_kernel_size: int = 31
    fsq_levels: int = 3
    fsq_dim: int = 8
    token_rate: int = S3_TOKEN_RATE
    max_frames_per_pass: int = 3000  # 30 s; longer audio is not a prompt

    @property
    def codebook_size(self) -> int:
        return self.fsq_levels ** self.fsq_dim


@dataclass
class S3GenMelConfig:
    n_fft: int = 1920
    num_mels: int = 80
    sample_rate: int = S3GEN_SR
    hop_size: int = 480
    win_size: int = 1920
    fmin: float = 0.0
    fmax: float = 8000.0


@dataclass
class S3GenEncoderConfig:
    """UpsampleConformerEncoder: 6 blocks at 25 Hz, x2 upsample, 4 blocks at 50 Hz."""

    input_size: int = 512
    output_size: int = 512
    attention_heads: int = 8
    linear_units: int = 2048
    num_blocks: int = 6
    num_up_blocks: int = 4
    pre_lookahead_len: int = 3
    up_stride: int = 2
    layer_norm_eps: float = 1e-12


@dataclass
class S3GenEstimatorConfig:
    """ConditionalDecoder: causal 1-D UNet with transformer blocks."""

    in_channels: int = 320  # 80 (x) + 80 (mu) + 80 (spk) + 80 (cond)
    out_channels: int = 80
    channels: tuple[int, ...] = (256,)
    attention_head_dim: int = 64
    n_blocks: int = 4
    num_mid_blocks: int = 12
    num_heads: int = 8
    causal: bool = True


@dataclass
class S3GenCFMConfig:
    sigma_min: float = 1e-6
    t_scheduler: str = "cosine"
    inference_cfg_rate: float = 0.7
    n_timesteps: int = 10


@dataclass
class S3GenHiFTConfig:
    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = S3GEN_SR
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    upsample_rates: tuple[int, ...] = (8, 5, 3)
    upsample_kernel_sizes: tuple[int, ...] = (16, 11, 7)
    istft_n_fft: int = 16
    istft_hop_len: int = 4
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    source_resblock_kernel_sizes: tuple[int, ...] = (7, 7, 11)
    source_resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99
    f0_cond_channels: int = 512

    @property
    def upsample_factor(self) -> int:
        n = self.istft_hop_len
        for r in self.upsample_rates:
            n *= r
        return n


@dataclass
class S3GenXVectorConfig:
    """CAMPPlus speaker encoder (80-bin Kaldi fbank, 16 kHz)."""

    feat_dim: int = 80
    embedding_size: int = 192
    growth_rate: int = 32
    bn_size: int = 4
    init_channels: int = 128
    block_num_layers: tuple[int, ...] = (12, 24, 16)
    block_kernel_sizes: tuple[int, ...] = (3, 3, 3)
    block_dilations: tuple[int, ...] = (1, 2, 2)


@dataclass
class S3GenConfig:
    """S3 speech tokens -> mel (flow matching) -> waveform (HiFT)."""

    sample_rate: int = S3GEN_SR
    vocab_size: int = S3_SPEECH_VOCAB
    token_mel_ratio: int = 2
    output_size: int = 80
    spk_embed_dim: int = 192
    mel: S3GenMelConfig = field(default_factory=S3GenMelConfig)
    encoder: S3GenEncoderConfig = field(default_factory=S3GenEncoderConfig)
    estimator: S3GenEstimatorConfig = field(default_factory=S3GenEstimatorConfig)
    cfm: S3GenCFMConfig = field(default_factory=S3GenCFMConfig)
    hift: S3GenHiFTConfig = field(default_factory=S3GenHiFTConfig)
    xvector: S3GenXVectorConfig = field(default_factory=S3GenXVectorConfig)
    # Turbo's distilled decoder: mean-flow estimator (t and r embeddings mixed),
    # two Euler steps, no guidance.
    meanflow: bool = False
    # Zero the first 20 ms and fade the next 20 ms in, to hide reference
    # spill-over (reference ``S3Token2Wav.trim_fade``).
    trim_fade_frames: int = S3GEN_SR // 50

    @classmethod
    def standard(cls) -> "S3GenConfig":
        return cls()

    @classmethod
    def meanflow_distilled(cls) -> "S3GenConfig":
        return cls(meanflow=True, cfm=S3GenCFMConfig(n_timesteps=2))


# ---------------------------------------------------------------------------
# Generation defaults and the whole-model config
# ---------------------------------------------------------------------------


@dataclass
class GenerationDefaults:
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.05
    repetition_penalty: float = 1.2
    cfg_weight: float = 0.5
    exaggeration: float = 0.5
    max_new_tokens: int = 1000
    n_cfm_timesteps: int = 10
    watermark: bool = True


@dataclass
class ChatterboxConfig:
    variant: str = "chatterbox"  # "chatterbox" | "multilingual" | "turbo"
    t3: T3Config = field(default_factory=T3Config.english)
    voice_encoder: VoiceEncoderConfig = field(default_factory=VoiceEncoderConfig)
    s3_tokenizer: S3TokenizerConfig = field(default_factory=S3TokenizerConfig)
    s3gen: S3GenConfig = field(default_factory=S3GenConfig.standard)
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)

    # Checkpoint file names inside the HF snapshot
    t3_weights: str = "t3_cfg.safetensors"
    s3gen_weights: str = "s3gen.safetensors"
    voice_encoder_weights: str = "ve.safetensors"
    text_tokenizer_file: str = "tokenizer.json"
    builtin_voice_file: str = "conds.pt"
    # Multilingual only: the Cangjie table for Chinese and the language the
    # text is read in when a request gives none (``language_id``).
    cangjie_file: str = ""
    default_language: str | None = None

    # Reference-audio windows: T3's speech prompt is tokenized from the first
    # ``enc_cond_seconds``; S3Gen embeds the first ``dec_cond_seconds``.
    enc_cond_seconds: float = 6.0
    dec_cond_seconds: float = 10.0
    # Turbo normalises the reference loudness before conditioning.
    normalize_reference_loudness: bool = False
    reference_target_lufs: float = -27.0
    # Silence tokens appended to Turbo's speech tokens before S3Gen.
    trailing_silence_tokens: int = 0

    # Serving limits
    max_text_tokens: int = 512
    voice_cache_size: int = 64

    # S3Gen streaming: the first waveform chunk is synthesised once
    # ``stream_first_chunk_tokens`` speech tokens exist (about 0.23 s on an
    # H100); later chunks start at ``stream_chunk_tokens`` and grow by
    # ``stream_chunk_growth`` up to ``stream_max_chunk_tokens`` (15, 50, 100,
    # 200, 200 ...): each chunk buys the playback time to produce a bigger one,
    # so an utterance costs three or four flow solves instead of one per 25
    # tokens (+40 % throughput at concurrency 8-32, playback still continuous
    # at concurrency 1). ``stream_chunk_tokens = 0`` = one chunk per utterance,
    # the reference's offline behaviour. Each chunk re-runs the flow decoder
    # over the prompt and the recent tokens with a fixed per-request noise
    # field and withholds the encoder's look-ahead tokens; the vocoder keeps
    # ``stream_mel_cache_frames`` mel frames of context and crossfades the
    # re-synthesised tail (CosyVoice 2's token2wav scheme).
    stream_first_chunk_tokens: int = 15
    stream_chunk_tokens: int = 50
    stream_chunk_growth: float = 2.0
    stream_max_chunk_tokens: int = 200
    stream_mel_cache_frames: int = 8
    # How many already-decoded tokens a chunk's flow solve keeps as left
    # context (plus the reference prompt). 0 = the whole history, as CosyVoice
    # 2 streams (cost grows with every chunk); the window bounds the work per
    # chunk and stays as close to the whole-utterance decode as the full
    # history does (log-mel correlation 0.988 vs 0.985 on CPU).
    stream_context_tokens: int = 25
    # torch.compile the flow-matching estimator instead of the CUDA graphs
    # below (the older, slower alternative: +14 % at concurrency 8, nothing at
    # 32; asking for it turns ``s3gen_graphs`` off). Dynamic shapes, so one
    # compile covers every chunk length; costs a few minutes at startup.
    s3gen_compile: bool = False
    # ``torch.compile`` mode for the estimator: "default" fuses kernels
    # (measured +14 % at concurrency 8, nothing at 32); "reduce-overhead"
    # would replay CUDA graphs but trips an Inductor assertion with dynamic
    # shapes on torch 2.11, so it is not usable yet.
    s3gen_compile_mode: str = "default"
    # Pad every flow solve to a multiple of this many mel frames (0 = exact
    # length). Padding is masked, so outputs stay the same up to float noise;
    # it bounds the number of distinct shapes the graph-captured (or compiled)
    # estimator sees (graphs need a bucket; 0 becomes 64 with them on).
    s3gen_frame_bucket: int = 64
    # Replay the S3Gen stages from CUDA graphs, one per shape: the estimator's
    # hundreds of tiny kernels per Euler step make an eager solve launch-bound
    # (a 10-step solve of one row went from 170 ms to 45 ms on an H100, the
    # streamed first chunk from 0.22 s to 0.12 s, throughput at concurrency
    # 8/32 up 1.8x/1.6x, WER unchanged). Rows are padded to 1/2/4/8, frames to
    # the bucket; the shapes of the built-in voice are captured at startup,
    # others on first use. Off = the eager reference path.
    s3gen_graphs: bool = True
    # Which S3Gen stages the graphs cover: the flow solve, the token encoder
    # (rows and tokens padded to the bucket) and the vocoder (exact lengths).
    s3gen_graph_stages: str = "solve,encoder,vocoder"
    # Precision of the flow estimator ("float32", "bfloat16", "float16"); the
    # Euler state and update stay in float32. float16 runs the estimator 2.5x
    # faster than float32 under a graph and stays within 0.02-0.15 of the
    # float32 log-mel (bfloat16: 0.1-1.8, same speed); float32 is the
    # reference path, bit-exact with the package at a fixed seed.
    s3gen_estimator_dtype: str = "float16"
    # Capture T3 prefill as packed CUDA graphs (token buckets x small batch
    # sizes, both guidance modes); off = eager prefill, decode still captured.
    t3_prefill_graphs: bool = True
    # Requests whose flow solves share one padded batch (the scheduler groups
    # up to this many); with graphs the captured row counts are the powers of
    # two up to it. 16 rows solve about 1.5x more efficiently per row than 8
    # under a graph, at the cost of a longer wait for the batch to form.
    s3gen_max_batch_size: int = 8

    @property
    def sample_rate(self) -> int:
        return self.s3gen.sample_rate

    @property
    def is_turbo(self) -> bool:
        return self.variant == "turbo"

    @property
    def is_multilingual(self) -> bool:
        return self.variant == "multilingual"

    @classmethod
    def chatterbox(cls) -> "ChatterboxConfig":
        return cls()

    @classmethod
    def multilingual(cls) -> "ChatterboxConfig":
        """Chatterbox Multilingual (reference ``mtl_tts.py``): the same S3Gen,
        voice encoder, built-in voice and sampling as the English model; the
        T3 weights ``t3_mtl23ls_v2.safetensors`` and the grapheme tokenizer
        with a language token."""
        return cls(
            variant="multilingual",
            t3=T3Config.multilingual(),
            t3_weights="t3_mtl23ls_v2.safetensors",
            text_tokenizer_file="grapheme_mtl_merged_expanded_v1.json",
            cangjie_file="Cangjie5_TC.json",
            default_language="en",
        )

    @classmethod
    def turbo(cls) -> "ChatterboxConfig":
        return cls(
            variant="turbo",
            t3=T3Config.turbo(),
            s3gen=S3GenConfig.meanflow_distilled(),
            generation=GenerationDefaults(
                temperature=0.8,
                top_p=0.95,
                top_k=1000,
                min_p=0.0,
                repetition_penalty=1.2,
                cfg_weight=0.0,
                exaggeration=0.0,
                max_new_tokens=1000,
                n_cfm_timesteps=2,
            ),
            t3_weights="t3_turbo_v1.safetensors",
            s3gen_weights="s3gen_meanflow.safetensors",
            text_tokenizer_file="",  # HF GPT-2 tokenizer files in the snapshot root
            enc_cond_seconds=15.0,
            normalize_reference_loudness=True,
            trailing_silence_tokens=3,
        )

    @classmethod
    def from_variant(cls, variant: str) -> "ChatterboxConfig":
        if variant in ("chatterbox", "english", "default"):
            return cls.chatterbox()
        if variant in ("multilingual", "chatterbox_multilingual", "chatterbox-multilingual", "mtl"):
            return cls.multilingual()
        if variant in ("turbo", "chatterbox_turbo", "chatterbox-turbo"):
            return cls.turbo()
        raise ValueError(f"Unknown Chatterbox variant {variant!r}")

    @classmethod
    def from_model_path(cls, model_path_hf: str) -> "ChatterboxConfig":
        """Pick the variant from the repo id (``ResembleAI/chatterbox-turbo``)."""
        name = model_path_hf.rstrip("/").split("/")[-1].lower()
        return cls.turbo() if "turbo" in name else cls.chatterbox()
