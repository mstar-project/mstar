from dataclasses import dataclass, field

from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy

# Constants
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
ROPE = "rope"

# SNAC captures one CUDA graph per window size at every capture batch size,
# and every size from 1 frame up to the widest window occurs (final flushes),
# so the widest window (left + right context + largest chunk) bounds startup
# and graph memory.
SNAC_MAX_WINDOW_FRAMES = 16

@dataclass
class OrpheusModelConfig:
    # Llama 3.2 3B architecture
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    hidden_size: int = 3072
    head_dim: int = 128
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    vocab_size: int = 156940
    tie_word_embeddings: bool = True
    rope_scaling: dict = field(default_factory=lambda: {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    })

    # Special token IDs
    start_token_id: int = 128259
    end_token_ids: list[int] = field(default_factory=lambda: [128009, 128260, 128261, 128257])
    stop_token_id: int = 128258
    pad_token_id: int = 128263
    custom_token_base_id: int = 128256  # vocab_id of <custom_token_0>

    # SNAC params
    snac_model_id: str = "hubertsiuzdak/snac_24khz"
    tokens_per_frame: int = 7
    sample_rate: int = 24000

    snac_hop_samples: int = 2048      # audio samples per 7-token frame

    # Streaming SNAC decode (async partition mode), in 7-token frames. The
    # decoder is not causal, so a frame's audio depends on frames on both
    # sides: each chunk decodes its frames with up to ``left_context`` earlier
    # and ``right_context`` later ones and emits only its own. The last
    # ``right_context`` frames of a chunk are emitted by the next one, and the
    # final flush decodes the real tail of the stream. Chunk k takes
    # ``snac_chunk_schedule[k]`` new frames (``snac_chunk_frames`` after it),
    # so the first audio leaves after ``snac_chunk_schedule[0]`` frames.
    # The first audio is 2 frames (the first chunk's 4 less its right
    # context); three 2-frame chunks then keep ahead of the listener until the
    # buffer covers a 4-frame chunk, at 32 streams too.
    # Deployment YAML ``model_kwargs`` override these; left + right context
    # plus the largest chunk may be at most ``SNAC_MAX_WINDOW_FRAMES``.
    snac_chunk_schedule: tuple[int, ...] = (4, 2, 2, 2)
    snac_chunk_frames: int = 4
    snac_left_context_frames: int = 3
    snac_right_context_frames: int = 2

    # Sampling defaults; the checkpoint's generation_config.json overrides them
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.3
    penalize_prompt: bool = True
    ignore_eos: bool = False  # benchmark parity: decode to max_tokens regardless of EOS

    # Available voices
    available_voices: list[str] = field(
        default_factory=lambda: ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
    )
    default_voice: str = "tara"

    def __post_init__(self):
        if self.default_voice not in self.available_voices:
            raise ValueError(
                f"default_voice {self.default_voice!r} is not in available_voices"
            )
        if self.snac_chunk_frames <= 0 or self.snac_left_context_frames < 0 \
                or self.snac_right_context_frames < 0:
            raise ValueError("snac_chunk_frames must be positive and SNAC context frames non-negative")
        self.snac_chunk_schedule = tuple(int(n) for n in self.snac_chunk_schedule)
        if any(n <= 0 for n in self.snac_chunk_schedule):
            raise ValueError("snac_chunk_schedule sizes must be positive")
        if (self.snac_chunk_schedule or (self.snac_chunk_frames,))[0] <= self.snac_right_context_frames:
            raise ValueError(
                "the first SNAC chunk must exceed snac_right_context_frames, "
                "or it emits no audio"
            )
        widest = self.snac_left_context_frames + self.snac_right_context_frames \
            + max((*self.snac_chunk_schedule, self.snac_chunk_frames))
        if widest > SNAC_MAX_WINDOW_FRAMES:
            raise ValueError(
                f"SNAC left + right context plus the largest chunk is {widest} frames; "
                f"at most {SNAC_MAX_WINDOW_FRAMES} are supported (one accelerator graph per window size)"
            )

    def snac_chunk_policy(self) -> ScheduledLeftContextChunkPolicy:
        """The LLM -> SNAC edge's chunking, in tokens.

        A chunk's leading ``context_items`` are its left context plus the
        previous chunk's right context, which this chunk emits.
        """
        tpf = self.tokens_per_frame
        return ScheduledLeftContextChunkPolicy(
            schedule=tuple(n * tpf for n in self.snac_chunk_schedule),
            chunk=self.snac_chunk_frames * tpf,
            left_context=(self.snac_left_context_frames + self.snac_right_context_frames) * tpf,
        )

    def snac_window_frames(self) -> list[int]:
        """Every window size, in frames, a stream can hand the decoder.

        The schedule's windows and every final flush: the policy runs over
        each stream length until the windows repeat. The windows do not depend
        on when tokens arrive, only on how many there are. A window shorter
        than a frame (a stream shorter than a frame, or a final flush holding
        only the EOS token) decodes one silent frame, as in ``prepare_inputs``.
        """
        tpf = self.tokens_per_frame
        steady = self.snac_left_context_frames + self.snac_right_context_frames + self.snac_chunk_frames
        sizes = {1}
        for frames in range(1, sum(self.snac_chunk_schedule) + 2 * steady + 1):
            policy, left = self.snac_chunk_policy(), frames * tpf + 1  # the EOS token streams last
            while policy.is_ready(left):
                sizes.add(policy.window_size() // tpf)
                stride = policy.next_chunk_size(left)
                policy.register_chunk(stride)
                left -= stride
            sizes.add(max(left // tpf, 1))
        return sorted(sizes)
