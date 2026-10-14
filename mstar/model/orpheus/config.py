from dataclasses import dataclass, field

from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy

# Constants
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
ROPE = "rope"

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
    # Two-frame chunks for the first ~1.9 s never wait on an LLM that is
    # only just faster than real time, while the listener's buffer is short;
    # four-frame chunks after that decode half as many frames per frame.
    # Deployment YAML ``model_kwargs`` override these.
    snac_chunk_schedule: tuple[int, ...] = (4,) + (2,) * 10
    snac_chunk_frames: int = 4
    snac_left_context_frames: int = 3
    snac_right_context_frames: int = 2

    # Generation defaults
    temperature: float = 0.6
    top_p: float = 0.8
    repetition_penalty: float = 1.3
    ignore_eos: bool = False  # benchmark parity: decode to max_tokens regardless of EOS
    max_new_tokens: int = 4096

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
        """Frames in each non-final window of a stream, one entry per distinct size."""
        context = self.snac_left_context_frames + self.snac_right_context_frames
        windows, delivered = [], 0
        for new in self.snac_chunk_schedule:
            windows.append(min(context, delivered) + new)
            delivered += new
        while True:
            window = min(context, delivered) + self.snac_chunk_frames
            if window in windows[len(self.snac_chunk_schedule):]:
                return sorted(set(windows))
            windows.append(window)
            delivered += self.snac_chunk_frames
