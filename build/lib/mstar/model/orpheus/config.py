from dataclasses import dataclass, field


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

    # Streaming chunking params (for async partition mode)
    snac_window_tokens: int = 28      # 4 frames * 7 tokens/frame
    snac_stride_tokens: int = 7       # 1 frame advance per chunk
    snac_audio_slice_start: int = 2048  # middle region start in decoded audio
    snac_audio_slice_end: int = 4096    # middle region end in decoded audio

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
