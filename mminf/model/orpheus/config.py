from dataclasses import dataclass, field


@dataclass
class OrpheusModelConfig:
    # Llama 3.2 3B architecture
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    hidden_size: int = 3072
    head_dim: int = 128
    max_position_embeddings: int = 131072

    # Special token IDs
    start_token_id: int = 128259
    end_token_ids: list[int] = field(default_factory=lambda: [128009, 128260, 128261, 128257])
    stop_token_id: int = 128258
    pad_token_id: int = 128263

    # SNAC params
    snac_model_id: str = "hubertsiuzdak/snac_24khz"
    tokens_per_frame: int = 7
    sample_rate: int = 24000

    # Generation defaults
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 4096

    # Available voices
    available_voices: list[str] = field(default_factory=lambda: ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"])
