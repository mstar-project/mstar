"""GLM-5.2 architecture + generation config.

Architecture values transcribed from the official checkpoint's config.json
(zai-org/GLM-5.2, architectures=["GlmMoeDsaForCausalLM"], model_type
"glm_moe_dsa", transformers 5.12.0). Fields the scaffold does not use yet
(DSA indexer, MTP) are kept here so the later steps read from one place.
"""

from dataclasses import dataclass, field


@dataclass
class Glm52ModelConfig:
    # --- backbone ---
    vocab_size: int = 154880
    hidden_size: int = 6144
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    # --- MLA (multi-head latent attention) ---
    # Latent-compressed KV: per token the cache stores kv_lora_rank
    # compressed dims + qk_rope_head_dim decoupled-RoPE dims, NOT
    # num_kv_heads x head_dim. Heads recover their K/V by up-projection.
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    # --- DSA sparse-attention indexer (unused by the scaffold) ---
    # indexer_types in the checkpoint alternate "full" every
    # index_topk_freq=4 layers with "shared" in between (IndexShare).
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    index_share_for_mtp_iteration: bool = True
    indexer_rope_interleave: bool = True

    # --- MoE ---
    first_k_dense_replace: int = 3  # layers 0..2 are dense
    intermediate_size: int = 12288  # dense-layer MLP
    moe_intermediate_size: int = 2048  # per expert
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_router_dtype: str = "float32"

    # --- RoPE / lengths ---
    rope_theta: float = 8_000_000.0
    rope_interleave: bool = True
    max_position_embeddings: int = 1_048_576
    # Dev-time serving cap; the checkpoint supports 1M positions but page
    # tables and CUDA-graph capture should not be sized for that by default.
    # Override per run via the YAML's max_seq_len.
    max_seq_len: int = 32768

    # --- MTP (speculative decoding; unused by the scaffold) ---
    num_nextn_predict_layers: int = 1

    # --- tokens / generation defaults ---
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int = 154820
    max_output_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    # Derived MLA cache geometry: the paged cache stores one 576-dim latent
    # vector per token per layer (512 compressed KV + 64 decoupled-RoPE key).
    cache_latent_dim: int = field(init=False)

    def __post_init__(self):
        self.cache_latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
