"""GLM-5.2 architecture + generation config.

Architecture values transcribed from the official checkpoint's config.json
(zai-org/GLM-5.2, architectures=["GlmMoeDsaForCausalLM"], model_type
"glm_moe_dsa", transformers 5.12.0). Fields the scaffold does not use yet
(DSA indexer, MTP) are kept here so the later steps read from one place.
"""

from dataclasses import dataclass, field

from mstar.model.glm52.quantization import Fp8BlockQuantConfig


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

    # Default absorbed MLA stores one compressed latent KV head (the kimi_k2_7
    # engine path). The naive path is the reduced-test parity fallback.
    mla_absorb: bool = True

    # --- DSA sparse-attention indexer (Phase C) ---
    # indexer_types in the checkpoint alternate "full" every
    # index_topk_freq=4 layers with "shared" in between (IndexShare);
    # components/indexer.py::is_full_indexer_layer holds the exact formula.
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    index_share_for_mtp_iteration: bool = True
    indexer_rope_interleave: bool = True
    # Engine half of DSA (opt-in; configs/glm52_tp8_longctx.yaml). Off: the
    # submodule guard holds every context to index_topk, where dense MLA IS
    # the exact DSA computation, and nothing about M1 serving changes. On:
    # the guard checks max_seq_len instead, FULL layers maintain a
    # per-request indexer k-store + compute selection, and decode beyond
    # index_topk runs sparse absorbed attention over the selected latents
    # (dsa.py / components/attention.py). v1 is decode-only beyond topk
    # (prefill prompts must still fit index_topk) and eager-only (selection
    # is host-side per-request work a captured graph would not replay).
    dsa_long_context: bool = False

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
    # Serving cap, consumed by get_kv_cache_config (the top-level YAML
    # max_seq_len key is only presence-checked by the conductor — it does
    # not override this). Held to index_topk while dsa_long_context is off:
    # dense MLA is exactly GLM-5.2's DSA computation only within the
    # top-2048 window, and the submodule's preprocess guard enforces the
    # same bound. With dsa_long_context on, Glm52Model raises this from the
    # ``max_seq_len`` model kwarg and the guard checks against it instead.
    max_seq_len: int = 2048

    # --- MTP (speculative decoding; unused by the scaffold) ---
    num_nextn_predict_layers: int = 1

    # --- tokens / generation defaults ---
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int = 154820
    # Kept under the ctx<=2048 exactness regime with room for the prompt;
    # Phase C restores the checkpoint's 8192 default alongside long context.
    max_output_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    # --- quantization / serving knobs ---
    # The official checkpoint is FP8 e4m3 with [128, 128] block scales
    # (`weight_scale_inv`). Populated from config.json by Glm52Model.
    quantization_config: Fp8BlockQuantConfig | None = None
    # Keep routed experts FP8-resident (uint8 container + block scales);
    # everything else dequantizes to bf16 on load. Disabling this dequantizes
    # experts too — fine for reduced tests, OOM on the real 753B checkpoint
    # (bf16 experts alone are ~181 GB/rank at TP8 vs the H200's 141 GB).
    moe_fp8_resident: bool = True

    prefill_token_buckets: list[int] | None = None
    prefill_capture_batch_sizes: list[int] | None = None

    # Derived MLA cache geometry: the paged cache stores one 576-dim latent
    # vector per token per layer (512 compressed KV + 64 decoupled-RoPE key).
    cache_latent_dim: int = field(init=False)

    def __post_init__(self):
        self.cache_latent_dim = self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def padded_head_dim(self) -> int:
        """Naive-MLA q/k/v pad target; FlashInfer paged kernels require 64/128/256.

        For the full model qk_head_dim is exactly 256, so padding is a no-op;
        the reduced configs (24 -> 64) exercise the pad path.
        """
        for supported in (64, 128, 256):
            if supported >= self.qk_head_dim:
                return supported
        raise ValueError(
            f"qk_head_dim={self.qk_head_dim} exceeds the largest FlashInfer SM90 "
            "head_dim (256); the naive-MLA pad mitigation cannot cover it."
        )

    @property
    def num_dense_layers(self) -> int:
        return min(self.first_k_dense_replace, self.num_hidden_layers)

    @classmethod
    def reduced(cls) -> "Glm52ModelConfig":
        """Tiny-dim variant for CPU tests: same shapes family, random weights."""
        return cls(
            mla_absorb=False,
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            q_lora_rank=48,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            # Indexer at test scale. offset=1 anchors the every-freq series
            # at layer 0, so the 2-layer model is layer 0 FULL / 1 SHARED.
            # index_topk stays serve-safe: the submodule's preprocess guard
            # refuses ctx > index_topk, and the GPU e2e serve tests run
            # ~16-token contexts. Truncation-regime unit tests override
            # cfg.index_topk locally.
            index_n_heads=4,
            index_head_dim=16,
            index_topk=64,
            index_topk_freq=4,
            index_skip_topk_offset=1,
            first_k_dense_replace=1,
            intermediate_size=256,
            moe_intermediate_size=64,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            max_seq_len=512,
            eos_token_ids=(250, 251, 252),
            pad_token_id=250,
            prefill_token_buckets=[64],
            prefill_capture_batch_sizes=[1],
        )

    @classmethod
    def reduced_fp8(cls, block: tuple[int, int] = (16, 16)) -> "Glm52ModelConfig":
        cfg = cls.reduced()
        cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=block)
        return cfg
