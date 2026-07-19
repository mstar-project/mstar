"""Configuration dataclass for Kimi-K2.7 (text backbone).

Kimi-K2.7's text architecture *is* DeepSeek-V3 — vLLM serves it as
``DeepseekV3ForCausalLM`` (``model_type: "kimi_k2"`` maps to
``DeepseekV3Config``). This dataclass therefore carries the full DeepSeek-V3
field set: MLA latent dims, fine-grained sigmoid-routed MoE grouping, and
``deepseek_yarn`` RoPE. Only a handful of these fields are read by the M0
scaffold (``num_hidden_layers``, the head dims, ``vocab_size``,
``max_position_embeddings``); the rest are declared now so this stays the single
source of truth for the later milestones (MoE router, MLA attention, weights).

The full-size defaults below are the real values from the
``moonshotai/Kimi-K2.7-Code`` HF ``config.json`` (``model_type: "kimi_k2"`` →
``DeepseekV3Config``), confirmed field by field against the published checkpoint
config (the weights themselves are not present in this workspace). M0 does not
depend on the full-size values being exact — the modular tests build from
:meth:`KimiK2Config.reduced`, a tiny self-consistent config.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KimiK2Config:
    # -- Core transformer dims --------------------------------------------
    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 18432  # dense-FFN size (first_k_dense_replace layers)
    num_hidden_layers: int = 61
    num_attention_heads: int = 64
    num_key_value_heads: int = 64  # MLA has no separate KV heads; kept for HF parity
    rms_norm_eps: float = 1e-5  # from config.json (Kimi uses 1e-5, not DeepSeek-V3's 1e-6)
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    # -- MLA (Multi-head Latent Attention) latent dims --------------------
    # Query is compressed to ``q_lora_rank`` then projected up to
    # ``num_attention_heads * qk_head_dim``; K/V share a ``kv_lora_rank`` latent
    # plus a decoupled ``qk_rope_head_dim`` RoPE slice. Per-head query/key dim is
    # ``qk_nope_head_dim + qk_rope_head_dim``; value head dim differs.
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # -- Fine-grained MoE (sigmoid router, group-limited top-k, noaux_tc) --
    n_routed_experts: int = 384          # from config.json
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8         # top-k
    moe_intermediate_size: int = 2048
    n_group: int = 1                     # from config.json
    topk_group: int = 1                  # from config.json (groups kept by group-limited routing)
    routed_scaling_factor: float = 2.827  # from config.json
    scoring_func: str = "sigmoid"        # DeepSeek-V3/Kimi: sigmoid (not softmax)
    topk_method: str = "noaux_tc"        # per-expert e_score_correction_bias
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1       # first N layers are dense, rest are MoE
    moe_layer_freq: int = 1

    # -- deepseek_yarn RoPE ------------------------------------------------
    rope_theta: float = 50000.0          # from config.json
    rope_scaling: dict = field(default_factory=lambda: {
        # from config.json (HF key is "type": "yarn"; mstar's internal id for the
        # DeepSeek/Kimi variant is "deepseek_yarn"). factor=64 yields the 262144
        # context (4096 * 64). K2.7-Code keeps beta_fast=32 (some other Kimi
        # checkpoints set beta_fast=1). mscale == mscale_all_dim == 1.0.
        "rope_type": "deepseek_yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    })

    # -- Special tokens / generation defaults -----------------------------
    bos_token_id: int = 163584           # from config.json
    eos_token_id: int = 163586           # from config.json
    pad_token_id: int = 163839           # from config.json
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False

    # -- MTP (multi-token prediction) — deferred, declared for completeness -
    num_nextn_predict_layers: int = 0

    # ---------------------------------------------------------------------
    # Derived dims (read by get_kv_cache_config / attention)
    # ---------------------------------------------------------------------
    @property
    def qk_head_dim(self) -> int:
        """Per-head query/key dim: nope + decoupled-rope slice (e.g. 128+64=192)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def padded_head_dim(self) -> int:
        """Head dim the naive-MLA q/k/v are zero-padded to for the paged cache.

        FlashInfer's SM90 (Hopper) prefill kernel ``static_assert``s
        ``head_dim_vo ∈ {64, 128, 256}`` (M4 finding), so it will not JIT-build for
        the real ``qk_head_dim=192`` or the reduced ``qk_head_dim=24``. The
        correctness-first mitigation (M6) pads q/k (from ``qk_head_dim``) and v
        (from ``v_head_dim``) up to the smallest supported dim ``>= qk_head_dim``,
        runs the paged attention there, and slices the output back to
        ``v_head_dim`` — compensating the softmax scale (see
        ``KimiMLAAttention.softmax_scale_boost``). Real Kimi 192 -> 256; reduced
        24 -> 64. Weight-absorbed MLA (which avoids the pad) is deferred to perf.
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
    def reduced(cls) -> "KimiK2Config":
        """A tiny, self-consistent config for CPU/dummy-mode modular tests and
        reduced-config golden runs. Keeps the *shape* of Kimi (MLA split heads,
        grouped MoE, one dense layer) while being small enough to run without
        the 1T checkpoint.
        """
        return cls(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            q_lora_rank=48,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
        )
