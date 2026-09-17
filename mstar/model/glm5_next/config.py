"""GLM-5.3-Flash (glm5_next) architecture + generation config."""

from dataclasses import dataclass, field

# fp8 e4m3 with [128, 128] `weight_scale_inv` block scales and dynamic
# activation scaling: the checkpoint's quantization scheme.
from mstar.model.glm5_next.quantization import Fp8BlockQuantConfig

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"

# Resource keys the model declares in get_node_resources and the layers
# bind by (bind_resources); the serving YAML tunes them under `resources:`.
KV_CACHE = "kv_cache"      # MLA latent pages of the 11 full-attention layers
ATTN = "attn"              # the absorbed-MLA attention backend over KV_CACHE
SAMPLER = "sampler"
KDA_STATE = "kda_state"    # slot-pooled recurrent + conv state of the 34 KDA layers
# The one cache stream label glm5_next uses (single-label model).
LABEL = "main"

# Full attention sits at every 4th layer starting at 3 (idx % 4 == 3); all
# other layers are KDA. A package invariant, validated against the schedule
# lists — reduced test configs must keep it so the layer mix is exercised.
_FULL_ATTN_PERIOD = 4
_FULL_ATTN_OFFSET = 3


def build_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
    """The checkpoint's attention schedule: MLA+DSA at idx % 4 == 3."""
    return tuple(
        FULL_ATTENTION
        if idx % _FULL_ATTN_PERIOD == _FULL_ATTN_OFFSET
        else LINEAR_ATTENTION
        for idx in range(num_hidden_layers)
    )


def build_mlp_layer_types(
    num_hidden_layers: int, first_k_dense_replace: int
) -> tuple[str, ...]:
    """Dense MLP on the first ``first_k_dense_replace`` layers, MoE after."""
    return tuple(
        DENSE_MLP if idx < first_k_dense_replace else SPARSE_MLP
        for idx in range(num_hidden_layers)
    )


def build_indexer_types(num_hidden_layers: int) -> tuple[str, ...]:
    """Every MLA layer runs its own full indexer; none are shared."""
    return tuple("full" for _ in range(num_hidden_layers))


@dataclass
class Glm5NextModelConfig:
    # --- backbone ---
    vocab_size: int = 154880
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64  # checkpoint requires == num_attention_heads
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    # --- layer schedule (from config.json, not formulas) ---
    # 34 x linear_attention + 11 x deepseek_sparse_attention (3, 7, ..., 43);
    # dense MLP on layers 0..2, 288+1-expert MoE everywhere else.
    layer_types: tuple[str, ...] = field(
        default_factory=lambda: build_layer_types(45)
    )
    mlp_layer_types: tuple[str, ...] = field(
        default_factory=lambda: build_mlp_layer_types(45, 3)
    )
    indexer_types: tuple[str, ...] = field(
        default_factory=lambda: build_indexer_types(45)
    )

    # --- KDA linear attention (config.json linear_attn_config) ---
    # Per-request fixed-size recurrent + short-conv state, NOT paged KV.
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_size: int = 4  # HF short_conv_kernel_size
    # "Safe gate" forget branch: g = gate_lower_bound * sigmoid(...); the
    # softplus branch in the HF class is dead for this checkpoint.
    gate_lower_bound: float = -5.0

    # --- mHC manifold-constrained hyper-connections ---
    # The residual is hc_mult parallel streams [B, S, 4, hidden]; each
    # sublayer site (attn_hc / ffn_hc) collapses them through learned pre
    # weights and writes back through post weights + a Sinkhorn-projected
    # doubly-stochastic mixing matrix. Sinkhorn runs at forward time (the
    # comb logits depend on the live streams — nothing to precompute).
    # hc_eps floors pre/comb and guards Sinkhorn denominators; the input
    # RMSNorm uses rms_norm_eps — two distinct epsilons.
    mhc: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # --- MLA (multi-head latent attention), NoPE ---
    # mla_use_nope: qk_rope_head_dim == 0 — no RoPE anywhere in the text
    # model, so the cache latent is exactly kv_lora_rank (512) and there is
    # no cos/sin threading. qk_head_dim = 256 either way, scale 256**-0.5.
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    mla_use_nope: bool = True

    # Absorbed MLA stores one compressed latent KV head per token in the
    # engine's MLA cache layout. This is the only path served (the engine's
    # MLA backend has an fp32 SDPA fallback, so reduced CPU configs run the
    # same path); the flag is kept for config parity and must stay True.
    mla_absorb: bool = True

    # --- DSA sparse-attention indexer (k-pool compression) ---
    # Scoring runs over pools of index_kpool consecutive tokens (learned
    # softmax over members + additive prior), top-(index_topk/index_kpool)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    index_share_for_mtp_iteration: bool = True
    indexer_rope_interleave: bool = True  # vestigial: rope_dim is 0 here
    # 0 = MTP off; k > 0 = draft k tokens per step with the layer-45 MTP
    # module. Gates both MTP construction and the layer-45 weight load.
    mtp_num_draft_tokens: int = 0
    # Engine half of DSA. Off: serving holds every context to index_topk,
    # where dense MLA computes exactly what DSA would (selecting
    # <= topk/kpool pools out of <= topk/kpool is the identity, tail
    # included).
    dsa_long_context: bool = False

    # --- MLP / MoE ---
    first_k_dense_replace: int = 3  # layers 0..2 are dense
    intermediate_size: int = 12288  # dense-layer MLP
    moe_intermediate_size: int = 2048  # per expert
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True  # denominator carries +1e-20 (HF parity)
    moe_router_dtype: str = "float32"
    # SwiGLU clamp on every MLP (dense, shared, routed experts alike):
    # gate.clamp(max=limit), up.clamp(-limit, limit), then silu(gate) * up.
    swiglu_limit: float = 10.0

    # --- lengths (NoPE: no rope_theta / interleave fields) ---
    max_position_embeddings: int = 1_048_576
    # Serving cap, consumed by get_node_resources. Held to index_topk
    # while dsa_long_context is off: dense MLA is exactly the DSA
    # computation only within the top-2048 window, and the submodule's
    # preprocess guard enforces the same bound.
    max_seq_len: int = 2048

    # --- MTP (speculative decoding) ---
    # Layer 45 in the checkpoint: DeepSeek-V3 glue (enorm/hnorm/eh_proj/
    # shared_head.norm) + one full MLA+DSA+MoE decoder layer with NO mHC
    # tensors — the draft layer runs plain-residual on the collapsed
    # stream.
    num_nextn_predict_layers: int = 1

    # --- tokens / generation defaults ---
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int = 154820
    max_output_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    # --- quantization / serving knobs ---
    # The official checkpoint is FP8 e4m3 with [128, 128] block scales
    # (`weight_scale_inv`). Populated from config.json by from_hf_config.
    quantization_config: Fp8BlockQuantConfig | None = None
    # Keep routed experts FP8-resident (uint8 container + block scales);
    # everything else dequantizes to bf16 on load. 288 experts x 43 layers
    # dominate the 306 GB checkpoint.
    moe_fp8_resident: bool = True
    moe_quant_kernel: str = "reference"

    prefill_token_buckets: list[int] | None = None
    prefill_capture_batch_sizes: list[int] | None = None

    # Derived MLA cache geometry: with NoPE the paged cache stores one
    # 512-dim latent per token per FULL layer (kv_lora_rank + 0 rope dims),
    # and only the 11 full-attention layers have planes at all.
    cache_latent_dim: int = field(init=False)

    def __post_init__(self):
        self.cache_latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._validate()

    def _validate(self) -> None:
        """Structural invariants every config (real or reduced) must keep."""
        n = self.num_hidden_layers
        for name, schedule in (
            ("layer_types", self.layer_types),
            ("mlp_layer_types", self.mlp_layer_types),
            ("indexer_types", self.indexer_types),
        ):
            if len(schedule) != n:
                raise ValueError(
                    f"{name} has {len(schedule)} entries for "
                    f"num_hidden_layers={n}"
                )
        for idx, layer_type in enumerate(self.layer_types):
            if layer_type not in (LINEAR_ATTENTION, FULL_ATTENTION):
                raise ValueError(f"layer_types[{idx}] = {layer_type!r} unknown")
            expect_full = idx % _FULL_ATTN_PERIOD == _FULL_ATTN_OFFSET
            if (layer_type == FULL_ATTENTION) != expect_full:
                raise ValueError(
                    f"layer_types[{idx}] = {layer_type!r} breaks the "
                    f"idx % {_FULL_ATTN_PERIOD} == {_FULL_ATTN_OFFSET} "
                    "full-attention rule"
                )
        for idx, mlp_type in enumerate(self.mlp_layer_types):
            if mlp_type not in (DENSE_MLP, SPARSE_MLP):
                raise ValueError(f"mlp_layer_types[{idx}] = {mlp_type!r} unknown")
            if (mlp_type == DENSE_MLP) != (idx < self.first_k_dense_replace):
                raise ValueError(
                    f"mlp_layer_types[{idx}] = {mlp_type!r} disagrees with "
                    f"first_k_dense_replace={self.first_k_dense_replace}"
                )
        if any(t != "full" for t in self.indexer_types):
            raise ValueError(
                "indexer_types must be all 'full' — GLM-5.3-Flash has no "
                "IndexShare; do not port the glm52 formula"
            )
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} != "
                f"num_key_value_heads={self.num_key_value_heads} "
                "(checkpoint config enforces equality)"
            )
        if self.mla_use_nope and self.qk_rope_head_dim != 0:
            raise ValueError(
                f"mla_use_nope with qk_rope_head_dim={self.qk_rope_head_dim}"
            )
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(
                f"index_topk={self.index_topk} must be a multiple of "
                f"index_kpool={self.index_kpool}"
            )
        if self.hc_mult < 1 or self.hc_sinkhorn_iters < 1:
            raise ValueError(
                f"hc_mult={self.hc_mult}, "
                f"hc_sinkhorn_iters={self.hc_sinkhorn_iters} must be >= 1"
            )
        if self.linear_conv_kernel_size < 2:
            raise ValueError(
                "linear_conv_kernel_size must be >= 2 (decode conv state has "
                f"width kernel-1); got {self.linear_conv_kernel_size}"
            )
        if self.num_nextn_predict_layers not in (0, 1):
            raise ValueError(
                "one iterated MTP module is the only supported layout; got "
                f"num_nextn_predict_layers={self.num_nextn_predict_layers}"
            )

    # --- derived schedule views ---

    @property
    def full_attn_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, t in enumerate(self.layer_types)
            if t == FULL_ATTENTION
        )

    @property
    def kda_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, t in enumerate(self.layer_types)
            if t == LINEAR_ATTENTION
        )

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True for MLA+DSA layers; the MTP layer (== num_hidden_layers)
        always is (it ships its own full indexer)."""
        if layer_idx >= self.num_hidden_layers:
            return True
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def is_dense_mlp_layer(self, layer_idx: int) -> bool:
        if layer_idx >= self.num_hidden_layers:
            return False  # the MTP layer is MoE
        return self.mlp_layer_types[layer_idx] == DENSE_MLP

    # --- derived dims ---

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_cache_kpe(self) -> int:
        """kpe (rope-slice) width stored in the MLA latent cache."""
        return 64

    @property
    def padded_head_dim(self) -> int:
        """Naive-MLA q/k/v pad target; FlashInfer paged kernels require
        64/128/256. The full model's qk_head_dim is exactly 256 so padding
        is a no-op; reduced configs exercise the pad path."""
        for supported in (64, 128, 256):
            if supported >= self.qk_head_dim:
                return supported
        raise ValueError(
            f"qk_head_dim={self.qk_head_dim} exceeds the largest FlashInfer "
            "SM90 head_dim (256); the naive-MLA pad mitigation cannot cover it."
        )

    @property
    def linear_qkv_dim(self) -> int:
        """Per-projection KDA width: heads x head_dim (8192 full-size)."""
        return self.linear_num_heads * self.linear_head_dim

    @property
    def linear_conv_channels(self) -> int:
        """Fused depthwise conv channels over cat(q, k, v): 3 x qkv_dim."""
        return 3 * self.linear_qkv_dim

    @property
    def num_dense_layers(self) -> int:
        return min(self.first_k_dense_replace, self.num_hidden_layers)

    @classmethod
    def from_hf_config(
        cls, hf_config: dict, strict: bool = True
    ) -> "Glm5NextModelConfig":
        """Build from the checkpoint's config.json dict."""
        text = hf_config.get("text_config", hf_config)
        model_type = text.get("model_type")
        if model_type not in (None, "glm5_next_text", "glm5_next"):
            raise ValueError(f"not a glm5_next config: model_type={model_type!r}")
        linear = text["linear_attn_config"]

        config = cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            hidden_act=str(text["hidden_act"]),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            layer_types=tuple(text["layer_types"]),
            mlp_layer_types=tuple(text["mlp_layer_types"]),
            indexer_types=tuple(text["indexer_types"]),
            linear_num_heads=int(linear["num_heads"]),
            linear_head_dim=int(linear["head_dim"]),
            linear_conv_kernel_size=int(linear["short_conv_kernel_size"]),
            gate_lower_bound=float(linear["gate_lower_bound"]),
            mhc=bool(text["mhc"]),
            hc_mult=int(text["hc_mult"]),
            hc_sinkhorn_iters=int(text["hc_sinkhorn_iters"]),
            hc_eps=float(text["hc_eps"]),
            q_lora_rank=int(text["q_lora_rank"]),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            mla_use_nope=bool(text["mla_use_nope"]),
            index_n_heads=int(text["index_n_heads"]),
            index_head_dim=int(text["index_head_dim"]),
            index_topk=int(text["index_topk"]),
            index_kpool=int(text["index_kpool"]),
            index_kpool_compress=bool(text["index_kpool_compress"]),
            index_kpool_always_select_tail=bool(
                text["index_kpool_always_select_tail"]
            ),
            index_share_for_mtp_iteration=bool(
                text["index_share_for_mtp_iteration"]
            ),
            indexer_rope_interleave=bool(text["indexer_rope_interleave"]),
            first_k_dense_replace=int(text["first_k_dense_replace"]),
            intermediate_size=int(text["intermediate_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            n_routed_experts=int(text["n_routed_experts"]),
            n_shared_experts=int(text["n_shared_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            routed_scaling_factor=float(text["routed_scaling_factor"]),
            scoring_func=str(text["scoring_func"]),
            topk_method=str(text["topk_method"]),
            norm_topk_prob=bool(text["norm_topk_prob"]),
            moe_router_dtype=str(text["moe_router_dtype"]),
            swiglu_limit=float(text["swiglu_limit"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            num_nextn_predict_layers=int(text["num_nextn_predict_layers"]),
            eos_token_ids=tuple(int(t) for t in text["eos_token_id"]),
            pad_token_id=int(text["pad_token_id"]),
            quantization_config=Fp8BlockQuantConfig.from_hf_config_dict(
                hf_config.get("quantization_config")
                or text.get("quantization_config")
            ),
        )

        # The linear_attn_config layer lists must agree with layer_types.
        if tuple(linear["kda_layers"]) != config.kda_layer_indices:
            raise ValueError(
                "linear_attn_config.kda_layers disagrees with layer_types"
            )
        if tuple(linear["full_attn_layers"]) != config.full_attn_layer_indices:
            raise ValueError(
                "linear_attn_config.full_attn_layers disagrees with layer_types"
            )
        # Group routing must be the identity; the expert-group machinery
        # is not implemented.
        if int(text.get("n_group", 1)) != 1 or int(text.get("topk_group", 1)) != 1:
            raise ValueError(
                "n_group/topk_group != 1: expert-group routing is not ported"
            )

        if strict:
            config._validate_glm53_flash()
        return config

    def _validate_glm53_flash(self) -> None:
        """Pin the known GLM-5.3-Flash checkpoint geometry (strict parse)."""
        expected = {
            "num_hidden_layers": 45,
            "hidden_size": 4096,
            "vocab_size": 154880,
            "num_attention_heads": 64,
            "linear_num_heads": 64,
            "linear_head_dim": 128,
            "linear_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_kpool": 4,
            "first_k_dense_replace": 3,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "swiglu_limit": 10.0,
            "num_nextn_predict_layers": 1,
            "cache_latent_dim": 512,
        }
        mismatched = {
            key: (getattr(self, key), want)
            for key, want in expected.items()
            if getattr(self, key) != want
        }
        if mismatched:
            raise ValueError(
                f"config.json does not match GLM-5.3-Flash (got, want): "
                f"{mismatched} — pass strict=False if this is intentional"
            )
        if len(self.full_attn_layer_indices) != 11:
            raise ValueError(
                f"expected 11 full-attention layers, got "
                f"{len(self.full_attn_layer_indices)}"
            )
        if len(self.kda_layer_indices) != 34:
            raise ValueError(
                f"expected 34 KDA layers, got {len(self.kda_layer_indices)}"
            )

    @classmethod
    def reduced(cls) -> "Glm5NextModelConfig":
        """Tiny-dim variant for CPU tests: same shapes family, random weights."""
        return cls(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            layer_types=build_layer_types(8),
            mlp_layer_types=build_mlp_layer_types(8, 3),
            indexer_types=build_indexer_types(8),
            linear_num_heads=4,
            linear_head_dim=16,
            linear_conv_kernel_size=4,
            hc_mult=2,  # the mHC math is H-generic; 2 streams for speed
            q_lora_rank=48,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=0,
            v_head_dim=16,
            index_n_heads=4,
            index_head_dim=16,
            index_topk=64,
            index_kpool=4,
            first_k_dense_replace=3,
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
    def reduced_fp8(
        cls, block: tuple[int, int] = (16, 16)
    ) -> "Glm5NextModelConfig":
        cfg = cls.reduced()
        cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=block)
        return cfg
