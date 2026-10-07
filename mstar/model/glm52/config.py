"""GLM-5.2 architecture + generation config."""

from dataclasses import dataclass, field

from mstar.model.glm52.quantization import Fp8BlockQuantConfig

# Engine resource labels for the one LLM node: the paged (latent) KV cache,
# the attention resource planned over it, the sampler. Layers bind the first
# two by these names (``bind_resources``); the model declares all three in
# ``get_node_resources``.
KV_RESOURCE = "kv"
ATTN_RESOURCE = "attn"
SAMPLER_RESOURCE = "sampler"
# dsa_long_context: the DSA indexer's keys, one index_head_dim latent per token per FULL layer
INDEX_KV_RESOURCE = "kv_index"

# config.json keys read by Glm52ModelConfig.from_hf_config under the same
# name (eos_token_id becomes eos_token_ids).
_HF_REQUIRED_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
    "rms_norm_eps", "hidden_act",
    "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim",
    "index_n_heads", "index_head_dim", "index_topk", "index_topk_freq",
    "index_skip_topk_offset",
    "first_k_dense_replace", "intermediate_size", "moe_intermediate_size",
    "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
    "routed_scaling_factor", "scoring_func", "topk_method", "norm_topk_prob",
    "max_position_embeddings", "eos_token_id", "pad_token_id",
)
_HF_OPTIONAL_KEYS = (
    "tie_word_embeddings", "index_share_for_mtp_iteration",
    "indexer_rope_interleave", "moe_router_dtype", "rope_interleave",
    "num_nextn_predict_layers",
)


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

    # Absorbed MLA stores one compressed latent KV head; the naive path is
    # the reduced-test parity fallback.
    mla_absorb: bool = True

    # --- DSA sparse-attention indexer ---
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
    # Speculative decoding: 0 = off, k > 0 = draft k tokens per step with the
    # layer-78 MTP module. Gates both the Glm52MTPModule construction and the
    # layer-78 weight load.
    mtp_num_draft_tokens: int = 0
    # Engine half of DSA (opt-in; configs/glm52_tp8_longctx.yaml). Off: the
    # submodule guard holds every context to index_topk, where dense MLA IS
    # the exact DSA computation. On (dsa_paged.py): the guard checks
    # max_seq_len instead; FULL layers write their index keys to a KV resource
    # and every row past index_topk attends its top index_topk latents, prefill
    # rows included, with selection and sparse attention on device.
    dsa_long_context: bool = False
    # a prefill step longer than this runs the trunk over row chunks of it in turn,
    # so its activations are a chunk's, whatever the prompt's length
    prefill_chunk_tokens: int = 8192
    # under TP: each rank selects a block of a long prefill's rows and the blocks are
    # all-gathered, instead of every rank selecting every row
    dsa_shard_prefill: bool = False

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

    # --- MTP layer (built and loaded only when mtp_num_draft_tokens > 0) ---
    num_nextn_predict_layers: int = 1

    # --- tokens / generation defaults ---
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int = 154820
    # Kept under the ctx<=2048 exactness regime with room for the prompt; the
    # checkpoint's own default (8192) belongs with long context.
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
    # experts too — fine for reduced tests, but bf16 routed experts alone
    # outsize a device at TP8 on the real 753B checkpoint.
    moe_fp8_resident: bool = True
    # Routed-expert dispatch for the fp8-resident path: "auto" = the fused
    # W8A8 kernel (fused_experts_fp8, capture-safe) where it can run, else
    # "reference"; "triton" = the fused kernel or an error; "reference" = the
    # per-hit-expert dequant loop, the numerics anchor (uncapturable). W8A8
    # activation quantization is the standard fp8 serving numerics, as in vLLM.
    moe_quant_kernel: str = "auto"
    # With the fused path resolved, small batches run the router and the
    # routed + shared experts in fused_moe's decode kernels instead: fp8
    # weights times bf16 activations (w8a16) and fp32 routing weights, so
    # numerics differ from the W8A8 runner.
    moe_decode_kernel: bool = False
    # On the fused path the router's sigmoid, bias, top-k and normalization run as
    # fused_moe's one-program-per-token router kernel on the fp32 logits: one launch
    # where torch takes several; ids can differ at exact ties, weights by an ulp.
    # Off by default.
    moe_router_kernel: bool = False
    # Batches above 64 tokens on the fused path run fused_moe's prefill kernels:
    # the runner's with wider tiles and a one-pass down GEMM, bit-identical output.
    moe_prefill_kernel: bool = False
    # Under TP, sum the shared expert's partial into the routed one and
    # all-reduce once per MoE layer (components/moe.py). Off by default: bf16
    # rounding order moves.
    moe_fused_allreduce: bool = False
    # Keep the non-expert linears the checkpoint stores in fp8 (q_a/kv_a, q_b, o_proj, the
    # shared expert, the dense-layer MLPs) fp8 with their block scales instead of
    # dequantizing them to bf16: half the bytes per step. kv_b_proj still dequantizes (the
    # absorbed MLA folds it into w_kc / w_vc), and so does the shared expert under
    # moe_decode_kernel, whose chain runs it faster in bf16. Numerics differ from the bf16
    # copies (w8a16 for decode batches, W8A8 above), so off until the quality gate.
    dense_fp8: bool = False
    # The absorbed MLA's latent norms, RoPEs and KV-cache write as two kernels (mla_prep.py)
    # instead of seven; the same bits as the compiled unfused path. Off by default.
    mla_fused_prep: bool = False
    # Under TP, o_proj and the FFN return partials, and each all-reduce runs fused into
    # the next residual add + RMSNorm (CommGroup.allreduce_add_rmsnorm); the MoE block
    # then sums its two partials as moe_fused_allreduce does. Off by default: the norm
    # rounds differently.
    fused_add_rmsnorm: bool = False

    # Prefill runs the last layer's FFN half (norm, MoE, all-reduce) on the rows it samples
    # only. Those few rows take the small-batch MoE path, so the first token's numerics
    # move. Off by default.
    prefill_last_layer_rows: bool = False
    prefill_token_buckets: list[int] | None = None
    # Token buckets for the prefill batch sizes above 1; None captures them at
    # prefill_token_buckets too.
    prefill_batched_token_buckets: list[int] | None = None
    prefill_capture_batch_sizes: list[int] | None = None
    # Most prompt tokens one prefill step takes ("auto": the largest captured bucket), so a
    # burst runs as captured chunks with decode steps between them. None: no cap.
    prefill_max_step_tokens: int | str | None = None

    # Derived MLA cache geometry: the paged cache stores one 576-dim latent
    # vector per token per layer (512 compressed KV + 64 decoupled-RoPE key).
    cache_latent_dim: int = field(init=False)

    def __post_init__(self):
        self.cache_latent_dim = self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def fp8_shared_expert(self) -> bool:
        return self.dense_fp8 and not self.moe_decode_kernel

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def padded_head_dim(self) -> int:
        """Naive-MLA q/k/v pad target; FlashInfer paged kernels require 64/128/256."""
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

    @property
    def max_prompt_tokens(self) -> int:
        """The longest prompt served. Decode runs at least one step after the
        prefill and the next may already be scheduled when it stops, so the
        prompt leaves two steps' rows under the context limit (an MTP step
        writes k + 1); a DSA prefill attends densely, within index_topk."""
        limit = self.max_seq_len if self.dsa_long_context else self.index_topk
        return min(limit - 2 * (self.mtp_num_draft_tokens + 1), self.index_topk)

    @classmethod
    def from_hf_config(cls, hf_config: dict) -> "Glm52ModelConfig":
        """Geometry from the checkpoint's config.json; serving knobs keep
        their defaults."""
        from mstar.model.glm52.components.indexer import is_full_indexer_layer

        model_type = hf_config.get("model_type")
        if model_type not in (None, "glm_moe_dsa"):
            raise ValueError(f"not a GLM-5.2 config: model_type={model_type!r}")
        if hf_config.get("n_group", 1) != 1 or hf_config.get("topk_group", 1) != 1:
            raise ValueError("n_group/topk_group != 1: the router is groupless")
        missing = [k for k in _HF_REQUIRED_KEYS if k not in hf_config]
        rope_theta = hf_config.get(
            "rope_theta", (hf_config.get("rope_parameters") or {}).get("rope_theta"))
        if rope_theta is None:
            missing.append("rope_theta")
        if missing:
            raise ValueError(f"config.json lacks {missing}")
        # plain RoPE only: a scaled one (YaRN, dynamic) ran as plain, its frequencies
        # and softmax scale wrong
        for rope in (hf_config.get("rope_parameters"), hf_config.get("rope_scaling")):
            kind = (rope or {}).get("rope_type", (rope or {}).get("type", "default"))
            if kind not in (None, "default"):
                raise ValueError(f"rope type {kind!r}: this model serves plain RoPE only")

        eos = hf_config["eos_token_id"]
        config = cls(
            **{k: hf_config[k] for k in _HF_REQUIRED_KEYS if k != "eos_token_id"},
            **{k: hf_config[k] for k in _HF_OPTIONAL_KEYS if k in hf_config},
            rope_theta=float(rope_theta),
            eos_token_ids=tuple(eos) if isinstance(eos, list) else (eos,),
            max_seq_len=hf_config["index_topk"],
            quantization_config=Fp8BlockQuantConfig.from_hf_config_dict(
                hf_config.get("quantization_config")),
        )

        n = config.num_hidden_layers
        schedules = {
            "indexer_types": [
                "full" if is_full_indexer_layer(config, i) else "shared"
                for i in range(n)
            ],
            "mlp_layer_types": [
                "dense" if i < config.first_k_dense_replace else "sparse"
                for i in range(n)
            ],
        }
        for key, want in schedules.items():
            if key in hf_config and list(hf_config[key]) != want:
                raise ValueError(
                    f"config.json {key} disagrees with the schedule this "
                    f"model derives: got {hf_config[key]}, want {want}"
                )
        return config

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
