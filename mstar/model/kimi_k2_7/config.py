"""Configuration dataclass for Kimi-K2.7 (text backbone).

Kimi-K2.7's text architecture *is* DeepSeek-V3 — vLLM serves it as
``DeepseekV3ForCausalLM`` (``model_type: "kimi_k2"`` -> ``DeepseekV3Config``), so
this dataclass carries the full DeepSeek-V3 field set: MLA latent dims, sigmoid-
routed MoE grouping, and ``deepseek_yarn`` RoPE.

The full-size defaults are the real ``moonshotai/Kimi-K2.7-Code`` values. That
repo is the multimodal ``KimiK25ForConditionalGeneration``; the text dims here
live NESTED under ``config.json``'s ``text_config``, and its ``quantization_config``
is nested there too (see :meth:`k27_code` and ``kimi_model.py``). The modular tests
build from :meth:`reduced`, a tiny self-consistent config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mstar.model.kimi_k2_7.quantization import CompressedTensorsQuantConfig


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

    # -- MLA weight absorption (DEFAULT) -----------------------------------
    # ``True`` (default) => weight-absorbed MLA: ``kv_b_proj``'s up-projection is
    # folded into the Q path (``W_UK``) and the O path (``W_UV``) at load (plus the
    # ``fused_qkv_a_proj`` down-proj fusion), attention runs as MQA over the
    # COMPRESSED latent via the ``mla_absorb`` cache backend, and the KV cache
    # stores only the ``kv_lora_rank + qk_rope_head_dim`` latent (1 KV head) — a
    # ~57x per-token cache shrink, numerically identical up to fp rounding.
    # ``False`` => naive/materialized MLA (latent projected up to full per-head
    # K/V, padded to ``padded_head_dim``, MHA cache): the M4-golden parity
    # reference / opt-out fallback.
    #
    # PERF CAVEAT: the absorbed backend currently runs on a torch SDPA-over-latent
    # path — correct + memory-lean but EAGER-ONLY (no CUDA-graph capture) and slow
    # on the real 1T. The FlashInfer MLA kernel + CUDA-graph capture (production
    # throughput) is a follow-up; until it lands, real large-scale serving should
    # set ``mla_absorb=False`` (naive) or accept eager execution. See
    # ``components/attention.py``, ``kimi_model.py::get_kv_cache_config``,
    # ``engine/cache_manager.py::MlaAbsorbCacheManager``.
    mla_absorb: bool = True

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
        # context (4096 * 64). K2.7-Code keeps beta_fast=32; mscale ==
        # mscale_all_dim == 1.0.
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

    # -- Quantization (compressed-tensors INT4/fp8) -----------------------
    # ``None`` => native-bf16 checkpoint. When set, the weight loader dequantizes
    # the checkpoint stream on load (:mod:`mstar.model.kimi_k2_7.quantization`)
    # before the bf16 remap + stacked rules. Populated from the real checkpoint's
    # ``config.json`` ``quantization_config`` (``kimi_model.py``) or set directly
    # for the reduced/synthetic tests (:meth:`reduced_quantized`).
    quantization_config: CompressedTensorsQuantConfig | None = None

    # -- Quantization: memory-lean packed experts (in-kernel dequant) ----------
    # ``False`` => quantized routed experts are dequantized to bf16 on load and fed
    # to the bf16 fused-expert GEMM. ``True`` (only meaningful when
    # ``quantization_config`` is set) => the routed experts stay PACKED int32 in
    # VRAM and the W4A16 ``fused_moe_kernel_w4a16`` dequantizes each tile in
    # registers. MLA / dense-FFN / shared-expert weights are always dequantized on
    # load. This is the only path that fits the real 1T checkpoint. See
    # ``components/moe.py`` / ``weight_loader.py``.
    moe_in_kernel_dequant: bool = False

    # -- Quantization: routed-expert W4A16 kernel backend ----------------------
    # Chooses the kernel for the PACKED routed experts (only meaningful when
    # ``moe_in_kernel_dequant`` is set — Marlin layers on top of the Hook B packed
    # params). Values:
    #   "auto"   => Marlin on sm80+ (Ampere/Hopper) when the build succeeds and the
    #               shapes/group_size are Marlin-legal, else the Triton
    #               ``fused_moe_kernel_w4a16`` fallback. This is the production default.
    #   "marlin" => force Marlin; raise if ineligible (must not silently downgrade).
    #   "triton" => force the Triton in-kernel dequant path (the pre-Marlin behavior).
    # The final resolution happens post-load in
    # ``KimiSparseMoeBlock.process_weights_after_loading`` (a real device is needed to
    # probe capability + build the kernel); ``__init__`` runs on ``meta``.
    quant_kernel: str = "auto"

    # -- Serving: CUDA-graph prefill capture grid (optional overrides) ------
    # ``None`` => ``KimiLLMSubmodule`` uses its full-size class-default grid.
    # ``reduced()`` sets a tiny grid so the synthetic bring-up serve captures a
    # single short-prompt graph instead of the full 6x5 compiled grid.
    prefill_token_buckets: list[int] | None = None
    prefill_capture_batch_sizes: list[int] | None = None

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
        ``head_dim_vo ∈ {64, 128, 256}``, so it will not JIT-build for the real
        ``qk_head_dim=192`` or the reduced ``qk_head_dim=24``. We pad q/k (from
        ``qk_head_dim``) and v (from ``v_head_dim``) up to the smallest supported
        dim ``>= qk_head_dim``, run the paged attention there, and slice the output
        back to ``v_head_dim`` — compensating the softmax scale (see
        ``KimiMLAAttention.softmax_scale_boost``). Real Kimi 192 -> 256; reduced 24 -> 64.
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

        NOTE ``mla_absorb=False``: this fixture pins the NAIVE MLA path, the
        M4-golden parity reference that the bulk of the reduced test suite
        validates. The absorbed path (the production default) is exercised by the
        dedicated ``test_kimi_mla_absorb*`` tests, which flip ``mla_absorb=True``
        on a reduced() instance explicitly.
        """
        return cls(
            mla_absorb=False,
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
            # Tiny CUDA-graph prefill capture grid for the synthetic bring-up serve:
            # one short-prompt bucket at batch size 1 (the full 6x5 grid is slow and
            # its larger buckets exceed this 512-token model). Serve/CUDA-graph path
            # only — the golden tests call forward() directly and are unaffected.
            prefill_token_buckets=[64],
            prefill_capture_batch_sizes=[1],
        )

    @classmethod
    def reduced_quantized(
        cls,
        num_bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
    ) -> "KimiK2Config":
        """:meth:`reduced` plus a compressed-tensors quant config, to exercise the
        dequant-on-load path on a synthetic quantized checkpoint.

        The reduced dims (``hidden_size=128``, ``moe_intermediate_size=64``,
        ``intermediate_size=256`` …) are all divisible by the default
        ``group_size=32`` and by ``pack_factor=8``, so the FFN / expert / MLA
        weights whose input dim divides ``group_size`` can be quantized while the
        rest stay bf16 — the mixed checkpoint the streaming parser handles.
        """
        cfg = cls.reduced()
        cfg.quantization_config = CompressedTensorsQuantConfig(
            num_bits=num_bits, group_size=group_size, symmetric=symmetric,
        )
        return cfg

    @classmethod
    def reduced_quantized_inkernel(
        cls,
        num_bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
    ) -> "KimiK2Config":
        """:meth:`reduced_quantized` plus ``moe_in_kernel_dequant=True`` — packed
        routed experts + in-kernel INT4 dequant on a synthetic quantized checkpoint.

        The reduced dims (``hidden_size=128``, ``moe_intermediate_size=64``) satisfy
        the packed-expert divisibility asserts (``% pack_factor`` and ``%
        group_size``) at tp=1 (``shard_inter=64``) and tp=2 (``shard_inter=32``);
        tp=4 (``shard_inter=16``) fails ``% group_size`` (32), so pin packed-expert
        goldens to tp<=2.
        """
        cfg = cls.reduced_quantized(
            num_bits=num_bits, group_size=group_size, symmetric=symmetric,
        )
        cfg.moe_in_kernel_dequant = True
        return cfg

    @classmethod
    def reduced_marlin(
        cls,
        num_bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
    ) -> "KimiK2Config":
        """:meth:`reduced_quantized_inkernel` with Marlin-legal shapes + the Marlin
        routed-expert backend forced on.

        Marlin's GEMM imposes ``n % 64 == 0`` and ``k % 128 == 0`` on each expert
        matmul, which the default reduced dims (``hidden_size=128``,
        ``moe_intermediate_size=64``) do NOT satisfy for the down projection
        (``k == shard_inter``). This variant bumps ``hidden_size=256`` and
        ``moe_intermediate_size=256`` so both expert GEMMs are Marlin-legal at
        tp<=2 (tp=1 ``shard_inter=256``, tp=2 ``shard_inter=128`` — both ``% 128``);
        tp=4 (``shard_inter=64``) fails ``k % 128``, so pin Marlin goldens to tp<=2.
        ``group_size=32`` and ``pack_factor=8`` still divide both axes.
        """
        cfg = cls.reduced_quantized_inkernel(
            num_bits=num_bits, group_size=group_size, symmetric=symmetric,
        )
        cfg.hidden_size = 256
        cfg.moe_intermediate_size = 256
        cfg.intermediate_size = 512
        cfg.quant_kernel = "marlin"
        return cfg

    @classmethod
    def k27_code(cls) -> "KimiK2Config":
        """Full-size ``moonshotai/Kimi-K2.7-Code`` text-only serve config.

        Full 1T dims plus ``moe_in_kernel_dequant=True``: Kimi-K2.7-Code is a ~1T
        INT4 ``pack-quantized`` checkpoint (num_bits=4, group_size=32, symmetric,
        routed experts only), so the routed experts are served packed and
        dequantized in-kernel — dequantizing them to bf16 would need ~2 TB of VRAM.
        The ``quantization_config`` is nested under ``text_config`` and auto-read at
        load by ``kimi_model.py::_maybe_apply_checkpoint_quant_config``; MLA /
        dense-FFN / shared-expert / lm_head / vision weights stay bf16, matching the
        checkpoint ``ignore`` list.

        Keeps the default ``beta_fast=32.0``.
        """
        cfg = cls()
        cfg.moe_in_kernel_dequant = True
        return cfg
