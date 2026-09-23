"""Kimi-K2.7 text config, using the DeepSeek-V3 architecture fields."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from mstar.model.components.quantization import CompressedTensorsQuantConfig

logger = logging.getLogger(__name__)

# Resource keys the model declares and the layers resolve against.
KV_CACHE = "kv_cache"
ATTN = "attn"
SAMPLER = "sampler"
ROPE = "rope"

_CHECKPOINT_CONFIG_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "rms_norm_eps", "max_position_embeddings",
    "tie_word_embeddings", "hidden_act", "q_lora_rank", "kv_lora_rank",
    "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "n_routed_experts",
    "n_shared_experts", "num_experts_per_tok", "moe_intermediate_size",
    "n_group", "topk_group", "routed_scaling_factor", "scoring_func",
    "topk_method", "norm_topk_prob", "first_k_dense_replace", "moe_layer_freq",
    "rope_theta", "rope_scaling", "bos_token_id", "eos_token_id", "pad_token_id",
    "num_nextn_predict_layers",
)


def _quant_raw_from_config_dict(raw: dict) -> dict | None:
    """The compressed-tensors sub-dict, wherever a flat or Kimi-K2.5
    ``text_config``-nested config.json puts it."""
    return raw.get("quantization_config") or (raw.get("text_config") or {}).get(
        "quantization_config"
    )


@dataclass
class KimiK2Config:
    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 18432
    num_hidden_layers: int = 61
    num_attention_heads: int = 64
    num_key_value_heads: int = 64  # MLA has no separate KV heads; kept for HF parity
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # Default absorbed MLA stores one compressed latent KV head. The naive path is
    # kept as the reduced-test parity fallback.
    mla_absorb: bool = True

    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.827
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1

    rope_theta: float = 50000.0
    rope_scaling: dict = field(default_factory=lambda: {
        "rope_type": "deepseek_yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    })

    bos_token_id: int = 163584
    eos_token_id: int = 163586
    pad_token_id: int = 163839
    # generation_config.json's eos_token_id may be an int or a list; check_stop
    # matches against this list. Defaults to [eos_token_id].
    eos_token_ids: list[int] = field(default_factory=lambda: [163586])
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    num_nextn_predict_layers: int = 0

    quantization_config: CompressedTensorsQuantConfig | None = None

    # Keeps routed experts packed; non-expert quantized weights still dequantize
    # on load.
    moe_in_kernel_dequant: bool = False

    # "auto" probes Marlin post-load on the real device; "marlin" must not silently
    # downgrade, and "triton" keeps the packed Triton path.
    quant_kernel: str = "auto"

    prefill_token_buckets: list[int] | None = None
    prefill_capture_batch_sizes: list[int] | None = None

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

    @classmethod
    def reduced(cls) -> "KimiK2Config":
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
        cfg = cls()
        cfg.moe_in_kernel_dequant = True
        return cfg

    @classmethod
    def from_checkpoint(
        cls, source: str | Path, base: "KimiK2Config | None" = None
    ) -> "KimiK2Config":
        """Override ``base`` (or the dataclass defaults) with a checkpoint
        directory's ``config.json`` / ``generation_config.json``.

        Handles both flat DeepSeek-V3-style config.json (fields at the top
        level) and Kimi-K2.5-style (text fields under ``text_config``). Falls
        back to ``base`` unchanged, with a warning, if ``config.json`` is
        missing. Fields absent from the checkpoint keep ``base``'s value.
        """
        base = base if base is not None else cls()
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            logger.warning(
                "KimiK2Config.from_checkpoint: no config.json in %s; using "
                "defaults.", source,
            )
            return base

        with open(config_json) as f:
            raw = json.load(f)
        text_raw = raw.get("text_config") or raw
        overrides = {
            k: v for k, v in text_raw.items() if k in _CHECKPOINT_CONFIG_FIELDS
        }

        if overrides.get("tie_word_embeddings"):
            raise ValueError(
                "Kimi-K2.7 checkpoint declares tie_word_embeddings=True; "
                "mstar's lm_head is a separate parameter and weight tying is "
                "not implemented."
            )
        hidden_act = overrides.get("hidden_act")
        if hidden_act is not None and hidden_act != "silu":
            raise ValueError(
                f"Kimi-K2.7 checkpoint declares hidden_act={hidden_act!r}; "
                "only 'silu' is implemented."
            )

        if base.quantization_config is None:
            quant = CompressedTensorsQuantConfig.from_hf_config_dict(
                _quant_raw_from_config_dict(raw)
            )
            if quant is not None:
                overrides["quantization_config"] = quant

        config_eos = overrides.get("eos_token_id", base.eos_token_id)
        if isinstance(config_eos, list):
            overrides["eos_token_ids"] = list(config_eos)
            if config_eos:
                overrides["eos_token_id"] = config_eos[0]
        else:
            overrides["eos_token_ids"] = [config_eos]
        gen_json = Path(source) / "generation_config.json"
        if gen_json.is_file():
            with open(gen_json) as f:
                gen_raw = json.load(f)
            gen_eos = gen_raw.get("eos_token_id")
            if gen_eos is not None:
                overrides["eos_token_ids"] = (
                    list(gen_eos) if isinstance(gen_eos, list) else [gen_eos]
                )
            if "temperature" in gen_raw:
                overrides["temperature"] = gen_raw["temperature"]
            if "top_p" in gen_raw:
                overrides["top_p"] = gen_raw["top_p"]

        return replace(base, **overrides)
