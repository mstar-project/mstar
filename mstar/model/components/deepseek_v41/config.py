"""DeepSeek-V4.1 architecture configuration and schedule helpers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mstar.model.components.deepseek_v41.moe import DeepSeekV41MoeConfig
from mstar.utils.fp8_block import Fp8BlockQuantConfig

_TEXT_KEYS = (
    "vocab_size",
    "hidden_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "hidden_act",
    "swiglu_limit",
    "rms_norm_eps",
    "max_position_embeddings",
    "rope_theta",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "scoring_func",
    "norm_topk_prob",
    "routed_scaling_factor",
    "sliding_window",
    "compress_rope_theta",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "candidate_topk_blocks",
    "candidate_block_size",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_pad_token_id",
    "engram_compressed_vocab_size",
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_num_experts_per_tok",
)


@dataclass
class DeepSeekV41Config:
    """DeepSeek-V4.1-Flash text backbone configuration."""

    # --- backbone ---
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-20
    max_position_embeddings: int = 1_048_576
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # --- attention: one latent per token that is both key and value ---
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128

    # --- compressed sparse attention ---
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # --- RoPE: plain on window-only layers; YaRN at compress_rope_theta on the others ---
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_position_embeddings: int = 65536

    # --- MoE ---
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    hidden_act: str = "silu"
    gate_temp: float = 1.0
    vision_enabled: bool = True

    # --- hyper-connections ---
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # --- Engram ---
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16_000_000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092

    # --- DSpark / MTP (metadata only in Phase 1) ---
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    # --- tokens ---
    bos_token_id: int = 0
    eos_token_ids: tuple[int, ...] = (1,)
    pad_token_id: int = 2
    image_token_id: int = 129264

    # --- quantization ---
    fp8_block: int = 32

    # --- serving ---
    max_seq_len: int = 65536
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        self.compress_ratios = tuple(int(r) for r in self.compress_ratios)
        self.kv_source_layer_ids = tuple(int(i) for i in self.kv_source_layer_ids)
        self.index_source_layer_ids = tuple(int(i) for i in self.index_source_layer_ids)
        self.engram_layer_ids = tuple(int(i) for i in self.engram_layer_ids)
        self.engram_num_embeddings = tuple(int(n) for n in self.engram_num_embeddings)
        self.dspark_target_layer_ids = tuple(int(i) for i in self.dspark_target_layer_ids)
        self.eos_token_ids = tuple(int(i) for i in self.eos_token_ids)
        self._validate()

    def _validate(self) -> None:
        n = self.num_hidden_layers
        if n < 1:
            raise ValueError("num_hidden_layers must be positive")
        if len(self.compress_ratios) < n:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {n} layers")
        if self.num_key_value_heads != 1:
            raise ValueError(f"DeepSeek-V4.1 requires num_key_value_heads=1, got {self.num_key_value_heads}")
        if self.head_dim <= self.qk_rope_head_dim or self.qk_rope_head_dim < 2 or self.qk_rope_head_dim % 2:
            raise ValueError("head_dim must exceed even positive qk_rope_head_dim")
        if self.q_lora_rank < 1 or self.o_lora_rank < 1:
            raise ValueError("q_lora_rank and o_lora_rank must be positive")
        if self.hidden_size % self.o_groups or self.num_attention_heads % self.o_groups:
            raise ValueError("hidden_size and num_attention_heads must be divisible by o_groups")
        if self.n_shared_experts != 1:
            raise ValueError(f"only n_shared_experts=1 is supported, got {self.n_shared_experts}")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(f"only scoring_func='sqrtsoftplus' is supported, got {self.scoring_func!r}")
        for width in (
            self.head_dim,
            self.index_head_dim,
            self.q_lora_rank,
            self.o_lora_rank,
            self.hidden_size,
            self.moe_intermediate_size,
            self.engram_head_dim,
        ):
            if width % self.fp8_block:
                raise ValueError(f"width {width} is not a multiple of fp8_block={self.fp8_block}")
        if set(self.kv_source_layer_ids) - set(self.index_source_layer_ids):
            raise ValueError("every kv_source_layer_id must also be in index_source_layer_ids")
        for layer in range(n):
            ratio = self.compress_ratios[layer]
            if ratio < 0:
                raise ValueError(f"compress_ratios[{layer}] must be >= 0, got {ratio}")
            src = self.kv_source_of(layer)
            if ratio and src is None:
                raise ValueError(f"layer {layer} attends compressed cache but has no preceding kv_source_layer_id")
            if ratio and self.compress_ratios[src] != ratio:
                raise ValueError(
                    f"layer {layer} has compress_ratio={ratio} != source layer {src} ratio {self.compress_ratios[src]}"
                )
            if layer in self.kv_source_layer_ids and not ratio:
                raise ValueError(f"kv_source_layer_id {layer} cannot have compress_ratio=0")
        cand = self.candidate_source_layer_id
        if cand >= 0:
            if cand not in self.kv_source_layer_ids:
                raise ValueError("candidate_source_layer_id must be in kv_source_layer_ids")
            later_kv = [i for i in self.kv_source_layer_ids if i > cand]
            if later_kv:
                raise ValueError(f"kv_source_layer_ids {later_kv} cannot appear after candidate_source_layer_id {cand}")
            if (self.candidate_topk_blocks - 1) * self.candidate_block_size + 1 < self.index_topk:
                raise ValueError("candidate_topk_blocks * candidate_block_size must cover index_topk")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings must have one entry per engram_layer_id")
        if self.engram_layer_ids and self.engram_max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size must be >= 2 when engram_layer_ids is non-empty")

    def kv_source_of(self, layer: int) -> int | None:
        if not self.compress_ratios[layer]:
            return None
        before = [i for i in self.kv_source_layer_ids if i <= layer]
        return before[-1] if before else None

    def index_source_of(self, layer: int) -> int | None:
        if not self.compress_ratios[layer]:
            return None
        before = [i for i in self.index_source_layer_ids if i <= layer]
        return before[-1] if before else None

    def uses_candidates(self, layer: int) -> bool:
        return 0 <= self.candidate_source_layer_id < layer

    @property
    def pooled_source_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i in self.kv_source_layer_ids if self.compress_ratios[i] > 1)

    @property
    def engram_hash_cols(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.eos_token_ids

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def moe_config(self) -> DeepSeekV41MoeConfig:
        return DeepSeekV41MoeConfig(
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            norm_topk_prob=self.norm_topk_prob,
            n_shared_experts=self.n_shared_experts,
            hidden_act=self.hidden_act,
            gate_temp=self.gate_temp,
            vision_enabled=self.vision_enabled,
            quantization_config=Fp8BlockQuantConfig(weight_block_size=(self.fp8_block, self.fp8_block)),
        )

    @classmethod
    def from_hf(cls, data: dict) -> DeepSeekV41Config:
        """Parse an official HuggingFace ``config.json`` dict."""
        if data.get("model_type") != "deepseek_v41":
            raise ValueError(f"unsupported model_type={data.get('model_type')!r}, expected 'deepseek_v41'")
        text = data["text_config"]
        missing = [k for k in _TEXT_KEYS if k not in text]
        if missing:
            raise ValueError(f"text_config is missing keys: {missing}")
        if text.get("topk_method") != "noaux_tc":
            raise ValueError(f"unsupported topk_method={text.get('topk_method')!r}")
        rope = text.get("rope_scaling") or {}
        if rope.get("rope_type") != "yarn":
            raise ValueError(f"unsupported rope_scaling={rope!r}, expected 'yarn'")
        quant = data.get("quantization_config") or {}
        block = tuple(quant.get("weight_block_size", (32, 32)))
        if (
            quant.get("quant_method") != "fp8"
            or quant.get("expert_dtype") != "fp4"
            or quant.get("scale_fmt") != "ue8m0"
            or len(block) != 2
            or block[0] != block[1]
        ):
            raise ValueError(f"unsupported quantization_config={quant!r}")
        eos = data.get("eos_token_id", 1)
        return cls(
            **{k: text[k] for k in _TEXT_KEYS},
            compress_ratios=tuple(text["compress_ratios"]),
            kv_source_layer_ids=tuple(text["kv_source_layer_ids"]),
            index_source_layer_ids=tuple(text["index_source_layer_ids"]),
            candidate_source_layer_id=int(text.get("candidate_source_layer_id", -1)),
            rope_factor=float(rope["factor"]),
            rope_beta_fast=float(rope["beta_fast"]),
            rope_beta_slow=float(rope["beta_slow"]),
            rope_original_max_position_embeddings=int(rope["original_max_position_embeddings"]),
            engram_layer_ids=tuple(text["engram_layer_ids"]),
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            dspark_target_layer_ids=tuple(text.get("dspark_target_layer_ids", ())),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            initializer_range=float(text.get("initializer_range", 0.02)),
            gate_temp=float(text.get("gate_temp", 1.0)),
            vision_enabled=bool(data.get("vision_config")),
            bos_token_id=int(data.get("bos_token_id", 0)),
            eos_token_ids=tuple(int(x) for x in eos) if isinstance(eos, list) else (int(eos),),
            pad_token_id=int(data.get("pad_token_id", 2)),
            image_token_id=int(data.get("image_token_id", 129264)),
            fp8_block=int(block[0]),
        )

    @classmethod
    def reduced(cls, **overrides) -> DeepSeekV41Config:
        """Small-dimension variant exercising every V4.1 mechanism on CPU."""
        base = {
            "vocab_size": 512,
            "hidden_size": 64,
            "num_hidden_layers": 7,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_lora_rank": 32,
            "o_groups": 2,
            "sliding_window": 8,
            "compress_ratios": (0, 2, 2, 2, 1, 1, 1, 0, 0, 0),
            "kv_source_layer_ids": (1, 3, 4),
            "index_source_layer_ids": (1, 3, 4, 6),
            "index_n_heads": 16,
            "index_head_dim": 32,
            "index_topk": 8,
            "candidate_source_layer_id": 4,
            "candidate_topk_blocks": 3,
            "candidate_block_size": 4,
            "rope_original_max_position_embeddings": 64,
            "rope_factor": 4.0,
            "moe_intermediate_size": 32,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "vision_enabled": False,
            "engram_layer_ids": (1,),
            "engram_num_embeddings": (0,),
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 50,
            "engram_n_heads": 2,
            "engram_head_dim": 32,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 512,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 500,
            "dspark_target_layer_ids": (4, 5, 6),
            "dspark_markov_rank": 16,
            "dspark_n_routed_experts": 4,
            "dspark_num_experts_per_tok": 2,
            "max_position_embeddings": 512,
            "max_seq_len": 256,
        }
        base.update(overrides)
        cfg = cls(**base)
        if cfg.engram_layer_ids and cfg.engram_num_embeddings == (0,) * len(cfg.engram_layer_ids):
            from mstar.model.components.deepseek_v41.engram_hash import bucket_primes

            counts = tuple(
                sum(p for per_n in layer_primes for p in per_n)
                for layer_primes in bucket_primes(
                    cfg.engram_layer_ids,
                    cfg.engram_max_ngram_size,
                    cfg.engram_vocab_size,
                    cfg.engram_n_heads,
                )
            )
            cfg = dataclasses.replace(cfg, engram_num_embeddings=counts)
        return cfg
