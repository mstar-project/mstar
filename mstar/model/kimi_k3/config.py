"""Kimi K3 configuration.

Parsed from the Hugging Face checkpoint's ``config.json`` (``KimiK3Config`` with a
``text_config`` of type ``kimi_linear``, a ``vision_config`` and a compressed-tensors
``quantization_config``). Everything the model, the submodules and the resource
declarations need is exposed here as plain dataclasses, so the rest of the package
never touches ``transformers`` config objects.

Layer index conventions (see ``KIMI_K3_ARCH_SPEC.md`` A.3):

* ``linear_attn_config.kda_layers`` / ``full_attn_layers`` in the checkpoint are
  **1-indexed**; this module converts them to 0-indexed sets once.
* ``first_k_dense_replace`` is a 0-indexed count: layers below it use the dense MLP.
* Attention-residual blocks start at ``layer_idx % attn_res_block_size == 0``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Resource keys shared by the model declaration, the submodules and the layers.
MLA_KV = "mla_kv"
MLA_ATTN = "mla_attn"
KDA_STATE = "kda_state"
POS = "pos"
SAMPLER = "sampler"


@dataclass
class KimiK3TextConfig:
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    rms_norm_eps: float = 1e-5
    # The two LoRA norms of MLA keep the KimiRMSNorm default (1e-6) in the reference.
    mla_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False

    # Dense MLP (layer 0) and activation
    intermediate_size: int = 33792
    hidden_act: str = "situ"
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0

    # MoE
    num_experts: int | None = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: int | None = 3584
    latent_moe_use_norm: bool = True
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    topk_method: str = "noaux_tc"
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1

    # MLA
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True

    # KDA (from linear_attn_config)
    kda_num_heads: int = 96
    kda_head_dim: int = 128
    kda_conv_kernel_size: int = 4
    kda_gate_lower_bound: float | None = -5.0
    kda_use_full_rank_gate: bool = True
    kda_layers: frozenset[int] = field(default_factory=frozenset)  # 0-indexed
    full_attn_layers: frozenset[int] = field(default_factory=frozenset)  # 0-indexed

    # Attention residuals
    attn_res_block_size: int | None = 12

    num_nextn_predict_layers: int = 0

    bos_token_id: int = 163584
    eos_token_id: int = 163586
    pad_token_id: int = 163839

    # ---- derived ----------------------------------------------------------------
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_kv_latent_dim(self) -> int:
        """Per-token MLA cache width: normalized latent + shared (un-rotated) rope part."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def mla_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @property
    def kda_projection_size(self) -> int:
        return self.kda_num_heads * self.kda_head_dim

    @property
    def shared_expert_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def use_latent_moe(self) -> bool:
        return self.routed_expert_hidden_size is not None

    @property
    def use_attn_res(self) -> bool:
        return self.attn_res_block_size is not None

    @property
    def num_attn_res_blocks(self) -> int:
        """Entries the residual stack can hold (embedding block + one per boundary)."""
        if not self.use_attn_res:
            return 0
        return -(-self.num_hidden_layers // self.attn_res_block_size)

    def is_kda_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.kda_layers

    def is_mla_layer(self, layer_idx: int) -> bool:
        return not self.is_kda_layer(layer_idx)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts is not None
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def is_attn_res_block_start(self, layer_idx: int) -> bool:
        return self.use_attn_res and layer_idx % self.attn_res_block_size == 0

    def attn_res_blocks_before(self, layer_idx: int) -> int:
        """Residual-stack entries valid when layer ``layer_idx`` starts (before its own
        boundary write): ``ceil(layer_idx / block_size)``."""
        if not self.use_attn_res:
            return 0
        return -(-layer_idx // self.attn_res_block_size)

    @property
    def num_kda_layers(self) -> int:
        return len(self.kda_layers)

    @property
    def num_mla_layers(self) -> int:
        return self.num_hidden_layers - self.num_kda_layers

    def mla_layer_slot(self, layer_idx: int) -> int:
        """Index of an MLA layer within the MLA KV cache (0..num_mla_layers-1)."""
        return sorted(self.full_attn_layers).index(layer_idx)

    def kda_layer_slot(self, layer_idx: int) -> int:
        """Index of a KDA layer within the recurrent-state resource."""
        return sorted(self.kda_layers).index(layer_idx)

    @classmethod
    def from_hf_dict(cls, tc: dict) -> KimiK3TextConfig:
        lac = tc.get("linear_attn_config") or {}
        n = tc["num_hidden_layers"]
        kda_1 = set(lac.get("kda_layers") or [])
        full_1 = set(lac.get("full_attn_layers") or [])
        kda0 = frozenset(i - 1 for i in kda_1)
        full0 = frozenset(i - 1 for i in full_1) if full_1 else frozenset(
            i for i in range(n) if i not in kda0
        )
        if kda0 | full0 != frozenset(range(n)) or (kda0 & full0):
            raise ValueError(
                "kda_layers and full_attn_layers must partition the layers; got "
                f"{sorted(kda0)} and {sorted(full0)} for {n} layers"
            )
        return cls(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            num_hidden_layers=n,
            num_attention_heads=tc["num_attention_heads"],
            rms_norm_eps=tc.get("rms_norm_eps", 1e-5),
            max_position_embeddings=tc.get("max_position_embeddings", 4096),
            tie_word_embeddings=tc.get("tie_word_embeddings", False),
            intermediate_size=tc["intermediate_size"],
            hidden_act=tc.get("hidden_act", "situ"),
            activation_situ_beta=tc.get("activation_situ_beta"),
            activation_situ_linear_beta=tc.get("activation_situ_linear_beta"),
            num_experts=tc.get("num_experts"),
            num_experts_per_token=tc.get("num_experts_per_token") or 0,
            num_shared_experts=tc.get("num_shared_experts", 0),
            moe_intermediate_size=tc.get("moe_intermediate_size") or 0,
            routed_expert_hidden_size=tc.get("routed_expert_hidden_size"),
            latent_moe_use_norm=tc.get("latent_moe_use_norm", False),
            moe_router_activation_func=tc.get("moe_router_activation_func", "sigmoid"),
            moe_renormalize=tc.get("moe_renormalize", True),
            routed_scaling_factor=tc.get("routed_scaling_factor", 1.0),
            topk_method=tc.get("topk_method", "noaux_tc"),
            use_grouped_topk=tc.get("use_grouped_topk", True),
            num_expert_group=tc.get("num_expert_group", 1),
            topk_group=tc.get("topk_group", 1),
            first_k_dense_replace=tc.get("first_k_dense_replace", 0),
            moe_layer_freq=tc.get("moe_layer_freq", 1),
            q_lora_rank=tc.get("q_lora_rank"),
            kv_lora_rank=tc["kv_lora_rank"],
            qk_nope_head_dim=tc["qk_nope_head_dim"],
            qk_rope_head_dim=tc["qk_rope_head_dim"],
            v_head_dim=tc["v_head_dim"],
            mla_use_nope=tc.get("mla_use_nope", False),
            mla_use_output_gate=tc.get("mla_use_output_gate", False),
            kda_num_heads=lac.get("num_heads", tc["num_attention_heads"]),
            kda_head_dim=lac.get("head_dim", 128),
            kda_conv_kernel_size=lac.get("short_conv_kernel_size", 4),
            kda_gate_lower_bound=lac.get("gate_lower_bound"),
            kda_use_full_rank_gate=lac.get("use_full_rank_gate", False),
            kda_layers=kda0,
            full_attn_layers=full0,
            attn_res_block_size=tc.get("attn_res_block_size"),
            num_nextn_predict_layers=tc.get("num_nextn_predict_layers", 0),
            bos_token_id=tc.get("bos_token_id", 163584),
            eos_token_id=tc.get("eos_token_id", 163586),
            pad_token_id=tc.get("pad_token_id", 163839),
        )


@dataclass
class KimiK3VisionConfig:
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    vt_num_attention_heads: int = 12
    vt_num_hidden_layers: int = 27
    vt_hidden_size: int = 1024
    vt_intermediate_size: int = 4096
    qkv_hidden_size: int = 1536
    merge_kernel_size: tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    mm_projector_type: str = "patchmergerv2"
    projector_ln_eps: float = 1e-5
    text_hidden_size: int = 7168
    activation_func: str = "gelu_pytorch_tanh"

    @classmethod
    def from_hf_dict(cls, vc: dict, text_hidden_size: int) -> KimiK3VisionConfig:
        mk = vc.get("merge_kernel_size", (2, 2))
        return cls(
            patch_size=vc.get("patch_size", 14),
            init_pos_emb_height=vc.get("init_pos_emb_height", 64),
            init_pos_emb_width=vc.get("init_pos_emb_width", 64),
            init_pos_emb_time=vc.get("init_pos_emb_time", 4),
            vt_num_attention_heads=vc.get("vt_num_attention_heads", 12),
            vt_num_hidden_layers=vc.get("vt_num_hidden_layers", 27),
            vt_hidden_size=vc.get("vt_hidden_size", 1024),
            vt_intermediate_size=vc.get("vt_intermediate_size", 4096),
            qkv_hidden_size=vc.get("qkv_hidden_size", 1536),
            merge_kernel_size=(int(mk[0]), int(mk[1])),
            merge_type=vc.get("merge_type", "sd2_tpool"),
            mm_projector_type=vc.get("mm_projector_type", "patchmergerv2"),
            projector_ln_eps=vc.get("projector_ln_eps", 1e-5),
            text_hidden_size=text_hidden_size,
            activation_func=vc.get("activation_func", "gelu_pytorch_tanh"),
        )


@dataclass
class KimiK3QuantConfig:
    """compressed-tensors ``mxfp4-pack-quantized`` description. In the shipped checkpoints
    only the routed experts' ``w1/w2/w3`` carry ``weight_packed``/``weight_scale``; the
    loader keys off those tensor names rather than the ignore regexes."""
    quant_method: str = "compressed-tensors"
    format: str = "mxfp4-pack-quantized"
    num_bits: int = 4
    group_size: int = 32
    scale_dtype: str = "torch.uint8"
    ignore: tuple[str, ...] = ()

    @property
    def is_mxfp4(self) -> bool:
        return self.format == "mxfp4-pack-quantized" and self.num_bits == 4

    @classmethod
    def from_hf_dict(cls, qc: dict | None) -> KimiK3QuantConfig | None:
        if not qc:
            return None
        groups = qc.get("config_groups") or {}
        g0 = next(iter(groups.values()), {}) if groups else {}
        w = g0.get("weights") or {}
        return cls(
            quant_method=qc.get("quant_method", "compressed-tensors"),
            format=qc.get("format", ""),
            num_bits=int(w.get("num_bits", 4)),
            group_size=int(w.get("group_size", 32)),
            scale_dtype=str(w.get("scale_dtype", "torch.uint8")),
            ignore=tuple(qc.get("ignore") or ()),
        )


@dataclass
class KimiK3Config:
    text: KimiK3TextConfig
    vision: KimiK3VisionConfig | None
    quant: KimiK3QuantConfig | None
    model_type: str = "kimi_k3"
    media_placeholder_token_id: int = 163605
    image_placeholder: str = "<|kimi_image_placeholder|>"
    # generation defaults (per request overridable)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    max_output_tokens: int = 2048
    # <|end_of_msg|> (163586) and [EOT] (163593); the model may extend this from the tokenizer
    stop_token_ids: frozenset[int] = frozenset({163586, 163593})

    @property
    def stop_token_id(self) -> int:
        return self.text.eos_token_id

    @classmethod
    def from_hf_dir(cls, path: str | Path) -> KimiK3Config:
        path = Path(path)
        with open(path / "config.json") as f:
            raw = json.load(f)
        return cls.from_hf_dict(raw, path)

    @classmethod
    def from_hf_dict(cls, raw: dict, path: Path | None = None) -> KimiK3Config:
        tc = raw.get("text_config") or raw
        text = KimiK3TextConfig.from_hf_dict(tc)
        vc = raw.get("vision_config")
        vision = KimiK3VisionConfig.from_hf_dict(vc, text.hidden_size) if vc else None
        quant = KimiK3QuantConfig.from_hf_dict(
            raw.get("quantization_config") or tc.get("quantization_config")
        )
        cfg = cls(
            text=text,
            vision=vision,
            quant=quant,
            model_type=raw.get("model_type", "kimi_k3"),
            media_placeholder_token_id=raw.get("media_placeholder_token_id", 163605),
            image_placeholder=raw.get("image_placeholder", "<|kimi_image_placeholder|>"),
        )
        gen = None
        if path is not None and (path / "generation_config.json").exists():
            with open(path / "generation_config.json") as f:
                gen = json.load(f)
        if gen:
            eos = gen.get("eos_token_id")
            if isinstance(eos, int):
                cfg.text.eos_token_id = eos
            elif isinstance(eos, list) and eos:
                cfg.text.eos_token_id = int(eos[0])
        return cfg
