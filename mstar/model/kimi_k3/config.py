"""Kimi K3 / K3.1 text model: architecture and serving config.

One config covers both checkpoints: K3.1 is K3 with a DSA indexer on its MLA layers
(``dsa``). ``from_hf_config`` reads the checkpoint's ``config.json``; the defaults are
K3.1 Pro's geometry. Layer lists in ``config.json`` are 1-based (``is_kda_layer(i)`` is
``i + 1 in kda_layers``); here they are 0-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Resource keys the model declares and the layers bind; a serving YAML tunes them under
# ``resources:``.
KV_CACHE = "kv_cache"      # the MLA layers' 576-wide latent pages
ATTN = "attn"              # absorbed MLA over KV_CACHE
DSA = "dsa"                # K3.1's sparse attention over KV_CACHE (dsa_long_context only)
SAMPLER = "sampler"
KDA_STATE = "kda_state"    # recurrent pool: conv and delta-rule state of the KDA layers
KDA = "kda"                # KDA linear attention over KDA_STATE
LABEL = "main"             # the one KV stream label

# the reference indexer's k_norm: LayerNorm(head_dim) with a bias, eps 1e-6 (DeepSeek-V3.2's)
INDEXER_K_NORM_EPS = 1e-6

_K31_FULL_ATTN = tuple(range(3, 92, 4)) + (92,)
# stop tokens: config.json's eos (<|end_of_msg|>) united with the tokenizer's [EOS] and [EOT]
_STOP_IDS = (163586, 163585, 163593)
# K3's default context bound (dense MLA serves any length; the KV pool sizes it)
_K3_MAX_SEQ_LEN = 32768


def _kda_layers(num_layers: int, full_attn: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i in range(num_layers) if i not in set(full_attn))


@dataclass
class KimiK3ModelConfig:
    # --- backbone ---
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # --- layer schedule (0-based) ---
    full_attn_layers: tuple[int, ...] = _K31_FULL_ATTN
    kda_layers: tuple[int, ...] = field(default_factory=lambda: _kda_layers(93, _K31_FULL_ATTN))
    first_k_dense_replace: int = 1

    # --- KDA (linear_attn_config) ---
    linear_num_heads: int = 96
    linear_head_dim: int = 128
    linear_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    # Kimi K3's output gate is one full-rank g_proj
    use_full_rank_gate: bool = True

    # --- gated MLA ---
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    # Kimi's NoPE keeps the decoupled dims, unrotated: the cache row is kv_lora_rank + 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    # absorbed MLA over the latent cache; False is the naive per-head parity path
    mla_absorb: bool = True

    # --- DSA (K3.1's dsa_config; K3 has none) ---
    dsa: bool = True
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    # Engine half of DSA (opt-in). Off: contexts are held to index_topk, where dense MLA is exactly
    # what DSA computes (top-k of at most k keys is every key), and the indexer is neither built
    # nor loaded. On: the DSA resource picks each query's top index_topk keys past it, prefill and
    # decode rows alike (eager: its selection runs per request on the host).
    dsa_long_context: bool = False

    # --- MoE ---
    intermediate_size: int = 33792          # layer 0's dense MLP
    num_experts: int = 896
    num_experts_per_tok: int = 16
    moe_intermediate_size: int = 3072
    num_shared_experts: int = 2
    # routed experts run in this latent, between routed_expert_down_proj and _up_proj
    routed_expert_hidden_size: int = 3584
    latent_moe_use_norm: bool = True
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    # expert parallelism for the routed experts (every rank owns num_experts / tp whole experts);
    # attention, the dense layers and the shared experts stay tensor parallel
    moe_expert_parallel: bool = False

    # --- SiTU-GLU ---
    situ_beta: float = 4.0
    situ_linear_beta: float | None = 25.0

    # --- block attention residuals ---
    attn_res_block_size: int = 12

    # --- lengths ---
    max_position_embeddings: int = 1_048_576
    # A request's context bound (prompt + generated); with DSA off it is held to index_topk.
    max_seq_len: int = 2048

    # --- tokens / generation defaults ---
    # <|end_of_msg|> ends a chat turn (config.json's eos); [EOS] and [EOT] are the tokenizer's
    eos_token_ids: tuple[int, ...] = _STOP_IDS
    bos_token_id: int = 163584
    pad_token_id: int = 163839
    max_output_tokens: int | None = None
    # the model card's sampling defaults
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    # --- quantization ---
    # How the checkpoint stores the routed experts: None (bf16) or "mxfp4" (compressed-tensors
    # mxfp4-pack-quantized, e8m0 scale per 32 along K). Informational here: the quantization
    # API reads the checkpoint's headers and the serving config picks the runtime format.
    experts_checkpoint_format: str | None = None

    def __post_init__(self) -> None:
        self.full_attn_layers = tuple(self.full_attn_layers)
        self.kda_layers = tuple(self.kda_layers)
        self.eos_token_ids = tuple(self.eos_token_ids)
        self._validate()

    def _validate(self) -> None:
        n = self.num_hidden_layers
        if sorted(self.full_attn_layers + self.kda_layers) != list(range(n)):
            raise ValueError(
                f"kda_layers {self.kda_layers} and full_attn_layers {self.full_attn_layers} must "
                f"partition the {n} layers"
            )
        if not self.kda_layers:
            raise ValueError("Kimi K3 needs at least one KDA layer")
        if not self.mla_use_nope:
            raise ValueError("Kimi K3's MLA is NoPE (mla_use_nope); a rotary MLA is not ported")
        if self.dsa_long_context and not self.dsa:
            raise ValueError("dsa_long_context needs a DSA checkpoint (K3.1's dsa_config)")
        if self.dsa and not self.dsa_long_context and self.max_seq_len > self.index_topk:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} > index_topk={self.index_topk} with "
                "dsa_long_context off: past index_topk dense MLA is not DSA"
            )
        if self.num_attention_heads % 1 or self.linear_conv_kernel_size < 2:
            raise ValueError(f"linear_conv_kernel_size={self.linear_conv_kernel_size} must be >= 2")
        if self.attn_res_block_size < 1:
            raise ValueError(f"attn_res_block_size={self.attn_res_block_size} must be >= 1")

    # --- schedule ---

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.full_attn_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace

    def is_attn_res_block_start(self, layer_idx: int) -> bool:
        """A layer that opens a residual block pushes its input onto the stack."""
        return layer_idx % self.attn_res_block_size == 0

    @property
    def num_attn_res_blocks(self) -> int:
        """Stack entries at the output read: one per block start."""
        return -(-self.num_hidden_layers // self.attn_res_block_size)

    # --- dims ---

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @property
    def mla_cache_kpe(self) -> int:
        """The cache row's slot after the latent: the decoupled key, unrotated."""
        return self.qk_rope_head_dim

    @property
    def cache_latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def padded_head_dim(self) -> int:
        """qk_head_dim rounded up to a paged-kernel head dim, for the naive path."""
        for supported in (64, 128, 256):
            if supported >= self.qk_head_dim:
                return supported
        raise ValueError(f"qk_head_dim={self.qk_head_dim} exceeds 256")

    @property
    def context_limit(self) -> int:
        return self.max_seq_len

    def kda_config(self):
        """The KDA layers' ``KDAConfig``, all heads (the layer divides them by TP)."""
        # imported here: the layer module pulls in the engine, which the config must not
        from mstar.model.components.kda import KDAConfig

        return KDAConfig(
            hidden_size=self.hidden_size,
            num_heads=self.linear_num_heads,
            head_dim=self.linear_head_dim,
            conv_kernel_size=self.linear_conv_kernel_size,
            gate_lower_bound=self.gate_lower_bound,
            rms_norm_eps=self.rms_norm_eps,
            full_rank_output_gate=self.use_full_rank_gate,
        )

    def situ(self):
        from mstar.utils.situ import SiTU

        return SiTU(beta=self.situ_beta, linear_beta=self.situ_linear_beta)

    # --- config.json ---

    @classmethod
    def from_hf_config(cls, hf_config: dict, strict: bool = True, **overrides) -> KimiK3ModelConfig:
        """Build from the checkpoint's ``config.json`` (the outer multimodal one or its
        ``text_config``). ``strict`` pins K3 / K3.1 Pro's geometry; ``overrides`` set serving
        fields (``max_seq_len``, ``dsa_long_context``, ...)."""
        text = hf_config.get("text_config", hf_config)
        if text.get("model_type") not in (None, "kimi_linear"):
            raise ValueError(f"not a Kimi Linear text config: model_type={text.get('model_type')!r}")
        linear = text["linear_attn_config"]
        n = int(text["num_hidden_layers"])
        kda = tuple(i - 1 for i in linear["kda_layers"])
        full = tuple(i - 1 for i in linear["full_attn_layers"])
        if text.get("hidden_act") != "situ":
            raise ValueError(f"hidden_act={text.get('hidden_act')!r}: only SiTU-GLU is ported")
        if text.get("moe_router_activation_func", "sigmoid") != "sigmoid" or text.get(
                "topk_method", "noaux_tc") != "noaux_tc":
            raise ValueError("only the sigmoid noaux_tc router is ported")
        if int(text.get("num_expert_group", 1)) != 1 or int(text.get("topk_group", 1)) != 1:
            raise ValueError("expert-group routing is not ported (num_expert_group/topk_group != 1)")
        if int(text.get("moe_layer_freq", 1)) != 1:
            raise ValueError("moe_layer_freq != 1 is not ported")
        if int(text.get("num_nextn_predict_layers", 0)) != 0:
            raise ValueError("MTP layers are not ported")
        if text.get("q_lora_rank") is None:
            raise ValueError("an MLA without q_lora_rank is not ported")

        dsa_cfg = text.get("dsa_config")
        dsa_fields: dict = {"dsa": dsa_cfg is not None}
        if dsa_cfg is not None:
            if tuple(i - 1 for i in dsa_cfg["dsa_layers"]) != full:
                raise ValueError("dsa_config.dsa_layers must be the full-attention layers")
            if int(dsa_cfg.get("index_rope_dim", 0)) != 0:
                raise ValueError("a rotary DSA indexer (index_rope_dim > 0) is not ported")
            if dsa_cfg.get("index_k_norm_type", "layernorm") != "layernorm":
                raise ValueError(f"index_k_norm_type={dsa_cfg.get('index_k_norm_type')!r} is not ported")
            if int(dsa_cfg.get("index_topk_freq", 1)) != 1:
                # every DSA layer runs its own indexer; shared (IndexShare) layers are glm52's
                raise ValueError("index_topk_freq != 1 (shared indexer layers) is not ported")
            dsa_fields.update(
                index_n_heads=int(dsa_cfg["index_num_heads"]),
                index_head_dim=int(dsa_cfg["index_head_dim"]),
                index_topk=int(dsa_cfg["index_topk"]),
            )

        eos = text.get("eos_token_id", hf_config.get("eos_token_id"))
        eos_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
        fields = dict(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=n,
            num_attention_heads=int(text["num_attention_heads"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            full_attn_layers=full,
            kda_layers=kda,
            first_k_dense_replace=int(text["first_k_dense_replace"]),
            linear_num_heads=int(linear["num_heads"]),
            linear_head_dim=int(linear["head_dim"]),
            linear_conv_kernel_size=int(linear["short_conv_kernel_size"]),
            gate_lower_bound=float(linear["gate_lower_bound"]),
            use_full_rank_gate=bool(linear.get("use_full_rank_gate", False)),
            q_lora_rank=int(text["q_lora_rank"]),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            mla_use_nope=bool(text.get("mla_use_nope", False)),
            mla_use_output_gate=bool(text.get("mla_use_output_gate", False)),
            intermediate_size=int(text["intermediate_size"]),
            num_experts=int(text["num_experts"]),
            num_experts_per_tok=int(text["num_experts_per_token"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            num_shared_experts=int(text.get("num_shared_experts") or 0),
            routed_expert_hidden_size=int(text.get("routed_expert_hidden_size") or text["hidden_size"]),
            latent_moe_use_norm=bool(text.get("latent_moe_use_norm", False)),
            moe_renormalize=bool(text["moe_renormalize"]),
            routed_scaling_factor=float(text["routed_scaling_factor"]),
            situ_beta=float(text.get("activation_situ_beta") or 1.0),
            situ_linear_beta=(None if text.get("activation_situ_linear_beta") is None
                              else float(text["activation_situ_linear_beta"])),
            attn_res_block_size=int(text["attn_res_block_size"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            eos_token_ids=tuple(dict.fromkeys(eos_ids + _STOP_IDS)),
            bos_token_id=int(text.get("bos_token_id", 163584)),
            pad_token_id=int(text.get("pad_token_id", 163839)),
            experts_checkpoint_format=_experts_format(text.get("quantization_config")
                                                      or hf_config.get("quantization_config")),
            **dsa_fields,
        )
        if fields["dsa"] and not overrides.get("dsa_long_context", False):
            # within index_topk unless the DSA resource serves past it
            fields["max_seq_len"] = min(overrides.pop("max_seq_len", fields["index_topk"]),
                                        fields["index_topk"])
        elif not fields["dsa"]:
            fields["max_seq_len"] = _K3_MAX_SEQ_LEN
        fields.update(overrides)
        config = cls(**fields)
        if strict:
            config._validate_kimi_k3()
        return config

    def _validate_kimi_k3(self) -> None:
        """Pin K3 / K3.1 Pro's geometry (a strict parse)."""
        expected = {
            "num_hidden_layers": 93, "hidden_size": 7168, "vocab_size": 163840,
            "num_attention_heads": 96, "linear_num_heads": 96, "linear_head_dim": 128,
            "linear_conv_kernel_size": 4, "gate_lower_bound": -5.0, "use_full_rank_gate": True,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64, "v_head_dim": 128, "mla_use_output_gate": True,
            "first_k_dense_replace": 1, "intermediate_size": 33792, "num_experts": 896,
            "num_experts_per_tok": 16, "moe_intermediate_size": 3072, "num_shared_experts": 2,
            "routed_expert_hidden_size": 3584, "latent_moe_use_norm": True,
            "attn_res_block_size": 12, "full_attn_layers": _K31_FULL_ATTN,
        }
        if self.dsa:
            expected.update(index_n_heads=32, index_head_dim=128, index_topk=2048)
        bad = {k: (getattr(self, k), v) for k, v in expected.items() if getattr(self, k) != v}
        if bad:
            raise ValueError(f"config.json does not match Kimi K3 / K3.1 Pro (got, want): {bad}; "
                             "pass strict=False if intended")

    @classmethod
    def k31(cls, **overrides) -> KimiK3ModelConfig:
        """K3.1 Pro's geometry without its config.json (dummy mode, the conductor)."""
        return cls(experts_checkpoint_format="mxfp4", **overrides)

    @classmethod
    def k3(cls, **overrides) -> KimiK3ModelConfig:
        """K3's: K3.1 without the indexer."""
        overrides.setdefault("max_seq_len", _K3_MAX_SEQ_LEN)
        return cls(dsa=False, experts_checkpoint_format="mxfp4", **overrides)

    @classmethod
    def reduced(cls, dsa: bool = True, **overrides) -> KimiK3ModelConfig:
        """Five tiny layers with every kind in them, for CPU tests: KDA + dense MLP, two KDA +
        MoE, then two MLA (+ DSA) + MoE like the checkpoint's tail; residual blocks of 2, so the
        stack has entries from layers 0, 2 and 4."""
        fields = dict(
            vocab_size=256, hidden_size=64, num_hidden_layers=5, num_attention_heads=4,
            full_attn_layers=(3, 4), kda_layers=(0, 1, 2), first_k_dense_replace=1,
            linear_num_heads=4, linear_head_dim=16, linear_conv_kernel_size=4,
            q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8,
            v_head_dim=16, dsa=dsa, index_n_heads=4, index_head_dim=16, index_topk=64,
            intermediate_size=96, num_experts=8, num_experts_per_tok=2,
            moe_intermediate_size=32, num_shared_experts=2, routed_expert_hidden_size=48,
            attn_res_block_size=2, max_position_embeddings=512, max_seq_len=64,
            eos_token_ids=(250, 251), bos_token_id=252, pad_token_id=255,
        )
        fields.update(overrides)
        return cls(**fields)


def _experts_format(quant: dict | None) -> str | None:
    """The routed experts' stored format from a compressed-tensors ``quantization_config``."""
    if not quant:
        return None
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError(f"quantization_config.quant_method={quant.get('quant_method')!r} is not "
                         "Kimi K3's (compressed-tensors)")
    if quant.get("format") != "mxfp4-pack-quantized":
        raise ValueError(f"quantization format {quant.get('format')!r} is not mxfp4-pack-quantized")
    for group in quant.get("config_groups", {}).values():
        weights = group.get("weights") or {}
        if (weights.get("num_bits"), weights.get("group_size"), weights.get("type")) != (4, 32, "float"):
            raise ValueError(f"unexpected MXFP4 weight scheme {weights}")
    return "mxfp4"
