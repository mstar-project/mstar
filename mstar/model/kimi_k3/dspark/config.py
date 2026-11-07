"""The DSpark drafter's configuration: ``config.json`` of ``Inferact/Kimi-K3.1-Pro-DSpark-preview``
(``K3DSparkModel``, ``model_type: k3_dspark``), after PR #257's ``DSparkConfig`` for K3's drafter.

Five dense layers of DeepSeek-style MLA (rope, no NoPE trick, no output gate) and SiLU MLPs over a
projection of five target layers' hidden states; a block of ``block_size`` rows (the anchor, the
target's last token, then mask tokens) drafts ``block_size`` tokens in one pass, each row predicting
the token after it, corrected left to right by a rank-``markov_rank`` Markov head. Embedding and
``lm_head`` are the target's. What the parse refuses is what this port does not implement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class YarnParams:
    """DeepSeek-style yarn over the 64 rope dims. ``attn_factor`` multiplies the cos/sin tables as
    vLLM's ``DeepseekScalingRotaryEmbedding`` does (the release sets it to 1 / mscale(32), so the
    tables carry no scale); HF reads no such key (hypothesis: the drafter was trained and evaluated
    under vLLM's convention, its README's)."""
    factor: float = 32.0
    original_max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    attn_factor: float = 0.7426255848312643


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int = 7168
    intermediate_size: int = 14336
    num_hidden_layers: int = 5
    num_attention_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    target_hidden_size: int = 7168
    # 0-based target layers whose output stream the drafter reads (vLLM captures ids + 1)
    target_layer_ids: tuple[int, ...] = (75, 79, 83, 87, 91)
    target_num_hidden_layers: int = 93
    mask_token_id: int = 163839
    markov_rank: int = 512
    # rows of a block: the anchor and block_size - 1 masks, as many drafts
    block_size: int = 7
    # the block's rows attend causally (dflash_config.causal / full_attention_causal); #257's K3
    # drafter attended both ways
    causal: bool = True
    rope: YarnParams = field(default_factory=YarnParams)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """A context cache row: the normed compressed kv and the roped key rope part."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def context_width(self) -> int:
        return self.target_hidden_size * len(self.target_layer_ids)

    @classmethod
    def from_dir(cls, path: str | Path) -> DSparkConfig:
        return cls.from_hf(json.loads((Path(path) / "config.json").read_text()))

    @classmethod
    def from_hf(cls, raw: dict) -> DSparkConfig:
        def refuse(why: str):
            raise ValueError(f"DSpark drafter config: {why}")

        if raw.get("architectures") != ["K3DSparkModel"] or raw.get("model_type") != "k3_dspark":
            refuse(f"not a K3DSparkModel checkpoint: {raw.get('architectures')} / {raw.get('model_type')}")
        if raw.get("mla_use_nope") or raw.get("mla_use_output_gate"):
            refuse("built for the rope MLA without an output gate")
        if raw.get("num_key_value_heads", raw["num_attention_heads"]) != raw["num_attention_heads"]:
            refuse("MLA keeps a key and value per head")
        if raw.get("hidden_act", "silu") != "silu" or raw.get("attention_bias"):
            refuse(f"hidden_act {raw.get('hidden_act')!r} / attention_bias {raw.get('attention_bias')}")
        if (raw.get("use_sliding_window") or raw.get("sliding_window") is not None
                or raw.get("sliding_window_non_causal")):
            refuse("sliding-window attention is not implemented")
        if any(t != "full_attention" for t in raw.get("layer_types", ())):
            refuse(f"layer_types {raw.get('layer_types')}")
        if raw.get("markov_head_type", "vanilla") != "vanilla":
            refuse(f"markov_head_type {raw.get('markov_head_type')!r}")
        if raw.get("draft_vocab_size", raw["vocab_size"]) != raw["vocab_size"]:
            refuse("a reduced draft vocabulary is not implemented")
        if raw.get("tie_word_embeddings"):
            refuse("tied embeddings")
        if not raw.get("rope_interleave", False):
            refuse("expected GPT-J interleaved rope (rope_interleave: true)")
        rp = raw.get("rope_parameters") or {}
        if rp.get("rope_type") != "yarn":
            refuse(f"expected yarn rope parameters, got {rp}")
        dflash = raw.get("dflash_config") or {}
        causal = dflash.get("causal", raw.get("full_attention_causal"))
        if causal is None:
            refuse("names neither dflash_config.causal nor full_attention_causal")
        layer_ids = tuple(int(i) for i in raw["target_layer_ids"])
        if len(layer_ids) != raw.get("num_target_layers", len(layer_ids)) or list(layer_ids) != sorted(layer_ids):
            refuse(f"target_layer_ids {layer_ids}")
        rope = YarnParams(
            factor=float(rp["factor"]),
            original_max_position_embeddings=int(rp["original_max_position_embeddings"]),
            rope_theta=float(rp.get("rope_theta", raw.get("rope_theta"))),
            beta_fast=float(rp.get("beta_fast", 32)), beta_slow=float(rp.get("beta_slow", 1)),
            mscale=float(rp.get("mscale", 1.0)), mscale_all_dim=float(rp.get("mscale_all_dim", 0.0)),
            attn_factor=float(rp.get("attn_factor", 1.0)),
        )
        return cls(
            hidden_size=raw["hidden_size"], intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"], num_attention_heads=raw["num_attention_heads"],
            q_lora_rank=raw["q_lora_rank"], kv_lora_rank=raw["kv_lora_rank"],
            qk_nope_head_dim=raw["qk_nope_head_dim"], qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"], vocab_size=raw["vocab_size"], rms_norm_eps=raw["rms_norm_eps"],
            target_hidden_size=raw["target_hidden_size"], target_layer_ids=layer_ids,
            target_num_hidden_layers=raw.get("target_num_hidden_layers", 93),
            mask_token_id=raw["mask_token_id"], markov_rank=raw["markov_rank"],
            block_size=int(raw.get("block_size", 7)), causal=bool(causal), rope=rope,
        )

    def check_target(self, hidden_size: int, num_hidden_layers: int, vocab_size: int) -> None:
        """The target this drafter reads: its width, depth (the taps) and vocabulary."""
        if (hidden_size, vocab_size) != (self.target_hidden_size, self.vocab_size):
            raise ValueError(f"DSpark drafter for a {self.target_hidden_size}-wide, {self.vocab_size}-token "
                             f"target, not {hidden_size} / {vocab_size}")
        if num_hidden_layers != self.target_num_hidden_layers or self.target_layer_ids[-1] >= num_hidden_layers - 1:
            raise ValueError(f"DSpark drafter taps {self.target_layer_ids} of a {self.target_num_hidden_layers}-"
                             f"layer target, not {num_hidden_layers} layers")

    @classmethod
    def reduced(cls, **overrides) -> DSparkConfig:
        """A tiny drafter for CPU tests (the reduced K3.1's widths: hidden 64, vocab 256)."""
        base = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                    q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                    vocab_size=256, target_hidden_size=64, target_layer_ids=(1, 2), target_num_hidden_layers=5,
                    mask_token_id=255, markov_rank=8, block_size=4,
                    rope=YarnParams(factor=4.0, original_max_position_embeddings=32, rope_theta=10000.0,
                                    mscale=1.0, mscale_all_dim=0.0, attn_factor=0.8))
        base.update(overrides)
        return replace(cls(), **base)
