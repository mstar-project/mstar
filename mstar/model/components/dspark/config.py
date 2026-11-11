"""A DSpark drafter's settings, read from its checkpoint's ``config.json``.

DSpark (arXiv 2607.05147) drafts a block in one parallel pass: the anchor (the
target's last token) and ``k`` mask tokens, row ``i`` predicting draft
``i + 1``, over a context of the target's own hidden states (a few target
layers, concatenated, projected by ``fc`` and normed) that the drafter keeps as
KV. A low-rank Markov head then adds a bias from each draft's predecessor, left
to right. A checkpoint's confidence head (adaptive verification) goes unused.

Taps are in the engine's convention (`DrafterNeeds.taps`): the stream entering a
layer. DSpark's Qwen3 checkpoints name the layer whose *output* is read
(``target_layer_ids``), so tap = id + 1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    # the target layers it reads (the stream entering each) and their width
    taps: tuple[int, ...]
    target_hidden_size: int
    # mask rows it was trained to draft over, so the most drafts a step may ask of it
    block_size: int
    mask_token_id: int
    markov_rank: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    # its head is its own embedding when the checkpoint ships no lm_head
    tie_word_embeddings: bool = False
    # the checkpoint's weight dtype, as HF names it
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.block_size < 1 or self.markov_rank < 1:
            raise ValueError(f"a DSpark drafter needs block_size and markov_rank >= 1; got {self}")
        if not self.taps:
            raise ValueError("a DSpark drafter reads at least one target layer")

    @classmethod
    def from_hf(cls, hf: dict) -> DSparkConfig:
        """From the ``config.json`` of DeepSeek's Qwen3 DSpark drafters (``Qwen3DSparkModel``), a
        Qwen3 decoder. Raises ``ValueError`` on a format or option the drafter here doesn't implement."""
        if "dspark_block_size" in hf.get("text_config", hf) or hf.get("model_type") == "k3_dspark":
            raise ValueError(
                "this DSpark drafter's layers are its target family's own blocks (MLA or MTP), "
                "which M* doesn't build as a drafter yet; only Qwen3-style DSpark drafters are supported"
            )
        if "markov_rank" not in hf or "speculators_model_type" in hf or "dflash_config" in hf:
            raise ValueError(f"not a DSpark drafter's config: model_type {hf.get('model_type')!r}")
        if hf.get("is_causal"):
            raise ValueError("a causal DSpark drafter is not supported")
        layer_ids = hf.get("target_layer_ids")
        if not layer_ids:
            raise ValueError("a DSpark drafter's config names no target_layer_ids")
        mask = hf.get("mask_token_id")
        if mask is None:
            raise ValueError("a DSpark drafter's config names no mask_token_id")
        layer_types = hf.get("layer_types") or []
        if hf.get("use_sliding_window") or "sliding_attention" in layer_types:
            raise ValueError("a DSpark drafter with sliding-window layers is not supported")
        rope = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"a DSpark drafter with rope scaling {rope!r} is not supported")
        if hf.get("attention_bias"):
            raise ValueError("a DSpark drafter with attention_bias is not supported")
        if hf.get("hidden_act", "silu") != "silu":
            raise ValueError(f"a DSpark drafter with hidden_act {hf['hidden_act']!r} is not supported")
        draft_vocab = hf.get("draft_vocab_size") or hf["vocab_size"]
        if draft_vocab != hf["vocab_size"]:
            raise ValueError("a DSpark drafter with a reduced draft vocabulary (d2t) is not supported")
        block = hf.get("block_size")
        if block is None:
            raise ValueError("a DSpark drafter's config names no block_size, the most drafts it was trained on")
        return cls(
            hidden_size=hf["hidden_size"], num_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"], num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"],
            vocab_size=hf["vocab_size"], taps=tuple(int(i) + 1 for i in layer_ids),
            target_hidden_size=hf.get("target_hidden_size") or hf["hidden_size"],
            block_size=int(block), mask_token_id=int(mask), markov_rank=int(hf["markov_rank"]),
            intermediate_size=hf["intermediate_size"], rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope.get("rope_theta", hf.get("rope_theta", 1_000_000.0))),
            max_position_embeddings=hf.get("max_position_embeddings", 32768),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=str(hf.get("dtype") or hf.get("torch_dtype") or "bfloat16"),
        )
