"""A DSpark drafter's settings, from any of the checkpoint formats that ship one.

DSpark (arXiv 2607.05147) drafts a block in one parallel pass: the anchor (the target's last
token) and ``block_size - 1`` mask tokens, each row predicting the token after it, over a
context of the target's own hidden states (a few target layers, concatenated, projected by
``fc`` and normed) that the drafter keeps as KV; then a low-rank Markov head adds a bias from
each draft's predecessor, left to right (a checkpoint's confidence head, which scores each
draft for adaptive verification, goes unused).

Everything that differs between drafters is a field here, read from the checkpoint's config,
not a subclass: the backbone's geometry (a Qwen3-style decoder: GQA with per-head q/k norms,
SwiGLU or routed experts), its rope, whether the block attends causally, a sliding window,
the taps, the mask token and the heads' sizes. A drafter whose layers are its target
family's own blocks (DeepSeek-V4.1's MTP blocks, an MLA drafter for Kimi K3) names
``backbone="family"``; the family's package hands ``DSparkDrafter`` its block stack.

Taps are in the engine's convention (``DraftPolicy.taps``): the stream entering a layer.
Checkpoints count otherwise: ``target_layer_ids`` (DSpark/DFlash Qwen3, TorchSpec) name the
layer whose output is read (the stream entering the next one), DeepSeek-V4.1's
``dspark_target_layer_ids`` the layers whose input is read, and the speculators format's
``aux_hidden_state_layer_ids`` the same. vLLM 0.31 reads them alike
(``eagle3_utils.get_eagle3_aux_layers_from_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    # rows of the block it was trained on: the anchor and block_size - 1 masks, as many drafts
    block_size: int
    mask_token_id: int
    markov_rank: int
    intermediate_size: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    # HF's rope_scaling / rope_parameters beyond the default (YaRN, ...), None for plain RoPE
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 32768
    # routed experts in place of the MLP (DeepSeek-V4.1's drafter: 128, top 3)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    # the block's rows attend to each other both ways (DFlash's backbone) unless causal
    causal: bool = False
    # attention over the context through a window of this many positions (None: all)
    sliding_window: int | None = None
    # "qwen3": the generic decoder here; "family": the target family's blocks (its package
    # builds them, see the module docstring)
    backbone: str = "qwen3"
    # the format it was read from, for messages
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.block_size < 1 or self.markov_rank < 1:
            raise ValueError(f"a DSpark drafter needs block_size and markov_rank >= 1; got {self}")
        if not self.taps:
            raise ValueError("a DSpark drafter reads at least one target layer")

    @classmethod
    def from_hf(cls, hf: dict) -> DSparkConfig:
        """From a drafter checkpoint's ``config.json`` (or a target's, whose text config
        carries ``dspark_*`` keys: DeepSeek-V4.1)."""
        text = hf.get("text_config", hf)
        if "dspark_block_size" in text:
            return cls._deepseek(text, hf)
        if hf.get("speculators_model_type") == "dspark":
            return cls._speculators(hf)
        if hf.get("model_type") == "k3_dspark":
            return cls._k3_mla(hf)
        if "markov_rank" in hf:
            return cls._qwen3(hf)
        raise ValueError(f"not a DSpark drafter's config: model_type {hf.get('model_type')!r}")

    @classmethod
    def _qwen3(cls, hf: dict) -> DSparkConfig:
        """DeepSeek's Qwen3 DSpark drafters (``Qwen3DSparkModel``) and the DFlash-format ones
        (TorchSpec, a ``dflash_config``): a Qwen3 decoder; taps after the named layers."""
        dflash = hf.get("dflash_config") or {}
        layer_ids = hf.get("target_layer_ids") or dflash.get("target_layer_ids")
        if not layer_ids:
            raise ValueError("a DSpark drafter's config names no target_layer_ids")
        mask = hf.get("mask_token_id", dflash.get("mask_token_id"))
        if mask is None:
            raise ValueError("a DSpark drafter's config names no mask_token_id")
        layer_types = hf.get("layer_types") or []
        sliding = bool(hf.get("use_sliding_window")) or "sliding_attention" in layer_types
        causal = hf.get("is_causal", dflash.get("causal"))
        if causal is None:
            # vLLM's rule (qwen3_dflash._dflash_layer_causal): sliding layers attend causally
            causal = sliding
        rope = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
        theta = rope.get("rope_theta", hf.get("rope_theta", 1_000_000.0))
        scaling = None if rope.get("rope_type", "default") == "default" else dict(rope)
        draft_vocab = hf.get("draft_vocab_size") or hf["vocab_size"]
        if draft_vocab != hf["vocab_size"]:
            raise ValueError("a DSpark drafter with a reduced draft vocabulary (d2t) is not supported yet")
        return cls(
            hidden_size=hf["hidden_size"], num_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"], num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"],
            vocab_size=hf["vocab_size"], taps=tuple(int(i) + 1 for i in layer_ids),
            target_hidden_size=hf.get("target_hidden_size") or hf["hidden_size"],
            block_size=int(hf.get("block_size", 7)), mask_token_id=int(mask), markov_rank=int(hf["markov_rank"]),
            intermediate_size=hf["intermediate_size"], rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
            rope_theta=float(theta), rope_scaling=scaling,
            max_position_embeddings=hf.get("max_position_embeddings", 32768),
            causal=bool(causal),
            sliding_window=(hf.get("sliding_window") or dflash.get("swa_window_size")) if sliding else None,
            source="qwen3_dspark",
        )

    @classmethod
    def _speculators(cls, hf: dict) -> DSparkConfig:
        """The speculators format (``speculators_model_type: dspark``): a Qwen3 decoder under
        ``transformer_layer_config``; taps are the streams entering the named layers."""
        layer = dict(hf["transformer_layer_config"])
        layer.update({key: hf[key] for key in (
            "markov_rank", "mask_token_id", "block_size", "draft_vocab_size",
        ) if key in hf})
        layer["target_layer_ids"] = [i - 1 for i in hf["aux_hidden_state_layer_ids"]]
        if hf.get("target_hidden_size"):
            layer["target_hidden_size"] = hf["target_hidden_size"]
        if hf.get("sliding_window_non_causal") is False and layer.get("use_sliding_window"):
            layer["is_causal"] = True
        out = cls._qwen3(layer)
        return _replace(out, source="speculators")

    @classmethod
    def _k3_mla(cls, hf: dict) -> DSparkConfig:
        """Kimi K3 drafters on MLA layers (``k3_dspark``): K3's attention blocks, so a family
        backbone; taps after the named layers."""
        rope = hf.get("rope_parameters") or {}
        return cls(
            hidden_size=hf["hidden_size"], num_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"], num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get("qk_nope_head_dim", 128) + hf.get("qk_rope_head_dim", 64),
            vocab_size=hf["vocab_size"], taps=tuple(int(i) + 1 for i in hf["target_layer_ids"]),
            target_hidden_size=hf.get("target_hidden_size") or hf["hidden_size"],
            block_size=int(hf.get("block_size", 7)), mask_token_id=int(hf["mask_token_id"]),
            markov_rank=int(hf["markov_rank"]), intermediate_size=hf.get("intermediate_size", 0),
            rms_norm_eps=hf.get("rms_norm_eps", 1e-5),
            rope_theta=float(rope.get("rope_theta", hf.get("rope_theta", 1e4))),
            rope_scaling=dict(rope) if rope.get("rope_type", "default") != "default" else None,
            max_position_embeddings=hf.get("max_position_embeddings", 32768),
            causal=bool((hf.get("dflash_config") or {}).get("causal", hf.get("full_attention_causal", False))),
            backbone="family", source="k3_dspark",
            extra={key: hf[key] for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                                              "v_head_dim") if key in hf},
        )

    @classmethod
    def _deepseek(cls, text: dict, hf: dict) -> DSparkConfig:
        """DeepSeek-V4.1's drafter, in the target's own checkpoint (``mtp.{i}``: its MTP
        blocks, sliding-window only, ``dspark_n_routed_experts`` experts, top
        ``dspark_num_experts_per_tok``): a family backbone; taps enter the named layers."""
        return cls(
            hidden_size=text["hidden_size"], num_layers=int(text.get("num_nextn_predict_layers", 1)),
            num_attention_heads=text["num_attention_heads"], num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"], vocab_size=text["vocab_size"],
            taps=tuple(int(i) for i in text["dspark_target_layer_ids"]),
            target_hidden_size=text["hidden_size"], block_size=int(text["dspark_block_size"]),
            mask_token_id=int(text["dspark_noise_token_id"]), markov_rank=int(text["dspark_markov_rank"]),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6), rope_theta=float(text.get("rope_theta", 1e4)),
            max_position_embeddings=text.get("max_position_embeddings", 32768),
            num_experts=int(text.get("dspark_n_routed_experts", text.get("n_routed_experts", 0))),
            num_experts_per_tok=int(text.get("dspark_num_experts_per_tok", text.get("num_experts_per_tok", 0))),
            moe_intermediate_size=int(text.get("moe_intermediate_size", 0)),
            causal=False, sliding_window=text.get("sliding_window"),
            backbone="family", source=hf.get("model_type", "deepseek"),
        )


def _replace(config: DSparkConfig, **changes) -> DSparkConfig:
    from dataclasses import replace

    return replace(config, **changes)
