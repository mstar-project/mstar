"""Fish S2-Pro Slow AR: the shared ``Qwen3DenseLM`` under Fish weight names.

``text_model.model.*`` is a Qwen3-4B decoder stored with Fish names (fused
``attention.wqkv``, ``feed_forward.w1/w2/w3``, ``attention_norm`` /
``ffn_norm``) and its output head tied to the 155,776-token embedding. The
model only ever emits the 4,096 semantic tokens or ``<|im_end|>``, so instead
of the full tied head it scores those 4,097 embedding rows
(``semantic_head``): the same logits a masked full head yields, at 1/38 of
the head's cost. RoPE is interleaved and rotates with a bfloat16-rounded
table, both configured on the node's position resource.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.fish_s2_pro.config import AR_ATTN, AR_KV, AR_POS, FishS2ProConfig

SLOW_AR_PREFIX = "text_model.model."

# Fish name fragment -> Qwen3DenseLM (HF) name fragment, per decoder layer.
_LAYER_RENAMES = (
    (".attention.wo.", ".self_attn.o_proj."),
    (".attention.q_norm.", ".self_attn.q_norm."),
    (".attention.k_norm.", ".self_attn.k_norm."),
    (".attention_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    (".feed_forward.w1.", ".mlp.gate_proj."),
    (".feed_forward.w3.", ".mlp.up_proj."),
    (".feed_forward.w2.", ".mlp.down_proj."),
)


def remap_slow_ar_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: FishS2ProConfig,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Fish ``text_model.model.*`` tensors -> HF Qwen3 names (prefix dropped).

    ``wqkv`` rows are ``[q heads, k heads, v heads]``; they split into the
    three projections ``LLAMA_STACKED_PARAMS`` fuses back into ``qkv_proj``.
    """
    cfg = config.slow_ar
    q_size = cfg.n_head * cfg.head_dim
    kv_size = cfg.n_local_heads * cfg.head_dim
    for name, tensor in weights:
        if not name.startswith(SLOW_AR_PREFIX):
            continue
        name = name.removeprefix(SLOW_AR_PREFIX)
        if name == "embeddings.weight":
            yield "embed_tokens.weight", tensor
            continue
        if name.endswith(".attention.wqkv.weight"):
            layer = name.removesuffix(".attention.wqkv.weight")
            q, k, v = tensor.split([q_size, kv_size, kv_size], dim=0)
            yield f"{layer}.self_attn.q_proj.weight", q
            yield f"{layer}.self_attn.k_proj.weight", k
            yield f"{layer}.self_attn.v_proj.weight", v
            continue
        name = f".{name}"
        for fish, hf in _LAYER_RENAMES:
            name = name.replace(fish, hf)
        yield name.removeprefix("."), tensor


class FishSlowAR(nn.Module):
    """Slow AR backbone plus the semantic-token head (replicated; no tensor parallelism)."""

    def __init__(self, config: FishS2ProConfig):
        super().__init__()
        self.config = config
        self.lm = Qwen3DenseLM(
            config.slow_ar.qwen3_lm_config(),
            attn_key=AR_ATTN,
            kv_key=AR_KV,
            pos_key=AR_POS,
        )
        # Never used: ``semantic_head`` replaces the tied 155,776-row head.
        self.lm.lm_head = nn.Identity()
        self.register_buffer(
            "semantic_head",
            torch.empty(config.semantic_head_size, config.slow_ar.dim),
            persistent=False,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``text_model.model.*`` into the backbone; fail on any gap."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        loaded = load_hf_weights(
            self.lm,
            remap_slow_ar_weights(weights, self.config),
            stacked_params=LLAMA_STACKED_PARAMS,
        )
        missing = sorted(set(dict(self.lm.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(f"Fish Slow AR checkpoint left {len(missing)} parameters unloaded: {missing[:5]}")
        self.build_semantic_head()
        return loaded

    def build_semantic_head(self) -> None:
        """Copy the embedding rows of the 4,096 semantic tokens and ``<|im_end|>``."""
        cfg = self.config
        embed = self.lm.embed_tokens.weight.data
        self.semantic_head = torch.cat(
            [
                embed[cfg.semantic_start_token_id : cfg.semantic_end_token_id + 1],
                embed[cfg.eos_token_id : cfg.eos_token_id + 1],
            ]
        ).contiguous()

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.embed(token_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, D)`` -> final-normed hidden states."""
        return self.lm(input_embeds, label=label)

    def semantic_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` -> ``[B, 4097]``: codes 0..4095, then ``<|im_end|>``."""
        return torch.nn.functional.linear(hidden, self.semantic_head)
