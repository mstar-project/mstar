"""Action expert transformer with adaRMS timestep conditioning.

The action expert is a Gemma-style transformer that processes action
suffix tokens during the flow-matching loop. It shares KV-cache
dimensions with the PaliGemma expert (same num_kv_heads and head_dim)
so it can attend to the prefix KV cache that PaliGemma wrote during
the prefill walk.

Composed from ``mstar.model.components`` — ``AdaRMSNorm`` for the
conditional norms, ``GatedDecoderLayer`` for the gated-residual block,
and the parallel-aware ``ParallelAttention`` / ``ParallelGatedMLP`` for
the inner blocks (with a trivial single-rank comm group for the non-TP
case). The model-specific pieces left here are ``Pi05TimeMLP`` (the
sincos → adaRMS conditioning MLP) and the stack assembly.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components import AdaRMSNorm, GatedDecoderLayer
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.pi05.config import Pi05Config


class Pi05TimeMLP(nn.Module):
    """Two-layer SiLU MLP that maps the sincos timestep embedding to the
    ``adarms_cond`` vector consumed by every norm in the action expert.

    Both layers operate in the action expert's hidden dimension (which
    may differ from PaliGemma's). Mirrors lerobot's ``time_mlp_in`` /
    ``time_mlp_out`` chain: Linear → silu → Linear → silu.

    Not a plain shared ``MLP`` because of the trailing SiLU — keeping it
    here.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.linear_in(time_emb))
        h = self.linear_out(h)
        return nn.functional.silu(h)


def Pi05ActionExpertLayer(config: Pi05Config) -> GatedDecoderLayer:
    """One action expert decoder layer.

    Operates in ``config.action_hidden_size`` (1024 for gemma_300m, 2048
    for gemma_2b). The attention's K/V dims still match PaliGemma's so
    the action expert can attend to the prefix KV cache PaliGemma wrote.
    """
    h = config.action_hidden_size
    return GatedDecoderLayer(
        self_attn=ParallelAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            input_hidden_size=h,
            rope_theta=config.rope_theta,
        ),
        mlp=ParallelGatedMLP(
            hidden_size=h,
            intermediate_size=config.action_intermediate_size,
            activation="gelu_tanh",
        ),
        input_layernorm=AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps),
        post_attention_layernorm=AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps),
    )


class Pi05ActionExpert(nn.Module):
    """Stack of action expert layers plus a final adaRMS norm.

    Operates entirely in ``config.action_hidden_size``.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        h = config.action_hidden_size
        self.layers = nn.ModuleList(
            [Pi05ActionExpertLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = layer(
                hidden_states=query_sequence,
                cache_handle=cache_handle,
                adarms_cond=adarms_cond,
            )
        out, _ = self.norm(query_sequence, adarms_cond)
        return out
