"""Action expert transformer with adaRMS timestep conditioning.

The action expert is a Gemma-style transformer that processes action
suffix tokens during the flow-matching loop. It shares KV-cache
dimensions with the PaliGemma expert (same num_kv_heads and head_dim)
so it can attend to the prefix KV cache that PaliGemma wrote during
the same walk.

Composed from ``mstar.model.components`` — ``AdaRMSNorm`` for the
conditional norms, ``GatedDecoderLayer`` for the gated-residual block,
and the parallel-aware ``ParallelAttention`` / ``ParallelGatedMLP`` for
the inner blocks (with a trivial single-rank comm group for the non-TP
case). The model-specific pieces left here are ``Pi05TimeMLP`` (the
sincos → adaRMS conditioning MLP), ``Pi05AdaRMSNorm`` and the stack assembly.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from mstar.model.components import AdaRMSNorm, GatedDecoderLayer
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.pi05.config import LLM_KV, LLM_POS, Pi05Config


class AdaRMSModulations(NamedTuple):
    """Every adaRMS norm's modulation, one row per denoising step."""

    per_layer: list[torch.Tensor]  # per layer [num_steps, 2, 3 * hidden]: input, post-attention norm
    final: torch.Tensor  # [num_steps, 1, 3 * hidden]


class Pi05TimeMLP(nn.Module):
    """Two-layer SiLU MLP that maps the sincos timestep embedding to the
    adaRMS condition, from which every norm in the action expert computes its
    modulation.

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


class Pi05AdaRMSNorm(AdaRMSNorm):
    """``AdaRMSNorm`` reading its row of a layer's modulations.

    The condition depends only on the denoising step, so every norm's
    ``dense(cond)`` is computed for all steps at once
    (:meth:`Pi05ActionExpert.modulations`). A layer passes both of its norms
    the same ``[2, 3 * hidden]`` block; ``slot`` picks this norm's row, which
    every request in the batch shares.
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float, slot: int):
        super().__init__(hidden_size, cond_dim=cond_dim, eps=eps)
        self.slot = slot

    def forward(
        self, x: torch.Tensor, modulations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, modulations[self.slot:self.slot + 1])


def Pi05ActionExpertLayer(config: Pi05Config, layer: int) -> GatedDecoderLayer:
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
            kv_key=LLM_KV,
            pos_key=LLM_POS,
            layer=layer,
        ),
        mlp=ParallelGatedMLP(
            hidden_size=h,
            intermediate_size=config.action_intermediate_size,
            activation="gelu_tanh",
        ),
        input_layernorm=Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps, slot=0),
        post_attention_layernorm=Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps, slot=1),
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
            [Pi05ActionExpertLayer(config, layer) for layer in range(config.num_layers)]
        )
        self.norm = Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps, slot=0)

    def modulations(self, conds: torch.Tensor) -> AdaRMSModulations:
        """Every norm's modulation for each ``[num_steps, hidden]`` condition row."""
        per_layer = [
            torch.stack(
                [layer.input_layernorm.dense(conds), layer.post_attention_layernorm.dense(conds)], dim=1,
            )
            for layer in self.layers
        ]
        return AdaRMSModulations(per_layer, self.norm.dense(conds).unsqueeze(1))

    def forward(
        self, query_sequence: torch.Tensor, layer_modulations: list[torch.Tensor], final_modulation: torch.Tensor,
    ) -> torch.Tensor:
        """One step: ``layer_modulations[i]`` is layer ``i``'s ``[2, 3 * hidden]``
        block and ``final_modulation`` the final norm's ``[1, 3 * hidden]``."""
        for layer, modulation in zip(self.layers, layer_modulations, strict=True):
            query_sequence = layer(hidden_states=query_sequence, modulation=modulation)
        out, _ = self.norm(query_sequence, final_modulation)
        return out
