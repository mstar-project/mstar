"""PaliGemma transformer expert for Pi0.5 (prefix processing).

A Gemma-style transformer that writes the paged KV cache through the node's
KV/attention/position resources. Used for the prefill graph walk where it
processes the prefix tokens (image + language + state) and writes the KV cache
that the action expert later reads during action generation.

Composed entirely from ``mstar.model.components`` — Gemma RMSNorm
(``gemma_mode=True``), GELU-tanh ``ParallelGatedMLP``, and a standard
``ParallelAttention`` block (with a trivial single-rank comm group for
the non-TP case, so the same code runs for TP=1 and TP>1).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.pi05.config import LLM_ATTN, LLM_KV, LLM_POS, Pi05Config


def _build_paligemma_layer(
    config: Pi05Config,
    input_hidden_size: int | None = None,
    intermediate_size: int | None = None,
) -> DecoderLayer:
    """One Gemma decoder layer for PaliGemma's prefix expert.

    ``input_hidden_size`` and ``intermediate_size`` are overridable so the
    same layer construction can be reused by the action expert (which has
    a different width but shares K/V dims with PaliGemma).
    """
    h = input_hidden_size if input_hidden_size is not None else config.hidden_size
    inter = intermediate_size if intermediate_size is not None else config.pali_intermediate_size
    return DecoderLayer(
        self_attn=ParallelAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            input_hidden_size=h,
            rope_theta=config.rope_theta,
            attn_key=LLM_ATTN,
            kv_key=LLM_KV,
            pos_key=LLM_POS,
        ),
        mlp=ParallelGatedMLP(
            hidden_size=h,
            intermediate_size=inter,
            activation="gelu_tanh",
        ),
        input_layernorm=RMSNorm(h, eps=config.rms_norm_eps, gemma_mode=True),
        post_attention_layernorm=RMSNorm(h, eps=config.rms_norm_eps, gemma_mode=True),
    )


class Pi05PaliGemmaExpert(nn.Module):
    """Stack of PaliGemma transformer layers.

    The submodule's input embeddings (image tokens + language tokens +
    state tokens) are passed in directly. This module owns only the
    transformer blocks plus a final Gemma RMSNorm; the embedding table is
    held by the parent submodule and shared with the action expert.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_build_paligemma_layer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_mode=True)

    def forward(
        self,
        query_sequence: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        # The label and layer index are cursors on the shared resources: bind
        # the label once, advance the index per layer. Passing them as
        # arguments instead would make inductor specialize on the int.
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            query_sequence = layer(hidden_states=query_sequence)

        # `write_cache` and the advance that followed it are the step
        # declaration's now: the runner commits the step after the forward.
        return self.norm(query_sequence)
