"""Kimi-K2.7 / DeepSeek-V3 assembled text backbone (M4 assembly).

Stacks the M4 :class:`KimiDecoderLayer` blocks between a token embedding and a
final RMSNorm (:class:`KimiLanguageModel`), then wraps that with the untied LM
head (:class:`KimiForCausalLM`). This is the full text forward: token ids →
logits.

The per-layer cache-handle contract mirrors ``OrpheusLanguageModel`` exactly:
each layer is preceded by ``cache_handle.set_layer_idx(layer_idx)`` (so the paged
KV cache writes/reads the right layer slice), and the loop is followed by a
single ``cache_handle.advance_seq_lens()`` (so every request's ``seq_len`` /
``position_id_start`` steps forward once per forward pass, not once per layer).
The naive MLA reads ``position_ids`` for its YARN RoPE, so unlike Orpheus we
thread ``position_ids`` through each layer.

Lives in its own module (not ``language_model.py``) to keep the import graph
acyclic: ``decoder_layer`` imports the ``language_model`` builders, so the
assembly that imports ``decoder_layer`` must sit downstream of both.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.kimi_k2_7.components.decoder_layer import KimiDecoderLayer
from mstar.model.kimi_k2_7.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.kimi_k2_7.config import KimiK2Config


class KimiLanguageModel(nn.Module):
    """Embedding + stacked decoder layers + final norm (returns hidden states)."""

    def __init__(
        self, config: KimiK2Config, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.embed_tokens = build_embedding(config, comm_group=comm_group)
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = build_rmsnorm(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states, cache_handle, position_ids
            )
        cache_handle.advance_seq_lens()
        return self.norm(hidden_states)


class KimiForCausalLM(nn.Module):
    """Text backbone + untied LM head (returns ``[..., vocab]`` logits)."""

    def __init__(
        self, config: KimiK2Config, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = KimiLanguageModel(config, comm_group=comm_group)
        self.lm_head = build_lm_head(config, comm_group=comm_group)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, cache_handle, position_ids)
        return self.lm_head(hidden_states)

    def load_weights(self, weights, **kwargs) -> set[str]:
        """Load an HF DeepSeek-V3 checkpoint stream (M5).

        Called by the shared ``mstar.model.loader.load_weights(model, source,
        device)`` driver (mirrors ``OrpheusForCausalLM.load_weights``). Delegates
        to :func:`mstar.model.kimi_k2_7.weight_loader.load_kimi_hf_weights` for
        the Kimi remap + fused-expert stacked rules. Returns the set of loaded
        param paths.
        """
        from mstar.model.kimi_k2_7.weight_loader import load_kimi_hf_weights

        return load_kimi_hf_weights(
            self, weights, self.config.n_routed_experts,
        )
