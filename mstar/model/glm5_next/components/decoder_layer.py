"""GLM-5.3-Flash decoder layers: the mHC hybrid trunk layer + the plain MTP layer."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.attention import Glm5NextMLAAttention
from mstar.model.glm5_next.components.language_model import (
    build_mlp_for_layer,
    build_rmsnorm,
)
from mstar.model.glm5_next.config import KDA_STATE, LINEAR_ATTENTION, Glm5NextModelConfig
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection, update_streams


def build_hyper_connection(config: Glm5NextModelConfig) -> Glm5NextHyperConnection:
    return Glm5NextHyperConnection(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        hc_eps=config.hc_eps,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        rms_norm_eps=config.rms_norm_eps,
    )


class Glm5NextDecoderLayer(nn.Module):
    """One mHC trunk layer over streams ``(1, T, hc_mult, hidden)``."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear_attention = config.layer_types[layer_idx] == LINEAR_ATTENTION
        if self.is_linear_attention:
            self.self_attn = ParallelKimiDeltaAttention(
                config.kda_config(), comm_group=comm_group, state_key=KDA_STATE, layer=layer_idx)
        else:
            self.self_attn = Glm5NextMLAAttention(config, comm_group=comm_group, layer_idx=layer_idx)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)
        self.attn_hc = build_hyper_connection(config)
        self.ffn_hc = build_hyper_connection(config)

    def forward(self, hidden_streams: torch.Tensor, decode: bool = False) -> torch.Tensor:
        """``decode``: the step decodes or verifies, which the KDA layer's output all-reduce
        follows."""
        if fused_decode.fused_available(hidden_streams):
            hidden_streams, pending = self._forward_fused(hidden_streams, decode=decode)
            return fused_decode.update_streams(hidden_streams, *pending)
        residual = hidden_streams
        post, comb, hidden = self.attn_hc(hidden_streams)
        hidden = self.input_layernorm(hidden.squeeze(0))
        hidden = self._attend(hidden, decode)
        hidden_streams = update_streams(residual, hidden.unsqueeze(0), post, comb)

        residual = hidden_streams
        post, comb, hidden = self.ffn_hc(hidden_streams)
        hidden = self.post_attention_layernorm(hidden.squeeze(0))
        hidden = self.mlp(hidden)
        return update_streams(residual, hidden.unsqueeze(0), post, comb)

    def _attend(self, hidden: torch.Tensor, decode: bool) -> torch.Tensor:
        if self.is_linear_attention:
            return self.self_attn(hidden, decode=decode)
        return self.self_attn(hidden)

    def forward_carry(self, hidden_streams, pending=None, rows=None, decode=False):
        """``forward`` that, at prefill sizes, returns its closing ``update_streams`` as
        ``pending`` for the next layer's first mHC site to apply. ``(streams, pending)``.
        With ``rows``, only those tokens go on past the attention: the last layer's FFN
        half feeds nothing but the sampled rows."""
        fused = fused_decode.fused_available(hidden_streams)
        if pending is None and not (fused and (rows is not None
                                               or fused_decode.prefill_sized(hidden_streams))):
            out = self(hidden_streams, decode)
            return (out if rows is None else out.index_select(1, rows)), None
        return self._forward_fused(hidden_streams, pending, rows, decode)

    def _forward_fused(self, hidden_streams, pending=None, rows=None, decode=False):
        """``forward`` with each mHC site fused with its norm and with the update before it
        (``pending``, then the attention's). Returns the streams and the MLP's update."""
        post, comb, hidden, hidden_streams = self.attn_hc.forward_fused(
            hidden_streams, self.input_layernorm, update=pending)
        hidden = self._attend(hidden, decode)
        if rows is not None:
            hidden_streams, post, comb = (t.index_select(1, rows)
                                          for t in (hidden_streams, post, comb))
            hidden = hidden.index_select(0, rows)
        post, comb, hidden, hidden_streams = self.ffn_hc.forward_fused(
            hidden_streams, self.post_attention_layernorm, update=(hidden, post, comb))
        return hidden_streams, (self.mlp(hidden), post, comb)

class Glm5NextPlainDecoderLayer(nn.Module):
    """The MTP layer (45): plain-residual MLA + MoE, no mHC."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Glm5NextMLAAttention(config, comm_group=comm_group, layer_idx=layer_idx)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(
        self, hidden_states: torch.Tensor, row: int | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """With ``row``, one query per request at that row of the step's block
        (``Glm5NextMLAAttention.forward_block_row``)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if row is None:
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn.forward_block_row(hidden_states, row, latents)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
