"""GR00T's VL bridge: a LayerNorm over the backbone's features, then four
self-attention blocks, before the rows are split into text and image.

The blocks are the reference's ``BasicTransformerBlock`` at its plainest
setting -- ``layer_norm``, no AdaLN, no cross-attention, no positional
embedding, no timestep -- which ``SelfAttentionTransformer`` gets by passing
none of those arguments, so the block's own defaults apply (eps 1e-5, affine).
The only piece that is ours is the attention: it calls M\\*'s ragged resource
where the reference calls diffusers' ``Attention``. The feed-forward *is*
diffusers' own, imported -- its ``net.0.proj`` and ``net.2`` are exactly the
checkpoint's keys, at exactly its shapes.

Class and parameter names are the reference's, so the checkpoint maps 1:1.
"""
from __future__ import annotations

import torch
from diffusers.models.attention import FeedForward
from torch import nn

from mstar.model.gr00t_n17.config import (
    BRIDGE_ATTN,
    Gr00tN17Config,
    Gr00tN17StackConfig,
)

# `BasicTransformerBlock`'s default, which `SelfAttentionTransformer` does not
# override, and `nn.LayerNorm`'s default for `vlln`
_NORM_EPS = 1e-5


class BridgeAttention(nn.Module):
    """Plain multi-head self-attention over one request's rows.

    Unmasked in the reference, so every row sees every other row of the same
    request; running it as one ragged segment is the same computation at
    B = 1 and keeps it independent of a request's neighbours.
    """

    def __init__(self, config: Gr00tN17StackConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        dim = config.inner_dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        # diffusers' `Attention` holds a dropout after the projection, which
        # is why the checkpoint's key is `to_out.0`; the dropout is inert at
        # eval, so only the linear is here
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True)])
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``."""
        self.ragged_attn = resources.get(BRIDGE_ATTN)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ragged_attn is None:
            raise RuntimeError(
                "GR00T's VL bridge has no ragged attention resource; declare "
                f"a RaggedAttentionSpec under {BRIDGE_ATTN!r} for the "
                "vl_bridge node"
            )
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, self.num_heads, self.head_dim)
        out = self.ragged_attn.run(
            self.to_q(hidden_states).view(shape),
            self.to_k(hidden_states).view(shape),
            self.to_v(hidden_states).view(shape),
        )
        return self.to_out[0](out.reshape(num_tokens, -1))


class BasicTransformerBlock(nn.Module):
    """``norm1 -> attn -> +x -> norm3 -> ff -> +x``. There is no ``norm2``
    because this setting has no cross-attention, and the checkpoint agrees."""

    def __init__(self, config: Gr00tN17StackConfig):
        super().__init__()
        dim = config.inner_dim
        self.norm1 = nn.LayerNorm(dim, eps=_NORM_EPS)
        self.attn1 = BridgeAttention(config)
        self.norm3 = nn.LayerNorm(dim, eps=_NORM_EPS)
        self.ff = FeedForward(
            dim, activation_fn="gelu-approximate", final_dropout=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class SelfAttentionTransformer(nn.Module):
    def __init__(self, config: Gr00tN17StackConfig):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(config) for _ in range(config.num_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class Gr00tN17VLBridge(nn.Module):
    """The backbone's features in, the action head's conditioning rows out."""

    def __init__(self, config: Gr00tN17Config):
        super().__init__()
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim, eps=_NORM_EPS)
        self.vl_self_attention = SelfAttentionTransformer(config.bridge)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))
