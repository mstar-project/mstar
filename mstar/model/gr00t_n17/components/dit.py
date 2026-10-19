"""GR00T's DT: the action head's transformer, and the pieces around it.

The DT is 32 blocks at 32 heads x 48, conditioned on the denoising timestep
through AdaLN, alternating cross-attention over the bridge's rows with
self-attention over its own 41 tokens.

Attention arrives as a callable, so nothing here knows whether it is a CPU
stand-in or one of M\\*'s ragged resources (C07 supplies the second).
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

# Every LayerNorm in the DT is non-affine, and the checkpoint says so: `norm1`
# carries only its `linear`, and `norm3` carries no tensor at all.
_NORM_EPS = 1e-5

# The sinusoid's width. Not a config field: the checkpoint's
# `timestep_encoder.timestep_embedder.linear_1` is 1536x256, so 256 is what
# the weights were trained against.
_TIME_PROJ_CHANNELS = 256


class TimestepEncoder(nn.Module):
    """The denoising timestep as a conditioning vector, ``[B] -> [B, dim]``.

    Both halves are diffusers' own classes, which is what the reference
    imports too, so this is an import rather than a port -- and
    ``timestep_embedder.linear_1`` / ``linear_2`` are the checkpoint's keys.
    Note the flip: ``flip_sin_to_cos=True`` here, where the action encoder's
    own sinusoid does not flip (see ``SinusoidalPositionalEncoding``).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=_TIME_PROJ_CHANNELS,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=_TIME_PROJ_CHANNELS, time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # the reference casts the projection to the weights' dtype before the
        # embedder, which matters once the model is bf16
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """Normalise, then scale and shift from the timestep, per request.

    ``scale`` comes first out of the chunk and is applied as ``1 + scale``, and
    both are indexed ``[:, None]`` so one request's pair applies to all of its
    tokens -- which is what lets a batch hold requests at different denoising
    steps (R10).

    The activation is ``nn.SiLU``, where the action encoder's swish is spelled
    ``x * sigmoid(x)``. The same function twice in one model, and the two are
    **not** bit-identical, so each has to keep its own spelling (C06 CHK-2
    measured the difference at up to 0.5 in bf16 on a trained slot).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, _NORM_EPS, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    r"""One DT block: AdaLN, attention, residual, LayerNorm, feed-forward,
    residual.

    ``attend`` is supplied by the caller, so this module never knows whether
    it is a CPU stand-in or one of M\*'s ragged resources. The even blocks
    cross-attend the bridge's rows and the odd blocks attend their own tokens;
    the checkpoint shows which is which, because a cross block's ``to_k`` and
    ``to_v`` are ``2048`` wide against the DT's own ``1536``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int | None = None,
        attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        kv_dim = cross_attention_dim if cross_attention_dim is not None else dim
        inner = num_heads * head_dim
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = _BlockAttention(dim, kv_dim, inner, num_heads, head_dim)
        self.norm3 = nn.LayerNorm(dim, _NORM_EPS, elementwise_affine=False)
        self.ff = FeedForward(dim, activation_fn="gelu-approximate", final_dropout=True)
        self.attend = attend

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attn1(
            self.norm1(hidden_states, temb), encoder_hidden_states, self.attend,
        )
        hidden_states = attended + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class _BlockAttention(nn.Module):
    """diffusers' ``Attention`` parameter names over a caller's attention."""

    def __init__(
        self, query_dim: int, kv_dim: int, inner: int, num_heads: int, head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_k = nn.Linear(kv_dim, inner, bias=True)
        self.to_v = nn.Linear(kv_dim, inner, bias=True)
        # as in the bridge: diffusers keeps a dropout at `to_out.1`, inert at
        # eval, so only the linear is here
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim, bias=True)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attend: Callable | None,
    ) -> torch.Tensor:
        if attend is None:
            raise RuntimeError(
                "a DT block has no attention; the node supplies one when it "
                "builds the stack"
            )
        source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, queries, _ = hidden_states.shape
        heads = (batch, -1, self.num_heads, self.head_dim)
        query = self.to_q(hidden_states).view(heads)
        out = attend(query, self.to_k(source).view(heads), self.to_v(source).view(heads))
        # `[B, H, T, D]` is the natural mistake for a kernel that wants heads
        # first, and it has the same element count as `[B, T, H, D]` -- so a
        # transposed return would reshape without complaint and scramble every
        # row. C07 swaps this callable, so the shape is checked here.
        if out.shape != query.shape:
            raise ValueError(
                f"attention returned {list(out.shape)}, expected "
                f"{list(query.shape)} (batch, tokens, heads, head_dim)"
            )
        return self.to_out[0](out.reshape(batch, queries, -1))
