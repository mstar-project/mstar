"""GR00T's DT: the action head's transformer, and the pieces around it.

The DT is 32 blocks at 32 heads x 48, conditioned on the denoising timestep
through AdaLN, alternating cross-attention over the bridge's rows with
self-attention over its own 41 tokens.

Attention arrives as a callable, so nothing here knows whether it is a CPU
stand-in or one of M\\*'s ragged resources.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.gr00t_n17.config import (
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
)

# What a block attends with when no ragged resource is bound: SDPA over the
# same `[B, S, H, D]` layout, which is the parity backend the DiT scaffold
# sanctions (`diffusion-scaffold/SKILL.md:47`). Module-level, so its identity
# is stable -- a compiled region guards on the identity of the callables it is
# handed, and a fresh one per step retraces that frame every step.
_SDPA_ATTEND = functools.partial(joint_attention, ragged=None)

# Every span a block can ask for. A label outside this set reaches no block, so
# without the check below a misspelt key would leave that span on SDPA --
# bitwise identical to passing nothing, off the planned resource and off the
# kernel a captured graph recorded. The engine side raises for an unplanned
# label; this makes our side raise too.
_SPANS = frozenset({SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN})

# Every LayerNorm in the DT is non-affine, and the checkpoint says so: `norm1`
# carries only its `linear`, and `norm3` carries no tensor at all.
_NORM_EPS = 1e-5

# The output head's norm is the one exception: the reference hardcodes 1e-6
# there while the blocks take the config's 1e-5. No weight records an eps and
# the bf16 recordings cannot resolve one either, so this is a reading of
# `dit.py:284`, not something parity against the checkpoint would recover.
_OUT_NORM_EPS = 1e-6

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
    steps.

    The activation is ``nn.SiLU``, where the action encoder's swish is spelled
    ``x * sigmoid(x)``. The same function twice in one model, and the two are
    **not** bit-identical, so each has to keep its own spelling (measured: up to
    0.5 apart in bf16 on a trained slot).

    This is written out rather than imported because **no argument to
    diffusers' own ``AdaLayerNorm`` reproduces it**: that class chunks
    ``shift, scale`` along dim 1 when ``chunk_dim=1`` and ``scale, shift``
    along dim 0 otherwise (its source calls its own inconsistency "a bit
    weird"). The reference copied the class, kept the ``chunk_dim=0`` *order*
    and hardcoded the ``dim=1`` *axis* -- a pairing diffusers cannot produce,
    and its ``chunk_dim`` survives as an attribute nothing reads. Importing
    with ``chunk_dim=1`` would silently swap a learned gain for a learned
    offset. ``DiT``'s output head below is the *other* convention.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        attended = self.attn1(
            self.norm1(hidden_states, temb), encoder_hidden_states, attend,
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
        attend: Callable,
    ) -> torch.Tensor:
        source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, queries, _ = hidden_states.shape
        heads = (batch, -1, self.num_heads, self.head_dim)
        query = self.to_q(hidden_states).view(heads)
        out = attend(query, self.to_k(source).view(heads), self.to_v(source).view(heads))
        # `[B, H, T, D]` is the natural mistake for a kernel that wants heads
        # first, and it has the same element count as `[B, T, H, D]` -- so a
        # transposed return would reshape without complaint and scramble every
        # row. The callable is swappable, so the shape is checked here.
        if out.shape != query.shape:
            raise ValueError(
                f"attention returned {list(out.shape)}, expected "
                f"{list(query.shape)} (batch, tokens, heads, head_dim)"
            )
        return self.to_out[0](out.reshape(batch, queries, -1))


def _attends_encoder(index: int) -> bool:
    """Whether block ``index`` cross-attends the bridge's rows.

    The even blocks cross-attend and the odd ones attend their own tokens.
    The stack reads this once to size ``to_k``/``to_v`` and again to decide
    what to hand each block, so the two cannot disagree -- a block given
    1536-wide rows will cross-attend them perfectly happily, so nothing below
    this function would catch a mismatch.

    Not to be confused with ``attend_text_every_n_blocks``, which chooses
    *which* rows a cross block reads -- that is ``_span_for`` below.
    """
    return index % 2 == 0


def _span_for(index: int, attend_text_every_n_blocks: int) -> str:
    """Which span block ``index`` attends: its own rows, the text rows, or the
    image rows.

    The odd blocks attend themselves. The cross blocks alternate between the
    bridge's two spans every ``attend_text_every_n_blocks`` *cross* blocks, so
    at its one value, 2, the text blocks are 0, 4, 8 ... and the image blocks
    are 2, 6, 10 ... (``dit.py:380-396``). Block 0 is a text block; starting
    on an image block is the easy off-by-one.

    One function decides both which rows a block reads and which attention it
    reads them with, so the two cannot disagree.
    """
    if not _attends_encoder(index):
        return SA_SPAN
    return VL_TEXT_SPAN if index % (2 * attend_text_every_n_blocks) == 0 else VL_IMAGE_SPAN


class ActionPositionEmbedding(nn.Module):
    """The learned table, added to the action rows and to nothing else.

    The reference adds it to ``action_features`` *before* the state token is
    concatenated (``gr00t_n1d7.py:408-413``), so of the DT's 41 input rows the
    40 action rows carry a position and the state row carries none. Adding 41
    rows, or adding the table after the concat, changes no shape and raises
    nothing.

    Rows ``0:T`` are taken as a slice, which is what the reference's
    ``self.position_embedding(arange(T))`` gathers -- the same rows in the
    same order, without the index tensor.
    """

    def __init__(self, config: Gr00tN17Config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.max_seq_len, config.input_embedding_dim)
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """``[B, T, D]`` in, the same with positions added."""
        return action_features + self.weight[: action_features.shape[1]]


class DiT(nn.Module):
    r"""GR00T's action-head transformer: the ``action_head.model`` tensors.

    32 AdaLN blocks alternating cross-attention over the bridge's rows with
    self-attention over the DT's own, then a conditioned output norm and two
    projections down to ``output_dim``.

    ``attend`` is handed to every block, so swapping the attention is one
    constructor argument and nothing else moves.
    """

    def __init__(self, config: Gr00tN17Config):
        super().__init__()
        stack = config.dit
        self.attend_text_every_n_blocks = config.attend_text_every_n_blocks
        dim = stack.inner_dim
        self.timestep_encoder = TimestepEncoder(dim)
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(
                dim,
                stack.num_attention_heads,
                stack.attention_head_dim,
                # a cross block reads the bridge's rows, which are as wide as
                # the backbone's; `cross_attention_dim` is in no config file,
                # so this field and the checkpoint's 2048-wide `to_k` are the
                # only two places the width is written down
                cross_attention_dim=config.backbone_embedding_dim if _attends_encoder(i) else None,
            )
            for i in range(stack.num_layers)
        )
        self.norm_out = nn.LayerNorm(dim, _OUT_NORM_EPS, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.proj_out_2 = nn.Linear(dim, stack.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vl_text: torch.Tensor,
        vl_image: torch.Tensor,
        timestep: torch.Tensor,
        attends: Mapping[str, Callable] | None = None,
    ) -> torch.Tensor:
        """The DT's ``[B, T, dim]`` rows and the bridge's two spans to
        ``[B, T, output_dim]``, conditioned on one ``[B]`` timestep.

        ``attends`` maps a span label to the attention it is read with, built
        once per resource binding by whoever owns this stack. A label with no
        entry falls back to SDPA, which is how the CPU parity tests run; a served
        model always binds its ragged resources.

        Passing the *same* rows as ``vl_text`` and ``vl_image`` makes every
        cross block read all of them, which is the reference's plain
        ``DiT.forward``, the unmasked case.
        """
        if attends is not None and not attends.keys() <= _SPANS:
            raise ValueError(
                f"the DT has no span {sorted(attends.keys() - _SPANS)}; it reads "
                f"{sorted(_SPANS)}. The node that binds the resources names the "
                "spans it declared in its step."
            )
        temb = self.timestep_encoder(timestep)
        # the reference makes both contiguous here, and it is load-bearing:
        # non-contiguous rows take a different GEMM path and move the output by
        # a whole fp32 floor (1.14e-05, measured). Free when they already
        # are, which is the usual case.
        hidden_states = hidden_states.contiguous()
        spans = {VL_TEXT_SPAN: vl_text.contiguous(), VL_IMAGE_SPAN: vl_image.contiguous()}
        for index, block in enumerate(self.transformer_blocks):
            span = _span_for(index, self.attend_text_every_n_blocks)
            hidden_states = block(
                hidden_states,
                temb,
                spans.get(span),
                _SDPA_ATTEND if attends is None else attends.get(span, _SDPA_ATTEND),
            )
        # `shift` first, where `AdaLayerNorm` takes `scale` first: the same
        # arithmetic from a differently-ordered linear (`dit.py:331`). Both
        # halves are learned and both look plausible, so only parity catches a
        # swap. `F.silu` is the reference's spelling here and `nn.SiLU`'s is
        # the module calling it, so the two are the same function.
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
