"""GR00T's DT: the action head's transformer, and the pieces around it.

32 blocks at 32 heads x 48, conditioned on the denoising timestep through
AdaLN, alternating cross-attention over the bridge's rows with self-attention
over its own 41 tokens. Attention arrives as a callable, so nothing here knows
whether it is a CPU stand-in or one of M\\*'s ragged resources.
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

# The attention when no ragged resource is bound: SDPA, the DiT scaffold's
# parity backend. Module-level so its identity is stable: a compiled region
# guards on the callables it is handed, and a fresh one per step retraces.
_SDPA_ATTEND = functools.partial(joint_attention, ragged=None)

# Every span a block can ask for. A misspelt label would reach no block and
# silently leave that span on SDPA, off the planned resource, so `DiT.forward`
# raises for one, as the engine does for an unplanned label.
_SPANS = frozenset({SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN})

# Every LayerNorm in the DT is non-affine: the checkpoint's `norm1` carries
# only its `linear`, and `norm3` carries no tensor at all.
_NORM_EPS = 1e-5

# The output norm alone is 1e-6, hardcoded in the reference (`dit.py:284`).
# No weight records an eps and bf16 parity cannot resolve one, so this is read
# off the source.
_OUT_NORM_EPS = 1e-6

# Not a config field: the checkpoint's `timestep_embedder.linear_1` is
# 1536x256, so 256 is what the weights were trained against.
_TIME_PROJ_CHANNELS = 256


class TimestepEncoder(nn.Module):
    """The denoising timestep as a conditioning vector, ``[B] -> [B, dim]``.

    Both halves are diffusers' own classes, imported as the reference does. This one flips sin
    to cos; the action encoder's sinusoid does not
    (``SinusoidalPositionalEncoding``).
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
        # cast before the embedder, as the reference does; matters in bf16
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    """Normalise, then scale and shift from the timestep, per request.

    Indexed ``[:, None]`` so one request's pair applies to all of its tokens,
    which lets a batch hold requests at different denoising steps.

    ``nn.SiLU`` here, where the action encoder writes ``x * sigmoid(x)``: the
    two are not bit-identical, so each keeps the reference's spelling.

    Written out because no argument to diffusers' ``AdaLayerNorm`` reproduces
    it: the reference keeps diffusers' ``chunk_dim=0`` order (``scale,
    shift``) but chunks along dim 1, a pairing diffusers cannot produce.
    Importing with ``chunk_dim=1`` would silently swap gain for offset.
    ``DiT``'s output head is the *other* order.
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
    """One DT block: AdaLN, attention, residual, LayerNorm, feed-forward,
    residual. A cross block's ``to_k``/``to_v`` are 2048 wide in the
    checkpoint, against the DT's own 1536."""

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
        # diffusers' dropout at `to_out.1` is inert at eval, so only the linear
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
        # a swappable kernel returning heads-first `[B, H, T, D]` would reshape
        # without complaint and scramble every row
        if out.shape != query.shape:
            raise ValueError(
                f"attention returned {list(out.shape)}, expected "
                f"{list(query.shape)} (batch, tokens, heads, head_dim)"
            )
        return self.to_out[0](out.reshape(batch, queries, -1))


def _attends_encoder(index: int) -> bool:
    """Whether block ``index`` cross-attends the bridge's rows (the even ones).

    The stack reads this both to size ``to_k``/``to_v`` and to decide what to
    hand each block, so the two cannot disagree. Which rows a cross block
    reads is ``_span_for``.
    """
    return index % 2 == 0


def _span_for(index: int, attend_text_every_n_blocks: int) -> str:
    """Which span block ``index`` attends: its own rows, the text rows, or the
    image rows.

    Every ``attend_text_every_n_blocks``-th *cross* block reads text and the
    rest read image, so at its one value, 2, text is 0, 4, 8 ... and image is
    2, 6, 10 ... (``dit.py:380-396``). Block 0 is text; starting on image is
    the easy off-by-one. This one function picks both the rows and the
    attention, so the two cannot disagree.
    """
    if not _attends_encoder(index):
        return SA_SPAN
    return VL_TEXT_SPAN if index % (2 * attend_text_every_n_blocks) == 0 else VL_IMAGE_SPAN


class ActionPositionEmbedding(nn.Module):
    """The learned table, added to the action rows and to nothing else.

    The reference adds it before the state token is concatenated
    (``gr00t_n1d7.py:408-413``), so the 40 action rows carry a position and the
    state row none. Adding it after the concat changes no shape and raises
    nothing.
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
    """The ``action_head.model`` tensors: the blocks, then a conditioned output
    norm and two projections down to ``output_dim``."""

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
                # the bridge's rows are backbone-wide; `cross_attention_dim`
                # is in no config file
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

        ``attends`` maps a span label to its attention, built once per resource
        binding. A missing label falls back to SDPA, which is how the CPU
        parity tests run; a served model always binds its ragged resources.
        The same rows as ``vl_text`` and ``vl_image`` give the reference's
        plain, unmasked ``DiT.forward``.
        """
        if attends is not None and not attends.keys() <= _SPANS:
            raise ValueError(
                f"the DT has no span {sorted(attends.keys() - _SPANS)}; it reads "
                f"{sorted(_SPANS)}. The node that binds the resources names the "
                "spans it declared in its step."
            )
        temb = self.timestep_encoder(timestep)
        # as the reference does, and load-bearing: non-contiguous rows take a
        # different GEMM path and move the output by a whole fp32 floor
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
        # `shift` first, where `AdaLayerNorm` takes `scale` first (`dit.py:331`);
        # both halves look plausible, so only parity catches a swap
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
