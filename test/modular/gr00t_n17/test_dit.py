"""GR00T N1.7's DT, checked against inline references written from NVIDIA's code: the two
timestep encodings, the action encoder (``embodiment.py``), ``AdaLayerNorm``, the block, the
stack with its output head and position table, and the cross/self and text/image span rules.
Every attention is this file's own callable, so the parity tests are ``torch.equal``.
"""

import inspect
from collections import Counter
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps

from mstar.model.gr00t_n17.components.dit import (
    _SDPA_ATTEND,
    ActionPositionEmbedding,
    AdaLayerNorm,
    BasicTransformerBlock,
    DiT,
    TimestepEncoder,
    _attends_encoder,
    _span_for,
)
from mstar.model.gr00t_n17.components.embodiment import (
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    build_action_decoder,
    build_action_encoder,
)
from mstar.model.gr00t_n17.config import (
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
    Gr00tN17StackConfig,
)

from .helpers import randomize

# Spelled out rather than imported from the modules under test: these are the numbers the
# tests below exist to catch changing there.
_TIME_PROJ = 256  # the DT's projection width, and the width the two encodings are compared at
_BASE = 10000.0  # the frequency base both encodings use
_CHECKPOINT_DIM = 1536  # `input_embedding_dim`: the width both encodings run at in the real model
_EMBED = 48  # a small DT width to fill with weights

# Denoising timesteps: both ends of a flow-matching schedule, a fraction between them, and
# two integers, since the DDPM form these encodings came from indexes discrete steps.
_TIMESTEPS = (0.0, 0.25, 1.0, 7.0, 999.0)


def _flat() -> torch.Tensor:
    """``[B]``, which is what the DT's encoder takes."""
    return torch.tensor(_TIMESTEPS)


def _rows(shape: tuple[int, int] = (1, len(_TIMESTEPS))) -> torch.Tensor:
    """``[B, T]``, which is what the action encoder's takes. The timesteps repeat to fill a
    shape bigger than the list, which also puts a repeated value in a batch."""
    count = shape[0] * shape[1]
    return torch.tensor((_TIMESTEPS * (count // len(_TIMESTEPS) + 1))[:count]).reshape(shape)


def _encoder(embedding_dim: int = _EMBED, seed: int = 0) -> TimestepEncoder:
    """Ours, filled at 0.1 so the SiLU between the two linears is not saturated. Deterministic
    in ``seed``, so two calls give two identical modules."""
    encoder = TimestepEncoder(embedding_dim)
    randomize(encoder, seed, scale=0.1)
    return encoder


# ----------------------------------------------------------------------------
# The action encoder's sinusoid against an inline reference
# ----------------------------------------------------------------------------

# Deliberate deviations: conventions a reader would expect instead, each of which would load
# the checkpoint and run, because this module has no weights.
_WRONG_DIVISOR = "half_dim - 1"  # the usual form, and the one the DT's own encoding uses
_WRONG_ORDER_COS_FIRST = "cos then sin"  # what `flip_sin_to_cos=True` does
_WRONG_ORDER_INTERLEAVED = "sin and cos interleaved"  # the RoPE-style layout
_WRONG_BASE = "base 1000"


def _reference(timesteps: torch.Tensor, embedding_dim: int, deviation: str | None = None) -> torch.Tensor:
    """``exponent = -arange(half_dim) * (log(base) / divisor)``, ``freqs = t * exponent.exp()``,
    then sines and cosines concatenated, sines first. The divisor is ``half_dim``, not the
    usual ``half_dim - 1``, and the log goes through torch, both following the reference op
    for op."""
    timesteps = timesteps.float()
    half_dim = embedding_dim // 2
    base = 1000.0 if deviation == _WRONG_BASE else _BASE
    divisor = half_dim - 1 if deviation == _WRONG_DIVISOR else half_dim
    exponent = -torch.arange(half_dim, dtype=torch.float) * (torch.log(torch.tensor(base)) / divisor)
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    sin, cos = torch.sin(freqs), torch.cos(freqs)
    if deviation == _WRONG_ORDER_COS_FIRST:
        return torch.cat([cos, sin], dim=-1)
    if deviation == _WRONG_ORDER_INTERLEAVED:
        return torch.stack([sin, cos], dim=-1).flatten(-2)
    return torch.cat([sin, cos], dim=-1)


@pytest.mark.parametrize(
    "embedding_dim",
    [_CHECKPOINT_DIM, _TIME_PROJ, 8],
    ids=["the checkpoint's width", "the compared width", "a narrow one"],
)
@pytest.mark.parametrize(
    "shape", [(1, len(_TIMESTEPS)), (len(_TIMESTEPS), 1), (3, 2)],
    ids=["one request's schedule", "one timestep each", "a batch with history"],
)
@torch.no_grad()
def test_the_sinusoid_is_the_reference_block(shape, embedding_dim):
    """Pure arithmetic, the same ops in the same order, so ``torch.equal``. The checkpoint's
    width is the case that matters: ``half_dim`` (768) is not a power of two, so
    ``log(base) / half_dim`` is inexact and the route through the arithmetic shows."""
    timesteps = _rows(shape)
    out = SinusoidalPositionalEncoding(embedding_dim)(timesteps)

    assert out.shape == (*shape, embedding_dim)
    assert torch.equal(out, _reference(timesteps, embedding_dim))


@pytest.mark.parametrize(
    "shape", [(), (len(_TIMESTEPS),), (2, 3, 4)], ids=["a scalar", "a [B] vector", "rank 3"],
)
def test_a_timestep_that_is_not_rank_2_is_refused_where_it_arrives(shape):
    """``[B]`` is the dangerous rank: it is the DT encoder's input, and this module would answer
    it with ``[B, embedding_dim]`` under the wrong convention and raise nothing. So the message
    names both modules rather than just the rank."""
    with pytest.raises(ValueError, match="TimestepEncoder is the one that takes"):
        SinusoidalPositionalEncoding(_TIME_PROJ)(torch.zeros(shape))
    assert SinusoidalPositionalEncoding(_TIME_PROJ)(_rows()).shape == (1, len(_TIMESTEPS), _TIME_PROJ)


# Properties argued from values rather than from a second copy of the formula. Each returns
# the check that tells a deviation apart first, then any further check the module also passes.
def _zero_layout(encode) -> tuple[bool, ...]:
    """At ``t = 0`` every sine is exactly 0 and every cosine exactly 1, whatever the base and
    spacing, so sines first is ``[0, ..., 0, 1, ..., 1]``; cos first and interleaved are the
    same multiset in another arrangement."""
    half_dim = _TIME_PROJ // 2
    at_zero = encode(torch.zeros(1, 3), _TIME_PROJ)
    return (torch.equal(at_zero, torch.cat([torch.zeros(1, 3, half_dim), torch.ones(1, 3, half_dim)], dim=-1)),)


def _ladder_halves(encode) -> tuple[bool, ...]:
    """Dividing by ``half_dim`` makes frequency ``j`` at width ``2h`` frequency ``2j`` at width
    ``4h``, whatever the base; ``half_dim - 1`` breaks that. Both ``half_dim`` are powers of
    two, so the ladders agree bit for bit; the ``half_dim - 1`` widths part by 1.26 on the
    sine channels."""
    timesteps = _rows()
    narrow, wide = encode(timesteps, _TIME_PROJ), encode(timesteps, 2 * _TIME_PROJ)
    half_dim = _TIME_PROJ // 2
    return (
        torch.equal(narrow[..., :half_dim], wide[..., : 2 * half_dim : 2]),
        torch.equal(narrow[..., half_dim:], wide[..., 2 * half_dim :: 2]),
    )


def _lowest_frequency(encode) -> tuple[bool, ...]:
    """The last sine at ``t = 1`` is the ladder's lowest frequency (``sin x = x`` to ten
    figures there). Measured at width 256: 1.0746e-04, where base 1000 gives 1.0554e-03, base
    100000 gives 1.0941e-05 and the ``half_dim - 1`` divisor gives 9.9999e-05; the bracket is
    2.5%. The ladder starts at a frequency of exactly 1."""
    ladder = encode(torch.ones(1, 1), _TIME_PROJ)[0, 0]
    return (
        1.05e-04 < ladder[_TIME_PROJ // 2 - 1].item() < 1.10e-04,
        bool(ladder[0] == torch.sin(torch.tensor(1.0))),
    )


_SINUSOID_DEVIATIONS = {
    "cos first, at t = 0": (_WRONG_ORDER_COS_FIRST, _zero_layout),
    "interleaved, at t = 0": (_WRONG_ORDER_INTERLEAVED, _zero_layout),
    "half_dim - 1, across two widths": (_WRONG_DIVISOR, _ladder_halves),
    "base 1000, at the lowest frequency": (_WRONG_BASE, _lowest_frequency),
    "half_dim - 1, at the lowest frequency": (_WRONG_DIVISOR, _lowest_frequency),
}


@pytest.mark.parametrize("deviation, check", list(_SINUSOID_DEVIATIONS.values()), ids=list(_SINUSOID_DEVIATIONS))
@torch.no_grad()
def test_a_sinusoid_built_any_other_way_fails_a_property_the_module_has(deviation, check):
    """The sinusoid has no parameters, so no weight can give a wrong order, divisor or base
    away; each is pinned by a value only the right convention produces."""
    assert all(check(lambda timesteps, width: SinusoidalPositionalEncoding(width)(timesteps)))
    assert not check(lambda timesteps, width: _reference(timesteps, width, deviation))[0]


@pytest.mark.parametrize("embedding_dim", [_TIME_PROJ, 8], ids=["the compared width", "a narrow one"])
@torch.no_grad()
def test_the_sinusoid_is_diffusers_ladder_unflipped_and_unshifted(embedding_dim):
    """Base, divisor and order at once, borrowed from code this file did not write: diffusers'
    sinusoid with ``flip_sin_to_cos=False`` and ``downscale_freq_shift=0``. Bit-exact only
    because ``half_dim`` is a power of two here; at 1536 diffusers' route through ``math``
    misses ours by 6.09e-05."""
    theirs = Timesteps(num_channels=embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=0)
    ours = SinusoidalPositionalEncoding(embedding_dim)(_rows())[0]

    assert torch.equal(ours, theirs(_flat()))


# ----------------------------------------------------------------------------
# `TimestepEncoder`: the DT's conditioning path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("embedding_dim", [_EMBED, 16], ids=["a small DT width", "a smaller one"])
def test_the_projection_is_256_wide_whatever_the_dt_is(embedding_dim):
    """256 is not a config field: the checkpoint's ``linear_1`` is 1536 x 256, so it must not
    track the DT's width. The projection is diffusers' flipped, shifted ladder, the other
    convention from the action encoder's, though both are 256-wide sinusoids of the same
    timestep and crossing them loads every weight and runs."""
    encoder = TimestepEncoder(embedding_dim)
    embedder = encoder.timestep_embedder

    assert {name for name, _ in encoder.named_parameters()} == {
        f"timestep_embedder.{linear}.{tensor}"
        for linear in ("linear_1", "linear_2")
        for tensor in ("weight", "bias")
    }
    assert embedder.linear_1.weight.shape == (embedding_dim, _TIME_PROJ)
    assert embedder.linear_1.bias.shape == (embedding_dim,)
    assert embedder.linear_2.weight.shape == (embedding_dim, embedding_dim)
    assert embedder.linear_2.bias.shape == (embedding_dim,)
    assert list(encoder.time_proj.parameters()) == []
    assert encoder.time_proj.num_channels == _TIME_PROJ
    # the two knobs that make this the DT's convention and not the action encoder's
    assert encoder.time_proj.flip_sin_to_cos is True
    assert encoder.time_proj.downscale_freq_shift == 1

    half_dim = _TIME_PROJ // 2
    # the flip: at timestep 0 the DT's cosines come first (the action encoder's come second)
    assert torch.equal(
        encoder.time_proj(torch.zeros(3)), torch.cat([torch.ones(3, half_dim), torch.zeros(3, half_dim)], -1)
    )
    # the shift: the DT's lowest frequency is 1 / 10000 (flipped, so the sines are the second half)
    assert encoder.time_proj(torch.ones(1))[0, -1].item() == pytest.approx(1.0 / _BASE, rel=1e-06)


@torch.no_grad()
def test_the_encoder_is_the_projection_through_two_linears_with_a_silu_between():
    """``[B] -> [B, embedding_dim]``, one conditioning vector per request. diffusers' own
    linears and activation in the same order, so ``torch.equal``; this pins the wiring."""
    encoder = _encoder()
    timesteps = _flat()
    out = encoder(timesteps)
    embedder = encoder.timestep_embedder

    assert out.shape == (len(_TIMESTEPS), _EMBED)
    assert torch.equal(out, embedder.linear_2(F.silu(embedder.linear_1(encoder.time_proj(timesteps)))))


@torch.no_grad()
def test_the_projection_is_rounded_to_the_weights_dtype_before_the_embedder():
    """diffusers builds the sinusoid in fp32 and the reference casts it to the weights' dtype
    on the way in. In bf16 that is a real rounding before the first GEMM, not what running in
    fp32 and rounding at the end gives (measured max gap 3.9e-03 on O(1) rows)."""
    bf16 = _encoder().to(torch.bfloat16)
    fp32 = _encoder()  # the same weights, unrounded, because `_encoder` is deterministic
    timesteps = _flat()
    projection = bf16.time_proj(timesteps)

    assert projection.dtype == torch.float32
    assert torch.equal(bf16(timesteps), bf16.timestep_embedder(projection.to(torch.bfloat16)))
    # the rounding moves the projection, so where it happens is visible in the output
    assert not torch.equal(projection.to(torch.bfloat16).float(), projection)
    assert not torch.equal(bf16(timesteps), fp32(timesteps).to(torch.bfloat16))


# ----------------------------------------------------------------------------
# `MultiEmbodimentActionEncoder`: the noisy action and the denoising time in
#
# Only slots 24-27 of the checkpoint's 32 embodiments are trained; the rest keep their random
# init, and `embodiment_id.json` maps several tags onto them. A real-weights test on another
# slot multiplies by noise, so the tests here use synthetic weights.
# ----------------------------------------------------------------------------

_CATEGORIES = 4
_ACTION_DIM = 5  # a different number from the width, so the two cannot be confused
_HIDDEN = 8  # the encoder's width; even, because half of the sinusoid is sines
_HORIZON = 3
# The embedded action and the sinusoid then have comparable RMS (measured 0.72 against 0.71),
# which is what makes the concatenation-order case a test of the order: the two halves are
# 2.34 apart element for element, but neither is the larger tensor.
_ACTION_SCALE = 3.0
# one id per timestep, out of order, repeated, and spanning both ends of the category axis
_CAT_IDS = (3, 0, 1, 3, 2)

# Deliberate deviations: readings of the five steps a reader could expect instead, each of
# which loads every weight and runs.
_ORDER_TAU_FIRST = "tau first"
_SWISH_BEFORE_W2 = "before W2"
_SWISH_BETWEEN = "between"
_SWISH_AFTER_W3 = "after W3"
_SWISH_NONE = "none"


def _action_encoder(seed: int = 1) -> MultiEmbodimentActionEncoder:
    """Filled at 0.1, so the swish runs where its two spellings are not the same function."""
    encoder = MultiEmbodimentActionEncoder(_CATEGORIES, _ACTION_DIM, _HIDDEN)
    randomize(encoder, seed, scale=0.1)
    return encoder


def _actions(batch: int = len(_CAT_IDS), horizon: int = _HORIZON, seed: int = 8) -> torch.Tensor:
    return _ACTION_SCALE * torch.randn(
        batch, horizon, _ACTION_DIM, generator=torch.Generator().manual_seed(seed),
    )


def _config() -> Gr00tN17Config:
    """The DT's ``output_dim`` (6), its ``inner_dim`` (3 x 4 = 12), ``hidden_size`` (9),
    ``max_action_dim`` (5) and ``input_embedding_dim`` (8) are five different numbers, so
    none can stand in for another."""
    return Gr00tN17Config(
        text=None, vision=None, bridge=None,
        dit=Gr00tN17StackConfig(
            num_layers=1, num_attention_heads=3, attention_head_dim=4, output_dim=6,
        ),
        deepstack_visual_indexes=(), backbone_embedding_dim=0,
        max_action_dim=_ACTION_DIM, hidden_size=9, input_embedding_dim=_HIDDEN,
        max_num_embodiments=_CATEGORIES,
    )


def _per_category(linear, x: torch.Tensor, cat_ids: list[int]) -> torch.Tensor:
    """``x @ W[id] + b[id]``, indexed row by row here rather than gathered, since a mis-gather
    returns a plausible shape. The matmul is ``torch.bmm`` as the module calls it: a 2-D ``@``
    differs in the last ulp (3.7e-08), and ``test_state.py`` pins ``bmm`` against it on the
    same ``CategorySpecificLinear``."""
    weights = torch.stack([linear.W[cat] for cat in cat_ids])
    biases = torch.stack([linear.b[cat] for cat in cat_ids])
    return torch.bmm(x, weights) + biases.unsqueeze(1)


def _encoder_reference(
    encoder: MultiEmbodimentActionEncoder,
    actions: torch.Tensor,
    timesteps: torch.Tensor,
    cat_ids: torch.Tensor,
    order: str = "action first",
    swish: str = _SWISH_BETWEEN,
    silu: bool = False,
) -> torch.Tensor:
    """One timestep per request spread over the horizon; the action through ``W1``; the time
    through ``_reference``; the two concatenated, action first; ``W2``; a swish; ``W3``.
    ``order``, ``swish`` and ``silu`` are the deviation knobs."""
    cats = cat_ids.tolist()
    horizon = actions.shape[1]
    # every row of a request's horizon carries that request's one denoising time
    tau = _reference(torch.stack([t.expand(horizon) for t in timesteps]), _HIDDEN)
    action = _per_category(encoder.W1, actions, cats)

    activation = F.silu if silu else (lambda x: x * torch.sigmoid(x))
    halves = [tau, action] if order == _ORDER_TAU_FIRST else [action, tau]
    mixed = torch.cat(halves, dim=-1)
    if swish == _SWISH_BEFORE_W2:
        mixed = activation(mixed)
    mixed = _per_category(encoder.W2, mixed, cats)
    if swish == _SWISH_BETWEEN:
        mixed = activation(mixed)
    out = _per_category(encoder.W3, mixed, cats)
    return activation(out) if swish == _SWISH_AFTER_W3 else out


_BATCHES = {
    "a batch across embodiments": (_CAT_IDS, _HORIZON),
    "one request": (_CAT_IDS[:1], _HORIZON),
    "the last embodiment": ((_CATEGORIES - 1,), _HORIZON),
    "one action step": (_CAT_IDS, 1),
    "a longer horizon": (_CAT_IDS[:2], 4),
}


@pytest.mark.parametrize("ids, horizon", list(_BATCHES.values()), ids=list(_BATCHES))
@torch.no_grad()
def test_the_action_encoder_is_the_reference_block(ids, horizon):
    """``[B, T, action_dim]`` and one timestep per request to ``[B, T, hidden]``, ``torch.equal``.
    The horizon varies because the timestep is spread along it and the biases broadcast over it."""
    encoder = _action_encoder()
    cat_ids = torch.tensor(ids)
    actions = _actions(len(ids), horizon)
    timesteps = _flat()[: len(ids)]
    out = encoder(actions, timesteps, cat_ids)

    assert out.shape == (len(ids), horizon, _HIDDEN)
    assert torch.equal(out, _encoder_reference(encoder, actions, timesteps, cat_ids))


# each deviation, and the floor on how far it moves the output where one was measured
_ENCODER_DEVIATIONS = {
    "the time before the action": ({"order": _ORDER_TAU_FIRST}, 0.1),
    "F.silu for the swish": ({"silu": True}, None),
    "a swish before W2": ({"swish": _SWISH_BEFORE_W2}, None),
    "a swish after W3": ({"swish": _SWISH_AFTER_W3}, None),
    "no swish at all": ({"swish": _SWISH_NONE}, None),
}


@pytest.mark.parametrize("deviation, floor", list(_ENCODER_DEVIATIONS.values()), ids=list(_ENCODER_DEVIATIONS))
@torch.no_grad()
def test_an_action_encoder_wired_any_other_way_is_a_different_answer(deviation, floor):
    """Swapped halves are a permutation of ``W2``'s input rows: nothing raises and the output
    moves 0.13 against a largest element of 0.38. ``F.silu`` differs from ``x * sigmoid(x)``
    only in floating point (1.5e-08 here), yet by 0.50 in bf16 on a trained embodiment; the
    0.1 fill keeps ``W2``'s output O(1), since at 100x the two round identically."""
    encoder = _action_encoder()
    cat_ids = torch.tensor(_CAT_IDS)
    actions, timesteps = _actions(), _flat()
    out = encoder(actions, timesteps, cat_ids)
    wrong = _encoder_reference(encoder, actions, timesteps, cat_ids, **deviation)

    assert torch.equal(out, _encoder_reference(encoder, actions, timesteps, cat_ids))
    assert not torch.equal(out, wrong)
    if floor is not None:
        # the halves are the same shape, far apart, and of one scale, so a swap is not a resize
        action = encoder.W1(actions, cat_ids)
        tau = encoder.pos_encoding(timesteps.unsqueeze(1).expand(-1, _HORIZON))
        assert action.shape == tau.shape
        assert (action - tau).abs().max() > 1.0
        assert action.pow(2).mean().sqrt() == pytest.approx(tau.pow(2).mean().sqrt(), rel=0.1)
        assert (out - wrong).abs().max() > floor * out.abs().max()


_BAD_TIMESTEPS = {
    # what the sinusoid itself takes, so a caller who spread the timestep first
    "already spread to [B, T]": torch.zeros(len(_CAT_IDS), _HORIZON),
    "one timestep for the batch": torch.tensor(0.5),
    "one timestep too few": torch.zeros(len(_CAT_IDS) - 1),
}


@pytest.mark.parametrize("timesteps", list(_BAD_TIMESTEPS.values()), ids=list(_BAD_TIMESTEPS))
def test_a_timestep_that_is_not_one_per_request_is_refused_here(timesteps):
    """This module takes ``[B]``, the sinusoid inside it ``[B, T]``. Without the guard each of
    these raises from an ``expand``, an ``unsqueeze`` or a ``torch.cat`` that never mentions the
    timestep, hence the match on the text."""
    encoder = _action_encoder()
    with pytest.raises(ValueError, match="one timestep per request"):
        encoder(_actions(), timesteps, torch.tensor(_CAT_IDS))


@torch.no_grad()
def test_the_time_encoding_is_cast_to_the_actions_dtype():
    """The sinusoid is fp32 whatever the module's dtype, so in bf16, the checkpoint's dtype, the
    concatenation would mix dtypes and ``torch.cat`` would raise; no fp32 test sees it."""
    encoder = _action_encoder().to(torch.bfloat16)
    out = encoder(_actions().to(torch.bfloat16), _flat(), torch.tensor(_CAT_IDS))

    assert encoder.pos_encoding(_rows()).dtype == torch.float32
    assert out.dtype == torch.bfloat16


def test_the_encoder_and_decoder_are_sized_from_the_config():
    """``W2`` twice the width is the concatenation in shapes. The decoder takes the DT's
    ``output_dim``, not its ``inner_dim``, which every other DT layer is; its relu and indexing
    are pinned in ``test_state.py`` on the same class."""
    config = _config()
    encoder = build_action_encoder(config)
    decoder = build_action_decoder(config)

    assert encoder.W1.W.shape == (_CATEGORIES, _ACTION_DIM, _HIDDEN)
    assert encoder.W2.W.shape == (_CATEGORIES, 2 * _HIDDEN, _HIDDEN)
    assert encoder.W3.W.shape == (_CATEGORIES, _HIDDEN, _HIDDEN)
    assert encoder.pos_encoding.embedding_dim == _HIDDEN

    assert config.dit.output_dim != config.dit.inner_dim  # the mistake the next line excludes
    assert decoder.layer1.W.shape == (_CATEGORIES, config.dit.output_dim, config.hidden_size)
    assert decoder.layer2.W.shape == (_CATEGORIES, config.hidden_size, _ACTION_DIM)


# ----------------------------------------------------------------------------
# `AdaLayerNorm` against an inline reference
# ----------------------------------------------------------------------------

# A tiny DT. Every width is a different number: the block is 12 wide, its attention 3 heads x
# 5 = 15 inside, and the bridge's rows it cross-attends are 7. In the real model the first two
# are both 1536 and only the bridge's 2048 stands apart.
_DT_DIM = 12
_DT_HEADS = 3
_DT_HEAD_DIM = 5
_DT_INNER = _DT_HEADS * _DT_HEAD_DIM
_BRIDGE_DIM = 7
# 2 requests x 4 action tokens attending 6 bridge rows: three different counts, so a query
# count standing in for a key count is a shape error and not a quiet reinterpretation.
_REQUESTS = 2
_QUERIES = 4
_BRIDGE_ROWS = 6
# every LayerNorm in the DT, including AdaLN's inner one; spelled out, not imported
_EPS = 1e-5

# Deliberate deviations: readings of the one line a reader could expect instead, each of
# which loads the checkpoint and runs.
_WRONG_SHIFT_FIRST = "shift first out of the chunk"
_WRONG_BARE_SCALE = "scale and not 1 + scale"
# `[:, None]` dropped: computable only where the request count equals the token count, where
# it silently gives token t request t's conditioning.
_WRONG_PER_TOKEN = "the pair indexed per token"
_SWISH_LONGHAND = "x * sigmoid(x) for the activation"


def _adaln_reference(
    norm: AdaLayerNorm, x: torch.Tensor, temb: torch.Tensor, deviation: str | None = None,
) -> torch.Tensor:
    """``layer_norm(x) * (1 + scale[:, None]) + shift[:, None]``, ``scale`` first out of the
    chunk and the activation ``F.silu``. The norm is written out (not affine, eps 1e-5), so
    parity pins it too; only the ``linear`` is borrowed."""
    activation = temb * torch.sigmoid(temb) if deviation == _SWISH_LONGHAND else F.silu(temb)
    first, second = norm.linear(activation).chunk(2, dim=1)
    scale, shift = (second, first) if deviation == _WRONG_SHIFT_FIRST else (first, second)
    normed = F.layer_norm(x, (x.shape[-1],), eps=_EPS)
    if deviation == _WRONG_PER_TOKEN:
        return normed * (1 + scale) + shift
    gain = scale[:, None] if deviation == _WRONG_BARE_SCALE else 1 + scale[:, None]
    return normed * gain + shift[:, None]


def _adaln(seed: int = 20) -> AdaLayerNorm:
    """Ours, filled at 0.1, which keeps the SiLU off its saturated ends."""
    norm = AdaLayerNorm(_DT_DIM)
    randomize(norm, seed, scale=0.1)
    return norm


def _temb(requests: int = _REQUESTS, seed: int = 21) -> torch.Tensor:
    """``[B, dim]``: one conditioning vector per request, O(1) so the SiLU's two spellings differ."""
    return torch.randn(requests, _DT_DIM, generator=torch.Generator().manual_seed(seed))


def _hidden(requests: int = _REQUESTS, tokens: int = _QUERIES, seed: int = 22) -> torch.Tensor:
    """``[B, T, dim]``: the DT's own action tokens."""
    return torch.randn(
        requests, tokens, _DT_DIM, generator=torch.Generator().manual_seed(seed),
    )


_ADALN_SHAPES = {
    "a batch of requests": (_REQUESTS, _QUERIES),
    "as many tokens as requests": (3, 3),
    "one request": (1, _QUERIES),
    "one token each": (3, 1),
}


@pytest.mark.parametrize("shape", list(_ADALN_SHAPES.values()), ids=list(_ADALN_SHAPES))
@torch.no_grad()
def test_adalayernorm_is_the_reference_block(shape):
    """``torch.equal``. The square shape is the only one where dropping ``[:, None]`` raises
    nothing; the random fill of every parameter is what would catch an affine inner norm."""
    requests, tokens = shape
    norm = _adaln()
    x = _hidden(requests, tokens)
    temb = _temb(requests)
    out = norm(x, temb)

    assert out.shape == (requests, tokens, _DT_DIM)
    assert torch.equal(out, _adaln_reference(norm, x, temb))


# each deviation, and the floor on how far it moves the output where one was measured
_ADALN_DEVIATIONS = {
    _WRONG_SHIFT_FIRST: 0.1,
    _WRONG_BARE_SCALE: None,
    _WRONG_PER_TOKEN: 0.1,
    _SWISH_LONGHAND: None,
}


@pytest.mark.parametrize("deviation, floor", list(_ADALN_DEVIATIONS.items()), ids=list(_ADALN_DEVIATIONS))
@torch.no_grad()
def test_adalayernorm_read_any_other_way_is_a_different_answer(deviation, floor):
    """The chunk's halves swap with no shape change (0.97 move on a 2.55 output). ``1 + scale``
    makes a zero ``linear`` the identity where bare ``scale`` zeroes the tokens. The pair is
    per request, so every token of a request given one row comes back as one row (a 1.07 move
    per token, on the square shape). ``F.silu`` here, unlike the action encoder's long hand,
    differs on 20.4% of elements by up to 2.4e-07."""
    norm = _adaln()
    if deviation == _WRONG_PER_TOKEN:
        requests = tokens = 3
        x = _hidden(1, 1).expand(requests, tokens, _DT_DIM).contiguous()
        temb = _temb(requests)
    else:
        x, temb = _hidden(), _temb()
    out = norm(x, temb)
    wrong = _adaln_reference(norm, x, temb, deviation)

    assert torch.equal(out, _adaln_reference(norm, x, temb))
    assert not torch.equal(out, wrong)
    if floor is not None:
        assert (out - wrong).abs().max() > floor * out.abs().max()
    if deviation == _WRONG_PER_TOKEN:
        for request in range(requests):
            for token in range(1, tokens):
                assert torch.equal(out[request, token], out[request, 0])
    if deviation == _WRONG_BARE_SCALE:
        zeroed = AdaLayerNorm(_DT_DIM)
        for parameter in zeroed.parameters():
            parameter.zero_()
        assert torch.equal(zeroed(x, temb), F.layer_norm(x, (_DT_DIM,), eps=_EPS))
        assert torch.equal(_adaln_reference(zeroed, x, temb, _WRONG_BARE_SCALE), torch.zeros_like(x))


@torch.no_grad()
def test_a_batch_can_hold_requests_at_different_denoising_steps():
    """Each row's answer depends on its own ``temb`` only, so moving one request's leaves its
    neighbours bit-identical. Same batch shape in both runs; batch against singleton is not
    exact (2.4e-07 from the GEMM)."""
    norm = _adaln()
    x = _hidden(3, _QUERIES)
    temb = _temb(3)
    out = norm(x, temb)

    moved = temb.clone()
    moved[1] += 0.5
    after = norm(x, moved)

    assert torch.equal(after[0], out[0])
    assert torch.equal(after[2], out[2])
    assert not torch.equal(after[1], out[1])
    assert torch.equal(norm(x[[2, 0, 1]], temb[[2, 0, 1]]), out[[2, 0, 1]])


# ----------------------------------------------------------------------------
# One DT block against an inline reference
# ----------------------------------------------------------------------------

# The block's fill, wider than 0.1 so the attention's contribution is comparable to the
# residual: at 0.1 dropping the attention residual moves the answer by 1.4%, a wiring test
# that reads as rounding. At 0.3 every deviation moves it by at least a third of its largest
# element.
_BLOCK_SCALE = 0.3

_WRONG_NO_ATTN_RESIDUAL = "the attention residual dropped"
_WRONG_NO_FF_RESIDUAL = "the feed-forward residual dropped"
_WRONG_NORM3_AFTER_FF = "norm3 after the feed-forward"
_WRONG_NORMS_SWAPPED = "AdaLN before the feed-forward instead"
# the feed-forward reading the block's input rather than what the attention residual made
_WRONG_FF_BEFORE_RESIDUAL = "the feed-forward reads the pre-residual tokens"

_BLOCK_DEVIATIONS = (
    _WRONG_NO_ATTN_RESIDUAL,
    _WRONG_NO_FF_RESIDUAL,
    _WRONG_NORM3_AFTER_FF,
    _WRONG_NORMS_SWAPPED,
    _WRONG_FF_BEFORE_RESIDUAL,
)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A CPU stand-in for the attention resource in the block's own contract,
    ``[B, T, heads, head_dim]`` in and out, unmasked. The module and the reference share it, so
    the comparison is of the wiring and exact."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _block(cross: bool, seed: int = 23) -> BasicTransformerBlock:
    """One block at either attention setting, filled. The attention is a ``forward`` argument."""
    block = BasicTransformerBlock(
        _DT_DIM,
        _DT_HEADS,
        _DT_HEAD_DIM,
        cross_attention_dim=_BRIDGE_DIM if cross else None,
    )
    randomize(block, seed, scale=_BLOCK_SCALE)
    return block


def _bridge_rows(
    cross: bool, requests: int = _REQUESTS, rows: int = _BRIDGE_ROWS, seed: int = 24,
) -> torch.Tensor | None:
    """What a cross block is handed: the bridge's rows, ``[B, S, 2048]`` in the real model. A
    self block is handed nothing, which is what makes its k and v its own tokens."""
    if not cross:
        return None
    return torch.randn(requests, rows, _BRIDGE_DIM, generator=torch.Generator().manual_seed(seed))


def _block_reference(
    block: BasicTransformerBlock,
    hidden: torch.Tensor,
    temb: torch.Tensor,
    encoder: torch.Tensor | None,
    deviation: str | None = None,
    attend=_sdpa,
) -> torch.Tensor:
    """AdaLN, attention, ``+ residual``, a plain LayerNorm, the feed-forward, ``+ residual``.
    Borrowed: the projections and ``block.ff`` (diffusers' ``FeedForward``, pinned in
    ``test_bridge.py``). ``attend`` lets the span tests hand both sides the same attention."""
    # the width comes off the tensor: `DiT`'s blocks are built at the inner width
    requests, queries, dim = hidden.shape
    heads = (requests, -1, _DT_HEADS, _DT_HEAD_DIM)
    plain = _adaln_reference(block.norm1, hidden, temb)
    if deviation == _WRONG_NORMS_SWAPPED:
        plain = F.layer_norm(hidden, (dim,), eps=_EPS)
    source = plain if encoder is None else encoder
    out = attend(
        block.attn1.to_q(plain).view(heads),
        block.attn1.to_k(source).view(heads),
        block.attn1.to_v(source).view(heads),
    )
    attended = block.attn1.to_out[0](out.reshape(requests, queries, -1))

    residual = attended if deviation == _WRONG_NO_ATTN_RESIDUAL else attended + hidden
    if deviation == _WRONG_NORM3_AFTER_FF:
        return F.layer_norm(block.ff(residual), (dim,), eps=_EPS) + residual
    normed = F.layer_norm(
        hidden if deviation == _WRONG_FF_BEFORE_RESIDUAL else residual, (dim,), eps=_EPS,
    )
    if deviation == _WRONG_NORMS_SWAPPED:
        normed = _adaln_reference(block.norm1, residual, temb)
    forwarded = block.ff(normed)
    return forwarded if deviation == _WRONG_NO_FF_RESIDUAL else forwarded + residual


# The even DT blocks cross-attend the bridge's rows, the odd ones their own tokens; the two
# differ only in where k and v come from, the half most easily got backwards.
_SETTINGS = {"a cross block": True, "a self block": False}


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@pytest.mark.parametrize(
    "requests, queries", [(_REQUESTS, _QUERIES), (1, 1)],
    ids=["a batch of requests", "one request, one token"],
)
@torch.no_grad()
def test_the_block_is_the_reference_block(cross, requests, queries):
    """``torch.equal``: the attention is the caller's callable, so nothing between module and
    reference is a kernel this file did not call itself."""
    block = _block(cross)
    hidden = _hidden(requests, queries)
    temb = _temb(requests)
    encoder = _bridge_rows(cross, requests)
    out = block(hidden, temb, encoder, _sdpa)

    assert out.shape == (requests, queries, _DT_DIM)
    assert torch.equal(out, _block_reference(block, hidden, temb, encoder))


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@pytest.mark.parametrize("deviation", _BLOCK_DEVIATIONS)
@torch.no_grad()
def test_a_block_wired_any_other_way_is_a_different_answer(cross, deviation):
    """Each is the same six pieces in another block's arrangement (one residual, post-norm,
    conditioned feed-forward, parallel halves); each loads the checkpoint and runs."""
    block = _block(cross)
    hidden, temb = _hidden(), _temb()
    encoder = _bridge_rows(cross)
    out = block(hidden, temb, encoder, _sdpa)
    wrong = _block_reference(block, hidden, temb, encoder, deviation)

    assert not torch.equal(out, wrong)
    # a floor, not a fitted threshold: over weight seeds 20-31 x both settings the weakest
    # moved the answer by 0.26 of its largest element (norm3 after the feed-forward, the only
    # one that keeps both residuals)
    assert (out - wrong).abs().max() > 0.2 * out.abs().max()


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@torch.no_grad()
def test_k_and_v_come_from_the_setting_and_q_always_from_the_tokens(cross):
    """A cross block's ``to_k``/``to_v`` are 2048 wide (the bridge's rows), a self block's 1536
    (its own tokens). In the self setting k and v are the AdaLN output, not the block's input."""
    block = _block(cross)
    hidden, temb = _hidden(), _temb()
    encoder = _bridge_rows(cross)
    seen = {}

    def record(q, k, v):
        seen.update(q=q, k=k, v=v)
        return _sdpa(q, k, v)

    block(hidden, temb, encoder, record)

    assert block.attn1.to_q.in_features == _DT_DIM
    assert block.attn1.to_k.in_features == block.attn1.to_v.in_features
    assert block.attn1.to_k.in_features == (_BRIDGE_DIM if cross else _DT_DIM)
    assert block.attn1.to_out[0].in_features == _DT_INNER
    assert block.attn1.to_out[0].out_features == _DT_DIM

    normed = _adaln_reference(block.norm1, hidden, temb)
    source = encoder if cross else normed
    heads = (_REQUESTS, -1, _DT_HEADS, _DT_HEAD_DIM)
    assert torch.equal(seen["q"], block.attn1.to_q(normed).view(heads))
    assert torch.equal(seen["k"], block.attn1.to_k(source).view(heads))
    assert torch.equal(seen["v"], block.attn1.to_v(source).view(heads))


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@torch.no_grad()
def test_the_callable_is_handed_heads_last_and_its_output_is_folded_back_the_same_way(cross):
    """The attention resource's contract: q ``[B, T, heads, head_dim]``, k and v
    ``[B, S, heads, head_dim]``, output folded back with ``reshape(B, T, -1)``. An attention
    returning q makes the fold checkable; a heads-first layout would pass a shape check and
    fold the heads back interleaved."""
    block = _block(cross)
    hidden, temb = _hidden(), _temb()
    encoder = _bridge_rows(cross)
    keys = _BRIDGE_ROWS if cross else _QUERIES
    seen = []

    def shapes(q, k, v):
        seen.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        return q

    out = block(hidden, temb, encoder, shapes)

    assert seen == [(
        (_REQUESTS, _QUERIES, _DT_HEADS, _DT_HEAD_DIM),
        (_REQUESTS, keys, _DT_HEADS, _DT_HEAD_DIM),
        (_REQUESTS, keys, _DT_HEADS, _DT_HEAD_DIM),
    )]
    normed = _adaln_reference(block.norm1, hidden, temb)
    attended = block.attn1.to_out[0](block.attn1.to_q(normed)) + hidden
    assert torch.equal(out, block.ff(F.layer_norm(attended, (_DT_DIM,), eps=_EPS)) + attended)


def test_a_block_takes_its_attention_per_call_and_has_nowhere_to_keep_a_stale_one():
    """No default on ``forward``: a block that fell back to its own tokens would silently
    ignore the language and images. No ``attend`` on ``__init__``: one bound attention cannot
    serve the three spans the stack reads."""
    block = BasicTransformerBlock(_DT_DIM, _DT_HEADS, _DT_HEAD_DIM)
    hidden, temb = _hidden(), _temb()

    with pytest.raises(TypeError):
        block(hidden, temb, None)
    with pytest.raises(TypeError):
        BasicTransformerBlock(_DT_DIM, _DT_HEADS, _DT_HEAD_DIM, attend=_sdpa)
    assert block(hidden, temb, None, _sdpa).shape == (_REQUESTS, _QUERIES, _DT_DIM)


# ----------------------------------------------------------------------------
# The DT's output head and the stack it sits on, against an inline reference
# ----------------------------------------------------------------------------

# `DiT` builds every block at `dim = stack.inner_dim` (both 1536 in the real model), so the
# numbers that stay distinct here are the stack's width (15), the bridge's rows (7), the output
# width (6) and the position table's (8 wide, 9 rows).
_DIT_DIM = _DT_HEADS * _DT_HEAD_DIM
_DIT_OUT = 6
_DIT_DEPTH = 2  # one of each kind of block, which is the shortest stack that alternates
# the position table: `max_seq_len` x `input_embedding_dim`, 1024 x 1536 in the real model,
# of which only the first `action_horizon` (40) rows are ever read
_TABLE_ROWS = 9
_POS_DIM = _HIDDEN
_ACTION_ROWS = 3

# `norm_out`'s eps: the reference hardcodes 1e-6 at the output norm (`dit.py:284`) and takes
# the config's 1e-5 everywhere else, and no weight and no bf16 dump records an eps.
_OUT_EPS = 1e-6

# The real schedule's four timesteps: `int(t / num_inference_timesteps * num_timestep_buckets)`
# for t = 0..3, which is what the reference fills `timesteps_tensor` with
# (`gr00t_n1d7.py:398-404`). A batch holds one per request, at whichever iteration each is on.
_DENOISE_STEPS = (0.0, 250.0, 500.0, 750.0)

# Deliberate deviations of the head's three lines, each of which loads the checkpoint and runs.
_WRONG_HEAD_SCALE_FIRST = "scale first out of the head's chunk"
_WRONG_HEAD_BLOCK_EPS = "the blocks' 1e-5 at the output norm"
# `[:, None]` dropped, as in `_WRONG_PER_TOKEN`: computable only where requests equal rows.
_WRONG_HEAD_PER_ROW = "the head's pair indexed per row"


def _dit_config(depth: int = _DIT_DEPTH) -> Gr00tN17Config:
    """``input_embedding_dim`` (8) differs from the stack's width (15): the table is sized from
    one and the blocks from the other, both 1536 in the real checkpoint."""
    return Gr00tN17Config(
        text=None, vision=None, bridge=None,
        dit=Gr00tN17StackConfig(
            num_layers=depth, num_attention_heads=_DT_HEADS,
            attention_head_dim=_DT_HEAD_DIM, output_dim=_DIT_OUT,
        ),
        deepstack_visual_indexes=(), backbone_embedding_dim=_BRIDGE_DIM,
        max_action_dim=_ACTION_DIM, hidden_size=9, input_embedding_dim=_POS_DIM,
        max_num_embodiments=_CATEGORIES, max_seq_len=_TABLE_ROWS, action_horizon=_ACTION_ROWS,
    )


def _dit(depth: int = _DIT_DEPTH, seed: int = 25) -> DiT:
    """A **shallow** DT, filled at ``_BLOCK_SCALE``. The real stack is 32 blocks, 1.12B
    parameters and 4.2 GiB in fp32, so nothing here builds one."""
    dit = DiT(_dit_config(depth))
    randomize(dit, seed, scale=_BLOCK_SCALE)
    return dit


# Every span read with this file's own stand-in, so module and reference share one callable
# and the parity tests stay `torch.equal`. The module's own fallback is pinned bit-identical
# to it in the span section.
_STAND_INS = dict.fromkeys((SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN), _sdpa)


def _dit_rows(
    requests: int = _REQUESTS, rows: int = _QUERIES, seed: int = 25, scale: float = 1.0,
) -> torch.Tensor:
    """``[B, T, dim]``: the state row and the action rows the DT takes in, 41 in the real model."""
    return scale * torch.randn(
        requests, rows, _DIT_DIM, generator=torch.Generator().manual_seed(seed),
    )


def _pos_rows(requests: int = _REQUESTS, rows: int = _ACTION_ROWS, seed: int = 26) -> torch.Tensor:
    """``[B, T, input_embedding_dim]``: what the action encoder returns, the only thing the
    position table is added to."""
    return torch.randn(requests, rows, _POS_DIM, generator=torch.Generator().manual_seed(seed))


def _steps(requests: int = _REQUESTS) -> torch.Tensor:
    """``[B]``: a *different* denoising timestep per request, the serving case."""
    return torch.tensor(_DENOISE_STEPS[:requests])


def _position_table(seed: int = 27) -> ActionPositionEmbedding:
    """Filled, which is not optional: the module allocates its table with ``torch.empty``."""
    table = ActionPositionEmbedding(_dit_config())
    randomize(table, seed, scale=_BLOCK_SCALE)
    return table


def _head_reference(
    dit: DiT, hidden: torch.Tensor, temb: torch.Tensor, deviation: str | None = None,
) -> torch.Tensor:
    """``shift, scale = proj_out_1(silu(temb)).chunk(2, dim=1)`` (**that** order), then
    ``norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]``, then ``proj_out_2``. The norm
    is written out (no weight, no bias, eps 1e-6); only the projections are borrowed."""
    first, second = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
    shift, scale = (second, first) if deviation == _WRONG_HEAD_SCALE_FIRST else (first, second)
    eps = _EPS if deviation == _WRONG_HEAD_BLOCK_EPS else _OUT_EPS
    normed = F.layer_norm(hidden, (hidden.shape[-1],), eps=eps)
    if deviation == _WRONG_HEAD_PER_ROW:
        return dit.proj_out_2(normed * (1 + scale) + shift)
    return dit.proj_out_2(normed * (1 + scale[:, None]) + shift[:, None])


def _reference_span(index: int, attend_text_every_n_blocks: int) -> str:
    """Which span block ``index`` reads, written out from ``AlternateVLDiT.forward``
    (``dit.py:379-396``): odd blocks self-attend; an even block reads the text rows
    (``non_image_attention_mask``) when ``idx % (2 * n) == 0`` and the image rows otherwise.
    Not imported from ``_span_for``, the function it checks."""
    if index % 2 == 1:
        return SA_SPAN
    return VL_TEXT_SPAN if index % (2 * attend_text_every_n_blocks) == 0 else VL_IMAGE_SPAN


def _dit_reference(
    dit: DiT,
    hidden: torch.Tensor,
    text: torch.Tensor,
    image: torch.Tensor,
    timestep: torch.Tensor,
    deviation: str | None = None,
    attends: dict | None = None,
) -> torch.Tensor:
    """The timestep encoder, every block through ``_block_reference`` with spans from
    ``_reference_span``, then the head. ``attends`` maps a span to its attention, as the
    module's does, defaulting to the stand-in."""
    temb = dit.timestep_encoder(timestep)
    rows = {SA_SPAN: None, VL_TEXT_SPAN: text, VL_IMAGE_SPAN: image}
    for index, block in enumerate(dit.transformer_blocks):
        span = _reference_span(index, dit.attend_text_every_n_blocks)
        # `deviation` is a reading of the head; the blocks here are always the right wiring
        hidden = _block_reference(
            block, hidden, temb, rows[span], None, (attends or {}).get(span, _sdpa),
        )
    return _head_reference(dit, hidden, temb, deviation)


# The tests in this section hand the *same* rows as `vl_text` and `vl_image`, so every cross
# block reads all of the bridge's rows: the reference's plain, unmasked `DiT.forward`. The
# split itself is pinned in the next section, with spans of different lengths.
_DIT_SHAPES = {
    # the real shape is 41 rows: one state row and `action_horizon` action rows
    "a batch of requests": (_REQUESTS, _QUERIES),
    "one request, one row": (1, 1),
    "as many rows as requests": (3, 3),
}


@pytest.mark.parametrize("shape", list(_DIT_SHAPES.values()), ids=list(_DIT_SHAPES))
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@torch.no_grad()
def test_the_dit_is_its_blocks_then_the_output_head(shape, depth):
    """``torch.equal``. Four depths, because ``index % 2`` agrees with ``index < depth // 2``,
    ``index % 3`` and ``index % 4`` on the first two blocks. The square shape is the one where
    the head's dropped ``[:, None]`` raises nothing; one request, one row is the serving edge."""
    requests, rows = shape
    dit = _dit(depth)
    hidden = _dit_rows(requests, rows)
    encoder = _bridge_rows(True, requests)
    timestep = _steps(requests)
    out = dit(hidden, encoder, encoder, timestep, _STAND_INS)

    assert out.shape == (requests, rows, _DIT_OUT)
    assert torch.equal(out, _dit_reference(dit, hidden, encoder, encoder, timestep))


@torch.no_grad()
def test_the_output_heads_chunk_is_shift_then_scale_and_not_adalns_order():
    """The head chunks ``shift, scale`` where ``AdaLayerNorm`` chunks ``scale, shift``
    (``dit.py:331`` against ``dit.py:95``); both halves are plausible (RMS 1.42 and 1.80), so
    only values see a swap. With the shift half 0 and the scale half -1 the right reading is
    ``proj_out_2.bias`` exactly and the swap is 4.03 against 0.50; on the trained-style fill
    the swap moves the answer by 1.39x its largest element."""
    dit = _dit()
    with torch.no_grad():
        dit.proj_out_1.weight.zero_()
        dit.proj_out_1.bias[:_DIT_DIM].zero_()  # the first half out of the chunk: `shift`
        dit.proj_out_1.bias[_DIT_DIM:].fill_(-1.0)  # the second: `scale`, so `1 + scale` is 0
    hidden = _dit_rows()
    encoder = _bridge_rows(True)
    timestep = _steps()
    out = dit(hidden, encoder, encoder, timestep, _STAND_INS)
    swapped = _dit_reference(dit, hidden, encoder, encoder, timestep, _WRONG_HEAD_SCALE_FIRST)

    assert torch.equal(out, dit.proj_out_2.bias.expand(_REQUESTS, _QUERIES, _DIT_OUT))
    assert not torch.equal(out, swapped)
    assert swapped.abs().max() > 2.0 * out.abs().max()

    trained = _dit()
    out = trained(hidden, encoder, encoder, timestep, _STAND_INS)
    swapped = _dit_reference(trained, hidden, encoder, encoder, timestep, _WRONG_HEAD_SCALE_FIRST)
    first, second = trained.proj_out_1(F.silu(trained.timestep_encoder(timestep))).chunk(2, 1)
    assert first.pow(2).mean().sqrt() == pytest.approx(second.pow(2).mean().sqrt(), rel=0.3)
    assert not torch.equal(out, swapped)
    assert (out - swapped).abs().max() > 0.2 * out.abs().max()


@torch.no_grad()
def test_the_output_norm_normalises_at_1e_6_where_every_other_norm_uses_1e_5():
    """An eps is invisible to parity against a bf16 dump, so it is pinned on rows of variance
    8.5e-07, where ``sqrt(v + 1e-5) / sqrt(v + 1e-6)`` is 2.3 and the output moves by 0.45 of
    its largest element. The stack has no blocks, since a block's norm would wash that out."""
    flat = _dit(depth=0)
    hidden = _dit_rows(scale=1e-3)
    encoder = _bridge_rows(True)
    timestep = _steps()
    out = flat(hidden, encoder, encoder, timestep, _STAND_INS)

    assert hidden.var(-1, unbiased=False).max() < 1e-05  # the input the eps is visible on
    assert torch.equal(out, _dit_reference(flat, hidden, encoder, encoder, timestep))
    at_1e_5 = _dit_reference(flat, hidden, encoder, encoder, timestep, _WRONG_HEAD_BLOCK_EPS)
    assert not torch.equal(out, at_1e_5)
    assert (out - at_1e_5).abs().max() > 0.2 * out.abs().max()

    # and the two norms are different numbers in the same model, which is the whole hazard
    deep = _dit()
    assert deep.norm_out.eps == _OUT_EPS
    assert {block.norm3.eps for block in deep.transformer_blocks} == {_EPS}
    assert {block.norm1.norm.eps for block in deep.transformer_blocks} == {_EPS}


def test_the_output_norm_is_not_affine_because_the_checkpoint_has_no_weight_for_it():
    """The checkpoint holds no ``norm_out`` tensor. An affine one would load and run at its
    identity init and be right until someone trained it, so the key set is the test."""
    dit = _dit()
    blocks = {f"transformer_blocks.{index}." for index in range(_DIT_DEPTH)}
    names = {name for name, _ in dit.named_parameters()}

    assert dit.norm_out.weight is None and dit.norm_out.bias is None
    assert {name for name in names if not name.startswith(tuple(blocks))} == {
        "timestep_encoder.timestep_embedder.linear_1.weight",
        "timestep_encoder.timestep_embedder.linear_1.bias",
        "timestep_encoder.timestep_embedder.linear_2.weight",
        "timestep_encoder.timestep_embedder.linear_2.bias",
        "proj_out_1.weight", "proj_out_1.bias", "proj_out_2.weight", "proj_out_2.bias",
    }
    assert dit.proj_out_1.out_features == 2 * _DIT_DIM  # the one projection both halves
    assert dit.proj_out_2.out_features == _DIT_OUT != _DIT_DIM


@torch.no_grad()
def test_the_heads_pair_is_one_per_request_and_reaches_every_row_of_it():
    """Dropping the head's ``[:, None]`` raises everywhere but ``B == T``, where row ``t`` gets
    request ``t``'s conditioning (a 1.17x move on O(5) rows). Compared against the wrong
    reading, since a request's rows differ among themselves in the last ulp (2.4e-07)."""
    requests = rows = 3
    dit = _dit()
    hidden = _dit_rows(requests, rows)
    encoder = _bridge_rows(True, requests)
    timestep = _steps(requests)
    out = dit(hidden, encoder, encoder, timestep, _STAND_INS)
    per_row = _dit_reference(dit, hidden, encoder, encoder, timestep, _WRONG_HEAD_PER_ROW)

    assert torch.equal(out, _dit_reference(dit, hidden, encoder, encoder, timestep))
    assert not torch.equal(out, per_row)
    assert (out - per_row).abs().max() > 0.2 * out.abs().max()

    # at any other shape the dropped index is a shape error, hence the square shape above
    with pytest.raises(RuntimeError):
        _dit_reference(dit, _dit_rows(), _bridge_rows(True), _bridge_rows(True), _steps(), _WRONG_HEAD_PER_ROW)


@torch.no_grad()
def test_a_dit_batch_can_hold_requests_at_different_denoising_steps():
    """Requests in one batch are each at their own iteration, so moving one request's timestep
    must leave its neighbours bit-identical. Isolation, not equivariance: permuting the batch
    reblocks the GEMMs and moves the answer by 9.5e-07."""
    dit = _dit()
    hidden = _dit_rows(3, _QUERIES)
    encoder = _bridge_rows(True, 3)
    timestep = _steps(3)
    out = dit(hidden, encoder, encoder, timestep, _STAND_INS)

    moved = timestep.clone()
    moved[1] = _DENOISE_STEPS[3]
    after = dit(hidden, encoder, encoder, moved, _STAND_INS)

    # the four steps the reference's loop produces, which is what a batch is drawn from
    assert _DENOISE_STEPS == tuple(float(int(t / 4 * 1000)) for t in range(4))
    assert len(set(timestep.tolist())) == 3  # three requests, three different steps
    assert torch.equal(after[0], out[0])
    assert torch.equal(after[2], out[2])
    assert not torch.equal(after[1], out[1])


# ----------------------------------------------------------------------------
# The alternation, the two spans it divides into, and the attention each is read with
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@torch.no_grad()
def test_the_stack_is_built_and_run_as_cross_then_self_from_block_zero(depth):
    """``_attends_encoder`` sizes ``to_k``/``to_v`` and picks what each block is handed, and the
    two must agree: a block handed 1536-wide rows cross-attends them happily. The key count
    (6 bridge rows against 4 own) says which source each block got, in block order."""
    dit = _dit(depth)
    hidden = _dit_rows()
    encoder = _bridge_rows(True)
    seen: list[int] = []

    def record(q, k, v):
        seen.append(k.shape[1])
        return _sdpa(q, k, v)

    dit(hidden, encoder, encoder, _steps(), dict.fromkeys(_STAND_INS, record))

    cross = [index % 2 == 0 for index in range(depth)]
    assert [_attends_encoder(index) for index in range(depth)] == cross
    assert _attends_encoder(0) is True  # the alternation starts with a cross block
    # one argument: it cannot read `attend_text_every_n_blocks`, which agrees with `% 2` at n = 2
    assert list(inspect.signature(_attends_encoder).parameters) == ["index"]
    assert _BRIDGE_ROWS != _QUERIES  # so the key count says where k came from
    assert seen == [_BRIDGE_ROWS if is_cross else _QUERIES for is_cross in cross]
    for kind in ("to_k", "to_v"):
        assert [
            getattr(block.attn1, kind).in_features for block in dit.transformer_blocks
        ] == [_BRIDGE_DIM if is_cross else _DIT_DIM for is_cross in cross]
    # and q is the block's own rows at every block, cross or self
    assert {block.attn1.to_q.in_features for block in dit.transformer_blocks} == {_DIT_DIM}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_text_image_rule_is_the_config_field_dividing_the_cross_blocks(n):
    """``attend_text_every_n_blocks`` splits the cross blocks into text and image and leaves
    cross/self alone. Four values, because at the checkpoint's n = 2 several wrong readings
    agree with ``idx % 4``."""
    indexes = range(16)
    spans = [_span_for(index, n) for index in indexes]

    assert spans == [_reference_span(index, n) for index in indexes]
    # the cross/self half is invariant in the field, and is `_attends_encoder`'s rule
    assert [span != SA_SPAN for span in spans] == [_attends_encoder(index) for index in indexes]
    assert [span != SA_SPAN for span in spans] == [index % 2 == 0 for index in indexes]
    # the text/image half is not: the period is `2 * n` cross blocks
    text = [index for index in indexes if spans[index] == VL_TEXT_SPAN]
    assert text == [index for index in indexes if index % (2 * n) == 0]
    assert len(text) == len(range(0, 16, 2 * n))
    assert set(spans) == ({SA_SPAN, VL_TEXT_SPAN} if n == 1 else {SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN})
    assert list(inspect.signature(_span_for).parameters) == [
        "index", "attend_text_every_n_blocks",
    ]


def test_the_span_rule_runs_block_for_block_over_the_checkpoints_thirty_two():
    """At the real depth and n, block 0 is a **text** block (``0 % 4 == 0``). Starting at the
    image rows is the same 32 blocks, tensors and shapes end to end, so only this comparison
    sees the phase."""
    config = _dit_config(32)
    n = config.attend_text_every_n_blocks
    spans = [_span_for(index, n) for index in range(config.dit.num_layers)]

    assert (config.dit.num_layers, n) == (32, 2)
    assert spans == [_reference_span(index, n) for index in range(config.dit.num_layers)]
    assert spans[0] == VL_TEXT_SPAN
    assert spans[:8] == [VL_TEXT_SPAN, SA_SPAN, VL_IMAGE_SPAN, SA_SPAN] * 2
    # 16 self blocks, and the 16 cross blocks split evenly between language and images
    assert Counter(spans) == {SA_SPAN: 16, VL_TEXT_SPAN: 8, VL_IMAGE_SPAN: 8}


# Three *different* kv lengths, so a span is told apart by what its attention was handed. The
# DT's rows are 41 in the real model; the text span is shorter than one prompt and the image
# span is one 256-token tile.
_SA_ROWS = 41
_TEXT_ROWS = 37
_IMAGE_ROWS = 256


def _spy(label: str, seen: list) -> Callable:
    """An attention that records the label it was registered under and the key count it was
    handed, then attends: the right span with the wrong attention, or the reverse, is a
    different list either way."""
    def attend(q, k, v):
        seen.append((label, k.shape[1]))
        return _sdpa(q, k, v)

    return attend


def _zeros(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """An attention whose output is nothing, in the contract's shape: it keeps ``to_out`` and
    both residuals alive, so a supplied attention is distinguishable from the fallback."""
    return torch.zeros_like(q)


@torch.no_grad()
def test_each_block_is_handed_the_attention_its_own_span_asked_for():
    """One attention per span over text, self, image, self, with lengths 41, 37 and 256, so a
    span swap, a map read once outside the loop or keyed by block index is a different list.
    The parity line says each block is handed its span's rows."""
    dit = _dit(4)
    hidden = _dit_rows(_REQUESTS, _SA_ROWS)
    text = _bridge_rows(True, _REQUESTS, _TEXT_ROWS, seed=30)
    image = _bridge_rows(True, _REQUESTS, _IMAGE_ROWS, seed=31)
    timestep = _steps()
    seen: list[tuple[str, int]] = []

    out = dit(hidden, text, image, timestep, {label: _spy(label, seen) for label in _STAND_INS})

    assert len({_SA_ROWS, _TEXT_ROWS, _IMAGE_ROWS}) == 3  # so the key count names the span
    assert seen == [
        (VL_TEXT_SPAN, _TEXT_ROWS),
        (SA_SPAN, _SA_ROWS),
        (VL_IMAGE_SPAN, _IMAGE_ROWS),
        (SA_SPAN, _SA_ROWS),
    ]
    assert torch.equal(out, _dit_reference(dit, hidden, text, image, timestep))


@torch.no_grad()
def test_a_span_label_no_block_reads_is_refused_rather_than_ignored():
    """Regression: a misspelt span key reaches no block, so that span falls back to SDPA,
    bitwise identical to passing nothing but off the planned resource and the captured graph's
    kernel. This matches the engine's refusal of an unplanned label."""
    dit = _dit(2)
    hidden = _dit_rows(_REQUESTS, _SA_ROWS)
    text = _bridge_rows(True, _REQUESTS, _TEXT_ROWS, seed=30)
    image = _bridge_rows(True, _REQUESTS, _IMAGE_ROWS, seed=31)
    timestep = _steps()

    with pytest.raises(ValueError, match=r"no span \['vl_txt'\]"):
        dit(hidden, text, image, timestep, {"vl_txt": _sdpa})
    # the message names the spans the caller could have meant, and who supplies them
    with pytest.raises(ValueError, match="names the spans it declared"):
        dit(hidden, text, image, timestep, {SA_SPAN: _sdpa, "sa_rows": _sdpa})
    # a subset is still legal: that is the documented fallback, not a mistake
    dit(hidden, text, image, timestep, {SA_SPAN: _sdpa})


@torch.no_grad()
def test_a_span_with_no_attention_of_its_own_is_read_with_sdpa():
    """The fallback the parity backend runs on: ``None``, an empty map and a partial map agree,
    and a supplied label is read with what was supplied. The fallback is bit-identical to this
    file's stand-in, so every parity test above also covers the default path."""
    dit = _dit(4)
    hidden = _dit_rows(_REQUESTS, _SA_ROWS)
    text = _bridge_rows(True, _REQUESTS, _TEXT_ROWS, seed=30)
    image = _bridge_rows(True, _REQUESTS, _IMAGE_ROWS, seed=31)
    timestep = _steps()
    default = dit(hidden, text, image, timestep)

    assert torch.equal(default, dit(hidden, text, image, timestep, {}))
    assert torch.equal(default, dit(hidden, text, image, timestep, {SA_SPAN: _SDPA_ATTEND}))
    assert torch.equal(
        default, dit(hidden, text, image, timestep, dict.fromkeys(_STAND_INS, _SDPA_ATTEND)),
    )
    assert torch.equal(default, _dit_reference(dit, hidden, text, image, timestep))

    supplied = dit(hidden, text, image, timestep, {VL_IMAGE_SPAN: _zeros})
    assert not torch.equal(supplied, default)
    assert torch.equal(
        supplied,
        _dit_reference(dit, hidden, text, image, timestep, attends={VL_IMAGE_SPAN: _zeros}),
    )


@torch.no_grad()
def test_the_sdpa_fallback_is_one_object_and_not_one_per_forward(monkeypatch):
    """A compiled region guards on the identity of the callables it is handed, so a fallback
    built per forward retraces every step (``RaggedAttentionCallable``, in
    ``mstar/engine/resources/convenience.py``, is one object per label for this reason). No parity
    test can see it."""
    seen: list = []
    forward = BasicTransformerBlock.forward

    def record(self, hidden_states, temb, encoder_hidden_states, attend):
        seen.append(attend)
        return forward(self, hidden_states, temb, encoder_hidden_states, attend)

    monkeypatch.setattr(BasicTransformerBlock, "forward", record)
    hidden = _dit_rows(_REQUESTS, _SA_ROWS)
    text = _bridge_rows(True, _REQUESTS, _TEXT_ROWS, seed=30)
    image = _bridge_rows(True, _REQUESTS, _IMAGE_ROWS, seed=31)
    timestep = _steps()

    dit = _dit(4)
    dit(hidden, text, image, timestep)
    dit(hidden, text, image, timestep)  # a second denoising step on the same stack
    _dit(4)(hidden, text, image, timestep)  # and a second stack, so it is not per instance
    assert len(seen) == 12
    assert all(attend is _SDPA_ATTEND for attend in seen)
    assert len({id(attend) for attend in seen}) == 1

    # the spans a partial map leaves out fall back to that same object, not to a copy per span
    dit(hidden, text, image, timestep, {SA_SPAN: _zeros})
    assert [attend is _SDPA_ATTEND for attend in seen[12:]] == [True, False, True, False]
    assert len({id(attend) for attend in seen}) == 2


# ----------------------------------------------------------------------------
# `ActionPositionEmbedding`
# ----------------------------------------------------------------------------

_POS_ROWS = {
    "the action horizon": _ACTION_ROWS,  # 40 rows in the real model
    "one action step": 1,
    "the whole table": _TABLE_ROWS,
}


@pytest.mark.parametrize("rows", list(_POS_ROWS.values()), ids=list(_POS_ROWS))
@torch.no_grad()
def test_the_table_adds_its_own_first_rows_and_reads_nothing_past_them(rows):
    """Rows ``0:T``, what the reference's ``position_embedding(arange(T))`` gathers. The real
    table is 1024 rows for 40 read, so an off-by-one raises nothing; perturbing the rows past
    ``T`` must leave the answer bit-identical."""
    table = _position_table()
    features = _pos_rows(rows=rows)
    out = table(features)

    assert out.shape == features.shape == (_REQUESTS, rows, _POS_DIM)
    assert torch.equal(out, features + table.weight[:rows])
    assert torch.equal(out, features + table.weight[torch.arange(rows)])  # the gather, as a slice

    if rows < _TABLE_ROWS:
        assert not torch.equal(out, features + table.weight[1 : rows + 1])  # the off-by-one
        with torch.no_grad():
            table.weight[rows:] += 1.0
        assert torch.equal(table(features), out)
