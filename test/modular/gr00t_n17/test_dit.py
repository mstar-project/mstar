"""GR00T N1.7's DT: the action head's transformer and the pieces around it.

This file starts with the two timestep encodings, because GR00T has two of them and they
live in one model. The action encoder's ``SinusoidalPositionalEncoding`` is a hand port of
NVIDIA's own class; the DT's ``TimestepEncoder`` is diffusers' ``Timesteps`` followed by
diffusers' ``TimestepEmbedding``, which is what the reference imports too. They disagree on
both of diffusers' knobs: the DT's reads cos first and spaces its frequencies over
``half_dim - 1``, the action encoder's reads sin first over ``half_dim``.

The sinusoid has no parameters, so no weight can say which convention it is, and no
magnitude can either -- both are half sines and half cosines, so their rows have the same
L2 norm to the bit. It is pinned three ways instead: against an inline reference written
from the algorithm, against values only one convention can produce (the encoding of
timestep 0, the span of the frequency ladder, the same ladder read at twice the width), and
against diffusers' own sinusoid, which is where the base 10000 and the shift come from
rather than from this test.

Then ``TimestepEncoder``: its 256-wide projection, which the checkpoint fixes and no config
does, and the cast to the weights' dtype, which is a real rounding in bf16.

Next comes ``MultiEmbodimentActionEncoder``, which is where that sinusoid is used:
the action through ``W1``, the time through the sinusoid, the two **concatenated** rather than
added, ``W2``, a swish spelled ``x * sigmoid(x)``, ``W3``. It lives in ``embodiment.py`` next
to the state encoder, but it is a DT piece and it depends on the sinusoid above, so it is
tested here. ``CategorySpecificLinear`` itself is pinned in ``test_gr00t_n17_state.py`` and
not again here.

Then the DT block: ``AdaLayerNorm``, which carries one request's denoising timestep into all
of that request's tokens, and ``BasicTransformerBlock``, which alternates cross-attention
over the bridge's rows with self-attention over its own. Its attention arrives as a
caller-supplied callable, **per call** rather than bound at construction, so an inline
reference can be handed the same one -- which leaves no kernel in the comparison that this
file did not call itself, and makes every test there ``torch.equal`` rather than a bar.

Then ``DiT`` itself -- the stack of those blocks and the output head on top of it -- with
``ActionPositionEmbedding`` beside it. Its tests are the details a dump comparison against
the reference cannot resolve: that the head chunks ``shift, scale`` where ``AdaLayerNorm``
chunks ``scale, shift``, that ``norm_out`` normalises at 1e-6 where every other norm in the
model uses 1e-5, that it is not affine, and that the position table reaches the action rows
and not the state row.

The last section is the alternation and the two spans. ``_attends_encoder`` says *whether* a
block cross-attends -- the parity of the index, and it starts with a cross block at every
depth. ``_span_for`` says *which* rows a cross block reads, and that is ``AlternateVLDiT``'s
rule rather than the plain ``DiT``'s (``dit.py:379-396``): odd blocks self-attend, and an even
block reads the text rows when ``idx % (2 * n) == 0`` and the image rows otherwise. So
``attend_text_every_n_blocks`` is not the cross/self rule -- ``_attends_encoder`` takes an
index and could not read the field if it wanted to -- but it *is* the text/image rule, which
is what ``_span_for`` reads it for. Beside those two: that each block is handed the attention
registered under its own span, that a span with no entry falls back to ``_SDPA_ATTEND``, and
that the fallback is one object for the life of the module rather than one per step.

Every value test over the stack runs a **shallow** one: the real stack is 32 blocks of
1536 x 32 x 48, 1.12B parameters and 4.2 GiB per fp32 copy, so the only places the full 32
blocks exist are the two key-set tests, on ``meta``, where nothing is allocated.
"""

import glob
import inspect
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from safetensors import safe_open
from torch import nn

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
from mstar.model.gr00t_n17.weight_loader import NODE_PREFIXES

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)
# the DT lives under `action_head.model.`, which the `denoise_step` node owns
_CKPT_PREFIX = "action_head.model.timestep_encoder."

# Spelled out rather than imported from the modules under test: these are the numbers the
# tests below exist to catch changing there.
_TIME_PROJ = 256  # the DT's projection width, and the width the two encodings are compared at
_BASE = 10000.0  # the frequency base both encodings use
_CHECKPOINT_DIM = 1536  # `input_embedding_dim`: the width both encodings run at in the real model
_EMBED = 48  # a small DT width to fill with weights

# Denoising timesteps: both ends of a flow-matching schedule, a fraction between them, and
# two integers, since the DDPM form these encodings came from indexes discrete steps.
_TIMESTEPS = (0.0, 0.25, 1.0, 7.0, 999.0)


@pytest.fixture(autouse=True)
def _pinned_threads():
    """fp32 CPU forwards compared with ``torch.equal``; reduction order follows the thread
    count, as it does in the other gr00t tests."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _flat() -> torch.Tensor:
    """``[B]``, which is what the DT's encoder takes."""
    return torch.tensor(_TIMESTEPS)


def _rows(shape: tuple[int, int] = (1, len(_TIMESTEPS))) -> torch.Tensor:
    """``[B, T]``, which is what the action encoder's takes -- and only that. The timesteps
    repeat to fill a shape bigger than the list, which also puts a repeated value in a batch.
    """
    count = shape[0] * shape[1]
    return torch.tensor((_TIMESTEPS * (count // len(_TIMESTEPS) + 1))[:count]).reshape(shape)


def _randomize(module: nn.Module, seed: int, scale: float = 0.1) -> None:
    """Scaled down so the SiLU between the two linears is not saturated on every row. The
    DT block's section below fills wider, for a reason stated there."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(scale * torch.randn(parameter.shape, generator=generator))


def _encoder(embedding_dim: int = _EMBED, seed: int = 0) -> TimestepEncoder:
    """Ours, filled. Deterministic in ``seed``, so two calls give two identical modules."""
    encoder = TimestepEncoder(embedding_dim)
    _randomize(encoder, seed)
    return encoder


# ----------------------------------------------------------------------------
# Inline reference: the action encoder's sinusoid, written from its description
# ----------------------------------------------------------------------------

# Deliberate deviations. Each is a convention a reader would expect instead, and each would
# load the checkpoint and run, because this module has no weights.
_WRONG_DIVISOR = "half_dim - 1"  # the usual form, and the one the DT's own encoding uses
_WRONG_ORDER_COS_FIRST = "cos then sin"  # what `flip_sin_to_cos=True` does
_WRONG_ORDER_INTERLEAVED = "sin and cos interleaved"  # the RoPE-style layout
_WRONG_BASE = "base 1000"


def _reference(timesteps: torch.Tensor, embedding_dim: int, deviation: str | None = None) -> torch.Tensor:
    """``exponent = -arange(half_dim) * (log(base) / divisor)``, ``freqs = t * exponent.exp()``,
    then the sine and the cosine of every frequency concatenated, sines first.

    The divisor is ``half_dim`` and not the usual ``half_dim - 1``, and the log is taken
    through torch rather than ``math``, both of which follow the reference op for op.
    """
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


# ----------------------------------------------------------------------------
# The action encoder's sinusoid against the reference
# ----------------------------------------------------------------------------


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
    """Pure arithmetic with no GEMM in it -- the same ops in the same order on the same
    tensors -- so this is ``torch.equal`` and not a bar.

    The checkpoint's width is in here because it is the one case where ``half_dim`` (768) is
    not a power of two, so ``log(base) / half_dim`` is inexact and the route the port takes
    through the arithmetic starts to matter; see the two tests against diffusers below.
    """
    timesteps = _rows(shape)
    out = SinusoidalPositionalEncoding(embedding_dim)(timesteps)

    assert out.shape == (*shape, embedding_dim)
    assert torch.equal(out, _reference(timesteps, embedding_dim))


@pytest.mark.parametrize(
    "shape", [(), (len(_TIMESTEPS),), (2, 3, 4)], ids=["a scalar", "a [B] vector", "rank 3"],
)
def test_a_timestep_that_is_not_rank_2_is_refused_where_it_arrives(shape):
    """The reference takes ``[B, T]`` and nothing else, and keeping that is the only thing
    standing between the two encodings at a call site. ``[B]`` is the dangerous rank: it is
    the DT encoder's own input shape, and this module would otherwise answer it with
    ``[B, embedding_dim]`` -- the DT encoder's own output shape -- raising nothing and
    producing plausible magnitudes under the wrong convention. So the message names both
    modules rather than just the rank.
    """
    with pytest.raises(ValueError, match="TimestepEncoder is the one that takes"):
        SinusoidalPositionalEncoding(_TIME_PROJ)(torch.zeros(shape))
    assert SinusoidalPositionalEncoding(_TIME_PROJ)(_rows()).shape == (1, len(_TIMESTEPS), _TIME_PROJ)


def test_the_sinusoid_has_no_parameters_so_only_its_values_pin_it():
    """The premise of this whole section: there is no weight whose shape or value would give
    a wrong convention away, and no buffer either -- the ladder is rebuilt every forward.
    """
    module = SinusoidalPositionalEncoding(_TIME_PROJ)
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    assert module.state_dict() == {}


@torch.no_grad()
def test_timestep_zero_is_every_sine_then_every_cosine():
    """The order argued from values rather than from a second copy of the formula: at
    timestep 0 every frequency is 0, so every sine is exactly 0 and every cosine exactly 1,
    whatever the base and the spacing are. Sines first means ``[0, ..., 0, 1, ..., 1]``.

    The two wrong readings give the same multiset in a different arrangement -- cos first is
    the halves swapped, interleaved is ``[0, 1, 0, 1, ...]`` -- so neither is this tensor,
    and nothing about the frequencies has to be known to say so.
    """
    half_dim = _TIME_PROJ // 2
    zeros = torch.zeros(1, 3)
    at_zero = SinusoidalPositionalEncoding(_TIME_PROJ)(zeros)

    assert torch.equal(
        at_zero, torch.cat([torch.zeros(1, 3, half_dim), torch.ones(1, 3, half_dim)], dim=-1)
    )
    for deviation in (_WRONG_ORDER_COS_FIRST, _WRONG_ORDER_INTERLEAVED):
        assert not torch.equal(at_zero, _reference(zeros, _TIME_PROJ, deviation=deviation))


@torch.no_grad()
def test_the_frequency_ladder_divides_by_half_dim_and_not_half_dim_minus_one():
    """The divisor argued from values. Dividing by ``half_dim`` makes the ladder a function of
    the *fraction* of the way down the channels: frequency ``j`` of a width-``2h`` sinusoid is
    ``base ** (-j / h)``, which is frequency ``2j`` of a width-``4h`` one. Dividing by
    ``half_dim - 1`` breaks that -- ``base ** (-j / (h - 1))`` against
    ``base ** (-2j / (2h - 1))`` -- and the relation says nothing about what the base is, so
    it isolates the divisor.

    Both widths' ``half_dim`` are powers of two here, so ``log(base) / half_dim`` is exact and
    the two ladders agree bit for bit rather than nearly. Measured: the real module matches
    exactly, where the ``half_dim - 1`` reference's two widths part company by 1.26 on the
    sine channels, which is most of a sine's range.
    """
    timesteps = _rows()
    narrow = SinusoidalPositionalEncoding(_TIME_PROJ)(timesteps)
    wide = SinusoidalPositionalEncoding(2 * _TIME_PROJ)(timesteps)
    half_dim = _TIME_PROJ // 2

    # the narrow sinusoid's sines, then its cosines, are the wide one's every other channel
    assert torch.equal(narrow[..., :half_dim], wide[..., : 2 * half_dim : 2])
    assert torch.equal(narrow[..., half_dim:], wide[..., 2 * half_dim :: 2])

    wrong_narrow = _reference(timesteps, _TIME_PROJ, deviation=_WRONG_DIVISOR)
    wrong_wide = _reference(timesteps, 2 * _TIME_PROJ, deviation=_WRONG_DIVISOR)
    assert not torch.equal(wrong_narrow[..., :half_dim], wrong_wide[..., : 2 * half_dim : 2])


@torch.no_grad()
def test_the_lowest_frequency_says_the_base_is_10000():
    """The base argued from values. The last sine channel at timestep 1 is
    ``sin(base ** (-(h - 1) / h))``, and at these frequencies ``sin x`` is ``x`` to ten
    significant figures, so that channel *is* the lowest frequency of the ladder.

    Measured at width 256: 1.0746e-04, where a base of 1000 gives 1.0554e-03 and a base of
    100000 gives 1.0941e-05 -- an order of magnitude either side, against a bracket of 2.5%.
    The bracket also excludes the ``half_dim - 1`` divisor at this base, which puts the lowest
    frequency at exactly ``1 / base``, 9.9999e-05; both of those are checked below rather
    than only quoted.
    """
    one = torch.ones(1, 1)
    half_dim = _TIME_PROJ // 2
    ladder = SinusoidalPositionalEncoding(_TIME_PROJ)(one)[0, 0]

    assert 1.05e-04 < ladder[half_dim - 1].item() < 1.10e-04
    for deviation in (_WRONG_BASE, _WRONG_DIVISOR):
        wrong = _reference(one, _TIME_PROJ, deviation=deviation)[0, 0, half_dim - 1].item()
        assert not 1.05e-04 < wrong < 1.10e-04
    # and the ladder starts at a frequency of exactly 1, which the span above is measured from
    assert ladder[0] == torch.sin(torch.tensor(1.0))


# ----------------------------------------------------------------------------
# The two encodings, which are not the same encoding
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("embedding_dim", [_TIME_PROJ, 8], ids=["the compared width", "a narrow one"])
@torch.no_grad()
def test_the_sinusoid_is_diffusers_ladder_unflipped_and_unshifted(embedding_dim):
    """Base, divisor and order at once, against code this test did not write: the port is
    diffusers' sinusoid with ``flip_sin_to_cos=False`` and ``downscale_freq_shift=0``, whose
    divisor is then ``half_dim - 0`` and whose ``max_period`` is 10000 by default. Where the
    tests above state the convention, this one borrows it from the library the DT's half
    already imports, so the three numbers are not this file's opinion.

    Bit-exact, which is not free: ours takes the log through torch and divides it once,
    diffusers takes it through ``math`` and divides the whole product. It holds at these
    widths because ``half_dim`` is a power of two, so the division is exact either way --
    and not at the checkpoint's width, which is the next test.
    """
    theirs = Timesteps(num_channels=embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=0)
    ours = SinusoidalPositionalEncoding(embedding_dim)(_rows())[0]

    assert torch.equal(ours, theirs(_flat()))


@torch.no_grad()
def test_the_diffusers_route_to_the_same_ladder_misses_at_the_checkpoints_width():
    """Why the port spells the arithmetic out instead of calling ``Timesteps(1536, False, 0)``,
    which is the same math by a different route. At 1536 ``half_dim`` is 768, not a power of
    two, so ``log(10000) / 768`` is inexact and the route shows: measured max gap 6.09e-05
    against a sinusoid's range of 2.

    That is the awkward size. It is far too large to be called rounding, and far too small
    for any bar a reader would set to separate a wrong convention (the crossed encodings below
    differ by 1.93), so there is no tolerance that would accept the wrong route and reject the
    wrong convention. Hence ``torch.equal`` against the reference's own ops, and hence the
    module's note about taking the log through torch.
    """
    ours = SinusoidalPositionalEncoding(_CHECKPOINT_DIM)(_rows())[0]
    route = Timesteps(num_channels=_CHECKPOINT_DIM, flip_sin_to_cos=False, downscale_freq_shift=0)

    gap = (ours - route(_flat())).abs().max().item()
    assert not torch.equal(ours, route(_flat()))
    assert 1e-05 < gap < 1e-03


@torch.no_grad()
def test_the_dts_projection_is_the_flipped_ladder_and_not_the_action_encoders():
    """The mistake this cycle is most exposed to. Both encodings are a 256-wide sinusoid of
    the denoising timestep, so crossing them loads every weight and runs. What they are not
    is the same function, and both of diffusers' knobs differ:

    - the flip, read off timestep 0: the DT's is every cosine then every sine, ours the
      other way round. One row at ``t = 0`` identifies which convention produced a tensor.
    - the shift, read off the lowest frequency: the DT's divides by ``half_dim - 1``, which
      puts it at exactly ``1 / 10000`` (9.9999e-05) where ours sits at 1.0746e-04.

    Nothing weaker than that separates them. Each is 128 sines and 128 cosines, so every
    row of both has L2 norm ``sqrt(128)`` to the bit -- no magnitude, norm or finiteness
    check can tell them apart -- while they differ by up to 1.93 element for element.
    """
    half_dim = _TIME_PROJ // 2
    ours = SinusoidalPositionalEncoding(_TIME_PROJ)
    dt = _encoder().time_proj
    assert dt.num_channels == _TIME_PROJ  # same width, so the two are directly comparable

    mine, theirs = ours(_rows())[0], dt(_flat())
    assert not torch.equal(mine, theirs)
    assert torch.equal(mine.norm(dim=-1), theirs.norm(dim=-1))
    assert (mine - theirs).abs().max() > 1.0

    # the flip: at timestep 0 the DT's cosines come first, and ours come second
    assert torch.equal(dt(torch.zeros(3)), torch.cat([torch.ones(3, half_dim), torch.zeros(3, half_dim)], -1))
    assert torch.equal(
        ours(torch.zeros(1, 3))[0], torch.cat([torch.zeros(3, half_dim), torch.ones(3, half_dim)], -1)
    )

    # the shift: the DT's lowest frequency is 1 / 10000, where ours is 10000 ** (-127 / 128)
    their_lowest = dt(torch.ones(1))[0, -1].item()  # flipped, so the sines are the second half
    our_lowest = ours(torch.ones(1, 1))[0, 0, half_dim - 1].item()
    assert their_lowest == pytest.approx(1.0 / _BASE, rel=1e-06)
    assert our_lowest / their_lowest == pytest.approx(_BASE ** (1.0 / half_dim), rel=1e-06)


# ----------------------------------------------------------------------------
# `TimestepEncoder`: the DT's conditioning path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("embedding_dim", [_EMBED, 16], ids=["a small DT width", "a smaller one"])
def test_the_projection_is_256_wide_whatever_the_dt_is(embedding_dim):
    """256 is not a config field. It is what the checkpoint's ``linear_1`` was trained against
    (1536 x 256), so it must not track the DT's width, and ``linear_1`` is the only place the
    two widths meet. ``time_proj`` holds nothing, so the four names below are the module's
    whole state dict, and they are the checkpoint's keys.
    """
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


@torch.no_grad()
def test_the_encoder_is_the_projection_through_two_linears_with_a_silu_between():
    """``[B] -> [B, embedding_dim]``: one conditioning vector per request's denoising step,
    which is what AdaLN reads. The arithmetic is diffusers' own two linears and its own
    activation in the same order, so this is ``torch.equal`` and not a bar -- it pins the
    wiring, which is ours, rather than the linears, which are not.
    """
    encoder = _encoder()
    timesteps = _flat()
    out = encoder(timesteps)
    embedder = encoder.timestep_embedder

    assert out.shape == (len(_TIMESTEPS), _EMBED)
    assert torch.equal(out, embedder.linear_2(F.silu(embedder.linear_1(encoder.time_proj(timesteps)))))


@torch.no_grad()
def test_the_projection_is_rounded_to_the_weights_dtype_before_the_embedder():
    """diffusers builds the sinusoid in fp32 whatever the module's dtype is, and the reference
    casts it to the weights' dtype on the way in. In bf16 that is a real rounding of the
    conditioning vector rather than a formality: it drops 16 bits of every channel before the
    first GEMM, and the answer is not what running the path in fp32 and rounding at the end
    gives (measured max gap 3.9e-03, on an output whose rows are O(1)).

    A module that did not cast could not run at all -- a bf16 ``nn.Linear`` on an fp32 input
    raises -- but that is the easy half. The equality below is what says the rounding lands
    before the embedder rather than after it.
    """
    bf16 = _encoder().to(torch.bfloat16)
    fp32 = _encoder()  # the same weights, unrounded, because `_encoder` is deterministic
    timesteps = _flat()
    projection = bf16.time_proj(timesteps)

    assert projection.dtype == torch.float32
    assert torch.equal(bf16(timesteps), bf16.timestep_embedder(projection.to(torch.bfloat16)))
    # the rounding moves the projection, so where it happens is visible in the output
    assert not torch.equal(projection.to(torch.bfloat16).float(), projection)
    assert not torch.equal(bf16(timesteps), fp32(timesteps).to(torch.bfloat16))


@pytest.mark.skipif(not _GR00T.exists(), reason="checkpoint not cached")
def test_the_checkpoints_timestep_encoder_tensors_name_this_modules_parameters():
    """Where the 256 comes from. The DT is 1536 wide, so ``linear_1`` is 1536 x 256 and
    nothing but the sinusoid's width explains the 256. Shapes come off the safetensors
    header, so no tensor is read, and the module is built on meta.
    """
    with torch.device("meta"):
        encoder = TimestepEncoder(_CHECKPOINT_DIM)
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    checkpoint = {
        name[len(_CKPT_PREFIX) :]: (name, shard)
        for name, shard in weight_map.items()
        if name.startswith(_CKPT_PREFIX)
    }

    assert set(checkpoint) == {name for name, _ in encoder.named_parameters()}
    for name, parameter in encoder.named_parameters():
        key, shard = checkpoint[name]
        with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
            assert list(parameter.shape) == shard_file.get_slice(key).get_shape()


# ----------------------------------------------------------------------------
# `MultiEmbodimentActionEncoder`: the noisy action and the denoising time in
# ----------------------------------------------------------------------------

_CATEGORIES = 4
_ACTION_DIM = 5  # a different number from the width, so the two cannot be confused
_HIDDEN = 8  # the encoder's width; even, because half of the sinusoid is sines
_HORIZON = 3
# The embedded action and the sinusoid then have comparable RMS (measured 0.72 against 0.71),
# which is what makes the concatenation-order test below a test of the order: the two halves
# are 2.34 apart element for element, but neither is the larger tensor.
_ACTION_SCALE = 3.0
# one id per timestep, out of order, repeated, and spanning both ends of the category axis
_CAT_IDS = (3, 0, 1, 3, 2)

# Deliberate deviations, as in the sinusoid's reference above: the readings of the five steps
# that a reader could plausibly expect instead, each of which loads every weight and runs.
_ORDER_TAU_FIRST = "tau first"
_SWISH_BEFORE_W2 = "before W2"
_SWISH_BETWEEN = "between"
_SWISH_AFTER_W3 = "after W3"
_SWISH_NONE = "none"


def _action_encoder(seed: int = 1) -> MultiEmbodimentActionEncoder:
    """Filled, and scaled by ``_randomize`` so the swish runs in the part of its range where
    the two spellings of it are not the same function -- see the ``F.silu`` test."""
    encoder = MultiEmbodimentActionEncoder(_CATEGORIES, _ACTION_DIM, _HIDDEN)
    _randomize(encoder, seed)
    return encoder


def _actions(batch: int = len(_CAT_IDS), horizon: int = _HORIZON, seed: int = 8) -> torch.Tensor:
    return _ACTION_SCALE * torch.randn(
        batch, horizon, _ACTION_DIM, generator=torch.Generator().manual_seed(seed),
    )


def _config() -> Gr00tN17Config:
    """The fields the two builders read. The DT's ``output_dim`` (6, what the decoder takes
    in), the DT's ``inner_dim`` (3 x 4 = 12, what it must not take), ``hidden_size`` (9, the
    decoder's middle), ``max_action_dim`` (5) and ``input_embedding_dim`` (8) are five
    different numbers here, so none of them can stand in for another.
    """
    return Gr00tN17Config(
        text=None, vision=None, bridge=None,
        dit=Gr00tN17StackConfig(
            num_layers=1, num_attention_heads=3, attention_head_dim=4, output_dim=6,
        ),
        deepstack_visual_indexes=(), backbone_embedding_dim=0,
        max_action_dim=_ACTION_DIM, hidden_size=9, input_embedding_dim=_HIDDEN,
        max_num_embodiments=_CATEGORIES,
    )


# ----------------------------------------------------------------------------
# Inline reference: the action encoder, written from its five steps
# ----------------------------------------------------------------------------


def _per_category(linear, x: torch.Tensor, cat_ids: list[int]) -> torch.Tensor:
    """``x @ W[id] + b[id]``, with the category axis indexed here, row by row, rather than
    gathered -- a mis-gather returns a plausibly shaped tensor, so the indexing is the half
    worth writing out twice.

    The matmul itself is ``torch.bmm`` at the width the module calls it, which is the one
    thing this reference borrows rather than rederives. A row-at-a-time 2-D ``@`` disagrees
    with it in the last ulp on CPU fp32 (measured max gap 3.7e-08 on the tensors below), and
    the small-integer trick that dodges this in ``test_gr00t_n17_state.py`` is not available
    here: the sinusoid's sines and cosines sit in the middle of the arithmetic, so ``W2``'s
    and ``W3``'s inputs cannot be made exactly representable. What that costs is the
    independent check that a batched ``bmm`` is a row-at-a-time matmul -- which is exactly
    what that file pins, on the same ``CategorySpecificLinear`` this module is built from.
    """
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
    through this file's own sinusoid reference; the two concatenated, action first; ``W2``;
    a swish; ``W3``.

    ``order``, ``swish`` and ``silu`` are the deviation knobs. The sinusoid comes from
    ``_reference`` above rather than from the module, so the time half is this file's
    arithmetic too.
    """
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


# ----------------------------------------------------------------------------
# The action encoder against the reference
# ----------------------------------------------------------------------------

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
    """``[B, T, action_dim]`` and one timestep per request to ``[B, T, hidden]``.

    ``torch.equal``, which holds because the reference calls the same matmul at the same
    width; everything around the matmuls -- the spread, the sinusoid, the concatenation, the
    swish and its placement -- is this file's. The horizon varies because the timestep is
    spread along it and the biases broadcast over it.
    """
    encoder = _action_encoder()
    cat_ids = torch.tensor(ids)
    actions = _actions(len(ids), horizon)
    timesteps = _flat()[: len(ids)]
    out = encoder(actions, timesteps, cat_ids)

    assert out.shape == (len(ids), horizon, _HIDDEN)
    assert torch.equal(out, _encoder_reference(encoder, actions, timesteps, cat_ids))


@torch.no_grad()
def test_the_concatenation_reads_the_action_before_the_time():
    """``W2`` is one matrix over the concatenation, so swapping the halves is a permutation of
    its input rows -- no shape changes, nothing raises, and the checkpoint loads.

    Two things make this a test of the order rather than of the magnitudes. The halves are
    the same size and, with this fill, nearly the same scale (RMS 0.72 against 0.71), so the
    swap is not visible as the output growing or shrinking. And they are far apart element
    for element (2.34 at the widest), which is what makes the permutation move anything: the
    swap shifts the output by 0.13 against an output whose largest element is 0.38.
    """
    encoder = _action_encoder()
    cat_ids = torch.tensor(_CAT_IDS)
    actions = _actions()
    timesteps = _flat()
    out = encoder(actions, timesteps, cat_ids)

    action = encoder.W1(actions, cat_ids)
    tau = encoder.pos_encoding(timesteps.unsqueeze(1).expand(-1, _HORIZON))
    assert action.shape == tau.shape  # a swap a shape check could not catch
    assert (action - tau).abs().max() > 1.0
    assert action.pow(2).mean().sqrt() == pytest.approx(tau.pow(2).mean().sqrt(), rel=0.1)

    swapped = _encoder_reference(encoder, actions, timesteps, cat_ids, order=_ORDER_TAU_FIRST)
    assert not torch.equal(out, swapped)
    assert (out - swapped).abs().max() > 0.1 * out.abs().max()


@torch.no_grad()
def test_the_swish_is_x_times_sigmoid_x_and_not_f_silu():
    """The reference is computed both ways and only one of them is the module's output, which
    is the whole claim -- there is nothing else to assert here that is not a second copy of
    the implementation.

    ``F.silu`` is the same function in exact arithmetic and a different one in floating point.
    On the tensor ``W2`` produces here the two disagree on a third of the elements (measured
    32.5%) by up to 3.0e-08, which survives ``W3`` into a 1.5e-08 gap -- enough for
    ``torch.equal``, and small enough that a cleanup pass would read ``x * torch.sigmoid(x)``
    as long-hand for ``F.silu``. The number that says it is not comes from the real
    checkpoint's trained embodiments: the substitution moves the encoder's output by up to
    1.5e-05 in fp32 and by **0.50 in bf16**, the dtype this model serves in, on 40% of its
    elements.

    The gap this test sees is small because ``_randomize`` keeps ``W2``'s output O(1) (absmax
    0.96 here), and that is deliberate: where the sigmoid saturates, both spellings round to
    the same thing on every element. Measured with a fill 100x wider, ``W2``'s absmax is
    6.2e+03 and the two outputs are bit-identical -- so a test written at that scale would
    pass against either spelling.
    """
    encoder = _action_encoder()
    cat_ids = torch.tensor(_CAT_IDS)
    actions, timesteps = _actions(), _flat()
    out = encoder(actions, timesteps, cat_ids)

    assert torch.equal(out, _encoder_reference(encoder, actions, timesteps, cat_ids))
    assert not torch.equal(
        out, _encoder_reference(encoder, actions, timesteps, cat_ids, silu=True),
    )


@pytest.mark.parametrize(
    "swish", [_SWISH_BEFORE_W2, _SWISH_AFTER_W3, _SWISH_NONE],
    ids=["a swish before W2", "a swish after W3", "no swish at all"],
)
@torch.no_grad()
def test_the_swish_sits_between_w2_and_w3(swish):
    """The only nonlinearity in this module, so with it gone ``W2`` and ``W3`` collapse into
    one linear; moved to either outside edge it is a different function again. Each is a
    different answer, or the parity above would hold for a model that is not this one.
    """
    encoder = _action_encoder()
    cat_ids = torch.tensor(_CAT_IDS)
    actions, timesteps = _actions(), _flat()
    assert not torch.equal(
        encoder(actions, timesteps, cat_ids),
        _encoder_reference(encoder, actions, timesteps, cat_ids, swish=swish),
    )


_BAD_TIMESTEPS = {
    # what the sinusoid itself takes, so a caller who spread the timestep first
    "already spread to [B, T]": torch.zeros(len(_CAT_IDS), _HORIZON),
    "one timestep for the batch": torch.tensor(0.5),
    "one timestep too few": torch.zeros(len(_CAT_IDS) - 1),
}


@pytest.mark.parametrize("timesteps", list(_BAD_TIMESTEPS.values()), ids=list(_BAD_TIMESTEPS))
def test_a_timestep_that_is_not_one_per_request_is_refused_here(timesteps):
    """This module takes ``[B]`` and spreads it; the sinusoid inside it takes ``[B, T]`` and
    refuses ``[B]``. Two neighbouring modules wanting different ranks of the same quantity is
    how a call site gets it wrong, so the error has to name what the caller owes rather than
    what some tensor downstream found.

    Each of these does raise without the guard -- but as an ``expand`` with too few sizes, an
    ``unsqueeze`` dimension out of range, and a ``torch.cat`` size mismatch, none of which
    mentions the timestep. Hence the match on the text and not just on the raise.
    """
    encoder = _action_encoder()
    with pytest.raises(ValueError, match="one timestep per request"):
        encoder(_actions(), timesteps, torch.tensor(_CAT_IDS))


@torch.no_grad()
def test_the_time_encoding_is_cast_to_the_actions_dtype():
    """The sinusoid builds itself in fp32 whatever the module's dtype is, so in bf16 -- which
    is the dtype the checkpoint runs in -- the concatenation has an fp32 half and a bf16 half.
    ``torch.cat`` raises on that, and no fp32 test above can see it.
    """
    encoder = _action_encoder().to(torch.bfloat16)
    out = encoder(_actions().to(torch.bfloat16), _flat(), torch.tensor(_CAT_IDS))

    assert encoder.pos_encoding(_rows()).dtype == torch.float32
    assert out.dtype == torch.bfloat16


# ----------------------------------------------------------------------------
# What the two builders are sized from
# ----------------------------------------------------------------------------


def test_the_encoder_and_decoder_are_sized_from_the_config():
    """``W2`` twice the width is the concatenation, stated in shapes: a module that added the
    two halves instead of concatenating them would be this module with ``W2`` half as deep.

    The decoder's input is the DT's ``output_dim`` and *not* its inner width, which is the
    distinction this cycle is most likely to get wrong -- the DT's own blocks are all
    ``inner_dim`` wide, and its last projection is the only place ``output_dim`` appears.

    Only the widths, because the decoder is a plain ``CategorySpecificMLP``: its relu -- which
    is not the encoder's swish -- and its per-embodiment indexing are pinned in
    ``test_gr00t_n17_state.py``, on the same class.
    """
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


# the two action-head modules this section builds, and the checkpoint prefix of each
_ACTION_PREFIXES = {
    "the action encoder": (build_action_encoder, "action_head.action_encoder."),
    "the action decoder": (build_action_decoder, "action_head.action_decoder."),
}


@pytest.mark.parametrize("build, prefix", list(_ACTION_PREFIXES.values()), ids=list(_ACTION_PREFIXES))
@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_action_tensors_name_these_modules_parameters(build, prefix):
    """Both modules keep the reference's own attribute names (``W1`` / ``W2`` / ``W3``,
    ``layer1`` / ``layer2``), so the checkpoint's keys land on them with only ``action_head.``
    stripped. Shapes come off the safetensors header, so no tensor is read.

    This is where the two widths above are pinned against something that is not a config read
    the same way twice. The encoder's ``W2`` is 32 x 3072 x 1536 -- twice the 1536 either side
    of it, which is the concatenation. The decoder's ``layer1`` is 32 x 1024 x 1024, so it
    takes the DT's ``output_dim`` (1024) and not its inner width (1536); that one would be a
    shape error at load, and this is the test that reports it as a width instead.

    Nothing here reads a weight, which is just as well: only embodiment slots 24-27 of the 32
    are trained, and the other 28 still hold NVIDIA's ``0.02 * randn`` initialisation with
    all-zero biases. A numeric test against this checkpoint has to pick one of those four and
    say so -- on an untrained slot it multiplies by near-isotropic noise and passes for a port
    that is merely plausible. ``embodiment_id.json`` is no guide: the ids it names land on
    untrained slots.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):  # 32 x 3072 x 1536 of it, allocated nowhere
        module = build(config)
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    checkpoint = {
        name[len(prefix) :]: (name, shard)
        for name, shard in weight_map.items()
        if name.startswith(prefix)
    }

    assert set(checkpoint) == {name for name, _ in module.named_parameters()}
    for name, parameter in module.named_parameters():
        key, shard = checkpoint[name]
        with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
            assert list(parameter.shape) == shard_file.get_slice(key).get_shape()


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_real_decoder_takes_the_dts_output_width_in_and_an_action_out():
    """The two numbers in one place: 1024 in, 132 out. 1024 is the DT's ``output_dim``, which
    is also ``hidden_size`` in this checkpoint, so ``layer1`` is square and its shape alone
    does not say which field it came from -- what rules out the inner width is that 1536 is
    neither of its two axes.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):
        decoder = build_action_decoder(config)

    assert config.dit.inner_dim == 1536
    assert decoder.layer1.W.shape == (32, 1024, 1024)
    assert decoder.layer2.W.shape == (32, 1024, 132)


# ----------------------------------------------------------------------------
# Inline reference: `AdaLayerNorm`, written from its one line
# ----------------------------------------------------------------------------

# A tiny DT. Every width here is a different number, so no test below can pass by reading
# one where it meant another: the block is 12 wide, its attention is 3 heads x 5 = 15 wide
# inside, and the bridge's rows it cross-attends are 7. In the real model the first two are
# both 1536 and only the bridge's 2048 stands apart, which is why they are spread out here.
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

# Deliberate deviations, as in the two references above. Each is a reading of the one line
# that a reader could plausibly expect instead, and each loads the checkpoint and runs.
_WRONG_SHIFT_FIRST = "shift first out of the chunk"
_WRONG_BARE_SCALE = "scale and not 1 + scale"
# `[:, None]` dropped. Only computable where the request count equals the token count, which
# is the whole hazard: at any other shape it raises, at that one shape it silently gives
# token t request t's conditioning.
_WRONG_PER_TOKEN = "the pair indexed per token"
_SWISH_LONGHAND = "x * sigmoid(x) for the activation"


def _adaln_reference(
    norm: AdaLayerNorm, x: torch.Tensor, temb: torch.Tensor, deviation: str | None = None,
) -> torch.Tensor:
    """``layer_norm(x) * (1 + scale[:, None]) + shift[:, None]``, with ``scale`` first out of
    the chunk and the activation ``F.silu``.

    The normalisation is written out as ``F.layer_norm`` with no weight and no bias rather
    than read off the module, so the parity test pins the inner norm's two settings -- not
    affine, eps 1e-5 -- as well as the arithmetic around it. Only the ``linear`` is borrowed.
    """
    activation = temb * torch.sigmoid(temb) if deviation == _SWISH_LONGHAND else F.silu(temb)
    first, second = norm.linear(activation).chunk(2, dim=1)
    scale, shift = (second, first) if deviation == _WRONG_SHIFT_FIRST else (first, second)
    normed = F.layer_norm(x, (x.shape[-1],), eps=_EPS)
    if deviation == _WRONG_PER_TOKEN:
        return normed * (1 + scale) + shift
    gain = scale[:, None] if deviation == _WRONG_BARE_SCALE else 1 + scale[:, None]
    return normed * gain + shift[:, None]


def _adaln(seed: int = 20) -> AdaLayerNorm:
    """Ours, filled. ``_randomize``'s default scale keeps the SiLU off its saturated ends,
    which is what the activation test below needs."""
    norm = AdaLayerNorm(_DT_DIM)
    _randomize(norm, seed)
    return norm


def _temb(requests: int = _REQUESTS, seed: int = 21) -> torch.Tensor:
    """``[B, dim]``: one conditioning vector per request, which is what ``TimestepEncoder``
    above returns. O(1), so the SiLU runs where its two spellings are not the same function.
    """
    return torch.randn(requests, _DT_DIM, generator=torch.Generator().manual_seed(seed))


def _hidden(requests: int = _REQUESTS, tokens: int = _QUERIES, seed: int = 22) -> torch.Tensor:
    """``[B, T, dim]``: the DT's own action tokens."""
    return torch.randn(
        requests, tokens, _DT_DIM, generator=torch.Generator().manual_seed(seed),
    )


# ----------------------------------------------------------------------------
# `AdaLayerNorm` against the reference
# ----------------------------------------------------------------------------

_ADALN_SHAPES = {
    "a batch of requests": (_REQUESTS, _QUERIES),
    "as many tokens as requests": (3, 3),
    "one request": (1, _QUERIES),
    "one token each": (3, 1),
}


@pytest.mark.parametrize("shape", list(_ADALN_SHAPES.values()), ids=list(_ADALN_SHAPES))
@torch.no_grad()
def test_adalayernorm_is_the_reference_block(shape):
    """One GEMM, one normalisation, a multiply and an add, in that order on the same tensors,
    so this is ``torch.equal`` and not a bar.

    The square shape is here because it is the only one where the ``[:, None]`` broadcast can
    be dropped without raising; the test below says what it would cost.

    Parity alone would also pass against an inner norm that was affine at its default
    initialisation (weight 1, bias 0). What rules that out numerically is ``_randomize``,
    which fills every parameter the module has -- so an affine norm would arrive here with a
    random gain and miss. The key set is pinned separately below.
    """
    requests, tokens = shape
    norm = _adaln()
    x = _hidden(requests, tokens)
    temb = _temb(requests)
    out = norm(x, temb)

    assert out.shape == (requests, tokens, _DT_DIM)
    assert torch.equal(out, _adaln_reference(norm, x, temb))


@torch.no_grad()
def test_the_scale_comes_first_out_of_the_chunk():
    """``linear`` is one 2x-wide projection, so the halves are interchangeable in shape and a
    swap raises nothing, loads every weight, and leaves the output's magnitude where it was.
    One of the two is multiplied and the other added, which is the only thing that tells
    them apart -- measured here as a 0.97 move on an output whose largest element is 2.55.
    """
    norm = _adaln()
    x, temb = _hidden(), _temb()
    out = norm(x, temb)
    swapped = _adaln_reference(norm, x, temb, _WRONG_SHIFT_FIRST)

    assert not torch.equal(out, swapped)
    assert (out - swapped).abs().max() > 0.1 * out.abs().max()


@torch.no_grad()
def test_the_scale_is_a_gain_about_one_and_not_the_gain_itself():
    """``1 + scale`` makes a zero ``linear`` the identity, which is how a DiT block starts
    trained: zeros in, the block passes its input through its normalisation untouched. Bare
    ``scale`` makes the same weights annihilate the tokens instead, which is the concrete
    difference rather than a second copy of the formula.

    The trained fill is checked too, since zeros are a special case: on it the two readings
    are 2.33 apart against an output whose largest element is 2.55.
    """
    zeroed = AdaLayerNorm(_DT_DIM)
    with torch.no_grad():
        for parameter in zeroed.parameters():
            parameter.zero_()
    x, temb = _hidden(), _temb()

    assert torch.equal(zeroed(x, temb), F.layer_norm(x, (_DT_DIM,), eps=_EPS))
    assert torch.equal(_adaln_reference(zeroed, x, temb, _WRONG_BARE_SCALE), torch.zeros_like(x))

    norm = _adaln()
    bare = _adaln_reference(norm, x, temb, _WRONG_BARE_SCALE)
    assert not torch.equal(norm(x, temb), bare)


@torch.no_grad()
def test_one_requests_pair_reaches_every_one_of_its_tokens():
    """Half of R10: the conditioning is per request, not per token. Argued from values rather
    than from the formula -- every token of a request is given the same row here, so a per
    request pair has to come back as identical rows, and the per-token reading gives each
    token a different gain and breaks that.

    The shape is square on purpose. ``scale`` is ``[B, dim]`` and the tokens are
    ``[B, T, dim]``, so dropping the ``[:, None]`` raises at every shape except this one,
    where it quietly applies request t's conditioning to token t of every request -- a 1.07
    move against a 2.28 output, and the kind of mistake that a 1-token test would never see.
    """
    requests = tokens = 3
    norm = _adaln()
    row = _hidden(1, 1)
    x = row.expand(requests, tokens, _DT_DIM).contiguous()
    temb = _temb(requests)
    out = norm(x, temb)

    for request in range(requests):
        for token in range(1, tokens):
            assert torch.equal(out[request, token], out[request, 0])
    per_token = _adaln_reference(norm, x, temb, _WRONG_PER_TOKEN)
    assert not torch.equal(out, per_token)
    assert (out - per_token).abs().max() > 0.1 * out.abs().max()


@torch.no_grad()
def test_a_batch_can_hold_requests_at_different_denoising_steps():
    """The other half of R10, and the reason the DT can batch at all: each row's answer
    depends on that row's timestep and on no other row's. Moving one request's ``temb``
    moves its own tokens and leaves its neighbours' bit-identical.

    Bit-identical, not close: the batch shape is the same in both runs, so the GEMM blocks
    its reductions the same way. (Running one request alone is *not* exact -- a 1-row
    ``linear`` and a 3-row one disagree by 2.4e-07 here -- so a "batching is independence"
    test written as batch against singleton would have to carry a bar, and would be about
    the GEMM rather than about this module.)
    """
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


def test_the_inner_norm_is_not_affine_because_the_checkpoint_has_no_weight_for_it():
    """``norm1`` carries ``linear.weight`` and ``linear.bias`` and nothing else, so an affine
    inner norm would arrive with two tensors the checkpoint cannot fill. It would also load
    and run: ``nn.LayerNorm``'s affine parameters initialise to the identity, so
    ``load_state_dict`` with ``strict=False`` anywhere upstream would leave them at 1 and 0
    and the forward would be right until someone trained it.
    """
    norm = _adaln()

    assert {name for name, _ in norm.named_parameters()} == {"linear.weight", "linear.bias"}
    assert norm.norm.weight is None and norm.norm.bias is None
    assert norm.norm.eps == _EPS


# ----------------------------------------------------------------------------
# Inline reference: one DT block, written from its six steps
# ----------------------------------------------------------------------------

# The block's fill, wider than `_randomize`'s default because the attention's contribution
# has to be comparable to the residual it is added to. At 0.1 the attention output's largest
# element is 0.18 against a block output of 2.61, so dropping the attention residual moves
# the answer by 1.4% -- a wiring test that reads as a rounding. At 0.3 every deviation below
# moves it by at least a third of its largest element.
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
    """A CPU stand-in for the resource C07 will bind: ``[B, T, heads, head_dim]`` in and out,
    unmasked, every query over every key. It is the contract's own shape, which is why the
    same callable can serve the module and the reference below -- so the comparison is of
    the wiring around the attention and is exact, with no attention backend in it.
    """
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _block(cross: bool, seed: int = 23) -> BasicTransformerBlock:
    """One block at either attention setting, filled. Nothing is bound here: the attention is
    a ``forward`` argument, so every call below says which one it is handing this block."""
    block = BasicTransformerBlock(
        _DT_DIM,
        _DT_HEADS,
        _DT_HEAD_DIM,
        cross_attention_dim=_BRIDGE_DIM if cross else None,
    )
    _randomize(block, seed, _BLOCK_SCALE)
    return block


def _bridge_rows(
    cross: bool, requests: int = _REQUESTS, rows: int = _BRIDGE_ROWS, seed: int = 24,
) -> torch.Tensor | None:
    """What a cross block is handed: the bridge's rows, ``[B, S, 2048]`` in the real model.
    A self block is handed nothing, which is what makes its k and v its own tokens."""
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

    The AdaLN is this file's reference from above, the normalisation before the feed-forward
    is written out, and the residuals and the order are this file's. Two things are borrowed:
    the projections, and ``block.ff`` -- diffusers' own ``FeedForward``, whose gating, widths
    and activation are pinned on the same class in ``test_gr00t_n17_bridge.py`` and are not
    this cycle's claim.

    ``attend`` is the attention this reference reads with, defaulting to the stand-in: the
    module takes one per call now, so the span tests below hand both sides the same one and
    stay exact.
    """
    # the width comes off the tensor, not from `_DT_DIM`: `DiT`'s own section below builds
    # its blocks at the inner width and hands them to this same reference
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


# ----------------------------------------------------------------------------
# The block against the reference, at both attention settings
# ----------------------------------------------------------------------------

# The alternation: the even DT blocks cross-attend the bridge's rows, the odd ones attend
# their own tokens. Everything below runs at both settings, because the two differ only in
# where k and v come from and that is the half most easily got backwards.
_SETTINGS = {"a cross block": True, "a self block": False}


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@pytest.mark.parametrize(
    "requests, queries", [(_REQUESTS, _QUERIES), (1, 1)],
    ids=["a batch of requests", "one request, one token"],
)
@torch.no_grad()
def test_the_block_is_the_reference_block(cross, requests, queries):
    """``torch.equal``, and it can be: the attention is the caller's callable, so the module
    and the reference reach the same one on the same tensors and nothing in between is a
    kernel this file did not call itself.
    """
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
    """What makes the parity above a test. Each of these is the same six pieces in a
    different arrangement, each loads the checkpoint and runs at every shape, and each is
    the arrangement of some other transformer block: a pre-norm block that keeps only one
    residual, a post-norm block that normalises after the feed-forward, a block that
    conditions its feed-forward instead of its attention, and a parallel block whose two
    halves both read the input.
    """
    block = _block(cross)
    hidden, temb = _hidden(), _temb()
    encoder = _bridge_rows(cross)
    out = block(hidden, temb, encoder, _sdpa)
    wrong = _block_reference(block, hidden, temb, encoder, deviation)

    assert not torch.equal(out, wrong)
    # a floor, not a fitted threshold: over weight seeds 20-31 x both settings the weakest
    # of these five moved the answer by 0.26 of its largest element, the norm3-after-the-
    # feed-forward reading every time -- the only one of the five that keeps both residuals
    assert (out - wrong).abs().max() > 0.2 * out.abs().max()


@pytest.mark.parametrize("cross", list(_SETTINGS.values()), ids=list(_SETTINGS))
@torch.no_grad()
def test_k_and_v_come_from_the_setting_and_q_always_from_the_tokens(cross):
    """The alternation's correctness, and what the checkpoint's shapes encode: a cross
    block's ``to_k`` and ``to_v`` are 2048 wide against the DT's own 1536, so they can only
    be fed the bridge's rows, and a self block's are 1536 and can only be fed its tokens.

    The widths are half the claim; the other half is which tensor is actually projected,
    which the stand-in reports. Note what k and v are in the self setting: the AdaLN output,
    not the block's input -- the conditioning reaches the keys as well as the queries.
    """
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
    """The contract C07's resource will be written against, pinned on both sides. In:
    ``[B, T, heads, head_dim]`` for q and ``[B, S, heads, head_dim]`` for k and v, where S is
    the bridge's row count in a cross block and the query count in a self one. Out: whatever
    it returns is folded back with ``reshape(B, T, -1)``, so the heads are concatenated in
    the order they were split.

    An attention that returns its queries untouched is what makes the second half checkable
    without a second copy of the fold: the view and the reshape have to round-trip, so the
    block's attention output must be exactly ``to_out.0`` of the unsplit ``to_q``. A
    ``[B, heads, T, head_dim]`` layout -- the one every kernel in M* and torch wants, and the
    one ``_sdpa`` transposes into -- would pass a shape check and fold the heads back
    interleaved.
    """
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
    """The callable is still the only seam, but it moved from the constructor to the call, and
    the enforcement has to move with it. Both halves are here.

    No default on ``forward``, so a block nobody hands an attention to does not run at all. A
    block that fell back to its own tokens instead would be a running model with every second
    block ignoring the language and the images, which no shape and no loss curve reports --
    the hazard the old ``RuntimeError`` named, now carried by the signature.

    And no ``attend`` on ``__init__``, which is the new half: one attention per block for the
    life of the stack cannot express the three spans the stack now reads (the odd blocks' own
    rows, the text rows, the image rows), so a constructor that still accepted one would be a
    block holding an attention for a span it is not being asked for. The spans are per call
    because the attention is.
    """
    block = BasicTransformerBlock(_DT_DIM, _DT_HEADS, _DT_HEAD_DIM)
    hidden, temb = _hidden(), _temb()

    with pytest.raises(TypeError):
        block(hidden, temb, None)
    with pytest.raises(TypeError):
        BasicTransformerBlock(_DT_DIM, _DT_HEADS, _DT_HEAD_DIM, attend=_sdpa)

    # handing one in is the whole of what the stack has to do, and it is positional
    parameters = list(inspect.signature(block.forward).parameters)
    assert parameters == ["hidden_states", "temb", "encoder_hidden_states", "attend"]
    assert all(
        inspect.signature(block.forward).parameters[name].default is inspect.Parameter.empty
        for name in parameters
    )
    assert "attend" not in inspect.signature(BasicTransformerBlock.__init__).parameters
    assert block(hidden, temb, None, _sdpa).shape == (_REQUESTS, _QUERIES, _DT_DIM)


@torch.no_grad()
def test_adalns_activation_is_f_silu_where_the_action_encoders_is_the_long_hand():
    """One model, the same function twice, two spellings, and they are not bit-identical: a
    cleanup that unified them would move one of the two. Both directions are here so the
    test says which spelling belongs where rather than that the two differ.

    ``nn.SiLU`` and ``x * sigmoid(x)`` disagree on a fifth of the elements (measured 20.4%
    here) by up to 2.4e-07, which survives into AdaLN's output. The action encoder's own
    section above measures what the substitution costs there: 0.50 in bf16 on a trained
    embodiment.
    """
    norm = _adaln()
    x, temb = _hidden(), _temb()
    out = norm(x, temb)

    assert torch.equal(out, _adaln_reference(norm, x, temb))
    assert not torch.equal(out, _adaln_reference(norm, x, temb, _SWISH_LONGHAND))

    encoder = _action_encoder()
    cat_ids = torch.tensor(_CAT_IDS)
    actions, timesteps = _actions(), _flat()
    encoded = encoder(actions, timesteps, cat_ids)

    assert torch.equal(encoded, _encoder_reference(encoder, actions, timesteps, cat_ids))
    assert not torch.equal(
        encoded, _encoder_reference(encoder, actions, timesteps, cat_ids, silu=True),
    )


# ----------------------------------------------------------------------------
# What the checkpoint says a block is
# ----------------------------------------------------------------------------

_BLOCK_PREFIX = "action_head.model.transformer_blocks."


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_two_block_variants_name_these_modules_parameters():
    """32 blocks, 14 tensors each, and two variants: ``attn1.to_k`` and ``to_v`` are 2048
    wide on the even blocks and 1536 on the odd ones. That is the whole of what says the
    alternation starts with cross-attention -- building it the other way round makes every
    block's k and v the wrong width here.

    The 14 also say what a block does *not* have: no ``norm2``, which is where diffusers'
    own block keeps the norm before its cross-attention, and no ``norm3.weight`` or
    ``norm3.bias``, which is the non-affine LayerNorm again. Shapes come off the safetensors
    header, so no tensor is read, and both variants are built on meta.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):
        variants = {
            cross: BasicTransformerBlock(
                config.dit.inner_dim,
                config.dit.num_attention_heads,
                config.dit.attention_head_dim,
                cross_attention_dim=config.backbone_embedding_dim if cross else None,
            )
            for cross in (True, False)
        }
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    checkpoint: dict[int, dict[str, tuple[str, str]]] = {}
    for name, shard in weight_map.items():
        if name.startswith(_BLOCK_PREFIX):
            index, _, key = name[len(_BLOCK_PREFIX) :].partition(".")
            checkpoint.setdefault(int(index), {})[key] = (name, shard)

    assert sorted(checkpoint) == list(range(config.dit.num_layers))
    # the bridge's rows are as wide as the backbone's, which is why one field sizes both
    assert config.backbone_embedding_dim == config.bridge.inner_dim

    for index, tensors in checkpoint.items():
        # the rule is the parity of the index and nothing else, which is what
        # `_attends_encoder` says; `attend_text_every_n_blocks` is a different question and
        # happens to give the same answer at its one value (see its own test below)
        cross = _attends_encoder(index)
        block = variants[cross]
        assert len(tensors) == 14
        assert set(tensors) == {name for name, _ in block.named_parameters()}
        for name, parameter in block.named_parameters():
            key, shard = tensors[name]
            with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
                assert list(parameter.shape) == shard_file.get_slice(key).get_shape()


# ----------------------------------------------------------------------------
# Inline reference: the DT's output head, and the stack it sits on
# ----------------------------------------------------------------------------

# `DiT`'s own widths, which are not the block section's: there the block is 12 wide and its
# attention 15 inside, where `DiT` builds every block at `dim = stack.inner_dim` -- as in the
# real model, where both are 1536. So the numbers that stay distinct here are the stack's
# width (15), the bridge's rows it cross-attends (7), its output width (6) and the position
# table's (8 wide, 9 rows).
_DIT_DIM = _DT_HEADS * _DT_HEAD_DIM
_DIT_OUT = 6
_DIT_DEPTH = 2  # one of each kind of block, which is the shortest stack that alternates
# the position table: `max_seq_len` x `input_embedding_dim`, 1024 x 1536 in the real model,
# of which only the first `action_horizon` (40) rows are ever read
_TABLE_ROWS = 9
_POS_DIM = _HIDDEN
_ACTION_ROWS = 3

# `norm_out`'s eps. Spelled out, like `_EPS` above, because it is the number this section
# exists to pin: the reference hardcodes 1e-6 at the output norm (`dit.py:284`) and takes the
# config's 1e-5 everywhere else, and no weight and no bf16 dump records an eps.
_OUT_EPS = 1e-6

# The real schedule's four timesteps: `int(t / num_inference_timesteps * num_timestep_buckets)`
# for t = 0..3, which is what the reference fills `timesteps_tensor` with
# (`gr00t_n1d7.py:398-404`). A batch holds one per request and not the same one in each row:
# requests reach the DT at whichever denoising iteration they are on.
_DENOISE_STEPS = (0.0, 250.0, 500.0, 750.0)

# Deliberate deviations, as in the sections above. Each is a reading of the head's three
# lines that a reader could expect instead, and each loads the checkpoint and runs.
_WRONG_HEAD_SCALE_FIRST = "scale first out of the head's chunk"
_WRONG_HEAD_BLOCK_EPS = "the blocks' 1e-5 at the output norm"
# `[:, None]` dropped, as in `_WRONG_PER_TOKEN` above: computable only where the request
# count equals the row count, and a different answer there.
_WRONG_HEAD_PER_ROW = "the head's pair indexed per row"


def _dit_config(depth: int = _DIT_DEPTH) -> Gr00tN17Config:
    """The fields `DiT` and `ActionPositionEmbedding` read. ``input_embedding_dim`` (8) is a
    different number from the stack's width (15) on purpose: the table is sized from the one
    and the blocks from the other, and in the real checkpoint both are 1536.
    """
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
    """A **shallow** DT, filled.

    Shallow is not a convenience. The real stack is 32 blocks of 1536 x 32 x 48 -- 1.12B
    parameters, 4.2 GiB in fp32 -- so nothing in this file builds one off ``meta``; the
    key-set test below is the only place the full stack appears, and it allocates nothing.
    Filled at ``_BLOCK_SCALE`` for the reason the block section gives.
    """
    dit = DiT(_dit_config(depth))
    _randomize(dit, seed, _BLOCK_SCALE)
    return dit


# Every span read with this file's own stand-in. Handed to each `DiT` forward below, so the
# module and the reference reach the same callable on the same tensors and the comparison
# holds no attention backend this file did not call itself -- which is what keeps the parity
# tests `torch.equal`. The module's own fallback is bit-identical to it, which the span
# section at the end of the file pins rather than assuming.
_STAND_INS = dict.fromkeys((SA_SPAN, VL_TEXT_SPAN, VL_IMAGE_SPAN), _sdpa)


def _dit_rows(
    requests: int = _REQUESTS, rows: int = _QUERIES, seed: int = 25, scale: float = 1.0,
) -> torch.Tensor:
    """``[B, T, dim]``: the state row and the action rows the node concatenates, which is what
    the DT takes in -- 41 of them in the real model."""
    return scale * torch.randn(
        requests, rows, _DIT_DIM, generator=torch.Generator().manual_seed(seed),
    )


def _pos_rows(requests: int = _REQUESTS, rows: int = _ACTION_ROWS, seed: int = 26) -> torch.Tensor:
    """``[B, T, input_embedding_dim]``: what the action encoder returns, which is the only
    thing the position table is added to."""
    return torch.randn(requests, rows, _POS_DIM, generator=torch.Generator().manual_seed(seed))


def _steps(requests: int = _REQUESTS) -> torch.Tensor:
    """``[B]``: one denoising timestep per request, and a *different* one per request -- the
    serving case, where a batch holds requests at different iterations of the schedule."""
    return torch.tensor(_DENOISE_STEPS[:requests])


def _position_table(seed: int = 27) -> ActionPositionEmbedding:
    """Filled, which is not optional: the module allocates its table with ``torch.empty``, so
    an unfilled one holds whatever the allocator last left there."""
    table = ActionPositionEmbedding(_dit_config())
    _randomize(table, seed, _BLOCK_SCALE)
    return table


def _head_reference(
    dit: DiT, hidden: torch.Tensor, temb: torch.Tensor, deviation: str | None = None,
) -> torch.Tensor:
    """``shift, scale = proj_out_1(silu(temb)).chunk(2, dim=1)`` -- **that** order -- then
    ``norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]``, then ``proj_out_2``.

    The normalisation is written out as ``F.layer_norm`` with no weight, no bias and eps 1e-6
    rather than read off the module, so the parity test pins all three of ``norm_out``'s
    settings. Only the two projections are borrowed.
    """
    first, second = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
    shift, scale = (second, first) if deviation == _WRONG_HEAD_SCALE_FIRST else (first, second)
    eps = _EPS if deviation == _WRONG_HEAD_BLOCK_EPS else _OUT_EPS
    normed = F.layer_norm(hidden, (hidden.shape[-1],), eps=eps)
    if deviation == _WRONG_HEAD_PER_ROW:
        return dit.proj_out_2(normed * (1 + scale) + shift)
    return dit.proj_out_2(normed * (1 + scale[:, None]) + shift[:, None])


def _reference_span(index: int, attend_text_every_n_blocks: int) -> str:
    """Which span block ``index`` reads, written out from ``AlternateVLDiT.forward``
    (``dit.py:379-396``) -- the variant this checkpoint selects, and the one with two spans.

    The reference carries one set of encoder rows and switches the *mask* over them, so the
    rule reads off its three branches: ``idx % 2 == 1`` takes the self-attention branch; an
    even block takes ``non_image_attention_mask`` when ``idx % (2 * n) == 0`` and
    ``image_attention_mask`` otherwise. ``non_image`` is the text rows, which is the span this
    port calls ``vl_text``.

    Written out here rather than imported, so neither this reference nor the span tests below
    borrow their expectation from ``_span_for``, the function they exist to check.
    """
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
    """The timestep encoder, then every block through this file's own block reference, then
    the head. The alternation and the text/image split are ``_reference_span`` above rather
    than ``_attends_encoder`` and ``_span_for``, so the stack's own rules are not borrowed
    from the module it tests.

    ``attends`` maps a span to the attention that span's blocks read with, as the module's
    does, and defaults to the stand-in everywhere -- so a test that supplies its own for one
    span has an exact expectation rather than a bar.
    """
    temb = dit.timestep_encoder(timestep)
    rows = {SA_SPAN: None, VL_TEXT_SPAN: text, VL_IMAGE_SPAN: image}
    for index, block in enumerate(dit.transformer_blocks):
        span = _reference_span(index, dit.attend_text_every_n_blocks)
        # `deviation` is a reading of the *head*, which is where it is applied; the blocks
        # here are the right wiring in every case, as they were before the spans existed
        hidden = _block_reference(
            block, hidden, temb, rows[span], None, (attends or {}).get(span, _sdpa),
        )
    return _head_reference(dit, hidden, temb, deviation)


# ----------------------------------------------------------------------------
# The whole DT against the reference
# ----------------------------------------------------------------------------

# Every test in this section hands the *same* rows as `vl_text` and as `vl_image`, which is
# not a shortcut around the split: with the two spans equal, every cross block reads all of
# the bridge's rows whichever span it asked for, and that is exactly the reference's plain
# `DiT.forward` -- the unmasked case C06 measured parity against. So these tests go on saying
# what they said about the head, the norms and the batching, and they say it at both readings
# of the split at once. The split itself is pinned in the next section, where the two spans
# are deliberately different lengths.
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
    """``[B, T, 1536]`` rows and the bridge's ``[B, S, 2048]`` to ``[B, T, output_dim]``, from
    one ``[B]`` timestep. ``torch.equal``, for the same reason the block's parity is: the
    attention is the caller's callable, so the module and the reference reach the same one --
    here one per span, all three of them this file's stand-in.

    Four depths, because a stack that alternates cannot be pinned at one. Depth 1 says block
    0 is a cross block, depth 2 that block 1 is not, and depths 3 and 4 separate ``index % 2``
    from the readings that agree with it on the first two blocks -- ``index < depth // 2``,
    ``index % 3``, ``index % 4``.

    The square shape is here for the head's ``[:, None]``, as it is in the AdaLN section: it
    is the one shape where dropping it raises nothing. One request and one row is the real
    serving edge, and the only shape where every tensor in the forward is square.
    """
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
    """The head chunks ``shift, scale`` where ``AdaLayerNorm`` three classes up chunks
    ``scale, shift`` (``dit.py:331`` against ``dit.py:201``). One projection, two halves of
    equal width, both learned and both plausible in magnitude -- here the trained-style fill
    gives them RMS 1.42 and 1.80 -- so a swap loads every weight, raises nothing, and leaves
    the output's scale where it was. Nothing but a value comparison sees it.

    So the swap is forced into view rather than measured. ``proj_out_1`` is rebuilt with its
    first half exactly 0 and its second half exactly -1, which makes the right reading
    ``norm_out(x) * (1 + -1) + 0`` -- exactly zero on every row, whatever the blocks did --
    so the DT's output is ``proj_out_2.bias`` and nothing else. Read the other way round the
    same weights give ``norm_out(x) * 1 - 1``, which is the normalised rows shifted, and
    4.03 in absolute value against the 0.50 the right reading produces.

    The trained-style fill is checked too, since the constructed one is degenerate: there the
    swap moves the answer by 1.39 times its own largest element.
    """
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
    """An eps is invisible to parity against a bf16 dump (C05): it moves the answer by far
    less than bf16's own floor on any ordinary input. So it is pinned on an input chosen to
    make it visible instead -- rows whose variance across the normalised dim is 8.5e-07, which
    is the same order as the eps itself, so the eps is most of the denominator rather than a
    correction to it. ``sqrt(v + 1e-5) / sqrt(v + 1e-6)`` is 2.3 there, and the DT's output
    moves by 0.45 of its own largest element.

    The stack is built with **no blocks**, which is the only way to hand ``norm_out`` an input
    this test chooses: a block's own normalisation would wash out any variance set here.
    """
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
    """``action_head.model.`` holds ``norm_out`` nowhere: the whole stack's tensors are the
    timestep encoder's four, 14 per block, and the two projections' four. An affine
    ``norm_out`` would want two more -- and would load and run without them, because
    ``nn.LayerNorm``'s affine parameters initialise to 1 and 0, so the forward would be right
    until someone trained it. Parity cannot see that; the key set is the test.

    The parity test above would also catch it, but only as a side effect: ``_randomize`` fills
    every parameter the module has, so an affine norm would arrive there with a random gain
    rather than with the identity. This names it directly.
    """
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
    """The head's half of R10. ``scale`` is ``[B, dim]`` and the rows are ``[B, T, dim]``, so
    dropping the ``[:, None]`` raises at every shape except ``B == T``, where it quietly gives
    row ``t`` request ``t``'s conditioning -- a 1.17x move on an output whose rows are O(5),
    and a mistake no 1-row and no non-square test would ever reach.

    Made by comparison with the wrong reading rather than by comparing a request's rows to
    each other, which cannot be exact: with identical rows in, the GEMMs block their
    reductions over the row axis and one block's output rows differ in the last ulp (measured
    2.4e-07). The comparison below is ``torch.equal`` on both sides.
    """
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

    # at any other shape the dropped index is a shape error, which is why the square one is
    # the shape this test runs at
    with pytest.raises(RuntimeError):
        _dit_reference(dit, _dit_rows(), _bridge_rows(True), _bridge_rows(True), _steps(), _WRONG_HEAD_PER_ROW)


@torch.no_grad()
def test_a_dit_batch_can_hold_requests_at_different_denoising_steps():
    """The serving case, end to end: the requests in one batch are each at their own iteration
    of the four-step schedule, so every row's answer must depend on that row's timestep and on
    no other row's. Moving one request's timestep to the last step of the schedule moves its
    own rows and leaves its neighbours' bit-identical.

    Bit-identical because the batch shape does not change between the two runs. Permuting the
    batch is *not* exact through the stack -- the GEMMs reblock and the answer moves by
    9.5e-07 -- so this asserts isolation, which is the claim, rather than equivariance, which
    would need a bar and would be about the GEMM.
    """
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
# The alternation, the two spans it divides into, and the attention each one is read with
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@torch.no_grad()
def test_the_stack_is_built_and_run_as_cross_then_self_from_block_zero(depth):
    """``_attends_encoder`` is read twice -- once to size ``to_k``/``to_v`` and once to decide
    what to hand each block -- and the two readings have to agree. They are checked against
    each other here, because neither alone is enough: a block handed 1536-wide rows will
    cross-attend them perfectly happily, so a forward that disagreed with the construction
    would raise nothing in the real model's self blocks.

    The forward half is read off the key count, not the width: a cross block is handed 6
    bridge rows and a self block its own 4, so the stand-in reports which source each block
    got. One stand-in serves all three spans and appends in call order, so its position in the
    list *is* the block index -- the blocks run in order. The construction half is the
    ``in_features``, which is what the checkpoint's 2048 pins (see the block-variant test
    below).

    Both spans are the same rows here, so this test is about cross against self only; which
    of the two a cross block reads is the next section's claim.

    Four depths, because an off-by-one in the rule survives a shallower test: ``index % 2``,
    ``index % 3`` and ``index % 4`` all agree on blocks 0 and 1.
    """
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
    assert _BRIDGE_ROWS != _QUERIES  # so the key count says where k came from
    assert seen == [_BRIDGE_ROWS if is_cross else _QUERIES for is_cross in cross]
    for kind in ("to_k", "to_v"):
        assert [
            getattr(block.attn1, kind).in_features for block in dit.transformer_blocks
        ] == [_BRIDGE_DIM if is_cross else _DIT_DIM for is_cross in cross]
    # and q is the block's own rows at every block, cross or self
    assert {block.attn1.to_q.in_features for block in dit.transformer_blocks} == {_DIT_DIM}


def test_the_cross_self_rule_is_the_parity_of_the_index_and_not_the_config_field():
    """Half of the field's story: it is **not** what decides cross against self.
    ``attend_text_every_n_blocks`` is 2 in this checkpoint and the blocks alternate in twos,
    so ``index % n == 0`` and ``_attends_encoder`` agree at every index -- a coincidence of
    one value, and the whole reason reading the field as the alternation is a bug waiting on
    a config change. ``_attends_encoder`` could not read it if it wanted to: it takes an index
    and has no config.

    What the field *does* govern is the text/image split among the cross blocks, which is
    ``_span_for``'s job and the next test's claim. The two are split apart because they are
    two rules over the same index, and a reader who merged them would get the alternation
    wrong at any ``n`` but 2.
    """
    config = _dit_config()
    indexes = range(8)
    cross = [_attends_encoder(index) for index in indexes]
    n = config.attend_text_every_n_blocks

    assert n == 2
    assert cross == [index % 2 == 0 for index in indexes]
    assert cross == [index % n == 0 for index in indexes]  # the coincidence, stated
    # at any other value of the field the two part company
    for other in (1, 3, 4):
        assert [index % other == 0 for index in indexes] != cross
    assert list(inspect.signature(_attends_encoder).parameters) == ["index"]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_text_image_rule_is_the_config_field_dividing_the_cross_blocks(n):
    """The other half: ``_span_for`` reads the field, and reads it for the one thing it
    governs. Against ``_reference_span``, which is ``AlternateVLDiT``'s three branches written
    out from the reference rather than taken from the module.

    The field divides the cross blocks and touches nothing else, which is the shape of the
    claim: at every ``n`` the set of blocks that read *some* encoder span is the same -- the
    even ones, ``_attends_encoder``'s rule -- while which of the two they read moves with
    ``n``. So a stack that read the field as the alternation would size half its ``to_k``
    tensors wrong, and a stack that ignored it in ``_span_for`` would feed the image blocks
    the text rows and run.

    Four values rather than the checkpoint's one, because at n = 2 the text blocks are
    ``idx % 4`` and several wrong readings agree with that: n = 1 makes every cross block a
    text block, and n = 3 and 4 move the period off a power of two.
    """
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
    """R7, at the depth the checkpoint actually has. The rule is a periodicity over the block
    index, so the place an off-by-one hides is the *phase*, and the phase is only visible at
    the real depth and against the real ``n``.

    Block 0 is a **text** block, because ``0 % 4 == 0``. Starting the alternation at the image
    rows instead -- which is what reading the reference's branches the other way round gives,
    and what ``idx % (2 * n) == 1`` or a 1-based index would give -- is the same 32 blocks, the
    same 456 tensors, the same widths and the same shapes end to end. Nothing but this
    comparison says which way round it is: the checkpoint's ``to_k`` widths distinguish cross
    from self and say nothing about text from image, and a bf16 dump of the whole stack would
    differ only where the two spans differ, which is a value difference and not a structural
    one.

    Built from the config dataclass, not from a stack: ``_span_for`` takes two ints, so the
    real 32 blocks cost nothing here.
    """
    config = _dit_config(32)
    n = config.attend_text_every_n_blocks
    spans = [_span_for(index, n) for index in range(config.dit.num_layers)]

    assert (config.dit.num_layers, n) == (32, 2)
    assert spans == [_reference_span(index, n) for index in range(config.dit.num_layers)]
    assert spans[0] == VL_TEXT_SPAN  # R7
    assert spans[:8] == [VL_TEXT_SPAN, SA_SPAN, VL_IMAGE_SPAN, SA_SPAN] * 2
    # 16 self blocks, and the 16 cross blocks split evenly -- the language is read as often as
    # the images, which is what `attend_text_every_n_blocks` = 2 buys
    assert Counter(spans) == {SA_SPAN: 16, VL_TEXT_SPAN: 8, VL_IMAGE_SPAN: 8}


# Three *different* kv lengths, so a span is told apart by what its attention was handed and
# not only by the label it was handed under. The DT's own rows are 41 in the real model too;
# the text span is shorter than one prompt's worth of tokens and the image span is one
# 256-token tile, which is the real asymmetry between the two.
_SA_ROWS = 41
_TEXT_ROWS = 37
_IMAGE_ROWS = 256


def _spy(label: str, seen: list) -> Callable:
    """An attention that records the span label it was registered under and the number of keys
    it was handed, then attends. The pair is the point: the label says which entry of
    ``attends`` the stack reached for and the key count says which rows it put in front of it,
    so a stack that read the right span with the wrong attention, or the right attention over
    the wrong span, is a different list either way.
    """
    def attend(q, k, v):
        seen.append((label, k.shape[1]))
        return _sdpa(q, k, v)

    return attend


def _zeros(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """An attention whose output is nothing, in the contract's own shape. Used where a test
    needs a *supplied* attention to be distinguishable from the fallback: it keeps ``to_out``
    and both residuals alive, so a block that ran it still moves its tokens, and it gives a
    different answer from SDPA's wherever SDPA's own output is not already zero.
    """
    return torch.zeros_like(q)


@torch.no_grad()
def test_each_block_is_handed_the_attention_its_own_span_asked_for():
    """The interface's whole claim in one list. One attention per span, each recording what it
    saw, over a stack deep enough to hold all three spans: text, self, image, self.

    What this catches that the parity tests above cannot: they pass the same rows as both
    encoder spans, so there a cross block reading the image entry for the text span is the
    same answer. Here the three lengths are 41, 37 and 256, so every such swap is a different
    list -- and a stack that read ``attends`` once outside the loop, or keyed it by the block
    index rather than by the span, would hand every cross block the same one entry.

    The parity assertion at the end is the other half: the rows a block is handed are the
    span's rows, which the reference's own span rule decides.
    """
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
    """A misspelt span key used to be indistinguishable from passing nothing.

    ``attends`` is a mapping, so an unknown key reaches no block and that span quietly
    falls back to SDPA -- bitwise identical to ``attends=None``, but off the resource the
    node planned and off the kernel a captured graph recorded. The engine side already
    raises for an unplanned label (``ragged attention has no plan for label ...``); this
    is the matching refusal on ours. Found by C07's blind QA, which measured the typo
    case as bitwise equal to the fallback.

    If this fails, a deployment can serve a span on the wrong kernel and look correct.
    """
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
    """The fallback, which is how the parity backend runs and how a deployment that declares
    no ragged resource still serves. Three readings of ``attends`` have to agree: ``None``, an
    empty map, and a map that names some labels and not others. A stack that raised on a
    missing label would make SDPA unreachable; one that skipped the block, or attended its own
    tokens instead, would run and be wrong.

    Then the half that makes it a fallback rather than a default: a label that *is* supplied is
    read with what was supplied, which is asserted against the reference with the same entry
    in it rather than against a bar.

    The last assertion is worth its line: the module's fallback is bit-identical to this
    file's stand-in, so every ``torch.equal`` in the sections above -- all of which hand the
    stand-in in explicitly -- is also a test of the default path.
    """
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
    """Identity, not equality, and the reason is not style. A compiled transformer region
    guards on the identity of the callables it is handed, so a fallback built inside the
    forward -- ``attends.get(span, functools.partial(joint_attention, ragged=None))``, which
    is the natural way to write it -- is a fresh object every step, retraces that frame every
    step, and keeps retracing until Dynamo gives up on it
    (``mstar/engine/resources/convenience.py:142-145``, where the ragged callables are held
    per label for exactly this reason). Nothing about the numbers changes, so no parity test
    in this file would see it; this is the only test that can.

    Every block's fourth argument is collected: over two forwards of one stack, over a second
    stack, and over a forward whose map covers one span and leaves the other two to fall back.
    One object throughout, and it is the module's own.
    """
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
# `ActionPositionEmbedding`: the table, and the row it is not added to
# ----------------------------------------------------------------------------

_POS_ROWS = {
    "the action horizon": _ACTION_ROWS,  # 40 rows in the real model
    "one action step": 1,
    "the whole table": _TABLE_ROWS,
}


@pytest.mark.parametrize("rows", list(_POS_ROWS.values()), ids=list(_POS_ROWS))
@torch.no_grad()
def test_the_table_adds_its_own_first_rows_and_reads_nothing_past_them(rows):
    """Rows ``0:T`` of the table, which is what the reference's
    ``position_embedding(arange(T))`` gathers -- the same rows in the same order, without the
    index tensor. The real table is 1024 rows against the 40 that are ever read, so the slice
    cannot run off the end and an off-by-one in it raises nothing: starting at row 1 instead
    is the same shape and a plausible tensor, and it is what adding the table after the state
    row is concatenated amounts to.

    The untouched tail is checked by moving it: the rows past ``T`` are perturbed and the
    answer has to be bit-identical, which says the slice is a slice and not a broadcast of
    some reduction over the table.
    """
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


@torch.no_grad()
def test_the_state_row_carries_no_position_because_the_table_lands_before_the_concat():
    """The reference adds the table to ``action_features`` and *then* concatenates the state
    row in front of it (``gr00t_n1d7.py:408-413``), so of the DT's 41 input rows the 40 action
    rows carry a position and the state row carries none.

    Adding the table after the concatenation instead is the same shape, raises nothing, loads
    the same weight -- and is wrong twice over: the state row picks up row 0 of a table that
    was never trained for it, and every action row picks up the row after its own. That moves
    the rows by 0.28 of the right answer's largest element here, and the bar below is a floor
    rather than a fitted threshold: over table seeds 20-31 the weakest move was 0.26.
    """
    table = _position_table()
    state = _pos_rows(rows=1, seed=28)
    actions = _pos_rows(rows=_ACTION_ROWS, seed=29)

    rows = torch.cat([state, table(actions)], dim=1)
    after = table(torch.cat([state, actions], dim=1))

    assert rows.shape == after.shape == (_REQUESTS, 1 + _ACTION_ROWS, _POS_DIM)
    assert torch.equal(rows[:, 0], state[:, 0])  # the state row, unpositioned
    assert torch.equal(rows[:, 1:], actions + table.weight[:_ACTION_ROWS])
    assert not torch.equal(after, rows)
    assert (after - rows).abs().max() > 0.2 * rows.abs().max()


# ----------------------------------------------------------------------------
# What the checkpoint says the `denoise_step` node is
# ----------------------------------------------------------------------------

# The node's four pieces and the tensor count of each. The 456 is 32 blocks x 14, the
# timestep encoder's 4 and the two output projections' 4 -- so it is also what says the stack
# has no `norm_out.weight`, no `norm_out.bias` and no 33rd block.
_DENOISE_PIECES = {"action_encoder": 6, "model": 456, "action_decoder": 4, "position_embedding": 1}


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_denoise_step_tensors_name_this_nodes_parameters():
    """Every tensor the node owns, named and shaped, against the four modules this file and
    the section above build. 467 of them, and the node's prefixes come from
    ``weight_loader`` rather than from a literal here, so a prefix moving there without a
    module moving with it fails this test.

    Built on ``meta``: the stack alone is 1.12B parameters, 4.2 GiB in fp32, and shapes come
    off the safetensors header, so nothing is read and nothing is allocated. This is the only
    place in the file where the full 32 blocks exist.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    root, prefixes = NODE_PREFIXES["denoise_step"]
    with torch.device("meta"):
        node = nn.Module()
        node.action_encoder = build_action_encoder(config)
        node.model = DiT(config)
        node.action_decoder = build_action_decoder(config)
        node.position_embedding = ActionPositionEmbedding(config)
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    checkpoint = {
        name[len(root) :]: (name, shard)
        for name, shard in weight_map.items()
        if name.startswith(prefixes)
    }

    assert set(checkpoint) == {name for name, _ in node.named_parameters()}
    assert Counter(name.partition(".")[0] for name in checkpoint) == _DENOISE_PIECES
    assert len(checkpoint) == 467
    for name, parameter in node.named_parameters():
        key, shard = checkpoint[name]
        with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
            assert list(parameter.shape) == shard_file.get_slice(key).get_shape()

    # the alternation in the checkpoint's own widths: 32 blocks, cross first
    assert (config.backbone_embedding_dim, config.dit.inner_dim) == (2048, 1536)
    assert [block.attn1.to_k.in_features for block in node.model.transformer_blocks] == [
        config.backbone_embedding_dim if index % 2 == 0 else config.dit.inner_dim
        for index in range(config.dit.num_layers)
    ]
    assert config.dit.num_layers == 32
    # 1024 rows for the 40 the DT reads, which is why a slice off the front never raises
    assert node.position_embedding.weight.shape == (config.max_seq_len, config.input_embedding_dim)
    assert node.position_embedding.weight.shape == (1024, 1536)
    assert config.action_horizon + 1 == 41 < config.max_seq_len
