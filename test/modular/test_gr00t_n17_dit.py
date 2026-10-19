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

The last section is ``MultiEmbodimentActionEncoder``, which is where that sinusoid is used:
the action through ``W1``, the time through the sinusoid, the two **concatenated** rather than
added, ``W2``, a swish spelled ``x * sigmoid(x)``, ``W3``. It lives in ``embodiment.py`` next
to the state encoder, but it is a DT piece and it depends on the sinusoid above, so it is
tested here. ``CategorySpecificLinear`` itself is pinned in ``test_gr00t_n17_state.py`` and
not again here.
"""

import glob
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from safetensors import safe_open
from torch import nn

from mstar.model.gr00t_n17.components.dit import TimestepEncoder
from mstar.model.gr00t_n17.components.embodiment import (
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    build_action_decoder,
    build_action_encoder,
)
from mstar.model.gr00t_n17.config import Gr00tN17Config, Gr00tN17StackConfig

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


def _randomize(module: nn.Module, seed: int) -> None:
    """Scaled down so the SiLU between the two linears is not saturated on every row."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator))


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
