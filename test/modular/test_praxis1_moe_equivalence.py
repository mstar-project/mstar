"""Numerical equivalence of the Praxis-1 feed-forward layers with the reference.

Companion to ``test_praxis1_reference_equivalence``: the same rule applies, that
the comparison is against the imported reference rather than against a second
copy of our own reading of it, and every comparison is bit-exact.

The reference is a checkout rather than a published package, so these tests
skip unless ``praxis`` is importable. Put the clone on ``PYTHONPATH``::

    PYTHONPATH=/path/to/praxis-1-oss pytest test/modular/test_praxis1_moe_equivalence.py

The negative controls carry most of the weight here. Six of the MoE's
conventions are inversions of what the DeepSeek-V3 family does, and every one
of them still produces gates that sum to one and an output of the right shape,
so a parity test alone cannot tell us we read the right one: it can only tell
us the two sides agree. Each control below therefore inverts exactly one
convention and asserts the result is a different number.

Two measured notes, both recorded as tests rather than as comments:

* the reference's missing ``1e-20`` in the renormalization denominator is a
  real difference from the DeepSeek and M* routers, but it is *inert* at any
  plausible score. It only moves a gate when the eight picked sigmoid scores
  sum to something near ``1e-20``, which takes logits around -50. Both regimes
  are asserted, so nobody has to guess which one they are in,
* the routed SwiGLU is the one place the reference does not match *itself*
  across devices: on CUDA it runs a hand-written kernel whose sigmoid rounds
  differently from ``F.silu``. These tests run on CPU and compare eager
  against eager; an accelerated port has to be checked against that kernel.

Both dispatch regimes are covered, unchunked (``dispatch_chunk_size`` null,
the reference's default) and chunked (what every published config sets).
The chunked regime's float32 sum order is invisible in bf16, measured, so
its float32 tests are the ones that protect it; see the notes in that
section.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip(
    "praxis",
    reason="the Praxis-1 reference checkout is not on PYTHONPATH",
)

import praxis  # noqa: E402
from praxis import layers as ref_layers  # noqa: E402

from mstar.model.praxis1.components import moe as port_moe  # noqa: E402
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError  # noqa: E402

# Small, but every shape the real layer cares about is still distinct: the
# hidden width is not the model width, 2 * hidden is not either, and there are
# more experts than a token picks.
DIM, MOE_HIDDEN, DENSE_HIDDEN = 64, 32, 160
NUM_EXPERTS, TOP_K, SCALE = 16, 8, 2.5

DTYPES = [torch.float32, torch.bfloat16]
SHAPES = [(5, DIM), (2, 7, DIM), (1, 1, DIM)]


@pytest.fixture(autouse=True)
def _one_thread():
    """Pin the intra-op pool.

    Parity here is safe either way, since both sides run in this process under
    the same pool. The negative controls are not: three of them assert that a
    reassociated float32 sum *does* change, and one asserts that a bf16 one
    does not, and both facts are properties of the reduction torch picks,
    which depends on the thread count.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fill(module: torch.nn.Module, seed: int) -> None:
    """Deterministic weights for every parameter, including the uninitialized
    ``torch.empty`` ones the reference's banks start as."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)


def _rand(shape, seed: int, dtype: torch.dtype, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen) * scale).to(dtype)


def _moe_pair(
    dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    top_k: int = TOP_K,
    seed: int = 0,
):
    """A reference MoE and our port, holding the same weights.

    The MoE parameter names are deliberately the reference's, so this is a
    plain ``load_state_dict``: if it ever needs a remap, a key drifted.
    """
    ref = ref_layers.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, top_k, SCALE)
    _fill(ref, seed)
    if bias is not None:
        with torch.no_grad():
            ref.expert_bias.copy_(bias)
    port = port_moe.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, top_k, SCALE)
    port.load_state_dict(ref.state_dict())
    return ref.to(dtype).eval(), port.to(dtype).eval()


def _dense_pair(dtype: torch.dtype = torch.float32, seed: int = 1):
    """A reference dense FFN and our port. The one place with a key remap: the
    reference is an ``nn.Sequential``, so it stores index names."""
    ref = ref_layers.SwiGLUFeedForward(DIM, DENSE_HIDDEN)
    _fill(ref, seed)
    sd = ref.state_dict()
    port = port_moe.SwiGLUFeedForward(DIM, DENSE_HIDDEN)
    port.load_state_dict(
        {
            "linear.a_weight": sd["0.linear.a_weight"],
            "linear.b_weight": sd["0.linear.b_weight"],
            "down.weight": sd["1.weight"],
        }
    )
    return ref.to(dtype).eval(), port.to(dtype).eval()


# A bias with enough spread to reorder the top-k of most tokens. Sigmoid
# scores live in (0, 1), so a bias of this size genuinely competes with them.
def _spread_bias(seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_EXPERTS, generator=gen) * 0.5


# ---------------------------------------------------------------------------
# The dense feed-forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_dense_ffn_matches_reference(dtype, shape):
    ref, port = _dense_pair(dtype)
    x = _rand(shape, seed=10, dtype=dtype)
    with torch.no_grad():
        want, got = ref(x), port(x)
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dense_projection_matches_reference(dtype):
    """The two halves separately, so a failure above points at the projection
    or at the activation rather than at either."""
    ref, port = _dense_pair(dtype)
    x = _rand((6, DIM), seed=11, dtype=dtype)
    with torch.no_grad():
        want_a, want_b = ref[0].linear(x)
        got_a, got_b = port.linear(x)
    assert torch.equal(got_a, want_a) and torch.equal(got_b, want_b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_dense_halves_are_not_interchangeable(dtype):
    """The activation is on the ``a`` half.

    Swapping the two projections keeps every shape and still produces a
    SwiGLU of the same weights, so the checkpoint is the only thing that says
    which half the ``silu`` goes on.
    """
    _, port = _dense_pair(dtype)
    x = _rand((6, DIM), seed=12, dtype=dtype)
    with torch.no_grad():
        a, b = port.linear(x)
        straight = port.down(F.silu(a) * b)
        swapped = port.down(F.silu(b) * a)
    assert not torch.equal(straight, swapped)


# ---------------------------------------------------------------------------
# The MoE, piece by piece
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_router_logits_match_reference(dtype):
    ref, port = _moe_pair(dtype)
    x = _rand((9, DIM), seed=20, dtype=dtype)
    with torch.no_grad():
        want = ref_layers._router_logits(ref.router.weight, x)
        got = port_moe.router_logits(port.router.weight, x)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("biased", [False, True])
def test_route_matches_reference(dtype, biased):
    bias = _spread_bias() if biased else None
    ref, port = _moe_pair(dtype, bias=bias)
    x = _rand((11, DIM), seed=21, dtype=dtype)
    with torch.no_grad():
        want_idx, want_gate = ref._route(x)
        got_idx, got_gate = port.route(x)
    assert got_gate.dtype == want_gate.dtype == dtype
    assert torch.equal(got_idx, want_idx)
    assert torch.equal(got_gate, want_gate)


@pytest.mark.parametrize("dtype", DTYPES)
def test_shared_expert_matches_reference(dtype):
    ref, port = _moe_pair(dtype)
    x = _rand((9, DIM), seed=22, dtype=dtype)
    with torch.no_grad():
        want, got = ref.shared(x), port.shared(x)
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dispatch_metadata_matches_reference(dtype):
    """The sort, the gather and the group offsets, which is where a port
    quietly loses the stable order."""
    from praxis.kernels.layers_triton import moe_dispatch

    ref, port = _moe_pair(dtype, bias=_spread_bias())
    x = _rand((13, DIM), seed=23, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        order = top_idx.reshape(-1).argsort(stable=True)
        want = moe_dispatch(
            x, top_idx, gate, order, NUM_EXPERTS, ref.routed.gate_up_weight.dtype
        )
        got = port.dispatch(x, top_idx, gate)
    names = ("rows", "sorted_gate", "inverse", "offsets")
    for name, g, w in zip(names, got, want, strict=True):
        assert g.dtype == w.dtype, name
        assert torch.equal(g, w), name


def test_dispatch_rounds_the_rows_to_the_expert_banks():
    """The row cast belongs to the dispatch, not to the block input.

    The served checkpoint is bf16 throughout, so inside ``forward`` the cast
    is a no-op and no end-to-end test can see it. Driving ``dispatch``
    directly with float32 rows against bf16 banks is the only way to check the
    boundary is where the reference puts it.
    """
    from praxis.kernels.layers_triton import moe_dispatch

    ref, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    x = _rand((13, DIM), seed=23, dtype=torch.float32)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        order = top_idx.reshape(-1).argsort(stable=True)
        want_rows = moe_dispatch(
            x, top_idx, gate, order, NUM_EXPERTS, ref.routed.gate_up_weight.dtype
        )[0]
        got_rows = port.dispatch(x, top_idx, gate)[0]
    assert got_rows.dtype == torch.bfloat16
    assert torch.equal(got_rows, want_rows)


@pytest.mark.parametrize("dtype", DTYPES)
def test_routed_forward_matches_reference(dtype):
    ref, port = _moe_pair(dtype, bias=_spread_bias())
    x = _rand((13, DIM), seed=24, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        want = ref._routed_forward(x, top_idx, gate)
        got = port.routed_forward(x, top_idx, gate)
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# The MoE, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("biased", [False, True])
def test_moe_forward_matches_reference(dtype, shape, biased):
    bias = _spread_bias() if biased else None
    ref, port = _moe_pair(dtype, bias=bias)
    x = _rand(shape, seed=25, dtype=dtype)
    with torch.no_grad():
        want, got = ref(x), port(x)
    assert got.shape == x.shape
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want)


@pytest.mark.parametrize("top_k", [1, 4, 8, NUM_EXPERTS])
def test_moe_forward_matches_reference_at_every_k(top_k):
    """Parity is not a property of the checkpoint's k alone, and k is also what
    the first control below varies, so both sides have to track it."""
    ref, port = _moe_pair(torch.bfloat16, bias=_spread_bias(), top_k=top_k)
    x = _rand((7, DIM), seed=26, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))


def test_expert_bias_stays_float32_under_a_module_cast():
    """``model.to(bfloat16)`` must not round the routing bias.

    The bias competes with float32 sigmoid scores. Rounded to bf16 it decides
    the near-ties it exists to decide by whichever way it happened to round,
    which is why both sides override ``_apply``.
    """
    ref, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    assert port.expert_bias.dtype == ref.expert_bias.dtype == torch.float32
    assert port.router.weight.dtype == torch.bfloat16
    assert torch.equal(port.expert_bias, ref.expert_bias)


# ---------------------------------------------------------------------------
# Negative controls: one inverted convention each
# ---------------------------------------------------------------------------


def test_control_top_k_is_load_bearing():
    """Top-8 is not top-anything-else."""
    _, port8 = _moe_pair(torch.bfloat16, bias=_spread_bias(), top_k=8)
    _, port4 = _moe_pair(torch.bfloat16, bias=_spread_bias(), top_k=4)
    x = _rand((7, DIM), seed=30, dtype=torch.bfloat16)
    with torch.no_grad():
        assert not torch.equal(port8(x), port4(x))


def test_control_bias_shifts_selection():
    """Half of "selection yes, weighting no": the bias must move the picks."""
    _, plain = _moe_pair(torch.float32)
    _, tilted = _moe_pair(torch.float32, bias=_spread_bias())
    x = _rand((24, DIM), seed=31, dtype=torch.float32)
    with torch.no_grad():
        idx_plain, _ = plain.route(x)
        idx_tilted, _ = tilted.route(x)
    assert not torch.equal(idx_plain, idx_tilted)


def test_control_bias_does_not_weight():
    """The other half: a bias that cannot change the picks must not change the
    gates either.

    A *uniform* bias shifts every score by the same amount, so the top-k order
    is untouched. The gates then have to come out identical, because they are
    read from the unbiased scores. An implementation that gathers from the
    biased scores instead passes every parity test about shapes and still
    fails here, which is the point.
    """
    _, plain = _moe_pair(torch.float32)
    _, lifted = _moe_pair(torch.float32, bias=torch.full((NUM_EXPERTS,), 0.75))
    x = _rand((24, DIM), seed=31, dtype=torch.float32)
    with torch.no_grad():
        idx_plain, gate_plain = plain.route(x)
        idx_lifted, gate_lifted = lifted.route(x)
    assert torch.equal(idx_plain, idx_lifted)
    assert torch.equal(gate_plain, gate_lifted)


def test_control_gathering_the_biased_scores_differs():
    """And the wrong gather is a different number, not a rounding of this one."""
    _, port = _moe_pair(torch.float32, bias=_spread_bias())
    x = _rand((24, DIM), seed=32, dtype=torch.float32)
    with torch.no_grad():
        logits = port_moe.router_logits(port.router.weight, x)
        scores = torch.sigmoid(logits)
        biased = scores + port.expert_bias
        _, top_idx = torch.topk(biased, TOP_K, dim=-1)
        unbiased_gate = scores.gather(-1, top_idx)
        biased_gate = biased.gather(-1, top_idx)
        unbiased_gate = unbiased_gate / unbiased_gate.sum(-1, keepdim=True)
        biased_gate = biased_gate / biased_gate.sum(-1, keepdim=True)
    # Both sum to one, which is why this inversion survives a sanity check.
    assert torch.allclose(unbiased_gate.sum(-1), torch.ones(x.shape[0]))
    assert torch.allclose(biased_gate.sum(-1), torch.ones(x.shape[0]))
    assert not torch.equal(unbiased_gate, biased_gate)


def test_control_topk_sorted_false_returns_another_permutation():
    """The two routers do not even return the same indices.

    M*'s shared ``NoAuxTcRouter`` passes ``sorted=False``. Measured on this
    input, torch then returns a different permutation of the same eight
    experts, so the two routers do not agree on ``top_idx``. How far that
    reaches into the output is the next two tests: in float32 it changes the
    combine, in bf16 eager it does not.
    """
    _, port = _moe_pair(torch.float32, bias=_spread_bias())
    x = _rand((24, DIM), seed=33, dtype=torch.float32)
    with torch.no_grad():
        biased = torch.sigmoid(
            port_moe.router_logits(port.router.weight, x)
        ) + port.expert_bias
        desc = torch.topk(biased, TOP_K, dim=-1)[1]
        any_order = torch.topk(biased, TOP_K, dim=-1, sorted=False)[1]
    assert torch.equal(desc.sort(-1).values, any_order.sort(-1).values)
    assert not torch.equal(desc, any_order)
    # And the sorted order really is by descending biased score.
    picked = biased.gather(-1, desc)
    assert torch.equal(picked, picked.sort(-1, descending=True).values)


def _slot_order_gap(dtype: torch.dtype, seed: int) -> int:
    """How many channels move when the eight slots are reversed.

    Reversing the slot order leaves the picked set, every gate value and every
    expert output untouched, so this isolates the order the combine sums in.
    """
    _, port = _moe_pair(dtype, bias=_spread_bias())
    x = _rand((24, DIM), seed=seed, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        straight = port.routed_forward(x, top_idx, gate)
        flipped = port.routed_forward(x, top_idx.flip(-1), gate.flip(-1))
    return int((straight != flipped).sum())


def test_control_slot_order_changes_the_combine_in_float32():
    """The slot order the sorted top-k fixes is load bearing, in float32.

    This is the control behind ``sorted=True``: the combine sums the eight
    gated rows in slot order, and float32 addition is not associative, so a
    permutation of the slots is a different number.
    """
    assert _slot_order_gap(torch.float32, seed=34) > 0


def test_slot_order_is_not_observable_in_bfloat16_eager():
    """And the measured limit of that control, which is worth pinning.

    In bf16 the reversal moves **nothing**: eager ``sum(dim=1)`` accumulates
    the eight terms in float32 and rounds once, and at these magnitudes that
    single rounding absorbs the reassociation. Measured here, and over a
    (256, 8, 3072) bf16 tensor under four different slot permutations, the
    count of changed channels is zero every time.

    Two consequences. A bf16-only check of the ``sorted=True`` convention
    passes vacuously, which is why the control above runs in float32. And the
    convention still has to be kept: the permutation also reorders the rows
    inside each expert's group, which on CPU cannot change a row's value but
    on an accelerator changes the tiles the grouped GEMM reduces in. This
    assertion is about eager CPU and claims nothing about a kernel.
    """
    assert _slot_order_gap(torch.bfloat16, seed=34) == 0


def test_control_dropping_the_routed_scale_changes_the_output():
    """The 2.5 is not optional and not absorbed anywhere else."""
    _, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    x = _rand((9, DIM), seed=35, dtype=torch.bfloat16)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        routed = port.routed_forward(x, top_idx, gate)
        shared = port.shared(x)
        assert torch.equal(port(x), (routed * SCALE + shared))
        assert not torch.equal(port(x), (routed + shared))


def test_control_folding_the_scale_into_the_gates_differs():
    """Scaling the gates instead of the sum is the same algebra and a
    different number.

    M*'s shared ``NoAuxTcRouter`` multiplies ``routed_scaling_factor`` into
    the gates in float32, before the experts. Praxis casts the unscaled gate
    to bf16 and scales the summed routed output. Eight roundings fed into a
    sum against one rounding applied to it: this control measures the gap.
    """
    _, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    x = _rand((24, DIM), seed=36, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = port_moe.router_logits(port.router.weight, x)
        scores = torch.sigmoid(logits)
        biased = scores + port.expert_bias
        _, top_idx = torch.topk(biased, TOP_K, dim=-1)
        picked = scores.gather(-1, top_idx)
        gate = picked / picked.sum(-1, keepdim=True)
        # The fold: scale in float32, then cast, then no scale on the sum.
        folded_gate = (gate * SCALE).to(x.dtype)
        folded = port.routed_forward(x, top_idx, folded_gate) + port.shared(x)
    assert not torch.equal(port(x), folded)


def test_control_shared_expert_is_ungated():
    """M*'s ``ParallelSparseMoeBlock`` requires a sigmoid gate on its shared
    expert. Praxis's shared expert is always on at full weight, and gating it
    is a different layer."""
    _, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    x = _rand((9, DIM), seed=37, dtype=torch.bfloat16)
    probe = torch.nn.Linear(DIM, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        routed = port.routed_forward(x, top_idx, gate) * SCALE
        gated = routed + torch.sigmoid(probe(x)) * port.shared(x)
    assert not torch.equal(port(x), gated)


def test_control_bf16_router_picks_other_experts():
    """Why the router GEMM is float32 even when everything else is bf16.

    A bf16 router does not perturb the output a little: it hands some tokens a
    different expert, which is a discontinuous change.
    """
    _, port = _moe_pair(torch.bfloat16, bias=_spread_bias())
    x = _rand((64, DIM), seed=38, dtype=torch.bfloat16)
    with torch.no_grad():
        narrow = F.linear(x, port.router.weight).to(torch.float32)
        narrow_idx = torch.topk(torch.sigmoid(narrow) + port.expert_bias, TOP_K, -1)[1]
        wide_idx, _ = port.route(x)
    assert not torch.equal(wide_idx, narrow_idx)


# ---------------------------------------------------------------------------
# The renormalization epsilon: measured, both ways
# ---------------------------------------------------------------------------


def test_renorm_epsilon_is_inert_at_realistic_scores():
    """The reference divides by the bare sum; DeepSeek, HF and M* add 1e-20.

    At any score a trained router produces, the eight picked sigmoids sum to
    something of order one, and 1e-20 is far below the float32 resolution
    there, so the two expressions are the *same* number. The difference is
    real but not reachable, and this is the half of it that matters in
    production. The practical consequence: adding the epsilon to our ``route``
    passes every parity test in this file, so nothing here protects the
    convention and only this test records why it does not have to.
    """
    _, port = _moe_pair(torch.float32, bias=_spread_bias())
    x = _rand((64, DIM), seed=40, dtype=torch.float32, scale=2.0)
    with torch.no_grad():
        scores = torch.sigmoid(port_moe.router_logits(port.router.weight, x))
        _, top_idx = torch.topk(scores + port.expert_bias, TOP_K, dim=-1)
        picked = scores.gather(-1, top_idx)
        total = picked.sum(-1, keepdim=True)
    assert total.min() > 1e-3
    assert torch.equal(picked / total, picked / (total + 1e-20))


def test_renorm_epsilon_bites_only_at_vanishing_scores():
    """And the half that does not: push the logits to about -50 and the eight
    sigmoids sum to ~1e-21, where the epsilon dominates its own denominator."""
    logits = torch.full((1, NUM_EXPERTS), -50.0)
    scores = torch.sigmoid(logits)
    _, top_idx = torch.topk(scores, TOP_K, dim=-1)
    picked = scores.gather(-1, top_idx)
    total = picked.sum(-1, keepdim=True)
    assert total.item() < 1e-20
    assert not torch.equal(picked / total, picked / (total + 1e-20))


# ---------------------------------------------------------------------------
# The chunked dispatch: the regime every published checkpoint selects
# ---------------------------------------------------------------------------

PUBLISHED_CHUNK = 16384
# Down to one row, through the published chunk exactly, and past it so the
# last one crosses a boundary (20000 is 16384 plus a ragged 3616).
CHUNKED_ROWS = [1, 40, 728, 4096, PUBLISHED_CHUNK, 20000]
# Chunks small enough that a cheap batch crosses many boundaries, the last
# chunk ragged. 1 is the degenerate one: every token its own dispatch.
SMALL_CHUNKS = [1, 7]


def _reference_with_chunk(chunk, dtype, bias, seed: int = 0, top_k: int = TOP_K):
    """A reference MoE with a dispatch chunk size, weights fixed by ``seed``."""
    ref = ref_layers.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, top_k, SCALE, chunk)
    _fill(ref, seed)
    with torch.no_grad():
        ref.expert_bias.copy_(bias)
    return ref.to(dtype).eval()


def _chunked_pair(
    dtype: torch.dtype = torch.float32,
    chunk: int | None = PUBLISHED_CHUNK,
    top_k: int = TOP_K,
    seed: int = 0,
):
    """A reference MoE and our port in the same dispatch regime, same weights."""
    ref = _reference_with_chunk(chunk, dtype, _spread_bias(), seed=seed, top_k=top_k)
    port = port_moe.MoEFeedForward(
        DIM, MOE_HIDDEN, NUM_EXPERTS, top_k, SCALE, dispatch_chunk_size=chunk
    )
    port.load_state_dict(ref.state_dict())
    return ref, port.to(dtype).eval()


def _chunk_pieces(port, x):
    """One chunk's expert outputs and dispatch metadata, as the port makes them.

    The controls below invert the combine on exactly these tensors, so the
    only thing that differs between the two sides of each one is the
    convention under test.
    """
    top_idx, gate = port.route(x)
    rows, sorted_gate, inverse, offsets = port.dispatch(x, top_idx, gate)
    gate_up = torch._grouped_mm(rows, port.routed.gate_up_weight, offsets)
    out = torch._grouped_mm(
        port_moe.expert_swiglu(gate_up), port.routed.down_weight, offsets
    )
    return out, sorted_gate, inverse


def _terms(out, sorted_gate, inverse, top_k=TOP_K):
    """The ``top_k`` scaled float32 terms of every token, slot by slot."""
    rows_of_token = inverse.view(-1, top_k)
    scale = sorted_gate.to(torch.float32) * SCALE
    return [
        out[rows_of_token[:, s]].float() * scale[rows_of_token[:, s]].unsqueeze(-1)
        for s in range(top_k)
    ]


def _one_accumulator(terms):
    total = torch.zeros_like(terms[0])
    for t in terms:
        total = total + t
    return total


def _pairwise_tree(terms):
    acc = [torch.zeros_like(terms[0]) for _ in range(4)]
    for s, t in enumerate(terms):
        acc[s % 4] = acc[s % 4] + t
    return (acc[0] + acc[1]) + (acc[2] + acc[3])


# --- parity, piece by piece -------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk_dispatch_metadata_matches_reference(dtype):
    """The reference cuts a chunk's metadata with a stable ``sort`` and a
    ``scatter_add`` histogram; the port reuses its ``argsort`` and
    ``searchsorted`` dispatch. That reuse is only sound if the two agree on
    every field, so this asserts it rather than arguing it."""
    _, port = _chunked_pair(dtype)
    x = _rand((13, DIM), seed=50, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        order, sorted_tokens, want_gate, want_offsets = ref_layers._dispatch_rows(
            top_idx, gate, NUM_EXPERTS
        )
        rows, got_gate, inverse, got_offsets = port.dispatch(x, top_idx, gate)
    want_inverse = torch.empty_like(order)
    want_inverse[order] = torch.arange(order.numel())
    assert torch.equal(rows, x[sorted_tokens].to(port.routed.gate_up_weight.dtype))
    assert torch.equal(got_gate, want_gate)
    assert torch.equal(inverse, want_inverse)
    assert got_offsets.dtype == want_offsets.dtype == torch.int32
    assert torch.equal(got_offsets, want_offsets)


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_combine_matches_reference(dtype):
    """The four-accumulator sum against both of the reference's eager
    spellings of it: the layer's own and the kernel module's fallback."""
    from praxis.kernels.layers_triton import moe_chunked_combine

    _, port = _chunked_pair(dtype)
    x = _rand((29, DIM), seed=51, dtype=dtype)
    with torch.no_grad():
        out, sorted_gate, inverse = _chunk_pieces(port, x)
        got = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
        layer_sum = ref_layers._sum_rows_of_tokens(
            out, inverse.view(-1, TOP_K), sorted_gate.to(torch.float32) * SCALE
        )
        kernel_sum = moe_chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
    assert got.dtype == torch.float32
    assert torch.equal(got, layer_sum)
    assert torch.equal(got, kernel_sum)


@pytest.mark.parametrize("dtype", DTYPES)
def test_routed_chunk_matches_reference(dtype):
    ref, port = _chunked_pair(dtype)
    x = _rand((31, DIM), seed=52, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        want = ref_layers._routed_chunk(
            x, top_idx, gate, ref.routed.gate_up_weight, ref.routed.down_weight, SCALE
        )
        got = port.routed_chunk(x, top_idx, gate)
    assert got.dtype == want.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunk", [*SMALL_CHUNKS, PUBLISHED_CHUNK])
def test_routed_forward_chunked_matches_reference(dtype, chunk):
    ref, port = _chunked_pair(dtype, chunk=chunk)
    x = _rand((40, DIM), seed=53, dtype=dtype)
    with torch.no_grad():
        top_idx, gate = port.route(x)
        want = ref._routed_forward_chunked(x, top_idx, gate)
        got = port.routed_forward_chunked(x, top_idx, gate)
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want)


# --- parity, end to end -----------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", CHUNKED_ROWS)
def test_chunked_forward_matches_reference(dtype, rows):
    """The published chunk, from one row to past one chunk."""
    ref, port = _chunked_pair(dtype)
    x = _rand((rows, DIM), seed=54, dtype=dtype)
    with torch.no_grad():
        want, got = ref(x), port(x)
    assert got.shape == x.shape
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunk", SMALL_CHUNKS)
@pytest.mark.parametrize("rows", [1, 40, 728])
def test_chunked_forward_matches_reference_across_many_chunks(dtype, chunk, rows):
    """Many boundaries and a ragged last chunk, cheaply.

    This is the test that pins *where* the chunks are cut. At the published
    chunk, moving the 16384 boundary by a row was measured to change nothing
    at these widths, so an off-by-one there passes the test above; at a chunk
    of 7 the GEMMs see different row counts per expert group and the same
    misreading shows up.
    """
    ref, port = _chunked_pair(dtype, chunk=chunk)
    x = _rand((rows, DIM), seed=55, dtype=dtype)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_chunked_forward_matches_reference_at_every_shape(dtype, shape):
    """Leading dims are flattened before the chunks are cut, on both sides."""
    ref, port = _chunked_pair(dtype, chunk=3)
    x = _rand(shape, seed=56, dtype=dtype)
    with torch.no_grad():
        want, got = ref(x), port(x)
    assert got.shape == want.shape == x.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("top_k", [1, 3, 4, 8, NUM_EXPERTS])
def test_chunked_forward_matches_reference_at_every_k(top_k):
    """Fewer slots than accumulators leaves some of the four at zero, and more
    than four wraps them; the reference's sum is defined for both and the
    port has to agree. Float32, because that is where the accumulators are
    visible (see the bf16 note below)."""
    ref, port = _chunked_pair(torch.float32, chunk=5, top_k=top_k)
    x = _rand((23, DIM), seed=57, dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))


def test_an_empty_batch_runs_in_both_regimes():
    """No chunks at all: the float32 buffer is empty and the cast is a no-op."""
    for chunk in (None, PUBLISHED_CHUNK):
        ref, port = _chunked_pair(torch.bfloat16, chunk=chunk)
        x = torch.empty(0, DIM, dtype=torch.bfloat16)
        with torch.no_grad():
            got, want = port(x), ref(x)
        assert got.shape == want.shape == (0, DIM)
        assert got.dtype == want.dtype == torch.bfloat16


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_chunk_is_refused_by_both_layers(bad):
    """The reference's own check. Zero would step the chunk loop by nothing
    and a negative step yields no chunks, leaving the output uninitialized."""
    for layer in (ref_layers.MoEFeedForward, port_moe.MoEFeedForward):
        with pytest.raises(ValueError, match="dispatch_chunk_size must be positive"):
            layer(DIM, MOE_HIDDEN, NUM_EXPERTS, TOP_K, SCALE, bad)


# --- negative controls: one inverted convention each ------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 40, 512])
def test_a_single_chunk_is_not_the_unchunked_path(dtype, rows):
    """The tempting reading of ``dispatch_chunk_size`` is that a served batch
    stays under it and the chunked path reduces to the unchunked one. It does
    not, in the reference.

    The reference picks the path on ``dispatch_chunk_size is not None`` alone,
    and the two paths differ in the *combine*: chunked multiplies the gate by
    ``routed_scaling_factor`` in float32, sums the ``top_k`` rows in float32
    through four interleaved accumulators and rounds once at the end, where the
    unchunked path gates in the row dtype, sums, and scales the sum. Every row
    count here is far below 16384, so no chunk boundary is ever crossed, and
    the answers still differ. One row is enough.
    """
    assert rows < PUBLISHED_CHUNK  # a single chunk, by construction
    bias = _spread_bias()
    unchunked = _reference_with_chunk(None, dtype, bias)
    chunked = _reference_with_chunk(PUBLISHED_CHUNK, dtype, bias)
    x = _rand((rows, DIM), seed=41, dtype=dtype)
    with torch.no_grad():
        want, got = unchunked(x), chunked(x)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert not torch.equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_port_selects_the_regime_the_reference_does(dtype):
    """And the same split, on our side: each regime of the port is bit-exact
    against the same regime of the reference and bit-different from the other.
    A port that ignored the field, or picked on a threshold, fails one half."""
    bias = _spread_bias()
    refs = {c: _reference_with_chunk(c, dtype, bias) for c in (None, PUBLISHED_CHUNK)}
    x = _rand((64, DIM), seed=42, dtype=dtype)
    with torch.no_grad():
        for chunk, other in ((None, PUBLISHED_CHUNK), (PUBLISHED_CHUNK, None)):
            port = port_moe.MoEFeedForward(
                DIM, MOE_HIDDEN, NUM_EXPERTS, TOP_K, SCALE, dispatch_chunk_size=chunk
            )
            port.load_state_dict(refs[None].state_dict())
            port = port.to(dtype).eval()
            assert torch.equal(port(x), refs[chunk](x))
            assert not torch.equal(port(x), refs[other](x))


def test_control_one_accumulator_differs():
    """Four interleaved accumulators are not one running sum.

    The same eight float32 terms, added left to right into a single
    accumulator, or as a pairwise tree of the four, are other numbers.
    Float32, because bf16 cannot see it (see the measured note below).
    """
    _, port = _chunked_pair(torch.float32)
    x = _rand((40, DIM), seed=60, dtype=torch.float32)
    with torch.no_grad():
        out, sorted_gate, inverse = _chunk_pieces(port, x)
        want = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
        terms = _terms(out, sorted_gate, inverse)
        single, tree = _one_accumulator(terms), _pairwise_tree(terms)
    assert not torch.equal(want, single)
    assert not torch.equal(want, tree)
    # Both pass a tolerance check, which is why parity here has to be bit-exact.
    torch.testing.assert_close(single, want)
    torch.testing.assert_close(tree, want)


def test_control_scaling_after_the_sum_differs():
    """The chunked regime scales the gate, not the sum.

    This is the unchunked regime's convention carried into the chunked one:
    the same float32 four-accumulator sum of the unscaled products, then the
    2.5. In exact arithmetic it is the same number.
    """
    _, port = _chunked_pair(torch.float32)
    x = _rand((40, DIM), seed=61, dtype=torch.float32)
    with torch.no_grad():
        out, sorted_gate, inverse = _chunk_pieces(port, x)
        want = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
        after = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, 1.0) * SCALE
    assert not torch.equal(want, after)
    torch.testing.assert_close(after, want)


def test_control_rounding_the_products_differs():
    """The products stay float32 until the one rounding.

    Rounding each scaled row to the row dtype before the float32 sum, which is
    where the unchunked regime rounds, moves bf16 outputs. This one is
    visible in bf16 where the two controls above are not.
    """
    _, port = _chunked_pair(torch.bfloat16)
    x = _rand((40, DIM), seed=62, dtype=torch.bfloat16)
    with torch.no_grad():
        out, sorted_gate, inverse = _chunk_pieces(port, x)
        want = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
        terms = [t.to(torch.bfloat16).float() for t in _terms(out, sorted_gate, inverse)]
        acc = [torch.zeros_like(want) for _ in range(4)]
        for s, t in enumerate(terms):
            acc[s % 4] = acc[s % 4] + t
        rounded = ((acc[0] + acc[1]) + acc[2]) + acc[3]
    assert not torch.equal(want.to(torch.bfloat16), rounded.to(torch.bfloat16))


def test_float32_sum_order_is_not_observable_in_bfloat16():
    """The measured limit of the two float32 controls above, worth pinning.

    In bf16 the one accumulator, the pairwise tree and the scale after the
    sum each change the float32 sum in many elements, and the single
    rounding to bf16 absorbs **every** one of those changes. Measured here at
    728 rows, and at 20000 rows (1.28M elements), the count of changed bf16
    outputs is zero for all three. This is elementwise IEEE arithmetic on
    fixed inputs, with no GEMM in it, so it is not a backend property.

    The consequence: every bf16 parity test in this file passes on any of
    those three misreadings. Only the float32 parity and the controls above
    protect them, the same limit the unchunked slot order has.
    """
    _, port = _chunked_pair(torch.bfloat16)
    x = _rand((728, DIM), seed=63, dtype=torch.bfloat16)
    with torch.no_grad():
        out, sorted_gate, inverse = _chunk_pieces(port, x)
        want = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, SCALE)
        terms = _terms(out, sorted_gate, inverse)
        after = port_moe.chunked_combine(out, sorted_gate, inverse, TOP_K, 1.0) * SCALE
        others = (_one_accumulator(terms), _pairwise_tree(terms), after)
    for other in others:
        # The float32 sums do differ, so the bf16 equality is not vacuous.
        assert not torch.equal(want, other)
        assert torch.equal(want.to(torch.bfloat16), other.to(torch.bfloat16))


@pytest.mark.parametrize(
    "rows, chunk",
    [
        # Many small chunks against one chunk over the same tokens.
        (40, 7),
        # The published chunk with a second, ragged chunk: 16384 + 3616.
        (20000, PUBLISHED_CHUNK),
    ],
)
def test_control_the_chunk_boundary_moves_the_result(rows, chunk):
    """Where the tokens are cut is part of the arithmetic, in float32.

    The combine is row-wise, so this is not the sum: it is the grouped GEMMs,
    which see one chunk's rows per call, and on CPU a row's result depends on
    how many rows its expert group holds. Measured here, one chunk against
    16384 + 3616 differs in a few dozen of 1.28M elements; 7-row chunks
    against one chunk differ in hundreds of 2560.

    Recorded alongside, not asserted, because it is a property of the GEMM
    backend rather than of the arithmetic: in bf16 the published boundary
    changed nothing at these widths, and moving it by one row (16383, 16385)
    changed nothing in float32 either.
    """
    assert rows > chunk  # more than one chunk, by construction
    _, port = _chunked_pair(torch.float32, chunk=chunk)
    x = _rand((rows, DIM), seed=41, dtype=torch.float32)
    with torch.no_grad():
        cut = port(x)
        port.dispatch_chunk_size = rows
        whole = port(x)
    assert not torch.equal(cut, whole)


# ---------------------------------------------------------------------------
# The published configs: the routed counts, and the dispatch regime they ask
# for. These read the reference's own checkpoint fixtures rather than a copy
# of them in this file, because a copy is exactly how the port came to believe
# dispatch_chunk_size was null.
# ---------------------------------------------------------------------------

_DATA = Path(praxis.__file__).resolve().parents[1] / "tests" / "data"
_FIXTURES = ("praxis-1-base", "praxis-1-droid", "praxis-1-droid-k", "praxis-1-yam")
# The servable ones. The base checkpoint has no action heads, so the config
# reader refuses it before it ever reaches the MoE fields.
_SERVABLE = ("praxis-1-droid", "praxis-1-droid-k", "praxis-1-yam")


def _published(name: str) -> dict:
    path = _DATA / f"{name}.json"
    if not path.exists():
        pytest.skip(f"the reference checkout has no {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", _FIXTURES)
def test_every_published_config_asks_for_the_chunked_dispatch(name):
    """The premise that the chunked regime is the served one, checked against
    the fixtures.

    The port once believed the published checkpoints set
    ``dispatch_chunk_size`` to null. ``null`` is the *reference's default*;
    every published config sets 16384. This test exists so that claim can
    never be made again from memory, and so that a checkpoint that really does
    leave it null shows up as a change here rather than as silence.
    """
    moe = _published(name)["backbone"]["moe"]
    assert moe["dispatch_chunk_size"] == PUBLISHED_CHUNK
    # And both pairs of expert counts, which is the other thing a hand-copied
    # fixture tends to lose.
    assert (moe["num_experts"], moe["top_k"]) == (129, 9)
    assert (moe["num_routed_experts"], moe["top_k_routed"]) == (128, 8)
    assert moe["shared_expert"] is True
    assert moe["routed_scaling_factor"] == 2.5


def test_the_base_config_is_refused_for_its_heads_not_its_moe():
    """The one published config that does not load, and why: it has no action
    heads. Asserted so that nobody reads its absence from ``_SERVABLE`` as a
    leftover of the old dispatch refusal."""
    with pytest.raises(Praxis1ConfigError, match="no action heads"):
        Praxis1Config.from_dict(_published("praxis-1-base"))


@pytest.mark.parametrize("name", _SERVABLE)
def test_a_published_config_loads_with_its_dispatch_chunk_size(name):
    """Every servable published config loads, and its chunk reaches the layer.

    Built on the meta device: the real widths are 128 banks of 3072 x 2560,
    and what is under test is which regime the arguments select, not the
    weights. Both layers are built from the same kwargs, so the reference
    also agrees the accessor's names are its own.
    """
    config = Praxis1Config.from_dict(_published(name))
    assert config.dispatch_chunk_size == PUBLISHED_CHUNK
    kwargs = config.moe_kwargs()
    assert kwargs["dispatch_chunk_size"] == PUBLISHED_CHUNK
    with torch.device("meta"):
        port = port_moe.MoEFeedForward(**kwargs)
        ref = ref_layers.MoEFeedForward(**kwargs)
    assert port.dispatch_chunk_size == ref.dispatch_chunk_size == PUBLISHED_CHUNK


@pytest.mark.parametrize("name", _SERVABLE)
def test_a_published_config_selects_the_chunked_combine(name):
    """And the regime it selects is the chunked one, measured.

    The published kwargs at this file's small widths: the published chunk,
    k and scale, with only the widths and the expert count shrunk. The port
    built that way is bit-exact against the reference built the same way and
    bit-different from the unchunked reference, so a config reader that lost
    the field would fail here rather than serve the other combine.
    """
    kwargs = Praxis1Config.from_dict(_published(name)).moe_kwargs()
    assert (kwargs["top_k"], kwargs["routed_scaling_factor"]) == (TOP_K, SCALE)
    small = {**kwargs, "dim": DIM, "hidden": MOE_HIDDEN, "num_experts": NUM_EXPERTS}
    ref = ref_layers.MoEFeedForward(**small)
    _fill(ref, seed=0)
    with torch.no_grad():
        ref.expert_bias.copy_(_spread_bias())
    port = port_moe.MoEFeedForward(**small)
    port.load_state_dict(ref.state_dict())
    unchunked = _reference_with_chunk(None, torch.bfloat16, _spread_bias())
    ref, port = ref.to(torch.bfloat16).eval(), port.to(torch.bfloat16).eval()
    x = _rand((40, DIM), seed=64, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(port(x), ref(x))
        assert not torch.equal(port(x), unchunked(x))


@pytest.mark.parametrize("name", _SERVABLE)
def test_a_published_config_gives_the_layer_the_routed_counts(name):
    """128 experts and top-8, read the way the reference reads them.

    ``praxis.backbone`` builds its own ``MoEFeedForward`` from
    ``moe["num_routed_experts"]`` and ``moe["top_k_routed"]``. This asserts our
    config arrives at the same two numbers from the same file, so the port
    cannot drift to the 129 and 9 that sit beside them under the names the
    layer's own arguments use.
    """
    config = _published(name)
    moe = config["backbone"]["moe"]
    kwargs = Praxis1Config.from_dict(config).moe_kwargs()

    assert kwargs["num_experts"] == moe["num_routed_experts"] == 128
    assert kwargs["top_k"] == moe["top_k_routed"] == 8
    assert kwargs["hidden"] == moe["expert_hidden"] == 1280
    assert kwargs["routed_scaling_factor"] == moe["routed_scaling_factor"] == 2.5
    assert kwargs["dim"] == config["backbone"]["dim"] == 3072
    assert kwargs["dispatch_chunk_size"] == moe["dispatch_chunk_size"]
    # The totals are the numbers that must not reach the layer.
    assert kwargs["num_experts"] != moe["num_experts"]
    assert kwargs["top_k"] != moe["top_k"]


def test_moe_kwargs_are_the_arguments_both_layers_take():
    """The accessor's names are the reference's constructor's names.

    A structural check rather than a numerical one: it is what catches the
    reference renaming ``top_k`` or ``dispatch_chunk_size``, which would
    otherwise show up as our kwargs silently landing in ``**kwargs`` or as a
    TypeError far from here. Bound against both constructors, since the
    port's signature tracks the reference's on purpose.
    """
    kwargs = Praxis1Config.for_rig("single_arm").moe_kwargs()
    for layer in (ref_layers.MoEFeedForward, port_moe.MoEFeedForward):
        bound = inspect.signature(layer.__init__).bind_partial(None, **kwargs)
        assert bound.arguments["num_experts"] == 128
        assert bound.arguments["top_k"] == 8
        assert bound.arguments["dispatch_chunk_size"] == PUBLISHED_CHUNK


def test_the_shared_inclusive_counts_build_a_layer_that_does_not_complain():
    """Why the config refuses to answer to ``num_experts`` at all.

    This is the failure the naming exists to prevent, shown rather than
    described: build the reference layer from the totals instead of the routed
    pair and nothing raises. The router is one expert wider, every token takes
    one expert more, the gates still sum to one and the output still has the
    right shape. The extra bank is one the checkpoint never filled, because the
    shared expert lives in its own tensors.
    """
    routed = ref_layers.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, TOP_K, SCALE)
    totals = ref_layers.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS + 1, TOP_K + 1, SCALE)
    for layer, experts, k in ((routed, NUM_EXPERTS, TOP_K), (totals, NUM_EXPERTS + 1, TOP_K + 1)):
        _fill(layer, seed=0)
        assert layer.router.weight.shape[0] == experts
        assert layer.routed.gate_up_weight.shape[0] == experts
        assert layer.top_k == k

    x = _rand((6, DIM), seed=11, dtype=torch.float32)
    with torch.no_grad():
        # Same shape out, and the gates of the wider one still renormalize.
        assert totals(x).shape == routed(x).shape
        _, gate = totals._route(x)
        assert gate.shape[1] == TOP_K + 1
        torch.testing.assert_close(gate.sum(-1), torch.ones(6))
    # The weights of the two cannot be exchanged, which is the only loud part
    # of this failure and it only fires if a loader is strict.
    with pytest.raises(RuntimeError):
        totals.load_state_dict(routed.state_dict())
