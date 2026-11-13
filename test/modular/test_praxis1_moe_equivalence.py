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
    """The premise this port was built on, checked against the fixtures.

    The module docstring of the port used to say the published checkpoints set
    ``dispatch_chunk_size`` to null and that the chunked combine was therefore
    unreachable. ``null`` is the *reference's default*; every published config
    sets 16384. This test exists so that claim can never be made again from
    memory, and so that a checkpoint that really does leave it null shows up as
    a change here rather than as silence.
    """
    moe = _published(name)["backbone"]["moe"]
    assert moe["dispatch_chunk_size"] == 16384
    # And both pairs of expert counts, which is the other thing a hand-copied
    # fixture tends to lose.
    assert (moe["num_experts"], moe["top_k"]) == (129, 9)
    assert (moe["num_routed_experts"], moe["top_k_routed"]) == (128, 8)
    assert moe["shared_expert"] is True
    assert moe["routed_scaling_factor"] == 2.5


@pytest.mark.parametrize("name", _SERVABLE)
def test_a_published_config_is_refused_for_its_dispatch_chunk_size(name):
    """Loudly, and naming the field, rather than falling through to this path.

    Every published checkpoint fails here today. That is the honest state of
    the port: it is bit-exact against the unchunked regime and no published
    checkpoint selects it.
    """
    with pytest.raises(Praxis1ConfigError, match="dispatch_chunk_size"):
        Praxis1Config.from_dict(_published(name))


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
    # Only the unported regime is in the way, so drop it to reach the counts.
    del config["backbone"]["moe"]["dispatch_chunk_size"]
    kwargs = Praxis1Config.from_dict(config).moe_kwargs()

    assert kwargs["num_experts"] == moe["num_routed_experts"] == 128
    assert kwargs["top_k"] == moe["top_k_routed"] == 8
    assert kwargs["hidden"] == moe["expert_hidden"] == 1280
    assert kwargs["routed_scaling_factor"] == moe["routed_scaling_factor"] == 2.5
    assert kwargs["dim"] == config["backbone"]["dim"] == 3072
    # The totals are the numbers that must not reach the layer.
    assert kwargs["num_experts"] != moe["num_experts"]
    assert kwargs["top_k"] != moe["top_k"]


def test_moe_kwargs_are_the_arguments_both_layers_take():
    """The accessor's names are the reference's constructor's names.

    A structural check rather than a numerical one: it is what catches the
    reference renaming ``top_k``, which would otherwise show up as our kwargs
    silently landing in ``**kwargs`` or as a TypeError far from here. Bound
    against both constructors, since the port's signature tracks the
    reference's on purpose.
    """
    kwargs = Praxis1Config.for_rig("single_arm").moe_kwargs()
    for layer in (ref_layers.MoEFeedForward, port_moe.MoEFeedForward):
        bound = inspect.signature(layer.__init__).bind_partial(None, **kwargs)
        assert bound.arguments["num_experts"] == 128
        assert bound.arguments["top_k"] == 8


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


# ---------------------------------------------------------------------------
# The chunked dispatch: measured, not reasoned about
# ---------------------------------------------------------------------------

PUBLISHED_CHUNK = 16384


def _reference_with_chunk(chunk, dtype, bias, seed: int = 0):
    """A reference MoE with a dispatch chunk size, weights fixed by ``seed``."""
    ref = ref_layers.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, TOP_K, SCALE, chunk)
    _fill(ref, seed)
    with torch.no_grad():
        ref.expert_bias.copy_(bias)
    return ref.to(dtype).eval()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 40, 512])
def test_a_single_chunk_is_not_the_unchunked_path(dtype, rows):
    """The tempting reading of ``dispatch_chunk_size`` is that a served batch
    stays under it and the chunked path reduces to this one. It does not.

    The reference picks the path on ``dispatch_chunk_size is not None`` alone,
    and the two paths differ in the *combine*: chunked multiplies the gate by
    ``routed_scaling_factor`` in float32, sums the ``top_k`` rows in float32
    through four interleaved accumulators and rounds once at the end, where the
    unchunked path gates in the row dtype, sums, and scales the sum. Every row
    count here is far below 16384, so no chunk boundary is ever crossed, and
    the answers still differ. One row is enough.

    This is the test that says the refusal in the config reader is a missing
    feature and not a guard against an exotic setting.
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
def test_our_port_is_the_unchunked_regime_exactly(dtype):
    """The other half of the finding above, and the reason it is a gap rather
    than a bug: the port is bit-exact against the unchunked reference and
    bit-different from the chunked one. So there is nothing wrong with what it
    computes, only with which of the two regimes it computes."""
    bias = _spread_bias()
    unchunked = _reference_with_chunk(None, dtype, bias)
    chunked = _reference_with_chunk(PUBLISHED_CHUNK, dtype, bias)
    port = port_moe.MoEFeedForward(DIM, MOE_HIDDEN, NUM_EXPERTS, TOP_K, SCALE)
    port.load_state_dict(unchunked.state_dict())
    port = port.to(dtype).eval()

    x = _rand((64, DIM), seed=42, dtype=dtype)
    with torch.no_grad():
        assert torch.equal(port(x), unchunked(x))
        assert not torch.equal(port(x), chunked(x))
