"""The Triton LoRA kernels at every tile configuration ``triton_backend._tiles`` picks: decode and
prefill widths, a rank held in one tile and ranks tiled with their split partials reduced first,
ranks off the tile, against the fp32 reference ops over mixed slots with ragged per-slot rows, a
rank-0 slot and base rows; and a tiled rank captured at a prefill width. GPU only.

    pytest test/integration/test_lora_triton_high_rank.py -v
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("triton")

import mstar.engine.resources.lora.triton_backend as tb  # noqa: E402  (registers the CUDA kernels)
from mstar.engine.resources.lora import backend as ref_ops  # noqa: E402
from mstar.engine.resources.lora import ops  # noqa: E402

SLOTS = 6


def check(got, want, what):  # bf16 inputs and weights, plus a bf16 rounding of the output
    err = (got.float() - want).norm() / want.norm().clamp_min(1e-6)
    assert err < 3e-2, f"{what}: relative error {err:.3g}"


def ragged_layout(tokens, g):
    """Tokens over the slots in uneven runs (one slot takes most), as ``(token slots, order, indptr)``."""
    weights = torch.tensor([1.0, 6.0, 2.0, 1.0, 0.5, 3.0], device="cuda")
    token_slots = torch.multinomial(weights, tokens, replacement=True, generator=g)
    order = torch.argsort(token_slots, stable=True).to(torch.int32)
    indptr = torch.zeros(SLOTS + 1, dtype=torch.int32, device="cuda")
    indptr[1:] = torch.cumsum(torch.bincount(token_slots, minlength=SLOTS), 0)
    return token_slots, order, indptr


def case(max_rank, in_features, slices, tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # slot 0 is the zero adapter and slot 3 has rank 0; the others are at, under and off the tile
    ranks = torch.tensor([0, max_rank, max(1, max_rank // 4), 0, max(1, max_rank // 2 + 3), max_rank - 1 or 1],
                         dtype=torch.int32, device="cuda")
    a = torch.randn(SLOTS, len(slices) * max_rank, in_features, generator=g, device="cuda") / in_features ** 0.5
    b = torch.randn(SLOTS, sum(slices), max_rank, generator=g, device="cuda") / max_rank ** 0.5
    scalings = torch.rand(SLOTS, generator=g, device="cuda") + 0.5
    token_slots, order, indptr = ragged_layout(tokens, g)
    x = torch.randn(tokens, in_features, generator=g, device="cuda")
    y0 = torch.randn(tokens, sum(slices), generator=g, device="cuda")
    return a, b, scalings, ranks, token_slots, order, indptr, x, y0


SHAPES = [  # max_rank, in_features, slices: Phi-4-multimodal's targets, then widths off the tiles
    (256, 3072, (3072, 1024, 1024)), (320, 3072, (8192, 8192)), (320, 8192, (3072,)), (256, 3072, (3072,)),
    (100, 1000, (700, 300)), (16, 3072, (3072, 1024, 1024)), (8, 8192, (3072,)), (64, 1000, (700,)),
]


@pytest.mark.parametrize("tokens", [1, 8, 33, 64, 128, 300, 2048])
@pytest.mark.parametrize("max_rank,in_features,slices", SHAPES)
def test_apply_matches_the_reference(max_rank, in_features, slices, tokens):
    """``lora_apply_`` (whose shrink writes zeros past each slot's rank for its expand) against
    the fp32 reference; base rows and the rank-0 slot's rows bitwise untouched."""
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, tokens)
    want = y0.clone()
    h_ref = ref_ops.shrink_reference(x, a, order, indptr, ranks, len(slices))
    ref_ops.expand_reference_(want, h_ref, b, order, indptr, ranks, scalings, list(slices))
    got = y0.to(torch.bfloat16)
    ops.lora_apply_(got, x.to(torch.bfloat16), a.to(torch.bfloat16), b.to(torch.bfloat16), order, indptr,
                    ranks, scalings, list(slices))
    check(got.float() - y0.to(torch.bfloat16).float(), want - y0, f"apply r{max_rank} T{tokens}")
    untouched = (token_slots == 0) | (token_slots == 3)
    assert torch.equal(got[untouched], y0.to(torch.bfloat16)[untouched])


@pytest.mark.parametrize("tokens", [5, 300])
@pytest.mark.parametrize("max_rank,in_features,slices", [(320, 3072, (3072, 1024, 1024)), (100, 1000, (700, 300))])
def test_shrink_then_expand_with_anything_past_the_rank(max_rank, in_features, slices, tokens):
    """The separate ops: ``lora_expand_`` masks ``h`` past each slot's rank itself, since its ``h``
    may come from anywhere (here, inf where the expand must not read)."""
    a, b, scalings, ranks, token_slots, order, indptr, x, y0 = case(max_rank, in_features, slices, tokens, 7)
    want = y0.clone()
    h_ref = ref_ops.shrink_reference(x, a, order, indptr, ranks, len(slices))
    ref_ops.expand_reference_(want, h_ref, b, order, indptr, ranks, scalings, list(slices))
    h = ops.lora_shrink(x.to(torch.bfloat16), a.to(torch.bfloat16), order, indptr, ranks, len(slices))
    rank = ranks.long()[token_slots]
    col = torch.arange(h.shape[1], device="cuda") % max_rank
    h = torch.where(col[None, :] < rank[:, None], h, float("inf"))
    got = y0.to(torch.bfloat16)
    ops.lora_expand_(got, h, b.to(torch.bfloat16), order, indptr, ranks, scalings, list(slices))
    check(got.float() - y0.to(torch.bfloat16).float(), want - y0, f"shrink + expand r{max_rank} T{tokens}")


@pytest.mark.parametrize("tokens", [8, 300])
def test_shrink_zeros_past_each_slots_rank(tokens):
    """What ``lora_apply_``'s expand relies on: every split's partial holds zeros from a live
    row's slot rank up to the max rank, whatever the buffer held before."""
    max_rank, slices = 320, (3072, 1024, 1024)
    a, _, _, ranks, token_slots, order, indptr, x, _ = case(max_rank, 3072, slices, tokens, 3)
    tiles = tb._tiles(tokens, max_rank)
    splits, k_per_split = 3, 1024
    h = torch.full((splits, tokens, len(slices) * max_rank), float("nan"), device="cuda")
    tb._shrink_into(h, x.to(torch.bfloat16), a.to(torch.bfloat16), order, indptr, ranks, len(slices), splits,
                    k_per_split, tiles)
    want = ref_ops.shrink_reference(x, a, order, indptr, ranks, len(slices))
    live = ranks.long()[token_slots] > 0
    check(h.sum(0)[live], want[live], "split partials")
    col = torch.arange(h.shape[2], device="cuda") % max_rank
    past = col[None, :] >= ranks.long()[token_slots][:, None]
    assert not h[:, live[:, None] & past].any(), "a partial past a slot's rank is not zero"


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@pytest.mark.parametrize("tokens", [24, 300])
def test_a_high_rank_inside_a_captured_graph(tokens):
    """``lora_apply_`` at max rank 320, decode and prefill tiles, captured once on the zero adapter
    and replayed for slot mixes and ranks that change between replays: bitwise equal to eager."""
    g = torch.Generator(device="cuda").manual_seed(11)
    max_rank, in_features, slices = 320, 3072, [3072, 1024, 1024]
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    a = (torch.randn(SLOTS, len(slices) * max_rank, in_features, generator=g, device="cuda") / 55).to(torch.bfloat16)
    b = (torch.randn(SLOTS, sum(slices), max_rank, generator=g, device="cuda") / 18).to(torch.bfloat16)
    scalings = torch.rand(SLOTS, generator=g, device="cuda") + 0.5
    ranks = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    _, order, indptr = ragged_layout(tokens, g)
    indptr.zero_()
    indptr[1:] = tokens  # every row on the zero adapter
    x, y = torch.zeros(tokens, in_features, **bf16), torch.zeros(tokens, sum(slices), **bf16)

    def run():
        out = y.clone()
        ops.lora_apply_(out, x, a, b, order, indptr, ranks, scalings, slices)
        return out

    graph, out = capture(run)
    for mix_ranks, seed in (([0, 320, 33, 0, 256, 100], 1), ([0, 16, 320, 64, 0, 319], 2), ([0] * SLOTS, 3)):
        gm = torch.Generator(device="cuda").manual_seed(seed)
        ranks.copy_(torch.tensor(mix_ranks, dtype=torch.int32))
        _, o, p = ragged_layout(tokens, gm)
        order.copy_(o)
        indptr.copy_(p)
        x.copy_(torch.randn(x.shape, generator=gm, device="cuda").to(torch.bfloat16))
        y.copy_(torch.randn(y.shape, generator=gm, device="cuda").to(torch.bfloat16))
        graph.replay()
        eager = run()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"replay differs from eager for ranks {mix_ranks}"
        if not any(mix_ranks):
            assert torch.equal(out, y)
