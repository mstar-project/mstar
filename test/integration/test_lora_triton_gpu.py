"""The Triton LoRA kernels in bf16 on a GPU (skipped elsewhere): against fp32 references over
slots of mixed ranks, a rank-0 slot and base rows, and inside one captured CUDA graph replayed
for changing slot mixes and ranks.

    pytest test/integration/test_lora_triton_gpu.py -v
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("triton")

import mstar.engine.resources.lora.triton_backend  # noqa: E402, F401  (registers the CUDA kernels)
from mstar.engine.resources.lora import backend as ref_ops  # noqa: E402
from mstar.engine.resources.lora import ops  # noqa: E402

BF16 = dict(device="cuda", dtype=torch.bfloat16)


def check(got, want, what):  # bf16 inputs and weights, plus a bf16 rounding of the output
    err = (got.float() - want).norm() / want.norm().clamp_min(1e-6)
    assert err < 3e-2, f"{what}: relative error {err:.3g}"


def capture(fn):
    """Warm ``fn`` on a side stream (Triton JIT and autotuning stay out of the graph), then capture it."""
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


def slot_layout(token_slots, slots):
    order = torch.argsort(token_slots, stable=True).to(torch.int32)
    indptr = torch.zeros(slots + 1, dtype=torch.int32, device=token_slots.device)
    indptr[1:] = torch.cumsum(torch.bincount(token_slots, minlength=slots), 0)
    return order, indptr


@pytest.mark.parametrize("op,max_rank,in_features,slices,tokens", [
    ("apply", 16, 1024, (4096, 512, 512), 1), ("apply", 32, 3584, (1024,), 5),
    ("apply", 64, 1000, (700,), 33), ("apply", 96, 1024, (4096, 512, 512), 200),
    ("apply", 128, 3584, (1024,), 1500), ("apply", 128, 1000, (700,), 5),
    ("apply", 256, 1000, (700,), 1), ("apply", 256, 1024, (4096, 512, 512), 1500),
    ("apply", 512, 3584, (1024,), 33), ("apply", 512, 1024, (4096, 512, 512), 200),
    ("gather", 128, 5000, (1024,), 1), ("gather", 128, 5000, (1024,), 7), ("gather", 256, 5000, (1024,), 300),
])
def test_kernels_match_the_reference(op, max_rank, in_features, slices, tokens):
    """``lora_apply_`` (split contraction, compact row tiles, an input width off the tile) and the
    embedding path ``lora_gather_shrink`` + ``lora_expand_`` (ids off the vocabulary get nothing)
    against the fp32 reference ops, over slots of mixed ranks with a rank-0 slot and base rows."""
    gather = op == "gather"
    g = torch.Generator(device="cuda").manual_seed(tokens + 100 * gather)
    slots, out_features = 6, sum(slices)
    ranks = torch.tensor([0, max_rank, max_rank // 4, 0, max_rank // 2 + 3, max_rank], dtype=torch.int32, device="cuda")
    a = torch.randn(slots, len(slices) * max_rank, in_features, generator=g, device="cuda") / (
        1 if gather else in_features ** 0.5)
    b = torch.randn(slots, out_features, max_rank, generator=g, device="cuda") / max_rank ** 0.5
    scalings = torch.rand(slots, generator=g, device="cuda") + 0.5
    token_slots = torch.randint(0, slots, (tokens,), generator=g, device="cuda")
    order, indptr = slot_layout(token_slots, slots)
    y0 = torch.randn(tokens, out_features, generator=g, device="cuda")
    want, got = y0.clone(), y0.to(torch.bfloat16)
    untouched = (token_slots == 0) | (token_slots == 3)  # the zero adapter and the rank-0 slot
    if gather:
        ids = torch.randint(0, in_features, (tokens,), generator=g, device="cuda")
        ids[::5], ids[3::7] = -1, in_features + 11
        untouched |= (ids < 0) | (ids >= in_features)
        h_ref = ref_ops.gather_shrink_reference(ids, a, order, indptr, ranks)
        h = ops.lora_gather_shrink(ids, a.to(torch.bfloat16), order, indptr, ranks)
        ops.lora_expand_(got, h, b.to(torch.bfloat16), order, indptr, ranks, scalings, list(slices))
    else:
        x = torch.randn(tokens, in_features, generator=g, device="cuda")
        h_ref = ref_ops.shrink_reference(x, a, order, indptr, ranks, len(slices))
        ops.lora_apply_(got, x.to(torch.bfloat16), a.to(torch.bfloat16), b.to(torch.bfloat16), order, indptr,
                        ranks, scalings, list(slices))
    ref_ops.expand_reference_(want, h_ref, b, order, indptr, ranks, scalings, list(slices))
    check(got.float() - y0.to(torch.bfloat16).float(), want - y0, op)
    assert torch.equal(got[untouched], y0.to(torch.bfloat16)[untouched])


def test_a_tiled_rank_inside_a_captured_graph():
    """``lora_apply_`` at max_rank 128 captured once (every row on the zero adapter), replayed for
    slot mixes and ranks that change between replays: bitwise equal to eager."""
    g = torch.Generator(device="cuda").manual_seed(5)
    slots, max_rank, in_features, slices, tokens = 4, 128, 1024, [1024, 256, 256], 24
    a = (torch.randn(slots, len(slices) * max_rank, in_features, generator=g, device="cuda") / 32).to(torch.bfloat16)
    b = (torch.randn(slots, sum(slices), max_rank, generator=g, device="cuda") / 4).to(torch.bfloat16)
    scalings = torch.rand(slots, generator=g, device="cuda") + 0.5
    ranks = torch.zeros(slots, dtype=torch.int32, device="cuda")
    order, indptr = slot_layout(torch.zeros(tokens, dtype=torch.long, device="cuda"), slots)
    x, y = torch.zeros(tokens, in_features, **BF16), torch.zeros(tokens, sum(slices), **BF16)

    def run():
        out = y.clone()
        ops.lora_apply_(out, x, a, b, order, indptr, ranks, scalings, slices)
        return out

    graph, out = capture(run)
    for mix_ranks, seed in (([0, 128, 33, 64], 1), ([0, 16, 128, 100], 2), ([0, 0, 0, 0], 3)):
        gm = torch.Generator(device="cuda").manual_seed(seed)
        ranks.copy_(torch.tensor(mix_ranks, dtype=torch.int32))
        order[:], indptr[:] = slot_layout(torch.randint(0, slots, (tokens,), generator=gm, device="cuda"), slots)
        x.copy_(torch.randn(x.shape, generator=gm, device="cuda").to(torch.bfloat16))
        y.copy_(torch.randn(y.shape, generator=gm, device="cuda").to(torch.bfloat16))
        graph.replay()
        eager = run()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"replay differs from eager for ranks {mix_ranks}"
        if not any(mix_ranks):
            assert torch.equal(out, y)
