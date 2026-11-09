"""The fused RAS draws give ``ras_sample``'s tokens: the hash runs in the same uint32
arithmetic, so a row's token doesn't depend on which path drew it."""

import pytest
import torch

from mstar.model.cosyvoice3.components import sampling
from mstar.model.cosyvoice3.components.sampling import ras_sample

VOCAB, WINDOW = 6761, 10
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused draws are Triton kernels")


def _args(rows, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    def full(value, **kw):
        return torch.full((rows,), value, device=device, **kw)

    return dict(
        logits=torch.randn(rows, VOCAB, device=device, generator=gen) * 3,
        seed=torch.randint(0, 2**62, (rows,), device=device, generator=gen),
        step=torch.randint(0, 500, (rows,), device=device, generator=gen),
        history=torch.randint(0, 200, (rows, WINDOW), device=device, generator=gen),
        temperature=full(1.0),
        top_p=full(0.8),
        top_k=full(25, dtype=torch.long),
        window=full(WINDOW, dtype=torch.long),
        repeat_threshold=full(1.0),
        blocked=torch.zeros(rows, VOCAB, dtype=torch.bool, device=device),
    )


def _both(args):
    return ras_sample(**args), sampling.ras_sample_plain(**args)


@cuda
@pytest.mark.parametrize("rows", [1, 3, 16, 32])
@pytest.mark.parametrize("repeats", [False, True], ids=["no_redraw", "redraw"])
def test_fused_draws_pick_the_same_tokens(rows, repeats):
    device = torch.device("cuda")
    for seed in range(12):
        args = _args(rows, seed, device)
        if repeats:  # the nucleus lands in the history, so draws repeat and redraw
            args["logits"][:, :200] += 8.0
        fused, plain = _both(args)
        assert torch.equal(fused, plain), seed


@cuda
def test_greedy_rows_and_blocked_ids_hold_on_the_fused_path():
    device = torch.device("cuda")
    args = _args(8, 0, device)
    args["temperature"][:4] = 0.0
    args["blocked"][:, 6561:] = True
    fused, plain = _both(args)
    assert torch.equal(fused, plain)
    assert (fused < 6561).all()


@cuda
def test_the_fused_draws_replay_in_a_graph():
    device = torch.device("cuda")
    args = _args(4, 1, device)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        out = ras_sample(**args)
    for seed in (2, 3):
        fresh = _args(4, seed, device)
        for name, value in fresh.items():
            args[name].copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ras_sample(**args))
