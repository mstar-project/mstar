"""mstar's sampling entry points on AMD GPUs (FlashInfer's ROCm fork underneath).

The draw must follow FlashInfer's truncation (top-k first, then top-p on the
renormalized top-k mass), return the first argmax for greedy rows even on exact
ties, repeat for a repeated (seed, offset), and replay identically from a
captured graph whose seed/offset buffers change between replays.
"""

import pytest
import torch
from scipy.stats import chisquare

from mstar.engine.resources.sampler.utils import sample_accelerator_graphable_gpu, sample_tokens
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _expected(probs: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """FlashInfer's top-k-first truncation of one row, in fp64."""
    q = probs.double().clone()
    if 0 < k < q.numel():
        q[q < torch.topk(q, k).values[-1]] = 0
    q /= q.sum()
    order = torch.argsort(q, descending=True)
    before = torch.cumsum(q[order], 0) - q[order]
    keep = torch.zeros_like(q, dtype=torch.bool)
    keep[order] = before < p
    q = torch.where(keep, q, torch.zeros_like(q))
    return q / q.sum()


def _row(vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(vocab, generator=g, dtype=torch.float64), 0)


@pytest.mark.parametrize("path", ["graphable", "eager"])
def test_distribution_matches_truncation(path):
    vocab, reps, calls = 48, 2048, 48
    configs = [(0, 1.0), (5, 1.0), (0, 0.7), (10, 0.8), (3, 0.5), (vocab, 0.95), (1, 1.0)]
    rows, ks, ps, expected = [], [], [], []
    for i, (k, p) in enumerate(configs):
        probs = _row(vocab, i)
        exp = _expected(probs, k, p)
        # a boundary token within fp32 rounding of p would make the support ambiguous
        sorted_q = torch.sort(_expected(probs, k, 1.0), descending=True).values
        before = (torch.cumsum(sorted_q, 0) - sorted_q)[sorted_q > 0]
        assert p == 1.0 or ((before - p).abs() > 1e-4).all()
        rows.append(probs.log().float().expand(reps, vocab))
        ks += [k] * reps
        ps += [p] * reps
        expected.append(exp)
    logits = torch.cat(rows).to(DEV)
    n = logits.shape[0]
    temperature = torch.ones(n, device=DEV)
    top_k = torch.tensor(ks, device=DEV, dtype=torch.int32)
    top_p = torch.tensor(ps, device=DEV)
    seed = torch.arange(n, device=DEV, dtype=torch.long) * 7919 + 1
    counts = torch.zeros(len(configs), vocab, dtype=torch.long)
    for call in range(calls):
        offset = torch.full((n,), call, device=DEV, dtype=torch.long)
        if path == "graphable":
            tokens = sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, offset)
        else:
            tokens = sample_tokens(
                logits, temperature, top_k, top_p, any_greedy=False,
                top_k_zero_count=int((top_k == 0).sum()), seed=seed, rand_offset=offset,
            )
        tokens = tokens.view(len(configs), reps).cpu()
        for i in range(len(configs)):
            counts[i] += torch.bincount(tokens[i], minlength=vocab)
    for i, (k, p) in enumerate(configs):
        exp = expected[i] * reps * calls
        outside = counts[i][exp == 0].sum().item()
        assert outside == 0, f"(k={k}, p={p}): {outside} draws outside the truncated support"
        support = exp > 0
        if support.sum() == 1:
            continue
        result = chisquare(counts[i][support].double().numpy(), exp[support].numpy())
        assert result.pvalue > 1e-4, f"(k={k}, p={p}): chi-square p={result.pvalue:.2e}"


@pytest.mark.parametrize("vocab", [64, 40000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_rows_take_the_first_argmax_on_ties(vocab, dtype):
    """Large vocabularies take the split softmax path, with ties across chunks."""
    batch = 6
    logits = torch.randn(batch, vocab, device=DEV, dtype=dtype)
    tie_at = [(3, 10), (0, vocab - 1), (17, 30)] if vocab == 64 else [(100, 20000), (16383, 16384), (5, 39999)]
    for row in range(batch):
        a, b = tie_at[row % len(tie_at)]
        logits[row, a] = logits[row, b] = 50.0
    expected = logits.float().argmax(-1)
    assert expected.tolist() == [tie_at[r % len(tie_at)][0] for r in range(batch)]
    # greedy rows mixed with sampled ones
    temperature = torch.tensor([0, 0.9, 0, 0.9, 0, 0.9], device=DEV, dtype=torch.float32)
    greedy = temperature == 0
    top_k = torch.full((batch,), 0, device=DEV, dtype=torch.int32)
    top_p = torch.ones(batch, device=DEV)
    for trial in range(8):
        seed = torch.randint(0, 2**31, (batch,), device=DEV)
        offset = torch.full((batch,), trial, device=DEV, dtype=torch.long)
        for tokens in (
            sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, offset),
            sample_tokens(logits, temperature, top_k, top_p, any_greedy=True,
                          top_k_zero_count=batch, seed=seed, rand_offset=offset),
        ):
            assert torch.equal(tokens[greedy].long(), expected[greedy])


def test_draws_repeat_per_seed_and_offset():
    batch, vocab = 64, 1000
    logits = torch.randn(batch, vocab, device=DEV)
    temperature = torch.ones(batch, device=DEV)
    top_k = torch.full((batch,), 50, device=DEV, dtype=torch.int32)
    top_p = torch.full((batch,), 0.95, device=DEV)
    seed = torch.arange(batch, device=DEV, dtype=torch.long) + 11

    def draw(offset, seed=seed):
        off = torch.full((batch,), offset, device=DEV, dtype=torch.long)
        return sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, off)

    assert torch.equal(draw(3), draw(3))
    assert (draw(3) != draw(4)).float().mean() > 0.5
    assert (draw(3) != draw(3, seed=seed + 1000)).float().mean() > 0.5


def test_captured_sampler_replays_the_eager_draw():
    batch, vocab = 8, 4096
    logits = torch.randn(batch, vocab, device=DEV, dtype=torch.bfloat16)
    temperature = torch.linspace(0, 1.2, batch, device=DEV)
    top_k = torch.tensor([0, 1, 5, 40, 0, 100, 4096, 7], device=DEV, dtype=torch.int32)
    top_p = torch.tensor([1.0, 1.0, 0.9, 0.8, 0.5, 1.0, 0.99, 0.7], device=DEV)
    seed = torch.arange(batch, device=DEV, dtype=torch.long) * 1000 + 1
    offset = torch.zeros(batch, device=DEV, dtype=torch.long)
    for _ in range(3):  # autotune and first-call work stay out of the capture
        sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, offset)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, offset)
    seen = set()
    for step in range(16):
        offset.fill_(step)
        logits.normal_()
        graph.replay()
        eager = sample_accelerator_graphable_gpu(logits, temperature, top_k, top_p, seed, offset)
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), f"step {step}"
        seen.add(tuple(captured.tolist()))
    assert len(seen) > 1, "replays drew fresh numbers"


def test_sampling_traced_by_dynamo_graph_breaks_instead_of_raising():
    """The fork's ops raise when dynamo traces into them (unless
    FLASHINFER_USE_TORCH_CUSTOM_OPS=1 was set before import); mstar's binding
    runs them outside the trace, so a compiled caller still samples."""
    import flashinfer

    batch, vocab = 4, 256
    logits = torch.randn(batch, vocab, device=DEV)
    temperature = torch.ones(batch, device=DEV)
    top_k = torch.full((batch,), 20, device=DEV, dtype=torch.int32)
    top_p = torch.full((batch,), 0.9, device=DEV)
    seed = torch.arange(batch, device=DEV, dtype=torch.long)
    offset = torch.zeros(batch, device=DEV, dtype=torch.long)

    def raw(probs):
        return flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, deterministic=True, seed=seed, offset=offset,
        )

    torch._dynamo.reset()
    with pytest.raises(Exception, match="FLASHINFER_USE_TORCH_CUSTOM_OPS"):
        torch.compile(raw)(torch.softmax(logits, -1))

    def step(logits):
        return sample_accelerator_graphable_gpu(logits * 1.0, temperature, top_k, top_p, seed, offset)

    torch._dynamo.reset()
    assert torch.equal(torch.compile(step)(logits), step(logits))
