"""``mstar::sample_filtered`` on AMD GPUs against its torch reference and the full sampler.

Greedy rows take the first argmax, exactly. Sampled rows draw the reference's Philox noise, so they take the
reference's token unless two scores sit within the rounding of the GPU's ``log``. The distribution must be the full
sampler's: the renormalised top-k / top-p of the softmax. In a captured graph, each replay draws afresh.
"""

import pytest
import torch
from scipy.stats import chisquare

from mstar.engine.resources.sampler import filtered as F
from mstar.engine.resources.sampler import unfiltered as U
from mstar.engine.resources.sampler.utils import sample_accelerator_graphable_gpu
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _case(batch, vocab, dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = (torch.randn(batch, vocab, device=DEV, generator=g) * 3).to(dtype)
    temperature = torch.where(torch.arange(batch, device=DEV) % 3 == 0, 0.0, 0.9).float()
    seeds = torch.arange(batch, device=DEV, dtype=torch.int64) * 1009 + 5
    offset = torch.arange(batch, device=DEV, dtype=torch.int64) * 3 + (1 << 33)
    return logits, temperature, seeds, offset


def _run(logits, temperature, top_k, top_p, seeds, offset):
    rows = logits.shape[0]
    tokens = torch.empty(rows, dtype=torch.int64, device=DEV)
    offset = offset.clone()
    F.SAMPLE_FILTERED(
        logits, temperature, torch.as_tensor(top_k, device=DEV, dtype=torch.int32).expand(rows).contiguous(),
        torch.as_tensor(top_p, device=DEV, dtype=torch.float32).expand(rows).contiguous(), seeds, offset, tokens, 1,
    )
    return tokens, offset


@pytest.mark.parametrize("batch, vocab", [(1, 2048), (3, 3072), (17, 2048), (64, 2048), (128, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k, top_p", [(50, 1.0), (0, 1.0), (50, 0.8), (0, 0.5)])
def test_tokens_match_the_reference(batch, vocab, dtype, top_k, top_p):
    logits, temperature, seeds, offset = _case(batch, vocab, dtype)
    tokens, advanced = _run(logits, temperature, top_k, top_p, seeds, offset)
    assert advanced.tolist() == (offset + 1).tolist()
    for r in range(batch):
        x = logits[r].float().cpu()
        t = int(tokens[r])
        if float(temperature[r]) == 0:
            assert t == int(torch.argmax(x)), f"greedy row {r}"
            continue
        scaled = x * (1.0 / temperature[r:r + 1].float().cpu())
        kept = F.kept_tokens(scaled, top_k, top_p)
        scores = torch.where(kept, scaled + U.gumbel_reference(int(seeds[r]), int(offset[r]), vocab), -float("inf"))
        if not bool(kept[t]):
            # the kept set's edge: top-p's float32 sum may round a token across it
            assert top_p < 1.0, f"row {r} drew a token top-k cuts"
        elif t != int(torch.argmax(scores)):
            best = float(scores.max())
            assert best - float(scores[t]) < 1e-3 * max(1.0, abs(best)), f"row {r}"


def _tv(tokens, expected):
    counts = torch.bincount(tokens.cpu(), minlength=expected.numel()).double() / tokens.numel()
    return 0.5 * float((counts - expected).abs().sum())


@pytest.mark.parametrize("top_k, top_p", [(50, 1.0), (10, 1.0), (0, 0.8), (30, 0.9)])
def test_the_draw_follows_the_full_samplers_distribution(top_k, top_p):
    """The same logits through the fused draw and the full captured sampler, rows enough for TV noise near 0.01."""
    vocab, rows, t = 2048, 1 << 14, 0.9
    g = torch.Generator(device=DEV).manual_seed(1)
    row = (torch.randn(vocab, device=DEV, generator=g) * 2.5).to(torch.bfloat16)
    logits = row.expand(rows, vocab).contiguous()
    temperature = torch.full((rows,), t, device=DEV)
    seeds = torch.arange(rows, device=DEV, dtype=torch.int64) * 31 + 7
    offset = torch.zeros(rows, dtype=torch.int64, device=DEV)
    fused, _ = _run(logits, temperature, top_k, top_p, seeds, offset)
    # per-row draws of the full sampler, whose noise is folded per row off row 0's seed
    full = sample_accelerator_graphable_gpu(
        logits, temperature, torch.full((rows,), top_k, device=DEV, dtype=torch.int32),
        torch.full((rows,), top_p, device=DEV, dtype=torch.float32), seeds[:1].expand(rows).contiguous(), offset,
    )
    scaled = row.float().cpu() * (1.0 / torch.tensor([t]))
    kept = F.kept_tokens(scaled, top_k, top_p)
    expected = torch.where(kept, torch.softmax(row.float().cpu().double() / t, 0), torch.zeros(vocab).double())
    expected /= expected.sum()
    fused_tv, full_tv = _tv(fused, expected), _tv(full, expected)
    assert fused_tv < 0.12, (fused_tv, full_tv)
    assert fused_tv < full_tv * 1.3 + 0.01, (fused_tv, full_tv)
    assert not (torch.bincount(fused.cpu(), minlength=vocab)[~kept] > 0).any()


def test_a_small_vocabulary_passes_a_chi_square():
    vocab, rows, t = 12, 1 << 16, 0.9
    row = torch.linspace(-2, 2, vocab, device=DEV)
    tokens, _ = _run(row.expand(rows, vocab).contiguous(), torch.full((rows,), t, device=DEV), 5, 1.0,
                     torch.arange(rows, device=DEV, dtype=torch.int64) * 31 + 7,
                     torch.zeros(rows, dtype=torch.int64, device=DEV))
    counts = torch.bincount(tokens, minlength=vocab).cpu().double()
    assert counts[:7].sum() == 0
    expected = torch.softmax(row.double().cpu()[7:] / t, 0) * rows
    assert chisquare(counts[7:].numpy(), expected.numpy()).pvalue > 1e-3


def test_exact_ties_take_the_first_index_on_greedy_rows():
    logits = torch.zeros(3, 2048, device=DEV, dtype=torch.bfloat16)
    logits[0, [5, 1000, 2047]] = 2.0
    logits[2, 77] = -1.0
    tokens, _ = _run(logits, torch.zeros(3, device=DEV), 1, 1.0, torch.arange(3, device=DEV),
                     torch.zeros(3, dtype=torch.int64, device=DEV))
    assert tokens.tolist() == [5, 0, 0]


def test_rows_of_nan_still_draw_an_id_inside_the_vocab():
    vocab = 2048
    logits = torch.randn(4, vocab, device=DEV).to(torch.bfloat16)
    logits[0] = float("nan")
    logits[1, : vocab // 2] = float("nan")
    tokens, _ = _run(logits, torch.tensor([0.0, 0.7, 0.7, 0.7], device=DEV), 50, 1.0, torch.arange(4, device=DEV),
                     torch.zeros(4, dtype=torch.int64, device=DEV))
    assert ((tokens >= 0) & (tokens < vocab)).all(), tokens.tolist()
    assert int(tokens[1]) >= vocab // 2


def test_a_captured_replay_draws_what_eager_does_and_advances():
    batch, vocab, steps = 8, 2048, 3
    logits, _, seeds, _ = _case(batch, vocab, torch.bfloat16)
    temperature = torch.full((batch,), 0.9, device=DEV)
    top_k = torch.full((batch,), 50, device=DEV, dtype=torch.int32)
    top_p = torch.ones(batch, device=DEV)
    offset = torch.zeros(batch, device=DEV, dtype=torch.int64)
    tokens = torch.empty(batch, device=DEV, dtype=torch.int64)

    def step():
        F.SAMPLE_FILTERED(logits, temperature, top_k, top_p, seeds, offset, tokens, 1)

    expected = []
    for _ in range(steps):
        step()
        expected.append(tokens.clone())
    torch.cuda.synchronize()
    assert offset.tolist() == [steps] * batch
    offset.zero_()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        step()  # warm on the capture stream
    offset.zero_()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        step()
    offset.zero_()
    for s in range(steps):
        graph.replay()
        torch.cuda.synchronize()
        assert tokens.tolist() == expected[s].tolist(), f"replay {s}"
    assert offset.tolist() == [steps] * batch
