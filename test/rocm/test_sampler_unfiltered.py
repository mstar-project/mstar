"""``mstar::sample_unfiltered`` on AMD GPUs against its torch reference.

Greedy rows must be the first argmax, exactly. Sampled rows draw the reference's
Philox noise, so they take the reference's token unless two scores sit within
the rounding of the GPU's ``log``; the distribution must follow the softmax. In a
captured graph, each replay must draw afresh off the offsets it advances.
"""

import pytest
import torch
from scipy.stats import chisquare

from mstar.engine.resources.sampler import unfiltered as U
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _case(batch, vocab, dtype, greedy_every=2, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = (torch.randn(batch, vocab, device=DEV, generator=g) * 3).to(dtype)
    temperature = torch.where(torch.arange(batch, device=DEV) % greedy_every == 0, 0.0, 0.7).float()
    seeds = torch.arange(batch, device=DEV, dtype=torch.int64) * 1009 + 5
    offset = torch.arange(batch, device=DEV, dtype=torch.int64) * 3 + (1 << 33)
    return logits, temperature, seeds, offset


def _run(logits, temperature, seeds, offset):
    tokens = torch.empty(logits.shape[0], dtype=torch.int64, device=DEV)
    offset = offset.clone()
    U.SAMPLE_UNFILTERED(logits, temperature, seeds, offset, tokens, 1)
    return tokens, offset


def _scores(logits, temperature, seeds, offset):
    """The reference's score per token of each row (the noise included)."""
    rows = []
    for r in range(logits.shape[0]):
        x = logits[r].float().cpu()
        t = float(temperature[r])
        if t > 0:
            x = x * (1.0 / temperature[r:r + 1].float().cpu()) + U.gumbel_reference(
                int(seeds[r]), int(offset[r]), x.shape[0])
        rows.append(x)
    return torch.stack(rows)


@pytest.mark.parametrize("batch, vocab", [(1, 248320), (3, 151936), (17, 50000), (64, 248320), (128, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tokens_match_the_reference(batch, vocab, dtype):
    logits, temperature, seeds, offset = _case(batch, vocab, dtype)
    tokens, advanced = _run(logits, temperature, seeds, offset)
    assert advanced.tolist() == (offset + 1).tolist()
    scores = _scores(logits, temperature, seeds, offset)
    best = scores.max(dim=1).values
    for r in range(batch):
        t = int(tokens[r])
        if float(temperature[r]) == 0:
            assert t == int(torch.argmax(logits[r].float())), f"greedy row {r}"
        elif t != int(torch.argmax(scores[r])):
            # only a near-tie in the noisy scores may resolve differently
            assert float(best[r] - scores[r, t]) < 1e-4 * max(1.0, abs(float(best[r]))), f"row {r}"


def test_exact_ties_take_the_first_index():
    logits = torch.zeros(4, 248320, device=DEV, dtype=torch.bfloat16)
    logits[0, [5, 100000, 248319]] = 2.0
    logits[1, [248318, 248319]] = 1.0
    logits[3, 77] = -1.0  # row 3: all zero but one, the first zero wins
    tokens, _ = _run(logits, torch.zeros(4, device=DEV), torch.arange(4, device=DEV),
                     torch.zeros(4, dtype=torch.int64, device=DEV))
    assert tokens.tolist() == [5, 248318, 0, 0]


def test_the_draw_follows_the_softmax():
    vocab, rows = 12, 1 << 16
    row = torch.linspace(-2, 2, vocab, device=DEV)
    logits = row.expand(rows, vocab).contiguous()
    t = 0.9
    tokens, _ = _run(logits, torch.full((rows,), t, device=DEV), torch.arange(rows, device=DEV) * 31 + 7,
                     torch.zeros(rows, dtype=torch.int64, device=DEV))
    counts = torch.bincount(tokens, minlength=vocab).cpu().double()
    expected = torch.softmax(row.double().cpu() / t, 0) * rows
    assert chisquare(counts.numpy(), expected.numpy()).pvalue > 1e-3


def test_slot_masters_through_a_captured_graph():
    batch, vocab, steps = 4, 248320, 3
    logits, temperature, seeds, _ = _case(batch, vocab, torch.bfloat16)
    slots = torch.tensor([3, 1, 4, 4], device=DEV)  # the last two are padding rows on the trash slot
    offset_master = torch.tensor([0, 10, 0, 20, 7], device=DEV, dtype=torch.int64)
    token_master = torch.zeros(5, device=DEV, dtype=torch.int64)
    offset = torch.zeros(batch, device=DEV, dtype=torch.int64)
    tokens = torch.empty(batch, device=DEV, dtype=torch.int64)

    def step():
        U.SAMPLE_UNFILTERED(logits, temperature, seeds, offset, tokens, 1, slots, offset_master, token_master)

    # eager first: what each step should draw from the masters it starts from
    expected = []
    start = offset_master.clone()
    for _ in range(steps):
        step()
        expected.append(tokens[:2].clone())
    torch.cuda.synchronize()
    assert offset_master.tolist()[:4] == [0, 10 + steps, 0, 20 + steps]
    assert token_master[slots[:2]].tolist() == tokens[:2].tolist()

    offset_master.copy_(start)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        step()  # warm on the capture stream
    offset_master.copy_(start)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        step()
    offset_master.copy_(start)
    for s in range(steps):
        graph.replay()
        torch.cuda.synchronize()
        assert tokens[:2].tolist() == expected[s].tolist(), f"replay {s}"
    assert offset_master.tolist()[:4] == [0, 10 + steps, 0, 20 + steps]


def test_rows_of_nan_still_draw_an_id_inside_the_vocab():
    """Garbage logits (a broken layer, a capture's dummy rows) must not turn into an out-of-range id, which the
    next step's embedding would read out of bounds."""
    vocab = 248320
    logits = torch.randn(4, vocab, device=DEV).to(torch.bfloat16)
    logits[0] = float("nan")
    logits[1, : vocab // 2] = float("nan")
    logits[2, -5:] = float("nan")
    tokens, _ = _run(logits, torch.tensor([0.0, 0.7, 0.0, 0.7], device=DEV), torch.arange(4, device=DEV),
                     torch.zeros(4, dtype=torch.int64, device=DEV))
    assert ((tokens >= 0) & (tokens < vocab)).all(), tokens.tolist()
    assert int(tokens[1]) >= vocab // 2
    assert int(tokens[2]) == int(torch.nan_to_num(logits[2].float(), nan=-float("inf")).argmax())
