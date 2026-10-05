"""XPU parity, batching and graph replay for the released per-row RNG API."""

import pytest
import torch

from mstar.engine.resources.sampler.utils import Sampler, sample_tokens

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("vllm_xpu_kernels._xpu_C")
    return torch.device("xpu:0")


def _inputs(device, batch=8, vocab=257):
    generator = torch.Generator(device="cpu").manual_seed(42)
    logits = torch.randn(batch, vocab, generator=generator).to(device)
    seeds = torch.arange(batch, dtype=torch.int64, device=device) + 100
    offsets = torch.arange(batch, dtype=torch.int64, device=device) * vocab
    return logits, seeds, offsets


@pytest.mark.parametrize(("top_k", "top_p"), [(0, 1.), (32, 1.), (0, 0.8), (32, 0.8)])
def test_batched_sampling_matches_legacy_cpu_rng_calls(device, top_k, top_p):
    logits, seeds, offsets = _inputs(device)
    actual = sample_tokens(
        logits, temperature=1., top_k=top_k, top_p=top_p,
        seed=seeds, rand_offset=offsets, any_greedy=False,
        top_k_zero_count=logits.shape[0] if top_k == 0 else 0,
    )
    legacy = []
    for row in range(logits.shape[0]):
        token = torch.empty(1, dtype=torch.int64, device=device)
        k = torch.tensor([top_k], dtype=torch.int64, device=device) if top_k else None
        p = torch.tensor([top_p], dtype=torch.float32, device=device) if top_p < 1 else None
        torch.ops._xpu_C.topk_topp_sampler(
            token, None, logits[row:row + 1].clone(), k, p, "raw_logits",
            torch.tensor([int(seeds[row]), int(offsets[row])], dtype=torch.int64), 1.,
        )
        legacy.append(token[0])
    torch.testing.assert_close(actual, torch.stack(legacy), atol=0, rtol=0)


def test_mixed_settings_are_invariant_to_batch_order_and_padding(device):
    logits, seeds, offsets = _inputs(device)
    batch, vocab = logits.shape
    params = dict(
        temperature=torch.tensor([0., 0.7, 1.2, 0., 0.9, 1., 0.6, 0.], device=device),
        top_k=torch.tensor([0, 32, 0, 16, 32, 0, 16, 0], device=device),
        top_p=torch.tensor([1., 0.8, 0.9, 1., 0.8, 1., 0.95, 1.], device=device),
        min_p=torch.tensor([0., 0.1, 0.05, 0., 0.02, 0., 0.05, 0.], device=device),
        repetition_penalty=torch.tensor([1., 1.2, 1.1, 1., 1.3, 1., 1.2, 1.], device=device),
        seen_token_mask=(torch.arange(vocab, device=device)[None, :] % 5 == 0).expand(batch, -1),
        seed=seeds, rand_offset=offsets,
    )
    actual = sample_tokens(logits, **params)
    individual = torch.cat([
        sample_tokens(logits[row:row + 1], **{key: value[row:row + 1] for key, value in params.items()})
        for row in range(batch)
    ])
    torch.testing.assert_close(actual, individual, atol=0, rtol=0)
    permutation = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4], device=device)
    permuted = sample_tokens(logits[permutation], **{key: value[permutation] for key, value in params.items()})
    torch.testing.assert_close(permuted, actual[permutation], atol=0, rtol=0)
    doubled = sample_tokens(
        torch.cat([logits, logits]),
        **{key: torch.cat([value, value]) for key, value in params.items()},
    )
    torch.testing.assert_close(doubled, actual.repeat(2), atol=0, rtol=0)
    greedy = params["temperature"] == 0
    torch.testing.assert_close(actual[greedy], logits[greedy].argmax(dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize("top_ks", [(32, 32, 32), (0, 32, 0)])
def test_request_rng_offsets_survive_reordered_decode_batches(device, top_ks):
    logits, _, _ = _inputs(device, batch=3)
    serial, batched = Sampler(device), Sampler(device)
    for sampler in (serial, batched):
        for rid in range(3):
            sampler.add_request(rid)
            sampler.set_config(rid, temperature=0.8, top_k=top_ks[rid], top_p=0.9)
            sampler._sampling_config[rid].set_seed(777 + rid)
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        expected = torch.cat([serial.sample([rid], logits[rid:rid + 1]) for rid in order])
        actual = batched.sample(order, logits[order])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert serial._step_offset == batched._step_offset == {rid: 3 * logits.shape[1] for rid in range(3)}


def test_unseeded_sampling_advances_default_generator_and_reproduces_after_reset(device):
    logits, _, _ = _inputs(device, batch=8)
    generator = torch.xpu.default_generators[device.index]
    original = generator.get_state()
    try:
        generator.manual_seed(777)
        before = generator.get_state()
        first = sample_tokens(logits, temperature=1., top_k=32)
        second = sample_tokens(logits, temperature=1., top_k=32)
        assert not torch.equal(before, generator.get_state())
        assert not torch.equal(first, second)
        generator.manual_seed(777)
        torch.testing.assert_close(sample_tokens(logits, temperature=1., top_k=32), first, atol=0, rtol=0)
    finally:
        generator.set_state(original)


@pytest.mark.parametrize(
    ("top_k", "zero_count"),
    [(0, 4), (32, 0), ([0, 32, 0, 16], 2), ([0, 32, 0, 16], None)],
)
def test_explicit_rng_sampling_captures_and_advances_in_xpu_graph(device, top_k, zero_count):
    logits, seeds, offsets = _inputs(device, batch=4)
    stride = logits.shape[1]
    start = offsets.clone()
    if isinstance(top_k, list):
        top_k = torch.tensor(top_k, dtype=torch.int32, device=device)

    def sample():
        tokens = sample_tokens(
            logits, temperature=0.8, top_k=top_k, top_p=0.9,
            top_k_zero_count=zero_count,
            seed=seeds, rand_offset=offsets,
        )
        offsets.add_(stride)
        return tokens

    with torch.xpu.device(device):
        expected = [sample().clone() for _ in range(3)]
        torch.xpu.synchronize(device)
        offsets.copy_(start)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            tokens = sample()
        offsets.copy_(start)
        for reference in expected:
            graph.replay()
            torch.xpu.synchronize(device)
            torch.testing.assert_close(tokens, reference, atol=0, rtol=0)
        torch.testing.assert_close(offsets, start + 3 * stride, atol=0, rtol=0)
