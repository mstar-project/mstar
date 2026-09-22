"""DiffusionSamplerResource: the contract an autoregressive sampler cannot hold.

Every position scored, log-probabilities returned alongside the tokens, and
per-request knobs honoured inside one batched call.
"""

import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.diffusion_sampler.config import (
    DiffusionSamplerSpec,
    DiffusionSamplingReqConfig,
)
from mstar.engine.resources.diffusion_sampler.resource import DiffusionSamplerResource

ROWS, VOCAB, MASK = 8, 64, 63


def _build(**spec_kwargs):
    spec = DiffusionSamplerSpec(
        resource_key="diffusion_sampler", nodes={"backbone"},
        vocab_size=VOCAB, num_rows=ROWS, forbidden_class=MASK, **spec_kwargs,
    )
    return DiffusionSamplerResource.build(
        spec, EngineResourceInfo(device=torch.device("cpu"))
    )


def _logits(positions, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(ROWS, positions, VOCAB, generator=g)


def test_returns_a_token_and_its_logprob_for_every_position():
    res = _build()
    res.ingest_request("a", DiffusionSamplingReqConfig())
    c = _logits(12)

    tokens, logprobs = res.sample(["a"], c, seq_lens=[12])

    assert tokens.shape == (ROWS, 12)
    assert logprobs.shape == (ROWS, 12)
    # The returned log-probability must be the winning class's, which is what
    # a confidence-ranked reveal sorts on.
    expect = torch.log_softmax(c, dim=-1)
    expect[..., MASK] = -float("inf")
    assert torch.allclose(logprobs, expect.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))
    assert (logprobs <= 0).all()


def test_the_forbidden_class_is_never_emitted():
    res = _build()
    res.ingest_request("a", DiffusionSamplingReqConfig())
    # Make the mask class the runaway favourite everywhere.
    c = _logits(6)
    c[..., MASK] = 50.0

    tokens, _ = res.sample(["a"], c, seq_lens=[6])
    assert (tokens != MASK).all()


def test_guidance_is_per_request_inside_one_batch():
    """Two requests, one guided and one not, scored in a single call."""
    res = _build()
    res.ingest_request("guided", DiffusionSamplingReqConfig(guidance_scale=2.0))
    res.ingest_request("plain", DiffusionSamplingReqConfig(guidance_scale=0.0))

    c, u = _logits(10, seed=1), _logits(10, seed=2)
    tokens, logprobs = res.sample(["guided", "plain"], c, u, seq_lens=[4, 6])
    assert tokens.shape == (ROWS, 10)

    # The unguided half must match what it would get scored on its own, and
    # the guided half must not.
    alone = _build()
    alone.ingest_request("plain", DiffusionSamplingReqConfig(guidance_scale=0.0))
    solo_t, solo_lp = alone.sample(["plain"], c[:, 4:], u[:, 4:], seq_lens=[6])
    assert torch.equal(tokens[:, 4:], solo_t)
    assert torch.allclose(logprobs[:, 4:], solo_lp)

    guided_only = _build()
    guided_only.ingest_request("guided", DiffusionSamplingReqConfig(guidance_scale=2.0))
    g_t, _ = guided_only.sample(["guided"], c[:, :4], u[:, :4], seq_lens=[4])
    assert torch.equal(tokens[:, :4], g_t)
    assert not torch.equal(tokens[:, :4], c[:, :4].argmax(-1)), (
        "guidance at 2.0 should move some tokens off the unguided argmax"
    )


def test_guidance_zero_is_not_double_normalised():
    """log_softmax twice is not log_softmax once, so the zero path must skip it."""
    res = _build()
    res.ingest_request("a", DiffusionSamplingReqConfig(guidance_scale=0.0))
    c, u = _logits(5, seed=3), _logits(5, seed=4)

    _, logprobs = res.sample(["a"], c, u, seq_lens=[5])
    plain = torch.log_softmax(c, dim=-1)
    plain[..., MASK] = -float("inf")
    tokens = plain.argmax(-1)
    assert torch.allclose(logprobs, plain.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))


def test_temperature_zero_is_greedy_and_deterministic():
    res = _build()
    res.ingest_request("a", DiffusionSamplingReqConfig(temperature=0.0))
    c = _logits(9, seed=5)

    first, _ = res.sample(["a"], c, seq_lens=[9])
    second, _ = res.sample(["a"], c, seq_lens=[9])
    assert torch.equal(first, second)

    expect = torch.log_softmax(c, dim=-1)
    expect[..., MASK] = -float("inf")
    assert torch.equal(first, expect.argmax(-1))


def test_a_seeded_request_replays():
    c = _logits(7, seed=6)

    def draw(seed, iteration=0):
        res = _build()
        cfg = DiffusionSamplingReqConfig(temperature=1.0)
        cfg.apply_conductor_config(seed=seed)
        res.ingest_request("a", cfg)
        return res.sample(["a"], c, seq_lens=[7], iterations=[iteration])[0]

    assert torch.equal(draw(1234), draw(1234))
    assert not torch.equal(draw(1234), draw(5678))
    # Different iterations of the same request must not repeat one draw.
    assert not torch.equal(draw(1234, iteration=0), draw(1234, iteration=1))


def test_mismatched_shapes_are_rejected_at_the_seam():
    res = _build()
    res.ingest_request("a", DiffusionSamplingReqConfig())
    c = _logits(5)

    for bad in (
        dict(request_ids=["a"], seq_lens=[4]),          # lengths do not cover
        dict(request_ids=["a", "b"], seq_lens=[5]),     # ids and lengths differ
    ):
        try:
            res.sample(c_logits=c, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")

    try:
        res.sample(["a"], c[0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 2-D logits tensor")


def test_a_seeded_draw_ignores_its_neighbours():
    """The whole point of per-request seeding: batching must not change a draw."""
    ca, cb = _logits(6, seed=11), _logits(9, seed=12)

    def cfg(seed):
        c = DiffusionSamplingReqConfig(temperature=1.0)
        c.apply_conductor_config(seed=seed)
        return c

    alone = _build()
    alone.ingest_request("a", cfg(99))
    solo, _ = alone.sample(["a"], ca, seq_lens=[6], iterations=[3])

    together = _build()
    together.ingest_request("a", cfg(99))
    together.ingest_request("b", cfg(1234))
    packed, _ = together.sample(
        ["a", "b"], torch.cat([ca, cb], dim=1),
        seq_lens=[6, 9], iterations=[3, 7],
    )
    assert torch.equal(packed[:, :6], solo), (
        "request a's draw changed because b was in the same step"
    )
