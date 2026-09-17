"""Min-p sampling: the filter's semantics, and that the knob reaches the eager
sampler's per-request config, the graph buffers, and is refused on nodes whose
``SamplerSpec`` does not enable it.

The filter itself is checked against the HF ``MinPLogitsWarper`` definition
(``probs < min_p * max_prob`` is dropped) re-derived here on logits, so the
probability-space implementation cannot drift from the reference order:
penalty -> temperature -> min-p -> top-k/top-p.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import (
    Sampler,
    SamplerBuffers,
    SamplingConfig,
    apply_min_p,
)

CPU = torch.device("cpu")


def _hf_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """transformers.MinPLogitsWarper on logits, then softmax."""
    probs = logits.softmax(dim=-1)
    threshold = min_p[:, None] * probs.amax(dim=-1, keepdim=True)
    return logits.masked_fill(probs < threshold, float("-inf")).softmax(dim=-1)


def test_apply_min_p_matches_the_hf_warper():
    torch.manual_seed(0)
    logits = torch.randn(4, 64) * 3
    min_p = torch.tensor([0.0, 0.05, 0.3, 1.0])

    ours = apply_min_p(logits.softmax(dim=-1), min_p)

    torch.testing.assert_close(ours, _hf_min_p(logits, min_p), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ours.sum(dim=-1), torch.ones(4))


def test_min_p_zero_is_the_identity_and_one_keeps_only_the_argmax():
    torch.manual_seed(1)
    probs = torch.randn(3, 16).softmax(dim=-1)

    torch.testing.assert_close(apply_min_p(probs, torch.zeros(3)), probs)
    one_hot = apply_min_p(probs, torch.ones(3))
    assert torch.equal(one_hot.argmax(dim=-1), probs.argmax(dim=-1))
    torch.testing.assert_close(one_hot.amax(dim=-1), torch.ones(3))


def test_greedy_one_hot_rows_pass_through():
    """The prep kernel turns temperature-0 rows into a one-hot; min-p must not
    disturb them (the threshold is the single kept probability itself)."""
    probs = torch.zeros(2, 8)
    probs[0, 5] = 1.0
    probs[1, 2] = 1.0

    torch.testing.assert_close(apply_min_p(probs, torch.tensor([0.1, 0.9])), probs)


def test_request_config_reaches_the_eager_sampler():
    sampler = Sampler(device=CPU)
    sampler.add_request("r")
    sampler.set_config("r", **asdict(SamplingReqConfig(min_p=0.05, temperature=0.8)))

    assert sampler._sampling_config["r"].min_p == 0.05
    assert sampler._sampling_config["r"].temperature == 0.8


def test_graph_buffers_carry_min_p_only_when_enabled():
    on = SamplerBuffers.allocate(max_batch_size=4, device=CPU, enable_min_p=True)
    off = SamplerBuffers.allocate(max_batch_size=4, device=CPU)

    assert off.slice_for_bs(2)["min_p_buf"] is None
    assert on.slice_for_bs(2)["min_p_buf"].shape == (2,)

    on._write_master_row(1, SamplingConfig(min_p=0.1, temperature=0.7))
    off._write_master_row(1, SamplingConfig(min_p=0.1, temperature=0.7))  # no buffer, no error
    assert on.min_p.master[1].item() == pytest.approx(0.1)
    # greedy rows sample a one-hot; the filter is written inert for them
    on._write_master_row(2, SamplingConfig(min_p=0.1, temperature=0.0))
    assert on.min_p.master[2].item() == 0.0


def test_resource_refuses_min_p_without_the_capability():
    plain = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=CPU)
    with pytest.raises(ValueError, match="enable_min_p=False"):
        plain.ingest_request("r", SamplingReqConfig(min_p=0.05))
    assert "r" not in plain._sampler._sampling_config

    plain.ingest_request("ok", SamplingReqConfig(min_p=0.0))  # default stays fine
    assert plain._sampler._sampling_config["ok"].min_p == 0.0

    capable = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=CPU, enable_min_p=True,
    )
    capable.ingest_request("r", SamplingReqConfig(min_p=0.05))
    assert capable._sampler._sampling_config["r"].min_p == 0.05


def test_spec_capability_defaults_off():
    spec = SamplerSpec(resource_key="sampler", nodes={"lm"}, vocab_size=None)
    assert spec.enable_min_p is False
    # and build() forwards it, so a spec that opts in gets a capable resource
    info = SimpleNamespace(device=CPU, joint_comm_group=None)
    assert SamplerResource.build(spec, info)._enable_min_p is False
    on = SamplerSpec(resource_key="sampler", nodes={"lm"}, vocab_size=None, enable_min_p=True)
    assert SamplerResource.build(on, info)._enable_min_p is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer sampler requires CUDA")
def test_min_p_one_is_greedy_for_every_seed_on_cuda():
    pytest.importorskip("flashinfer")
    from mstar.engine.resources.sampler.utils import sample_cuda_graphable_gpu, sample_tokens

    dev = torch.device("cuda")
    torch.manual_seed(0)
    logits = torch.randn(4, 128, device=dev)
    expected = logits.argmax(dim=-1)
    for seed in range(8):
        seeds = torch.full((4,), seed, device=dev, dtype=torch.long)
        offsets = torch.zeros(4, device=dev, dtype=torch.long)
        eager = sample_tokens(
            logits, temperature=1.0, min_p=1.0, seed=seeds, rand_offset=offsets,
            any_greedy=False, any_top_k_zero=True, all_top_k_zero=True,
        )
        graph = sample_cuda_graphable_gpu(
            logits, torch.ones(4, device=dev), torch.zeros(4, device=dev, dtype=torch.int32),
            torch.ones(4, device=dev), seeds, offsets, min_p=torch.ones(4, device=dev),
        )
        assert torch.equal(eager.to(expected.dtype), expected)
        assert torch.equal(graph, expected)
