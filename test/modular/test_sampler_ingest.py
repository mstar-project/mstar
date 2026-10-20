"""A worker serving two partitions of a request ingests it twice.

The conductor sends one NewRequest per partition, and the second can arrive after
the first partition already sampled. Re-adding the sampler state there clears the
seen-token mask the repetition penalty reads and restarts the RNG offset.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext

RID = "r0"
VOCAB = 32
PROMPT = torch.tensor([3, 5, 7])
SAMPLED = 11
CONFIG = SamplingReqConfig(temperature=0.9, top_k=50, repetition_penalty=1.05)


@pytest.fixture(autouse=True)
def _cpu_sampling(monkeypatch):
    # the sampling kernel needs a device, and these tests don't care which token it picks
    monkeypatch.setattr(sampler_utils, "sample_tokens", lambda **kwargs: torch.tensor([SAMPLED]))


def _sampler() -> SamplerResource:
    return SamplerResource(
        vocab_size=VOCAB, enable_repetion_penalty=True, device=torch.device("cpu"),
    )


def _prefill(resource: SamplerResource) -> None:
    ctx = StepContext(request_ids=(RID,), graph_walk="prefill", slot=0, capture=False)
    step = SamplerStep(prefill_tracked_tokens={RID: PROMPT})
    resource.plan(step, ctx)
    resource.sample(torch.zeros(1, VOCAB))
    resource.commit(step, ctx)


def _seen(resource: SamplerResource) -> list[int]:
    mask = resource._sampler.get_token_mask(RID)._seen_token_mask
    return mask.nonzero().flatten().tolist()


def test_a_repeat_ingest_after_a_step_keeps_the_seen_tokens_and_offset():
    resource = _sampler()
    resource.ingest_request(RID, CONFIG)
    _prefill(resource)
    offset = resource._sampler._step_offset[RID]
    assert _seen(resource) == [3, 5, 7, SAMPLED]

    # the second partition's NewRequest
    resource.ingest_request(RID, CONFIG)

    assert _seen(resource) == [3, 5, 7, SAMPLED]
    assert resource._sampler._step_offset[RID] == offset


def test_an_ingest_after_removal_starts_clean():
    resource = _sampler()
    resource.ingest_request(RID, CONFIG)
    _prefill(resource)
    resource.remove_request(RID)

    # a recycled handle
    resource.ingest_request(RID, CONFIG)

    assert _seen(resource) == []
    assert resource._sampler._step_offset[RID] == 0
