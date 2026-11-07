"""Declared prompt tokens seed only admitted real rows, before their first read."""
import pytest
import torch

from mstar.engine.accelerator_graph_runner import dummy_rid_handle
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext


def test_padded_prompt_seeds_only_the_real_rows():
    vocab = 65
    res = SamplerResource(vocab, True, torch.device("cpu"))
    real = (0, 1)
    pad = tuple(dummy_rid_handle(f"__cg_test_pad_{i}__") for i in range(2))
    prompts = {0: torch.tensor([0, 31, 32, 64, 31]), 1: torch.tensor([11, 13])}
    for rid in real:
        res.ingest_request(rid, SamplingReqConfig(repetition_penalty=1.3))
    for rid in pad:
        res.ingest_capture_row(rid)
    tracked = {**prompts, **{rid: torch.tensor([vocab + 3]) for rid in pad}}
    ctx = StepContext(request_ids=real, graph_walk="prefill", slot=0, capture=False)
    ctx.set_padded_rids((*real, *pad))
    res.plan(SamplerStep(prefill_tracked_tokens=tracked), ctx)
    masks = res._rows.read_seen(res._rows.indices(real))
    for i, rid in enumerate(real):
        assert masks[i].nonzero().flatten().tolist() == sorted(set(prompts[rid].tolist()))
    assert all(res._rows.slot_of(rid) == 0 for rid in pad)
    assert not res._rows.seen_words[0].any()


@pytest.mark.parametrize("vocab", [None, 0, -1, True])
def test_presence_capability_requires_a_positive_declared_width(vocab):
    with pytest.raises(ValueError, match="positive vocab_size"):
        SamplerResource(vocab, True, torch.device("cpu"))


def test_presence_rejects_undeclared_penalties_and_mismatched_logits():
    plain = SamplerResource(None, False, torch.device("cpu"))
    with pytest.raises(ValueError, match="disables"):
        plain.ingest_request(1, SamplingReqConfig(repetition_penalty=1.3))
    res = SamplerResource(33, True, torch.device("cpu"))
    res.ingest_request(1)
    with pytest.raises(ValueError, match="logits width"):
        res.sample([1], torch.zeros(1, 34))
