"""A row whose sampled token is dropped must leave no sampler state behind.

Prefill walks that only keep the last prefill's token (BAGEL, the Qwen3-Omni
Thinker, Higgs Audio) still sample every row under a captured graph. Without
``SamplerStep.kept_rids`` the dropped draws advanced the RNG offset and, under a
repetition penalty, marked the dropped token as seen, so seeded output depended
on how the prompt was split into steps.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import SlotLease, StepContext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer sampler requires CUDA"
)

V = 64
KEPT, DROPPED = 0, 1


def _resource(graph: bool) -> SamplerResource:
    res = SamplerResource(
        vocab_size=V, enable_repetion_penalty=True, device=torch.device("cuda"),
    )
    if graph:
        res.build_cuda_graph_buffers([SimpleNamespace(slot=0)], max_bs=2, max_seq_len=1)
    for rid in (KEPT, DROPPED):
        res.ingest_request(
            rid, SamplingReqConfig(temperature=1.0, repetition_penalty=1.3),
        )
    return res


def _step(res: SamplerResource, step: SamplerStep, graph: bool) -> torch.Tensor:
    ctx = StepContext(
        request_ids=(KEPT, DROPPED), graph_walk="prefill", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=SimpleNamespace(bs=2)) if graph else None,
    )
    res.plan(step, ctx)
    logits = torch.randn(2, V, device="cuda")
    tokens = res.sample([KEPT, DROPPED], logits)
    res.commit(step, ctx)
    torch.cuda.synchronize()
    return tokens


def _offset(res: SamplerResource, rid: int, graph: bool) -> int:
    if graph:
        slot = res._cg_buffers._rid_to_slot[rid]
        return int(res._cg_buffers.offset.master[slot].item())
    return res._sampler._step_offset[rid]


def _seen(res: SamplerResource, rid: int) -> int:
    mask = res._sampler.get_token_mask(rid)._seen_token_mask
    return 0 if mask is None else int(mask.sum().item())


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_dropped_row_commits_nothing(graph):
    res = _resource(graph)
    _step(res, SamplerStep(kept_rids=frozenset({KEPT})), graph)

    assert _offset(res, KEPT, graph) > 0
    assert _offset(res, DROPPED, graph) == 0
    assert _seen(res, KEPT) == 1
    assert _seen(res, DROPPED) == 0


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_default_commits_every_row(graph):
    res = _resource(graph)
    _step(res, SamplerStep(), graph)

    for rid in (KEPT, DROPPED):
        assert _offset(res, rid, graph) > 0
        assert _seen(res, rid) == 1


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_dropped_steps_do_not_shift_the_kept_stream(graph):
    """Two dropped steps ahead of a kept one draw what a lone kept step does."""
    torch.manual_seed(0)
    direct = _resource(graph)
    want = _step(direct, SamplerStep(), graph)[0].item()

    torch.manual_seed(0)
    split = _resource(graph)
    none_kept = SamplerStep(kept_rids=frozenset())
    _step(split, none_kept, graph)
    _step(split, none_kept, graph)
    torch.manual_seed(0)  # same logits as the direct run
    got = _step(split, SamplerStep(), graph)[0].item()

    assert got == want
