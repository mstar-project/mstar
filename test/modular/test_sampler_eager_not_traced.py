"""A step outside a captured graph samples eagerly, even from a compiled forward.

Traced along with the forward, the eager sampler cost about half a second of
dynamo time per uncaptured step with the GPU idle. On ROCm every prefill step
is uncaptured.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer sampler requires CUDA"
)

V = 64


def test_eager_sampler_is_not_traced():
    res = SamplerResource(
        vocab_size=V, enable_repetion_penalty=False, device=torch.device("cuda"),
    )
    traced = []
    sample = res._sampler.sample

    def spy(*args, **kwargs):
        traced.append(torch.compiler.is_compiling())
        return sample(*args, **kwargs)

    res._sampler.sample = spy

    @torch.compile(backend="eager")
    def head(logits, request_ids):
        return res.sample(request_ids, logits * 2)

    for batch in (("a", "b"), ("c", "d")):
        for rid in batch:
            res.ingest_request(rid, SamplingReqConfig(temperature=1.0))
        ctx = StepContext(
            request_ids=batch, graph_walk="prefill", slot=0, capture=False, slot_lease=None,
        )
        res.plan(SamplerStep(), ctx)
        head(torch.randn(2, V, device="cuda"), list(batch))
        res.commit(SamplerStep(), ctx)

    assert traced and not any(traced), f"sampler traced by dynamo: {traced}"
