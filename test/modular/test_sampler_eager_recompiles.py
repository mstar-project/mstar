"""An uncaptured step samples from a compiled forward without recompiling it
for every new batch.

The eager sampler looks rows up by request handle. Traced with the forward, the
call guards on the handles and on the kept set, so each new mix of requests
recompiled it while the GPU sat idle.
"""

import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext

V = 32


def _resource(monkeypatch) -> tuple[SamplerResource, list[bool]]:
    res = SamplerResource(vocab_size=V, enable_repetion_penalty=False, device=torch.device("cpu"))
    traced = []

    # Stands in for the FlashInfer kernels, which stay out of dynamo the same way.
    @torch.compiler.disable
    def greedy(request_ids, logits, **kwargs):
        return logits.argmax(-1)

    def spy(request_ids, logits, **kwargs):
        traced.append(torch.compiler.is_compiling())
        return greedy(request_ids, logits, **kwargs)

    monkeypatch.setattr(res._sampler, "sample", spy)
    return res, traced


def _step(res, head, handles, kept):
    for h in handles:
        if h not in res._sampler._sampling_config:
            res.ingest_request(h, SamplingReqConfig(temperature=0.0))
    step = SamplerStep(kept_rids=frozenset(kept))
    ctx = StepContext(request_ids=handles, graph_walk="prefill", slot=0, capture=False, slot_lease=None)
    res.plan(step, ctx)
    out = head(torch.randn(len(handles), V), list(handles))
    res.commit(step, ctx)
    return out


def test_a_new_batch_of_a_seen_size_reuses_the_compiled_forward(monkeypatch):
    res, traced = _resource(monkeypatch)

    @torch.compile(backend="eager", dynamic=None)
    def head(hidden, request_ids):
        return res.sample(request_ids, hidden * 2)

    torch._dynamo.reset()
    try:
        _step(res, head, [0, 1], kept=[0, 1])
        with torch._dynamo.config.patch(error_on_recompile=True):
            _step(res, head, [2, 3], kept=[2])
            _step(res, head, [1, 4], kept=[1, 4])
    finally:
        torch._dynamo.reset()
    assert traced == [False, False, False]
