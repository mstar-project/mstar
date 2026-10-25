"""An uncaptured step samples from a compiled forward without recompiling it
for every new batch.

The sampler takes only the logits and reads the step's rows from its plan, so
the forward hands dynamo no ids to guard on: not their values (the eager
sampler looks rows up by request handle) and not their count. Either guard
recompiled the forward for each new batch while the GPU sat idle.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.model.submodule_base import ModelInputsFromEngine

V = 32


class _Masters:
    """Stands in for the slot masters an eager loop-back reads."""

    def __init__(self):
        self.read: list[tuple] = []

    def register_request(self, rid, sampling_config):
        pass

    def last_tokens_for(self, rows):
        self.read.append(tuple(rows))
        return torch.tensor([r % V for r in rows])


class _GraphSampler:
    """Stands in for the AcceleratorGraphableSampler a leased plan hands back."""

    def __init__(self):
        self.calls: list[tuple] = []

    def sample(self, request_ids, logits, apply_penalty=False, apply_filters=True):
        self.calls.append((tuple(request_ids), apply_penalty))
        return torch.zeros(len(request_ids), dtype=torch.long)


class _GraphBuffers:
    """Stands in for SamplerBuffers on the plan path of a leased step."""

    def __init__(self):
        self.sampler = _GraphSampler()

    def register_request(self, rid, sampling_config):
        pass

    def gather_static(self, request_ids, padded_bs, cg_slot):
        pass

    def gather_dynamic(self, request_ids, padded_bs, cg_slot, gather_seen_tokens=True):
        pass

    def sampler_for(self, padded_bs, cg_slot):
        return self.sampler


def _resource(monkeypatch, buffers=None) -> tuple[SamplerResource, list[bool]]:
    res = SamplerResource(vocab_size=V, enable_repetion_penalty=False, device=torch.device("cpu"))
    res._cg_buffers = buffers
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


def _plan(res, handles, kept=None, lease=None, padded=None):
    for h in handles:
        if h not in res._sampler._sampling_config:
            res.ingest_request(h, SamplingReqConfig(temperature=0.0))
    step = SamplerStep(kept_rids=None if kept is None else frozenset(kept))
    ctx = StepContext(
        request_ids=tuple(handles), graph_walk="prefill", slot=0, capture=False, slot_lease=lease,
    )
    if padded is not None:
        ctx.set_padded_rids(tuple(padded))
    res.plan(step, ctx)
    return step, ctx


def _step(res, head, handles, kept):
    step, ctx = _plan(res, handles, kept)
    engine_inputs = ModelInputsFromEngine(
        request_ids=list(handles), per_request_info={}, resources={"sampler": res},
    )
    out = head(torch.randn(len(handles), V), engine_inputs)
    res.commit(step, ctx)
    return out


def _compiled_forward():
    # built like a model forward: the ids ride on the inputs, the sampler is not handed them
    @torch.compile(backend="eager", dynamic=None)
    def forward(hidden, engine_inputs):
        sampler = engine_inputs.resources["sampler"]
        last = torch.nn.functional.one_hot(sampler.loopback_tokens(), V)
        return sampler.sample(hidden * 2 + last)

    return forward


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_a_new_batch_of_a_seen_size_reuses_the_compiled_forward(monkeypatch, fresh_dynamo):
    res, traced = _resource(monkeypatch, _Masters())
    forward = _compiled_forward()
    _step(res, forward, [0, 1], kept=[0, 1])
    with torch._dynamo.config.patch(error_on_recompile=True):
        _step(res, forward, [2, 3], kept=[2])
        _step(res, forward, [1, 4], kept=[1, 4])
    assert traced == [False, False, False]


def test_a_new_batch_size_reuses_the_compiled_forward(monkeypatch, fresh_dynamo):
    res, traced = _resource(monkeypatch, _Masters())
    forward = _compiled_forward()
    # the second size makes the batch dimension dynamic
    _step(res, forward, [0], kept=[0])
    _step(res, forward, [1, 2], kept=[1, 2])
    with torch._dynamo.config.patch(error_on_recompile=True):
        for n in (3, 5, 7):
            rows = list(range(10 * n, 11 * n))
            _step(res, forward, rows, kept=rows)
    assert traced == [False] * 5
    assert res._cg_buffers.read[-1] == tuple(range(70, 77))


def test_the_sampler_reads_only_a_planned_step(monkeypatch):
    res, _ = _resource(monkeypatch)
    with pytest.raises(RuntimeError, match="outside a planned step"):
        res.sample(torch.zeros(2, V))
    step, ctx = _plan(res, [0, 1])
    # a pre-plan of the next step, on the plan thread, leaves this step's rows be
    res.plan(SamplerStep(), StepContext(
        request_ids=(5, 6, 7), graph_walk="decode", slot=1, capture=False, is_preplan=True,
    ))
    with pytest.raises(RuntimeError, match="3 rows of logits for a step planned with 2"):
        res.sample(torch.zeros(3, V))
    assert res.sample(torch.zeros(2, V)).shape == (2,)
    res.commit(step, ctx)
    with pytest.raises(RuntimeError, match="outside a planned step"):
        res.sample(torch.zeros(2, V))


def test_a_leased_step_hands_the_graph_sampler_its_padded_rows(monkeypatch):
    res, _ = _resource(monkeypatch, _GraphBuffers())
    lease = SlotLease(slot=0, bucket=SimpleNamespace(bs=4))
    _plan(res, [0, 1], lease=lease, padded=[0, 1, -1, -2])
    res.sample(torch.zeros(4, V))
    assert res._cg_buffers.sampler.calls == [((0, 1, -1, -2), res._apply_penalty_this_step)]


def test_an_eager_loop_back_reads_the_planned_rows(monkeypatch):
    res, _ = _resource(monkeypatch, _Masters())
    _plan(res, [3, 1, 4])
    assert res.loopback_tokens().tolist() == [3, 1, 4]
    assert res._cg_buffers.read == [(3, 1, 4)]
