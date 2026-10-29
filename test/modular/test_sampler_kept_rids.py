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

V = 64
KEPT, DROPPED = 0, 1


@pytest.fixture(params=["cuda", "xpu"])
def device(request):
    runtime = getattr(torch, request.param)
    if not runtime.is_available():
        pytest.skip(f"requires {request.param}")
    if request.param == "xpu":
        pytest.importorskip("vllm_xpu_kernels._xpu_C")
    return torch.device(request.param, 0)


@pytest.fixture(params=["cpu", "cuda", "xpu"])
def buffer_device(request):
    if request.param != "cpu" and not getattr(torch, request.param).is_available():
        pytest.skip(f"requires {request.param}")
    return torch.device(request.param)


def _resource(graph: bool, device: torch.device) -> SamplerResource:
    res = SamplerResource(
        vocab_size=V, enable_repetion_penalty=True, device=device,
    )
    if graph:
        res.build_accelerator_graph_buffers(
            [SimpleNamespace(slot=0)], max_bs=4, max_seq_len=1,
        )
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
    logits = torch.randn(2, V).to(res._device)
    tokens = res.sample([KEPT, DROPPED], logits)
    res.commit(step, ctx)
    torch.accelerator.synchronize(res._device)
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
def test_dropped_row_commits_nothing(graph, device):
    res = _resource(graph, device)
    _step(res, SamplerStep(kept_rids=frozenset({KEPT})), graph)

    assert _offset(res, KEPT, graph) > 0
    assert _offset(res, DROPPED, graph) == 0
    assert _seen(res, KEPT) == 1
    assert _seen(res, DROPPED) == 0


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_default_commits_every_row(graph, device):
    res = _resource(graph, device)
    _step(res, SamplerStep(), graph)

    for rid in (KEPT, DROPPED):
        assert _offset(res, rid, graph) > 0
        assert _seen(res, rid) == 1


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_dropped_steps_do_not_shift_the_kept_stream(graph, device):
    """Two dropped steps ahead of a kept one draw what a lone kept step does."""
    torch.manual_seed(0)
    direct = _resource(graph, device)
    want = _step(direct, SamplerStep(), graph)[0].item()

    torch.manual_seed(0)
    split = _resource(graph, device)
    none_kept = SamplerStep(kept_rids=frozenset())
    _step(split, none_kept, graph)
    _step(split, none_kept, graph)
    torch.manual_seed(0)  # same logits as the direct run
    got = _step(split, SamplerStep(), graph)[0].item()

    assert got == want


@pytest.mark.parametrize(
    "kept_rids",
    [None, frozenset({KEPT, DROPPED}), frozenset({KEPT}), frozenset()],
    ids=["default", "all", "mixed", "none"],
)
def test_commit_keeps_only_selected_rows_and_ignores_padding(buffer_device, kept_rids):
    """Commit staged draw results without depending on a sampling kernel."""
    res = _resource(True, buffer_device)
    step = SamplerStep(kept_rids=kept_rids)
    ctx = StepContext(
        request_ids=(DROPPED, KEPT), graph_walk="prefill", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=SimpleNamespace(bs=4)),
    )
    res.plan(step, ctx)
    sampler = res._cg_sampler
    sampler.offset_buf.add_(1)
    tokens = torch.tensor([[3], [5], [7], [9]], device=buffer_device)
    sampler.seen_tokens_buf.scatter_(1, tokens, True)
    res.commit(step, ctx)

    for rid, token in ((DROPPED, 3), (KEPT, 5)):
        kept = kept_rids is None or rid in kept_rids
        assert _offset(res, rid, True) == int(kept)
        assert _seen(res, rid) == int(kept)
        mask = res._sampler.get_token_mask(rid)._seen_token_mask
        assert bool(mask[token]) == kept
    assert res._cg_buffers.offset.master[0].item() == 0
