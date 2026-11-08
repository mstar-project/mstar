"""A request ingested after its handle was removed starts from a clean slate.

The worker opens a request's resources once, on its first NewRequest, and handles
are recycled after removal, so the next request to get a handle must not inherit
the position counter, penalty mask or RNG offset of the one before it.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import PagedKVConfig, PositionConfig, StepContext, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import Segment, SubmoduleStep

CPU = torch.device("cpu")
KV = "kv"
ROPE = "rope"
RID = "r0"
NODE = "LLM"


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _runner() -> StepRunner:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=16,
        ),
        name=KV, joint_comm_group=None, transfer_engine_info=None,
        device=CPU, dtype=torch.float32,
    )
    rope = RopeManager(config=PositionConfig(kv_cache=KV), device=CPU, dtype=torch.float32)
    return StepRunner({KV: kv, ROPE: rope}, node_resources={NODE: [KV, ROPE]})


def _step(runner: StepRunner, span: int) -> list[int]:
    step = SubmoduleStep(
        steps={KV: KVStep(), ROPE: PositionStep()},
        segments=[Segment(RID, "main", span)],
    )
    step.set_ctx(StepContext(request_ids=(RID,), graph_walk="prefill", slot=0, capture=False))
    assert runner.admit(step).outcome.ok
    pos_ids = runner.plan(step)[ROPE]["main"].tolist()
    runner.commit(step)
    return pos_ids


def _sampled(sampler, token: int = 5) -> int:
    """What one sampled step leaves behind (sampling itself needs CUDA or XPU)."""
    sampler.get_token_mask(RID).add_tokens(token)
    sampler._step_offset[RID] += 1
    return token


def test_remove_then_ingest_still_starts_fresh():
    """A recycled handle starts clean."""
    runner = _runner()
    runner.ingest_request(RID)
    _step(runner, 4)
    runner.remove_request(RID)

    runner.ingest_request(RID)

    assert _step(runner, 2) == [0, 1]

    res = SamplerResource(vocab_size=16, enable_repetion_penalty=True, device=CPU)
    res.ingest_request(RID, SamplingReqConfig(repetition_penalty=1.3))
    _sampled(res._sampler)
    res.remove_request(RID)

    res.ingest_request(RID, SamplingReqConfig(temperature=0.5))

    assert res._sampler._step_offset[RID] == 0
    assert res._sampler._sampling_config[RID].temperature == pytest.approx(0.5)
    assert RID not in res._penalty_rids
