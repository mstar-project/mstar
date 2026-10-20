"""A worker serving two partitions of a request ingests it twice.

The conductor sends one NewRequest per partition, and the second can arrive after
the first partition's prefill already committed. A position counter reset there
puts the next decode token at rope position 0 while the KV holds the whole prompt.
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
from mstar.engine.resources.step import Segment, SubmoduleStep

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
    device = torch.device("cpu")
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=16,
        ),
        name=KV, joint_comm_group=None, transfer_engine_info=None,
        device=device, dtype=torch.float32,
    )
    rope = RopeManager(config=PositionConfig(kv_cache=KV), device=device, dtype=torch.float32)
    return StepRunner({KV: kv, ROPE: rope}, node_resources={NODE: [KV, ROPE]})


def _step(runner: StepRunner, span: int, walk: str) -> list[int]:
    step = SubmoduleStep(
        steps={KV: KVStep(), ROPE: PositionStep()},
        segments=[Segment(RID, "main", span)],
    )
    step.set_ctx(StepContext(request_ids=(RID,), graph_walk=walk, slot=0, capture=False))
    assert runner.admit(step).outcome.ok
    pos_ids = runner.plan(step)[ROPE]["main"].tolist()
    runner.commit(step)
    return pos_ids


def test_a_repeat_ingest_after_the_prefill_keeps_the_position():
    runner = _runner()
    runner.ingest_request(RID)
    assert _step(runner, 119, "prefill") == list(range(119))

    # the second partition's NewRequest, after the prefill committed
    runner.ingest_request(RID)

    assert _step(runner, 1, "decode") == [119]
    assert _step(runner, 1, "decode") == [120]


def test_a_repeat_ingest_before_any_step_changes_nothing():
    runner = _runner()
    runner.ingest_request(RID)
    runner.ingest_request(RID)
    assert _step(runner, 4, "prefill") == [0, 1, 2, 3]
