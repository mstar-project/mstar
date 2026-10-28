"""Warmup zeroes SINK_PAGE once every capture has run.

Every capture row runs on SINK_PAGE (#729), on dummy inputs. On an H100,
Qwen3.5-0.8B's SINK_PAGE was non-finite in every layer after warmup and still
was after 67 served requests, while every real page was zero. A real row's
padded block table names SINK_PAGE too (``XPUAttentionManager._build_plan``, and
the AITER tables of #685, whose tail tile reads it), so a kernel that masks the
row's tail arithmetically would turn it into NaN.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

BUCKET = BucketKey(graph_walk="mixed", bs=4, num_tokens=40)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=8, page_size=4,
        ),
        name="kv", joint_comm_group=SimpleNamespace(world_size=1, rank=0), transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = BUCKET.num_tokens
    return kv


def _capture(kv: KVManager) -> set[int]:
    """A capture admitted and planned as the runner does it, with inf written
    to every page its plan sends the K/V to. Returns those pages."""
    rids = (-1, -2, -3, -4)
    for rid in rids:
        kv.ingest_capture_row(rid)
    step = KVStep(segments=tuple(
        Segment(rid, "main", span) for rid, span in zip(rids, (1, 1, 1, 37), strict=True)
    ))
    ctx = StepContext(
        request_ids=rids, graph_walk=BUCKET.graph_walk, slot=0, capture=True,
        slot_lease=SlotLease(slot=0, bucket=BUCKET), force_prefill=True,
    )
    assert kv.admit(step, ctx).ok
    pages = kv.plan(step, ctx)["main"].cpu_indptrs.paged_kv_indices
    kv.kv_cache.tensor[:, pages.long()] = math.inf
    return set(pages.tolist())


def _warmup(resources: dict) -> None:
    """The real ``Engine.warmup`` on an engine with no nodes to capture."""
    engine = Engine.__new__(Engine)
    engine._submodules = {}
    engine._device = torch.device("cpu")
    engine._resources = resources
    Engine.warmup(engine)


def test_warmup_zeroes_what_capture_wrote():
    kv = _manager()
    assert _capture(kv) == {SINK_PAGE}
    _warmup({"kv": kv})
    assert torch.count_nonzero(kv.kv_cache.tensor) == 0


def test_only_the_sink_page_is_cleared():
    kv = _manager()
    _capture(kv)
    kv.kv_cache.tensor[:, 1] = 2.0
    kv.after_capture()
    assert torch.count_nonzero(kv.kv_cache.tensor[:, SINK_PAGE]) == 0
    assert bool((kv.kv_cache.tensor[:, 1] == 2.0).all())
