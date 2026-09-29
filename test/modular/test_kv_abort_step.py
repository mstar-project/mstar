"""`KVManager.abort_step`: a step admitted but never committed must not
leave its streams marked in flight, or the OOM handler can't evict them."""

import pytest
import torch

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager

PAGE = 8


class _NoTransfer:
    def __init__(self, *a, **k):
        pass

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **k):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _NoTransfer)


def _mgr(pages=6):
    cfg = KVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=pages * PAGE, max_num_pages=pages, page_size=PAGE,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(rid):
    return StepContext(request_ids=(rid,), graph_walk="w", slot=0, capture=False)


def test_abort_clears_the_mark_and_keeps_length_and_pages():
    m = _mgr()
    m.ingest_request("a")
    step = KVStep(segments=(Segment("a", "main", 10),))
    assert m.admit(step, _ctx("a")).ok
    m.plan(step, _ctx("a"))
    stream = m._streams["a"]["main"]
    assert stream.step_in_flight

    m.abort_step(step, _ctx("a"))

    assert not stream.step_in_flight
    assert stream.stored_len == 0 and len(stream.page_indices) == 2
    claimed, _ = m._claim_for_offload("a")
    assert claimed, "an aborted rid is evictable again"


def test_abort_tolerates_a_zero_span_segment_with_no_stream():
    m = _mgr()
    m.ingest_request("a")
    step = KVStep(segments=(Segment("a", "never_made", 0),))
    assert m.admit(step, _ctx("a")).ok

    m.abort_step(step, _ctx("a"))


def test_abort_unblocks_a_rid_admitted_before_a_refusal():
    """`_exec_per_request` admits every rid before driving any: when a later
    one is refused, the earlier ones never commit."""
    m = _mgr(pages=4)  # 1 sink + 3 usable
    for rid in ("a", "b"):
        m.ingest_request(rid)
    sa = KVStep(segments=(Segment("a", "main", 16),))
    sb = KVStep(segments=(Segment("b", "main", 16),))
    assert m.admit(sa, _ctx("a")).ok
    assert not m.admit(sb, _ctx("b")).ok
    claimed, _ = m._claim_for_offload("a")
    assert claimed == [], "still in flight, so the OOM handler can't take it"

    m.abort_step(sa, _ctx("a"))

    claimed, _ = m._claim_for_offload("a")
    assert claimed
