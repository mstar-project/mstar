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


def _mgr(pages=6, name="kv"):
    cfg = KVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=pages * PAGE, max_num_pages=pages, page_size=PAGE,
    )
    return KVManager(
        cfg=cfg, name=name, joint_comm_group=None, transfer_engine_info=None,
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


def test_a_refused_admit_leaves_no_resource_marked():
    """Whisper's decoder: cross_kv_cache admits before kv_cache. When kv_cache
    refuses, the cross stream must not stay in flight, or offload refuses it."""
    from mstar.engine.resources import SubmoduleStep
    from mstar.engine.resources.runner import StepRunner

    cross, kv = _mgr(6, "cross_kv_cache"), _mgr(2, "kv_cache")
    runner = StepRunner({"cross_kv_cache": cross, "kv_cache": kv})
    for m in (cross, kv):
        m.ingest_request("a")
    prefill = SubmoduleStep(steps={"cross_kv_cache": KVStep(segments=(Segment("a", "main", 8),))})
    prefill.set_ctx(_ctx("a"))
    assert runner.admit(prefill).ok
    runner.plan(prefill)
    runner.commit(prefill)

    # reads the encoder context (zero span) and needs more self-attn pages than exist
    decode = SubmoduleStep(steps={
        "cross_kv_cache": KVStep(segments=(Segment("a", "main", 0),)),
        "kv_cache": KVStep(segments=(Segment("a", "main", 16),)),
    })
    decode.set_ctx(_ctx("a"))
    outcome = runner.admit(decode)
    assert not outcome.ok and outcome.failed_resource == "kv_cache"
    assert not cross._streams["a"]["main"].step_in_flight
