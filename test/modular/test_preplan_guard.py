"""A staged pre-plan promotes only into the step it was planned for.

The plan thread pre-plans the speculated next step of a node while the current
one runs; if the scheduler then dispatches a *different* batch first (a new
request's prefill while a decode step sits pre-planned), that batch must plan
inline and the staged plan must be dropped — not served the other step's
pages (KV) or dereferenced against a lease it does not hold (sampler).
"""

from __future__ import annotations

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext

PS = 8


def _manager() -> KVManager:
    cfg = KVConfig(num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=64 * PS, max_num_pages=64, page_size=PS)
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids: str, preplan: bool = False) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False, is_preplan=preplan)


def _grow(m: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),), commit=True)
    ctx = _ctx(rid)
    assert m.admit(step, ctx).ok
    m.plan(step, ctx)
    m.commit(step, ctx)


def test_kv_preplan_promotes_only_into_its_own_step() -> None:
    m = _manager()
    for rid in ("a", "b"):
        m.ingest_request(rid)
    _grow(m, "a", 2 * PS)

    # Stage a's next (decode) step ahead.
    step_a = KVStep(segments=(Segment("a", "main", 1),), commit=True)
    ctx_a = _ctx("a", preplan=True)
    assert m.admit(step_a, ctx_a).ok
    staged = m.plan(step_a, ctx_a)
    assert m._preplanned and staged["main"].views[0].request_id == "a"

    # A different batch (b's prefill) reaches the GPU thread first: it is
    # planned inline, on its own pages, and the staged plan is dropped.
    step_b = KVStep(segments=(Segment("b", "main", 3 * PS),), commit=True)
    ctx_b = _ctx("b")
    assert m.admit(step_b, ctx_b).ok
    out_b = m.plan(step_b, ctx_b)
    assert not m._preplanned
    (view,) = out_b["main"].views
    assert view.request_id == "b" and view.to_compute == 3 * PS
    m.commit(step_b, ctx_b)
    assert m._streams["b"]["main"].stored_len == 3 * PS

    # a's step then plans inline like any un-staged step and commits its token.
    assert m.admit(step_a, _ctx("a")).ok
    out_a = m.plan(step_a, _ctx("a"))
    assert out_a["main"].views[0].request_id == "a"
    m.commit(step_a, _ctx("a"))
    assert m._streams["a"]["main"].stored_len == 2 * PS + 1


def test_kv_preplan_still_promotes_for_its_own_step() -> None:
    m = _manager()
    m.ingest_request("a")
    _grow(m, "a", PS)
    step = KVStep(segments=(Segment("a", "main", 1),), commit=True)
    assert m.admit(step, _ctx("a", preplan=True)).ok
    staged = m.plan(step, _ctx("a", preplan=True))
    promoted = m.plan(step, _ctx("a"))
    assert promoted is staged and not m._preplanned
    m.commit(step, _ctx("a"))
    assert m._streams["a"]["main"].stored_len == PS + 1
