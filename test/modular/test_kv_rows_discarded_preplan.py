"""A persistent walk's rows keep their KV state across a discarded pre-plan.

A pre-plan's admit reserves the page a row's next token crosses onto, and the
page stays with the stream when the pre-plan is discarded. The step that then
runs reserves nothing new, so the row must still be given that page."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import SINK_PAGE  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402
from mstar.engine.rows import RowSpace  # noqa: E402

PAGE = 4


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


@pytest.fixture
def kv() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64,
            max_num_pages=64, page_size=PAGE,
        ),
        name="ctx", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    return kv


def _prefill(kv: KVManager, rid: int, n: int) -> None:
    kv.ingest_request(rid, KVReqConfig())
    step = KVStep(segments=(Segment(rid, "main", n),))
    ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _decode_ctx(space: RowSpace, rids: tuple[int, ...], preplan: bool = False):
    step = KVStep(segments=tuple(Segment(r, "main", 1) for r in rids))
    bucket = BucketKey(graph_walk="decode", bs=len(rids), num_tokens=len(rids))
    ctx = StepContext(
        request_ids=rids, graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=bucket),
    )
    ctx.set_padded_rids(rids)
    ctx.rows = space.propose(list(rids))
    ctx.is_preplan = preplan
    return step, ctx


def _run(kv: KVManager, space: RowSpace, rids: tuple[int, ...]) -> None:
    """One decode step, as the engine drives it: admit, plan, the rows'
    upload, commit and gather, the forward's length step, then KV's commit."""
    step, ctx = _decode_ctx(space, rids)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    ctx.rows.stage()
    ctx.rows.commit()
    ctx.rows.apply()
    space.gather(len(rids))
    space.after_forward()
    kv.commit(step, ctx)


def _discarded_preplan(kv: KVManager, space: RowSpace, rids: tuple[int, ...]) -> None:
    step, ctx = _decode_ctx(space, rids, preplan=True)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.clear_preplan()  # the engine's reset of a pre-plan it discards


def _check_rows(kv: KVManager, space: RowSpace, rids, when: str) -> None:
    """Each row holds its stream: the stored length, on the mirror and the
    device, and every page."""
    lens, pages = kv._kv_rows["decode"].labels["main"]
    for rid in rids:
        stream = kv._streams[rid]["main"]
        slot = space.batch.slot_of(rid)
        held = stream.page_indices.tolist()
        row = pages.resident[slot, :len(held)].tolist()
        assert row == held, f"{when}: row {rid} holds pages {row}, its stream {held}"
        assert SINK_PAGE not in row
        assert int(lens.resident[slot]) == stream.stored_len, f"{when}: row {rid}'s device length"
        assert int(space.committed(lens, rid)) == stream.stored_len, f"{when}: row {rid}'s mirror length"


@pytest.mark.parametrize("discard_at", [0, 1, 2, 3, "every"])
def test_a_row_keeps_the_page_a_discarded_pre_plan_reserved(kv, discard_at):
    # row 0 starts 3 tokens in; its steps cross onto a new page at lengths 4
    # and 8. Row 1 is offset by one so the two cross on different steps.
    _prefill(kv, 0, 3)
    _prefill(kv, 1, 2)
    space = RowSpace(4, torch.device("cpu"), labels={"ctx": ("main",)})
    kv.row_state(space, "decoder", "decode")
    rids = (0, 1)
    for i in range(8):
        if discard_at == "every" or i % 4 == discard_at:
            _discarded_preplan(kv, space, rids)
        _run(kv, space, rids)
        _check_rows(kv, space, rids, f"step {i}")
