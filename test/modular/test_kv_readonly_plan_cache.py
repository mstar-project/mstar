"""The KV manager's read-only plan reuse: a step whose segments are all span
0 (a decoder reading an encoder context written at prefill) gets the last
read-only plan back while nothing touched the streams of its real rows, and
plans again as soon as something did."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import PinnedIndexRing  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

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


def _manager(device="cpu", cpu_offload_pages=0) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=256,
            max_num_pages=64, page_size=PAGE, cpu_offload_pages=cpu_offload_pages,
        ),
        name="ctx", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device(device), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    return kv


def _write(kv: KVManager, rid: str, n: int, ingest: bool = True, **step_kw) -> None:
    """One eager step writing ``n`` tokens of ``rid``'s context."""
    if ingest:
        kv.ingest_request(rid, KVReqConfig())
    step = KVStep(segments=(Segment(rid, "main", n),), **step_kw)
    ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _read(
    kv: KVManager, rids: tuple[str, ...], pad: int = 2, slot: int = 0, sink=frozenset(),
    pad_name: str = "pad", preplan: bool = False,
):
    """A captured read-only step over ``rids`` plus ``pad`` padding rows. A
    pre-plan is promoted by the step's own plan before the commit, as the
    worker's GPU thread does."""
    padded = rids + tuple(f"{pad_name}{i}" for i in range(pad))
    step = KVStep(segments=tuple(Segment(r, "main", 0) for r in padded), sink=sink)
    bucket = BucketKey(graph_walk="decode", bs=len(padded), num_tokens=len(padded))

    def ctx(is_preplan: bool) -> StepContext:
        c = StepContext(
            request_ids=rids, graph_walk="decode", slot=slot, capture=False,
            slot_lease=SlotLease(slot=slot, bucket=bucket), is_preplan=is_preplan,
        )
        c.set_padded_rids(padded)
        return c

    first = ctx(preplan)
    assert kv.admit(step, first).ok
    res = kv.plan(step, first)
    if preplan:
        assert kv._preplanned
        promoted = ctx(False)
        assert kv.plan(step, promoted) is res and not kv._preplanned
        first = promoted
    kv.commit(step, first)
    return res


def _lengths(res) -> list[int]:
    return [v.length for v in res["main"].views]


@pytest.fixture
def kv():
    kv = _manager()
    _write(kv, "a", 6)
    _write(kv, "b", 9)
    return kv


def test_an_identical_read_only_step_reuses_the_plan(kv):
    first = _read(kv, ("a", "b"))
    assert _lengths(first)[:2] == [6, 9]
    assert _read(kv, ("a", "b")) is first
    # its slot's static buffers already hold it: not staged again
    assert kv._staged_outputs[(0, "main")] is first["main"]
    assert kv._current_plan_states["main"] is kv._static_plan_state(0, "main")


def test_the_padding_rows_reset_after_a_replay_keeps_the_plan(kv):
    first = _read(kv, ("a", "b"))
    # what the graph runner's release does after every padded replay
    for pad in ("pad0", "pad1"):
        kv.reset_request(pad, free=True)
    assert _read(kv, ("a", "b")) is first


def test_another_requests_ingest_and_write_keep_the_plan(kv):
    first = _read(kv, ("a", "b"))
    _write(kv, "c", 5)
    assert _read(kv, ("a", "b")) is first


def test_a_write_to_a_real_row_plans_again(kv):
    first = _read(kv, ("a", "b"))
    _write(kv, "a", 3, ingest=False)
    again = _read(kv, ("a", "b"))
    assert again is not first and _lengths(again)[:2] == [9, 9]


def test_a_reset_of_a_real_row_plans_again(kv):
    first = _read(kv, ("a", "b"))
    kv.reset_request("b")
    _write(kv, "b", 2, ingest=False)
    again = _read(kv, ("a", "b"))
    assert again is not first and _lengths(again)[:2] == [6, 2]


def test_a_removed_and_recycled_handle_plans_again(kv):
    first = _read(kv, ("a", "b"))
    kv.remove_request("b")
    # the handle comes back for another request with another context
    _write(kv, "b", 3)
    again = _read(kv, ("a", "b"))
    assert again is not first and _lengths(again)[:2] == [6, 3]


def test_a_touch_of_every_stream_plans_again(kv):
    first = _read(kv, ("a", "b"))
    kv._touch_plan_cache()
    assert _read(kv, ("a", "b")) is not first


def test_a_step_with_sink_rows_plans_its_own_views(kv):
    """The same segments with a row on the sink page are another plan."""
    first = _read(kv, ("a", "b"))
    sunk = _read(kv, ("a", "b"), sink=frozenset({("b", "main")}))
    assert sunk is not first and _lengths(sunk)[:2] == [6, 0]
    # the plain step still reuses its plan, staged again over the sink step's
    assert _read(kv, ("a", "b")) is first
    assert kv._staged_outputs[(0, "main")] is first["main"]


def test_another_slot_stages_the_reused_plan(kv):
    first = _read(kv, ("a", "b"), slot=0)
    # the same step on slot 1 reuses the plan, but slot 1's static buffers
    # do not hold it yet, so it is staged there
    assert (1, "main") not in kv._staged_outputs
    assert _read(kv, ("a", "b"), slot=1) is first
    assert kv._staged_outputs[(1, "main")] is first["main"]
    assert kv._current_plan_states["main"] is kv._static_plan_state(1, "main")


def test_slots_that_pad_with_rows_of_their_own_keep_a_plan_each(kv):
    """Replay slots pad with their own rows, so steps that alternate slots
    (the pre-plan's double buffer) are two keys; neither evicts the other."""
    for pad in ("s0_", "s1_"):
        for i in range(2):
            kv.ingest_capture_row(f"{pad}{i}")
    first = _read(kv, ("a", "b"), slot=0, pad_name="s0_")
    other = _read(kv, ("a", "b"), slot=1, pad_name="s1_")
    assert other is not first
    assert _read(kv, ("a", "b"), slot=0, pad_name="s0_") is first
    assert _read(kv, ("a", "b"), slot=1, pad_name="s1_") is other


def test_a_preplanned_read_only_step_reuses_the_plan(kv):
    first = _read(kv, ("a", "b"), preplan=True)
    again = _read(kv, ("a", "b"), preplan=True)
    assert again is first
    assert kv._current_plan_states["main"] is kv._static_plan_state(0, "main")
    # a plan made inline and one made ahead are the same entry
    assert _read(kv, ("a", "b")) is first


def test_a_preplan_that_is_not_promoted_plans_again(kv):
    first = _read(kv, ("a", "b"), preplan=True)
    padded = ("a", "b", "pad0", "pad1")
    step = KVStep(segments=tuple(Segment(r, "main", 0) for r in padded))
    bucket = BucketKey(graph_walk="decode", bs=4, num_tokens=4)
    ctx = StepContext(
        request_ids=("a", "b"), graph_walk="decode", slot=1, capture=False,
        slot_lease=SlotLease(slot=1, bucket=bucket), is_preplan=True,
    )
    ctx.set_padded_rids(padded)
    assert kv.admit(step, ctx).ok
    assert kv.plan(step, ctx) is first
    kv.clear_preplan()
    assert _read(kv, ("a", "b")) is not first


def test_the_cached_plan_keeps_its_indptrs_while_the_ring_turns(kv):
    """Index arrays from the pinned ring are rewritten two plans later; a
    cached plan staged on a new slot reads its own."""
    kv._indptr_ring = PinnedIndexRing(numel=256, depth=2)
    first = _read(kv, ("a", "b"))
    qo = first["main"].cpu_indptrs.qo_indptr.clone()
    _write(kv, "c", 5)
    _write(kv, "d", 7)
    assert _read(kv, ("a", "b"), slot=1) is first
    assert torch.equal(first["main"].cpu_indptrs.qo_indptr, qo)
    assert kv._static_plan_state(1, "main").total_tokens == 0


def test_a_plan_built_across_a_touch_of_its_rows_is_not_reused(kv, monkeypatch):
    """A pre-plan builds on the plan thread while the main thread can tear a
    request down: the entry keeps the epochs from before the views."""
    build = kv._sequence_views

    def touched(*args, **kwargs):
        kv._touch_plan_cache("b")
        return build(*args, **kwargs)

    monkeypatch.setattr(kv, "_sequence_views", touched)
    first = _read(kv, ("a", "b"))
    monkeypatch.setattr(kv, "_sequence_views", build)
    assert _read(kv, ("a", "b")) is not first


def test_an_eager_write_step_is_not_staged_and_write_target_addresses_it():
    """``KVStep.eager_write``: the step's K/V are written outside the graph,
    so its plan stays out of the slot's static buffers, and ``write_target``
    hands the layer the step's own addressing."""
    kv = _manager()
    kv.ingest_request("a", KVReqConfig())
    step = KVStep(segments=(Segment("a", "main", 6),), eager_write=True)
    bucket = BucketKey(graph_walk="prefill", bs=1, num_tokens=8)
    ctx = StepContext(
        request_ids=("a",), graph_walk="prefill", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=bucket),
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    assert (0, "main") not in kv._staged_outputs
    assert kv._current_plan_states["main"] is not kv._static_plan_state(0, "main")
    layer, pages, slots, n = kv.write_target(layer_idx=0, label="main")
    assert n == 6 and layer.data_ptr() == kv.kv_cache.tensor[0].data_ptr()
    stream = kv._streams["a"]["main"]
    assert pages[:n].tolist() == [stream.page_indices[i // PAGE] for i in range(6)]
    assert slots[:n].tolist() == [i % PAGE for i in range(6)]
    kv.commit(step, ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs pinned host memory")
def test_an_offload_and_reload_of_a_real_row_plans_again():
    kv = _manager(device="cuda", cpu_offload_pages=16)
    _write(kv, "a", 6)
    _write(kv, "b", 9)
    first = _read(kv, ("a", "b"))
    assert kv.offload("b") > 0
    assert kv.reload("b")
    again = _read(kv, ("a", "b"))
    assert again is not first and _lengths(again)[:2] == [6, 9]
