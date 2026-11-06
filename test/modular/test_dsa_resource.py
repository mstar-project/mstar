"""The DSA resource on CPU: its build over an MLA cache with index planes, the rows its declared
step plans from the KV plan (``plan``), and the plan the layers read (``current``, ``rows`` /
``pick`` / ``dense``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec, DSAStep
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.engine.resources.kv.config import AuxPlanes, KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import PagedIndptrs, SequenceView
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

KV = "kv"
PAGE, TOPK, HEADS, MAX_SEQ_LEN = 4, 4, 16, 64


D = 16  # index_head_dim


def _spec(topk=TOPK, **config) -> DSASpec:
    config.setdefault("index_head_dim", D)
    return DSASpec(resource_key="dsa", nodes={"LLM"},
                   config=DSAConfig(kv_cache=KV, topk=topk, **config))


def _kv_spec(index_planes: bool = True, width: int = D) -> KVSpec:
    return KVSpec(resource_key=KV, nodes={"LLM"}, config=PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=48, max_seq_len=MAX_SEQ_LEN,
        max_num_pages=32, page_size=PAGE, num_qo_heads=HEADS, layout=KVLayout.MLA,
        kv_lora_rank=32, qk_rope_head_dim=16,
        aux_planes={INDEX_PLANES: AuxPlanes(num_layers=2, width=width)} if index_planes else {}))


def _manager(world_size: int = 1, kpool: int = 1, topk: int = TOPK) -> DSAManager:
    """The resource as the engine builds it, from its spec and its cache's spec."""
    info = EngineResourceInfo(
        device=torch.device("cpu"), joint_comm_group=SimpleNamespace(world_size=world_size),
        dependencies={KV: _kv_spec(width=D if kpool == 1 else 3 * D)})
    return build_resource(_spec(topk=topk, kpool=kpool, sm_scale=0.1), info)


def _step(walk: str, pages, lengths, to_compute, lease=None, is_preplan=False) -> StepContext:
    """A step whose KV plan holds one view per request: request i has ``lengths[i]`` tokens
    after the step, its last ``to_compute[i]`` new."""
    views = [SequenceView(f"r{i}", "main", p, length=n, to_compute=c)
             for i, (p, n, c) in enumerate(zip(pages, lengths, to_compute, strict=True))]
    indptr = torch.tensor([0, *torch.tensor([len(p) for p in pages]).cumsum(0).tolist()])
    flat = torch.tensor([page for p in pages for page in p], dtype=torch.int32)
    plan = SimpleNamespace(views=views, cpu_indptrs=PagedIndptrs(
        qo_indptr=None, paged_kv_indptr=indptr, paged_kv_indices=flat,
        paged_kv_last_page_len=None))
    return StepContext(
        request_ids=tuple(f"r{i}" for i in range(len(lengths))), graph_walk=walk,
        slot=0 if lease is None else lease.slot, capture=False,
        plan_results={KV: {"main": plan}}, slot_lease=lease, is_preplan=is_preplan)


def _stage(dsa: DSAManager, walk: str, pages, lengths, to_compute, lease=None):
    """The step's plan for its one label, as the engine runs it (``DSAStep()``)."""
    return dsa.plan(DSAStep(), _step(walk, pages, lengths, to_compute, lease))["main"]


def _prefill(dsa: DSAManager, pages=([0], [5, 6, 7])):
    """A fresh 3-token prompt and a 4-token one past 5 cached tokens: 9 keys, past topk."""
    return _stage(dsa, "prefill", pages, lengths=[3, 9], to_compute=[3, 4])


# ---------------------------------------------------------------------------
# spec and build
# ---------------------------------------------------------------------------


def test_the_resource_plans_this_ranks_query_heads():
    assert _manager(world_size=2).num_heads == HEADS // 2


@pytest.mark.parametrize("kpool, width, head_dim", [
    (1, 3 * D, D),      # kpool 1: a row is one key
    (4, D, D),          # kpool 4: [key | gate | pool key]
    (4, 3 * D, None),   # the key width undeclared
])
def test_index_planes_of_another_width_are_refused_at_build(kpool, width, head_dim):
    """A mis-sized INDEX_PLANES row booted, then failed on the first prefill."""
    info = EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=None,
                              dependencies={KV: _kv_spec(width=width)})
    with pytest.raises(ValueError, match="wide"):
        build_resource(_spec(kpool=kpool, index_head_dim=head_dim), info)


def test_a_cache_without_index_planes_is_refused():
    info = EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=None,
                              dependencies={KV: _kv_spec(index_planes=False)})
    with pytest.raises(ValueError, match=INDEX_PLANES):
        build_resource(_spec(), info)


# ---------------------------------------------------------------------------
# plan: the declared step's rows, from the KV plan
# ---------------------------------------------------------------------------


def test_a_capture_past_one_sparse_plan_is_refused_before_it_runs():
    """A captured step plans its whole bucket as one sparse plan, which FlashInfer faults on
    past ``MAX_ROWS`` rows: the resource refuses such a bucket when the captures are sized."""
    from mstar.engine.resources.base import CGSlotSpec
    from mstar.engine.resources.dsa.sparse_mla import MAX_ROWS
    from mstar.engine.resources.step import BucketKey

    def specs(*tokens):
        return [CGSlotSpec(bucket=BucketKey("prefill", 1, n), slot=0, config=None) for n in tokens]

    dsa = _manager()
    dsa.build_accelerator_graph_buffers(specs(64, MAX_ROWS), max_bs=1, max_seq_len=MAX_ROWS)
    with pytest.raises(ValueError, match="sparse plan"):
        dsa.build_accelerator_graph_buffers(specs(64, 2 * MAX_ROWS), max_bs=1,
                                            max_seq_len=2 * MAX_ROWS)


def test_an_eager_prefill_plans_its_rows_and_pages():
    """Each request's new rows and its pages, flat, from the KV plan, as the layers read them:
    the plan of the step's ``DSAStep``, with nothing from preprocess."""
    dsa = _manager()
    ctx = _prefill(dsa)
    assert dsa.current() is ctx
    assert ctx.row_req.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert ctx.pos.tolist() == ctx.host_pos == [0, 1, 2, 5, 6, 7, 8]
    assert ctx.pages.dtype == torch.int32 and ctx.pages.tolist() == [0, 5, 6, 7]
    assert ctx.page_start.tolist() == ctx.host_page_start == [0, 1]
    assert ctx.spans == [(0, 3, 0), (3, 4, 1)]  # (first row, rows, request)
    assert ctx.max_pools == 9  # the step's longest context
    # a row sees 9 keys, past topk; prefill rows score in chunks of one request
    assert ctx.needs_selection and not ctx.decode_rows and ctx.sparse_plan is None
    assert ctx.attn_lens() == [1, 2, 3, 4, 4, 4, 4]
    # pooled keys score in pools: 9 keys are 4 complete pools of 2
    assert _prefill(_manager(kpool=2)).max_pools == 4
    # a step that declared no DSAStep has no plan to read
    with pytest.raises(KeyError, match="DSAStep"):
        _manager().current()


def test_an_eager_decode_scores_every_row_on_its_own():
    dsa = _manager()
    # r0 at position 6 (past topk), r1 at position 2
    ctx = _stage(dsa, "decode", pages=[[1, 2], [4]], lengths=[7, 3], to_compute=[1, 1])
    assert ctx.decode_rows and ctx.needs_selection and ctx.max_pools == 7
    # below topk every row attends all it sees: dense MLA is the same computation
    short = _stage(dsa, "decode", pages=[[1], [4]], lengths=[4, 3], to_compute=[1, 1])
    assert short.needs_selection is False


def test_a_pre_plan_is_promoted_not_redone():
    """A plan made a step ahead (the plan thread's) stays pending until the step runs, which
    takes it rather than planning again."""
    dsa = _manager()
    assert dsa.supports_preplan
    _stage(dsa, "decode", [[1, 2]], [7], [1])
    ahead = dsa.plan(DSAStep(), _step("decode", [[1, 2]], [8], [1], is_preplan=True))["main"]
    assert dsa.current().host_pos == [6]  # the live plan stays until the step runs
    assert _stage(dsa, "decode", [[3]], [2], [1]) is ahead and dsa.current().host_pos == [7]


def test_plans_under_one_capture_lease_fill_the_same_buffers():
    """A captured graph reads a slot's rows where the capture saw them: each later plan under
    that lease refills them in place, from the buffers its captures were sized with."""
    from mstar.engine.resources.base import CGSlotSpec

    dsa = _manager()
    bucket = BucketKey("decode", bs=2, num_tokens=2)
    dsa.build_accelerator_graph_buffers([CGSlotSpec(bucket=bucket, slot=1, config=None)], 2, 2)
    lease = SlotLease(slot=1, bucket=bucket)
    first = _stage(dsa, "decode", [[1, 2], [4]], [7, 3], [1, 1], lease)
    second = _stage(dsa, "decode", [[3, 5], [6]], [8, 2], [1, 1], lease)
    assert second.pos.data_ptr() == first.pos.data_ptr()
    assert second.pages.data_ptr() == first.pages.data_ptr()
    assert second.pos.tolist() == [7, 1] and second.pages[:3].tolist() == [3, 5, 6]
    # every row scores on its own over the serving window's keys
    assert second.host_pos is None and second.decode_rows and second.max_pools == MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# rows / pick / dense: passes over part of the step, or none of it
# ---------------------------------------------------------------------------


def test_rows_and_pick_are_passes_over_part_of_the_step():
    ctx = _prefill(_manager())
    ctx.last_selection = torch.zeros(7, TOPK, dtype=torch.int32)
    tail = ctx.rows(2, 6)  # r0's last row and r1's first three
    assert tail.pos.tolist() == [2, 5, 6, 7] and tail.row_req.tolist() == [0, 1, 1, 1]
    assert tail.host_pos == [2, 5, 6, 7]
    assert tail.spans == [(0, 1, 0), (1, 3, 1)]
    assert tail.max_pools == 8
    # its own selection, its own eager plan: dense attention would plan the whole step
    assert tail.needs_selection and tail.last_selection is None and tail.sparse_plan is None
    assert tail.pages is ctx.pages and tail.page_start is ctx.page_start
    last = ctx.pick([2, 6])  # each request's last row
    assert last.pos.tolist() == [2, 8] and last.row_req.tolist() == [0, 1]
    assert last.spans == [(0, 1, 0), (1, 1, 1)]


def test_rows_pick_and_dense_narrow_the_plan_for_a_block_and_restore_it():
    dsa = _manager()
    step = _prefill(dsa)
    with dsa.rows(3, 7):  # r1's rows, a pass of their own over the step's plan
        assert dsa.current().host_pos == [5, 6, 7, 8] and dsa.current().spans == [(0, 4, 1)]
        with dsa.dense():  # a dense block inside it
            assert dsa.current() is None
        assert dsa.current().host_pos == [5, 6, 7, 8]
    assert dsa.current() is step
    with dsa.pick([2, 6]):
        assert dsa.current().host_pos == [2, 8]
    assert dsa.current() is step
    # a block that fails still restores the step's plan
    with pytest.raises(RuntimeError, match="boom"), dsa.dense():
        raise RuntimeError("boom")
    assert dsa.current() is step


def test_a_node_with_dsa_double_buffers_its_captures_without_pre_plan(monkeypatch):
    """A replay reads its slot's sparse plan, which ``plan`` re-plans through FlashInfer's
    pinned staging: the node keeps two capture slots and fences their reuse even when no
    pre-plan asks for them."""
    from mstar.engine import engine as engine_mod

    monkeypatch.setattr(engine_mod.knobs, "pre_plan_enabled", lambda: False)
    mgmt = engine_mod.SubmoduleManagement(
        submodule=None, forward=None, forward_batched=None, joint_comm_group=None,
        resources={"dsa": _manager()})
    assert mgmt.num_slots == 2 and mgmt.needs_slot_fence
