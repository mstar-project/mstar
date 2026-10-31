"""The DSA resource on CPU: its build over an MLA cache with index planes, the rows it stages
from the KV plan (``step_inputs``), and the context the layers read (``context``, ``rows`` /
``pick``, ``use``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.engine.resources.kv.config import AuxPlanes, KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import PagedIndptrs, SequenceView
from mstar.engine.resources.step import StepContext

KV = "kv"
PAGE, TOPK, HEADS, MAX_SEQ_LEN = 4, 4, 16, 64


def _spec(topk=TOPK, **config) -> DSASpec:
    return DSASpec(resource_key="dsa", nodes={"LLM"},
                   config=DSAConfig(kv_cache=KV, topk=topk, **config))


def _kv_spec(index_planes: bool = True) -> KVSpec:
    return KVSpec(resource_key=KV, nodes={"LLM"}, config=PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=48, max_seq_len=MAX_SEQ_LEN,
        max_num_pages=32, page_size=PAGE, num_qo_heads=HEADS, layout=KVLayout.MLA,
        kv_lora_rank=32, qk_rope_head_dim=16,
        aux_planes={INDEX_PLANES: AuxPlanes(num_layers=2, width=16)} if index_planes else {}))


def _manager(world_size: int = 1, kpool: int = 1, topk: int = TOPK) -> DSAManager:
    """The resource as the engine builds it, from its spec and its cache's spec."""
    info = EngineResourceInfo(
        device=torch.device("cpu"), joint_comm_group=SimpleNamespace(world_size=world_size),
        dependencies={KV: _kv_spec()})
    return build_resource(_spec(topk=topk, kpool=kpool, sm_scale=0.1), info)


def _stage(dsa: DSAManager, walk: str, pages, lengths, to_compute) -> dict:
    """``step_inputs`` of an eager step whose KV plan holds one view per request: request i
    has ``lengths[i]`` tokens after the step, its last ``to_compute[i]`` new."""
    views = [SequenceView(f"r{i}", "main", p, length=n, to_compute=c)
             for i, (p, n, c) in enumerate(zip(pages, lengths, to_compute, strict=True))]
    indptr = torch.tensor([0, *torch.tensor([len(p) for p in pages]).cumsum(0).tolist()])
    flat = torch.tensor([page for p in pages for page in p], dtype=torch.int32)
    plan = SimpleNamespace(views=views, cpu_indptrs=PagedIndptrs(
        qo_indptr=None, paged_kv_indptr=indptr, paged_kv_indices=flat,
        paged_kv_last_page_len=None))
    step_ctx = StepContext(
        request_ids=tuple(f"r{i}" for i in range(len(lengths))), graph_walk=walk, slot=0,
        capture=False, plan_results={KV: {"main": plan}})
    return dsa.step_inputs(step_ctx, captured=False)


def _prefill(dsa: DSAManager, pages=([0], [5, 6, 7])) -> dict:
    """A fresh 3-token prompt and a 4-token one past 5 cached tokens: 9 keys, past topk."""
    return _stage(dsa, "prefill", pages, lengths=[3, 9], to_compute=[3, 4])


# ---------------------------------------------------------------------------
# spec and build
# ---------------------------------------------------------------------------


def test_the_resource_plans_this_ranks_query_heads():
    assert _manager(world_size=2).num_heads == HEADS // 2


def test_a_cache_without_index_planes_is_refused():
    info = EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=None,
                              dependencies={KV: _kv_spec(index_planes=False)})
    with pytest.raises(ValueError, match=INDEX_PLANES):
        build_resource(_spec(), info)


# ---------------------------------------------------------------------------
# step_inputs: the step's rows, from the KV plan
# ---------------------------------------------------------------------------


def test_an_eager_prefill_stages_its_rows_and_pages():
    """Each request's new rows and its pages, flat, from the KV plan, as the layers read them."""
    dsa = _manager()
    ctx = dsa.context(_prefill(dsa))
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
    assert dsa.context(_prefill(_manager(kpool=2))).max_pools == 4
    # a step without the resource's inputs (flag off) has no context
    assert dsa.context({}) is None


def test_an_eager_decode_scores_every_row_on_its_own():
    dsa = _manager()
    # r0 at position 6 (past topk), r1 at position 2
    meta = _stage(dsa, "decode", pages=[[1, 2], [4]], lengths=[7, 3], to_compute=[1, 1])["dsa_meta"]
    assert meta["decode_rows"] and meta["needs_selection"] and meta["max_pools"] == 7
    # below topk every row attends all it sees: dense MLA is the same computation
    short = _stage(dsa, "decode", pages=[[1], [4]], lengths=[4, 3], to_compute=[1, 1])
    assert short["dsa_meta"]["needs_selection"] is False


# ---------------------------------------------------------------------------
# context / rows / pick / use: the forward's context, open while its layers run
# ---------------------------------------------------------------------------


def test_rows_and_pick_are_passes_over_part_of_the_step():
    ctx = _manager().context(_prefill(_manager()))
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


def test_use_opens_a_context_for_the_block_and_restores_the_one_before():
    dsa = _manager()
    step = dsa.context(_prefill(dsa))
    tail = step.rows(3, 7)  # r1's rows, a pass of their own over the step's plan
    assert dsa.ctx is None
    with dsa.use(step):
        assert dsa.ctx is step
        with dsa.use(tail):
            assert dsa.ctx is tail
        assert dsa.ctx is step
        with dsa.use(None):  # a dense block inside the forward
            assert dsa.ctx is None
        assert dsa.ctx is step
    assert dsa.ctx is None
    # a forward that fails still closes its context
    with pytest.raises(RuntimeError, match="boom"), dsa.use(step):
        raise RuntimeError("boom")
    assert dsa.ctx is None
