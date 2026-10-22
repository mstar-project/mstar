"""The DSA resource on CPU: its spec and build over the MLA cache whose pages hold the index keys,
the step's rows and pages it stages from the KV plan (``step_inputs``), the forward's context it
builds from them (``context``), the rows of it a pass takes (``rows`` / ``pick``) and the one it
holds open for the layers (``use``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.engine.resources.dsa.paged import DSAContext
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


def test_the_spec_builds_one_resource_over_the_mla_cache():
    for kpool in (1, 4):  # a key per token, or GLM-5.3's pooled keys: the same resource
        spec = _spec(kpool=kpool)
        assert spec.depends_on() == {KV}
        assert spec.resource_class is DSAManager


def test_the_resource_reads_its_geometry_off_the_cache():
    dsa = _manager(world_size=2)
    assert dsa.depends_on() == {KV}
    assert (dsa.page_size, dsa.kv_rows, dsa.kv_lora_rank, dsa.kpe) == (PAGE, MAX_SEQ_LEN, 32, 16)
    assert dsa.num_heads == HEADS // 2  # this rank's query heads
    assert dsa.ctx is None  # no forward is running


def test_a_cache_without_index_planes_is_refused():
    info = EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=None,
                              dependencies={KV: _kv_spec(index_planes=False)})
    with pytest.raises(ValueError, match=INDEX_PLANES):
        build_resource(_spec(), info)


# ---------------------------------------------------------------------------
# step_inputs: the step's rows, from the KV plan
# ---------------------------------------------------------------------------


def test_an_eager_prefill_stages_its_rows_and_pages():
    dsa = _manager()
    out = _prefill(dsa)
    assert out["dsa_row_req"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert out["dsa_pos"].tolist() == [0, 1, 2, 5, 6, 7, 8]
    # the step's requests' pages, flat, and where each request's begin
    assert out["dsa_pages"].dtype == torch.int32
    assert out["dsa_pages"].tolist() == [0, 5, 6, 7]
    assert out["dsa_page_start"].tolist() == [0, 1]
    meta = out["dsa_meta"]
    assert meta["host_pos"] == [0, 1, 2, 5, 6, 7, 8]
    assert meta["spans"] == [(0, 3, 0), (3, 4, 1)]  # (first row, rows, request)
    assert meta["host_page_start"] == [0, 1]
    assert meta["max_pools"] == 9  # the step's longest context
    assert meta["needs_selection"] is True  # a row sees 9 keys, past topk
    assert meta["decode_rows"] is False  # prefill rows score in chunks of one request
    assert meta["sparse_plan"] is None  # eager: the first layer to attend plans it


def test_an_eager_decode_scores_every_row_on_its_own():
    dsa = _manager()
    # r0 at position 6 (past topk), r1 at position 2
    out = _stage(dsa, "decode", pages=[[1, 2], [4]], lengths=[7, 3], to_compute=[1, 1])
    assert out["dsa_row_req"].tolist() == [0, 1]
    assert out["dsa_pos"].tolist() == [6, 2]
    assert out["dsa_pages"].tolist() == [1, 2, 4]
    meta = out["dsa_meta"]
    assert meta["spans"] == [(0, 1, 0), (1, 1, 1)]
    assert meta["max_pools"] == 7
    assert meta["needs_selection"] is True
    assert meta["decode_rows"] is True
    # below topk every row attends all it sees: dense MLA is the same computation
    short = _stage(dsa, "decode", pages=[[1], [4]], lengths=[4, 3], to_compute=[1, 1])
    assert short["dsa_meta"]["needs_selection"] is False


def test_pooled_keys_score_in_pools():
    out = _prefill(_manager(kpool=2, topk=4))
    assert out["dsa_meta"]["max_pools"] == 4  # 9 keys: 4 complete pools of 2


# ---------------------------------------------------------------------------
# context / rows / pick / use: the forward's context, open while its layers run
# ---------------------------------------------------------------------------


def test_the_context_is_the_staged_step():
    dsa = _manager()
    out = _prefill(dsa)
    ctx = dsa.context(out)
    assert isinstance(ctx, DSAContext)
    assert ctx.pos is out["dsa_pos"] and ctx.row_req is out["dsa_row_req"]
    assert ctx.pages is out["dsa_pages"] and ctx.page_start is out["dsa_page_start"]
    assert ctx.host_pos == [0, 1, 2, 5, 6, 7, 8]
    assert ctx.spans == [(0, 3, 0), (3, 4, 1)]
    assert (ctx.max_pools, ctx.page_size, ctx.topk, ctx.kpool) == (9, PAGE, TOPK, 1)
    assert ctx.needs_selection and not ctx.decode_rows and ctx.sparse_plan is None
    assert ctx.width == TOPK
    assert ctx.attn_lens() == [1, 2, 3, 4, 4, 4, 4]  # a row past topk attends its top topk
    assert ctx.last_selection is None  # a fresh step
    # without the resource's inputs (a flag-off step) there is no context
    assert dsa.context({}) is None
    assert dsa.context({k: v for k, v in out.items() if k != "dsa_meta"}) is None


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
