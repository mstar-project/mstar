"""The DSA resource on CPU: its spec and build over the latent cache and the index keys, the
step's rows and page tables it stages from the KV plans (``step_inputs``), the forward's context
it builds from them (``context``) and the one it holds open for the layers (``use``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.dsa.config import DSAConfig, DSASpec
from mstar.engine.resources.dsa.kpool import KPoolDSA
from mstar.engine.resources.dsa.manager import DSAManager
from mstar.engine.resources.dsa.paged import PagedContext
from mstar.engine.resources.kv.config import KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import SequenceView
from mstar.engine.resources.step import StepContext

KV, INDEX = "kv", "kv_index"
# the index store's pages differ from the latent cache's, so each table reads its own plan
PAGE, INDEX_PAGE, TOPK, HEADS, MAX_SEQ_LEN = 4, 8, 4, 16, 64


def _spec(topk=TOPK, **config) -> DSASpec:
    return DSASpec(resource_key="dsa", nodes={"LLM"},
                   config=DSAConfig(kv_cache=KV, topk=topk, **config))


def _kv_spec(key: str, page_size: int, latent: int, rope: int) -> KVSpec:
    return KVSpec(resource_key=key, nodes={"LLM"}, config=PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=latent + rope, max_seq_len=MAX_SEQ_LEN,
        max_num_pages=32, page_size=page_size, num_qo_heads=HEADS, layout=KVLayout.MLA,
        kv_lora_rank=latent, qk_rope_head_dim=rope))


def _manager(world_size: int = 1) -> DSAManager:
    """The resource as the engine builds it, from its spec and its caches' specs."""
    info = EngineResourceInfo(
        device=torch.device("cpu"), joint_comm_group=SimpleNamespace(world_size=world_size),
        dependencies={KV: _kv_spec(KV, PAGE, 32, 16), INDEX: _kv_spec(INDEX, INDEX_PAGE, 16, 0)})
    return build_resource(_spec(index_cache=INDEX, sm_scale=0.1), info)


def _stage(dsa: DSAManager, walk: str, kv_pages, index_pages, lengths, to_compute,
           needs_selection: bool) -> dict:
    """``step_inputs`` of an eager step whose KV plans hold one view per request in each cache:
    request i has ``lengths[i]`` tokens after the step, its last ``to_compute[i]`` new."""
    def plan(pages):
        return {"main": SimpleNamespace(views=[
            SequenceView(f"r{i}", "main", p, length=n, to_compute=c)
            for i, (p, n, c) in enumerate(zip(pages, lengths, to_compute, strict=True))])}

    step_ctx = StepContext(
        request_ids=tuple(f"r{i}" for i in range(len(lengths))), graph_walk=walk, slot=0,
        capture=False, plan_results={KV: plan(kv_pages), INDEX: plan(index_pages)})
    positions = [p for n, c in zip(lengths, to_compute, strict=True) for p in range(n - c, n)]
    return dsa.step_inputs(step_ctx, positions, list(to_compute), needs_selection)


def _prefill(dsa: DSAManager, kv_pages=([0], [5, 6, 7]), index_pages=([2], [3, 4])) -> dict:
    """A fresh 3-token prompt and a 4-token one past 5 cached tokens: 9 keys, past topk."""
    return _stage(dsa, "prefill", kv_pages, index_pages, lengths=[3, 9], to_compute=[3, 4],
                  needs_selection=True)


# ---------------------------------------------------------------------------
# spec and build
# ---------------------------------------------------------------------------


def test_the_spec_builds_over_the_latent_cache_and_the_index_keys():
    spec = _spec(index_cache=INDEX)
    assert spec.depends_on() == {KV, INDEX}
    assert spec.resource_class is DSAManager
    # pooled keys (GLM-5.3's k-pool indexer) live in the latent cache's own index planes
    pooled = _spec(kpool=4)
    assert pooled.depends_on() == {KV}
    assert pooled.resource_class is KPoolDSA


def test_the_resource_reads_its_geometry_off_both_caches():
    dsa = _manager(world_size=2)
    assert isinstance(dsa, DSAManager)
    assert dsa.depends_on() == {KV, INDEX}
    assert (dsa.page_size, dsa.index_page_size, dsa.max_seq_len) == (PAGE, INDEX_PAGE, MAX_SEQ_LEN)
    assert dsa.num_heads == HEADS // 2  # this rank's query heads
    assert dsa.ctx is None  # no forward is running


# ---------------------------------------------------------------------------
# step_inputs: the step's rows, from the KV plans
# ---------------------------------------------------------------------------


def test_an_eager_prefill_stages_its_rows_and_both_page_tables():
    dsa = _manager()
    kv_pages, index_pages = [[0], [5, 6, 7]], [[2], [3, 4]]
    out = _prefill(dsa, kv_pages, index_pages)
    assert out["dsa_row_req"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert out["dsa_lens"].tolist() == [1, 2, 3, 6, 7, 8, 9]  # each row's keys: position + 1
    # one row per request, zero padded to the step's longest, from each cache's own plan
    assert out["dsa_kv_table"].dtype == out["dsa_index_table"].dtype == torch.int32
    assert out["dsa_kv_table"].tolist() == [[0, 0, 0], [5, 6, 7]]
    assert out["dsa_index_table"].tolist() == [[2, 0], [3, 4]]
    # the tables are snapshots: later plan growth must not alias
    kv_pages[0].append(99)
    index_pages[1].append(98)
    assert out["dsa_kv_table"].tolist() == [[0, 0, 0], [5, 6, 7]]
    assert out["dsa_index_table"].tolist() == [[2, 0], [3, 4]]
    meta = out["dsa_meta"]
    assert meta["host_lens"] == [1, 2, 3, 6, 7, 8, 9]
    assert meta["spans"] == [(0, 3, 0), (3, 4, 1)]  # (first row, rows, request)
    assert meta["width"] == 9  # the step's longest context
    assert (meta["page_size"], meta["topk"]) == (PAGE, TOPK)
    assert meta["needs_selection"] is True
    assert meta["decode_rows"] is False  # prefill rows score in chunks of one request
    assert meta["sparse_plan"] is None  # eager: the first layer to attend plans it


def test_an_eager_decode_scores_every_row_on_its_own():
    dsa = _manager()
    # r0 at position 6 (past topk), r1 at position 2
    for needs_selection in (True, False):  # an eager step keeps the caller's needs_selection
        out = _stage(dsa, "decode", kv_pages=[[1, 2], [4]], index_pages=[[3], [5]],
                     lengths=[7, 3], to_compute=[1, 1], needs_selection=needs_selection)
        assert out["dsa_row_req"].tolist() == [0, 1]
        assert out["dsa_lens"].tolist() == [7, 3]
        assert out["dsa_kv_table"].tolist() == [[1, 2], [4, 0]]
        assert out["dsa_index_table"].tolist() == [[3], [5]]
        meta = out["dsa_meta"]
        assert meta["spans"] == [(0, 1, 0), (1, 1, 1)]
        assert meta["width"] == 7
        assert meta["needs_selection"] is needs_selection
        assert meta["decode_rows"] is True
        assert meta["sparse_plan"] is None


# ---------------------------------------------------------------------------
# context / use: the forward's context, open while its layers run
# ---------------------------------------------------------------------------


def test_the_context_is_the_staged_step():
    out = _prefill(_manager())
    ctx = DSAManager.context(out)
    assert isinstance(ctx, PagedContext)
    assert ctx.row_req is out["dsa_row_req"] and ctx.lens is out["dsa_lens"]
    assert ctx.kv_table is out["dsa_kv_table"] and ctx.index_table is out["dsa_index_table"]
    assert ctx.host_lens == [1, 2, 3, 6, 7, 8, 9]
    assert ctx.spans == [(0, 3, 0), (3, 4, 1)]
    assert (ctx.width, ctx.page_size, ctx.topk) == (9, PAGE, TOPK)
    assert ctx.needs_selection and not ctx.decode_rows and ctx.sparse_plan is None
    assert ctx.attn_lens == [1, 2, 3, 4, 4, 4, 4]  # a row past topk attends its top topk
    assert ctx.last_selection is None and ctx.step_rows is None  # a fresh, whole step
    # without the resource's inputs (a flag-off step) there is no context
    assert DSAManager.context({}) is None
    assert DSAManager.context({k: v for k, v in out.items() if k != "dsa_meta"}) is None


def test_use_opens_a_context_for_the_block_and_restores_the_one_before():
    dsa = _manager()
    step = DSAManager.context(_prefill(dsa))
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
