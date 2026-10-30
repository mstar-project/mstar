"""Cross-attention's captured-graph wrapper is sized by the plan, not the batch.

A FlashInfer wrapper in CUDA-graph mode fixes its internal buffers at
construction from ``batch_size``, and that has to be the qo_indptr row count
the label actually plans. One row per request is the ordinary case, but a
query plan that combines labels (batched CFG) contributes several rows per
request, and sizing off ``bucket.bs`` there under-allocates: every later
``plan()`` overruns the buffers.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn import cross as cross_mod
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv import page_table as page_table_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.page_table import PageTable
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    HostLens,
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
)
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

CTX_KV = "ctx_kv"
Q_KV = "q_kv"
CONTEXT = "context"


class _RecordingWrapper:
    """Stands in for FlashInferPrefillWrapper; records how it was built."""

    built: list[dict] = []
    kind = "prefill"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.plans = 0
        type(self).built.append({**kwargs, "kind": self.kind})

    def plan(self, **kwargs):
        self.plans += 1
        self.last_plan = kwargs
        # a real CUDA-graph wrapper raises here when the row count exceeds the
        # batch_size it was built with
        rows = kwargs["qo_indptr"].shape[0] - 1
        batch_size = self.kwargs.get("batch_size")
        if batch_size is not None and rows > batch_size:
            raise RuntimeError(
                f"runtime batch size {rows} exceeds the wrapper's {batch_size}"
            )


class _RecordingDecodeWrapper(_RecordingWrapper):
    """Stands in for FlashInferDecodeWrapper (rows of one query token)."""

    kind = "decode"


@pytest.fixture(autouse=True)
def _stub_wrapper(monkeypatch):
    _RecordingWrapper.built = []
    monkeypatch.setattr(cross_mod, "FlashInferPrefillWrapper", _RecordingWrapper)
    monkeypatch.setattr(cross_mod, "FlashInferDecodeWrapper", _RecordingDecodeWrapper)
    # WorkspacePool allocates this many MB per (label, slot)
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")


def _manager() -> cross_mod.FlashInferCrossManager:
    return cross_mod.FlashInferCrossManager(
        kv_cache=CTX_KV,
        query_kv_cache=Q_KV,
        context_label=CONTEXT,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_config=PagedKVConfig(
            num_layers=2, num_kv_heads=4, head_dim=64, max_seq_len=512,
            max_num_pages=64, page_size=16,
        ),
    )


def _plan_output(views: list[SequenceView]) -> KVPlanOutput:
    qo = [0]
    for view in views:
        qo.append(qo[-1] + view.to_compute)
    return KVPlanOutput(
        cpu_indptrs=PagedIndptrs(
            qo_indptr=torch.tensor(qo, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0], dtype=torch.int32),
            paged_kv_indices=torch.tensor([], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([], dtype=torch.int32),
        ),
        views=views,
    )


def _ctx(rids: tuple[str, ...], query: dict[str, KVPlanOutput]) -> StepContext:
    """A leased step whose context streams are written and whose query side
    plans ``query``."""
    ctx = StepContext(
        request_ids=rids, graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(
            slot=0,
            bucket=BucketKey(
                graph_walk="decode", bs=len(rids), num_tokens=len(rids),
            ),
        ),
    )
    context_views = [
        SequenceView(
            request_id=rid, label=CONTEXT, page_idxs=[1 + i],
            length=8, to_compute=0,
        )
        for i, rid in enumerate(rids)
    ]
    ctx.plan_results = {
        CTX_KV: KVPlanOutputs({CONTEXT: _plan_output(context_views)}),
        Q_KV: KVPlanOutputs(query),
    }
    return ctx


def test_an_ordinary_label_is_sized_one_row_per_request():
    rids = ("r0", "r1")
    query = {"main": _plan_output([
        SequenceView(request_id=rid, label="main", page_idxs=[9], length=4,
                     to_compute=1)
        for rid in rids
    ])}

    _manager().plan(AttentionStep(causal=False), _ctx(rids, query))

    (built,) = _RecordingWrapper.built
    assert built["batch_size"] == 2, "one qo row per request"


def test_a_combined_query_plan_is_sized_by_its_rows_not_the_batch_size():
    """The regression: two guidance branches over two requests plan four qo
    rows, so a wrapper built at bucket.bs=2 overruns on the first plan()."""
    rids = ("r0", "r1")
    # label-major, the ordering a combined plan concatenates in
    views = [
        SequenceView(request_id=rid, label="cfg_batched", page_idxs=[9],
                     length=4, to_compute=1)
        for _branch in ("cond", "uncond") for rid in rids
    ]
    query = {"cfg_batched": _plan_output(views)}

    # plan() raising is the failure mode a real wrapper would show
    _manager().plan(AttentionStep(causal=False), _ctx(rids, query))

    (built,) = _RecordingWrapper.built
    assert built["batch_size"] == 4, (
        "two branches over two requests is four qo rows, not bucket.bs"
    )


def test_the_eager_wrapper_is_unsized():
    """No lease means no CUDA graph, so FlashInfer sizes per plan call."""
    rids = ("r0",)
    query = {"main": _plan_output([
        SequenceView(request_id="r0", label="main", page_idxs=[9], length=4,
                     to_compute=1)
    ])}
    ctx = _ctx(rids, query)
    ctx.slot_lease = None

    _manager().plan(AttentionStep(causal=False), ctx)

    (built,) = _RecordingWrapper.built
    assert "batch_size" not in built


def test_one_token_rows_plan_the_decode_kernel_and_longer_rows_the_prefill_kernel():
    """A decoder's cross attention queries one token per request every step;
    a prompt prefill queries several."""
    rids = ("r0", "r1")
    for to_compute, kind in ((1, "decode"), (3, "prefill")):
        _RecordingWrapper.built = []
        query = {"main": _plan_output([
            SequenceView(request_id=rid, label="main", page_idxs=[9], length=4,
                         to_compute=to_compute)
            for rid in rids
        ])}
        _manager().plan(AttentionStep(causal=False), _ctx(rids, query))
        (built,) = _RecordingWrapper.built
        assert built["kind"] == kind


def _query(spans):
    return {"main": _plan_output([
        SequenceView(request_id=f"r{i}", label="main", page_idxs=[9], length=4, to_compute=n)
        for i, n in enumerate(spans)
    ])}


@pytest.mark.parametrize("spans,kind", [
    ([1, 1], "decode"),
    # as many tokens as rows, but not one a row: the prefill kernel
    ([2, 0], "prefill"),
    ([3, 2], "prefill"),
])
def test_the_kernel_follows_every_rows_span(spans, kind):
    for leased in (True, False):
        _RecordingWrapper.built = []
        ctx = _ctx(("r0", "r1"), _query(spans))
        if not leased:
            ctx.slot_lease = None
        _manager().plan(AttentionStep(causal=False), ctx)
        (built,) = _RecordingWrapper.built
        assert built["kind"] == kind


def test_force_prefill_takes_the_prefill_kernel():
    ctx = _ctx(("r0", "r1"), _query([1, 1]))
    ctx.force_prefill = True
    _manager().plan(AttentionStep(causal=False), ctx)
    (built,) = _RecordingWrapper.built
    assert built["kind"] == "prefill"


def test_an_unchanged_step_is_not_planned_again():
    manager = _manager()
    ctx = _ctx(("r0", "r1"), _query([1, 1]))
    manager.plan(AttentionStep(causal=False), ctx)
    wrapper = manager._current_plan_states["main"]
    # the same context plan object over the same rows: nothing rebuilt
    manager.plan(AttentionStep(causal=False), ctx)
    # a new context plan over the same pages: FlashInfer's plan skipped
    manager.plan(AttentionStep(causal=False), _ctx(("r0", "r1"), _query([1, 1])))
    assert manager._current_plan_states["main"] is wrapper and wrapper.plans == 1
    # other pages plan again
    moved = _ctx(("r0", "r1"), _query([1, 1]))
    moved.plan_results[CTX_KV][CONTEXT].views[1].page_idxs[0] = 7
    manager.plan(AttentionStep(causal=False), moved)
    assert wrapper.plans == 2


def test_a_captured_key_keeps_its_kind():
    manager = _manager()
    manager.plan(AttentionStep(causal=False), _ctx(("r0", "r1"), _query([1, 1])))
    with pytest.raises(AssertionError, match="captured as decode"):
        manager.plan(AttentionStep(causal=False), _ctx(("r0", "r1"), _query([2, 0])))


def test_the_plan_hands_the_wrapper_host_kv_lens():
    """A captured decode wrapper calls FlashInfer's planner directly only when
    the plan carries host lengths; they are FlashInfer's ``get_seq_lens`` of
    the context rows, and a padding row's empty context is one sink token."""
    manager = _manager()
    ctx = _ctx(("r0", "r1", "r2"), _query([1, 1, 1]))
    ctx.plan_results[CTX_KV][CONTEXT] = _plan_output([
        SequenceView(request_id="r0", label=CONTEXT, page_idxs=[1, 2, 9], length=20, to_compute=0),
        SequenceView(request_id="r1", label=CONTEXT, page_idxs=[3], length=16, to_compute=0),
        SequenceView(request_id="r2", label=CONTEXT, page_idxs=[], length=0, to_compute=0),
    ])
    manager.plan(AttentionStep(causal=False), ctx)
    plan = manager._current_plan_states["main"].last_plan
    assert isinstance(plan["kv_lens"], HostLens)
    assert plan["kv_lens"].tolist() == [20, 16, 1]
    # only the pages the written length covers
    assert plan["paged_kv_indices"].tolist() == [1, 2, 3, SINK_PAGE]
    assert plan["paged_kv_indptr"].tolist() == [0, 2, 3, 4]
    assert plan["paged_kv_last_page_len"].tolist() == [4, 16, 1]


def _unbuilt():
    raise AssertionError("the plan built views it had no need for")


def _columns_plan(rows, *, views: bool = False) -> KVPlanOutput:
    """The context label's plan as ``KVManager._plan_label`` builds it, off
    ``PageTable.plan_columns``. ``rows``: (request id, held pages, written
    length); a padding row holds nothing and has length 0. With ``views``
    the output has no columns and the cross manager must read the views."""
    table = PageTable()
    offs, held, lens = [], [], []
    for _, pages, length in rows:
        stream = table.new()
        stream.extend(pages)
        offs.append(stream.off)
        held.append(stream.n)
        lens.append(length)
    cols = table.plan_columns(offs, held, lens, [0] * len(rows), 16)
    bounds = cols.kv_indptr.tolist()
    built = [
        SequenceView(request_id=rid, label=CONTEXT, page_idxs=cols.indices[bounds[i]:bounds[i + 1]].tolist(),
                     length=length, to_compute=0)
        for i, (rid, _, length) in enumerate(rows)
    ]
    if views:
        return KVPlanOutput(cpu_indptrs=None, views=built)
    # cpu_indptrs stands for the pinned ring, which a later plan may have rewritten
    return KVPlanOutput(
        cpu_indptrs=None, rows=[(rid, CONTEXT, 0) for rid, _, _ in rows],
        view_builder=_unbuilt, columns=cols,
    )


def _rows_query(rids, spans) -> dict[str, KVPlanOutput]:
    qo = [0]
    for n in spans:
        qo.append(qo[-1] + n)
    return {"main": KVPlanOutput(
        cpu_indptrs=PagedIndptrs(
            qo_indptr=torch.tensor(qo, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0], dtype=torch.int32),
            paged_kv_indices=torch.tensor([], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([], dtype=torch.int32),
        ),
        rows=[(rid, "main", n) for rid, n in zip(rids, spans, strict=True)],
        view_builder=_unbuilt,
    )}


def _planned(context: KVPlanOutput, rids, spans) -> dict[str, list[int]]:
    ctx = _ctx(tuple(rids), _rows_query(rids, spans))
    ctx.plan_results[CTX_KV][CONTEXT] = context
    manager = _manager()
    manager.plan(AttentionStep(causal=False), ctx)
    plan = manager._current_plan_states["main"].last_plan
    assert isinstance(plan["kv_lens"], HostLens)
    return {k: plan[k].tolist() for k in (
        "qo_indptr", "paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len", "kv_lens",
    )}


# r0 holds a page more than its length covers; pad0 and pad1 are a capture's
# padding rows, one of them between real rows
_CONTEXT_ROWS = [
    ("r0", [1, 2, 9], 20), ("pad0", [], 0), ("r1", [3], 16), ("r2", [4, 5], 17), ("pad1", [], 0),
]


@pytest.fixture(params=["numpy", "rust"])
def plan_columns_backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(page_table_mod, "_rust_plan_columns", None)
    elif page_table_mod._rust_plan_columns is None:
        pytest.skip("the mstar_rust extension is not built")


@pytest.mark.parametrize("spans", [[1] * 5, [3, 1, 2, 1, 1]])
def test_query_rows_in_context_order_take_the_context_plans_arrays(plan_columns_backend, spans):
    """Same arrays as planning off the views, a padding row reading one sink
    token, without building a view on either side."""
    rids = [rid for rid, _, _ in _CONTEXT_ROWS]
    fast = _planned(_columns_plan(_CONTEXT_ROWS), rids, spans)
    assert fast == _planned(_columns_plan(_CONTEXT_ROWS, views=True), rids, spans)
    assert fast["paged_kv_indices"] == [1, 2, SINK_PAGE, 3, 4, 5, SINK_PAGE]
    assert fast["kv_lens"] == [20, 1, 16, 17, 1]


def test_query_rows_in_another_order_plan_off_the_views(plan_columns_backend):
    context = _columns_plan(_CONTEXT_ROWS)
    built = []

    def build():
        built.append(True)
        return _columns_plan(_CONTEXT_ROWS, views=True).views

    context._view_builder = build
    rids = ["r2", "r0", "pad1", "r1", "pad0"]
    planned = _planned(context, rids, [1] * 5)
    assert built
    assert planned["paged_kv_indices"] == [4, 5, 1, 2, SINK_PAGE, 3, SINK_PAGE]
    assert planned["kv_lens"] == [17, 20, 1, 16, 1]
