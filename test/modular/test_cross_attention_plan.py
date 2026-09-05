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
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.plan import (
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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).built.append(kwargs)

    def plan(self, **kwargs):
        # a real CUDA-graph wrapper raises here when the row count exceeds the
        # batch_size it was built with
        rows = kwargs["qo_indptr"].shape[0] - 1
        batch_size = self.kwargs.get("batch_size")
        if batch_size is not None and rows > batch_size:
            raise RuntimeError(
                f"runtime batch size {rows} exceeds the wrapper's {batch_size}"
            )


@pytest.fixture(autouse=True)
def _stub_wrapper(monkeypatch):
    _RecordingWrapper.built = []
    monkeypatch.setattr(cross_mod, "FlashInferPrefillWrapper", _RecordingWrapper)
    # WorkspacePool allocates this many MB per (label, slot)
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")


def _manager() -> cross_mod.FlashInferCrossManager:
    return cross_mod.FlashInferCrossManager(
        kv_cache=CTX_KV,
        query_kv_cache=Q_KV,
        context_label=CONTEXT,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_config=KVConfig(
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
