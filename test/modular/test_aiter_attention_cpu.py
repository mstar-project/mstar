"""CPU checks of what the AITER attention managers stage for their kernels.

The kernels only run on ROCm (``test/rocm/test_attention_aiter.py``), but the
plans are host work and are what a replay reads. These cover the block table
rows and tail padding, per-row KV lengths, the row counts and host ints a
captured bucket must keep, and preplan.
"""

import numpy as np
import pytest
import torch

from mstar.engine.resources.attn import aiter as rocm_attn
from mstar.engine.resources.attn.aiter import (
    AiterCrossManager,
    AiterPagedAttentionManager,
    kernel_rows,
    per_query_rows,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged import aiter as rocm_ragged
from mstar.engine.resources.attn.ragged.aiter import AiterRaggedCrossManager, AiterRaggedManager
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    KVPlanOutput,
    KVPlanOutputs,
    SequenceView,
    build_paged_indptrs,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

CPU = torch.device("cpu")
PAGE = 16
POOL = 32


@pytest.fixture(autouse=True)
def kernels_importable(monkeypatch):
    monkeypatch.setattr(rocm_attn, "aiter_paged_unavailable_reason", lambda: None)
    monkeypatch.setattr(rocm_ragged, "aiter_varlen_unavailable_reason", lambda: None)


def kv_config(**kw) -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=64, num_qo_heads=8,
        max_seq_len=512, max_num_pages=POOL, page_size=PAGE, **kw,
    )


def manager() -> AiterPagedAttentionManager:
    return AiterPagedAttentionManager("kv", CPU, torch.bfloat16, kv_config())


def view(rid, pages, length, to_compute) -> SequenceView:
    return SequenceView(request_id=rid, label="main", page_idxs=pages, length=length, to_compute=to_compute)


def context(views, lease=None, preplan=False, slot=0) -> StepContext:
    ctx = StepContext(
        request_ids=[v.request_id for v in views], graph_walk="w", slot=slot, capture=False,
        is_preplan=preplan, slot_lease=lease,
    )
    ctx.plan_results = {"kv": KVPlanOutputs(
        {"main": KVPlanOutput(cpu_indptrs=build_paged_indptrs(views, PAGE), views=views)},
    )}
    return ctx


def lease(bs, tokens, slot=0) -> SlotLease:
    return SlotLease(slot=slot, bucket=BucketKey(graph_walk="w", bs=bs, num_tokens=tokens))


PAD_COLS = -(-128 // PAGE)  # the tail tile the kernel may read past a row's pages


def test_an_eager_step_stages_its_own_layout():
    m = manager()
    views = [view("a", [3, 5], 20, 20), view("b", [7, 9, 11], 40, 1), view("c", [2], 1, 1)]
    m.plan(AttentionStep(), context(views))
    plan = m._current_plans["main"]
    assert plan.cu_seqlens_q.tolist() == [0, 20, 21, 22]
    assert plan.seqused_k.tolist() == [20, 40, 1]
    # an eager step uses the batch's maxima and a table only as wide as it needs
    assert (plan.max_seqlen_q, plan.max_seqlen_k, plan.num_tokens) == (20, 40, 22)
    assert plan.block_table.shape == (3, 3 + PAD_COLS)
    table = plan.block_table.tolist()
    assert table[0][:2 + PAD_COLS] == [3, 5] + [SINK_PAGE] * PAD_COLS
    assert table[1][:3 + PAD_COLS] == [7, 9, 11] + [SINK_PAGE] * PAD_COLS
    assert table[2][:1 + PAD_COLS] == [2] + [SINK_PAGE] * PAD_COLS


def test_a_captured_decode_step_pins_the_kernel_variant_and_keeps_its_buffers():
    m = manager()
    first = [view("a", [3], 5, 1), view("pad", [SINK_PAGE], 1, 1)]
    m.plan(AttentionStep(), context(first, lease(2, 2)))
    plan = m._current_plans["main"]
    # one query per row, and the cache's max context instead of the batch's
    assert (plan.max_seqlen_q, plan.max_seqlen_k) == (1, 512)
    # a row can hold every page of the pool
    assert plan.block_table.shape == (2, POOL + PAD_COLS)
    assert plan.seqused_k.tolist() == [5, 1]
    address = plan.block_table.data_ptr()

    second = [view("a", [3, 4], 17, 1), view("b", [6], 9, 1)]
    m.plan(AttentionStep(), context(second, lease(2, 2)))
    plan = m._current_plans["main"]
    assert plan.block_table.data_ptr() == address, "a replay reads the captured address"
    assert plan.seqused_k.tolist() == [17, 9]
    assert plan.block_table[0, :2].tolist() == [3, 4]


def test_a_decode_bucket_refuses_a_row_with_several_queries():
    m = manager()
    m.plan(AttentionStep(), context([view("a", [3], 5, 1), view("b", [4], 3, 1)], lease(2, 2)))
    with pytest.raises(RuntimeError, match="more than one query token"):
        m.plan(AttentionStep(), context([view("a", [3], 6, 2), view("b", [4], 3, 0)], lease(2, 2)))


def test_a_packed_bucket_pins_its_token_count():
    m = manager()
    views = [view("a", [3, 4], 30, 30), view("b", [5], 10, 2)]
    m.plan(AttentionStep(), context(views, lease(2, 64)))
    plan = m._current_plans["main"]
    assert (plan.max_seqlen_q, plan.max_seqlen_k, plan.num_tokens) == (64, 512, 32)


def test_a_packed_bucket_with_a_token_per_row_replays_a_mixed_step():
    """A packed capture can have as many rows as tokens and then replay a
    combined walk's step, decode rows beside a prompt. The engine marks a
    packed lease (`force_prefill`), so the capture pins the packed variant."""
    m = manager()
    captured = [view(rid, [i + 3], 5, 1) for i, rid in enumerate("abcd")]
    ctx = context(captured, lease(4, 4))
    ctx.force_prefill = True
    m.plan(AttentionStep(), ctx)
    assert m._current_plans["main"].max_seqlen_q == 4

    mixed = [view("a", [3], 6, 1), view("p", [7], 3, 3), view("pad0", [], 0, 0), view("pad1", [], 0, 0)]
    ctx = context(mixed, lease(4, 4))
    ctx.force_prefill = True
    m.plan(AttentionStep(), ctx)
    plan = m._current_plans["main"]
    assert plan.cu_seqlens_q.tolist() == [0, 1, 4, 4, 4]
    assert (plan.max_seqlen_q, plan.num_tokens) == (4, 4)
    assert m.select_last_hidden(torch.arange(4.0)[:, None])[:2, 0].tolist() == [0.0, 3.0]


def test_a_split_walk_plans_as_one_launch():
    """AITER runs a combined walk's step in one launch, so the decode/prompt
    split FlashInfer takes for a split walk (`AttentionConfig.splits`) is not
    needed: the mixed bucket stages one plan, like any packed bucket."""
    m = manager()
    views = [view("a", [3], 6, 1), view("b", [4], 9, 1), view("p", [5, 6], 20, 12)]
    mixed = SlotLease(slot=0, bucket=BucketKey(graph_walk="mixed", bs=4, num_tokens=16))
    ctx = context(views + [view("pad", [], 0, 0)], mixed)
    ctx.force_prefill = True
    m.plan(AttentionStep(), ctx)
    plan = m._current_plans["main"]
    assert plan.cu_seqlens_q.tolist() == [0, 1, 2, 14, 14]
    assert (plan.max_seqlen_q, plan.num_tokens) == (16, 14)


def test_a_row_with_nothing_to_attend_points_at_the_sink():
    m = manager()
    # a packed bucket's padding row, with no tokens and no pages
    views = [view("a", [3], 7, 7), view("pad", [], 0, 0)]
    m.plan(AttentionStep(), context(views, lease(2, 16)))
    plan = m._current_plans["main"]
    assert plan.seqused_k.tolist() == [7, 1]
    assert plan.block_table[1, :PAD_COLS].tolist() == [SINK_PAGE] * PAD_COLS


def test_non_causal_single_query_rows_stay_on_unified_attention():
    m = manager()
    m.plan(AttentionStep(causal=False), context([view("a", [3], 5, 1)]))
    assert m._current_plans["main"].cu_seqlens_q.tolist() == [0, 1]


def test_non_causal_rows_of_several_queries_need_flashinfer(monkeypatch):
    monkeypatch.setattr(
        rocm_attn, "flashinfer_rocm_unavailable_reason", lambda: "ImportError: no flashinfer.rocm",
    )
    m = manager()
    with pytest.raises(RuntimeError, match="AMD's FlashInfer build"):
        m.plan(AttentionStep(causal=False), context([view("a", [3], 5, 5)]))


def test_preplan_is_staged_on_its_own_slot_and_promoted():
    m = manager()
    live = [view("a", [3], 5, 1)]
    m.plan(AttentionStep(), context(live, lease(1, 1, slot=0)))
    current = m._current_plans["main"]
    ahead = [view("a", [3], 6, 1)]
    m.plan(AttentionStep(), context(ahead, lease(1, 1, slot=1), preplan=True))
    assert m._current_plans["main"] is current, "a pre-plan leaves the live plan alone"
    assert current.seqused_k.tolist() == [5]
    m.plan(AttentionStep(), context(ahead, lease(1, 1, slot=1)))
    promoted = m._current_plans["main"]
    assert promoted.seqused_k.tolist() == [6]
    assert promoted.block_table.data_ptr() != current.block_table.data_ptr()

    m.plan(AttentionStep(), context(live, lease(1, 1, slot=0), preplan=True))
    m.clear_preplan()
    assert not m._preplanned and m._preplan_plans == {}


def test_per_query_rows_give_every_query_its_requests_pages():
    views = [view("a", [3, 4], 20, 2), view("b", [5], 3, 0), view("c", [6], 9, 1)]
    rows = kernel_rows(build_paged_indptrs(views, PAGE), PAGE)
    expanded = per_query_rows(rows, num_rows=5)
    assert expanded.qo_indptr.tolist() == [0, 1, 2, 3, 4, 5]
    # a's two queries, c's one, two padding rows
    assert expanded.kv_lens.tolist() == [20, 20, 9, 1, 1]
    assert np.diff(expanded.kv_indptr).tolist() == [2, 2, 1, 0, 0]
    assert expanded.kv_indices.tolist() == [3, 4, 3, 4, 6]
    with pytest.raises(RuntimeError, match="do not fit"):
        per_query_rows(rows, num_rows=2)


def test_cross_attention_runs_one_kernel_row_per_query():
    m = AiterCrossManager(
        kv_cache="ctx", query_kv_cache="kv", context_label="context",
        device=CPU, dtype=torch.bfloat16, kv_config=kv_config(),
    )
    query = [view("a", [3], 4, 2), view("b", [4], 1, 1)]
    ctx = context(query)
    ctx_views = [
        SequenceView(request_id="a", label="context", page_idxs=[10, 11], length=20, to_compute=0),
        SequenceView(request_id="b", label="context", page_idxs=[12], length=5, to_compute=0),
    ]
    ctx.plan_results["ctx"] = KVPlanOutputs({
        "context": KVPlanOutput(cpu_indptrs=build_paged_indptrs(ctx_views, PAGE), views=ctx_views),
    })
    m.plan(AttentionStep(causal=False), ctx)
    plan = m._current_plans["main"]
    assert plan.cu_seqlens_q.tolist() == [0, 1, 2, 3]
    assert plan.seqused_k.tolist() == [20, 20, 5]
    assert plan.max_seqlen_q == 1
    assert [row[:n] for row, n in zip(plan.block_table.tolist(), (2, 2, 1), strict=True)] == [[10, 11], [10, 11], [12]]


def ragged_config() -> RaggedAttentionConfig:
    return RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=72, max_segments_per_request=2)


def test_a_captured_ragged_step_pads_to_the_buckets_segments():
    m = AiterRaggedManager(device=CPU, dtype=torch.bfloat16, config=ragged_config())
    step = AttentionStep(segments=(Segment("a", "img", 30), Segment("b", "img", 10)), causal=False)
    ctx = StepContext(request_ids=["a", "b"], graph_walk="w", slot=0, capture=False, slot_lease=lease(2, 64))
    m.plan(step, ctx)
    plan = m._current_plans["img"]
    # bs 2 x 2 segments per request, the last two empty
    assert plan.cu_seqlens_q.tolist() == [0, 30, 40, 40, 40]
    assert plan.cu_seqlens_k.tolist() == plan.cu_seqlens_q.tolist()
    assert (plan.max_seqlen_q, plan.num_segments, plan.num_tokens) == (64, 2, 40)
    assert plan.out.shape == (64, 4, 72)
    assert m.num_segments("img") == 2


def test_an_eager_ragged_step_is_exact():
    m = AiterRaggedManager(device=CPU, dtype=torch.bfloat16, config=ragged_config())
    step = AttentionStep(segments=(Segment("a", "img", 30), Segment("b", "img", 10)), causal=False)
    m.plan(step, StepContext(request_ids=["a", "b"], graph_walk="w", slot=0, capture=False))
    plan = m._current_plans["img"]
    assert plan.cu_seqlens_q.tolist() == [0, 30, 40]
    assert (plan.max_seqlen_q, plan.out) == (30, None)


def test_ragged_cross_pairs_plan_their_own_key_layout():
    m = AiterRaggedCrossManager(device=CPU, dtype=torch.bfloat16, config=ragged_config())
    step = RaggedCrossAttentionStep(
        segments=(Segment("a", "vid", 8), Segment("a", "txt", 3), Segment("b", "vid", 4), Segment("b", "txt", 5)),
        pairs=(("vid", "txt"),),
    )
    m.plan(step, StepContext(request_ids=["a", "b"], graph_walk="w", slot=0, capture=False))
    plan = m._current_plans[cross_label("vid", "txt")]
    assert plan.cu_seqlens_q.tolist() == [0, 8, 12]
    assert plan.cu_seqlens_k.tolist() == [0, 3, 8]
    assert (plan.max_seqlen_q, plan.max_seqlen_k, plan.causal) == (8, 5, False)
