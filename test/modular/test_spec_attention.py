"""Speculative decoding: verify attention. FA3's paged wrapper reads a verify
step's KV lengths off the resident tables (asynchronous mode) or a host plan
(synchronous); DSpark's hidden-last-row mask is its ``kv_extent``, a span short,
non-causal. FlashInfer serves the same mask on its host plan.

The page tables and lengths are checked on the CPU; the kernels against a
dense reference need CUDA, ``flash_attn_3`` and ``flashinfer`` (Hopper).
"""
from __future__ import annotations

import dataclasses
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.config import AttentionStep  # noqa: E402
from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper, fa3_paged_unavailable_reason  # noqa: E402
from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, hide_tail_views  # noqa: E402

PAGE = 4
CPU = torch.device("cpu")


def _views(lengths, span, pages_of):
    return [
        SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=n + span, to_compute=span)
        for i, n in enumerate(lengths)
    ]


def test_a_hidden_last_row_needs_a_non_causal_step():
    with pytest.raises(ValueError, match="causal=False"):
        AttentionStep(hide_last=1)
    AttentionStep(causal=False, hide_last=1, verify=True)


def test_hiding_the_tail_drops_its_keys_not_its_queries():
    views = _views([5, 7], 4, {0: [3, 4, 5], 1: [8, 9, 10]})
    hidden = hide_tail_views(views, 1, PAGE)
    assert [v.length for v in hidden] == [8, 10]
    assert [v.to_compute for v in hidden] == [4, 4]
    assert hidden[0].page_idxs == [3, 4] and hidden[1].page_idxs == [8, 9, 10]
    with pytest.raises(ValueError, match="hide_last"):
        hide_tail_views(views, 5, PAGE)


@pytest.mark.parametrize("hide", [0, 1])
def test_the_host_plan_stages_the_block_lengths(hide):
    span, lengths = 4, [5, 9, 2]
    pages_of = {0: [3, 4, 5], 1: [6, 7, 8, 11], 2: [12, 13]}
    indptrs = build_paged_indptrs(_views(lengths, span, pages_of), PAGE)
    wrapper = FA3PagedWrapper(page_size=PAGE, table_width=16, device=CPU, span=span, kv_extent=span - hide,
                              causal=hide == 0)
    wrapper.plan(**indptrs.to_kwargs_dict())
    assert wrapper._seqused.tolist() == [n + span - hide for n in lengths]
    assert wrapper._qo_indptr_buf.tolist() == [0, 4, 8, 12]
    for row, pages in pages_of.items():
        assert wrapper._page_table[row, : len(pages)].tolist() == pages
    assert wrapper.causal == (hide == 0)
    with pytest.raises(ValueError, match="rows of 4 queries"):
        bad = build_paged_indptrs(_views([3], 2, {0: [1, 2]}), PAGE)
        wrapper.plan(**bad.to_kwargs_dict())


def test_a_resident_step_reads_lengths_and_pages_off_the_tables():
    span, hide, width = 4, 1, 5
    stride, col = 1 + width, 1
    # three slots plus the trash row: [kv_len, block table...]
    store = torch.zeros(4 * stride, dtype=torch.long)
    table = store.view(4, stride)
    table[1, 0], table[1, 1:4] = 6, torch.tensor([7, 8, 9])
    table[2, 0], table[2, 1:3] = 3, torch.tensor([10, 11])
    wrapper = FA3PagedWrapper(
        page_size=PAGE, table_width=width, device=CPU, span=span, kv_extent=span - hide, causal=False,
        batch_size=3, accelerator_graph=True,
    )
    slots = torch.tensor([2, 1, 0])  # the last a padding row on the trash row
    kv_len = table[slots, 0]
    # main's in-graph gather: the table is exactly `width` columns from `col`
    wrapper.gather_resident(store, slots, kv_len, stride, col)
    assert wrapper._seqused.tolist() == [3 + span - hide, 6 + span - hide, span - hide]
    assert wrapper._page_table[0, :2].tolist() == [10, 11]
    assert wrapper._page_table[1, :3].tolist() == [7, 8, 9]
    assert wrapper._page_table[2].eq(0).all()
    assert wrapper._qo_indptr_buf.tolist() == [0, 4, 8, 12]


def test_fa3_is_refused_off_hopper_with_a_reason():
    reason = fa3_paged_unavailable_reason(CPU, 128, torch.bfloat16)
    if torch.cuda.is_available() and reason is None:
        pytest.skip("this machine runs paged FA3")
    assert isinstance(reason, str) and reason


# ---- the kernels (GPU) ----

def _dense(q, k_ctx, v_ctx, lengths, span, causal, hide):
    """Each row's block of queries over its context, explicitly masked."""
    out = []
    at = 0
    for i, n in enumerate(lengths):
        keys = n + span - hide
        qi = q[at: at + span].float()
        k, v = k_ctx[i][:keys].float(), v_ctx[i][:keys].float()
        heads = qi.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(heads, 1), v.repeat_interleave(heads, 1)
        scores = torch.einsum("qhd,khd->hqk", qi, k) / qi.shape[-1] ** 0.5
        if causal:
            # bottom-right aligned: query j sees keys up to n + j
            allowed = torch.arange(keys)[None, :] <= (n + torch.arange(span))[:, None]
            scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        at += span
    return torch.cat(out)


def _paged_case(lengths, span, heads=8, kv_heads=2, dim=128, page=16, device="cuda"):
    torch.manual_seed(0)
    pages_per = [-(-(n + span) // page) for n in lengths]
    total = sum(pages_per) + 1
    cache = torch.randn(total, 2, page, kv_heads, dim, device=device, dtype=torch.bfloat16)
    perm = torch.randperm(total - 1).tolist()
    pages_of, at = {}, 0
    for i, count in enumerate(pages_per):
        pages_of[i] = [p + 1 for p in perm[at: at + count]]
        at += count
    k_ctx = [cache[pages_of[i], 0].reshape(-1, kv_heads, dim) for i in range(len(lengths))]
    v_ctx = [cache[pages_of[i], 1].reshape(-1, kv_heads, dim) for i in range(len(lengths))]
    q = torch.randn(len(lengths) * span, heads, dim, device=device, dtype=torch.bfloat16)
    views = [
        SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=n + span, to_compute=span)
        for i, n in enumerate(lengths)
    ]
    return cache, q, k_ctx, v_ctx, views


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("hide", [0, 1])
@pytest.mark.parametrize("dim", [128, 256])
def test_a_captured_plan_stages_the_tables_an_eager_plan_builds(hide, dim):
    # rows of different page counts, so a transposed or offset write shows
    span, lengths = 4, [5, 9, 2, 30]
    pages_of = {0: [3, 4, 5], 1: [6, 7, 8, 11], 2: [12, 13], 3: [20, 21, 22, 23, 24, 25, 26, 27, 28]}
    indptrs = build_paged_indptrs(_views(lengths, span, pages_of), PAGE).to_kwargs_dict()
    device = torch.device("cuda", torch.cuda.current_device())
    common = dict(page_size=PAGE, table_width=16, device=device, span=span, kv_extent=span - hide, causal=hide == 0)
    eager = FA3PagedWrapper(**common)
    eager.plan(**indptrs)
    captured = FA3PagedWrapper(**common, batch_size=len(lengths), accelerator_graph=True)
    captured.plan(**indptrs)
    torch.cuda.synchronize()
    assert torch.equal(captured._page_table.cpu(), eager._page_table.cpu())
    assert captured._seqused.tolist() == eager._seqused.tolist() == [n + span - hide for n in lengths]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("causal,hide", [(True, 0), (False, 1)])
# 128: Qwen3 and DSpark; 256: Qwen3.5's full-attention layers and its MTP head
@pytest.mark.parametrize("dim", [128, 256])
def test_fa3_paged_attention_matches_a_dense_reference(causal, hide, dim):
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), dim, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    lengths, span = [37, 130, 5], 8
    cache, q, k_ctx, v_ctx, views = _paged_case(lengths, span, dim=dim)
    wrapper = FA3PagedWrapper(
        page_size=16, table_width=512 // 16, device=torch.device("cuda"), span=span, kv_extent=span - hide,
        causal=causal,
    )
    wrapper.plan(dtype=torch.bfloat16, **build_paged_indptrs(views, 16).to_kwargs_dict())
    out = wrapper.run(q, cache)
    torch.testing.assert_close(
        out.float(), _dense(q, k_ctx, v_ctx, lengths, span, causal, hide), rtol=2e-2, atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("heads, kv_heads, page, lengths, span, dim, hide", [
    (8, 2, 16, [37, 130, 5], 8, 128, 1),
    # DSpark on Qwen3-8B: 32 query and 8 KV heads of 128, pages of 128, k = 7
    (32, 8, 128, [37, 300, 5, 1021], 8, 128, 1),
    # Qwen3.5-9B's verify (and MTP) attention: 16 query and 4 KV heads of 256, causal, k = 3
    (16, 4, 16, [37, 300, 5, 1021], 4, 256, 0),
], ids=["toy", "qwen3_8b_dspark", "qwen3_5_9b_mtp"])
def test_fa3_and_flashinfer_agree_on_the_hidden_last_row(heads, kv_heads, page, lengths, span, dim, hide):
    """Each verify mask on FA3 (asynchronous) and FlashInfer (synchronous)."""
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), dim, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    flashinfer = pytest.importorskip("flashinfer")
    cache, q, k_ctx, v_ctx, views = _paged_case(lengths, span, heads=heads, kv_heads=kv_heads, dim=dim, page=page)
    fa3 = FA3PagedWrapper(
        page_size=page, table_width=2048 // page, device=torch.device("cuda"), span=span, kv_extent=span - hide,
        causal=not hide,
    )
    fa3.plan(dtype=torch.bfloat16, **build_paged_indptrs(views, page).to_kwargs_dict())
    hidden = build_paged_indptrs(hide_tail_views(views, hide, page), page)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    fi.plan(
        hidden.qo_indptr.cuda(), hidden.paged_kv_indptr.cuda(), hidden.paged_kv_indices.cuda(),
        hidden.paged_kv_last_page_len.cuda(), heads, kv_heads, dim, page, causal=not hide, q_data_type=torch.bfloat16,
    )
    a = fa3.run(q, cache).float()
    b = fi.run(q, (cache[:, 0], cache[:, 1])).float()
    torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(a, _dense(q, k_ctx, v_ctx, lengths, span, not hide, hide), rtol=2e-2, atol=2e-2)


def test_hidden_rows_are_a_verify_steps_mask_only():
    """Elsewhere the mask would plan from the host's (reserved) lengths."""
    with pytest.raises(ValueError, match="verify=True"):
        AttentionStep(causal=False, hide_last=1)


def _stub_manager(dtype=torch.bfloat16):
    from types import SimpleNamespace

    from mstar.engine.resources.kv.config import PagedKVConfig

    config = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=64, num_qo_heads=2, max_seq_len=64,
                           max_num_pages=40, page_size=PAGE)
    return SimpleNamespace(_verify_wrappers={}, _kv_config=config, _group=SimpleNamespace(page_size=PAGE),
                           _device=CPU, _dtype=dtype)


@pytest.mark.parametrize("causal,hide", [(True, 0), (False, 0), (False, 1)])
def test_the_manager_hands_fa3_the_steps_mask(causal, hide):
    """A non-causal verify step runs non-causal on FA3 as on FlashInfer; a hidden last row is a
    ``kv_extent`` that many keys short of the span."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    step = AttentionStep(causal=causal, hide_last=hide, verify=True)
    wrapper = FlashInferManager._verify_wrapper(_stub_manager(), step, num_rows=2, num_tokens=8, lease=None)
    assert wrapper.causal is causal and wrapper.kv_extent == 4 - hide
    # its query offsets are where the manager's select_last_hidden reads them
    assert hasattr(wrapper, "_qo_indptr_buf")


def test_the_page_table_is_the_pools_block_table_width():
    """Sized at construction as the resident tables are (`PagedKVConfig.block_table_width`): the
    context in pages, and no more than the pool's request pages."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    manager = _stub_manager()
    step = AttentionStep(verify=True)
    wrapper = FlashInferManager._verify_wrapper(manager, step, num_rows=2, num_tokens=8, lease=None)
    assert wrapper.width == manager._kv_config.block_table_width(PAGE) == 64 // PAGE
    small = dataclasses.replace(manager._kv_config, max_num_pages=6)
    assert small.block_table_width(PAGE) == small.request_pages == 5


@pytest.mark.parametrize("kv_extent", [5, 0], ids=["past_the_span", "no_keys"])
def test_a_kv_extent_outside_its_span_is_refused(kv_extent):
    with pytest.raises(ValueError, match="kv_extent"):
        FA3PagedWrapper(page_size=PAGE, table_width=4, device=CPU, span=4, kv_extent=kv_extent)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_verify_wrapper_takes_the_managers_dtype(dtype):
    """A resident (asynchronous) step never plans on the host,
    so the cast of q comes from construction, from the manager's dtype."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    step = AttentionStep(causal=False, hide_last=1, verify=True)
    wrapper = FlashInferManager._verify_wrapper(_stub_manager(dtype), step, num_rows=2, num_tokens=8, lease=None)
    assert wrapper.dtype is dtype


def test_a_verify_step_on_fa3_that_plans_two_labels_is_refused():
    """FA3 keeps one verify wrapper per (bucket, slot), whatever the label, so a second label's
    plan would overwrite the first's under the same key."""
    from types import SimpleNamespace

    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.kv.plan import KVPlanOutputs

    manager = FlashInferManager.__new__(FlashInferManager)
    manager._window_left, manager._preplanned, manager._group_idx = -1, False, 0
    manager._kv_cache_name, manager._verify_fa3 = "kv", True
    manager._current_plan_states, manager._preplan_states = {}, {}
    ctx = SimpleNamespace(
        resident=None, slot_lease=None, is_preplan=False, plan_phase="plan",
        plan_results={"kv": KVPlanOutputs({"main": object(), "uncond": object()})},
    )

    with pytest.raises(RuntimeError, match="plans one KV label"):
        manager.plan(AttentionStep(verify=True), ctx)
