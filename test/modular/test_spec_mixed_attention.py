"""Speculative decoding: a mixed step's attention on FA3. A mixed prompt-plus-verify step's rows, its verify rows of
``k + 1`` queries and its prompt rows, run on FA3's ragged wrapper (`FA3RaggedWrapper`) where the verify walk runs
on FA3's paged one; each verify row's last keys are hidden in the KV plan's arrays (`hide_tail_views`), as on
FlashInfer's host plan.

The plans are checked on the CPU; the kernel against FlashInfer and a dense reference, eager and captured, needs
CUDA, ``flash_attn_3`` and ``flashinfer`` (Hopper).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.config import AttentionStep  # noqa: E402
from mstar.engine.resources.attn.fa3_paged import FA3RaggedWrapper, fa3_paged_unavailable_reason  # noqa: E402
from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, hide_tail_views  # noqa: E402

PAGE = 4
CPU = torch.device("cpu")
BLOCK = 4


def _mixed_views(verify, prompts, pages_of, padding=0):
    """Verify rows of ``BLOCK`` queries over ``verify`` stored tokens each, then prompt rows of ``prompts``
    (cached, new) tokens each, then ``padding`` empty rows."""
    views = [
        SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=n + BLOCK, to_compute=BLOCK)
        for i, n in enumerate(verify)
    ]
    for j, (cached, new) in enumerate(prompts):
        i = len(verify) + j
        views.append(SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=cached + new,
                                  to_compute=new))
    views += [SequenceView(request_id=-1, label="main", page_idxs=[], length=0, to_compute=0)] * padding
    return views


def _hidden(views, n_verify, hide, page=PAGE):
    return hide_tail_views(views, hide, page, [True] * n_verify) if hide else views


@pytest.mark.parametrize("hide", [0, 1])
def test_the_host_plan_stages_each_rows_keys_and_queries(hide):
    """A verify row attends to its stored tokens and block, less the hidden rows; a prompt row to its cached and
    new tokens; a padding row to nothing. The query offsets are the rows' spans."""
    verify, prompts = [5, 9], [(0, 3), (6, 5)]
    pages_of = {0: [3, 4, 5], 1: [6, 7, 8, 11], 2: [12], 3: [13, 14, 15]}
    views = _hidden(_mixed_views(verify, prompts, pages_of, padding=1), len(verify), hide)
    wrapper = FA3RaggedWrapper(page_size=PAGE, table_width=16, device=CPU, causal=hide == 0)
    wrapper.plan(**build_paged_indptrs(views, PAGE).to_kwargs_dict())
    assert wrapper._seqused.tolist() == [5 + BLOCK - hide, 9 + BLOCK - hide, 3, 11, 0]
    assert wrapper._qo_indptr_buf.tolist() == [0, 4, 8, 11, 16, 16]
    assert wrapper._max_q == 5
    for row, view in enumerate(views):
        assert wrapper._page_table[row, : len(view.page_idxs)].tolist() == list(view.page_idxs)
    assert wrapper.causal == (hide == 0)


def test_a_captured_plan_fills_the_buckets_buffers_and_refuses_another_shape():
    """Under a captured graph the page table, lengths and query offsets are the bucket's static buffers, and the
    kernel's query bound stays the bucket's tokens; a plan of other rows, or more tokens, is refused."""
    verify, prompts = [5], [(2, 3)]
    pages_of = {0: [3, 4, 5], 1: [6, 7]}
    views = _mixed_views(verify, prompts, pages_of, padding=2)
    wrapper = FA3RaggedWrapper(page_size=PAGE, table_width=8, device=CPU, batch_size=4, max_tokens=12,
                               accelerator_graph=True)
    buffers = (wrapper._page_table, wrapper._seqused, wrapper._qo_indptr_buf)
    wrapper.plan(**build_paged_indptrs(views, PAGE).to_kwargs_dict())
    assert (wrapper._page_table, wrapper._seqused, wrapper._qo_indptr_buf) == buffers
    assert wrapper._seqused.tolist() == [9, 5, 0, 0]
    assert wrapper._qo_indptr_buf.tolist() == [0, 4, 7, 7, 7]
    assert wrapper._max_q == 12
    with pytest.raises(ValueError, match="captured for 4 rows"):
        wrapper.plan(**build_paged_indptrs(views[:3], PAGE).to_kwargs_dict())
    long = _mixed_views(verify, [(0, 9)], {0: [3, 4, 5], 1: [6, 7, 8]}, padding=2)
    with pytest.raises(ValueError, match="captured for 12 queries"):
        wrapper.plan(**build_paged_indptrs(long, PAGE).to_kwargs_dict())


def test_eager_and_captured_steps_run_the_kernel_with_the_same_pinned_schedule(monkeypatch):
    """FA3's split and head-packing heuristics read ``max_seqlen_q``, a capture's bucket tokens but an eager step's
    longest row, so both pass the wrapper's pinned values and reduce each row in the same order."""
    import mstar.engine.resources.attn.fa3_paged as fa3_mod

    seen = []

    def kernel(q, k, v, **kwargs):
        seen.append({key: kwargs[key] for key in ("num_splits", "pack_gqa", "max_seqlen_q")})
        return torch.zeros_like(q)

    monkeypatch.setattr(fa3_mod, "flash_attn_with_kvcache", kernel)
    views = _mixed_views([5], [(2, 3)], {0: [3, 4, 5], 1: [6, 7]})
    cache = torch.zeros(8, 2, PAGE, 1, 8)
    eager = FA3RaggedWrapper(page_size=PAGE, table_width=8, device=CPU)
    captured = FA3RaggedWrapper(page_size=PAGE, table_width=8, device=CPU, batch_size=2, max_tokens=12,
                                accelerator_graph=True)
    for wrapper, tokens in ((eager, 7), (captured, 12)):
        wrapper.plan(**build_paged_indptrs(views, PAGE).to_kwargs_dict())
        wrapper.run(torch.zeros(tokens, 2, 8), cache)
    pinned = (FA3RaggedWrapper.NUM_SPLITS, FA3RaggedWrapper.PACK_GQA)
    assert [(s["num_splits"], s["pack_gqa"]) for s in seen] == [pinned, pinned]
    assert [s["max_seqlen_q"] for s in seen] == [4, 12]


def _stub_manager(verify_fa3: bool):
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.kv.config import PagedKVConfig

    manager = FlashInferManager.__new__(FlashInferManager)
    manager._window_left, manager._preplanned, manager._group_idx = -1, False, 0
    manager._kv_cache_name, manager._verify_fa3, manager._verify_walk = "kv", verify_fa3, "decode"
    manager._current_plan_states, manager._preplan_states, manager._step_qo = {}, {}, {}
    manager._verify_wrappers, manager._dtype, manager._device = {}, torch.bfloat16, CPU
    manager._mixed_cg, manager._verify_block = {}, BLOCK
    manager._kv_config = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=64, num_qo_heads=2, max_seq_len=64,
                                       max_num_pages=40, page_size=PAGE)
    manager._group = SimpleNamespace(page_size=PAGE)
    manager._shard = manager._placement = None
    manager._split_exchange = None
    return manager


def _plan_mixed(manager, step):
    from mstar.engine.resources.kv.plan import KVPlanOutputs

    views = _mixed_views([5], [(2, 3)], {0: [3, 4, 5], 1: [6, 7]})
    kv_out = SimpleNamespace(views=views, cpu_indptrs=build_paged_indptrs(views, PAGE))
    ctx = SimpleNamespace(
        resident=None, slot_lease=None, is_preplan=False, plan_phase="plan", graph_walk="mixed", slot=0,
        request_ids=(0, 1), walk_of={0: "decode", 1: "prefill"}.get, force_prefill=False,
        plan_results={"kv": KVPlanOutputs({"main": kv_out})},
    )
    manager.plan(step, ctx)
    return manager._current_plan_states["main"]


@pytest.mark.parametrize("causal,hide", [(True, 0), (False, 1)], ids=["target", "dspark"])
def test_a_mixed_step_plans_on_fa3s_ragged_wrapper_with_its_verify_rows_keys_hidden(causal, hide):
    """With FA3 bound for the verify walk, a mixed step's attention plans its rows on the ragged wrapper, a
    drafter's non-causal, last-row-hidden binding too; its verify row's last keys come off its arrays."""
    wrapper = _plan_mixed(_stub_manager(True), AttentionStep(verify=True, causal=causal, hide_last=hide))
    assert isinstance(wrapper, FA3RaggedWrapper) and wrapper.causal is causal
    # what a resident gather of a captured mixed step's verify rows attends past each row's length
    assert wrapper.verify_extent == BLOCK - hide
    assert wrapper._seqused.tolist() == [5 + BLOCK - hide, 5]
    assert wrapper._qo_indptr_buf.tolist() == [0, 4, 7]


def test_without_fa3_a_mixed_step_stays_on_flashinfer(monkeypatch):
    """FlashInfer's own plan where FA3 doesn't run (no flash_attn_3, or ``verify_backend: flashinfer``)."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    planned = SimpleNamespace(plan=lambda **kwargs: None)
    monkeypatch.setattr(FlashInferManager, "_eager_wrapper", lambda self, *args, **kwargs: planned, raising=False)
    wrapper = _plan_mixed(_stub_manager(False), AttentionStep(verify=True))
    assert wrapper is planned


# ── on CUDA ──


def _dense(q, keys_of, values_of, rows, causal):
    """Each row's queries over its keys, explicitly masked: ``rows`` (queries, keys) a row, bottom-right
    aligned when ``causal``."""
    out, at = [], 0
    for (n_q, n_k), k, v in zip(rows, keys_of, values_of, strict=True):
        qi = q[at: at + n_q].float()
        k, v = k[:n_k].float(), v[:n_k].float()
        heads = qi.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(heads, 1), v.repeat_interleave(heads, 1)
        scores = torch.einsum("qhd,khd->hqk", qi, k) / qi.shape[-1] ** 0.5
        if causal:
            allowed = torch.arange(n_k)[None, :] <= (n_k - n_q + torch.arange(n_q))[:, None]
            scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        at += n_q
    return torch.cat(out)


def _cuda_case(verify, prompts, heads, kv_heads, dim, page, hide):
    """A paged cache, the mixed step's views with the verify rows' last keys hidden, its queries, and each row's
    (queries, keys) for the dense reference."""
    torch.manual_seed(0)
    totals = [n + BLOCK for n in verify] + [c + n for c, n in prompts]
    pages_per = [-(-t // page) for t in totals]
    total = sum(pages_per) + 1
    cache = torch.randn(total, 2, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    perm = torch.randperm(total - 1).tolist()
    pages_of, at = {}, 0
    for i, count in enumerate(pages_per):
        pages_of[i] = [p + 1 for p in perm[at: at + count]]
        at += count
    views = _hidden(_mixed_views(verify, prompts, pages_of), len(verify), hide, page)
    keys = [cache[pages_of[i], 0].reshape(-1, kv_heads, dim) for i in range(len(totals))]
    values = [cache[pages_of[i], 1].reshape(-1, kv_heads, dim) for i in range(len(totals))]
    rows = [(BLOCK, n + BLOCK - hide) for n in verify] + [(n, c + n) for c, n in prompts]
    q = torch.randn(sum(n for n, _ in rows), heads, dim, device="cuda", dtype=torch.bfloat16)
    return cache, q, views, keys, values, rows


CASES = [
    # heads, kv heads, page, verify rows' stored tokens, prompt rows' (cached, new), hide
    pytest.param(8, 2, 16, [37, 300, 5], [(0, 17), (64, 130)], 0, id="target"),
    pytest.param(32, 8, 16, [37, 300, 5, 1021], [(96, 33)], 0, id="qwen3_8b"),
    pytest.param(32, 8, 16, [37, 300, 5, 1021], [(96, 33), (0, 1)], 1, id="qwen3_8b_dspark"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("heads,kv_heads,page,verify,prompts,hide", CASES)
def test_fa3s_ragged_rows_agree_with_flashinfer_and_a_dense_reference(heads, kv_heads, page, verify, prompts, hide):
    """A mixed step's mask on FA3's ragged wrapper and on FlashInfer's prefill wrapper, from the same hidden
    arrays: causal for the target, non-causal with each verify row's last key hidden for DSpark's binding."""
    dim = 128
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), dim, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    flashinfer = pytest.importorskip("flashinfer")
    cache, q, views, keys, values, rows = _cuda_case(verify, prompts, heads, kv_heads, dim, page, hide)
    causal = not hide
    indptrs = build_paged_indptrs(views, page)
    fa3 = FA3RaggedWrapper(page_size=page, table_width=2048 // page, device=torch.device("cuda"), causal=causal)
    fa3.plan(dtype=torch.bfloat16, **indptrs.to_kwargs_dict())
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    fi.plan(
        indptrs.qo_indptr.cuda(), indptrs.paged_kv_indptr.cuda(), indptrs.paged_kv_indices.cuda(),
        indptrs.paged_kv_last_page_len.cuda(), heads, kv_heads, dim, page, causal=causal, q_data_type=torch.bfloat16,
    )
    a = fa3.run(q, cache).float()
    b = fi.run(q, (cache[:, 0], cache[:, 1])).float()
    torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(a, _dense(q, keys, values, rows, causal), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("heads,kv_heads,page,verify,prompts,hide", CASES)
def test_a_captured_ragged_replay_is_the_eager_plans_and_zeroes_its_padding(
    heads, kv_heads, page, verify, prompts, hide,
):
    """Captured once at a bucket of more rows and tokens than the step's, replayed after a host plan: its rows are
    the eager wrapper's, bit for bit, and the queries past them (a replay's padding) come out zero. A second plan
    of the same bucket, other rows, replays to that plan's eager result."""
    dim = 128
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), dim, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    device = torch.device("cuda", torch.cuda.current_device())
    causal = not hide
    cache, q, views, _, _, _ = _cuda_case(verify, prompts, heads, kv_heads, dim, page, hide)
    pad_rows, pad_tokens = 2, 64
    bs, max_tokens = len(views) + pad_rows, q.shape[0] + pad_tokens
    padding = [SequenceView(request_id=-1, label="main", page_idxs=[], length=0, to_compute=0)] * pad_rows
    captured = FA3RaggedWrapper(page_size=page, table_width=2048 // page, device=device, batch_size=bs,
                                max_tokens=max_tokens, accelerator_graph=True, causal=causal)
    q_static = torch.zeros(max_tokens, heads, dim, device=device, dtype=torch.bfloat16)
    captured.plan(dtype=torch.bfloat16, **build_paged_indptrs(views + padding, page).to_kwargs_dict())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        captured.run(q_static, cache)  # warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = captured.run(q_static, cache)

    for step_views, step_q in ((views, q), (views[1:], q[BLOCK:])):
        captured.plan(dtype=torch.bfloat16, **build_paged_indptrs(
            step_views + padding + padding[: len(views) - len(step_views)], page).to_kwargs_dict())
        q_static.zero_()
        q_static[: step_q.shape[0]] = step_q
        graph.replay()
        torch.cuda.synchronize()
        eager = FA3RaggedWrapper(page_size=page, table_width=2048 // page, device=device, causal=causal)
        eager.plan(dtype=torch.bfloat16, **build_paged_indptrs(step_views, page).to_kwargs_dict())
        want = eager.run(step_q, cache)
        got = out[: step_q.shape[0]]
        rows = (got != want).flatten(1).any(1).nonzero().flatten().tolist()
        diff = (got.float() - want.float()).abs().max().item()
        assert torch.equal(got, want), f"max |diff| {diff} at queries {rows[:16]}{'...' if len(rows) > 16 else ''}"
        assert out[step_q.shape[0]:].eq(0).all()


@pytest.mark.parametrize("page,warns", [(128, True), (64, True), (32, False), (16, False)])
def test_mixed_steps_on_fa3_over_large_pages_warn_once(caplog, page, warns):
    """FA3's paged kernel reads a long prompt row's last page past its length on pages over 32 tokens (the CUDA
    test below documents it): the manager says so once, when it first builds a mixed step's wrapper."""
    import logging

    manager = _stub_manager(True)
    manager._group = SimpleNamespace(page_size=page)
    with caplog.at_level(logging.WARNING, logger="mstar.engine.resources.attn.flashinfer"):
        manager._mixed_wrapper("main", 2, None, 0, causal=True)
        manager._mixed_wrapper("main", 2, None, 1, causal=True)
    said = [r for r in caplog.records if "past its length" in r.getMessage()]
    assert len(said) == (1 if warns else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("page", [128, 32, 16])
@pytest.mark.parametrize("prompt", [25, 64, 127], ids=lambda n: f"prompt{n}")
def test_fa3_reads_a_long_rows_last_page_past_its_length_only_on_large_pages(page, prompt):
    """What `FA3_TAIL_SAFE_PAGE_SIZE` rests on, for this FA3 wheel: with NaN in every slot past each row's length, a
    mixed step whose prompt row has 64 or more queries comes out NaN on 128-token pages, and finite on pages of 32
    or fewer, or with a shorter prompt row. A wheel that changes either side fails here."""
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), 128, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    from mstar.engine.resources.attn.fa3_paged import FA3_TAIL_SAFE_PAGE_SIZE

    heads, kv_heads, dim = 32, 8, 128
    torch.manual_seed(0)
    verify = [37, 300, 5, 1021]
    totals = [n + BLOCK for n in verify] + [700 + prompt]
    pages_per = [-(-t // page) for t in totals]
    total = sum(pages_per) + 1
    cache = torch.randn(total, 2, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    perm = torch.randperm(total - 1).tolist()
    pages_of, at = {}, 0
    for i, (t, count) in enumerate(zip(totals, pages_per, strict=True)):
        pages_of[i] = [p + 1 for p in perm[at: at + count]]
        at += count
        if t % page:
            cache[pages_of[i][-1], :, t % page:] = float("nan")
    views = _mixed_views(verify, [(700, prompt)], pages_of)
    q = torch.randn(len(verify) * BLOCK + prompt, heads, dim, device="cuda", dtype=torch.bfloat16)
    wrapper = FA3RaggedWrapper(page_size=page, table_width=4096 // page, device=torch.device("cuda"), causal=True)
    wrapper.plan(dtype=torch.bfloat16, **build_paged_indptrs(views, page).to_kwargs_dict())
    finite = bool(torch.isfinite(wrapper.run(q, cache)).all())
    assert finite == (page <= FA3_TAIL_SAFE_PAGE_SIZE or prompt < 64)
