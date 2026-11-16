"""Speculative decoding's FA3 wrappers at the pool's page sizes: 16 and 32 as well as main's default 128.

A smaller page leaves a prefix hit fewer tokens to recompute, but more of each row's keys sit in a last page it
only partly fills. Every slot a row doesn't own holds NaN here: the sink page (a padding row's) and the slots past
each row's length in its last page. So a kernel that reads one key too many, or a padding row that leaks into a
real one, fails loudly. Real rows must be finite and match a dense fp32 reference, eager and replayed, on the
verify wrapper (spans 4 and 8, a hidden last key for DSpark) and on the ragged mixed wrapper (verify rows beside
prompt rows whose new tokens are no multiple of the page). A replay's padding queries come out zero.

Needs CUDA and FA3's paged kernel (Hopper).
"""
from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.fa3_paged import (  # noqa: E402
    FA3PagedWrapper,
    FA3RaggedWrapper,
    fa3_paged_unavailable_reason,
)
from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, hide_tail_views  # noqa: E402

HEADS, KV_HEADS, DIM = 32, 8, 128
SINK = 0
PAGES = [16, 32, 128]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _need_fa3():
    reason = fa3_paged_unavailable_reason(torch.device("cuda"), DIM, torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)


def _cache(totals, page):
    """A pool whose sink page holds NaN, and each row's pages (none the sink). Past a row's length its last page holds
    NaN too at 16 and 32; at 128 it holds finite keys, as a pool's stale slots do: FA3's ragged kernel at 128-token
    pages loads the last page's whole tile and masks the scores past the length, so a NaN there would reach the row
    through the zeroed probabilities, though a finite stale key contributes exactly nothing."""
    torch.manual_seed(0)
    pages_per = [-(-t // page) for t in totals]
    total = sum(pages_per) + 1
    cache = torch.randn(total, 2, page, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    cache[SINK] = float("nan")
    perm = torch.randperm(total - 1).tolist()
    pages_of, at = {}, 0
    for i, (t, count) in enumerate(zip(totals, pages_per, strict=True)):
        pages_of[i] = [p + 1 for p in perm[at: at + count]]
        at += count
        if t % page and page < 128:
            cache[pages_of[i][-1], :, t % page:] = float("nan")
    return cache, pages_of


def _keys(cache, pages):
    return cache[pages, 0].reshape(-1, KV_HEADS, DIM), cache[pages, 1].reshape(-1, KV_HEADS, DIM)


def _dense(q, rows, kv, causal):
    """Each row's (queries, keys) over its own keys, bottom-right causal when ``causal``."""
    out, at = [], 0
    for (n_q, n_k), (k, v) in zip(rows, kv, strict=True):
        qi = q[at: at + n_q].float()
        k, v = k[:n_k].float(), v[:n_k].float()
        k, v = k.repeat_interleave(HEADS // KV_HEADS, 1), v.repeat_interleave(HEADS // KV_HEADS, 1)
        scores = torch.einsum("qhd,khd->hqk", qi, k) / DIM ** 0.5
        if causal:
            allowed = torch.arange(n_k)[None, :] <= (n_k - n_q + torch.arange(n_q))[:, None]
            scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        at += n_q
    return torch.cat(out)


def _check(got, want, what):
    assert torch.isfinite(got).all(), f"{what}: a real row read a NaN slot"
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2, msg=lambda m: f"{what}: {m}")


# stored lengths no multiple of 16, 32 or 128, short and past several pages
VERIFY = [37, 300, 5, 1021, 127, 129]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("span,hide", [(4, 0), (8, 1)], ids=["eagle3", "dspark"])
def test_verify_rows_over_small_pages_read_only_their_own_keys(page, span, hide):
    _need_fa3()
    causal = hide == 0
    totals = [n + span for n in VERIFY]
    cache, pages_of = _cache(totals, page)
    views = [SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=t, to_compute=span)
             for i, t in enumerate(totals)]
    rows = [(span, t - hide) for t in totals]
    kv = [_keys(cache, pages_of[i]) for i in range(len(totals))]
    q = torch.randn(len(totals) * span, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    common = dict(page_size=page, table_width=4096 // page, device=torch.device("cuda"), span=span,
                  kv_extent=span - hide, causal=causal)
    eager = FA3PagedWrapper(**common)
    eager.plan(dtype=torch.bfloat16, **build_paged_indptrs(views, page).to_kwargs_dict())
    want = _dense(q, rows, kv, causal)
    _check(eager.run(q, cache), want, f"eager, page {page}")

    # captured with a padding row on the sink (as a replay's padding reads it), replayed at the step's rows
    pad = SequenceView(request_id=-1, label="main", page_idxs=[SINK], length=span, to_compute=span)
    bs = len(views) + 1
    captured = FA3PagedWrapper(**common, batch_size=bs, accelerator_graph=True)
    q_static = torch.zeros(bs * span, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    captured.plan(dtype=torch.bfloat16, **build_paged_indptrs(views + [pad], page).to_kwargs_dict())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        captured.run(q_static, cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = captured.run(q_static, cache)
    q_static[: q.shape[0]] = q
    graph.replay()
    torch.cuda.synchronize()
    _check(out[: q.shape[0]], want, f"replayed, page {page}")


# prompt rows' (cached, new): tails of 1 to page-1 new tokens past a cached prefix, and a prompt of its own
PROMPTS = [(1024, 1), (1088, 15), (96, 33), (0, 127), (2048, 9)]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("block,hide", [(4, 0), (8, 1)], ids=["eagle3", "dspark"])
def test_mixed_rows_over_small_pages_read_only_their_own_keys_and_pad_with_zeros(page, block, hide):
    _need_fa3()
    causal = hide == 0
    verify = VERIFY[:4]
    totals = [n + block for n in verify] + [c + n for c, n in PROMPTS]
    cache, pages_of = _cache(totals, page)
    views = [SequenceView(request_id=i, label="main", page_idxs=pages_of[i], length=totals[i], to_compute=block)
             for i in range(len(verify))]
    views += [SequenceView(request_id=len(verify) + j, label="main", page_idxs=pages_of[len(verify) + j],
                           length=c + n, to_compute=n) for j, (c, n) in enumerate(PROMPTS)]
    if hide:
        views = hide_tail_views(views[: len(verify)], hide, page) + views[len(verify):]
    rows = [(block, n + block - hide) for n in verify] + [(n, c + n) for c, n in PROMPTS]
    kv = [_keys(cache, pages_of[i]) for i in range(len(totals))]
    n_q = sum(r[0] for r in rows)
    q = torch.randn(n_q, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    want = _dense(q, rows, kv, causal)
    device = torch.device("cuda", torch.cuda.current_device())
    eager = FA3RaggedWrapper(page_size=page, table_width=4096 // page, device=device, causal=causal)
    eager.plan(dtype=torch.bfloat16, **build_paged_indptrs(views, page).to_kwargs_dict())
    _check(eager.run(q, cache), want, f"eager, page {page}")

    pad_rows, pad_tokens = 2, 37
    bs, max_tokens = len(views) + pad_rows, n_q + pad_tokens
    padding = [SequenceView(request_id=-1, label="main", page_idxs=[], length=0, to_compute=0)] * pad_rows
    captured = FA3RaggedWrapper(page_size=page, table_width=4096 // page, device=device, batch_size=bs,
                                max_tokens=max_tokens, accelerator_graph=True, causal=causal)
    q_static = torch.full((max_tokens, HEADS, DIM), float("nan"), device=device, dtype=torch.bfloat16)
    captured.plan(dtype=torch.bfloat16, **build_paged_indptrs(views + padding, page).to_kwargs_dict())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        captured.run(q_static, cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = captured.run(q_static, cache)
    q_static[:n_q] = q
    graph.replay()
    torch.cuda.synchronize()
    _check(out[:n_q], want, f"replayed, page {page}")
    assert out[n_q:].eq(0).all(), "a replay's padding queries must come out zero"
