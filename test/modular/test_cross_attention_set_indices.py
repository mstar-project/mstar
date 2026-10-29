"""A captured cross-attention wrapper re-planned over other pages of the same
lengths takes only the new indices: FlashInfer's fa2 schedule depends on the
indptrs and lengths, so the output must match a fresh plan bit for bit, in a
replay of a graph captured under the old plan too."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer runs on CUDA")

ROWS, PAGES, PAGE, HEADS, DIM, POOL = 8, 12, 128, 4, 64, 1 + 8 * 12 * 2


def _wrapper():
    from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper

    return FlashInferPrefillWrapper(
        workspace_buffer=torch.empty(64 << 20, dtype=torch.uint8, device="cuda"),
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM, page_size=PAGE,
        batch_size=ROWS, max_total_tokens=ROWS, max_num_pages=POOL, accelerator_graph=True,
    )


def _plan(wrapper, indices):
    wrapper.plan(
        qo_indptr=torch.arange(ROWS + 1, dtype=torch.int32),
        paged_kv_indptr=torch.arange(ROWS + 1, dtype=torch.int32) * PAGES,
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.full((ROWS,), 92, dtype=torch.int32),
        causal=False,
    )


def test_set_indices_matches_a_fresh_plan():
    torch.manual_seed(0)
    cache = torch.randn(POOL, 2, PAGE, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(ROWS, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    old = torch.randperm(POOL - 1, dtype=torch.int32)[: ROWS * PAGES] + 1
    new = torch.randperm(POOL - 1, dtype=torch.int32)[: ROWS * PAGES] + 1

    fresh = _wrapper()
    _plan(fresh, new)
    want = fresh.run(q, cache)

    reused = _wrapper()
    _plan(reused, old)
    assert reused.replans_from_indices
    static_q = q.clone()
    graph = torch.cuda.CUDAGraph()
    reused.run(static_q, cache)  # warm-up outside the capture
    with torch.cuda.graph(graph):
        static_out = reused.run(static_q, cache)
    reused.set_indices(new)
    assert torch.equal(reused.run(q, cache), want)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, want)
