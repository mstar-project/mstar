"""On a GPU: FlashInfer's FA2 prefill wrapper under device rewind. A captured plan made for rows
of some lengths runs after the device shortened them (`FlashInferPrefillWrapper.correct_kv_lens`),
and must attend as a plan of the shortened lengths does. FA2's kernel counts a row's KV chunks from
its length at run time and its merge from the planned count, so rows shortened across a chunk
boundary (128 tokens at least) need the split plan's arrays planned again in the graph
(`_SplitPlan.replan`); without it they don't attend right. Skips without CUDA."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
flashinfer = pytest.importorskip("flashinfer")

from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper, _SplitPlan  # noqa: E402

GPU = torch.device("cuda", 0)
PAGE, HQ, HKV, HEAD_DIM, BLOCK = 128, 32, 8, 128, 8


def _arrays(lens, pages_of):
    npages = [-(-n // PAGE) for n in lens]
    indptr = torch.tensor([0, *torch.tensor(npages).cumsum(0).tolist()], dtype=torch.int32)
    indices = torch.cat([pages_of[i][: npages[i]] for i in range(len(lens))]).to(torch.int32)
    last = torch.tensor([n - (p - 1) * PAGE for n, p in zip(lens, npages, strict=True)], dtype=torch.int32)
    qo = torch.arange(0, (len(lens) + 1) * BLOCK, BLOCK, dtype=torch.int32)
    return qo, indptr, indices, last


def _worst_error(rows: int, causal: bool) -> float:
    """The largest difference from a fresh unsplit plan of the cut rows, over rows planned just
    past 128, 256 and 512 tokens and cut by up to a block's drafts."""
    torch.manual_seed(0)
    pages_of = [torch.randperm(64)[:6] for _ in range(rows)]
    cache = torch.randn(64, 2, PAGE, HKV, HEAD_DIM, dtype=torch.bfloat16, device=GPU)
    q = torch.randn(rows * BLOCK, HQ, HEAD_DIM, dtype=torch.bfloat16, device=GPU)
    wrapper = FlashInferPrefillWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device=GPU), HQ, HKV, HEAD_DIM, PAGE, batch_size=rows,
        max_total_tokens=rows * BLOCK, max_num_pages=64, device=GPU, accelerator_graph=True, backend="fa2",
    )
    ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device=GPU), "NHD", backend="fa2")
    cut = torch.zeros(rows, dtype=torch.int32, device=GPU)
    out = torch.empty_like(q)

    def plan(lens):
        qo, indptr, indices, last = _arrays(lens, pages_of)
        wrapper.plan(qo, indptr, indices, last, causal=causal, dtype=torch.bfloat16,
                     kv_lens=torch.tensor(lens, dtype=torch.int32))
        wrapper.locate_kv_lens()

    plan([600] * rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.copy_(wrapper.run(q, cache))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.correct_kv_lens(cut)
        out.copy_(wrapper.run(q, cache))
    worst = 0.0
    for edge in (128, 256, 512):
        for top in range(edge + 1, edge + BLOCK):
            lens = [top - i % 2 for i in range(rows)]
            cuts = [(top + i) % BLOCK for i in range(rows)]
            plan(lens)
            cut.copy_(torch.tensor(cuts, dtype=torch.int32))
            graph.replay()
            ref.plan(*_arrays([n - c for n, c in zip(lens, cuts, strict=True)], pages_of), HQ, HKV, HEAD_DIM, PAGE,
                     causal=causal, q_data_type=torch.bfloat16, disable_split_kv=True)
            worst = max(worst, (out.float() - ref.run(q, cache).float()).abs().max().item())
    return worst


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "non_causal"])
@pytest.mark.parametrize("rows", [4, 16])
def test_a_captured_split_plan_cut_on_the_device_attends_as_a_plan_of_the_cut_rows(rows, causal, monkeypatch):
    assert _worst_error(rows, causal) < 0.02
    monkeypatch.setattr(_SplitPlan, "replan", lambda self, *args: None)
    assert _worst_error(rows, causal) > 0.1
