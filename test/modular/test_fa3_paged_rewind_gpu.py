"""On an SM90 GPU with the flash_attn_3 wheel: paged FA3 as a verify step's attention under device
rewind (`FA3PagedWrapper`). A captured step planned for rows of some lengths, whose rows the
device then shortens (`correct_kv_lens`), must attend exactly as an eager plan of the shortened
rows does, and as FlashInfer's FA2 does within bf16 rounding. Skips elsewhere."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flash_attn_3")
flashinfer = pytest.importorskip("flashinfer")

from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper, fa3_paged_unavailable_reason  # noqa: E402

GPU = torch.device("cuda", 0)
PAGE, HQ, HKV, HEAD_DIM, MAX_SEQ, LAYERS = 128, 32, 8, 128, 4096, 3
if (reason := fa3_paged_unavailable_reason(GPU, HEAD_DIM, torch.bfloat16)) is not None:
    pytest.skip(reason, allow_module_level=True)


def _arrays(lens, pages_of, span):
    npages = [-(-n // PAGE) for n in lens]
    indptr = torch.tensor([0, *torch.tensor(npages).cumsum(0).tolist()], dtype=torch.int32)
    indices = torch.cat([pages_of[i][: npages[i]] for i in range(len(lens))]).to(torch.int32)
    last = torch.tensor([n - (p - 1) * PAGE for n, p in zip(lens, npages, strict=True)], dtype=torch.int32)
    qo = torch.arange(0, (len(lens) + 1) * span, span, dtype=torch.int32)
    return qo, indptr, indices, last


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "non_causal"])
@pytest.mark.parametrize("rows,span", [(1, 8), (4, 8), (32, 8), (16, 1)])
def test_a_captured_plan_cut_on_the_device_attends_as_a_plan_of_the_cut_rows(rows, span, causal):
    """Three layers a step, as a model's attend through one plan and one schedule."""
    torch.manual_seed(rows + span)
    pages_of = [torch.randperm(255)[:16] + 1 for _ in range(rows)]  # page 0 is the sink
    caches = torch.randn(LAYERS, 256, 2, PAGE, HKV, HEAD_DIM, dtype=torch.bfloat16, device=GPU)
    q = torch.randn(rows * span, HQ, HEAD_DIM, dtype=torch.bfloat16, device=GPU)
    kw = dict(num_qo_heads=HQ, num_kv_heads=HKV, head_dim=HEAD_DIM, page_size=PAGE, max_seq_len=MAX_SEQ, device=GPU)
    captured = FA3PagedWrapper(batch_size=rows, accelerator_graph=True, **kw)
    eager = FA3PagedWrapper(**kw)
    fa2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=GPU), "NHD", backend="fa2")
    cut = torch.zeros(rows, dtype=torch.int32, device=GPU)
    out = torch.empty(LAYERS, *q.shape, dtype=q.dtype, device=GPU)

    def step():
        captured.correct_kv_lens(cut)
        for layer in range(LAYERS):
            out[layer].copy_(captured.run(q, caches[layer]))

    def plan(wrapper, lens):
        wrapper.plan(*_arrays(lens, pages_of, span), causal=causal, dtype=torch.bfloat16,
                     kv_lens=torch.tensor(lens, dtype=torch.int32))

    plan(captured, [600] * rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    worst = 0.0
    # rows planned on either side of page and split-block edges, cut by up to a block's drafts;
    # every row keeps at least its block, as a verify step's does
    for top in (span + 16, 127 + span, 129 + span, 511 + span, 1500, 2048):
        lens = [top - i % 3 for i in range(rows)]
        cuts = [(top + i) % 8 for i in range(rows)]
        short = [n - c for n, c in zip(lens, cuts, strict=True)]
        plan(captured, lens)
        cut.copy_(torch.tensor(cuts, dtype=torch.int32))
        graph.replay()
        plan(eager, short)
        fa2.plan(*_arrays(short, pages_of, span), HQ, HKV, HEAD_DIM, PAGE, causal=causal,
                 q_data_type=torch.bfloat16)
        assert torch.isfinite(out).all()
        for layer in range(LAYERS):
            assert torch.equal(out[layer], eager.run(q, caches[layer])), f"layer {layer}, lengths {short}"
            worst = max(worst, (out[layer].float() - fa2.run(q, caches[layer]).float()).abs().max().item())
    assert worst < 0.02
