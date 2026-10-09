"""GPU (sm90 with FlashMLA built): the FlashMLA wrapper against the FlashInfer wrapper on one paged latent
cache, eager and captured, at a 12-head per-rank shape."""
import pytest
import torch

from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAWrapper
from mstar.engine.resources.attn.flashmla import FlashMLAWrapper, flashmla_available

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and flashmla_available()),
                                reason="needs a Hopper GPU with FlashMLA")
DEV = torch.device("cuda")
H, L, R, PAGE = 12, 512, 64, 64


def plan_inputs(lengths):
    pages = [(n + PAGE - 1) // PAGE for n in lengths]
    indptr = [0]
    for p in pages:
        indptr.append(indptr[-1] + p)
    idx = torch.randperm(indptr[-1] + 3)[: indptr[-1]].to(torch.int32)  # pages in a scrambled order, a few unused
    last = [n - (p - 1) * PAGE for n, p in zip(lengths, pages, strict=True)]
    return (torch.arange(len(lengths) + 1, dtype=torch.int32), torch.tensor(indptr, dtype=torch.int32), idx,
            torch.tensor(last, dtype=torch.int32)), indptr[-1] + 3


@pytest.mark.parametrize("lengths", [[1500], [1, 64, 65, 1023, 1536], [700] * 64])
def test_flashmla_matches_flashinfer(lengths):
    torch.manual_seed(0)
    (qo, kv_indptr, kv_idx, last), num_pages = plan_inputs(lengths)
    cache = torch.randn(num_pages, PAGE, L + R, device=DEV).to(torch.bfloat16)
    rows = len(lengths)
    q_nope = torch.randn(rows, H, L, device=DEV).to(torch.bfloat16)
    q_pe = torch.randn(rows, H, R, device=DEV).to(torch.bfloat16)
    scale = (128 + R) ** -0.5
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    fi = FlashInferMLAWrapper(ws, H, L, R, PAGE, scale, device=DEV)
    fi.plan(qo, kv_indptr, kv_idx, last, causal=True, dtype=torch.bfloat16)
    want = fi.run(q_nope, q_pe, cache)
    fm = FlashMLAWrapper(H, L, R, PAGE, scale, max_pages_per_row=32, device=DEV)
    fm.plan(qo, kv_indptr, kv_idx, last, causal=True, dtype=torch.bfloat16)
    got = fm.run(q_nope, q_pe, cache)
    assert got.shape == want.shape == (rows, H, L)
    assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2), (got.float() - want.float()).abs().max()
    # the lse too (natural log), against FlashInfer's
    want_o, want_lse = fi.run(q_nope, q_pe, cache, return_lse=True)
    got_o, got_lse = fm.run(q_nope, q_pe, cache, return_lse=True)
    assert torch.allclose(got_lse, want_lse, atol=1e-2, rtol=1e-3), (got_lse - want_lse).abs().max()


def test_flashmla_captured_replay_equals_eager():
    torch.manual_seed(1)
    lengths = [300, 1536, 64, 900]
    (qo, kv_indptr, kv_idx, last), num_pages = plan_inputs(lengths)
    cache = torch.randn(num_pages, PAGE, L + R, device=DEV).to(torch.bfloat16)
    rows = len(lengths)
    q_nope = torch.randn(rows, H, L, device=DEV).to(torch.bfloat16)
    q_pe = torch.randn(rows, H, R, device=DEV).to(torch.bfloat16)
    fm = FlashMLAWrapper(H, L, R, PAGE, (128 + R) ** -0.5, max_pages_per_row=32, batch_size=rows, device=DEV,
                         use_cuda_graph=True)
    fm.plan(qo, kv_indptr, kv_idx, last, causal=True, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    eager = fm.run(q_nope, q_pe, cache).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fm.run(q_nope, q_pe, cache)
    torch.cuda.current_stream().wait_stream(s)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    # a new plan with other lengths on the same buffers, replayed
    lengths2 = [1000, 1, 1536, 65]
    (qo2, kv_indptr2, kv_idx2, last2), _ = plan_inputs(lengths2)
    fm.plan(qo2, kv_indptr2, kv_idx2, last2, causal=True, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    fi = FlashInferMLAWrapper(workspace, H, L, R, PAGE, (128 + R) ** -0.5, device=DEV)
    fi.plan(qo2, kv_indptr2, kv_idx2, last2, causal=True, dtype=torch.bfloat16)
    assert torch.allclose(out.float(), fi.run(q_nope, q_pe, cache).float(), atol=2e-2, rtol=2e-2)
