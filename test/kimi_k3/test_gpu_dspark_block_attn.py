"""GPU: the one-launch block attention and merge (``dspark_block_attention``) against the torch
glue of ``DSparkAttention.forward_block`` in true fp32: real-dims and tiny latent shapes, a row with
an empty context (log-sum-exp -inf) included."""
import pytest
import torch

from mstar.model.kimi_k3.dspark.block_attn_kernel import dspark_block_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def reference(q_lat, q_pe, lat, o_ctx, lse_ctx, rows, l, scale):
    h = q_lat.shape[1]
    c, k_pe = lat.split([l, lat.shape[-1] - l], dim=-1)
    ql = q_lat.view(rows, -1, h, l).float()
    qp = q_pe.view(rows, -1, h, k_pe.shape[-1]).float()
    c = c.view(rows, -1, l).float()
    k_pe = k_pe.view(rows, -1, k_pe.shape[-1]).float()
    scores = (torch.einsum("rqhl,rkl->rhqk", ql, c) + torch.einsum("rqhe,rke->rhqk", qp, k_pe)) * scale
    lse_blk = torch.logsumexp(scores, dim=-1)
    o_blk = torch.einsum("rhqk,rkl->rqhl", torch.softmax(scores, dim=-1), c).reshape(-1, h, l)
    lse_blk = lse_blk.transpose(1, 2).reshape(-1, h)
    m = torch.maximum(lse_ctx, lse_blk)
    w_ctx, w_blk = torch.exp(lse_ctx - m), torch.exp(lse_blk - m)
    o = (o_ctx.float() * w_ctx[..., None] + o_blk * w_blk[..., None]) / (w_ctx + w_blk)[..., None]
    return o.to(q_lat.dtype)


@pytest.mark.parametrize("rows,k,h,l,r", [(3, 7, 4, 512, 64), (2, 3, 2, 128, 32), (1, 7, 8, 512, 64), (5, 4, 1, 8, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_block_attention_matches_the_torch_glue(rows, k, h, l, r, dtype, strided):
    torch.manual_seed(0)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        t = rows * k
        if strided:
            # the queries as the draft hands them over: the latent and rope parts of one projection
            q = torch.randn(t, h, l + r + 8, device=DEV).to(dtype)
            q_lat, q_pe = q[:, :, :l], q[:, :, l:l + r]
            assert not q_lat.is_contiguous() and not q_pe.is_contiguous()
        else:
            q_lat = torch.randn(t, h, l, device=DEV).to(dtype)
            q_pe = torch.randn(t, h, r, device=DEV).to(dtype)
        lat = torch.randn(t, l + r, device=DEV).to(dtype)
        o_ctx = torch.randn(t, h, l, device=DEV).to(dtype)
        lse_ctx = torch.randn(t, h, device=DEV) * 3 + 5
        lse_ctx[:k] = float("-inf")  # the first row has no context yet
        scale = (l // 4 + r) ** -0.5 * 1.8
        want = reference(q_lat, q_pe, lat, o_ctx, lse_ctx, rows, l, scale)
        got = dspark_block_attention(q_lat, q_pe, lat, o_ctx, lse_ctx, rows, l, scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert got.shape == want.shape and got.dtype == dtype
    assert torch.isfinite(got).all()
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(got.float(), want.float(), atol=tol, rtol=tol), (got.float() - want.float()).abs().max()


def test_block_attention_is_capturable():
    rows, k, h, l, r = 2, 7, 4, 512, 64
    t = rows * k
    torch.manual_seed(1)
    args = (torch.randn(t, h, l, device=DEV).to(torch.bfloat16), torch.randn(t, h, r, device=DEV).to(torch.bfloat16),
            torch.randn(t, l + r, device=DEV).to(torch.bfloat16), torch.randn(t, h, l, device=DEV).to(torch.bfloat16),
            torch.randn(t, h, device=DEV))
    want = dspark_block_attention(*args, rows, l, 0.1)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = dspark_block_attention(*args, rows, l, 0.1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, want)
