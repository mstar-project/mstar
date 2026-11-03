"""Qwen3Attention's fused q/k RMSNorm + RoPE (``mstar::qk_norm_rope_inplace``) matches an fp64 reference, is no
less accurate in bf16 than the unfused path it replaces (RMSNorm, then FlashInfer's non-interleaved RoPE from
position ids), and works in place on q/k sliced out of one qkv projection, eagerly, under torch.compile and in a
CUDA graph, on the shapes Qwen3-ASR's language model runs."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton + FlashInfer)")

HQ, HKV, D, EPS, THETA = 16, 8, 128, 1e-6, 1_000_000.0
HIDDEN = 256


def _reference(x, w, pos):
    x = x.double()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
    inv = THETA ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float64) / D)
    ang = pos[: x.shape[0]].double()[:, None] * inv
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def _split(qkv):
    n = qkv.shape[0]
    q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
    return q.view(n, HQ, D), k.view(n, HKV, D), v.view(n, HKV, D)


def _inputs(num_tokens, dtype, pos_len=None):
    torch.manual_seed(0)
    qkv = torch.randn(num_tokens, (HQ + 2 * HKV) * D, device="cuda", dtype=dtype)
    wq = (torch.rand(D, device="cuda") + 0.5).to(dtype)
    wk = (torch.rand(D, device="cuda") + 0.5).to(dtype)
    # the position resource's buffer can be longer than the step; only the first num_tokens count
    pos = torch.randint(0, 4000, (pos_len or num_tokens,), device="cuda", dtype=torch.long)
    return qkv, wq, wk, pos


@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float32, 5e-3)])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_matches_fp64_reference_in_place(dtype, tol, num_tokens):
    from mstar.utils.attention import qk_norm_rope_inplace

    qkv, wq, wk, pos = _inputs(num_tokens, dtype, pos_len=num_tokens + 5)
    q, k, v = _split(qkv)
    ref_q, ref_k, v_before = _reference(q, wq, pos), _reference(k, wk, pos), v.clone()
    qk_norm_rope_inplace(q, k, wq, wk, pos, EPS, EPS, THETA)
    torch.testing.assert_close(q.double(), ref_q, atol=tol, rtol=tol)
    torch.testing.assert_close(k.double(), ref_k, atol=tol, rtol=tol)
    assert torch.equal(v, v_before)


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_no_worse_than_unfused_bf16(num_tokens):
    import flashinfer

    from mstar.model.components.norm import RMSNorm
    from mstar.utils.attention import qk_norm_rope_inplace

    qkv, wq, wk, pos = _inputs(num_tokens, torch.bfloat16)
    q, k, _ = _split(qkv)
    ref_q, ref_k = _reference(q, wq, pos), _reference(k, wk, pos)

    def norm(x, w):
        m = RMSNorm(D, eps=EPS).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w)
        return m(x.reshape(-1, D)).view(x.shape).contiguous()

    uq, uk = flashinfer.rope.apply_rope_pos_ids(
        norm(q, wq), norm(k, wk), pos, rotary_dim=D, interleave=False, rope_theta=THETA,
    )
    qk_norm_rope_inplace(q, k, wq, wk, pos, EPS, EPS, THETA)
    for fused, unfused, ref in ((q, uq, ref_q), (k, uk, ref_k)):
        err_fused = (fused.double() - ref).abs().mean()
        err_unfused = (unfused.double() - ref).abs().mean()
        assert err_fused <= err_unfused * 1.1, (err_fused, err_unfused)


def _layer(x, w_qkv, wq, wk, pos):
    """The attention layer's prefix: one qkv projection, split, q/k fused in place."""
    from mstar.utils.attention import qk_norm_rope_inplace

    q, k, v = _split(x @ w_qkv.t())
    qk_norm_rope_inplace(q, k, wq, wk, pos, EPS, EPS, THETA)
    return q, k, v


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_under_compile_from_one_projection(num_tokens):
    """torch.compile of mm -> split -> in-place fused op: inductor functionalizes the mutation through the custom
    op and the result matches eager."""
    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w_qkv = torch.randn((HQ + 2 * HKV) * D, HIDDEN, device="cuda", dtype=torch.bfloat16) / HIDDEN ** 0.5
    _, wq, wk, pos = _inputs(num_tokens, torch.bfloat16)
    eager = _layer(x, w_qkv, wq, wk, pos)
    compiled = torch.compile(_layer, fullgraph=True)(x, w_qkv, wq, wk, pos)
    torch.cuda.synchronize()
    for a, b in zip(compiled, eager, strict=True):
        torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)


def test_cuda_graph_replay_follows_inputs():
    from mstar.utils.attention import qk_norm_rope_inplace

    qkv, wq, wk, pos = _inputs(8, torch.bfloat16)
    src = qkv.clone()
    q, k, _ = _split(qkv)
    qk_norm_rope_inplace(q, k, wq, wk, pos, EPS, EPS, THETA)  # warm the JIT
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        qk_norm_rope_inplace(q, k, wq, wk, pos, EPS, EPS, THETA)
    src.normal_()
    qkv.copy_(src)
    pos.copy_(torch.randint(0, 4000, (8,), device="cuda"))
    sq, sk, _ = _split(src)
    ref_q, ref_k = _reference(sq, wq, pos), _reference(sk, wk, pos)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(q.double(), ref_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.double(), ref_k, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens,n_write", [(1, 1), (7, 7), (8, 5)])
def test_kv_write_variant_matches_and_writes_the_planned_slots(num_tokens, n_write):
    """``qk_norm_rope_kv_write``: q as ``qk_norm_rope_inplace`` leaves it; the
    normed, rotated k and the raw v at each written token's (page, slot) of the
    NHD layer; tokens past ``n_write`` (plan padding) and every other slot
    untouched; the k input as it was."""
    from mstar.utils.attention import qk_norm_rope_inplace, qk_norm_rope_kv_write

    qkv, wq, wk, pos = _inputs(num_tokens, torch.bfloat16)
    q, k, v = _split(qkv)
    ref = qkv.clone()
    rq, rk, _ = _split(ref)
    qk_norm_rope_inplace(rq, rk, wq, wk, pos, EPS, EPS, THETA)
    pages, page_size = 6, 4
    cache = torch.randn(pages, 2, page_size, HKV, D, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()
    flat = torch.randperm(pages * page_size, device="cuda")[:num_tokens]
    page_idx, slot_idx = flat // page_size, flat % page_size
    k_before = k.clone()
    qk_norm_rope_kv_write(q, k, v, wq, wk, pos, cache, page_idx, slot_idx, n_write, EPS, EPS, THETA)
    torch.cuda.synchronize()
    assert torch.equal(q, rq)
    assert torch.equal(k, k_before)
    expect = before.clone()
    expect[page_idx[:n_write], 0, slot_idx[:n_write]] = rk[:n_write]
    expect[page_idx[:n_write], 1, slot_idx[:n_write]] = v[:n_write]
    assert torch.equal(cache, expect)
