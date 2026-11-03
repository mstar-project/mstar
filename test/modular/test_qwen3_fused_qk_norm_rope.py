"""Qwen3Attention's fused q/k RMSNorm + RoPE (``mstar::qk_norm_rope_inplace``) matches an fp64 reference, is no
less accurate in bf16 than the unfused path it replaces (RMSNorm, then FlashInfer's non-interleaved RoPE from
position ids), and works in place on q/k sliced out of one qkv projection, eagerly, under torch.compile and in a
CUDA graph, on the shapes Qwen3-ASR's language model runs; and the table-driven variant Qwen3-Omni's Thinker uses
for its multimodal RoPE."""
import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton + FlashInfer)")

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


@needs_cuda
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


@needs_cuda
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


@needs_cuda
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


@needs_cuda
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


@needs_cuda
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


@needs_cuda
@pytest.mark.parametrize("num_tokens,n_write", [(1, 1), (9, 6)])
def test_rotary_table_variant_applies_the_multimodal_rotation(num_tokens, n_write):
    """``qk_norm_rope_table_kv_write`` rotates by per-token cos/sin rows, here Qwen3-Omni's interleaved 3D MRoPE for
    an image span (its three position grids differ, so no 1D position reproduces it), and writes K/V as
    ``qk_norm_rope_kv_write`` does: k normed and rotated plus v at the planned slots of the first ``n_write``
    tokens, every other slot and the k input untouched; and no less accurate in bf16 than the unfused path (RMSNorm,
    then the rotate-half arithmetic in bf16)."""
    from mstar.model.components.norm import RMSNorm
    from mstar.model.qwen3_omni.components.rope import compute_3d_cos_sin, compute_rope_freqs
    from mstar.utils.attention import qk_norm_rope_table_kv_write

    qkv, wq, wk, _ = _inputs(num_tokens, torch.bfloat16)
    q, k, v = _split(qkv)
    t = torch.arange(num_tokens, device="cuda", dtype=torch.float)
    grids = torch.stack([torch.full_like(t, 700.0), t % 5 + 700, t % 3 + 700])
    cos, sin = compute_3d_cos_sin(grids, compute_rope_freqs(D, THETA, device="cuda"), (24, 20, 20),
                                  target_dtype=torch.bfloat16)

    def reference(x, w):
        x = x.double()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
        c, s = cos.double()[:, None], sin.double()[:, None]
        return x * c + torch.cat([-x[..., D // 2:], x[..., : D // 2]], dim=-1) * s

    def unfused(x, w):
        m = RMSNorm(D, eps=EPS).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w)
        x = m(x.reshape(-1, D)).view(x.shape)
        return x * cos[:, None] + torch.cat([-x[..., D // 2:], x[..., : D // 2]], dim=-1) * sin[:, None]

    ref_q, ref_k = reference(q, wq), reference(k, wk)
    unfused_q, unfused_k = unfused(q, wq), unfused(k, wk)  # before the op rotates q in place
    pages, page_size = 6, 4
    cache = torch.randn(pages, 2, page_size, HKV, D, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()
    flat = torch.randperm(pages * page_size, device="cuda")[:num_tokens]
    page_idx, slot_idx = flat // page_size, flat % page_size
    k_before = k.clone()
    qk_norm_rope_table_kv_write(q, k, v, wq, wk, cos, sin, cache, page_idx, slot_idx, n_write, EPS, EPS)
    torch.cuda.synchronize()
    torch.testing.assert_close(q.double(), ref_q, atol=2e-2, rtol=2e-2)
    assert torch.equal(k, k_before)
    w = slice(None, n_write)
    torch.testing.assert_close(cache[page_idx[w], 0, slot_idx[w]].double(), ref_k[w], atol=2e-2, rtol=2e-2)
    assert torch.equal(cache[page_idx[w], 1, slot_idx[w]], v[w])
    untouched = torch.ones(pages, page_size, dtype=torch.bool, device="cuda")
    untouched[page_idx[w], slot_idx[w]] = False
    assert torch.equal(cache.transpose(1, 2)[untouched], before.transpose(1, 2)[untouched])
    written_k = cache[page_idx[w], 0, slot_idx[w]]
    for fused, unfused_x, ref in ((q, unfused_q, ref_q), (written_k, unfused_k[w], ref_k[w])):
        err_fused = (fused.double() - ref).abs().mean()
        err_unfused = (unfused_x.double() - ref).abs().mean()
        assert err_fused <= err_unfused * 1.1, (err_fused, err_unfused)


@pytest.mark.parametrize("table", [False, True])
def test_fused_norm_helpers_refuse_a_gemma_norm(table):
    """The fused kernels scale by ``weight``; a Gemma-style norm scales by ``1 + weight``."""
    from types import SimpleNamespace

    from mstar.engine.resources.convenience import AttentionCallable
    from mstar.model.components.norm import RMSNorm

    attend = AttentionCallable(SimpleNamespace(), SimpleNamespace(requires_kv_write=False))
    q, k, v = torch.zeros(1, HQ, D), torch.zeros(1, HKV, D), torch.zeros(1, HKV, D)
    norm = RMSNorm(D, gemma_mode=True)
    with pytest.raises(ValueError, match="gemma_mode"):
        if table:
            attend.norm_rope_table_attend(q, k, v, norm, norm, (torch.ones(1, D), torch.zeros(1, D)))
        else:
            attend.norm_rope_attend(q, k, v, norm, norm, torch.zeros(1, dtype=torch.long), THETA)
