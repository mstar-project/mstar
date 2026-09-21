"""GPU test: the Marlin MXFP4 MoE backend (vendored vLLM kernel, JIT-built) vs. the in-tree
Triton kernel on a small latent MoE, eager and under CUDA-graph replay."""
import pytest
import torch

from mstar.model.kimi_k3.reference.mxfp4 import quant_mxfp4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def _packed_experts(e, rows, k):
    w = torch.randn(e, rows, k) * 0.2
    p, s = zip(*[quant_mxfp4(w[i]) for i in range(e)], strict=True)
    return torch.stack(p).to(DEV), torch.stack(s).to(DEV)


def _small_moe():
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE

    torch.manual_seed(0)
    # Marlin tiles: latent % 256 == 0, inter % 128 == 0
    hidden, latent, inter, e, k = 256, 512, 256, 16, 4
    moe = KimiLatentMoE(hidden_size=hidden, latent_size=latent, num_experts=e, top_k=k,
                        moe_intermediate_size=inter, num_shared_experts=0, quantized=True).to(DEV, torch.bfloat16)
    with torch.no_grad():
        p13, s13 = _packed_experts(e, 2 * inter, latent)
        p2, s2 = _packed_experts(e, latent, inter)
        moe.experts.gate_up_packed.copy_(p13)
        moe.experts.gate_up_scale.copy_(s13)
        moe.experts.down_packed.copy_(p2)
        moe.experts.down_scale.copy_(s2)
        for prm in (moe.gate.weight, moe.in_proj.weight, moe.routed_expert_up_proj.weight):
            prm.normal_(std=0.05)
        moe.gate.e_score_correction_bias.normal_(std=0.01)
    return moe, hidden


@cuda
def test_marlin_backend_matches_triton():
    moe, hidden = _small_moe()
    with torch.no_grad():
        x = torch.randn(37, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_down(x)
        idx, w = moe.gate(x)
        ref = moe._routed(z, idx, w).float()  # Triton MXFP4 kernel
        moe.prepare_experts_backend("marlin", DEV)
        out = moe._routed(z, idx, w).float()
    rel = (out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel < 3e-2, f"marlin: rel rms {rel:.4f}"
    # a long prefill runs in token slices; the slices must reproduce the one-shot result
    moe._backend.max_chunk_tokens = 8
    with torch.no_grad():
        chunked = moe._routed(z, idx, w).float()
    moe._backend.max_chunk_tokens = 2048
    torch.testing.assert_close(chunked, out, rtol=0, atol=0)
    # the parameters were rebound to Marlin's layouts (int32 tiles, E8M0 scales), nothing dangling
    assert moe.experts.gate_up_packed.dtype == torch.int32 and moe.experts.gate_up_scale.dtype == torch.float8_e8m0fnu
    with torch.no_grad():
        y = moe(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # a later device / dtype pass (the engine's ``.to``) must keep the kernel layouts and the
    # backend on the parameters: no uint8 or bf16 copies, identical outputs
    moe.to(DEV)
    moe.to(torch.bfloat16)
    assert moe.experts.gate_up_packed.dtype == torch.int32 and moe.experts.gate_up_scale.dtype == torch.float8_e8m0fnu
    assert moe.experts.down_packed.dtype == torch.int32 and moe.experts.down_scale.dtype == torch.float8_e8m0fnu
    assert moe._backend.w13.data_ptr() == moe.experts.gate_up_packed.data_ptr()
    assert moe._backend.s13.data_ptr() == moe.experts.gate_up_scale.data_ptr()
    with torch.no_grad():
        again = moe._routed(z, idx, w).float()
    torch.testing.assert_close(again, out, rtol=0, atol=0)


@cuda
def test_marlin_backend_replays_in_cuda_graph():
    moe, hidden = _small_moe()
    moe.prepare_experts_backend("marlin", DEV)
    with torch.no_grad():
        x = torch.randn(3, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_down(x)
        idx, w = moe.gate(x)
        eager = moe._routed(z, idx, w).clone()
        static_z, static_idx, static_w = z.clone(), idx.clone(), w.clone()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(2):
                moe._routed(static_z, static_idx, static_w)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = moe._routed(static_z, static_idx, static_w)
        # new routing through the same graph: the kernels read every address from tensors
        x2 = torch.randn(3, hidden, device=DEV, dtype=torch.bfloat16)
        z2 = moe.routed_down(x2)
        idx2, w2 = moe.gate(x2)
        static_z.copy_(z2); static_idx.copy_(idx2); static_w.copy_(w2)
        g.replay()
        torch.cuda.synchronize()
        expect = moe._routed(z2, idx2, w2)
    torch.testing.assert_close(out, expect, rtol=0, atol=0)
    torch.testing.assert_close(moe._routed(z, idx, w), eager, rtol=0, atol=0)


@cuda
def test_marlin_backend_writes_into_a_given_output():
    """``out=`` receives the summed expert outputs bit for bit (the caller's all-reduce buffer),
    including through the token-chunked path."""
    moe, hidden = _small_moe()
    moe.prepare_experts_backend("marlin", DEV)
    with torch.no_grad():
        x = torch.randn(5, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_down(x)
        idx, w = moe.gate(x)
        plain = moe._routed(z, idx, w)
        out = torch.empty_like(plain)
        res = moe._routed(z, idx, w, out=out)
        assert res is out and torch.equal(out, plain)
        moe._backend.max_chunk_tokens = 2  # chunked: slices of the output buffer
        out2 = torch.empty_like(plain)
        res2 = moe._routed(z, idx, w, out=out2)
        assert res2 is out2 and torch.equal(out2, plain)
        assert torch.equal(moe._routed(z, idx, w), plain)


def _load_expert_shards(moe, p13, s13, p2, s2, inter):
    """Route the checkpoint-style per-expert tensors through the module's loaders (global ids)."""
    ex = moe.experts
    for e in range(p13.shape[0]):
        ex.gate_up_packed.weight_loader(ex.gate_up_packed, p13[e, :inter], f"gate:{e}")
        ex.gate_up_packed.weight_loader(ex.gate_up_packed, p13[e, inter:], f"up:{e}")
        ex.gate_up_scale.weight_loader(ex.gate_up_scale, s13[e, :inter], f"gate:{e}")
        ex.gate_up_scale.weight_loader(ex.gate_up_scale, s13[e, inter:], f"up:{e}")
        ex.down_packed.weight_loader(ex.down_packed, p2[e], f"down:{e}")
        ex.down_scale.weight_loader(ex.down_scale, s2[e], f"down:{e}")


def _sharded_copies(full, world, ep, p13, s13, p2, s2, backend):
    """One module per rank of a ``world``-rank group with ``ep`` expert-parallel groups, holding
    the same experts as ``full`` (trivial comm group: ``_routed`` returns the rank's partial)."""
    from mstar.model.components.expert_sharding import ExpertSharding
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE

    ranks = []
    for r in range(world):
        sh = ExpertSharding(full.num_experts, full.moe_intermediate_size, world, r, ep_size=ep)
        m = KimiLatentMoE(hidden_size=full.hidden_size, latent_size=full.latent_size, num_experts=full.num_experts,
                          top_k=full.top_k, moe_intermediate_size=full.moe_intermediate_size, num_shared_experts=0,
                          quantized=True, expert_sharding=sh).to(DEV, torch.bfloat16)
        with torch.no_grad():
            _load_expert_shards(m, p13, s13, p2, s2, full.moe_intermediate_size)
            m.gate.weight.copy_(full.gate.weight)
            m.gate.e_score_correction_bias.copy_(full.gate.e_score_correction_bias)
            m.in_proj.weight.copy_(full.in_proj.weight)
        if backend is not None:
            m.prepare_experts_backend(backend, DEV)
        ranks.append(m)
    return ranks


@cuda
@pytest.mark.parametrize("world,ep", [(2, 2), (4, 2), (4, 4)])
def test_marlin_expert_parallel_partials_sum_to_full(world, ep):
    """Expert parallelism on the Marlin backend: each rank computes only its experts (global ids
    mapped to local ones, the other ranks' assignments skipped at alignment, their top-k slots
    zero) and the per-rank partials add up to the single-rank result."""
    moe, hidden = _small_moe()
    ex = moe.experts
    inter = moe.moe_intermediate_size
    p13, s13, p2, s2 = (t.clone() for t in (ex.gate_up_packed, ex.gate_up_scale, ex.down_packed, ex.down_scale))
    ranks = _sharded_copies(moe, world, ep, p13, s13, p2, s2, "marlin")
    moe.prepare_experts_backend("marlin", DEV)
    with torch.no_grad():
        x = torch.randn(37, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_down(x)
        idx, w = moe.gate(x)
        ref = moe._routed(z, idx, w).float()
        parts = [m._routed(z, idx, w) for m in ranks]
    for m, part in zip(ranks, parts, strict=True):
        sh = m.sharding
        assert m.experts.gate_up_packed.shape[0] == sh.local_experts
        if sh.is_partial:
            none_here = (sh.localize(idx) == sh.invalid_id).all(1)
            assert torch.equal(part[none_here], torch.zeros_like(part[none_here]))
    total = sum(p.float() for p in parts)
    rel = (total - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel < 2e-2, f"ep{ep}/tp{world // ep}: rel rms {rel:.4f}"  # bf16 rounding of each partial
    # the partial is also correct when written into a caller's buffer (the all-reduce buffer path)
    with torch.no_grad():
        out = torch.empty_like(parts[0])
        assert ranks[0]._routed(z, idx, w, out=out) is out and torch.equal(out, parts[0])


@cuda
def test_bf16_path_matches_marlin_on_long_slices():
    """From ``bf16_min_tokens`` tokens a slice runs as bf16 grouped GEMMs on weights dequantized from the
    Marlin tiles (``_forward_bf16``): the same result as Marlin's kernels up to the GEMMs' accumulation
    order, on a plain output, a given output and a column-chunked view of it."""
    pytest.importorskip("triton")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm missing")
    moe, hidden = _small_moe()
    moe.prepare_experts_backend("marlin", DEV)
    be = moe._backend
    torch.manual_seed(5)
    m = 300
    z = (torch.randn(m, moe.latent_size, device=DEV) * 0.5).to(torch.bfloat16)
    idx = torch.rand(m, moe.num_experts, device=DEV).topk(moe.top_k, dim=1).indices.to(torch.int32)
    w = torch.softmax(torch.randn(m, moe.top_k, device=DEV), dim=1)
    be.bf16_min_tokens = 0
    want = be(z, idx, w)
    be.bf16_min_tokens = 1
    assert be._bf16_applies(z)
    got = be(z, idx, w)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 1e-2, rel
    # both against the fp32 reference on the same dequantized weights: the bf16 path is no farther from it
    from mstar.model.kimi_k3.reference.moe import routed_experts_loop
    from mstar.utils.fused_moe.marlin.dequant import dequant_marlin_experts

    w13 = dequant_marlin_experts(be.w13, be.s13, 2 * moe.inter_local, moe.latent_size).float()
    w2 = dequant_marlin_experts(be.w2, be.s2, moe.latent_size, moe.inter_local).float()
    ref = routed_experts_loop(z.float(), idx.long(), w, w13, w2, moe.situ_beta, moe.situ_linear_beta)
    err_marlin = ((want.float() - ref).norm() / ref.norm()).item()
    err_bf16 = ((got.float() - ref).norm() / ref.norm()).item()
    assert err_bf16 < 1.5 * err_marlin + 1e-3, (err_bf16, err_marlin)
    out = torch.empty_like(want)
    assert be(z, idx, w, out=out) is out and torch.equal(out, got)
    chunks = torch.empty(4, m, moe.latent_size // 4, dtype=z.dtype, device=DEV)
    view = chunks.permute(1, 0, 2)
    assert be(z, idx, w, out=view) is view
    assert torch.equal(view.reshape(m, moe.latent_size), got)
    # slices: a long input runs slice by slice through the same path
    be.max_chunk_tokens = 128
    sliced = be(z, idx, w)
    assert torch.equal(sliced, got)
    be.max_chunk_tokens = type(be).max_chunk_tokens
    # no host sync anywhere in the path (a prefill of 92 layers must not stall on it)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        again = be(z, idx, w)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.equal(again, got)
    # under a CUDA-graph capture the path steps aside and Marlin's kernels are captured
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert not be._bf16_applies(z)
            captured = be(z, idx, w)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, want)
    be.bf16_min_tokens = type(be).bf16_min_tokens


@cuda
def test_bf16_path_warm_up_prepares_the_shared_buffers():
    """``warm_bf16_path`` runs the path once at setup (kernels compiled, the weight buffers allocated), so a
    server's first long prefill does not pay for it; a no-op while the path is off."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm missing")
    from mstar.utils.fused_moe.marlin import _BF16_BUFFERS

    moe, hidden = _small_moe()
    moe.prepare_experts_backend("marlin", DEV)
    be = moe._backend
    be.bf16_min_tokens = 0
    assert be.warm_bf16_path() == 0.0
    be.bf16_min_tokens = 64
    shape = (moe.num_experts, moe.inter_local, moe.latent_size)
    for key in [k for k in _BF16_BUFFERS if k[:3] == shape]:
        _BF16_BUFFERS.pop(key)
    took = be.warm_bf16_path()
    keys = [k for k in _BF16_BUFFERS if k[:3] == shape]
    assert took > 0 and len(keys) == 1, (took, keys)
    w13, w2 = _BF16_BUFFERS[keys[0]]
    assert w13.shape == (moe.num_experts, 2 * moe.inter_local, moe.latent_size) and w2.shape == (moe.num_experts, moe.latent_size, moe.inter_local)
    be.bf16_min_tokens = type(be).bf16_min_tokens
