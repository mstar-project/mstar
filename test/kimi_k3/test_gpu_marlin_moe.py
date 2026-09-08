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
        for prm in (moe.gate.weight, moe.routed_expert_down_proj.weight, moe.routed_expert_up_proj.weight):
            prm.normal_(std=0.05)
        moe.gate.e_score_correction_bias.normal_(std=0.01)
    return moe, hidden


@cuda
def test_marlin_backend_matches_triton():
    moe, hidden = _small_moe()
    with torch.no_grad():
        x = torch.randn(37, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_expert_down_proj(x)
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
        z = moe.routed_expert_down_proj(x)
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
        z2 = moe.routed_expert_down_proj(x2)
        idx2, w2 = moe.gate(x2)
        static_z.copy_(z2); static_idx.copy_(idx2); static_w.copy_(w2)
        g.replay()
        torch.cuda.synchronize()
        expect = moe._routed(z2, idx2, w2)
    torch.testing.assert_close(out, expect, rtol=0, atol=0)
    torch.testing.assert_close(moe._routed(z, idx, w), eager, rtol=0, atol=0)
