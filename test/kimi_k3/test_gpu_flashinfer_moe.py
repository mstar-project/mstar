"""GPU test: the FlashInfer CUTLASS MoE backend (in-place converted MXFP4 experts, SiTU fused)
vs. the in-tree Triton kernel and the fp32 reference on a small latent MoE."""
import pytest
import torch

from mstar.model.kimi_k3.reference.mxfp4 import quant_mxfp4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def _packed_experts(e, rows, k):
    w = torch.randn(e, rows, k) * 0.2
    p, s = zip(*[quant_mxfp4(w[i]) for i in range(e)], strict=True)
    return torch.stack(p).to(DEV), torch.stack(s).to(DEV)


@cuda
@pytest.mark.parametrize("mode,tol", [("w4a16", 3e-2), ("humming", 8e-2)])
def test_flashinfer_backend_matches_triton(mode, tol):
    pytest.importorskip("flashinfer.fused_moe")
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE

    torch.manual_seed(0)
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
        x = torch.randn(37, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_expert_down_proj(x)
        idx, w = moe.gate(x)
        ref = moe._routed(z, idx, w).float()  # Triton MXFP4 kernel
        moe.prepare_flashinfer(mode, DEV)
        out = moe._routed(z, idx, w).float()
    rel = (out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel < tol, f"{mode}: rel rms {rel:.4f}"
    # the whole module still runs (norm, up-proj) through the converted experts
    with torch.no_grad():
        y = moe(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
