"""Backend-neutral RMSNorm operator tests."""

import pytest
import torch

from mstar.engine.resources.rms_norm import run_rms_norm


def test_rms_norm_portable_kernel_matches_reference():
    x = torch.tensor([[3.0, 4.0], [1.0, -2.0]])
    weight = torch.tensor([0.5, 1.5])
    eps = 1e-6

    actual = run_rms_norm(x, weight, eps)
    expected = (
        x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    ) * weight

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("gemma_mode", [False, True])
def test_fused_add_rms_norm_portable_kernel_is_the_unfused_add_and_norm(gemma_mode):
    torch.manual_seed(0)
    x, residual, weight = torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)
    eps = 1e-6
    total = x + residual
    expected = total * torch.rsqrt(total.square().mean(dim=-1, keepdim=True) + eps) * (
        1 + weight if gemma_mode else weight
    )

    torch.ops.mstar.fused_add_rms_norm(x, residual, weight, eps, gemma_mode)

    assert torch.equal(residual, total)
    torch.testing.assert_close(x, expected)


def test_fused_add_rms_norm_cuda_kernel_runs_a_dtype_mix_unfused():
    """FlashInfer's fused kernel reads ``residual`` and ``weight`` as ``x``'s
    dtype, so the CUDA kernel hands any other mix (here an fp32 weight) to the
    unfused add and norm, which casts the weight."""
    from mstar.engine.resources.rms_norm.rms_norm import _fused_add_rms_norm_cuda

    torch.manual_seed(0)
    x, residual = torch.randn(3, 8, dtype=torch.bfloat16), torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8)
    want_x, want_residual = x.clone(), residual.clone()
    torch.ops.mstar.fused_add_rms_norm(want_x, want_residual, weight, 1e-6, False)  # the portable kernel

    _fused_add_rms_norm_cuda(x, residual, weight, 1e-6, False)

    assert torch.equal(residual, want_residual) and torch.equal(x, want_x)


def test_rms_norm_operator_has_backend_neutral_name():
    assert hasattr(torch.ops.mstar, "rms_norm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused add+norm is the CUDA path")
def test_fused_add_rms_norm_compiles_without_a_graph_break():
    """FlashInfer's fused add+norm wrapper is plain Python, so traced it breaks
    the graph, and each norm call site then compiles (and recompiles) as a frame
    of its own. Behind ``mstar::fused_add_rms_norm`` a residual stream is one
    graph."""
    pytest.importorskip("flashinfer")
    from mstar.model.components.norm import RMSNorm

    norms = [RMSNorm(64).cuda().to(torch.bfloat16) for _ in range(3)]

    def stream(x):
        h, residual = x, None
        for norm in norms:
            h, residual = norm.forward_residual(h, residual)
        return h, residual

    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    want = stream(x.clone())
    torch._dynamo.reset()
    try:
        got = torch.compile(stream, fullgraph=True, backend="aot_eager")(x.clone())
    finally:
        torch._dynamo.reset()
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)
