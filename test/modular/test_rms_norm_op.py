"""Backend-neutral RMSNorm operator tests."""

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


def test_rms_norm_operator_has_backend_neutral_name():
    assert hasattr(torch.ops.mstar, "rms_norm")
