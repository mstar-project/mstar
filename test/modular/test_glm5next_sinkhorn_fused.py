"""Fused Sinkhorn (Triton) vs the pure-torch reference — parity (M3 Phase 2)."""
import pytest
import torch

from mstar.model.glm5_next.mhc import _dispatch_sinkhorn, sinkhorn_normalize
from mstar.model.glm5_next.sinkhorn_kernel import (
    _HAS_TRITON,
    fused_sinkhorn_available,
    sinkhorn_normalize_fused,
)

_GPU = torch.cuda.is_available() and _HAS_TRITON


def _comb(shape, device):
    """A comb matrix as the mHC forward produces it: softmax(dim=-1) + eps."""
    torch.manual_seed(0)
    logits = torch.randn(*shape, device=device, dtype=torch.float32)
    return torch.softmax(logits, dim=-1) + 1e-6


@pytest.mark.skipif(not _GPU, reason="fused Sinkhorn needs CUDA + triton")
@pytest.mark.parametrize("shape", [(1, 1, 4, 4), (2, 3, 4, 4), (17, 4, 4), (256, 4, 4)])
def test_fused_matches_reference(shape):
    m = _comb(shape, "cuda")
    assert fused_sinkhorn_available(m)
    ref = sinkhorn_normalize(m, 20, 1e-6)
    fused = sinkhorn_normalize_fused(m, 20, 1e-6)
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-6)
    # Sinkhorn ends on a column step -> column sums are ~1 (doubly-stochastic).
    col_sums = fused.sum(dim=-2)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _GPU, reason="needs CUDA + triton")
def test_fused_does_not_mutate_input():
    m = _comb((8, 4, 4), "cuda")
    before = m.clone()
    _ = sinkhorn_normalize_fused(m, 20, 1e-6)
    torch.testing.assert_close(m, before)  # functional, like the reference


def test_dispatch_falls_back_on_cpu():
    """No CUDA/triton -> the dispatch returns the pure reference, unchanged."""
    m = _comb((5, 4, 4), "cpu")
    torch.testing.assert_close(
        _dispatch_sinkhorn(m, 20, 1e-6), sinkhorn_normalize(m, 20, 1e-6)
    )
