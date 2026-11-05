"""The fused Sinkhorn kernel against the torch reference, on CUDA."""
import pytest
import torch

from mstar.model.glm5_next.mhc import sinkhorn_normalize
from mstar.model.glm5_next.sinkhorn_kernel import (
    _HAS_TRITON,
    fused_sinkhorn_available,
    sinkhorn_normalize_fused,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON), reason="fused Sinkhorn needs CUDA + triton",
)


def _comb(shape):
    """A comb matrix as the mHC forward produces it: softmax(dim=-1) + eps."""
    torch.manual_seed(0)
    logits = torch.randn(*shape, device="cuda", dtype=torch.float32)
    return torch.softmax(logits, dim=-1) + 1e-6


@pytest.mark.parametrize("shape", [(1, 1, 4, 4), (2, 3, 4, 4), (17, 4, 4), (256, 4, 4)])
def test_fused_matches_reference(shape):
    m = _comb(shape)
    before = m.clone()
    assert fused_sinkhorn_available(m)
    ref = sinkhorn_normalize(m, 20, 1e-6)
    fused = sinkhorn_normalize_fused(m, 20, 1e-6)
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-6)
    # Sinkhorn ends on a column step, so the columns sum to 1
    col_sums = fused.sum(dim=-2)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(m, before)  # functional, like the reference
