"""GPU golden for per-token-group fp8 activation quant."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="per-token-group fp8 quant kernel needs a GPU",
)

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn


def test_per_token_group_quant_fp8_roundtrip():
    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    torch.manual_seed(2)
    x = torch.randn(64, 256, device=DEVICE).to(torch.bfloat16)
    x_q, x_s = per_token_group_quant_fp8(x, 128)

    assert x_q.dtype == FP8
    assert x_s.dtype == torch.float32 and x_s.shape == (64, 2)
    deq = x_q.to(torch.float32) * x_s.repeat_interleave(128, dim=1)
    # e4m3's 3-bit mantissa bounds the relative error at 2^-4 once the group
    # scale is divided out; atol covers the subnormal tail near zero.
    torch.testing.assert_close(deq, x.to(torch.float32), rtol=0.07, atol=1e-3)
