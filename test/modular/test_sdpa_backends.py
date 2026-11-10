"""``no_cudnn_sdpa`` turns off cuDNN attention and leaves every other SDPA backend as it was."""

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from mstar.model.components.sdpa_backends import no_cudnn_sdpa

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sdpa_ops(fn) -> set[str]:
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if "dot_product" in e.key}


def test_only_cudnn_is_turned_off():
    with no_cudnn_sdpa():
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        assert torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()


def test_cpu_keeps_its_flash_kernel():
    q, k, v = (torch.randn(2, 4, 50, 8) for _ in range(3))
    bias = torch.randn(2, 4, 50, 50)
    ref_ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
    with no_cudnn_sdpa():
        ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
    assert ops == ref_ops, (ops, ref_ops)


@_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("t", [137, 300])
def test_dense_bias_matches_unpinned_off_cudnn(dtype, t):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(4, 8, t, 128, device="cuda", dtype=dtype, generator=g) for _ in range(3))
    bias = torch.randn(4, 8, t, t, device="cuda", dtype=dtype, generator=g)
    # a FastConformer padding mask: the last keys of row 0 masked out
    bias[0, :, :, t - 20:] = torch.finfo(dtype).min

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    with no_cudnn_sdpa():
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        ops = _sdpa_ops(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
    tol = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 1e-4}[dtype]
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
    assert not any("cudnn" in op for op in ops), ops
    assert any("efficient_attention" in op or "math" in op for op in ops), ops
