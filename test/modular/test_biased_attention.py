"""``dense_bias_attention`` keeps a dense-bias SDPA off cuDNN and matches the unpinned call."""

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from mstar.model.components.biased_attention import dense_bias_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _attention_ops(fn) -> set[str]:
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if "attention" in e.key}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("t", [137, 300])
def test_matches_unpinned_sdpa_off_cudnn(dtype, t):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(4, 8, t, 128, device="cuda", dtype=dtype, generator=g) for _ in range(3))
    bias = torch.randn(4, 8, t, t, device="cuda", dtype=dtype, generator=g)
    # a FastConformer padding mask: the last keys of row 0 masked out
    bias[0, :, :, t - 20:] = torch.finfo(dtype).min

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    out = dense_bias_attention(q, k, v, bias)
    tol = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 1e-4}[dtype]
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)

    ops = _attention_ops(lambda: dense_bias_attention(q, k, v, bias))
    assert not any("cudnn" in op for op in ops), ops
    assert any("efficient_attention" in op or "math" in op for op in ops), ops
