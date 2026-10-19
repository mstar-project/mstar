"""The varlen kernel the dense attention backend resolves to on ROCm (AITER) at the Cosmos3 DiT
shapes: head dim 128, GQA 32/8 (Nano) and 16/8 (Edge), a short text prefix in front of 390 tokens per
latent frame at 480x832, non-causal, one or two (CFG) segments, plus the causal text prefill. Scored
against fp32 SDPA next to torch's own bf16 SDPA on the same inputs."""

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.attn import dense
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_rocm() and dense.flash_attn_varlen_func is not None),
    reason="needs a ROCm GPU with AITER varlen",
)

TOKENS_PER_FRAME = 390  # 480x832 -> 30x52 latent -> 15x26 patches


def _sdpa(q, k, v, causal, dtype):
    q, k, v = (x.transpose(0, 1).unsqueeze(0).to(dtype) for x in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    return out.squeeze(0).transpose(0, 1)


def _rel(x, ref):
    return ((x.float() - ref).norm() / ref.norm()).item()


def _inputs(segments, hq, hkv, peak, seed=0):
    """segments: [(q_len, prefix_len)]; q/k unit-RMS per head like the QK-normed DiT, times ``peak``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nq = sum(s[0] for s in segments)
    nk = sum(s[0] + s[1] for s in segments)
    q = torch.randn(nq, hq, 128, device="cuda", generator=g) * peak
    k = torch.randn(nk, hkv, 128, device="cuda", generator=g)
    v = torch.randn(nk, hkv, 128, device="cuda", generator=g)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


@pytest.mark.parametrize("frames", [1, 5, 9])
@pytest.mark.parametrize("heads", [(32, 8), (16, 8)])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("peak", [1.0, 3.0])
def test_gen_attention_matches_fp32(frames, heads, cfg, peak):
    n = frames * TOKENS_PER_FRAME
    segments = [(n, 23), (n, 7)] if cfg else [(n, 23)]
    q, k, v = _inputs(segments, *heads, peak)
    cu_q = torch.tensor([0] + [s[0] for s in segments], device="cuda").cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0] + [s[0] + s[1] for s in segments], device="cuda").cumsum(0).to(torch.int32)
    max_q = max(s[0] for s in segments)
    max_k = max(s[0] + s[1] for s in segments)
    out = dense.flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=False)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(out).all()
    errs, base = [], []
    for i in range(len(segments)):
        qs, ks, vs = q[cu_q[i]:cu_q[i + 1]], k[cu_k[i]:cu_k[i + 1]], v[cu_k[i]:cu_k[i + 1]]
        ref = _sdpa(qs.float(), ks.float(), vs.float(), False, torch.float32)
        errs.append(_rel(out[cu_q[i]:cu_q[i + 1]], ref))
        base.append(_rel(_sdpa(qs, ks, vs, False, torch.bfloat16), ref))
    # bf16 output rounding alone is ~2e-3 relative; the kernel may not be much worse than torch's
    for e, b in zip(errs, base, strict=True):
        assert e <= max(1.5 * b, 4e-3), f"aiter {e:.3e} vs torch bf16 {b:.3e}"


@pytest.mark.parametrize("und_len", [7, 23, 130, 300])
def test_text_prefill_causal_matches_fp32(und_len):
    q, k, v = _inputs([(und_len, 0)], 32, 8, 1.0, seed=und_len)
    cu = torch.tensor([0, und_len], device="cuda", dtype=torch.int32)
    out = dense.flash_attn_varlen_func(q, k, v, cu, cu, und_len, und_len, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    ref = _sdpa(q.float(), k.float(), v.float(), True, torch.float32)
    base = _rel(_sdpa(q, k, v, True, torch.bfloat16), ref)
    assert _rel(out, ref) <= max(1.5 * base, 4e-3)
