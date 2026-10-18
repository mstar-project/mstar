"""RoPE through ``PositionManager`` on AMD GPUs, against FlashInfer's semantics.

The ROCm kernel reads the fp32 cos/sin cache the manager builds, so these check
the cache (frequency scale, Llama 3.1 blend, partial rotary width) together with
the kernel, on q/k as the strided views of a fused QKV projection that attention
layers pass.
"""

import math

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = torch.device("cuda")
LABEL = "text"


def _inv_freq(rotary_dim, theta, scale, llama31):
    """FlashInfer's per-pair frequency (``pos_enc.cuh``), in fp64."""
    inv = 1.0 / theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    if llama31 is None:
        return inv / scale
    low, high, old = llama31
    smooth = (inv * old / (2 * math.pi * (high - low)) - low / (high - low)).clamp(0, 1)
    return (1 - smooth) * inv / scale + smooth * inv


def _reference(x, pos, rotary_dim, interleave, inv_freq):
    xf = x.double()
    ang = pos.double()[:, None] * inv_freq.to(x.device)[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    rot = xf[..., :rotary_dim]
    if interleave:
        a, b = rot[..., 0::2], rot[..., 1::2]
        out = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)
    else:
        a, b = rot[..., : rotary_dim // 2], rot[..., rotary_dim // 2:]
        out = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    return torch.cat((out, xf[..., rotary_dim:]), dim=-1)


CASES = {
    # name: (heads, kv_heads, head_dim, PositionConfig kwargs)
    "neox": (24, 8, 128, dict(rope_theta=500000.0)),
    "llama31": (24, 8, 128, dict(
        rope_theta=500000.0, rope_scale=32.0,
        low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192,
    )),
    "interleave": (16, 8, 128, dict(interleave=True)),
    "scaled": (16, 8, 64, dict(rope_scale=4.0)),
    "partial-0.25": (16, 4, 256, dict(rotary_dim=64, rope_theta=1e7)),
    "partial-0.5": (8, 2, 128, dict(rotary_dim=64)),
}


@pytest.mark.parametrize("tokens", [1, 37, 1024])
@pytest.mark.parametrize("case", list(CASES))
def test_rope_matches_flashinfer_semantics(case, tokens):
    heads, kv_heads, head_dim, kwargs = CASES[case]
    config = PositionConfig(kv_cache="kv", **kwargs)
    manager = RopeManager(config, DEV, max_seq_len=32768, head_dim=head_dim)
    assert manager._rope_cache.dtype == torch.float32

    torch.manual_seed(tokens)
    pos = torch.randint(0, 32768, (tokens,), device=DEV)
    manager._current_pos_ids[LABEL] = pos
    qkv = torch.randn(tokens, (heads + 2 * kv_heads) * head_dim, device=DEV, dtype=torch.bfloat16)
    q = qkv[:, : heads * head_dim].view(tokens, heads, head_dim)
    k = qkv[:, heads * head_dim:(heads + kv_heads) * head_dim].view(tokens, kv_heads, head_dim)
    v_before = qkv[:, (heads + kv_heads) * head_dim:].clone()

    rotary_dim = config.rotary_dim or head_dim
    llama31 = tuple(config.llama31_params.values()) or None
    inv = _inv_freq(rotary_dim, config.rope_theta, config.rope_scale, llama31)
    ref_q = _reference(q, pos, rotary_dim, config.interleave, inv)
    ref_k = _reference(k, pos, rotary_dim, config.interleave, inv)

    out_q, out_k = manager.apply_qk(q, k, LABEL, **config.llama31_params)
    for out, ref in ((out_q, ref_q), (out_k, ref_k)):
        # against the exact result rounded once: what is left is the kernel's own
        # fp32 arithmetic, an occasional one-ulp difference
        ref = ref.to(out.dtype)
        rel = ((out.double() - ref.double()).norm() / ref.double().norm()).item()
        assert rel < 6e-4, f"{case}: relative error {rel:.2e}"
        torch.testing.assert_close(out, ref, rtol=2 ** -7, atol=1e-2)
    assert out_q.data_ptr() == q.data_ptr(), "rotated in place"
    assert torch.equal(qkv[:, (heads + kv_heads) * head_dim:], v_before), "v untouched"
    if rotary_dim < head_dim:
        assert torch.equal(out_q[..., rotary_dim:], ref_q[..., rotary_dim:].bfloat16())


def test_runtime_arguments_must_match_the_cache():
    config = PositionConfig(kv_cache="kv", rope_theta=10000.0)
    manager = RopeManager(config, DEV, max_seq_len=128, head_dim=64)
    manager._current_pos_ids[LABEL] = torch.arange(4, device=DEV)
    q = torch.randn(4, 2, 64, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(4, 2, 64, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="do not match the cache"):
        manager.apply_qk(q, k, LABEL, rope_theta=1e6)


def test_rope_under_graph_capture():
    config = PositionConfig(kv_cache="kv", rope_theta=500000.0)
    manager = RopeManager(config, DEV, max_seq_len=4096, head_dim=128)
    pos = torch.zeros(8, dtype=torch.long, device=DEV)
    manager._current_pos_ids[LABEL] = pos
    q = torch.randn(8, 4, 128, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(8, 2, 128, device=DEV, dtype=torch.bfloat16)
    manager.apply_qk(q.clone(), k.clone(), LABEL)
    torch.cuda.synchronize()
    q_static, k_static = q.clone(), k.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        manager.apply_qk(q_static, k_static, LABEL)
    for step in range(3):
        pos.copy_(torch.arange(8, device=DEV) + 100 * step)
        q_static.copy_(q)
        k_static.copy_(k)
        graph.replay()
        eq, ek = q.clone(), k.clone()
        manager.apply_qk(eq, ek, LABEL)
        torch.cuda.synchronize()
        assert torch.equal(q_static, eq) and torch.equal(k_static, ek)
