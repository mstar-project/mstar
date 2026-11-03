"""``mstar::qk_norm_rope`` on an AMD GPU against the unfused pair it replaces.

The unfused path is what the Thinker's attention ran before: the registry's
RMSNorm on each of q and k, then ``apply_interleaved_mrope``, eager and
compiled. The fused kernel matches it to within a bf16 rounding and is mostly
bit-equal; through the real attention module, under graph replay and inside a
compiled module without a graph break. Qwen3-Omni's Thinker sizes (32 q heads,
4 kv heads, head_dim 128) and a TP2 shard of them.

    pytest test/rocm/test_qk_norm_rope.py
"""

from __future__ import annotations

import pytest
import torch

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernels run on AMD GPUs")

DEV = torch.device("cuda")
EPS = 1e-6


def _inputs(tokens, heads, kv_heads, head_dim=128, seed=0):
    from mstar.kernels.qk_norm_rope import _inputs as make

    g = torch.Generator(device=DEV).manual_seed(seed)
    return make({"tokens": tokens, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim},
                torch.bfloat16, DEV, g)


def _unfused(qkv, q_weight, k_weight, cos, sin, heads, kv_heads, head_dim, eps):
    from mstar.engine.resources.rms_norm import run_rms_norm
    from mstar.model.qwen3_omni.components.rope import apply_interleaved_mrope

    tokens = qkv.shape[0]
    q, k, _ = qkv.split([heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1)
    q = run_rms_norm(q.reshape(-1, head_dim), q_weight, eps).view(tokens, heads, head_dim)
    k = run_rms_norm(k.reshape(-1, head_dim), k_weight, eps).view(tokens, kv_heads, head_dim)
    return apply_interleaved_mrope(q, k, cos, sin)


def _fused(*args):
    import mstar.kernels.qk_norm_rope  # noqa: F401  declares the op

    return torch.ops.mstar.qk_norm_rope(*args)


def _close(a, b):
    # within two bf16 roundings at unit scale, elementwise (the inputs are O(1),
    # so a rotation that cancels to near zero keeps their absolute error)
    assert a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous()
    torch.testing.assert_close(a.float(), b.float(), rtol=1.6e-2, atol=1.6e-2)
    return (a == b).float().mean().item()


@pytest.mark.parametrize(("heads", "kv_heads"), [(32, 4), (16, 2)])
@pytest.mark.parametrize("tokens", [1, 3, 32, 64, 513])
def test_matches_the_unfused_pair(tokens, heads, kv_heads):
    args = _inputs(tokens, heads, kv_heads, seed=tokens)
    q, k = _fused(*args)
    # eager torch rounds q * cos, rot * sin and their sum to bf16 one by one,
    # so it is only within a rounding
    q_ref, k_ref = _unfused(*args)
    _close(q, q_ref)
    _close(k, k_ref)
    # compiled, as a served step runs it, the rotation is fp32 rounded once
    torch._dynamo.reset()
    q_ref, k_ref = torch.compile(_unfused)(*args)
    assert _close(q, q_ref) > 0.95
    assert _close(k, k_ref) > 0.95


def test_matches_its_reference():
    from mstar.kernels.qk_norm_rope import _reference

    args = _inputs(16, 32, 4, seed=3)
    for out, ref in zip(_fused(*args), _reference(*args), strict=True):
        assert _close(out, ref) > 0.99


@pytest.mark.parametrize("tokens", [1, 32])
def test_graph_replay(tokens):
    args = list(_inputs(tokens, 32, 4, seed=1))
    static = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    _fused(*static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fused(*static)
    for seed in (5, 6):
        fresh = _inputs(tokens, 32, 4, seed=seed)
        for dst, src in zip(static, fresh, strict=True):
            if isinstance(dst, torch.Tensor):
                dst.copy_(src)
        graph.replay()
        for got, want in zip(out, _fused(*fresh), strict=True):
            assert torch.equal(got, want)


def _attention():
    from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention

    torch.manual_seed(0)
    attn = Qwen3OmniAttention(hidden_size=512, num_heads=32, num_kv_heads=4, head_dim=128,
                              rms_norm_eps=EPS, use_mrope=True).to(DEV, torch.bfloat16)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0, 0.05)
        attn.q_norm.weight.add_(1)
        attn.k_norm.weight.add_(1)
    seen = []

    def attend(q, k, v):
        seen.append((q.clone(), k.clone(), v.clone()))
        return q

    attn.attend = attend
    return attn, seen


def test_the_attention_layer_takes_the_fused_kernel_with_the_same_inputs():
    attn, seen = _attention()
    assert attn._qk_norm_rope is torch.ops.mstar.qk_norm_rope
    x = torch.randn(8, 512, device=DEV, dtype=torch.bfloat16)
    _, _, _, cos, sin, *_ = _inputs(8, 32, 4, seed=9)
    with torch.no_grad():
        fused_out = attn(x, cos_sin_3d=(cos, sin))
        attn._qk_norm_rope = None
        plain_out = attn(x, cos_sin_3d=(cos, sin))
    (qf, kf, vf), (qp, kp, vp) = seen
    assert torch.equal(vf, vp)
    _close(qf, qp)
    _close(kf, kp)
    # a rounding apart in q moves the projected output by about that much
    rel = ((fused_out.float() - plain_out.float()).norm() / plain_out.float().norm()).item()
    assert rel < 5e-3, rel


def test_one_node_under_dynamo():
    args = _inputs(4, 32, 4, seed=2)
    torch._dynamo.reset()
    def step(*a):
        return torch.ops.mstar.qk_norm_rope(*a)

    explain = torch._dynamo.explain(step)(*args)
    assert explain.graph_break_count == 0, explain.break_reasons
