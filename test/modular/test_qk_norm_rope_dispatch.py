"""Which platforms take ``mstar::qk_norm_rope`` in the Thinker's attention, and
its CPU reference against the unfused norm + multimodal RoPE path through the
real ``Qwen3OmniAttention`` (attention itself stubbed). The kernel is checked
on an AMD GPU in test/rocm/test_qk_norm_rope.py."""

from __future__ import annotations

import torch

from mstar.kernels import qk_norm_rope
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention


def test_amd_gpus_take_the_op_and_the_rest_none():
    dispatch = qk_norm_rope.qk_norm_rope_kernel
    assert dispatch._factories["rocm"] is qk_norm_rope._fused
    assert dispatch._factories["cuda"] is None and dispatch._default() is None


def _attention():
    torch.manual_seed(0)
    attn = Qwen3OmniAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, use_mrope=True)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0, 0.3)
        attn.q_norm.weight.add_(1)
        attn.k_norm.weight.add_(1)
    seen = []

    def attend(q, k, v):
        seen.append((q, k, v))
        return q

    attn.attend = attend
    return attn, seen


def test_the_reference_is_the_unfused_path_on_cpu():
    attn, seen = _attention()
    x = torch.randn(5, 64)
    angles = torch.randn(5, 8)
    cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)
    with torch.no_grad():
        attn._qk_norm_rope = None
        plain = attn(x, cos_sin_3d=(cos, sin))
        attn._qk_norm_rope = torch.ops.mstar.qk_norm_rope
        fused = attn(x, cos_sin_3d=(cos, sin))
    (qp, kp, vp), (qf, kf, vf) = seen
    assert torch.equal(vp, vf)
    torch.testing.assert_close(qf, qp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(kf, kp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused, plain, rtol=1e-5, atol=1e-6)


def test_without_mrope_the_layer_keeps_its_path():
    attn = Qwen3OmniAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, use_mrope=False)
    assert attn._qk_norm_rope is None
