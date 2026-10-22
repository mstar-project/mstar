"""The fp32 paths FlashInfer cannot take: RMSNorm and RoPE fall back to torch
implementations that follow the fused kernels' math (CUDA)."""
import pytest
import torch

from mstar.engine.resources.position.rope import _rope_portable_inplace
from mstar.engine.resources.rms_norm import run_rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = torch.device("cuda")


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("rotary_dim", [None, 64])
@pytest.mark.parametrize("llama31", [False, True])
def test_portable_rope_matches_flashinfer(interleave, rotary_dim, llama31):
    import flashinfer

    torch.manual_seed(0)
    q = torch.randn(37, 4, 128, device=DEVICE)
    k = torch.randn(37, 2, 128, device=DEVICE)
    pos = torch.randint(0, 9000, (37,), device=DEVICE, dtype=torch.int32)
    kw = dict(rotary_dim=rotary_dim, interleave=interleave, rope_scale=8.0 if llama31 else 2.0,
              rope_theta=5e5 if llama31 else 1e4)
    smooth = dict(low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192) if llama31 else {}

    fq, fk = q.bfloat16(), k.bfloat16()
    if llama31:
        flashinfer.rope.apply_llama31_rope_pos_ids_inplace(fq, fk, pos, **kw, **smooth)
    else:
        flashinfer.rope.apply_rope_pos_ids_inplace(fq, fk, pos, **kw)
    pq, pk = q.clone(), k.clone()
    _rope_portable_inplace(pq, pk, pos, kw["rotary_dim"], interleave, kw["rope_scale"], kw["rope_theta"],
                           smooth.get("low_freq_factor"), smooth.get("high_freq_factor"),
                           smooth.get("old_context_len"))
    # FlashInfer's fast sincos and bf16 storage bound the agreement
    torch.testing.assert_close(pq, fq.float(), atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(pk, fk.float(), atol=3e-2, rtol=2e-2)


def test_fp32_rope_runs_in_fp32_through_the_op():
    q = torch.randn(5, 2, 64, device=DEVICE)
    k = torch.randn(5, 2, 64, device=DEVICE)
    q0 = q.clone()
    torch.ops.mstar.rope_apply_qk_inplace(q, k, torch.arange(5, device=DEVICE), None, None, False, 1.0, 1e4)
    assert q.dtype is torch.float32
    assert not torch.equal(q, q0)
    # position 0 is the identity
    torch.testing.assert_close(q[0], q0[0])


def test_fp32_rms_norm_runs_in_fp32_and_matches_bf16_kernel():
    torch.manual_seed(0)
    x = torch.randn(16, 256, device=DEVICE)
    w = torch.randn(256, device=DEVICE)
    out = run_rms_norm(x, w, eps=1e-6)
    assert out.dtype is torch.float32
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(out, ref)
    fused = run_rms_norm(x.bfloat16(), w.bfloat16(), eps=1e-6)
    torch.testing.assert_close(out, fused.float(), atol=5e-2, rtol=2e-2)
