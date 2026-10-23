"""Fused MLA latent norms (Triton) vs an fp32 reference and the torch path — parity."""
import pytest
import torch

from mstar.model.glm5_next.components import attention
from mstar.model.glm5_next.components.attention import Glm5NextMLAAttention
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.mla_kernels import (
    _HAS_TRITON,
    mla_latent_norm,
    mla_latent_norm_available,
)

_GPU = torch.cuda.is_available() and _HAS_TRITON
_BF16_ULP = 2.0 ** -7  # relative spacing of bf16 at the bottom of a binade


def _rmsnorm_fp32(x, weight, eps):
    """fp32 RMSNorm rounded to bf16 once, as FlashInfer's rmsnorm computes it."""
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(torch.bfloat16)


def _assert_bf16_parity(out, ref):
    """Within one bf16 ulp everywhere and bit-identical almost everywhere: fp32 summation
    order may flip the rounding of a value sitting on a bf16 boundary, nothing more."""
    torch.testing.assert_close(out, ref, rtol=_BF16_ULP, atol=1e-6)
    assert (out != ref).float().mean().item() < 1e-3


def _inputs(T, Q, L, scale, device, extra_cols=0):
    torch.manual_seed(T + Q)
    wide = torch.randn(T, Q + L + extra_cols, device=device) * scale
    fused = wide.to(torch.bfloat16)[:, : Q + L]  # extra_cols > 0 -> row stride > Q + L
    q_w = (1.0 + 0.1 * torch.randn(Q, device=device)).to(torch.bfloat16)
    kv_w = (1.0 + 0.1 * torch.randn(L, device=device)).to(torch.bfloat16)
    return fused, q_w, kv_w


@pytest.mark.skipif(not _GPU, reason="fused MLA latent norm needs CUDA + triton")
@pytest.mark.parametrize("Q,L,kpe", [(1536, 512, 64), (48, 32, 64)])  # GLM-5.3, reduced
@pytest.mark.parametrize("T", [1, 5, 64, 257])
@pytest.mark.parametrize("scale", [3.0, 1e-3])  # 1e-3: mean(x^2) ~ eps, so eps must be right
def test_kernel_matches_fp32_reference(Q, L, kpe, T, scale):
    fused, q_w, kv_w = _inputs(T, Q, L, scale, "cuda")
    assert mla_latent_norm_available(fused, q_w, kv_w)
    q_c, latent = mla_latent_norm(fused, q_w, 1e-5, kv_w, 1e-6, kpe)
    assert q_c.shape == (T, Q) and latent.shape == (T, L + kpe)
    assert q_c.is_contiguous() and latent.is_contiguous()
    _assert_bf16_parity(q_c, _rmsnorm_fp32(fused[:, :Q], q_w, 1e-5))
    _assert_bf16_parity(latent[:, :L], _rmsnorm_fp32(fused[:, Q:], kv_w, 1e-6))
    assert torch.equal(latent[:, L:], torch.zeros_like(latent[:, L:]))


@pytest.mark.skipif(not _GPU, reason="needs CUDA + triton")
def test_kernel_reads_a_strided_row():
    """The decode input is a fresh GEMM output, but nothing requires row stride == Q + L."""
    fused, q_w, kv_w = _inputs(7, 1536, 512, 1.0, "cuda", extra_cols=40)
    assert fused.stride(0) == 1536 + 512 + 40
    q_c, latent = mla_latent_norm(fused, q_w, 1e-5, kv_w, 1e-5, 64)
    _assert_bf16_parity(q_c, _rmsnorm_fp32(fused[:, :1536], q_w, 1e-5))
    _assert_bf16_parity(latent[:, :512], _rmsnorm_fp32(fused[:, 1536:], kv_w, 1e-5))


@pytest.mark.skipif(not _GPU, reason="needs CUDA + triton")
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("T", [1, 32, 300])
def test_module_fused_path_matches_torch_path(monkeypatch, reduced, T):
    """``_latent_norm`` through the kernel vs through FlashInfer rmsnorm + split/clone/cat."""
    cfg = Glm5NextModelConfig.reduced() if reduced else Glm5NextModelConfig()
    with torch.device("cuda"):
        attn = Glm5NextMLAAttention(cfg, layer_idx=3, kv_plane=0).to(torch.bfloat16)
    with torch.no_grad():
        attn.q_a_layernorm.weight.normal_(1.0, 0.1)
        attn.kv_a_layernorm.weight.normal_(1.0, 0.1)
    fused, _, _ = _inputs(T, cfg.q_lora_rank, cfg.kv_lora_rank, 2.0, "cuda")
    fused = fused.contiguous()

    q_fused, latent_fused = attn._latent_norm(fused)
    monkeypatch.setattr(attention, "mla_latent_norm_available", lambda *a: False)
    q_ref, latent_ref = attn._latent_norm(fused)

    _assert_bf16_parity(q_fused, q_ref)
    _assert_bf16_parity(latent_fused, latent_ref)


def test_torch_path_on_cpu():
    """No CUDA -> the torch path: q_c normed, cache row = normed kv || zero kpe."""
    cfg = Glm5NextModelConfig.reduced()
    attn = Glm5NextMLAAttention(cfg, layer_idx=3, kv_plane=0)
    fused = torch.randn(4, cfg.q_lora_rank + cfg.kv_lora_rank)
    assert not mla_latent_norm_available(
        fused, attn.q_a_layernorm.weight, attn.kv_a_layernorm.weight)
    q_c, latent = attn._latent_norm(fused)
    L = cfg.kv_lora_rank
    assert q_c.shape == (4, cfg.q_lora_rank) and latent.shape == (4, L + cfg.mla_cache_kpe)
    torch.testing.assert_close(q_c, attn.q_a_layernorm(fused[:, : cfg.q_lora_rank]))
    torch.testing.assert_close(latent[:, :L], attn.kv_a_layernorm(fused[:, cfg.q_lora_rank:]))
    assert torch.equal(latent[:, L:], torch.zeros_like(latent[:, L:]))
