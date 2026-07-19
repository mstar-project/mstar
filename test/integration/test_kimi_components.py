"""M1 golden tests for Kimi-K2.7 cheap reused components.

For each cheap component of the DeepSeek-V3 text backbone — RMSNorm, the dense
SwiGLU MLP, the token embedding, and the LM head — build the mstar component from
``KimiK2Config.reduced()``, load identical random weights into it and an
independent reference, and assert the outputs match. The reference formulas are
inlined here (self-contained; no dependency on the local golden harness) and each
is cited to the vLLM DeepSeek-V3 source so the golden is authoritative.

This is a GPU test: mstar's standard RMSNorm dispatches to a FlashInfer fused
kernel, so the suite runs on ``cuda``. It skips automatically without a GPU.

Run:  pytest test/integration/test_kimi_components.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.language_model import (
    build_dense_mlp,
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="M1 golden tests need a GPU (mstar RMSNorm uses a FlashInfer kernel)",
)

DEVICE = "cuda"


def _cfg() -> KimiK2Config:
    return KimiK2Config.reduced()


# --------------------------------------------------------------------------
# Independent references (cited to vllm-project/vllm .../models/deepseek_v2.py)
# --------------------------------------------------------------------------

def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Standard Llama/DeepSeek RMSNorm (HF ``LlamaRMSNorm`` / vLLM
    # ``RMSNorm.forward_native``): normalize in fp32, scale by weight in the
    # input dtype.
    orig_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return weight * x32.to(orig_dtype)


def _ref_swiglu(
    x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
) -> torch.Tensor:
    # ``DeepseekV2MLP.forward`` = ``down_proj(SiluAndMul(gate_up_proj(x)))``,
    # bias=False, silu-only. ``SiluAndMul([g, u]) = silu(g) * u``.
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    return F.linear(F.silu(gate) * up, down_w)


def _ref_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(ids, weight)


def _ref_lm_head(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Untied head (``tie_word_embeddings=False``): plain ``x @ weight.T``.
    return F.linear(x, weight)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    cfg = _cfg()
    dtype = torch.bfloat16  # FlashInfer rmsnorm runs in half precision
    n_tokens = 7

    norm = build_rmsnorm(cfg).to(device=DEVICE, dtype=dtype)
    weight = torch.randn(cfg.hidden_size, device=DEVICE, dtype=dtype)
    norm.weight.data.copy_(weight)

    x = torch.randn(n_tokens, cfg.hidden_size, device=DEVICE, dtype=dtype)

    got = norm(x)
    expected = _ref_rmsnorm(x, weight, cfg.rms_norm_eps)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_dense_mlp_matches_reference():
    torch.manual_seed(1)
    cfg = _cfg()
    dtype = torch.float32
    n_tokens = 5
    h, i = cfg.hidden_size, cfg.intermediate_size

    mlp = build_dense_mlp(cfg).to(device=DEVICE, dtype=dtype)
    gate_w = torch.randn(i, h, device=DEVICE, dtype=dtype) * 0.05
    up_w = torch.randn(i, h, device=DEVICE, dtype=dtype) * 0.05
    down_w = torch.randn(h, i, device=DEVICE, dtype=dtype) * 0.05
    # Load through the real (stacked) loaders: gate -> shard 0, up -> shard 1.
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate_w, loaded_shard_id=0)
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up_w, loaded_shard_id=1)
    mlp.down_proj.weight_loader(mlp.down_proj.weight, down_w)

    x = torch.randn(n_tokens, h, device=DEVICE, dtype=dtype)

    got = mlp(x)
    expected = _ref_swiglu(x, gate_w, up_w, down_w)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_embedding_matches_reference():
    torch.manual_seed(2)
    cfg = _cfg()
    dtype = torch.float32

    emb = build_embedding(cfg).to(device=DEVICE, dtype=dtype)
    weight = torch.randn(cfg.vocab_size, cfg.hidden_size, device=DEVICE, dtype=dtype)
    emb.weight_loader(emb.weight, weight)

    ids = torch.randint(0, cfg.vocab_size, (9,), device=DEVICE)

    got = emb(ids)
    expected = _ref_embedding(ids, weight)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_lm_head_matches_reference():
    torch.manual_seed(3)
    cfg = _cfg()
    dtype = torch.float32
    n_tokens = 4

    head = build_lm_head(cfg).to(device=DEVICE, dtype=dtype)
    weight = torch.randn(cfg.vocab_size, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.02
    head.weight_loader(head.weight, weight)

    x = torch.randn(n_tokens, cfg.hidden_size, device=DEVICE, dtype=dtype)

    got = head(x)
    expected = _ref_lm_head(x, weight)
    assert got.shape == (n_tokens, cfg.vocab_size)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
