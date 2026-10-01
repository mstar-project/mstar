"""KimiDecoderLayer's fused residual-stream contract: each layer returns an
unreduced MLP partial and the pre-norm residual instead of a fully resolved
hidden state, deferring the all-reduce + residual add + RMSNorm to one fused
call (``all_reduce_add_rmsnorm``) consumed by the next layer (or the model's
final norm).

CPU-only, TP=1, so every fused call here takes the plain-torch path (no
comm_group to reduce over); the FlashInfer-backed path is covered by
``test_flashinfer_allreduce.py``. ``KimiMLAAttention`` needs bound engine
resources this host doesn't have, so ``self_attn`` and ``mlp`` are replaced
with deterministic ``nn.Linear`` stubs -- the residual-stream threading under
test doesn't depend on what produces each sub-layer's output.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.flashinfer_allreduce import all_reduce_add_rmsnorm
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k2_7.components.decoder_layer import KimiDecoderLayer
from mstar.model.kimi_k2_7.config import KimiK2Config


def _stub_linear(hidden_size: int, seed: int) -> nn.Linear:
    lin = nn.Linear(hidden_size, hidden_size, bias=False)
    lin.weight.data = torch.randn(
        hidden_size, hidden_size, generator=torch.Generator().manual_seed(seed),
    ) * 0.05
    return lin


def _build_layers(cfg: KimiK2Config, num_layers: int, seed: int) -> list[KimiDecoderLayer]:
    layers = []
    for layer_idx in range(num_layers):
        layer = KimiDecoderLayer(cfg, layer_idx)
        layer.input_layernorm.weight.data.normal_(
            mean=1.0, std=0.02, generator=torch.Generator().manual_seed(seed + layer_idx),
        )
        layer.post_attention_layernorm.weight.data.normal_(
            mean=1.0, std=0.02,
            generator=torch.Generator().manual_seed(seed + layer_idx + 100),
        )
        # Replace the real sub-layers: they only need to be Tensor -> Tensor.
        layer.self_attn = _stub_linear(cfg.hidden_size, seed + layer_idx + 200)
        layer.mlp = _stub_linear(cfg.hidden_size, seed + layer_idx + 300)
        layers.append(layer)
    return layers


def _ref_rmsnorm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def _ref_forward(layers, norm_weight, eps, h):
    for layer in layers:
        residual = h
        h = _ref_rmsnorm(h, layer.input_layernorm.weight, eps)
        h = layer.self_attn(h)
        h = residual + h

        residual = h
        h = _ref_rmsnorm(h, layer.post_attention_layernorm.weight, eps)
        h = layer.mlp(h)
        h = residual + h
    return _ref_rmsnorm(h, norm_weight, eps)


def test_decoder_stack_matches_unfused_reference():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    eps = cfg.rms_norm_eps
    layers = _build_layers(cfg, num_layers=3, seed=0)
    norm = RMSNorm(cfg.hidden_size, eps=eps)
    norm.weight.data.normal_(
        mean=1.0, std=0.02, generator=torch.Generator().manual_seed(999),
    )

    h0 = torch.randn(5, cfg.hidden_size, dtype=torch.float32) * 0.1

    expected = _ref_forward(layers, norm.weight, eps, h0)

    hidden_states, residual = h0, None
    for layer in layers:
        hidden_states, residual = layer(hidden_states, residual)
    _, got = all_reduce_add_rmsnorm(None, hidden_states, residual, norm)

    assert got.shape == h0.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
