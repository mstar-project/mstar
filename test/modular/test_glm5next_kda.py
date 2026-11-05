"""The KDA layer's dense reference on CPU: chunked prefill against the token-by-token
recurrence, and the forget gate against an independent fp64 reference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from mstar.model.components.kda import KDAConfig, KDAForgetGate, KimiDeltaAttention

OUT_ATOL = 1e-3  # bf16 outputs
STATE_ATOL = 1e-5  # fp32 recurrent states


def _build(seed: int) -> tuple[KimiDeltaAttention, KDAConfig]:
    """Reduced-geometry layer with small random weights, grad-free."""
    torch.manual_seed(seed)
    cfg = KDAConfig(hidden_size=64, num_heads=4, head_dim=32)
    layer = KimiDeltaAttention(cfg, dtype=torch.bfloat16)
    for lin in (
        layer.q_proj, layer.k_proj, layer.v_proj, layer.b_proj,
        layer.g_a_proj, layer.g_b_proj, layer.o_proj,
        layer.forget_gate.f_a_proj, layer.forget_gate.f_b_proj,
    ):
        lin.weight.data.normal_(0, 0.05)
    layer.conv1d.weight.data.normal_(0, 0.2)
    layer.forget_gate.dt_bias.data.normal_(0, 0.5)
    layer.forget_gate.A_log.data.normal_(0, 0.5)
    layer.o_norm.weight.data.normal_(1.0, 0.02)
    layer.requires_grad_(False)
    return layer, cfg


def _input(cfg: KDAConfig, batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, cfg.hidden_size, dtype=torch.bfloat16) * 0.1


@pytest.mark.parametrize("seq_len", [1, 4, 63, 64, 65, 130])
def test_prefill_matches_decode_loop(seq_len):
    """Chunked prefill == running the decode step token by token."""
    layer, cfg = _build(seed=1)
    batch_size = 2
    x = _input(cfg, batch_size, seq_len)

    out_prefill, s_prefill, cs_prefill = layer.prefill(x)

    s_decode, cs_decode = layer.init_state(batch_size)
    steps = [layer.decode_step(x[:, t : t + 1], s_decode, cs_decode) for t in range(seq_len)]
    out_decode = torch.cat(steps, dim=1)

    assert torch.equal(cs_prefill, cs_decode)  # conv path: bit-exact contract
    assert torch.allclose(out_prefill, out_decode, atol=OUT_ATOL, rtol=0)
    assert torch.allclose(s_prefill, s_decode, atol=STATE_ATOL, rtol=0)
    assert s_prefill.dtype == torch.float32 and s_decode.dtype == torch.float32


@pytest.mark.parametrize("prefix_len, tail_len", [(10, 1), (10, 5), (70, 60)])
def test_resumed_prefill_matches_decode_loop(prefix_len, tail_len):
    """A prefill resumed from carried state, shorter than a chunk or past one, equals decoding
    its tokens one by one from that state, and updates the carried state in place."""
    layer, cfg = _build(seed=7)
    batch_size = 2
    x = _input(cfg, batch_size, prefix_len + tail_len)

    _, s, cs = layer.prefill(x[:, :prefix_len])
    s_dec, cs_dec = s.clone(), cs.clone()
    out_chunk, s_ret, cs_ret = layer.prefill(
        x[:, prefix_len:], recurrent_state=s, conv_state=cs)
    assert s_ret is s and cs_ret is cs

    steps = [
        layer.decode_step(x[:, t : t + 1], s_dec, cs_dec)
        for t in range(prefix_len, prefix_len + tail_len)
    ]
    assert torch.equal(cs, cs_dec)  # conv path: bit-exact contract
    assert torch.allclose(out_chunk, torch.cat(steps, dim=1), atol=OUT_ATOL, rtol=0)
    assert torch.allclose(s, s_dec, atol=STATE_ATOL, rtol=0)


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_forget_gate_matches_independent_reference():
    torch.manual_seed(5)
    b, s, hidden, heads, head_dim = 2, 3, 16, 2, 4
    cfg = KDAConfig(hidden_size=hidden, num_heads=heads, head_dim=head_dim)
    x = torch.randn(b, s, hidden)
    w_a = torch.randn(head_dim, hidden) * 0.5
    w_b = torch.randn(heads * head_dim, head_dim) * 0.5
    # +30 saturates the sigmoid, so the closed bound is reached, not just the interior
    dt_bias = torch.randn(heads * head_dim) * 0.5
    dt_bias[::3] += 30.0
    a_log = torch.randn(heads) * 0.3

    f64 = np.float64
    g_np = (
        x.numpy().astype(f64)
        @ w_a.numpy().astype(f64).T
        @ w_b.numpy().astype(f64).T
        + dt_bias.numpy().astype(f64)
    ).reshape(b, s, heads, head_dim)
    decay_np = np.exp(a_log.numpy().astype(f64))[None, None, :, None]

    gate = KDAForgetGate(cfg, dtype=torch.float32)
    with torch.no_grad():
        gate.f_a_proj.weight.copy_(w_a)
        gate.f_b_proj.weight.copy_(w_b)
        gate.dt_bias.copy_(dt_bias)
        gate.A_log.copy_(a_log)
        out = gate(x)

    assert out.shape == (b, s, heads, head_dim)
    assert out.dtype == torch.float32
    expected = cfg.gate_lower_bound * _np_sigmoid(decay_np * g_np)
    assert bool((out < 0).all()) and bool((out >= cfg.gate_lower_bound).all())
    np.testing.assert_allclose(
        out.numpy().astype(f64), expected, rtol=1e-4, atol=1e-5)

    # the HF softplus branch (gate_lower_bound=None) is dead for this checkpoint: refused
    # rather than served by a second, untested implementation
    with pytest.raises(ValueError, match="softplus"):
        KDAForgetGate(KDAConfig(hidden_size=hidden, num_heads=heads, head_dim=head_dim,
                                gate_lower_bound=None))
