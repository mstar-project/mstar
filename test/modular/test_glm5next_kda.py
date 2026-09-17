"""GLM-5.3-Flash KDA parity tests — CPU-only, pure torch (no flashinfer/triton)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
import torch.nn.functional as F

from mstar.model.glm5_next.kda import (
    Glm5NextKdaConfig,
    Glm5NextLinearAttention,
)

OUT_ATOL = 1e-3  # bf16 outputs
STATE_ATOL = 1e-5  # fp32 recurrent states


def _build(seed: int) -> tuple[Glm5NextLinearAttention, Glm5NextKdaConfig]:
    """Reduced-geometry layer with small random weights, grad-free."""
    torch.manual_seed(seed)
    cfg = Glm5NextKdaConfig.reduced()
    layer = Glm5NextLinearAttention(cfg, dtype=torch.bfloat16)
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


def _input(cfg: Glm5NextKdaConfig, batch_size: int, seq_len: int) -> torch.Tensor:
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


def test_conv_state_stores_raw_projection_tail():
    """Conv state == last kernel-1 RAW pre-SiLU columns of the padded conv input."""
    layer, cfg = _build(seed=2)
    batch_size, seq_len = 2, 9
    x = _input(cfg, batch_size, seq_len)

    _, _, cs = layer.prefill(x)

    mixed = torch.cat(
        [layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)], dim=-1
    ).transpose(1, 2)
    tail = cfg.linear_conv_kernel_size - 1
    expected = F.pad(mixed, (tail, 0))[:, :, -tail:]
    assert torch.equal(cs, expected)
    assert cs.shape == (batch_size, cfg.linear_conv_channels, tail)
    assert cs.is_contiguous()

    # L < kernel-1: the zero left-pad must show up in the stored tail.
    x1 = _input(cfg, batch_size, 1)
    _, _, cs1 = layer.prefill(x1)
    mixed1 = torch.cat(
        [layer.q_proj(x1), layer.k_proj(x1), layer.v_proj(x1)], dim=-1
    ).transpose(1, 2)
    assert torch.equal(cs1[:, :, :2], torch.zeros_like(cs1[:, :, :2]))
    assert torch.equal(cs1[:, :, 2:], mixed1)


def test_chunked_prefill_resume_matches_full_prefill():
    """prefill(70) + prefill(60, carried state) == prefill(130)."""
    layer, cfg = _build(seed=3)
    batch_size, seq_len, split = 2, 130, 70
    x = _input(cfg, batch_size, seq_len)

    out_full, s_full, cs_full = layer.prefill(x)

    out_a, s, cs = layer.prefill(x[:, :split])
    out_b, s_ret, cs_ret = layer.prefill(
        x[:, split:], recurrent_state=s, conv_state=cs
    )
    assert s_ret is s and cs_ret is cs  # carried slots are updated in place
    out_resumed = torch.cat([out_a, out_b], dim=1)

    assert torch.equal(cs, cs_full)
    assert torch.allclose(out_resumed, out_full, atol=OUT_ATOL, rtol=0)
    assert torch.allclose(s, s_full, atol=STATE_ATOL, rtol=0)


@pytest.mark.parametrize("tail_len", [1, 3, 5])
def test_small_continue_chunk_matches_decode_loop(tail_len):
    """A small continue window (L << 64) runs the chunk kernel at an
    effective chunk covering L (no 64-pad, no 63-iteration substitution
    loop), and must agree with decoding the same tokens one by one from
    the same carried state.
    """
    layer, cfg = _build(seed=7)
    batch_size, prefill_len = 2, 10
    x = _input(cfg, batch_size, prefill_len + tail_len)

    _, s, cs = layer.prefill(x[:, :prefill_len])
    s_dec, cs_dec = s.clone(), cs.clone()

    out_chunk, s_ret, cs_ret = layer.prefill(
        x[:, prefill_len:], recurrent_state=s, conv_state=cs)
    assert s_ret is s and cs_ret is cs

    steps = [
        layer.decode_step(x[:, t : t + 1], s_dec, cs_dec)
        for t in range(prefill_len, prefill_len + tail_len)
    ]
    out_decode = torch.cat(steps, dim=1)

    assert torch.equal(cs, cs_dec)  # conv path: bit-exact contract
    assert torch.allclose(out_chunk, out_decode, atol=OUT_ATOL, rtol=0)
    assert torch.allclose(s, s_dec, atol=STATE_ATOL, rtol=0)


def test_forget_gate_respects_lower_bound():
    """g stays inside (gate_lower_bound, 0) — parameterization, not a clamp."""
    layer, cfg = _build(seed=4)
    batch_size, seq_len = 2, 8

    x = _input(cfg, batch_size, seq_len) * 5.0
    g = layer.forget_gate(x)
    assert g.dtype == torch.float32
    assert g.shape == (batch_size, seq_len, cfg.linear_num_heads, cfg.linear_head_dim)
    assert (g > cfg.gate_lower_bound).all()
    assert (g < 0).all()

    # Saturating inputs may pin the fp32 sigmoid to exactly 0 or 1; the
    # bound must still hold closed — and never overshoot.
    x_big = _input(cfg, batch_size, seq_len) * 1e4
    g_big = layer.forget_gate(x_big)
    assert (g_big >= cfg.gate_lower_bound).all()
    assert (g_big <= 0).all()
    # Per-token decay exp(g) in (exp(lower_bound), 1].
    assert (g_big.exp() <= 1.0).all()
    assert (g_big.exp() >= torch.tensor(cfg.gate_lower_bound).exp()).all()


def test_left_padding_mask_is_inert():
    """Masked (left-pad) garbage tokens must not touch conv windows or S."""
    layer, cfg = _build(seed=5)
    batch_size, seq_len, pad = 1, 12, 5
    x = _input(cfg, batch_size, seq_len)

    out_ref, s_ref, cs_ref = layer.prefill(x)

    garbage = torch.randn(batch_size, pad, cfg.hidden_size, dtype=torch.bfloat16) * 3.0
    x_padded = torch.cat([garbage, x], dim=1)
    mask = torch.cat(
        [
            torch.zeros(batch_size, pad, dtype=torch.bool),
            torch.ones(batch_size, seq_len, dtype=torch.bool),
        ],
        dim=1,
    )
    out_pad, s_pad, cs_pad = layer.prefill(x_padded, attention_mask=mask)

    assert torch.equal(cs_pad, cs_ref)  # tail covers only real tokens
    assert torch.allclose(out_pad[:, pad:], out_ref, atol=OUT_ATOL, rtol=0)
    assert torch.allclose(s_pad, s_ref, atol=STATE_ATOL, rtol=0)


def test_shapes_and_dtypes_at_real_config_dims():
    """Full-size geometry (4096 hidden, 64x128 heads, 24576 conv channels)."""
    torch.manual_seed(6)
    cfg = Glm5NextKdaConfig()
    layer = Glm5NextLinearAttention(cfg, dtype=torch.bfloat16)
    layer.requires_grad_(False)
    batch_size, seq_len = 1, 5
    x = torch.randn(batch_size, seq_len, cfg.hidden_size, dtype=torch.bfloat16) * 0.05

    # Checkpoint-facing parameter geometry.
    assert layer.conv1d.weight.shape == (24576, 1, 4)
    assert layer.conv1d.weight.dtype == torch.float32
    assert layer.forget_gate.A_log.shape == (64,)
    assert layer.forget_gate.A_log.dtype == torch.float32
    assert layer.forget_gate.dt_bias.shape == (8192,)
    assert layer.forget_gate.dt_bias.dtype == torch.float32
    assert layer.o_norm.weight.shape == (128,)
    assert layer.q_proj.weight.shape == (8192, 4096)
    assert layer.o_proj.weight.shape == (4096, 8192)

    out, s, cs = layer.prefill(x)
    assert out.shape == (batch_size, seq_len, 4096)
    assert out.dtype == torch.bfloat16
    assert s.shape == (batch_size, 64, 128, 128)  # (heads, k-dim, v-dim)
    assert s.dtype == torch.float32
    assert cs.shape == (batch_size, 24576, 3)
    assert cs.dtype == torch.bfloat16

    # Decode step: fixed output shape, states updated in place (no realloc).
    s_ptr, cs_ptr = s.data_ptr(), cs.data_ptr()
    step = layer.decode_step(x[:, :1], s, cs)
    assert step.shape == (batch_size, 1, 4096)
    assert step.dtype == torch.bfloat16
    assert s.data_ptr() == s_ptr
    assert cs.data_ptr() == cs_ptr
