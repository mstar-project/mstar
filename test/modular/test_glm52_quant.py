import pytest
import torch

from mstar.model.glm52._testing import fake_quantize_fp8_block
from mstar.model.glm52.quantization import (
    FP8_DTYPE,
    Fp8BlockQuantConfig,
    dequant_fp8_block_stream,
    dequantize_fp8_block_weight,
)


def test_dequantize_single_block_known_answer():
    w = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(FP8_DTYPE)  # exact in e4m3
    scale_inv = torch.tensor([[2.0]])
    got = dequantize_fp8_block_weight(
        w, scale_inv, block_size=(2, 2), out_dtype=torch.float32,
    )
    assert torch.equal(got, torch.tensor([[2.0, -4.0], [1.0, 8.0]]))


def test_dequantize_block_broadcast():
    w = torch.ones(4, 4).to(FP8_DTYPE)
    scale_inv = torch.tensor([[3.0, 5.0], [7.0, 11.0]])  # 2x2 blocks of 2x2
    got = dequantize_fp8_block_weight(
        w, scale_inv, block_size=(2, 2), out_dtype=torch.float32,
    )
    assert got[0, 0] == 3.0 and got[0, 3] == 5.0
    assert got[3, 1] == 7.0 and got[2, 2] == 11.0


def test_dequantize_ragged_tail_blocks():
    # 5x6 with 4x4 blocks -> 2x2 scale grid; tail blocks are cropped.
    w = torch.ones(5, 6).to(FP8_DTYPE)
    scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = dequantize_fp8_block_weight(
        w, scale_inv, block_size=(4, 4), out_dtype=torch.float32,
    )
    assert got.shape == (5, 6)
    assert got[0, 0] == 1.0 and got[0, 5] == 2.0
    assert got[4, 0] == 3.0 and got[4, 5] == 4.0


def test_dequantize_shape_mismatch_raises():
    w = torch.ones(4, 4).to(FP8_DTYPE)
    with pytest.raises(ValueError, match="scale_inv shape"):
        dequantize_fp8_block_weight(w, torch.ones(3, 3), block_size=(2, 2))


def test_dequantize_accepts_uint8_view():
    torch.manual_seed(0)
    w_fp8, scale_inv, deq = fake_quantize_fp8_block(
        torch.randn(8, 8) * 0.1, block_size=(4, 4),
    )
    got = dequantize_fp8_block_weight(
        w_fp8.view(torch.uint8), scale_inv, block_size=(4, 4),
    )
    assert torch.equal(got, deq)


def test_fake_quantize_dequantize_exact():
    torch.manual_seed(1)
    w = torch.randn(12, 32) * 0.1
    w_fp8, scale_inv, deq = fake_quantize_fp8_block(w, block_size=(4, 16))
    assert w_fp8.dtype == FP8_DTYPE
    assert scale_inv.shape == (3, 2)
    got = dequantize_fp8_block_weight(w_fp8, scale_inv, block_size=(4, 16))
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, deq)
    # e4m3 has ~2 decimal digits; block scaling keeps error small.
    assert (got.float() - w).abs().max() < 0.05


def test_stream_pairs_dequantizes_and_passes_through():
    torch.manual_seed(2)
    cfg = Fp8BlockQuantConfig(weight_block_size=(4, 4))
    a_fp8, a_scale, a_deq = fake_quantize_fp8_block(torch.randn(8, 8) * 0.1, (4, 4))
    b_fp8, b_scale, b_deq = fake_quantize_fp8_block(torch.randn(4, 8) * 0.1, (4, 4))
    norm = torch.randn(8)

    # Pairing is by base name, independent of order (scale-first included);
    # bf16 keys (modules_to_not_convert) pass straight through.
    stream = [
        ("l.0.a_proj.weight_scale_inv", a_scale),
        ("l.0.norm.weight", norm),
        ("l.0.b_proj.weight", b_fp8),
        ("l.0.a_proj.weight", a_fp8),
        ("l.0.b_proj.weight_scale_inv", b_scale),
    ]
    out = dict(dequant_fp8_block_stream(iter(stream), cfg))

    assert set(out) == {"l.0.a_proj.weight", "l.0.b_proj.weight", "l.0.norm.weight"}
    assert torch.equal(out["l.0.a_proj.weight"], a_deq)
    assert torch.equal(out["l.0.b_proj.weight"], b_deq)
    assert torch.equal(out["l.0.norm.weight"], norm)


def test_stream_keep_fp8_passthrough():
    cfg = Fp8BlockQuantConfig(weight_block_size=(4, 4))
    exp_base = "model.layers.1.mlp.experts.3.gate_proj"
    mla_base = "model.layers.1.self_attn.o_proj"
    e_fp8, e_scale, _ = fake_quantize_fp8_block(torch.randn(8, 8) * 0.1, (4, 4))
    m_fp8, m_scale, m_deq = fake_quantize_fp8_block(torch.randn(4, 8) * 0.1, (4, 4))

    def keep_fp8(base):
        return ".experts." in base

    stream = [
        (f"{exp_base}.weight", e_fp8),
        (f"{exp_base}.weight_scale_inv", e_scale),
        (f"{mla_base}.weight", m_fp8),
        (f"{mla_base}.weight_scale_inv", m_scale),
    ]
    out = dict(dequant_fp8_block_stream(iter(stream), cfg, keep_fp8=keep_fp8))

    assert out[f"{exp_base}.weight"].dtype == FP8_DTYPE
    assert f"{exp_base}.weight_scale_inv" in out
    assert torch.equal(out[f"{mla_base}.weight"], m_deq)
    assert f"{mla_base}.weight_scale_inv" not in out


def test_stream_unpaired_raises():
    cfg = Fp8BlockQuantConfig(weight_block_size=(4, 4))
    w_fp8, _, _ = fake_quantize_fp8_block(torch.randn(4, 4) * 0.1, (4, 4))
    with pytest.raises(ValueError, match="unpaired"):
        list(dequant_fp8_block_stream(iter([("x.proj.weight", w_fp8)]), cfg))
    with pytest.raises(ValueError, match="unpaired"):
        list(dequant_fp8_block_stream(
            iter([("x.proj.weight_scale_inv", torch.ones(1, 1))]), cfg,
        ))


def test_quant_config_from_hf_dict():
    raw = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.embed_tokens", "lm_head", "re:.*gate$"],
    }
    cfg = Fp8BlockQuantConfig.from_hf_config_dict(raw)
    assert cfg is not None
    assert cfg.weight_block_size == (128, 128)
    assert cfg.fmt == "e4m3"
    assert cfg.ignore == ("model.embed_tokens", "lm_head", "re:.*gate$")
    assert Fp8BlockQuantConfig.from_hf_config_dict(None) is None
    assert Fp8BlockQuantConfig.from_hf_config_dict({}) is None
    # A compressed-tensors dict must not parse as fp8.
    assert Fp8BlockQuantConfig.from_hf_config_dict(
        {"quant_method": "compressed-tensors"},
    ) is None
