"""CPU unit tests for the Kimi-K2.7 compressed-tensors dequant utilities (dequant-on-load).

These pin the *numerics and bit layout* of the dequant-on-load parser without a
GPU or checkpoint — the cheapest level that guards the correctness harness:

  1. known-answer pack/unpack (the exact int32 bit layout, incl. the sign-bit
     nibble), so a future refactor can't silently change the on-disk convention;
  2. pack/unpack round-trip on random nibbles;
  3. symmetric dequant math (offset-binary ``(nibble - bias) * scale``);
  4. ``fake_quantize_weight`` -> ``dequantize_weight`` exactness (the golden
     harness relies on the loader reproducing the fake-quant result bit-for-bit);
  5. the streaming generator: quant components collapse to one bf16 ``*.weight``,
     non-quant keys pass through, incomplete groups raise;
  6. ``CompressedTensorsQuantConfig.from_hf_config_dict`` parsing.

The full weight-loading + forward golden (needs the fused-expert GEMM + RMSNorm)
lives in ``test/integration/test_kimi_quant_weight_loading.py``.

Run:  pytest test/modular/test_kimi_quant.py -v
"""
import pytest
import torch

from mstar.model.kimi_k2_7._testing import fake_quantize_weight
from mstar.model.kimi_k2_7.quantization import (
    CompressedTensorsQuantConfig,
    dequant_compressed_tensors_stream,
    dequantize_weight,
    pack_int32,
    unpack_int32,
)

# --------------------------------------------------------------------------
# 1. Known-answer pack/unpack — pins the int32 bit layout.
# --------------------------------------------------------------------------

def test_pack_known_answer():
    # Eight INT4 nibbles 0..7 along the last axis pack low-order-first:
    #   sum(j << 4*j for j in 0..7) == 0x76543210.
    nibbles = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    packed = pack_int32(nibbles, num_bits=4)
    assert packed.dtype == torch.int32
    assert packed.shape == (1, 1)
    assert packed.item() == 0x76543210

    # A top nibble >= 8 sets bit 31, so the int32 container is negative — the
    # unpack must still recover it (reads the 32-bit pattern as unsigned).
    top = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 8]], dtype=torch.int64)
    packed_top = pack_int32(top, num_bits=4)
    assert packed_top.item() == -(2**31)  # 0x80000000 as signed int32
    back = unpack_int32(packed_top, num_bits=4)
    assert torch.equal(back, top)


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    nibbles = torch.randint(0, 16, (5, 32), dtype=torch.int64)  # in=32 -> packed 4
    packed = pack_int32(nibbles, num_bits=4)
    assert packed.shape == (5, 4)
    assert torch.equal(unpack_int32(packed, num_bits=4), nibbles)


# --------------------------------------------------------------------------
# 2. Dequant math — symmetric offset-binary (nibble - bias) * scale.
# --------------------------------------------------------------------------

def test_dequantize_symmetric_known_answer():
    # One row, one group of 8 (group_size=8). Unsigned nibbles minus bias 8 give
    # the signed quantized values, times a per-group scale of 2.0.
    unsigned = torch.tensor([[8, 9, 7, 8, 10, 6, 8, 8]], dtype=torch.int64)
    signed = torch.tensor([[0, 1, -1, 0, 2, -2, 0, 0]], dtype=torch.float32)
    packed = pack_int32(unsigned, num_bits=4)
    scale = torch.tensor([[2.0]])  # (out=1, groups=1)
    got = dequantize_weight(
        packed, scale, num_bits=4, group_size=8, symmetric=True,
        out_dtype=torch.float32,
    )
    assert torch.equal(got, signed * 2.0)


def test_dequantize_two_groups_broadcast():
    # in=16, group_size=8 -> two groups with distinct scales; check the scale
    # broadcasts per-group along the input axis.
    unsigned = torch.full((1, 16), 8, dtype=torch.int64)
    unsigned[0, 0] = 9   # +1 in group 0
    unsigned[0, 8] = 9   # +1 in group 1
    packed = pack_int32(unsigned, num_bits=4)
    scale = torch.tensor([[3.0, 5.0]])  # group0=3, group1=5
    got = dequantize_weight(
        packed, scale, num_bits=4, group_size=8, symmetric=True,
        out_dtype=torch.float32,
    )
    assert got[0, 0] == 3.0
    assert got[0, 8] == 5.0
    assert got[0, 1] == 0.0


# --------------------------------------------------------------------------
# 3. fake_quantize -> dequantize exactness (the golden's core invariant).
# --------------------------------------------------------------------------

@pytest.mark.parametrize("group_size", [8, 16, -1])
def test_fake_quantize_dequantize_exact(group_size):
    torch.manual_seed(1)
    w = torch.randn(12, 32) * 0.1
    packed, scale, deq = fake_quantize_weight(
        w, num_bits=4, group_size=group_size, symmetric=True,
    )
    # Reconstructing from the on-disk tensors must reproduce the fake-quant bf16
    # result bit-for-bit (both compute q*scale in fp32 then cast to bf16).
    got = dequantize_weight(
        packed, scale, num_bits=4, group_size=group_size, symmetric=True,
    )
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, deq)
    # And the quantization is lossy but bounded (sanity: close to the original).
    assert (got.float() - w).abs().max() < 0.05


# --------------------------------------------------------------------------
# 4. The dequant-on-load streaming generator.
# --------------------------------------------------------------------------

def _quant_components(base, w, cfg):
    # Store the scale in bf16 (as a real compressed-tensors checkpoint does); the
    # returned dequant is derived from that same bf16 scale, so it matches the
    # stream's reconstruction bit-for-bit.
    packed, scale, deq = fake_quantize_weight(
        w, num_bits=cfg.num_bits, group_size=cfg.group_size,
        symmetric=cfg.symmetric, scale_dtype=torch.bfloat16,
    )
    return {
        f"{base}.weight_packed": packed,
        f"{base}.weight_scale": scale,
        f"{base}.weight_shape": torch.tensor(list(w.shape), dtype=torch.int64),
    }, deq


def test_stream_dequantizes_and_passes_through():
    torch.manual_seed(2)
    cfg = CompressedTensorsQuantConfig(num_bits=4, group_size=16, symmetric=True)
    w_a = torch.randn(8, 32) * 0.1
    w_b = torch.randn(4, 16) * 0.1
    comp_a, deq_a = _quant_components("layer.0.a_proj", w_a, cfg)
    comp_b, deq_b = _quant_components("layer.0.b_proj", w_b, cfg)
    norm = torch.randn(8)  # a non-quant key that must pass straight through

    # Interleave/shuffle keys — the generator must reassemble by base name,
    # independent of order.
    stream = [
        ("layer.0.a_proj.weight_scale", comp_a["layer.0.a_proj.weight_scale"]),
        ("layer.0.norm.weight", norm),
        ("layer.0.b_proj.weight_packed", comp_b["layer.0.b_proj.weight_packed"]),
        ("layer.0.a_proj.weight_shape", comp_a["layer.0.a_proj.weight_shape"]),
        ("layer.0.a_proj.weight_packed", comp_a["layer.0.a_proj.weight_packed"]),
        ("layer.0.b_proj.weight_scale", comp_b["layer.0.b_proj.weight_scale"]),
    ]
    out = dict(dequant_compressed_tensors_stream(iter(stream), cfg))

    # Exactly: two dequantized *.weight keys + the passthrough norm; no quant subkeys.
    assert set(out) == {"layer.0.a_proj.weight", "layer.0.b_proj.weight", "layer.0.norm.weight"}
    assert torch.equal(out["layer.0.a_proj.weight"], deq_a)
    assert torch.equal(out["layer.0.b_proj.weight"], deq_b)
    assert torch.equal(out["layer.0.norm.weight"], norm)


def test_stream_incomplete_group_raises():
    cfg = CompressedTensorsQuantConfig(num_bits=4, group_size=16, symmetric=True)
    w = torch.randn(4, 16) * 0.1
    comp, _ = _quant_components("x.proj", w, cfg)
    # Only the packed tensor, no scale -> the group can never complete.
    stream = [("x.proj.weight_packed", comp["x.proj.weight_packed"])]
    with pytest.raises(ValueError, match="incomplete"):
        list(dequant_compressed_tensors_stream(iter(stream), cfg))


# --------------------------------------------------------------------------
# 4b. dequant-on-load + packed-expert coexistence: keep_packed passes routed experts
#     through raw while MLA/dense keys still dequantize (the streaming half of the
#     mixed-load path; the GPU golden proves the packed params then load + run).
# --------------------------------------------------------------------------

def test_stream_keep_packed_passthrough():
    cfg = CompressedTensorsQuantConfig(num_bits=4, group_size=16, symmetric=True)
    exp_base = "model.layers.1.mlp.experts.3.gate_proj"   # a routed expert -> packed experts
    mla_base = "model.layers.1.self_attn.o_proj"          # an MLA weight  -> dequant-on-load
    comp_exp, _ = _quant_components(exp_base, torch.randn(8, 32) * 0.1, cfg)
    comp_mla, deq_mla = _quant_components(mla_base, torch.randn(4, 16) * 0.1, cfg)

    def keep_packed(base):
        return ".experts.3.gate_proj" in base

    # Expert carries all three sub-keys (packed/scale/shape); MLA carries the two
    # that complete a dequant (no shape) so no dangling buffer trips the end check.
    stream = [
        (f"{exp_base}.weight_packed", comp_exp[f"{exp_base}.weight_packed"]),
        (f"{exp_base}.weight_scale", comp_exp[f"{exp_base}.weight_scale"]),
        (f"{exp_base}.weight_shape", comp_exp[f"{exp_base}.weight_shape"]),
        (f"{mla_base}.weight_packed", comp_mla[f"{mla_base}.weight_packed"]),
        (f"{mla_base}.weight_scale", comp_mla[f"{mla_base}.weight_scale"]),
    ]
    out = dict(dequant_compressed_tensors_stream(iter(stream), cfg, keep_packed=keep_packed))

    # Routed-expert sub-keys pass through RAW — packed int32 + scale + shape, and
    # crucially NO collapsed ``.weight`` (they load into the packed params instead).
    assert out[f"{exp_base}.weight_packed"].dtype == torch.int32
    assert f"{exp_base}.weight_scale" in out
    assert f"{exp_base}.weight_shape" in out
    assert f"{exp_base}.weight" not in out
    # The MLA weight still collapses to one dequantized bf16 ``.weight`` (dequant-on-load).
    assert torch.equal(out[f"{mla_base}.weight"], deq_mla)
    assert f"{mla_base}.weight_packed" not in out


# --------------------------------------------------------------------------
# 4c. Pure-torch reference for the W4A16 kernel math (no GPU): the per-group
#     ``(unpack - 8) * scale`` -> bf16 grouped GEMM the Triton kernel replicates,
#     pinning the packed-K layout and the top-nibble (bit-31) sign case.
# --------------------------------------------------------------------------

def test_grouped_gemm_reference_math_and_top_nibble():
    torch.manual_seed(7)
    N, K, gs = 6, 32, 16  # two groups along the packed K axis
    # Wide init so per-group amax uses the full nibble range -> some top nibbles
    # land >= 8 (int32 container bit 31 set), exercising the sign path.
    w = torch.randn(N, K) * 0.5
    packed, scale, deq = fake_quantize_weight(
        w, num_bits=4, group_size=gs, symmetric=True, scale_dtype=torch.bfloat16,
    )
    assert packed.shape == (N, K // 8)  # packed along the last (input/K) axis
    assert (packed < 0).any(), "no negative container — top-nibble sign path untested"

    # Reproduce the kernel's in-register arithmetic in pure torch: unpack the
    # nibble, offset-binary subtract 8, scale per group (broadcast along K).
    nibbles = unpack_int32(packed, num_bits=4).to(torch.float32)  # (N, K) unsigned
    scale_bc = scale.to(torch.float32).repeat_interleave(gs, dim=-1)  # (N, K)
    manual_deq = ((nibbles - 8.0) * scale_bc).to(torch.bfloat16)
    assert torch.equal(manual_deq, deq)  # kernel math == dequantize_weight

    # Grouped GEMM equivalence: the packed path must produce the same y as feeding
    # the bf16 dequant directly (both contract over the same bf16 weight values).
    x = torch.randn(4, K)
    y_manual = torch.einsum("tk,nk->tn", x, manual_deq.float())
    y_deq = torch.einsum("tk,nk->tn", x, deq.float())
    assert torch.equal(y_manual, y_deq)


# --------------------------------------------------------------------------
# 5. Config parsing.
# --------------------------------------------------------------------------

def test_quant_config_from_hf_dict():
    raw = {
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "ignore": ["lm_head", "re:.*gate$"],
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "group_size": 32,
                    "symmetric": True,
                    "strategy": "group",
                    "type": "int",
                },
                "targets": ["Linear"],
            }
        },
    }
    cfg = CompressedTensorsQuantConfig.from_hf_config_dict(raw)
    assert cfg is not None
    assert cfg.num_bits == 4
    assert cfg.group_size == 32
    assert cfg.symmetric is True
    assert cfg.pack_factor == 8
    assert cfg.ignore == ("lm_head", "re:.*gate$")
    assert CompressedTensorsQuantConfig.from_hf_config_dict(None) is None
    assert CompressedTensorsQuantConfig.from_hf_config_dict({}) is None
