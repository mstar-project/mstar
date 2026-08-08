"""Quantization descriptors: the tagged union that replaced ``fused_experts``'
per-scheme kwargs, and the checkpoint-config -> kernel-data seam."""

import dataclasses

import pytest
import torch

from mstar.model.components.quantization import (
    CompressedTensorsQuantConfig,
    MarlinMoEMethod,
    QuantizationData,
    QuantizationType,
    W4A16Data,
)


def _data(**overrides) -> W4A16Data:
    kwargs = {
        "w1_scale": torch.ones(2, 4, 2),
        "w2_scale": torch.ones(2, 4, 2),
        "group_size": 32,
    }
    kwargs.update(overrides)
    return W4A16Data(**kwargs)


# --- the union ------------------------------------------------------------


def test_quantization_data_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        QuantizationData()


def test_w4a16_tags_itself_and_derives_pack_factor():
    d = _data()
    assert d.quant_type is QuantizationType.W4A16
    # Derived from the class's NUM_BITS, so it cannot disagree with the kernel's
    # hardcoded 4-bit nibble extraction.
    assert d.pack_factor == 8
    assert "pack_factor" not in {f.name for f in dataclasses.fields(d)}
    assert "NUM_BITS" not in {f.name for f in dataclasses.fields(d)}


def test_zero_points_are_the_symmetry_discriminant():
    assert _data().symmetric is True
    assert _data(w1_zp=torch.zeros(2, 4, 2), w2_zp=torch.zeros(2, 4, 2)).symmetric is False


def test_frozen_blocks_rebinding_but_not_tensor_mutation():
    d = _data()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.group_size = 64
    # Documented limit: frozen freezes the binding, not the pointee.
    d.w1_scale[0, 0, 0] = 7.0
    assert d.w1_scale[0, 0, 0] == 7.0


# --- the checkpoint-config -> kernel-data seam ----------------------------


def test_config_maps_num_bits_to_scheme():
    assert CompressedTensorsQuantConfig(num_bits=4).quant_type is QuantizationType.W4A16


@pytest.mark.parametrize("num_bits", [2, 8, 16])
def test_config_rejects_unimplemented_width_at_load(num_bits):
    # The whole point: an INT8 checkpoint fails here, naming the checkpoint,
    # instead of reaching a kernel that would mask its 8-bit values to nibbles.
    cfg = CompressedTensorsQuantConfig(num_bits=num_bits)
    assert cfg.quant_type is None  # the query answers; only the ensure raises
    with pytest.raises(ValueError, match="mstar does not implement"):
        cfg.ensure_kernel_support()


def test_ensure_kernel_support_returns_the_scheme():
    cfg = CompressedTensorsQuantConfig(num_bits=4)
    assert cfg.ensure_kernel_support() is QuantizationType.W4A16


def test_moe_quant_data_round_trips_group_size_and_scales():
    cfg = CompressedTensorsQuantConfig(num_bits=4, group_size=64)
    w1s, w2s = torch.ones(2, 4, 2), torch.zeros(2, 4, 2)
    data = cfg.moe_quant_data(w1s, w2s)

    assert isinstance(data, W4A16Data)
    assert data.quant_type is QuantizationType.W4A16
    assert data.group_size == 64
    assert data.pack_factor == cfg.pack_factor
    assert data.w1_scale is w1s and data.w2_scale is w2s


def test_symmetric_config_drops_zero_points():
    sym = CompressedTensorsQuantConfig(num_bits=4, symmetric=True)
    zp = torch.zeros(2, 4, 2)
    data = sym.moe_quant_data(torch.ones(2, 4, 2), torch.ones(2, 4, 2), w1_zp=zp, w2_zp=zp)
    assert data.w1_zp is None and data.w2_zp is None and data.symmetric

    asym = CompressedTensorsQuantConfig(num_bits=4, symmetric=False)
    data = asym.moe_quant_data(torch.ones(2, 4, 2), torch.ones(2, 4, 2), w1_zp=zp, w2_zp=zp)
    assert data.w1_zp is zp and not data.symmetric


def test_marlin_reads_num_bits_forward_from_the_config():
    cfg = CompressedTensorsQuantConfig(num_bits=4, group_size=128)
    method = MarlinMoEMethod.from_quant_config(cfg)
    assert method.num_bits == 4
    assert method.group_size == 128
    assert method.quant_type is QuantizationType.W4A16

    with pytest.raises(ValueError, match="mstar does not implement"):
        MarlinMoEMethod.from_quant_config(CompressedTensorsQuantConfig(num_bits=8))


# --- kernel dispatch ------------------------------------------------------


def test_fused_experts_rejects_an_unhandled_scheme():
    """A future QuantizationType with no branch must raise, not silently take
    the bf16 path — that is what the sentinel-based dispatch could not do."""
    pytest.importorskip("triton")
    from mstar.utils.fused_moe.runner import _validate_expert_shapes

    class _FutureData(QuantizationData):
        @property
        def quant_type(self):
            return "int8-someday"

    with pytest.raises(ValueError, match="no kernel for"):
        _validate_expert_shapes(
            hidden=128,
            w1=torch.zeros(2, 8, 16, dtype=torch.int32),
            w2=torch.zeros(2, 128, 1, dtype=torch.int32),
            quant=_FutureData(),
        )
