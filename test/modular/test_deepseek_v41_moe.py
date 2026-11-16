"""CPU golden values from the pinned DeepSeek-V4.1-Flash inference/model.py MoE.

Reference revision: 314113f8a7dadb388215396ba5fbee2c610a04d6.
The seed, packed weights, FP8 shared weights, image mask and FP32 outputs were
computed with the official Gate, Expert and MoE classes at TP=1.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from mstar.model.components.deepseek_v41.moe import DeepSeekV41MoE, DeepSeekV41MoeConfig

GOLDEN = [
    [
        0.0967235118150711,
        0.23208311200141907,
        0.730434238910675,
        1.3717312812805176,
        -0.026181694120168686,
        -0.7595370411872864,
        0.6460593938827515,
        0.4300660490989685,
        -0.9418591856956482,
        -1.0701720714569092,
        -0.6851847767829895,
        -0.33781737089157104,
        1.16988205909729,
        0.17715857923030853,
        -0.11364889144897461,
        0.017122432589530945,
        0.5873024463653564,
        -1.0002604722976685,
        0.21678537130355835,
        0.6836243867874146,
        -0.6847501993179321,
        -0.02737884409725666,
        -0.6055359840393066,
        0.013464843854308128,
        0.5445977449417114,
        0.1688499003648758,
        0.06335538625717163,
        -0.0865279883146286,
        -0.9381409287452698,
        0.7616317272186279,
        0.3192085027694702,
        0.1876574158668518,
    ],
    [
        -0.5890516042709351,
        -1.2249109745025635,
        -0.07964470237493515,
        1.0425461530685425,
        -1.9857593774795532,
        0.6861953735351562,
        -0.416248083114624,
        -0.7702357769012451,
        -1.122851848602295,
        -0.029921308159828186,
        -1.4033737182617188,
        0.5891767740249634,
        -0.7683882713317871,
        -0.2986398935317993,
        -0.5043566823005676,
        -0.16122272610664368,
        -1.4920165538787842,
        -0.437555193901062,
        0.09592946618795395,
        1.1705094575881958,
        0.10183718800544739,
        0.9150964617729187,
        0.3728281259536743,
        1.0719263553619385,
        -0.04494299739599228,
        0.10923614352941513,
        0.805385172367096,
        -0.0581875741481781,
        0.40987467765808105,
        -0.3060453236103058,
        -0.6923273801803589,
        -1.2980892658233643,
    ],
    [
        1.3628028631210327,
        -0.5601880550384521,
        -1.3522039651870728,
        -2.0153849124908447,
        0.2649870216846466,
        -0.3949770927429199,
        -0.9805670976638794,
        -0.7608382701873779,
        -0.93145352602005,
        -0.7007978558540344,
        -1.3265718221664429,
        0.19479982554912567,
        -1.7790091037750244,
        -2.464322805404663,
        2.693314552307129,
        -0.8078276515007019,
        -2.157189130783081,
        0.7329844236373901,
        1.9423935413360596,
        2.8719615936279297,
        -0.08915194869041443,
        -2.279881477355957,
        -0.5504593253135681,
        0.11732085049152374,
        -2.2046303749084473,
        3.516132116317749,
        -1.860108494758606,
        -1.1210479736328125,
        -1.5332201719284058,
        -0.7858134508132935,
        -2.4550015926361084,
        -1.1021374464035034,
    ],
]
ROUTER_IDS = [[2, 1], [3, 0], [1, 2]]


def _loaded_moe() -> tuple[DeepSeekV41MoE, torch.Tensor, torch.Tensor]:
    model = DeepSeekV41MoE(
        3,
        DeepSeekV41MoeConfig(32, 32, 4, 2, 1.5, 10.0, vision_enabled=True),
    )
    generator = torch.Generator().manual_seed(718)
    with torch.no_grad():
        for suffix, target in (
            ("weight", model.gate.weight),
            ("bias", model.gate.e_score_correction_bias),
            ("bias_vl", model.gate.bias_vl),
        ):
            value = torch.randn(target.shape, generator=generator) * 0.1
            model.load_gate(f"layers.3.ffn.gate.{suffix}", value)
        for expert in range(4):
            for projection in ("w1", "w3", "w2"):
                packed = torch.randint(-128, 128, (32, 16), dtype=torch.int8, generator=generator)
                scale = torch.full((32, 1), 0.125).to(torch.float8_e8m0fnu)
                model.load_projection_pair(
                    f"layers.3.ffn.experts.{expert}.{projection}",
                    packed.view(torch.float4_e2m1fn_x2),
                    scale,
                )
        for projection in ("w1", "w3", "w2"):
            weight = (torch.randn((32, 32), generator=generator) * 0.1).to(torch.float8_e4m3fn)
            scale = torch.ones((1, 1)).to(torch.float8_e8m0fnu)
            model.load_projection_pair(f"layers.3.ffn.shared_experts.{projection}", weight, scale)
    model.process_weights_after_loading("cpu")
    hidden = torch.randn((3, 32), generator=generator) * 0.5
    mask = torch.tensor([False, True, False])
    return model, hidden, mask


def test_moe_matches_pinned_reference_with_image_routing() -> None:
    model, hidden, mask = _loaded_moe()
    _, ids, _ = model.gate(hidden, image_mask=mask)
    assert ids.tolist() == ROUTER_IDS
    torch.testing.assert_close(model(hidden, mask), torch.tensor(GOLDEN), rtol=1e-5, atol=1e-5)
    assert not torch.equal(model(hidden, mask), model(hidden))


def test_multiblock_moe_matches_pinned_reference_past_swiglu_clamps() -> None:
    model = DeepSeekV41MoE(
        3,
        DeepSeekV41MoeConfig(64, 96, 4, 2, 1.5, 10.0, vision_enabled=True),
    )
    generator = torch.Generator().manual_seed(718)
    with torch.no_grad():
        for suffix, target in (
            ("weight", model.gate.weight),
            ("bias", model.gate.e_score_correction_bias),
            ("bias_vl", model.gate.bias_vl),
        ):
            value = torch.randn(target.shape, generator=generator) * 0.1
            model.load_gate(f"layers.3.ffn.gate.{suffix}", value)
        for expert in range(4):
            for projection in ("w1", "w3", "w2"):
                rows, cols = (64, 96) if projection == "w2" else (96, 64)
                packed = torch.randint(-128, 128, (rows, cols // 2), dtype=torch.int8, generator=generator)
                row = torch.arange(rows)[:, None]
                block = torch.arange(cols // 32)[None, :]
                scale = (0.125 * (2.0 ** ((row + block) % 3))).to(torch.float8_e8m0fnu)
                model.load_projection_pair(
                    f"layers.3.ffn.experts.{expert}.{projection}",
                    packed.view(torch.float4_e2m1fn_x2),
                    scale,
                )
        for projection in ("w1", "w3", "w2"):
            rows, cols = (64, 96) if projection == "w2" else (96, 64)
            weight = (torch.randn((rows, cols), generator=generator) * 0.35).to(torch.float8_e4m3fn)
            row = torch.arange((rows + 31) // 32)[:, None]
            col = torch.arange((cols + 31) // 32)[None, :]
            scale = (0.5 * (2.0 ** ((row + col) % 3))).to(torch.float8_e8m0fnu)
            model.load_projection_pair(f"layers.3.ffn.shared_experts.{projection}", weight, scale)
    model.process_weights_after_loading("cpu")
    hidden = torch.randn((2, 64), generator=generator) * 3.0
    mask = torch.tensor([False, True])
    gate_up, _ = model._expert_weights(0, torch.float32)
    routed_gate, routed_up = torch.mm(hidden, gate_up.T).chunk(2, dim=-1)
    shared_gate, shared_up = model.shared_expert.gate_up_proj(hidden).chunk(2, dim=-1)
    for gate, up in ((routed_gate, routed_up), (shared_gate, shared_up)):
        assert gate.max() > model.swiglu_limit
        assert up.max() > model.swiglu_limit
        assert up.min() < -model.swiglu_limit
    _, ids, _ = model.gate(hidden, image_mask=mask)
    assert ids.tolist() == [[3, 0], [0, 3]]
    golden = json.loads((Path(__file__).parent / "data/deepseek_v41_moe_multiblock_golden.json").read_text())
    torch.testing.assert_close(model(hidden, mask), torch.tensor(golden), rtol=1e-5, atol=1e-3)


def test_loader_rejects_incomplete_and_duplicate_weights() -> None:
    config = DeepSeekV41MoeConfig(32, 32, 4, 2, 1.5, 10.0)
    model = DeepSeekV41MoE(3, config)
    with pytest.raises(ValueError, match="missing weights"):
        model.process_weights_after_loading("cpu")
    gate = torch.zeros((4, 32))
    model.load_gate("layers.3.ffn.gate.weight", gate)
    with pytest.raises(ValueError, match="duplicate"):
        model.load_gate("layers.3.ffn.gate.weight", gate)
    with pytest.raises(ValueError, match="wrong.*name"):
        model.load_gate("layers.4.ffn.gate.weight", gate)
    with pytest.raises(ValueError, match="wrong.*name"):
        model.load_projection_pair("layers.4.ffn.shared_experts.w1", gate, torch.zeros(1))


def test_config_reads_published_text_fields_and_refuses_drift() -> None:
    hf = json.loads((Path(__file__).parent / "data/deepseek_v41_pinned_config.json").read_text())
    config = DeepSeekV41MoeConfig.from_hf(hf)
    assert (config.hidden_size, config.moe_intermediate_size, config.n_routed_experts) == (5120, 2304, 384)
    assert config.vision_enabled and config.gate_temp == 1.0
    bad = copy.deepcopy(hf)
    bad["quantization_config"]["scale_fmt"] = "other"
    with pytest.raises(ValueError, match="unsupported"):
        DeepSeekV41MoeConfig.from_hf(bad)
