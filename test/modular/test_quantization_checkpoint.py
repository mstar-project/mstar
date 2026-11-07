"""What a checkpoint stores, per linear, from its headers: the formats vLLM and SGLang read are recognised by
their tensors, whatever the config's ignore list says."""

import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.quantization.checkpoint import (
    CheckpointQuant,
    StoredBase,
    TensorInfo,
    classify,
    read_headers,
    read_quant_config,
    scan_checkpoint,
    split_name,
)
from mstar.utils.fp8_block import FP8_DTYPE


def T(dtype, *shape):
    return TensorInfo(dtype, tuple(shape))


@pytest.mark.parametrize("tensors, method, stored", [
    ({"weight": T("BF16", 256, 512)}, "", StoredBase("bf16")),
    # DeepSeek / Qwen / GLM / MiniMax `quant_method: fp8` block scales; the block comes from the shapes
    ({"weight": T("F8_E4M3", 256, 512), "weight_scale_inv": T("F32", 2, 4)}, "fp8",
     StoredBase("fp8_block", "fp8", (128, 128))),
    # a partial last block rounds up
    ({"weight": T("F8_E4M3", 200, 512), "weight_scale_inv": T("F32", 2, 4)}, "fp8",
     StoredBase("fp8_block", "fp8", (128, 128))),
    # compressed-tensors float-quantized: per channel, per tensor
    ({"weight": T("F8_E4M3", 256, 512), "weight_scale": T("F32", 256, 1)}, "compressed-tensors",
     StoredBase("fp8_channel", "compressed-tensors", (1, 0))),
    ({"weight": T("F8_E4M3", 256, 512), "weight_scale": T("F32", 1), "input_scale": T("F32", 1)}, "modelopt",
     StoredBase("fp8_tensor", "modelopt")),
    # MiniMax-M3-MXFP8: `quant_method: mxfp8`, e8m0 scales per 32
    ({"weight": T("F8_E4M3", 256, 512), "weight_scale_inv": T("U8", 256, 16)}, "mxfp8",
     StoredBase("mxfp8", "mxfp8", (1, 32))),
    # modelopt NVFP4 (nvidia/*-NVFP4)
    ({"weight": T("U8", 256, 256), "weight_scale": T("F8_E4M3", 256, 32), "weight_scale_2": T("F32"),
      "input_scale": T("F32")}, "modelopt", StoredBase("nvfp4", "modelopt", (1, 16))),
    # compressed-tensors pack-quantized int4 (Kimi K2.x, RedHat w4a16) and mxfp4 (Kimi K3)
    ({"weight_packed": T("I32", 256, 64), "weight_scale": T("BF16", 256, 16), "weight_shape": T("I64", 2)},
     "compressed-tensors", StoredBase("w4a16", "compressed-tensors", (1, 32)).with_params(
         group=32, symmetric=True, scale_dtype="BF16")),
    ({"weight_packed": T("U8", 256, 256), "weight_scale": T("U8", 256, 16)}, "compressed-tensors",
     StoredBase("mxfp4", "compressed-tensors", (1, 32))),
    ({"weight_packed": T("U8", 256, 256), "weight_scale": T("F8_E4M3", 256, 32),
      "weight_global_scale": T("F32", 1)}, "compressed-tensors", StoredBase("nvfp4", "compressed-tensors", (1, 16))),
    ({"qweight": T("I32", 512, 32), "qzeros": T("I32", 4, 32), "scales": T("F16", 4, 256)}, "awq",
     StoredBase("w4a16", "awq", (1, 128)).with_params(group=128, symmetric=False, scale_dtype="F16")),
    ({"qweight": T("I32", 64, 256), "qzeros": T("I32", 4, 32), "scales": T("F16", 4, 256),
      "g_idx": T("I32", 512)}, "gptq", StoredBase("w4a16", "gptq", (1, 128)).with_params(
          group=128, symmetric=True, scale_dtype="F16", zero_offset=1)),
    # e4m3 with no scale anywhere: not something to load as bytes
    ({"weight": T("F8_E4M3", 256, 512)}, "fp8", StoredBase("unknown", "fp8")),
])
def test_a_linear_is_classified_by_its_tensors(tensors, method, stored):
    assert classify(tensors, method) == stored


def test_suffixes_split_longest_first():
    assert split_name("a.b.weight_scale_inv") == ("a.b", "weight_scale_inv")
    assert split_name("a.b.weight_scale") == ("a.b", "weight_scale")
    assert split_name("a.b.weight") == ("a.b", "weight")
    assert split_name("a.b.inv_freq") == ("a.b.inv_freq", "")


def test_the_ignore_list_is_the_tensors_themselves():
    # Qwen3.5-27B-FP8 keeps in_proj_a bf16 (it is in modules_to_not_convert) beside fp8 in_proj_qkv
    tensors = {
        "model.layers.0.linear_attn.in_proj_qkv.weight": T("F8_E4M3", 1024, 512),
        "model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv": T("F32", 8, 4),
        "model.layers.0.linear_attn.in_proj_a.weight": T("BF16", 48, 512),
        "model.layers.0.input_layernorm.weight": T("BF16", 512),
    }
    ckpt = CheckpointQuant.from_tensors(tensors, {"quant_method": "fp8", "weight_block_size": [128, 128]})
    assert ckpt.bases["model.layers.0.linear_attn.in_proj_qkv"].format == "fp8_block"
    assert ckpt.bases["model.layers.0.linear_attn.in_proj_a"].format == "bf16"
    assert ckpt.quantized and ckpt.block == (128, 128)


def test_headers_and_quant_config_are_read_without_the_data(tmp_path):
    save_file({
        "a.weight": torch.zeros(256, 128, dtype=torch.bfloat16).to(FP8_DTYPE),
        "a.weight_scale_inv": torch.ones(2, 1),
        "b.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}}))
    headers = read_headers(tmp_path)
    assert headers["a.weight"] == TensorInfo("F8_E4M3", (256, 128))
    assert headers["a.weight_scale_inv"] == TensorInfo("F32", (2, 1))
    assert read_quant_config(tmp_path)["quant_method"] == "fp8"
    ckpt = scan_checkpoint(tmp_path)
    assert ckpt.bases["a"] == StoredBase("fp8_block", "fp8", (128, 128))
    assert ckpt.bases["b"].format == "bf16"


def test_modelopt_declares_itself_in_its_own_file(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "NVFP4"}}))
    assert read_quant_config(tmp_path) == {"quant_method": "modelopt", "quant_algo": "NVFP4"}
