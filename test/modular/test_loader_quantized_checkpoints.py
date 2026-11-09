"""A quantized checkpoint that a model can't load is refused at load, not served as garbage.

Each case writes a small synthetic checkpoint with a real producer's tensor names and dtypes:
HF fp8, compressed-tensors, ModelOpt, AWQ and GPTQ. The model holds only unquantized weights,
as every model but GLM-5.2 does. Without the refusal, fp8 values copy into bf16 parameters
unscaled, and the scales and packed weights load into nothing.
"""
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import QKVParallelLinear
from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights, load_weights
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

OUT, IN = 256, 128
F8 = torch.float8_e4m3fn


class TinyLM(nn.Module):
    """One projection and a norm under HF names, loaded the way the docs show."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16)
        self.norm = nn.Parameter(torch.zeros(IN, dtype=torch.bfloat16))

    def load_weights(self, weights):
        return load_hf_weights(self, weights)


class FusedQKV(nn.Module):
    """q/k/v fused into one parameter through the stacked rules."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = QKVParallelLinear(CommGroup.trivial(), IN, 64, 2, 1, bias=False)

    def load_weights(self, weights):
        return load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)


def _write(path, tensors, config=None, extra_files=None):
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "tiny", **(config or {})}))
    for name, body in (extra_files or {}).items():
        (path / name).write_text(json.dumps(body))
    return path


def _norm():
    return {"norm": torch.ones(IN, dtype=torch.bfloat16)}


def _fp8_block():
    return {"q_proj.weight": torch.full((OUT, IN), 0.5).to(F8),
            "q_proj.weight_scale_inv": torch.full((OUT // 128, IN // 128), 0.01), **_norm()}


def _int4_ct():
    return {"q_proj.weight_packed": torch.zeros(OUT, IN // 8, dtype=torch.int32),
            "q_proj.weight_scale": torch.ones(OUT, IN // 128, dtype=torch.bfloat16),
            "q_proj.weight_shape": torch.tensor([OUT, IN]), **_norm()}


def _fp8_channel():
    return {"q_proj.weight": torch.full((OUT, IN), 0.5).to(F8),
            "q_proj.weight_scale": torch.full((OUT, 1), 0.01, dtype=torch.bfloat16), **_norm()}


def _fp8_tensor():
    return {"q_proj.weight": torch.full((OUT, IN), 0.5).to(F8),
            "q_proj.weight_scale": torch.tensor(0.01), "q_proj.input_scale": torch.tensor(0.02),
            **_norm()}


def _nvfp4():
    return {"q_proj.weight": torch.zeros(OUT, IN // 2, dtype=torch.uint8),
            "q_proj.weight_scale": torch.ones(OUT, IN // 16).to(F8),
            "q_proj.weight_scale_2": torch.tensor(0.01), "q_proj.input_scale": torch.tensor(0.02),
            **_norm()}


def _awq():
    return {"q_proj.qweight": torch.zeros(IN, OUT // 8, dtype=torch.int32),
            "q_proj.qzeros": torch.zeros(1, OUT // 8, dtype=torch.int32),
            "q_proj.scales": torch.ones(1, OUT, dtype=torch.float16), **_norm()}


def _gptq():
    return {"q_proj.qweight": torch.zeros(IN // 8, OUT, dtype=torch.int32),
            "q_proj.qzeros": torch.zeros(1, OUT // 8, dtype=torch.int32),
            "q_proj.scales": torch.ones(1, OUT, dtype=torch.float16),
            "q_proj.g_idx": torch.zeros(IN, dtype=torch.int32), **_norm()}


# (tensors, config.json's quantization_config, other config files, the quant_method the error names)
DECLARED = {
    "hf_fp8_block": (_fp8_block, {"quant_method": "fp8", "weight_block_size": [128, 128], "fmt": "e4m3"},
                     None, "fp8"),
    "compressed_tensors_int4": (_int4_ct, {"quant_method": "compressed-tensors", "format": "pack-quantized"},
                                None, "compressed-tensors"),
    "compressed_tensors_fp8": (_fp8_channel, {"quant_method": "compressed-tensors", "format": "float-quantized"},
                               None, "compressed-tensors"),
    "modelopt_nvfp4": (_nvfp4, {"quant_method": "modelopt", "quant_algo": "NVFP4"}, None, "modelopt"),
    "modelopt_fp8_hf_quant_config": (_fp8_tensor, None, {"hf_quant_config.json": {
        "producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8"}}}, "modelopt"),
    "awq": (_awq, {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "zero_point": True},
            None, "awq"),
    "gptq_quantize_config": (_gptq, None, {"quantize_config.json": {"bits": 4, "group_size": 128,
                                                                    "desc_act": False}}, "gptq"),
}


@pytest.mark.parametrize("case", sorted(DECLARED))
def test_a_declared_quantization_is_refused(tmp_path, case):
    tensors, quant, files, method = DECLARED[case]
    _write(tmp_path, tensors(), {"quantization_config": quant} if quant else None, files)
    with pytest.raises(ValueError, match="quantized checkpoint") as err:
        load_weights(TinyLM(), tmp_path)
    assert f"quant_method {method})" in str(err.value) and "only unquantized checkpoints" in str(err.value)


@pytest.mark.parametrize("tensors", [_fp8_block, _fp8_tensor, _int4_ct, _awq])
def test_quantized_parts_no_parameter_loads_are_refused(tmp_path, tensors):
    """A checkpoint whose config doesn't declare its quantization: the scales or packed weights
    that load into nothing give it away."""
    _write(tmp_path, tensors())
    with pytest.raises(ValueError, match="part of a quantized weight"):
        load_weights(TinyLM(), tmp_path)


def test_a_fused_projections_scales_are_refused(tmp_path):
    """The stacked rules route q_proj.weight_scale_inv to qkv_proj.weight_scale_inv, which no
    parameter is."""
    tensors = {f"attn.{p}_proj.{part}": t for p, rows in (("q", 128), ("k", 64), ("v", 64)) for part, t in (
        ("weight", torch.full((rows, IN), 0.5).to(F8)), ("weight_scale_inv", torch.ones(rows // 64, 1)))}
    _write(tmp_path, tensors)
    with pytest.raises(ValueError, match="part of a quantized weight"):
        load_weights(FusedQKV(), tmp_path)


def test_the_shared_shard_loader_refuses_a_quantized_checkpoint(tmp_path):
    _write(tmp_path, _fp8_block(), {"quantization_config": DECLARED["hf_fp8_block"][1]})
    with pytest.raises(ValueError, match="quantized checkpoint"):
        load_weights_from_hf_shards(tmp_path, [ModuleAndPrefix(module=TinyLM())])


def test_an_unquantized_checkpoint_loads(tmp_path):
    q = torch.randn(OUT, IN).to(torch.bfloat16)
    _write(tmp_path, {"q_proj.weight": q, **_norm()}, {"torch_dtype": "bfloat16"})
    model = TinyLM()
    assert load_weights(model, tmp_path) == {"q_proj.weight", "norm"}
    assert torch.equal(model.q_proj.weight, q)
    out = TinyLM()
    load_weights_from_hf_shards(tmp_path, [ModuleAndPrefix(module=out)])
    assert torch.equal(out.q_proj.weight, q)


def test_a_model_reads_the_formats_it_names(tmp_path):
    """GLM-5.2's case: its loader names fp8 and reads the fp8 checkpoint itself."""
    _write(tmp_path, _fp8_block(), {"quantization_config": DECLARED["hf_fp8_block"][1]})
    keys = {k for k, _ in iter_safetensors_shards(tmp_path, quant_methods=("fp8",))}
    assert keys == set(_fp8_block())
    with pytest.raises(ValueError, match="unquantized checkpoints and quant_method awq"):
        list(iter_safetensors_shards(tmp_path, quant_methods=("awq",)))
