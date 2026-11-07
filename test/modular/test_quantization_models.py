"""The models' declarations end to end on CPU: a tiny Qwen3.5 in the checkpoint layout transformers writes, its
block-fp8 conversion, the boot check the conductor runs, and the LLM node loaded under the plan."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_qwen3_5_video import write_synthetic_checkpoint  # noqa: E402

from mstar.quantization.convert import convert_checkpoint  # noqa: E402
from mstar.quantization.resolve import Action  # noqa: E402
from mstar.quantization.scope import node_scope  # noqa: E402
from mstar.quantization.validate import quantization_problems  # noqa: E402

PLACED = {"node_groups": [{"node_names": ["LLM", "vision_encoder"], "ranks": [0]}]}
# 128-multiples, so every projection but the delta net's fused input (its b/a rows are a head each) takes block fp8
TEXT = dict(hidden_size=256, intermediate_size=256, head_dim=128, num_attention_heads=2, num_key_value_heads=1,
            linear_key_head_dim=128, linear_value_head_dim=128, tie_word_embeddings=False)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    root = write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen3_5_bf16"))
    config = json.loads((root / "config.json").read_text())
    config["text_config"].update(TEXT)
    config["tie_word_embeddings"] = False
    torch.manual_seed(0)
    hf = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config)).to(torch.bfloat16)
    hf.save_pretrained(root)
    fp8 = tmp_path_factory.mktemp("qwen3_5_fp8")
    # Qwen's own FP8 layout: the decoder's linears, but not the delta net's a/b heads
    names = convert_checkpoint(root, fp8, include=["model.language_model.layers.*"],
                               exclude=["*.in_proj_a", "*.in_proj_b"])
    assert any(n.endswith("mlp.gate_proj") for n in names) and not any(n.endswith("in_proj_a") for n in names)
    # what stayed bf16 is listed, so readers that trust the list (vLLM) don't expect fp8 there
    kept = json.loads((fp8 / "config.json").read_text())["quantization_config"]["modules_to_not_convert"]
    assert "model.language_model.layers.0.linear_attn.in_proj_a" in kept and "lm_head" in kept
    return {"bf16": root, "fp8": fp8}


def model_of(path):
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    return Qwen3_5DenseModel(str(path))


def load_llm(path, quantization=None):
    model = model_of(path)
    config = {} if quantization is None else {"quantization": quantization}
    with node_scope(model, "LLM", config) as scope:
        submodule = model.get_submodule("LLM", "cpu")
        scope.finish(submodule)
    return submodule.model, next(iter(scope.plans.values()))[1]


def test_the_boot_check_passes_a_servable_block_and_refuses_the_rest(checkpoints):
    model = model_of(checkpoints["bf16"])
    assert quantization_problems(model, {**PLACED, "quantization": "fp8_block"}) == []
    assert quantization_problems(model, {**PLACED, "quantization": {"nodes": {"vision_encoder": "fp8_block"}}}) == [
        "quantization.nodes.vision_encoder: Qwen3_5DenseModel declares no quantizable layers there",
    ]
    problems = quantization_problems(model, {**PLACED, "quantization": {
        "layers": [{"match": "model.layers.*.self_attn.in_proj_fused", "format": "fp8_block"}]}})
    assert len(problems) == 1 and "whole 128x128 blocks" in problems[0]
    fp8 = model_of(checkpoints["fp8"])
    assert quantization_problems(fp8, PLACED) == []
    problems = quantization_problems(fp8, {**PLACED, "quantization": "fp8_channel"})
    assert any("compounds two roundings" in p for p in problems)


def test_qwen3_5_quantizes_its_projections_at_load(checkpoints):
    lm, plan = load_llm(checkpoints["bf16"], "fp8_block")
    full = lm.model.layers[1]  # layer_types: [linear_attention, full_attention]
    assert full.self_attn.qkv_proj.quant_format.name == "fp8_block"
    assert full.mlp.down_proj.quant_format.name == "fp8_block"
    assert lm.model.layers[0].self_attn.out_proj.quant_format.name == "fp8_block"
    assert lm.model.layers[0].self_attn.in_proj_fused.quant_format is None
    assert plan.layers["lm_head"].action is Action.KEEP


def test_qwen3_5_loads_its_fp8_checkpoint_as_stored(checkpoints):
    direct, plan = load_llm(checkpoints["fp8"])
    assert plan.layers["model.layers.1.self_attn.qkv_proj"].action is Action.DIRECT
    assert plan.layers["model.layers.0.mlp.gate_up_proj"].action is Action.DIRECT
    # in_proj_qkv/z fp8 beside bf16 in_proj_a/b: the fused GEMM dequantizes and serves bf16
    assert plan.layers["model.layers.0.self_attn.in_proj_fused"].action is Action.DEQUANT
    fly, _ = load_llm(checkpoints["bf16"], "fp8_block")
    for name, tensor in direct.state_dict().items():
        if tensor.dtype in (torch.uint8, torch.int32):
            assert torch.equal(tensor, fly.state_dict()[name]), name
    x = torch.randn(3, TEXT["hidden_size"], dtype=torch.bfloat16)
    leaf = direct.model.layers[1].mlp.gate_up_proj
    assert torch.equal(leaf(x), fly.model.layers[1].mlp.gate_up_proj(x))


def test_qwen3_omni_routes_expert_scales_beside_their_weights():
    from mstar.model.loader.base import apply_stacked
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    omni = Qwen3OmniModel.__new__(Qwen3OmniModel)
    omni.config = SimpleNamespace(thinker_text=SimpleNamespace(num_experts=4))
    omni._THINKER_STACKED_PARAMS = []
    rules = omni._get_thinker_stacked_params()
    for suffix, param in (("weight", "gate_up_proj"), ("weight_scale_inv", "gate_up_proj_scale_inv")):
        name = Qwen3OmniModel._thinker_remap(f"model.layers.0.mlp.experts.3.up_proj.{suffix}")
        assert apply_stacked(name, rules) == (f"model.layers.0.mlp.experts.{param}", "up:3")
