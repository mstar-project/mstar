"""The per-layer plan from a node's spec, its model's declaration and its checkpoint, and the specs it refuses
at boot. One rank's root is built on the meta device at the deployment's TP size, as the boot check does."""

import pytest
import torch
from quant_helpers import EXPERTS, HEADS, HIDDEN, INTER, KV_HEADS, LAYERS, TinyRoot, declaration, group

from mstar.model.components.quant_linear import QuantizableLinears
from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo
from mstar.quantization.resolve import Action, resolve_plan
from mstar.quantization.spec import parse_quantization

PER_LAYER = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "moe.experts")


def meta_root(tp=1):
    with torch.device("meta"):
        return TinyRoot(group(0, tp))


def plan_for(quantization, ckpt=None, tp=1, capability=None, linears=None):
    decl = declaration(None) if linears is None else declaration(None, linears)
    return resolve_plan(meta_root(tp), decl, parse_quantization(quantization).for_node("LLM"), ckpt, capability)


def fp8_checkpoint(fp8=lambda base: True) -> CheckpointQuant:
    """Headers of the tiny model's checkpoint with the bases ``fp8`` picks in block fp8."""
    shapes = {"lm_head": (384, HIDDEN)}
    for i in range(LAYERS):
        p = f"layers.{i}"
        shapes |= {f"{p}.self_attn.q_proj": (HEADS * 128, HIDDEN), f"{p}.self_attn.k_proj": (KV_HEADS * 128, HIDDEN),
                   f"{p}.self_attn.v_proj": (KV_HEADS * 128, HIDDEN), f"{p}.self_attn.o_proj": (HIDDEN, HEADS * 128),
                   f"{p}.mlp.gate_proj": (INTER, HIDDEN), f"{p}.mlp.up_proj": (INTER, HIDDEN),
                   f"{p}.mlp.down_proj": (HIDDEN, INTER)}
        for e in range(EXPERTS):
            shapes |= {f"{p}.moe.experts.{e}.gate_proj": (INTER, HIDDEN),
                       f"{p}.moe.experts.{e}.up_proj": (INTER, HIDDEN),
                       f"{p}.moe.experts.{e}.down_proj": (HIDDEN, INTER)}
    tensors = {}
    for base, (n, k) in shapes.items():
        if fp8(base):
            tensors[f"{base}.weight"] = TensorInfo("F8_E4M3", (n, k))
            tensors[f"{base}.weight_scale_inv"] = TensorInfo("F32", (-(-n // 128), -(-k // 128)))
        else:
            tensors[f"{base}.weight"] = TensorInfo("BF16", (n, k))
    return CheckpointQuant.from_tensors(tensors, {"quant_method": "fp8", "weight_block_size": [128, 128]})


def actions(plan):
    return {path: (layer.runtime, layer.action) for path, layer in plan.layers.items()}


def test_a_bf16_checkpoint_quantizes_every_candidate_once_loaded():
    plan = plan_for("fp8_block")
    assert not plan.problems and not plan.notes
    for i in range(LAYERS):
        for leaf in PER_LAYER:
            assert actions(plan)[f"layers.{i}.{leaf}"] == ("fp8_block", Action.QUANTIZE)
    assert plan.layers["lm_head"].action is Action.KEEP and plan.layers["lm_head"].origin == "model"
    assert plan.layers["layers.0.moe.experts"].kind == "experts"


def test_no_block_and_a_bf16_checkpoint_change_nothing():
    plan = plan_for(None)
    assert not plan.active and not plan.problems


def test_rules_go_first_and_say_where_they_came_from():
    plan = plan_for({"format": "fp8_block", "layers": [{"match": "layers.0.**", "format": "bf16"},
                                                       {"match": "layers.*.self_attn.*", "format": "fp8_channel"}]})
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("bf16", Action.KEEP)
    assert actions(plan)["layers.1.self_attn.qkv_proj"] == ("fp8_channel", Action.QUANTIZE)
    assert plan.layers["layers.1.self_attn.qkv_proj"].origin == "quantization.layers[1]"
    assert actions(plan)["layers.1.mlp.down_proj"] == ("fp8_block", Action.QUANTIZE)


def test_a_pre_quantized_checkpoint_serves_as_stored_by_default():
    plan = plan_for(None, fp8_checkpoint(lambda base: base != "lm_head"))
    assert not plan.problems
    assert actions(plan)["layers.1.self_attn.qkv_proj"] == ("fp8_block", Action.DIRECT)
    assert actions(plan)["layers.1.moe.experts"] == ("fp8_block", Action.DIRECT)
    assert plan.layers["layers.1.moe.experts"].block == (128, 128)


def test_keeping_a_stored_layer_bf16_dequantizes_it_as_it_loads():
    plan = plan_for({"keep_bf16": ["layers.0.**"]}, fp8_checkpoint(lambda base: base != "lm_head"))
    assert actions(plan)["layers.0.mlp.gate_up_proj"] == ("bf16", Action.DEQUANT)
    assert actions(plan)["layers.1.mlp.gate_up_proj"] == ("fp8_block", Action.DIRECT)


def test_a_stored_linear_the_model_keeps_bf16_is_dequantized():
    plan = plan_for(None, fp8_checkpoint())
    assert actions(plan)["lm_head"] == ("bf16", Action.DEQUANT)


def test_a_fused_layer_stored_half_quantized_is_dequantized_or_requantized():
    # Qwen3.5-27B-FP8 keeps in_proj_a/b bf16 beside fp8 in_proj_qkv/z, all one fused GEMM here
    mixed = fp8_checkpoint(lambda base: base.endswith("q_proj"))
    plan = plan_for(None, mixed)
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("bf16", Action.DEQUANT)
    assert any("different formats" in note for note in plan.notes)
    plan = plan_for("fp8_block", mixed)
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("fp8_block", Action.REQUANT)


def test_at_tp4_the_checkpoint_blocks_straddle_and_the_shard_is_requantized():
    plan = plan_for(None, fp8_checkpoint(lambda base: base != "lm_head"), tp=4)
    assert not plan.problems
    # gate and up are 64 rows a rank: a 128-row scale block covers both, so the loaded bf16 is re-quantized
    assert actions(plan)["layers.0.mlp.gate_up_proj"] == ("fp8_block", Action.REQUANT)
    # down's K and the experts' width are 64 a rank: no 128 group fits, they serve bf16
    assert actions(plan)["layers.0.mlp.down_proj"] == ("bf16", Action.DEQUANT)
    assert actions(plan)["layers.0.moe.experts"] == ("bf16", Action.DEQUANT)
    assert actions(plan)["layers.0.self_attn.o_proj"] == ("fp8_block", Action.DIRECT)


def test_a_default_that_doesnt_fit_a_layer_is_a_note_a_rule_is_a_problem():
    plan = plan_for("fp8_block", tp=4)
    assert actions(plan)["layers.0.mlp.down_proj"] == ("bf16", Action.KEEP)
    assert any("layers.0.mlp.down_proj: stays bf16" in note for note in plan.notes)
    plan = plan_for({"layers": [{"match": "layers.0.mlp.down_proj", "format": "fp8_block"}]}, tp=4)
    assert len(plan.problems) == 1 and "whole 128x128 blocks" in plan.problems[0]


@pytest.mark.parametrize("quantization, ckpt, capability, message", [
    ({"layers": [{"match": "lm_head", "format": "fp8_block"}]}, None, None, "which the model keeps bf16"),
    ({"layers": [{"match": "layers.*.attn.*", "format": "bf16"}]}, None, None, "matches none of its"),
    ("fp8_channel", "fp8", None, "compounds two roundings"),
    ("fp8_block", None, (8, 0), "needs compute capability 8.9"),
    ("mxfp4", None, None, "serves mxfp4 on no linear layer yet"),
    ({"layers": [{"match": "layers.*.moe.experts", "format": "fp8_channel"}]}, None, None,
     "serves fp8_channel on no experts layer"),
])
def test_specs_the_build_cant_serve_are_refused(quantization, ckpt, capability, message):
    plan = plan_for(quantization, fp8_checkpoint() if ckpt else None, capability=capability)
    assert any(message in problem for problem in plan.problems), plan.problems


def test_a_checkpoint_format_without_a_loader_is_refused_by_name():
    tensors = {"layers.0.mlp.gate_proj.weight_packed": TensorInfo("I32", (INTER, HIDDEN // 8)),
               "layers.0.mlp.gate_proj.weight_scale": TensorInfo("BF16", (INTER, HIDDEN // 32))}
    plan = plan_for(None, CheckpointQuant.from_tensors(tensors, {"quant_method": "compressed-tensors"}))
    assert any("as w4a16" in p and "no loader for w4a16" in p for p in plan.problems), plan.problems


def test_a_linear_the_model_didnt_declare_is_a_problem_not_a_silent_skip():
    plan = plan_for("fp8_block", linears=QuantizableLinears(candidates=("layers.*.mlp.*",), keep_bf16=("lm_head",)))
    assert any("neither a low-precision candidate nor kept bf16" in p for p in plan.problems)
