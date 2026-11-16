"""Regenerate the two CPU MoE goldens from DeepSeek's pinned reference.

Download ``inference/model.py`` from ``deepseek-ai/DeepSeek-V4.1-Flash`` at
revision ``314113f8a7dadb388215396ba5fbee2c610a04d6``, then run
``python test/modular/data/generate_deepseek_v41_moe_golden.py PATH_TO_MODEL_PY``.
This extracts the published Gate, Expert and MoE classes. CPU Linear is the
bias-free dense equivalent of the reference's quantized Linear; packed FP4 and
FP8 weights are dequantized before insertion. The M* converters have separate
golden tests for their byte and scale interpretation.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.deepseek_v41.moe import DeepSeekV41MoE, DeepSeekV41MoeConfig
from mstar.utils.deepseek_v41_expert_weights import dequantize_expert_fp4
from mstar.utils.fp8_block import dequantize_fp8_block_weight


def reference_moe(source: Path) -> type[nn.Module]:
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"Gate", "Expert", "MoE"}]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Linear": lambda inputs, outputs, dtype=None: nn.Linear(
            inputs,
            outputs,
            bias=False,
            dtype=dtype,
        ),
        "linear": F.linear,
        "ModelArgs": object,
        "world_size": 1,
        "rank": 0,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["MoE"]


def run_case(moe_type: type[nn.Module], multi_block: bool) -> dict:
    hidden, intermediate = (64, 96) if multi_block else (32, 32)
    args = SimpleNamespace(
        dim=hidden,
        moe_inter_dim=intermediate,
        n_shared_experts=1,
        expert_dtype=None,
        swiglu_limit=10.0,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.5,
        vision_enabled=True,
        get_moe_config=lambda layer: (4, 2),
    )
    reference = moe_type(3, args)
    ours = DeepSeekV41MoE(
        3,
        DeepSeekV41MoeConfig(hidden, intermediate, 4, 2, 1.5, 10.0, vision_enabled=True),
    )
    generator = torch.Generator().manual_seed(718)
    with torch.no_grad():
        for suffix, target in (
            ("weight", reference.gate.weight),
            ("bias", reference.gate.bias),
            ("bias_vl", reference.gate.bias_vl),
        ):
            value = torch.randn(target.shape, generator=generator) * 0.1
            target.copy_(value)
            ours.load_gate(f"layers.3.ffn.gate.{suffix}", value)
        for expert in range(4):
            for projection in ("w1", "w3", "w2"):
                rows, cols = (hidden, intermediate) if projection == "w2" else (intermediate, hidden)
                packed = torch.randint(-128, 128, (rows, cols // 2), dtype=torch.int8, generator=generator)
                if multi_block:
                    row = torch.arange(rows)[:, None]
                    block = torch.arange(cols // 32)[None, :]
                    scale = (0.125 * (2.0 ** ((row + block) % 3))).to(torch.float8_e8m0fnu)
                else:
                    scale = torch.full((rows, cols // 32), 0.125).to(torch.float8_e8m0fnu)
                dense = dequantize_expert_fp4(packed, scale, dtype=torch.float32)
                getattr(reference.experts[expert], projection).weight.copy_(dense)
                ours.load_projection_pair(
                    f"layers.3.ffn.experts.{expert}.{projection}",
                    packed.view(torch.float4_e2m1fn_x2),
                    scale,
                )
        for projection in ("w1", "w3", "w2"):
            rows, cols = (hidden, intermediate) if projection == "w2" else (intermediate, hidden)
            weight = torch.randn((rows, cols), generator=generator) * (0.35 if multi_block else 0.1)
            weight = weight.to(torch.float8_e4m3fn)
            if multi_block:
                row = torch.arange((rows + 31) // 32)[:, None]
                col = torch.arange((cols + 31) // 32)[None, :]
                scale = (0.5 * (2.0 ** ((row + col) % 3))).to(torch.float8_e8m0fnu)
            else:
                scale = torch.ones((1, 1)).to(torch.float8_e8m0fnu)
            dense = dequantize_fp8_block_weight(weight, scale, (32, 32), out_dtype=torch.float32)
            getattr(reference.shared_experts, projection).weight.copy_(dense)
            ours.load_projection_pair(f"layers.3.ffn.shared_experts.{projection}", weight, scale)
    ours.process_weights_after_loading("cpu")
    tokens = 2 if multi_block else 3
    hidden_states = torch.randn((tokens, hidden), generator=generator) * (3.0 if multi_block else 0.5)
    image_mask = torch.tensor([False, True] + ([False] if not multi_block else []))
    expected = reference(hidden_states, image_mask).detach()
    observed = ours(hidden_states, image_mask).detach()
    return {
        "reference_output": expected.tolist(),
        "expert_ids": reference.gate(hidden_states, image_mask)[1].tolist(),
        "max_abs_diff": (expected - observed).abs().max().item(),
    }


if __name__ == "__main__":
    moe = reference_moe(Path(sys.argv[1]))
    for multi in (False, True):
        print(json.dumps({"multi_block": multi, **run_case(moe, multi)}))
