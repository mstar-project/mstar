"""Generate a small compressor golden from the pinned DeepSeek V4.1 reference.

Usage: python generate_deepseek_v41_compressor_golden.py /path/to/inference/model.py
The source is deepseek-ai/DeepSeek-V4.1-Flash@314113f8a7dadb388215396ba5fbee2c610a04d6.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def main(reference_path: Path) -> None:
    source = reference_path.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    if sha != "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d":
        raise ValueError("compressor golden needs the pinned DeepSeek V4.1 model.py")
    selected = {"linear", "Linear", "RMSNorm", "Compressor"}
    tree = ast.parse(source)
    code = ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "F": F, "nn": nn, "ModelArgs": object, "default_dtype": torch.bfloat16}
    exec(compile(code, str(reference_path), "exec", dont_inherit=True), namespace)
    args = SimpleNamespace(compress_ratios=(1, 2), head_dim=8, norm_eps=1e-20, dim=16, max_batch_size=2)
    ratio_one = namespace["Compressor"](args, 0).eval()
    ratio_two = namespace["Compressor"](args, 1).eval()
    generator = torch.Generator().manual_seed(81142)
    with torch.no_grad():
        weights = {}
        for name, model in (("one", ratio_one), ("two", ratio_two)):
            for key, param in model.named_parameters():
                value = torch.randn(param.shape, generator=generator) * 0.15
                if key == "norm.weight":
                    value += 1
                param.copy_(value.to(param.dtype))
                weights[f"{name}.{key}"] = param.float().tolist()
        seq = (torch.randn((2, 8, 16), generator=generator) * 0.4).to(torch.bfloat16)
        one_prefill = ratio_one(seq[:, :5], 0)
        one_decode = ratio_one(seq[:, 5:6], 5)
        two_prefill = ratio_two(seq[:, :5], 0)
        after_prefill = (ratio_two.kv_state[:2].clone(), ratio_two.score_state[:2].clone())
        two_complete5 = ratio_two(seq[:, 5:6], 5)
        after_complete5 = (ratio_two.kv_state[:2].clone(), ratio_two.score_state[:2].clone())
        two_partial6 = ratio_two(seq[:, 6:7], 6)
        after_partial6 = (ratio_two.kv_state[:2].clone(), ratio_two.score_state[:2].clone())
        two_complete7 = ratio_two(seq[:, 7:8], 7)
        after_complete7 = (ratio_two.kv_state[:2].clone(), ratio_two.score_state[:2].clone())
        short = namespace["Compressor"](args, 1).eval()
        short.load_state_dict(ratio_two.state_dict())
        short_prefill = short(seq[:, :1], 0)
        short_complete = short(seq[:, 1:2], 1)
        exact = namespace["Compressor"](args, 1).eval()
        exact.load_state_dict(ratio_two.state_dict())
        exact_prefill = exact(seq[:, :4], 0)
        exact_state = (exact.kv_state[:2].clone(), exact.score_state[:2].clone())
    # JSON cannot encode the unused-slot -inf scores, so use a reversible sentinel.
    payload = {
        "reference_sha256": sha,
        "weights": weights,
        "input": seq.float().tolist(),
        "one_prefill": one_prefill.float().tolist(),
        "one_decode": one_decode.float().tolist(),
        "two_prefill": two_prefill.float().tolist(),
        "two_complete5": two_complete5.float().tolist(),
        "two_partial6": two_partial6,
        "two_complete7": two_complete7.float().tolist(),
        "short_prefill": short_prefill,
        "short_complete": short_complete.float().tolist(),
        "exact_prefill": exact_prefill.float().tolist(),
        "after_prefill": [part.nan_to_num(neginf=-1e30).tolist() for part in after_prefill],
        "after_complete5": [part.nan_to_num(neginf=-1e30).tolist() for part in after_complete5],
        "after_partial6": [part.nan_to_num(neginf=-1e30).tolist() for part in after_partial6],
        "after_complete7": [part.nan_to_num(neginf=-1e30).tolist() for part in after_complete7],
        "exact_state": [part.nan_to_num(neginf=-1e30).tolist() for part in exact_state],
    }
    Path(__file__).with_name("deepseek_v41_compressor_golden.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
