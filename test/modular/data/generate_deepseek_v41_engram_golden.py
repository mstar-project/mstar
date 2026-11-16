"""Generate a small Engram golden from the pinned DeepSeek V4.1 reference.

Usage: python generate_deepseek_v41_engram_golden.py /path/to/inference/model.py
The reference is deepseek-ai/DeepSeek-V4.1-Flash@314113f8a7dadb388215396ba5fbee2c610a04d6.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def main(reference_path: Path) -> None:
    source = reference_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d":
        raise ValueError("Engram golden needs the pinned DeepSeek V4.1 model.py")
    tree = ast.parse(source)
    selected = {"linear", "Linear", "ParallelEngramEmbedding", "Engram"}
    code = ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "dist": dist,
        "F": F,
        "nn": nn,
        "world_size": 1,
        "rank": 0,
        "fp8_block_size": 32,
        "scale_dtype": torch.float8_e8m0fnu,
        "default_dtype": torch.bfloat16,
        "EngramLayout": object,
        "ModelArgs": object,
    }
    # The selected source is executed unchanged; unrelated model imports and
    # GPU kernels are not needed for the BF16 wkv path in this fixture.
    exec(compile(code, str(reference_path), "exec", dont_inherit=True), namespace)
    layout = SimpleNamespace(layer_ids=(1,), num_embeddings=(5,), head_dim=64, max_ngram_size=2, n_heads=2)
    args = SimpleNamespace(dim=8, hc_mult=4, norm_eps=1e-20)
    engram = namespace["Engram"](args, 1, layout).eval()
    generator = torch.Generator().manual_seed(81141)
    with torch.no_grad():
        table = (torch.randn((5, 64), generator=generator) * 0.3).to(torch.float8_e4m3fn)
        table_scales = torch.tensor(
            [[0.5, 2], [4, 0.25], [2, 0.5], [1, 4], [0.25, 1]], dtype=torch.float8_e8m0fnu
        )
        engram.embed.weight.copy_(table)
        engram.embed.scale.copy_(table_scales)
        dense = (torch.randn((40, 128), generator=generator) * 0.1).to(torch.float8_e4m3fn)
        wkv_scales = torch.tensor(
            [[0.5, 2, 1, 4], [2, 0.5, 4, 0.25]], dtype=torch.float8_e8m0fnu
        )
        dequant = dense.float()
        for row in range(dense.shape[0]):
            for col in range(dense.shape[1]):
                dequant[row, col] *= wkv_scales[row // 32, col // 32].float()
        engram.wkv.weight.copy_(dequant.to(torch.bfloat16))
        engram.q_weight.copy_(torch.randn((4, 8), generator=generator) * 0.15 + 1)
        engram.k_weight.copy_(torch.randn((4, 8), generator=generator) * 0.15 + 1)
        streams = (torch.randn((1, 2, 4, 8), generator=generator) * 0.4).to(torch.bfloat16)
        hash_ids = torch.tensor([[[0, 4], [1, 3]]])
        mask = torch.tensor([[True, False]])
        output = engram(streams, hash_ids, mask)
    payload = {
        "reference_sha256": source_sha256,
        "table": table.float().tolist(),
        "table_scales": table_scales.float().tolist(),
        "wkv": dense.float().tolist(),
        "wkv_scales": wkv_scales.float().tolist(),
        "q_weight": engram.q_weight.tolist(),
        "k_weight": engram.k_weight.tolist(),
        "streams": streams.float().tolist(),
        "hash_ids": hash_ids.tolist(),
        "mask": mask.tolist(),
        "output": output.float().tolist(),
    }
    Path(__file__).with_name("deepseek_v41_engram_golden.json").write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
