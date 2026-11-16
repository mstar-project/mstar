"""Regenerate the two-layer CPU golden from the pinned DeepSeek Block.

Download ``inference/model.py`` from ``deepseek-ai/DeepSeek-V4.1-Flash`` at
revision ``314113f8a7dadb388215396ba5fbee2c610a04d6``, then run
``python test/modular/data/generate_deepseek_v41_layer_golden.py PATH_TO_MODEL_PY``.
The published Block and RMSNorm are extracted without importing its GPU-only
kernel package. The CPU Sinkhorn below follows its published kernel's order.
Attention and FFN are deterministic tiny modules, so this fixture isolates the
layer's mHC order and the incoming pre-mix carried between layers.
"""

from __future__ import annotations

import ast
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


@contextmanager
def set_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class ToyAttention(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(args.dim, args.dim))

    def forward(self, x, start_pos, *args):
        return torch.tanh(F.linear(x, self.weight)) + start_pos * 0.001


class ToyFFN(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(args.dim, args.dim))

    def forward(self, x, image_mask=None):
        image_offset = image_mask.unsqueeze(-1) * 0.01 if image_mask is not None else 0
        return torch.sin(F.linear(x, self.weight)) + image_offset


def hc_split_sinkhorn(mixes, scale, base, hc, iterations, eps):
    """CPU translation of inference/kernel.py's hc_split_sinkhorn_kernel."""
    pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
    pre = torch.sigmoid(pre_w * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(post_w * scale[1] + base[hc : 2 * hc])
    comb = (
        torch.softmax(
            comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scale[2] + base[2 * hc :].reshape(hc, hc),
            dim=-1,
        )
        + eps
    )
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def reference_block(source: Path):
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"RMSNorm", "Block"}]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Attention": ToyAttention,
        "MoE": ToyFFN,
        "ModelArgs": object,
        "EngramLayout": object,
        "set_dtype": set_dtype,
        "hc_split_sinkhorn": hc_split_sinkhorn,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Block"]


def golden(source: Path) -> dict:
    block = reference_block(source)
    args = SimpleNamespace(dim=8, norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    layers = [block(index, args) for index in range(2)]
    generator = torch.Generator().manual_seed(811)
    with torch.no_grad():
        for layer in layers:
            for name in ("attn", "ffn"):
                getattr(layer, name).weight.copy_(torch.randn((8, 8), generator=generator) * 0.15)
            for name in ("attn_norm", "ffn_norm"):
                getattr(layer, name).weight.copy_(0.8 + torch.randn((8,), generator=generator) * 0.1)
            for name in ("attn", "ffn"):
                for suffix in ("fn", "base", "scale"):
                    target = getattr(layer, f"hc_{name}_{suffix}")
                    multiplier = {"fn": 0.1, "base": 0.2, "scale": 0.1}[suffix]
                    value = torch.randn(target.shape, generator=generator) * multiplier
                    if suffix == "scale":
                        value += 1
                    target.copy_(value)
        streams = torch.randn((1, 3, 4, 8), generator=generator) * 0.5
        image_mask = torch.tensor([[False, True, False]])
        pre = torch.zeros((1, 3, 4))
        pre[:, :, 0] = 1
        result = {}
        for index, layer in enumerate(layers):
            streams, pre = layer(streams, 7, pre, image_mask)
            prefix = "golden_first_" if index == 0 else "golden_"
            result[f"{prefix}streams"] = streams.tolist()
            result[f"{prefix}pre"] = pre.tolist()
        result["golden_final"] = layers[-1].hc_pre(streams, pre).tolist()
    return result


if __name__ == "__main__":
    print(json.dumps(golden(Path(sys.argv[1])), indent=2))
