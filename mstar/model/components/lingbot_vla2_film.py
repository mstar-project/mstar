"""LingBot-VLA 2.0's checkpoint-shaped action-expert FiLM norm.

The released action tower stores a learned RMS weight and separate gamma and
beta projections. M*'s shared AdaRMSNorm has a fused scale/shift/gate layout,
so it cannot load these weights or return the reference's single tensor.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards


class LingBotVLAAdaRMSNorm(nn.Module):
    """RMS normalize in FP32, then apply the reference's per-step FiLM."""

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        if min(hidden_size, cond_dim) <= 0 or eps <= 0:
            raise ValueError("LingBot-VLA FiLM dimensions and epsilon must be positive")
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("LingBot-VLA FiLM input must be [batch, tokens, hidden_size]")
        if cond.shape != (hidden_states.shape[0], self.cond_dim):
            raise ValueError("LingBot-VLA FiLM condition must be [batch, cond_dim]")
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        x = self.weight * x
        cond = cond.to(self.gamma.weight.dtype)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return ((1 + gamma.to(torch.float32)) * x + beta.to(torch.float32)).to(input_dtype)


def load_action_film_weights(
    norm: LingBotVLAAdaRMSNorm, checkpoint_dir: str | Path, layer: int, kind: str,
    *, device: str | torch.device = "cpu",
) -> set[str]:
    """Load one action layer's input or post-attention norm from indexed shards."""
    if layer < 0 or kind not in ("input_layernorm", "post_attention_layernorm"):
        raise ValueError("LingBot-VLA FiLM requires a nonnegative layer and a supported norm kind")
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"LingBot-VLA FiLM requires {index_path}")
    prefix = f"model.qwenvl_with_expert.qwen_expert.model.layers.{layer}.{kind}."
    expected = {prefix + name for name in dict(norm.named_parameters())}
    actual = {name for name in json.loads(index_path.read_text())["weight_map"] if name.startswith(prefix)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"LingBot-VLA FiLM checkpoint keys differ: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]})"
        )
    weights = iter_safetensors_shards(checkpoint_dir, device=device, keys=expected)
    return load_weights_into(
        norm, ((name.removeprefix(prefix), value) for name, value in weights), strict=True,
    )
