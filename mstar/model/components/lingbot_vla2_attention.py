"""LingBot-VLA 2.0 action-attention projections and read pattern.

The reference concatenates cached prefix K/V with each denoise step's action
K/V. The state row reads the prefix and itself; action rows also read every
action row. The read pattern is a declaration for a future engine binding,
not a mask or an attention kernel executed by this model component.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelAttention
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards


@dataclass(frozen=True)
class LingBotVLASuffixReads:
    """Key regions read by the state and action queries, after prefix prefill."""

    state: tuple[str, ...] = ("prefix", "state")
    actions: tuple[str, ...] = ("prefix", "state", "actions")


SUFFIX_READS = LingBotVLASuffixReads()


@dataclass(frozen=True)
class LingBotVLAActionAttentionConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


class LingBotVLAActionProjections(nn.Module):
    """Qwen2 action-expert Q/K/V and output linears on shared TP primitives.

    This does not run attention or apply MRoPE. A future resource binding must
    join these suffix rows to the cached Qwen3-VL prefix on every layer.
    """

    def __init__(self, config: LingBotVLAActionAttentionConfig, *, comm_group: CommGroup | None = None):
        super().__init__()
        if min(config.hidden_size, config.num_heads, config.num_kv_heads, config.head_dim) <= 0:
            raise ValueError("LingBot-VLA action attention dimensions must be positive")
        if config.num_heads % config.num_kv_heads:
            raise ValueError("LingBot-VLA action query heads must divide by KV heads")
        self.config = config
        self.self_attn = ParallelAttention(
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=False,
            qk_norm=False,
            pos_key=None,
        )

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project already normalized ``[batch, tokens, hidden]`` rows."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("LingBot-VLA action attention input must be [batch, tokens, hidden_size]")
        batch, tokens = hidden_states.shape[:2]
        rows = hidden_states.to(self.self_attn.qkv_proj.weight.dtype)
        q, k, v = self.self_attn._project_qkv(rows.reshape(-1, self.config.hidden_size))
        return (
            q.reshape(batch, tokens, self.self_attn.num_heads, self.config.head_dim),
            k.reshape(batch, tokens, self.self_attn.num_kv_heads, self.config.head_dim),
            v.reshape(batch, tokens, self.self_attn.num_kv_heads, self.config.head_dim),
        )

    def project_output(self, attention_rows: torch.Tensor) -> torch.Tensor:
        """Project attended suffix rows back to action-expert hidden width."""
        if attention_rows.ndim != 4 or attention_rows.shape[2:] != (
            self.self_attn.num_heads, self.config.head_dim,
        ):
            raise ValueError("LingBot-VLA attended rows must be [batch, tokens, heads, head_dim]")
        batch, tokens = attention_rows.shape[:2]
        rows = attention_rows.to(self.self_attn.o_proj.weight.dtype)
        out = self.self_attn.o_proj(rows.reshape(-1, self.self_attn.num_heads * self.config.head_dim))
        return out.reshape(batch, tokens, self.config.hidden_size)


def load_action_attention_weights(
    projections: LingBotVLAActionProjections, checkpoint_dir: str | Path, layer: int,
    *, device: str | torch.device = "cpu",
) -> set[str]:
    """Load one action layer's seven Q/K/V and output projection tensors."""
    if layer < 0:
        raise ValueError("LingBot-VLA action attention layer must be nonnegative")
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"LingBot-VLA action attention requires {index_path}")
    prefix = f"model.qwenvl_with_expert.qwen_expert.model.layers.{layer}.self_attn."
    expected = {prefix + f"{part}_proj.{kind}" for part in "qkv" for kind in ("weight", "bias")}
    expected.add(prefix + "o_proj.weight")
    actual = {name for name in json.loads(index_path.read_text())["weight_map"] if name.startswith(prefix)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"LingBot-VLA action attention checkpoint keys differ: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]})"
        )
    c = projections.config
    q_width, kv_width = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
    shapes = {
        prefix + "q_proj.weight": (q_width, c.hidden_size),
        prefix + "q_proj.bias": (q_width,),
        prefix + "k_proj.weight": (kv_width, c.hidden_size),
        prefix + "k_proj.bias": (kv_width,),
        prefix + "v_proj.weight": (kv_width, c.hidden_size),
        prefix + "v_proj.bias": (kv_width,),
        prefix + "o_proj.weight": (c.hidden_size, q_width),
    }
    weights = dict(iter_safetensors_shards(checkpoint_dir, device=device, keys=expected))
    if set(weights) != expected:
        raise ValueError("LingBot-VLA action attention checkpoint did not yield every indexed tensor")
    for name, shape in shapes.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"LingBot-VLA action attention {name} has shape {tuple(weights[name].shape)}, expected {shape}"
            )
    return load_weights_into(
        projections,
        (("self_attn." + name.removeprefix(prefix), value) for name, value in weights.items()),
        stacked_params=LLAMA_STACKED_PARAMS[:3], strict=True,
    )
