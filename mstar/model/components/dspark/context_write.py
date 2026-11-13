"""DSpark's context write on its Qwen3 layers: each layer's K/V of the projected taps, without attending.

Where a layer's forward fuses (CUDA, planned positions, plain RoPE, the plain fused projection), only the
projection's K/V rows run, and k's norm, RoPE and the write are one launch of the forward's kernel
(`draft.context_write.norm_rope_write`). Elsewhere each layer takes the unfused path, as
`Qwen3DenseLM.write_kv` does."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mstar.engine.resources.draft.context_write import norm_rope_write
from mstar.model.components.distributed import QKVParallelLinear

if TYPE_CHECKING:
    from mstar.model.components.qwen3_lm import Qwen3Attention, Qwen3DenseLM


def write_context(backbone: Qwen3DenseLM, hidden_states: torch.Tensor, *, label: str) -> None:
    """``Qwen3DenseLM.write_kv``, with each layer's write through `write_layer_kv`."""
    backbone.layers[0].self_attn.attend.bind_step(label)
    for i, layer in enumerate(backbone.layers):
        layer.self_attn.attend.set_layer_idx(i)
        write_layer_kv(layer.self_attn, hidden_states)


def write_layer_kv(attn: Qwen3Attention, hidden_states: torch.Tensor) -> None:
    # the same fuse decision Qwen3Attention.forward makes, from what its bind_resources set
    pos_ids = attn.pos.pos_ids(attn.attend.label) if attn._fuse_qk else None
    if pos_ids is None or not hidden_states.is_cuda or type(attn.qkv_proj) is not QKVParallelLinear:
        # a LoRA or quantized projection keeps its own forward
        attn.write_kv(hidden_states)
        return
    k, v = project_kv(attn, hidden_states)
    norm_rope_write(attn.attend, k, v, attn.k_norm, pos_ids, attn._rope_theta)


def project_kv(attn: Qwen3Attention, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``_project_qkv``'s k and v alone, from the fused weight's K/V rows."""
    q_size, kv_size = attn.num_heads * attn.head_dim, attn.num_kv_heads * attn.head_dim
    bias = attn.qkv_proj.bias
    kv = F.linear(hidden_states, attn.qkv_proj.weight[q_size:], None if bias is None else bias[q_size:])
    k, v = kv.split([kv_size, kv_size], dim=-1)
    shape = (hidden_states.shape[0], attn.num_kv_heads, attn.head_dim)
    return k.view(shape), v.view(shape)
