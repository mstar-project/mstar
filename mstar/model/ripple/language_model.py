"""Ripple's text stack: four sliding-window layers to one global.

Ported from the checkpoint's ``modeling_ricursive.py`` (Apache-2.0). Per layer:
four RMSNorms, two around each sublayer with the second inside the residual, a
gated GQA attention and a SwiGLU MLP with clamped inputs. Each layer's values
read a blend of its input and the scaled token embeddings. Queries and keys are
RMS-normed before RoPE, which only the sliding layers apply; the full layers
are NoPE.

All layers write one paged KV cache, which keeps only the window of the sliding
layers' K/V; they read it through a windowed read, the full ones through a
causal one.

TP shards by head. The post-norms sit between each row-parallel projection and
its residual add, so each reduces its projection's partial sums in one fused
all-reduce + norm kernel.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.engine.state import add_read, rope
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.ripple.config import KV_CACHE, RippleConfig


class RippleMLP(ParallelGatedMLP):
    """SwiGLU whose gate is capped and whose up projection is clamped both ways."""

    def __init__(self, config: RippleConfig, comm_group: CommGroup):
        super().__init__(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            comm_group=comm_group, activation="silu",
        )
        self.clamp = config.ffn_clamp_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.intermediate_size_per_partition, dim=-1)
        return self.down_proj(self.act(gate.clamp(max=self.clamp)) * up.clamp(-self.clamp, self.clamp))


class RippleAttention(nn.Module):
    def __init__(self, config: RippleConfig, sliding: bool, comm_group: CommGroup, layer: int):
        super().__init__()
        tp = comm_group.world_size
        self.head_dim = config.head_dim
        self.num_heads = divide(config.num_attention_heads, tp)
        # also refuses a TP past the KV heads, which qkg_proj's even split would cut through a head
        self.num_kv_heads = divide(config.num_key_value_heads, tp)
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        # q, k and the output gate read the layer's input; v reads its blend with the embeddings
        self.qkg_proj = MergedColumnParallelLinear(
            comm_group=comm_group, input_size=config.hidden_size,
            output_sizes=[q_width, kv_width, q_width], bias=False,
        )
        self.v_proj = ColumnParallelLinear(comm_group, config.hidden_size, kv_width, bias=False)
        self.o_proj = RowParallelLinear(
            comm_group=comm_group, input_size=q_width, output_size=config.hidden_size,
            bias=False, input_is_parallel=True,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.value_res_lambda = nn.Parameter(torch.zeros(1))
        self.sliding = sliding
        self.layer = torch.tensor(layer, device="cpu")
        self.kv = None

    def bind_state(self, handles: dict) -> None:
        self.kv = handles[KV_CACHE]

    def forward(self, hidden_states: torch.Tensor, value_residual: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, gate = self.qkg_proj(hidden_states).split([q_dim, kv_dim, q_dim], dim=-1)
        blend = torch.sigmoid(self.value_res_lambda[0])
        v = self.v_proj(torch.lerp(value_residual, hidden_states, blend))
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(n, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(n, self.num_kv_heads, self.head_dim)
        v = v.view(n, self.num_kv_heads, self.head_dim)
        # NoPE: the full layers never rotate
        if self.sliding:
            q, k = rope(self.kv, q, k)
        out = add_read(self.kv, self.layer, q, k=k, v=v).reshape(n, q_dim)
        return self.o_proj(out * torch.sigmoid(gate.float()).to(out.dtype))


class RippleDecoderLayer(nn.Module):
    def __init__(self, config: RippleConfig, sliding: bool, comm_group: CommGroup, layer: int):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = RippleAttention(config, sliding, comm_group, layer)
        self.mlp = RippleMLP(config, comm_group)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)


class RippleTextModel(nn.Module):
    def __init__(self, config: RippleConfig, comm_group: CommGroup | None = None):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        sliding = config.is_sliding
        self.comm_group = comm_group
        self.embed_scale = config.embed_scale
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, comm_group=comm_group)
        self.layers = nn.ModuleList(RippleDecoderLayer(config, s, comm_group, i) for i, s in enumerate(sliding))
        # `forward` reduces both row-parallel projections in their post-norms,
        # one fused all-reduce + norm kernel each, so they return partial sums
        for layer in self.layers:
            layer.self_attn.o_proj.reduce_results = False
            layer.mlp.down_proj.reduce_results = False
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale

    def forward(self, embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """Each row's label comes from the step's segments, so ``label`` goes unread."""
        cg = self.comm_group
        # every layer's values read ``embeds``, and the residual stream is updated in place
        residual = embeds.clone()
        h = self.layers[0].input_layernorm(embeds)
        out = None
        for layer in self.layers:
            if out is not None:
                h, residual = layer.input_layernorm.forward_residual(out, residual)
            a = layer.post_attention_layernorm(layer.self_attn(h, embeds), cg)
            h, residual = layer.pre_feedforward_layernorm.forward_residual(a, residual)
            out = layer.post_feedforward_layernorm(layer.mlp(h), cg)
        normed, _ = self.norm.forward_residual(out, residual)
        return normed


class RippleForCausalLM(nn.Module):
    """The decoder contract `SampledTokenDecoder` drives."""

    def __init__(self, config: RippleConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.model = RippleTextModel(config, comm_group)
        # gathers the vocab so the sampler is TP-oblivious
        self.lm_head = ColumnParallelLinear(
            comm_group=self.model.comm_group, input_size=config.hidden_size,
            output_size=config.vocab_size, bias=False, gather_output=True,
        )

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed(input_ids)

    def forward(self, embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        return self.model(embeds, label=label)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
