"""Ripple's text stack: four sliding-window layers to one global.

Ported from the checkpoint's ``modeling_ricursive.py`` (Apache-2.0). Per layer:
four RMSNorms, two around each sublayer with the second inside the residual, a
gated GQA attention and a SwiGLU MLP with clamped inputs. Each layer's values
read a blend of its input and the scaled token embeddings. Queries and keys are
RMS-normed before RoPE, which only the sliding layers apply; the full layers
are NoPE.

All layers write one paged KV cache, which keeps only the window of the sliding
layers' K/V; they attend it through a windowed attention resource, the full
ones through an unwindowed one.

TP shards by head. The post-norms sit between each row-parallel projection and
its residual add, so those projections reduce their own outputs.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.engine.resources.convenience import AttentionCallable
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.ripple.config import ATTN_FULL, ATTN_SLIDING, KV_CACHE, ROPE, RippleConfig


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
    def __init__(self, config: RippleConfig, sliding: bool, comm_group: CommGroup):
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
            bias=False, input_is_parallel=True, reduce_results=True,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.value_res_lambda = nn.Parameter(torch.zeros(1))
        self.sliding = sliding
        self._attn_key = ATTN_SLIDING if sliding else ATTN_FULL
        self.attn = self.kv = self.pos = None
        self.attend: AttentionCallable | None = None

    def bind_resources(self, resources: dict) -> None:
        self.attn = resources.get(self._attn_key)
        self.kv = resources.get(KV_CACHE)
        # NoPE: the full layers never rotate
        self.pos = resources.get(ROPE) if self.sliding else None
        self.attend = AttentionCallable(kv=self.kv, attn=self.attn)

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
        if self.pos is not None:
            q, k = self.pos.apply_qk(q, k, label=self.attend.label)
        out = self.attend(q, k, v).reshape(n, q_dim)
        return self.o_proj(out * torch.sigmoid(gate.float()).to(out.dtype))


class RippleDecoderLayer(nn.Module):
    def __init__(self, config: RippleConfig, sliding: bool, comm_group: CommGroup):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = RippleAttention(config, sliding, comm_group)
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
        self.layers = nn.ModuleList(RippleDecoderLayer(config, s, comm_group) for s in sliding)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # a layer of each kind, to bind the step's label on both attention resources
        self._bind_through = (sliding.index(True), sliding.index(False))

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale

    def forward(self, embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        for idx in self._bind_through:
            self.layers[idx].self_attn.attend.bind_step(label)
        # every layer's values read ``embeds``, and the residual stream is updated in place
        residual = embeds.clone()
        h = self.layers[0].input_layernorm(embeds)
        out = None
        for i, layer in enumerate(self.layers):
            if out is not None:
                h, residual = layer.input_layernorm.forward_residual(out, residual)
            # every layer is a layer of the one cache, so its stack index
            layer.self_attn.attend.set_layer_idx(i)
            a = layer.post_attention_layernorm(layer.self_attn(h, embeds))
            h, residual = layer.pre_feedforward_layernorm.forward_residual(a, residual)
            out = layer.post_feedforward_layernorm(layer.mlp(h))
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
