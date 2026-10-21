"""Muse Glimmer's text stack: three sliding-window layers to one global.

Ported from transformers' ``modeling_muse_glimmer.py`` (Apache-2.0). Per layer:
four Gemma-style ``(1 + w)`` sandwich norms around a gated GQA attention and a
SwiGLU MLP. Queries and keys are RMS-normed without a weight before RoPE, which
only the sliding layers apply; the full layers are NoPE. The query multiplier
``qk_scale_factor`` is folded into the attention resources' softmax scale.

All 52 layers write one paged KV cache; the sliding layers attend it through
a windowed attention resource, the full ones through an unwindowed one.

TP shards by head, as Qwen3.5 does. The post-norms sit between each
row-parallel projection and its residual add, so those projections reduce
their own outputs rather than deferring the all-reduce into a fused add+norm.
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
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.norm import rms_norm_no_weight
from mstar.model.muse.config import (
    ATTN_FULL,
    ATTN_SLIDING,
    KV_CACHE,
    ROPE,
    MuseGlimmerConfig,
)


class GatedQKVParallelLinear(QKVParallelLinear):
    """q, its output gate, k and v as one GEMM.

    Each rank's slice is ``[q | gate | k | v]`` for its own heads: the gate
    is sharded exactly as q is. The checkpoint stores the gate as its own
    ``self_attn.gate_proj``, loaded under shard id ``"gate"``.
    """

    def __init__(
        self, comm_group: CommGroup, hidden_size: int, head_size: int,
        total_num_heads: int, total_num_kv_heads: int,
    ):
        # the gate is as wide as q, so the q region holds twice the heads
        super().__init__(
            comm_group=comm_group, hidden_size=hidden_size, head_size=head_size,
            total_num_heads=2 * total_num_heads,
            total_num_kv_heads=total_num_kv_heads, bias=False,
        )
        self.q_size = self.num_heads // 2 * head_size

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if loaded_shard_id not in ("q", "gate"):
            return super().weight_loader(param, loaded_weight, loaded_shard_id)
        src = loaded_weight.narrow(0, self.tp_rank * self.q_size, self.q_size)
        offset = 0 if loaded_shard_id == "q" else self.q_size
        param.data.narrow(0, offset, self.q_size).copy_(src)


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self, config: MuseGlimmerConfig, sliding: bool, comm_group: CommGroup,
    ):
        super().__init__()
        tp = comm_group.world_size
        self.head_dim = config.head_dim
        self.num_heads = divide(config.num_attention_heads, tp)
        self.eps = config.rms_norm_eps
        self.qkvg_proj = GatedQKVParallelLinear(
            comm_group=comm_group, hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
        )
        self.num_kv_heads = self.qkvg_proj.num_kv_heads
        self.o_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size, bias=False,
            input_is_parallel=True, reduce_results=True,
        )
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, gate, k, v = self.qkvg_proj(hidden_states).split(
            [q_dim, q_dim, kv_dim, kv_dim], dim=-1,
        )
        q = rms_norm_no_weight(q.reshape(n, self.num_heads, self.head_dim), self.eps)
        k = rms_norm_no_weight(k.reshape(n, self.num_kv_heads, self.head_dim), self.eps)
        v = v.reshape(n, self.num_kv_heads, self.head_dim)
        if self.pos is not None:
            # plain NEOX rope; no llama-3.1 scaling arguments
            q, k = self.pos.apply_qk(q, k, label=self.attend.label)
        out = self.attend(q, k, v).reshape(n, q_dim)
        return self.o_proj(out * torch.sigmoid(gate))


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self, config: MuseGlimmerConfig, sliding: bool, comm_group: CommGroup,
    ):
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = MuseGlimmerAttention(config, sliding, comm_group)
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden, intermediate_size=config.intermediate_size,
            comm_group=comm_group, activation="silu",
        )
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps, gemma_mode=True)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.post_norm_eps, gemma_mode=True)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps, gemma_mode=True)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=config.post_norm_eps, gemma_mode=True)


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        sliding = config.sliding_layers
        self.config = config
        self.comm_group = comm_group
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, comm_group=comm_group,
        )
        self.layers = nn.ModuleList(
            MuseGlimmerDecoderLayer(config, s, comm_group) for s in sliding
        )
        # plain `w`, unlike the layers' norms
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # a layer of each kind, to bind the step's label on both attention resources
        self._bind_through = (sliding.index(True), sliding.index(False))

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings enter the stack at unit RMS, as image ones do."""
        return rms_norm_no_weight(self.embed_tokens(input_ids), self.config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, *, label: str) -> torch.Tensor:
        for idx in self._bind_through:
            self.layers[idx].self_attn.attend.bind_step(label)
        residual = hidden
        h = self.layers[0].input_layernorm(hidden)
        out = None
        for i, layer in enumerate(self.layers):
            if out is not None:
                h, residual = layer.input_layernorm.forward_residual(out, residual)
            # every layer is a layer of the one cache, so its stack index
            layer.self_attn.attend.set_layer_idx(i)
            a = layer.post_attention_layernorm(layer.self_attn(h))
            h, residual = layer.pre_feedforward_layernorm.forward_residual(a, residual)
            out = layer.post_feedforward_layernorm(layer.mlp(h))
        normed, _ = self.norm.forward_residual(out, residual)
        return normed


class MuseGlimmerForCausalLM(nn.Module):
    def __init__(self, config: MuseGlimmerConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        self.model = MuseGlimmerTextModel(config, comm_group)
        # untied; gathers the vocab so the sampler is TP-oblivious
        self.lm_head = ColumnParallelLinear(
            comm_group=self.model.comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )
        self._cap = config.final_logit_softcapping
        self._multiplier = config.output_multiplier or 1.0

    # The decoder contract `SampledTokenDecoder` drives (as `Qwen3DenseLM`).

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, hidden)`` in, the stack's normed output out."""
        return self.model(input_embeds, label=label)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``cap * tanh(lm_head(x) * output_multiplier / cap)``, in fp32."""
        x = self.lm_head(hidden).float() * self._multiplier
        return x if self._cap is None else self._cap * torch.tanh(x / self._cap)
