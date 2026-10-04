"""Qwen3.5's hybrid text stack: 3 gated-delta-net layers to 1 full-attention.

Each layer indexes its own resource (recurrent pool or KV cache) by its
position among its own kind; see ``Qwen3_5Config.resource_layer_index``.

TP shards by head (q-heads for attention, k/v-heads for the delta net), and the
engine shards the KV cache and recurrent pool to match, so
``get_node_resources`` stays unsharded. ``q/k/v``, ``gate/up`` and the delta
net's four projections are each fused into one GEMM, since at decode width they
are latency-bound. ``num_key_value_heads`` is 4 or 2, so past
that TP degree the K/V heads replicate (see ``QKVParallelLinear``).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components import Attention, DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.linear_attn import GDNProjLayout
from mstar.model.qwen3_5.components.rope import (
    apply_partial_mrope,
    compute_3d_cos_sin,
    compute_inv_freq,
)
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTENTION,
    LINEAR_ATTN,
    ROPE,
    Qwen3_5Config,
)


class RopeCache:
    """This step's cos/sin, shared by every full-attention layer.

    Set once per step, like the label cursor, so cos/sin need not be threaded
    through ``DecoderLayer.forward``.
    """

    def __init__(self):
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

    def set(self, cos: torch.Tensor | None, sin: torch.Tensor | None) -> None:
        self.cos, self.sin = cos, sin


class Qwen3_5Attention(Attention):
    """Full attention with Qwen3.5's output gate.

    The checkpoint's ``q_proj`` interleaves query and gate **per head**
    (``[q | gate]`` per head block), so the q part of ``qkv_proj`` is chunked
    after the per-head view, never before.
    """

    def __init__(
        self, *, output_gate: bool = True, rope_cache: RopeCache | None = None,
        comm_group: CommGroup | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_gate = output_gate
        self.rope_cache = rope_cache
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp = comm_group.world_size
        if self.q_norm is not None:
            # Gemma-style, not the parent's Llama-style; head_dim does not shard
            eps = self.q_norm.variance_epsilon
            self.q_norm = RMSNorm(self.head_dim, eps=eps, gemma_mode=True)
            self.k_norm = RMSNorm(self.head_dim, eps=eps, gemma_mode=True)

        qkv_bias = self.q_proj.bias is not None
        o_bias = self.o_proj.bias is not None
        self.total_num_heads = self.num_heads
        self.num_heads = divide(self.num_heads, tp)

        # One GEMM for q, k and v. The gate is inside each query head's block,
        # so q counts as twice the heads and a head shard keeps it intact.
        self.q_proj = self.k_proj = self.v_proj = None
        self.qkv_proj = QKVParallelLinear(
            comm_group=comm_group, hidden_size=self.input_hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads * (2 if output_gate else 1),
            total_num_kv_heads=self.num_kv_heads, bias=qkv_bias,
        )
        self.total_num_kv_heads = self.num_kv_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.o_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.input_hidden_size, bias=o_bias,
            input_is_parallel=True, reduce_results=True,
        )

    def _apply_rope(self, q, k, label):
        """Interleaved 3D MRoPE over the partial rotary dim; 1D RoPE when no
        cos/sin was set (a text-only bring-up harness)."""
        if self.rope_cache is None or self.rope_cache.cos is None:
            return super()._apply_rope(q, k, label)
        return apply_partial_mrope(q, k, self.rope_cache.cos, self.rope_cache.sin)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q_dim = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden_states).split([q_dim, kv_dim, kv_dim], dim=-1)
        if self.output_gate:
            q, gate = torch.chunk(
                q.view(num_tokens, self.num_heads, self.head_dim * 2), 2, dim=-1,
            )
        else:
            q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = self._apply_qk_norm(q, k)
        q, k = self._apply_rope(q, k, self.attend.label)
        out = self.attend(q, k, v).reshape(num_tokens, self.num_heads * self.head_dim)
        if self.output_gate:
            out = out * torch.sigmoid(gate.reshape(num_tokens, -1))
        return self.o_proj(out)


def _norm(config: Qwen3_5Config) -> RMSNorm:
    """Gemma-style RMSNorm (scales by ``1 + weight``); the gated norm is not."""
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_mode=True)


def _build_mlp(config: Qwen3_5Config, comm_group: CommGroup) -> nn.Module:
    # fused gate/up; see `weight_loader._STACKED_PARAMS`
    return ParallelGatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        comm_group=comm_group,
        activation="silu",
    )


def _build_linear_attn_layer(
    config: Qwen3_5Config, comm_group: CommGroup,
) -> DecoderLayer:
    return DecoderLayer(
        self_attn=ParallelGatedDeltaNet(
            hidden_size=config.hidden_size,
            num_k_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            layout=GDNProjLayout.SPLIT,
            rms_norm_eps=config.rms_norm_eps,
            linear_attn_key=LINEAR_ATTN,
            state_key=GDN_STATE,
            comm_group=comm_group,
        ),
        mlp=_build_mlp(config, comm_group),
        input_layernorm=_norm(config),
        post_attention_layernorm=_norm(config),
    )


def _build_full_attn_layer(
    config: Qwen3_5Config, rope_cache: RopeCache, comm_group: CommGroup,
) -> DecoderLayer:
    return DecoderLayer(
        self_attn=Qwen3_5Attention(
            output_gate=config.attn_output_gate,
            rope_cache=rope_cache,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attn_key=ATTN,
            kv_key=KV_CACHE,
            pos_key=ROPE,
            comm_group=comm_group,
        ),
        mlp=_build_mlp(config, comm_group),
        input_layernorm=_norm(config),
        post_attention_layernorm=_norm(config),
    )


class Qwen3_5LanguageModel(nn.Module):
    def __init__(
        self, config: Qwen3_5Config, comm_group: CommGroup | None = None,
    ):
        super().__init__()
        self.config = config
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, comm_group=comm_group,
        )
        self.rope_cache = RopeCache()
        self.layers = nn.ModuleList(
            _build_linear_attn_layer(config, comm_group)
            if kind == LINEAR_ATTENTION
            else _build_full_attn_layer(config, self.rope_cache, comm_group)
            for kind in config.layer_types
        )
        self.register_buffer(
            "inv_freq",
            compute_inv_freq(config.rotary_dim, config.rope_theta),
            persistent=False,
        )
        self.norm = _norm(config)

        # precomputed: which cursor each layer advances, and to what index
        self._cursor_attr = [
            "mix" if kind == LINEAR_ATTENTION else "attend"
            for kind in config.layer_types
        ]
        self._resource_idx = [
            config.resource_layer_index(i) for i in range(config.num_hidden_layers)
        ]
        # the cursors are on shared resources, so bind once per kind
        self._bind_through = {
            attr: self._cursor_attr.index(attr) for attr in set(self._cursor_attr)
        }

        # Defer each residual add into the next norm: one fused add+norm (under
        # TP, all-reduce+add+norm) per block, so row-parallel outputs return
        # partial sums.
        for layer in self.layers:
            mixer = layer.self_attn
            out = mixer.o_proj if hasattr(mixer, "o_proj") else mixer.out_proj
            out.reduce_results = False
            layer.mlp.down_proj.reduce_results = False

    def build_cos_sin(
        self, position_ids_3d: torch.Tensor, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for ``[3, tokens]`` positions, to hand to ``forward``."""
        return compute_3d_cos_sin(
            position_ids_3d, self.inv_freq, self.config.mrope_section,
            target_dtype=dtype,
        )

    def forward(
        self, query_sequence: torch.Tensor, *, label: str,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.rope_cache.set(*(cos_sin or (None, None)))
        # bind the label once per resource kind, advance the index per layer
        for attr, idx in self._bind_through.items():
            getattr(self.layers[idx].self_attn, attr).bind_step(label)
        return self._forward_fused_residual(query_sequence)

    def _forward_fused_residual(self, hidden: torch.Tensor) -> torch.Tensor:
        """``DecoderLayer.forward`` with residual adds deferred into the next
        norm, which also reduces the TP partial sums."""
        cg = self.comm_group if self.comm_group.world_size > 1 else None
        residual = None
        for i, layer in enumerate(self.layers):
            getattr(layer.self_attn, self._cursor_attr[i]).set_layer_idx(
                self._resource_idx[i]
            )
            if residual is None:
                residual = hidden
                h = layer.input_layernorm(hidden)
            else:
                h, residual = layer.input_layernorm.forward_residual(
                    hidden, residual, cg,
                )
            h = layer.self_attn(h)
            h, residual = layer.post_attention_layernorm.forward_residual(
                h, residual, cg,
            )
            hidden = layer.mlp(h)
        normed, _ = self.norm.forward_residual(hidden, residual, cg)
        return normed


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(
        self, config: Qwen3_5Config, comm_group: CommGroup | None = None,
    ):
        super().__init__()
        self.config = config
        self.model = Qwen3_5LanguageModel(config, comm_group)
        # Gathers the vocab so the sampler is TP-oblivious. Its `[vocab/tp,
        # hidden]` shard matches the embedding's, so the tie works per rank.
        self.lm_head = ColumnParallelLinear(
            comm_group=self.model.comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, *, label: str,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden = self.model(
            self.model.embed_tokens(input_ids), label=label, cos_sin=cos_sin,
        )
        return self.lm_head(hidden)
