"""Qwen3.5's hybrid text stack: 3 gated-delta-net layers to 1 full-attention.

The two layer types index *different* resources — the recurrent pool is sized
by the linear layers and the KV cache by the full ones — so each layer is given
its position among its own kind, not its position in the stack. See
``Qwen3_5Config.resource_layer_index``.

Tensor parallelism shards by head throughout — q-heads for attention, k/v-heads
for the delta net — and the engine shards the KV cache and the recurrent pool to
match without being told, so ``get_node_resources`` stays unsharded. The
checkpoint keeps ``q/k/v`` and ``gate/up`` apart, so every projection maps to
one parallel linear and the weight loader's names stay one-to-one.

Two head counts are not the q-heads: ``num_key_value_heads`` is 4 or 2, so past
that degree the K/V heads replicate (see ``KVColumnParallelLinear``), and the
delta net's are its own (see ``GatedDeltaNet``).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components import Attention, DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    KVColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
from mstar.model.components.distributed.mlp import ParallelGatedMLPUnfused
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

    A cursor like the label and the layer index: the stack sets it once
    per step rather than threading cos/sin through every layer's forward,
    which would mean a decoder layer of our own.
    """

    def __init__(self):
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

    def set(self, cos: torch.Tensor | None, sin: torch.Tensor | None) -> None:
        self.cos, self.sin = cos, sin


class Qwen3_5Attention(Attention):
    """Full attention with Qwen3.5's output gate.

    ``q_proj`` carries the query and the gate **interleaved per head** — each
    head's block is ``[q(head_dim) | gate(head_dim)]``, not all queries then
    all gates — so it is chunked after the per-head view, never before. The
    gate multiplies the attention output before ``o_proj``.
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
            # the parent builds Llama-style; Qwen3.5's are Gemma-style too.
            # `head_dim` is a head *dim*, so it does not shard.
            eps = self.q_norm.variance_epsilon
            self.q_norm = RMSNorm(self.head_dim, eps=eps, gemma_mode=True)
            self.k_norm = RMSNorm(self.head_dim, eps=eps, gemma_mode=True)

        qkv_bias = self.q_proj.bias is not None
        o_bias = self.o_proj.bias is not None
        self.total_num_heads = self.num_heads
        self.num_heads = divide(self.num_heads, tp)

        # The gate rides inside each head's block — `[q(head_dim) | gate]` per
        # head — so sharding *across* heads leaves it intact and this is an
        # ordinary column shard at double width.
        self.q_proj = ColumnParallelLinear(
            comm_group=comm_group, input_size=self.input_hidden_size,
            output_size=self.total_num_heads * self.head_dim * (2 if output_gate else 1),
            bias=qkv_bias,
        )
        self.k_proj = KVColumnParallelLinear(
            comm_group=comm_group, input_size=self.input_hidden_size,
            head_size=self.head_dim, total_num_kv_heads=self.num_kv_heads,
            bias=qkv_bias,
        )
        self.v_proj = KVColumnParallelLinear(
            comm_group=comm_group, input_size=self.input_hidden_size,
            head_size=self.head_dim, total_num_kv_heads=self.num_kv_heads,
            bias=qkv_bias,
        )
        self.total_num_kv_heads = self.num_kv_heads
        self.num_kv_heads = self.k_proj.num_kv_heads
        self.o_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.input_hidden_size, bias=o_bias,
            input_is_parallel=True, reduce_results=True,
        )

    def _apply_rope(self, q, k, label):
        """Interleaved 3D MRoPE over the partial rotary dim.

        Falls back to the position resource's 1D RoPE when no cos/sin was
        set, which is what a text-only bring-up harness does.
        """
        if self.rope_cache is None or self.rope_cache.cos is None:
            return super()._apply_rope(q, k, label)
        return apply_partial_mrope(q, k, self.rope_cache.cos, self.rope_cache.sin)

    def consolidate_qkv_weight(self) -> None:
        if self.output_gate:
            # q_proj is twice as wide as the fused layout assumes
            return
        super().consolidate_qkv_weight()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.output_gate:
            return super().forward(hidden_states)
        num_tokens = hidden_states.shape[0]
        q, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                num_tokens, self.num_heads, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        k = self.k_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        q, k = self._apply_qk_norm(q, k)
        q, k = self._apply_rope(q, k, self.attend.label)
        out = self.attend(q, k, v).reshape(num_tokens, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate.reshape(num_tokens, -1))
        return self.o_proj(out)


def _norm(config: Qwen3_5Config) -> RMSNorm:
    """Qwen3.5's plain RMSNorm is Gemma-style: the checkpoint stores
    ``weight - 1`` and the norm scales by ``1 + weight``. Its *gated* norm
    is not — see ``RMSNormGated``.
    """
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_mode=True)


def _build_mlp(config: Qwen3_5Config, comm_group: CommGroup) -> nn.Module:
    # Unfused, so `gate_proj` / `up_proj` keep the checkpoint's own names and
    # the weight loader needs no stacked-shard rules.
    return ParallelGatedMLPUnfused(
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

        # Precomputed so the forward does no bookkeeping: which cursor each
        # layer advances, and the index it advances to.
        self._cursor_attr = [
            "mix" if kind == LINEAR_ATTENTION else "attend"
            for kind in config.layer_types
        ]
        self._resource_idx = [
            config.resource_layer_index(i) for i in range(config.num_hidden_layers)
        ]
        # one layer of each kind to bind the label through; the cursors live on
        # the shared resources, so binding once per kind covers the stack
        self._bind_through = {
            attr: self._cursor_attr.index(attr) for attr in set(self._cursor_attr)
        }

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
        # cos/sin are the same for every layer, so they ride a cursor too
        self.rope_cache.set(*(cos_sin or (None, None)))
        # The label and layer index are cursors on the shared resources: bind
        # the label once per resource kind, advance the index per layer.
        for attr, idx in self._bind_through.items():
            getattr(self.layers[idx].self_attn, attr).bind_step(label)
        for i, layer in enumerate(self.layers):
            getattr(layer.self_attn, self._cursor_attr[i]).set_layer_idx(
                self._resource_idx[i]
            )
            query_sequence = layer(hidden_states=query_sequence)
        return self.norm(query_sequence)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(
        self, config: Qwen3_5Config, comm_group: CommGroup | None = None,
    ):
        super().__init__()
        self.config = config
        self.model = Qwen3_5LanguageModel(config, comm_group)
        # All-gathers the vocab before returning, so the sampler stays
        # vocab-oblivious. Its shard is `[vocab/tp, hidden]`, the same shape
        # the vocab-parallel embedding holds — which is what lets the tie
        # below keep working per rank.
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
