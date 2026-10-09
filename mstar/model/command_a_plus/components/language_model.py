import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    CommandAPlusTextConfig,
)
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.distributed.embedding import VocabParallelEmbedding
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.moe import ParallelSparseMoeBlock


class CommandAPlusLayerNorm(nn.Module):
    """Bias-free LayerNorm with FP32 statistics and learned scaling."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * normalized).to(input_dtype)


class CommandAPlusRouter(nn.Module):
    """Normalized sigmoid top-k router implementing M*'s stateless contract."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self,
        x: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del router_states  # This router carries no state between calls.

        logits = F.linear(x, self.weight)  # [tokens, hidden] -> [tokens, experts]
        top_k_logits, top_k_indices = torch.topk(logits, k=self.num_experts_per_tok, dim=-1)
        weights = torch.sigmoid(top_k_logits)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(x.dtype), top_k_indices, None


class CommandAPlusMoeBlock(ParallelSparseMoeBlock):
    """Average sigmoid-routed experts with an always-active shared SwiGLU MLP.

    Both branches use the same tensor-parallel group. Neither reduces on its
    own: the average is a per-rank partial that ``CommandAPlusDecoderLayer``
    all-reduces once, together with attention's. Inheriting the routed branch
    preserves the ``gate`` and ``experts`` parameter paths used by checkpoint
    loading.
    """

    def __init__(
        self,
        config: CommandAPlusTextConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.intermediate_size,
            router=CommandAPlusRouter(
                config.hidden_size, config.num_experts, config.num_experts_per_tok,
            ),
            comm_group=comm_group,
            reduce_results=False,
        )
        self.shared_experts = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_intermediate_size,
            comm_group=self.comm_group,
            activation="silu",
            bias=False,
            reduce_results=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """This rank's partial of ``(routed + shared) / 2``, NOT reduced."""
        routed, next_state = super().forward(
            hidden_states, router_states, return_router_states=True,
        )
        shared = self.shared_experts(hidden_states)
        output = (routed + shared) / 2
        return (output, next_state) if return_router_states else output


class CommandAPlusAttention(ParallelAttention):
    """Shared TP attention with unscaled, interleaved RoPE on local layers."""

    def __init__(
        self,
        config: CommandAPlusTextConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        is_local = config.layer_types[layer_idx] == "sliding_attention"
        super().__init__(
            # the decoder layer reduces this partial with the MoE branch's
            reduce_results=False,
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=False,
            rope_theta=config.rope_theta,
            attn_key=LOCAL_ATTN if is_local else GLOBAL_ATTN,
            kv_key=KV_CACHE,
            pos_key=ROPE if is_local else None,
        )

    def bind_resources(self, resources: dict) -> None:
        # Missing local RoPE must fail instead of silently disabling positions.
        required = [self._attn_key, self._kv_key]
        if self._pos_key is not None:
            required.append(self._pos_key)
        for key in required:
            if resources.get(key) is None:
                raise ValueError(f"Command A+ attention requires resource {key!r}")
        super().bind_resources(resources)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pos_key is None:
            return q, k
        return self.pos.apply_qk(
            q, k, label=label,
            rotary_dim=self.head_dim,
            interleave=True,
            rope_theta=self.rope_theta,
            rope_scale=1.0,
            rope_dtype=q.dtype,
        )


class CommandAPlusDecoderLayer(nn.Module):
    """Parallel residual block: x + Attention(LN(x)) + MoE(LN(x))."""

    def __init__(
        self,
        config: CommandAPlusTextConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group if comm_group is not None else CommGroup.trivial()
        self.input_layernorm = CommandAPlusLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
        )
        self.self_attn = CommandAPlusAttention(config, layer_idx, comm_group)
        self.mlp = CommandAPlusMoeBlock(config, comm_group)
        self.input_weight: torch.Tensor | None = None
        self.output_weight: torch.Tensor | None = None

    @torch.no_grad()
    def fuse_weights(self) -> None:
        """Concatenate the projections that share an input into two GEMMs.

        Attention's QKV, the shared MLP's gate/up and the router all read the
        normalized input, so they become one ``[qkv | gate_up | router]``
        column GEMM. Attention's o_proj and the shared MLP's down projection
        are both row-parallel into the residual, so they become one GEMM over
        ``[attn_out | shared_act]``. The original parameters are rebound as
        views of the fused storage, so no weight is held twice.
        """
        attn, moe = self.self_attn, self.mlp
        columns = [attn.qkv_proj.weight, moe.shared_experts.gate_up_proj.weight, moe.gate.weight]
        self.input_weight = torch.cat([w.data for w in columns], dim=0)
        start = 0
        for w in columns:
            w.data = self.input_weight[start:start + w.shape[0]]
            start += w.shape[0]
        self._input_splits = [w.shape[0] for w in columns]

        rows = [attn.o_proj.weight, moe.shared_experts.down_proj.weight]
        self.output_weight = torch.cat([w.data for w in rows], dim=1)
        start = 0
        for w in rows:
            w.data = self.output_weight[:, start:start + w.shape[1]]
            start += w.shape[1]
        self._attn_width = rows[0].shape[1]

    def _fused_branches(self, normalized: torch.Tensor) -> torch.Tensor:
        """This rank's unreduced ``attn + (routed + shared) / 2``, five GEMMs in two."""
        from mstar.model.command_a_plus.kernels import (
            ROUTE_ALIGN_MAX_SLOTS,
            moe_combine,
            route_align,
            sigmoid_topk,
            silu_mul_into,
            splitk_linear,
            splitk_supported,
        )
        from mstar.utils.fused_moe import fused_experts, moe_block_m

        attn, moe = self.self_attn, self.mlp
        tokens = normalized.shape[0]
        projected = F.linear(normalized, self.input_weight)
        qkv, gate_up, router_logits = projected.split(self._input_splits, dim=-1)
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(tokens, attn.num_heads, attn.head_dim)
        k = k.view(tokens, attn.num_kv_heads, attn.head_dim)
        v = v.view(tokens, attn.num_kv_heads, attn.head_dim)
        q, k = attn._apply_rope(q, k, attn.attend.label)

        attn_out = attn.attend(q, k, v).reshape(tokens, -1)
        shared_act = torch.empty(
            (tokens, self.output_weight.shape[1] - self._attn_width),
            dtype=normalized.dtype, device=normalized.device,
        )
        # The 1/2 of the shared/routed average is a power of two, so applying it
        # to the activation is exact and lets one GEMM produce attn + shared / 2.
        silu_mul_into(gate_up, shared_act, scale=0.5)

        top_k, w1 = moe.num_experts_per_tok, moe.experts.gate_up_proj
        alignment = None
        if tokens * top_k <= ROUTE_ALIGN_MAX_SLOTS and (top_k & (top_k - 1)) == 0:
            weights, ids, alignment = route_align(
                router_logits, top_k, moe_block_m(tokens, w1, top_k), normalized.dtype,
            )
        else:
            weights, ids = sigmoid_topk(router_logits, top_k, normalized.dtype)
        routed = fused_experts(
            normalized, w1, moe.experts.down_proj, weights, ids,
            reduce_results=False, alignment=alignment,
        )
        if splitk_supported((attn_out, shared_act), self.output_weight):
            row_output = splitk_linear((attn_out, shared_act), self.output_weight)
        else:
            row_output = F.linear(torch.cat((attn_out, shared_act), dim=-1), self.output_weight)
        return moe_combine(row_output, routed, scale=0.5)

    def branches(self, normalized: torch.Tensor) -> torch.Tensor:
        """This rank's unreduced partial of ``attn(x) + moe(x)`` for normalized ``x``."""
        if self.input_weight is not None:
            return self._fused_branches(normalized)
        partial = self.self_attn(normalized)
        partial += self.mlp(normalized)
        return partial

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Both branches run on the same input, so their partials can be summed
        on-rank and reduced together: one all-reduce per layer instead of three
        (attention's o_proj, the routed experts', and the shared MLP's). The
        collectives here are [tokens, 4096] bf16 and latency-bound, so at decode
        batch sizes the count matters far more than the bytes. Reassociating the
        sum is exact in real arithmetic and reorders bf16 rounding; prefill and
        decode both go through here, so the two stay consistent."""
        normalized = self.input_layernorm(hidden_states)
        return hidden_states + self.comm_group.all_reduce(self.branches(normalized))


class CommandAPlusLanguageModel(nn.Module):
    """Text backbone over packed token embeddings, returning final hidden states.

    As in Orpheus, the node's preprocessing calls ``embed_tokens`` separately.
    ``forward`` takes ``[total_step_tokens, hidden_size]``; resource plans carry
    request boundaries, positions and cached history. The causal-LM wrapper
    provides tied output logits and checkpoint loading.
    """

    def __init__(
        self,
        config: CommandAPlusTextConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group if comm_group is not None else CommGroup.trivial()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=self.comm_group,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList([
            CommandAPlusDecoderLayer(config, layer_idx, self.comm_group)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CommandAPlusLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fused = False

    def fuse_for_inference(self, max_allreduce_tokens: int = 2048) -> None:
        """Switch to fused kernels and concatenated GEMMs. CUDA only, after loading.

        Under TP this is collective: every rank must call it, because it also
        sets up the one-shot NVLink all-reduce workspace.
        """
        for layer in self.layers:
            layer.fuse_weights()
        self.fused = True
        self._allreduce_workspace = None
        group = self.comm_group
        if group.world_size > 1:
            # A replicated table makes the lookup a plain gather. The sharded
            # lookup all-reduces rows that are zero on every other rank, so the
            # result is bit-identical.
            shard = self.embed_tokens.weight.data
            self._replicated_embedding = torch.empty(
                (shard.shape[0] * group.world_size, shard.shape[1]), dtype=shard.dtype, device=shard.device,
            )
            torch.distributed.all_gather_into_tensor(
                self._replicated_embedding, shard.contiguous(), group=group.device_group,
            )
        if group.world_size > 1 and not os.environ.get("COMMAND_A_PLUS_NCCL_ALLREDUCE"):
            from flashinfer.comm import create_allreduce_fusion_workspace

            weight = self.embed_tokens.weight
            self._allreduce_workspace = create_allreduce_fusion_workspace(
                backend="trtllm", world_size=group.world_size, rank=group.rank,
                max_token_num=max_allreduce_tokens, hidden_dim=weight.shape[1],
                dtype=weight.dtype, group=group.device_group,
            )
            self._allreduce_max_tokens = max_allreduce_tokens

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        table = getattr(self, "_replicated_embedding", None)
        if table is None:
            return self.embed_tokens(input_ids)
        return F.embedding(input_ids, table)

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Small reductions (decode, short prefill) take FlashInfer's one-shot
        NVLink all-reduce, ~4x lower latency than NCCL at decode sizes; larger
        ones go through NCCL. Both give every rank identical sums."""
        workspace = getattr(self, "_allreduce_workspace", None)
        if workspace is None or x.shape[0] > self._allreduce_max_tokens:
            return self.comm_group.all_reduce(x)
        from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

        out = torch.empty_like(x)
        allreduce_fusion(
            input=x, workspace=workspace, pattern=AllReduceFusionPattern.kAllReduce,
            output=out, fp32_acc=True,
        )
        return out

    def forward(self, query_sequence: torch.Tensor, *, label: str) -> torch.Tensor:
        if self.fused:
            return self._fused_forward(query_sequence, label=label)
        for layer_idx, layer in enumerate(self.layers):
            # Both attention managers need the current label. The shared KV
            # cache uses the actual transformer index, not an index within
            # the local/global subset of layers.
            layer.self_attn.attend.bind_step(label)
            layer.self_attn.attend.set_layer_idx(layer_idx)
            query_sequence = layer(query_sequence)
        return self.norm(query_sequence)

    def _fused_forward(self, query_sequence: torch.Tensor, *, label: str) -> torch.Tensor:
        """Same math as ``forward``, with each layer's residual add fused into
        the next layer's LayerNorm."""
        from mstar.model.command_a_plus.kernels import add_layernorm

        norms = [layer.input_layernorm for layer in self.layers] + [self.norm]
        residual, normalized = add_layernorm(query_sequence, None, norms[0].weight, norms[0].eps)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.bind_step(label)
            layer.self_attn.attend.set_layer_idx(layer_idx)
            delta = self._all_reduce(layer.branches(normalized))
            norm = norms[layer_idx + 1]
            residual, normalized = add_layernorm(residual, delta, norm.weight, norm.eps)
        return normalized


class CommandAPlusForCausalLM(nn.Module):
    def __init__(
        self,
        config: CommandAPlusTextConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = CommandAPlusLanguageModel(config, comm_group)
        self.logit_scale = config.logit_scale

    def forward(
        self,
        query_sequence: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        return self.model(query_sequence, label=label)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # [T, H] @ [V / tp, H].T -> [T, V / tp]
        local_logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        logits = self.model.comm_group.all_gather(local_logits, dim=-1)
        return logits if self.logit_scale == 1.0 else logits * self.logit_scale

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from mstar.model.command_a_plus.weight_loading import load_command_a_plus_weights

        return load_command_a_plus_weights(self, weights, config=self.config)
