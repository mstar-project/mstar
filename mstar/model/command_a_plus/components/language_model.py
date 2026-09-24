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

    Both branches use the same tensor-parallel group and independently reduce
    their outputs before averaging. Inheriting the routed branch preserves the
    ``gate`` and ``experts`` parameter paths used by checkpoint loading.
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
        )
        self.shared_experts = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_intermediate_size,
            comm_group=self.comm_group,
            activation="silu",
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
        *,
        return_router_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
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
        self.input_layernorm = CommandAPlusLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
        )
        self.self_attn = CommandAPlusAttention(config, layer_idx, comm_group)
        self.mlp = CommandAPlusMoeBlock(config, comm_group)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normalized)
        moe_output = self.mlp(normalized)
        return residual + attn_output + moe_output


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

    def forward(self, query_sequence: torch.Tensor, *, label: str) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            # Both attention managers need the current label. The shared KV
            # cache uses the actual transformer index, not an index within
            # the local/global subset of layers.
            layer.self_attn.attend.bind_step(label)
            layer.self_attn.attend.set_layer_idx(layer_idx)
            query_sequence = layer(query_sequence)
        return self.norm(query_sequence)


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
        return logits * self.logit_scale

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from mstar.model.command_a_plus.weight_loading import load_command_a_plus_weights

        return load_command_a_plus_weights(self, weights, config=self.config)
