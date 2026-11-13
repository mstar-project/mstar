"""Native Ministral 3 decoder stack, composed from M*'s shared components.

This component owns tensor computation. The model wrapper declares its resources
and graph walks; the engine owns request state and execution.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.position.config import YaRNConfig
from mstar.model.components.decoder_layer import DecoderLayer, run_layers
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.ministral3.components.attention import Ministral3Attention


@dataclass(frozen=True)
class Ministral3LMConfig:
    """Text architecture; defaults describe the 3B variant, with an explicit head dim."""

    hidden_size: int = 3072
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    vocab_size: int = 131072
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    rope_theta: float = 1_000_000.0
    yarn: YaRNConfig = field(default_factory=lambda: YaRNConfig(
        factor=16, original_max_position_embeddings=16384,
    ))
    llama_4_scaling_beta: float = 0.1

    def __post_init__(self):
        if self.num_hidden_layers < 1:
            raise ValueError("Ministral 3 requires at least one decoder layer")


class Ministral3DenseLM(nn.Module):
    """Packed tokens -> hidden states -> logits, using resources bound to its layers.

    ``embed``, ``forward`` and ``logits`` follow the shared sampled-token decoder's
    interface. The stack holds no request state. Its position resource must use
    ``config.rope_theta``, ``config.head_dim`` and ``config.yarn``; that same YaRN
    original context length controls the model's query scaling.
    """

    def __init__(
        self,
        config: Ministral3LMConfig,
        *,
        attn_key: str,
        kv_key: str,
        pos_key: str,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        self.config = config
        comm_group = comm_group or CommGroup.trivial()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=comm_group,
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                self_attn=Ministral3Attention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    original_max_position_embeddings=config.yarn.original_max_position_embeddings,
                    llama_4_scaling_beta=config.llama_4_scaling_beta,
                    comm_group=comm_group,
                    attn_key=attn_key,
                    kv_key=kv_key,
                    pos_key=pos_key,
                ),
                mlp=ParallelGatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    activation="silu",
                    comm_group=comm_group,
                ),
                input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )
        if config.tie_word_embeddings:
            # Both parameters shard the vocabulary in the same order.
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        self.layers[0].self_attn.attend.bind_step(label)
        return run_layers(self.layers, input_embeds, self.norm)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
