"""Dense Qwen3 causal language model from the shared components.

Qwen3 = Llama-style pre-norm decoder with GQA, per-head QK RMSNorm, SwiGLU
MLP and plain RoPE (``rope_theta`` 1e6, no Llama-3 scaling). Several
audio-LLM checkpoints ship one as their text decoder (Qwen3-ASR, Higgs-Audio,
Qwen3-Omni's dense layers); this is the one place to build it from
``ParallelAttention`` / ``ParallelGatedMLP`` / ``RMSNorm``, bound to the
KV, attention and position resources of the node that owns it.

Parameter paths mirror HF's ``Qwen3ForCausalLM`` with the ``model.`` prefix
dropped (``embed_tokens.*``, ``layers.N.*``, ``norm.*``, ``lm_head.*``); a
checkpoint that ties ``lm_head`` to the embedding loads without the head key.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm


@dataclass
class Qwen3LMConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, hf: dict) -> "Qwen3LMConfig":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in hf.items() if k in names}
        values.setdefault("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
        return cls(**values)


class Qwen3DenseLM(nn.Module):
    def __init__(
        self,
        config: Qwen3LMConfig,
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
                self_attn=ParallelAttention(
                    comm_group=comm_group,
                    hidden_size=config.num_attention_heads * config.head_dim,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    qk_norm=True,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    input_hidden_size=config.hidden_size,
                    attn_key=attn_key,
                    kv_key=kv_key,
                    pos_key=pos_key,
                ),
                mlp=ParallelGatedMLP(
                    comm_group=comm_group,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    activation="silu",
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

    def load_weights(self, weights) -> set[str]:
        """Stream HF ``(name, tensor)`` pairs (``model.`` prefix already
        dropped); tie the head to the embedding when the checkpoint does;
        raise on anything left unloaded."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        loaded = load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)
        if "lm_head.weight" not in loaded:
            if not self.config.tie_word_embeddings:
                raise RuntimeError("Qwen3 checkpoint has no lm_head.weight and does not tie embeddings")
            self.lm_head.weight_loader(self.lm_head.weight, self.embed_tokens.weight.data.clone())
            loaded.add("lm_head.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"Qwen3 checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}"
            )
        return loaded

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm.

        The label and layer index are cursors on the shared resources: bind
        the label once, advance the index per layer. Passing them as
        arguments instead would make inductor specialize on the int.
        """
        hidden_states = input_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states=hidden_states)
        return self.norm(hidden_states)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
