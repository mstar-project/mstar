"""Phi-4-mini, Phi-4-multimodal's LLM, from the shared components.

A Llama-style pre-norm decoder: GQA, SwiGLU, RMSNorm, a tied LM head, and RoPE
over the first 3/4 of each head with LongRoPE scaling. The rotation is the
position resource's (``PositionConfig(rotary_dim=..., longrope=...)``), so the
layers are plain ``ParallelAttention``.

The checkpoint stores ``qkv_proj`` and ``gate_up_proj`` fused, under PEFT's
``base_layer`` names, beside the modality LoRAs (``lora_A.vision`` ...). Only the
base weights load here; the LoRAs are served from their PEFT directories by the
engine's LoRA pool (``mstar/model/phi4mm/lora.py``).
"""

from __future__ import annotations

import re

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import DecoderLayer, run_layers
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.loader.base import CheckpointRules
from mstar.model.phi4mm.config import ATTN, KV_CACHE, ROPE, Phi4MMConfig

# the LLM's tensors in the checkpoint: everything under `model.` but the media towers,
# with PEFT's `.base_layer` dropped and its LoRA tensors left to the pool
_LORA = re.compile(r"\.lora_[AB]\.")


def _checkpoint_name(name: str) -> str | None:
    if name.startswith("embed_tokens_extend.") or _LORA.search(name):
        return None
    return name.replace(".base_layer.", ".")


class Phi4MiniLM(nn.Module):
    def __init__(self, config: Phi4MMConfig, comm_group: CommGroup | None = None):
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
                    qkv_bias=False,
                    o_bias=False,
                    rope_theta=config.rope_theta,
                    input_hidden_size=config.hidden_size,
                    attn_key=ATTN,
                    kv_key=KV_CACHE,
                    pos_key=ROPE,
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

    @staticmethod
    def checkpoint_rules() -> CheckpointRules:
        return CheckpointRules(remap=_checkpoint_name, prefix="model.")

    def load_weights(self, weights) -> set[str]:
        """Stream the checkpoint's ``(name, tensor)`` pairs; the head is tied to the
        embedding; raise on anything left unloaded."""
        rules = self.checkpoint_rules()
        loaded = rules.load(self, ((n[len(rules.prefix):], t) for n, t in weights if n.startswith(rules.prefix)))
        if "lm_head.weight" not in loaded:
            if not self.config.tie_word_embeddings:
                raise RuntimeError("Phi-4 checkpoint has no lm_head.weight and does not tie embeddings")
            # both shard the vocab by TP rank over the same rows
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
            loaded.add("lm_head.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(f"Phi-4 checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")
        return loaded

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm."""
        self.layers[0].self_attn.attend.bind_step(label)
        return run_layers(self.layers, input_embeds, self.norm)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
