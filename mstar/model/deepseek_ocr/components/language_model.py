"""DeepSeek-V2 MoE decoder with plain multi-head attention, as DeepSeek-OCR ships it.

Parameter paths match the checkpoint with ``model.`` dropped (``embed_tokens``,
``layers.N.*``, ``norm``) plus ``lm_head``; per-expert ``gate/up/down_proj``
load into the fused expert tensors the Triton MoE kernel reads.
"""

import re
from collections.abc import Iterable

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
from mstar.model.components.moe import ParallelSparseMoeBlock, TopKRouter
from mstar.model.components.norm import RMSNorm
from mstar.model.deepseek_ocr.config import ATTN, KV_CACHE, ROPE, DeepseekOCRTextConfig

_EXPERT_KEY = re.compile(r"^(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


class DeepseekOCRAttention(ParallelAttention):
    """Llama attention: rotate-half RoPE over the whole head, no scaling."""

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos.apply_qk(
            q, k, label=label, rotary_dim=self.head_dim, interleave=False,
            rope_theta=self.rope_theta, rope_scale=1.0,
        )


class DeepseekOCRMoE(ParallelSparseMoeBlock):
    """Routed experts (softmax top-k, weights not renormalized) plus the
    always-on shared experts, summed."""

    def __init__(self, config: DeepseekOCRTextConfig, comm_group: CommGroup | None = None):
        if config.routed_scaling_factor != 1.0:
            raise ValueError("DeepSeek-OCR: routed_scaling_factor other than 1.0 is not ported")
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            router=TopKRouter(
                hidden_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
            ),
            comm_group=comm_group,
        )
        self.shared_experts = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_intermediate_size,
            comm_group=self.comm_group,
            activation="silu",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states) + self.shared_experts(hidden_states)


class DeepseekOCRLanguageModel(nn.Module):
    def __init__(self, config: DeepseekOCRTextConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        comm_group = comm_group or CommGroup.trivial()
        self.comm_group = comm_group
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, comm_group=comm_group,
        )
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if config.is_moe_layer(layer_idx):
                mlp = DeepseekOCRMoE(config, comm_group)
            else:
                mlp = ParallelGatedMLP(
                    hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                    comm_group=comm_group, activation="silu",
                )
            self.layers.append(DecoderLayer(
                self_attn=DeepseekOCRAttention(
                    comm_group=comm_group,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rope_theta=config.rope_theta,
                    attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE,
                ),
                mlp=mlp,
                input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            ))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group, input_size=config.hidden_size, output_size=config.vocab_size,
            bias=False, gather_output=True,
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm.

        The label and layer index are cursors on the shared resources: bind the
        label once, advance the index per layer.
        """
        hidden = input_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden = layer(hidden)
        return self.norm(hidden)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """HF ``(name, tensor)`` pairs with ``model.`` already dropped; raises on
        anything left unloaded."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        params = dict(self.named_parameters())
        # every expert's slice of a fused tensor: gate and up per expert, or down
        expert_slices: dict[str, set[tuple[str, str]]] = {}

        def dense(pairs):
            for name, tensor in pairs:
                match = _EXPERT_KEY.match(name)
                if match is None:
                    yield name, tensor
                    continue
                prefix, expert, kind = match.groups()
                target = f"{prefix}.{'down_proj' if kind == 'down' else 'gate_up_proj'}"
                params[target].weight_loader(params[target], tensor, f"{kind}:{expert}")
                expert_slices.setdefault(target, set()).add((kind, expert))

        loaded = load_hf_weights(self, dense(weights), stacked_params=LLAMA_STACKED_PARAMS)
        experts = self.config.n_routed_experts
        loaded |= {
            target for target, slices in expert_slices.items()
            if len(slices) == (experts if target.endswith("down_proj") else 2 * experts)
        }
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(f"DeepSeek-OCR decoder left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")
        return loaded
