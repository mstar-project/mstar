"""Thinker MoE transformer with FlashInfer paged KV cache for Qwen3-Omni.

Architecture: embed_tokens -> N decoder layers -> final RMSNorm -> lm_head
Each decoder layer: input_layernorm -> attention -> residual -> post_attention_layernorm -> MLP/MoE -> residual

Key differences from Orpheus:
- MoE (SparseMoeBlock) on most layers, dense MLP on ``mlp_only_layers``
- QK-norm in attention (handled by ``Qwen3OmniAttention``)
- 3D MRoPE (passed through as ``cos_sin_3d``)
- Captures layer-0 embeddings and layer-N hidden states for Talker conditioning

Weight name prefix: ``thinker.``
  - thinker.model.embed_tokens.weight
  - thinker.model.layers.{i}.input_layernorm.weight
  - thinker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  - thinker.model.layers.{i}.self_attn.{q,k}_norm.weight
  - thinker.model.layers.{i}.block_sparse_moe.gate.weight
  - thinker.model.layers.{i}.block_sparse_moe.experts.{j}.{gate,up,down}_proj.weight
  - thinker.model.layers.{i}.mlp.{gate,up,down}_proj.weight (dense layers)
  - thinker.model.norm.weight
  - thinker.lm_head.weight
"""

import functools
import re
from typing import Optional, Tuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import (
    MoeParallelConfig,
    MoeParallelMixin,
    RMSNorm,
    instantiate_moe_block,
)
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.loader.base import CheckpointRules, StackedParamRule
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_KV, THINKER_POS, Qwen3OmniModelConfig


def thinker_remap(name: str) -> str | None:
    """HF checkpoint names (after the ``thinker.`` prefix) as this model's parameters.

    The ``block_sparse_moe`` -> ``mlp`` rename, and each expert's
    ``experts.{N}.{gate,up,down}_proj.weight`` rewritten to
    ``experts.{proj}.__expert{N}__.weight``, which ``thinker_stacked_params`` routes
    into the fused expert tensors with a shard id naming the projection and expert.
    """
    if "rotary_emb" in name:
        return None
    name = name.replace("block_sparse_moe.", "mlp.")
    m = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
    if m:
        prefix, expert_idx, proj = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.weight"
    return name


@functools.cache
def thinker_stacked_params(num_experts: int) -> tuple[StackedParamRule, ...]:
    """The fused-parameter routing: per-expert projections into the packed expert
    tensors (first: the dense ``.gate_proj`` suffix would also match a remapped expert
    key, and the first matching rule wins), then dense gate/up and q/k/v."""
    rules = []
    for i in range(num_experts):
        # the source suffix includes ".weight", so the target (a bare parameter, not a
        # Linear) has none
        rules.append(StackedParamRule(".experts.gate_up_proj", f".experts.gate_proj.__expert{i}__.weight", f"gate:{i}"))
        rules.append(StackedParamRule(".experts.gate_up_proj", f".experts.up_proj.__expert{i}__.weight", f"up:{i}"))
        rules.append(StackedParamRule(".experts.down_proj", f".experts.down_proj.__expert{i}__.weight", f"down:{i}"))
    rules += [
        StackedParamRule(".gate_up_proj", ".gate_proj", 0),
        StackedParamRule(".gate_up_proj", ".up_proj", 1),
        StackedParamRule(".qkv_proj", ".q_proj", "q"),
        StackedParamRule(".qkv_proj", ".k_proj", "k"),
        StackedParamRule(".qkv_proj", ".v_proj", "v"),
    ]
    return tuple(rules)


class Qwen3OmniThinkerLayer(nn.Module):
    """Single Thinker decoder layer: attention + MoE/dense MLP.

    Uses the MoE block ``instantiate_moe_block`` picks for ``moe_parallel``
    for most layers, and a dense ``ParallelGatedMLP`` (SiLU SwiGLU) for
    layers in ``mlp_only_layers``.

    Args:
        config: top-level Qwen3-Omni model configuration.
        layer_idx: index of this layer in the stack.
        comm_group: TP communication group for MoE/MLP sharding.
        moe_parallel: TP or EP for the routed experts; TP when None.
    """

    def __init__(
        self, config: Qwen3OmniModelConfig, layer_idx: int,
        comm_group: CommGroup | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size

        # Pre-attention layernorm
        self.input_layernorm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

        # Self-attention with QK-norm and 3D MRoPE
        self.self_attn = Qwen3OmniAttention(
            hidden_size=tc.hidden_size,
            num_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            rope_theta=tc.rope_theta,
            rms_norm_eps=tc.rms_norm_eps,
            use_mrope=True,
            comm_group=comm_group,
            attn_key=THINKER_ATTN,
            kv_key=THINKER_KV,
            pos_key=THINKER_POS
        )

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )

        # MoE or dense MLP depending on layer index.
        use_moe = (
            layer_idx not in tc.mlp_only_layers
            and tc.num_experts > 0
            and (layer_idx + 1) % tc.decoder_sparse_step == 0
        )
        self.is_moe = use_moe
        if use_moe:
            self.mlp = instantiate_moe_block(
                hidden_size=tc.hidden_size,
                moe_intermediate_size=tc.moe_intermediate_size,
                num_experts=tc.num_experts,
                num_experts_per_tok=tc.num_experts_per_tok,
                norm_topk_prob=tc.norm_topk_prob,
                comm_group=comm_group,
                moe_parallel=moe_parallel,
            )
        else:
            self.mlp = ParallelGatedMLP(
                hidden_size=tc.hidden_size,
                intermediate_size=tc.intermediate_size,
                activation="silu",
                comm_group=comm_group,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE.
            token_valid: [tokens] bool; False rows are padding the MoE skips.

        Returns:
            hidden_states: [tokens, hidden_size]
        """
        # Pre-attention norm + self-attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
        )
        hidden_states = residual + hidden_states

        # Post-attention norm + MLP/MoE + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if token_valid is not None and self.is_moe:
            hidden_states = self.mlp(hidden_states, token_valid=token_valid)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniThinkerTextModel(nn.Module):
    """Inner text model (maps to ``thinker.model.*`` in HF weights)."""

    def __init__(
        self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3OmniThinkerLayer(config, layer_idx=i, comm_group=comm_group, moe_parallel=moe_parallel)
            for i in range(tc.num_hidden_layers)
        ])
        self.norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)


class Qwen3OmniThinkerModel(MoeParallelMixin, nn.Module):
    """Thinker: MoE transformer backbone for Qwen3-Omni.

    HF weight layout::

        thinker.model.embed_tokens.weight
        thinker.model.layers.{i}.*
        thinker.model.norm.weight
        thinker.lm_head.weight

    Produces:
    - Final hidden states (after all layers + final norm) for text logits
    - Layer-0 embeddings (before any transformer layers) for Talker conditioning
    - Layer-N hidden states (``accept_hidden_layer``) for Talker conditioning
    """

    def __init__(
        self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size
        self.num_layers = tc.num_hidden_layers
        self.num_experts = tc.num_experts
        self.accept_hidden_layer = config.accept_hidden_layer

        self.model = Qwen3OmniThinkerTextModel(config, comm_group=comm_group, moe_parallel=moe_parallel)

        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

        self.init_moe_parallel(moe_parallel)

    def checkpoint_rules(self) -> CheckpointRules:
        """How the full model's checkpoint (``thinker.*``) lands here, and where LoRA
        adapters land."""
        return CheckpointRules(
            stacked_params=thinker_stacked_params(self.num_experts), remap=thinker_remap,
            prefix="thinker.",
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_embeds: torch.Tensor
    ):
        # NOTE: must ensure that visual_embeds is the same shape as hidden_states,
        # and zero where we do not have visual tokens!!
        hidden_states += visual_embeds
        return hidden_states

    def forward(
        self,
        input_embeds: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        label: str = "main",
        num_valid_tokens: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_embeds: [tokens, hidden_size] -- pre-embedded input
                (token embeddings possibly merged with multimodal features).
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE,
                e.g. [24, 20, 20].
            num_valid_tokens: ``(1,)`` real token count; rows past it are
                CUDA-graph padding. Used only under ``mask_padding``.
            label: the plan key every layer runs against. The stream's
                stored length and position counter advance when the runner
                commits the declared step (vision prefill declares its
                MRoPE 3D-grid span there).

        Returns:
            hidden_states: [tokens, hidden_size] -- final normed hidden states
            layer_0_embed: [tokens, hidden_size] -- input before any layers
            layer_n_hidden: [tokens, hidden_size] or None -- hidden states
                after ``accept_hidden_layer`` (for Talker conditioning)
        """
        hidden_states = input_embeds

        # Capture input embeddings BEFORE any transformer layers
        layer_0_embed = hidden_states.clone()
        layer_n_hidden = None
        token_valid = self.moe_token_valid(hidden_states, num_valid_tokens)

        # The label and layer index are cursors on the shared resources: bind
        # the label once, advance the index per layer. Passing them as
        # arguments instead would make inductor specialize on the int.
        self.model.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            decoder_layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states,
                cos_sin_3d=cos_sin_3d,
                mrope_section=mrope_section,
                token_valid=token_valid,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    deepstack_visual_embeds[layer_idx],
                )

            # Capture hidden states at the accept_hidden_layer for Talker
            if layer_idx == self.accept_hidden_layer:
                layer_n_hidden = hidden_states.clone()

        # Final layer norm
        hidden_states = self.model.norm(hidden_states)

        return hidden_states, layer_0_embed, layer_n_hidden
