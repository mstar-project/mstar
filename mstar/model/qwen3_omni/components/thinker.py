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

from typing import Optional, Tuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import ParallelSparseMoeBlock, RMSNorm
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.qwen3_omni.components.attention import Qwen3OmniAttention
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_KV, THINKER_POS, Qwen3OmniModelConfig


class Qwen3OmniThinkerLayer(nn.Module):
    """Single Thinker decoder layer: attention + MoE/dense MLP.

    Uses ``ParallelSparseMoeBlock`` for most layers, and a dense
    ``ParallelGatedMLP`` (SiLU SwiGLU) for layers in ``mlp_only_layers``.

    Args:
        config: top-level Qwen3-Omni model configuration.
        layer_idx: index of this layer in the stack.
        comm_group: TP communication group for MoE/MLP sharding.
    """

    def __init__(
        self, config: Qwen3OmniModelConfig, layer_idx: int,
        comm_group: CommGroup | None = None,
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
        if use_moe:
            self.mlp = ParallelSparseMoeBlock(
                hidden_size=tc.hidden_size,
                moe_intermediate_size=tc.moe_intermediate_size,
                num_experts=tc.num_experts,
                num_experts_per_tok=tc.num_experts_per_tok,
                norm_topk_prob=tc.norm_topk_prob,
                comm_group=comm_group,
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
        layer_idx: int | None = None,
        label: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE.
            layer_idx: this layer's index in the stack.
            label: the plan key this layer's attention runs against.

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
            layer_idx=layer_idx,
            label=label,
        )
        hidden_states = residual + hidden_states

        # Post-attention norm + MLP/MoE + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniThinkerTextModel(nn.Module):
    """Inner text model (maps to ``thinker.model.*`` in HF weights)."""

    def __init__(self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None):
        super().__init__()
        tc = config.thinker_text
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3OmniThinkerLayer(config, layer_idx=i, comm_group=comm_group)
            for i in range(tc.num_hidden_layers)
        ])
        self.norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)


class Qwen3OmniThinkerModel(nn.Module):
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

    def __init__(self, config: Qwen3OmniModelConfig, comm_group: CommGroup | None = None):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size
        self.num_layers = tc.num_hidden_layers
        self.accept_hidden_layer = config.accept_hidden_layer

        self.model = Qwen3OmniThinkerTextModel(config, comm_group=comm_group)

        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_embeds: [tokens, hidden_size] -- pre-embedded input
                (token embeddings possibly merged with multimodal features).
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE,
                e.g. [24, 20, 20].
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

        for layer_idx, decoder_layer in enumerate(self.model.layers):
            # NOTE: Set layer_idx here and pass in layer_idx=None so that inductor doesn't
            # try to specialize on the layer_idx int
            decoder_layer.self_attn.kv.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states,
                cos_sin_3d=cos_sin_3d,
                mrope_section=mrope_section,
                layer_idx=None,
                label=label,
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
