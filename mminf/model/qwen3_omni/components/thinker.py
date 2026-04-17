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

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.qwen3_omni.components.attention import (
    Qwen3OmniAttention,
    Qwen3OmniRMSNorm,
)
from mminf.model.qwen3_omni.components.moe import (
    Qwen3OmniMLP,
    Qwen3OmniSparseMoeBlock,
)
from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig
from mminf.utils.flashinfer_utils import run_rms_norm


class Qwen3OmniThinkerLayer(nn.Module):
    """Single Thinker decoder layer: attention + MoE/dense MLP.

    Uses MoE (``Qwen3OmniSparseMoeBlock``) for most layers, and a dense
    ``Qwen3OmniMLP`` for layers listed in ``mlp_only_layers``.

    Args:
        config: top-level Qwen3-Omni model configuration.
        layer_idx: index of this layer in the stack.
    """

    def __init__(self, config: Qwen3OmniModelConfig, layer_idx: int):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size

        # Pre-attention layernorm
        self.input_layernorm = Qwen3OmniRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

        # Self-attention with QK-norm and 3D MRoPE
        self.self_attn = Qwen3OmniAttention(
            hidden_size=tc.hidden_size,
            num_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            rope_theta=tc.rope_theta,
            rms_norm_eps=tc.rms_norm_eps,
            use_mrope=True,
        )

        # Post-attention layernorm
        self.post_attention_layernorm = Qwen3OmniRMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )

        # MoE or dense MLP depending on layer index.
        # HF condition: use MoE when not in mlp_only_layers AND
        # num_experts > 0 AND (layer_idx + 1) % decoder_sparse_step == 0.
        use_moe = (
            layer_idx not in tc.mlp_only_layers
            and tc.num_experts > 0
            and (layer_idx + 1) % tc.decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = Qwen3OmniSparseMoeBlock(
                hidden_size=tc.hidden_size,
                moe_intermediate_size=tc.moe_intermediate_size,
                num_experts=tc.num_experts,
                num_experts_per_tok=tc.num_experts_per_tok,
                norm_topk_prob=tc.norm_topk_prob,
            )
        else:
            self.mlp = Qwen3OmniMLP(
                hidden_size=tc.hidden_size,
                intermediate_size=tc.intermediate_size,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cache_handle: BatchedCacheManager with pre-planned attention.
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE.

        Returns:
            hidden_states: [tokens, hidden_size]
        """
        # Pre-attention norm + self-attention + residual
        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states,
            self.input_layernorm.weight,
            eps=self.input_layernorm.variance_epsilon,
        )
        hidden_states = self.self_attn(
            hidden_states,
            cache_handle=cache_handle,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
        )
        hidden_states = residual + hidden_states

        # Post-attention norm + MLP/MoE + residual
        residual = hidden_states
        hidden_states = run_rms_norm(
            hidden_states,
            self.post_attention_layernorm.weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniThinkerTextModel(nn.Module):
    """Inner text model (maps to ``thinker.model.*`` in HF weights)."""

    def __init__(self, config: Qwen3OmniModelConfig):
        super().__init__()
        tc = config.thinker_text
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3OmniThinkerLayer(config, layer_idx=i) for i in range(tc.num_hidden_layers)]
        )
        self.norm = Qwen3OmniRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)


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

    def __init__(self, config: Qwen3OmniModelConfig):
        super().__init__()
        tc = config.thinker_text

        self.hidden_size = tc.hidden_size
        self.num_layers = tc.num_hidden_layers
        self.accept_hidden_layer = config.accept_hidden_layer

        # Inner text model: embed_tokens + layers + norm
        # Maps to thinker.model.* in HF weights
        self.model = Qwen3OmniThinkerTextModel(config)

        # Language model head (at top level: thinker.lm_head)
        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_embeds: [tokens, hidden_size] -- pre-embedded input
                (token embeddings possibly merged with multimodal features).
            cache_handle: BatchedCacheManager with pre-planned attention
                and RoPE.
            cos_sin_3d: (cos, sin) for 3D MRoPE, each [tokens, head_dim].
            mrope_section: section sizes for interleaved 3D MRoPE,
                e.g. [24, 20, 20].

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
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states,
                cache_handle=cache_handle,
                cos_sin_3d=cos_sin_3d,
                mrope_section=mrope_section,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            # Capture hidden states at the accept_hidden_layer for Talker
            if layer_idx == self.accept_hidden_layer:
                layer_n_hidden = hidden_states.clone()

        # Advance sequence lengths after all layers
        cache_handle.advance_seq_lens()

        # Final layer norm
        hidden_states = run_rms_norm(
            hidden_states, self.model.norm.weight, eps=self.model.norm.variance_epsilon
        )

        return hidden_states, layer_0_embed, layer_n_hidden
