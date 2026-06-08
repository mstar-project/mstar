"""Ling-2.0 thinker LLM (full forward, no KV cache yet).

Composes :class:`LingDecoderLayer` × N with a shared rope, vocab
embedding, final RMSNorm, and an untied lm_head. The shape downstream
mminf code will eventually wrap is one of these :class:`LingMoeModel`
instances behind a :class:`NodeSubmodule` (step 3c).

Reference structure: vllm-omni's :class:`BailingMoeV2Model` +
:class:`BailingMoeV2ForCausalLM`
``/tmp/vllm-omni/.../modeling_bailing_moe_v2.py:662-895``.
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.distributed.communication import TPCommGroup
from mminf.model.components.norm import RMSNorm
from mminf.model.ming_omni_flash.components.decoder_layer import (
    LingDecoderLayer,
)
from mminf.model.ming_omni_flash.components.rope import (
    LingPartialMRotaryEmbedding,
)


class LingMoeModel(nn.Module):
    """Full Ling-2.0 thinker forward (embed + layers + lm_head).

    All shape-relevant config flattens into the constructor so callers
    don't need a :class:`MingFlashOmniModelConfig` instance — useful for
    small-dim unit tests. The eventual mminf submodule (step 3c) builds
    one of these from the real config.

    Args (all required, but small-dim test configs only need plausible
    values; nothing here is hard-coded to Ming-specific dims):
        vocab_size: e.g. 157184 on released ckpt.
        hidden_size: e.g. 4096.
        intermediate_size: dense layer-0 MLP intermediate; e.g. 9216.
        moe_intermediate_size: per-expert intermediate; e.g. 1024.
        num_hidden_layers: e.g. 32.
        num_attention_heads, num_kv_heads, head_dim: e.g. 32 / 4 / 128.
        rms_norm_eps: 1e-6.
        rope_theta: 2_400_000.
        max_position_embeddings: 32768.
        partial_rotary_factor: 0.5.
        mrope_section: [8, 12, 12].
        num_experts: 256.
        num_experts_per_tok: 8.
        num_shared_experts: 1.
        n_group: 8.
        topk_group: 4.
        routed_scaling_factor: 2.5.
        first_k_dense_replace: 1.
        tie_word_embeddings: False on released ckpt — lm_head is a
            separate matrix from embed_tokens.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        partial_rotary_factor: float,
        mrope_section: list[int],
        num_experts: int,
        num_experts_per_tok: int,
        num_shared_experts: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        first_k_dense_replace: int,
        tie_word_embeddings: bool = False,
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        comm_group: TPCommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # embed_tokens + lm_head stay replicated. At hidden_size=4096
        # they're 1.3 GB each — cheap compared to the layers.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Single rotary instance shared across every layer — inv_freq is
        # config-only, no per-layer state.
        rotary = LingPartialMRotaryEmbedding(
            head_dim=head_dim,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

        self.layers = nn.ModuleList([
            LingDecoderLayer(
                layer_idx=i,
                first_k_dense_replace=first_k_dense_replace,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                moe_intermediate_size=moe_intermediate_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                rotary=rotary,
                use_qkv_bias=use_qkv_bias,
                use_bias=use_bias,
                comm_group=comm_group,
            )
            for i in range(num_hidden_layers)
        ])

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        cache_handle,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full thinker forward.

        Args:
            cache_handle: :class:`BatchedCacheManager` from the engine
                (or a unit-test mock with ``set_layer_idx`` +
                ``run_attention``). Required — the attention layer
                writes K/V to its paged cache and runs FlashInfer
                attention against it.
            input_ids: ``(T,)`` token ids — if provided, ``embed_tokens``
                turns them into embeddings.
            input_embeds: ``(T, hidden_size)`` precomputed embeddings —
                used directly (multimodal callers pass this with vision /
                audio embeddings already spliced in).
            position_ids: ``(T,)`` for 1D rope, or ``(3, T)`` for 3D
                video_rope. Defaults to ``torch.arange(T)`` if None.
            image_mask, audio_mask: per-token modality masks for
                :class:`LingMoeBlock`. ``None`` ⇒ all text routing.

        Returns:
            ``(T, vocab_size)`` logits. The caller (the submodule)
            slices the last position for next-token sampling.
        """
        if (input_ids is None) == (input_embeds is None):
            raise ValueError(
                "Exactly one of input_ids / input_embeds must be provided"
            )

        if input_embeds is None:
            assert input_ids is not None
            h = self.embed_tokens(input_ids)
        else:
            h = input_embeds

        if h.dim() != 2:
            raise ValueError(
                f"LingMoeModel expects packed (T, hidden) input; got "
                f"shape {tuple(h.shape)}."
            )

        T = h.shape[0]
        if position_ids is None:
            position_ids = torch.arange(T, device=h.device)

        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            h = layer(
                h, cache_handle, position_ids,
                image_mask=image_mask,
                audio_mask=audio_mask,
            )

        h = self.norm(h)
        return self.lm_head(h)
