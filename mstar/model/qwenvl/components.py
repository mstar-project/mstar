"""Qwen3-VL-30B-A3B text components built from M* shared primitives."""

from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components import ParallelSparseMoeBlock, RMSNorm
from mstar.model.components.distributed import ColumnParallelLinear, ParallelAttention, VocabParallelEmbedding
from mstar.model.components.mrope import compute_3d_cos_sin, compute_rope_freqs


def compute_mrope_cos_sin(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen3-VL's interleaved temporal/height/width RoPE."""
    if position_ids.shape[0] != 3:
        raise ValueError(f"Expected [3, tokens] MRoPE position ids, got {tuple(position_ids.shape)}.")
    inv_freq = compute_rope_freqs(head_dim, rope_theta, position_ids.device)
    return compute_3d_cos_sin(
        position_ids,
        inv_freq,
        mrope_section=mrope_section,
        target_dtype=dtype,
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


class QwenVLAttention(ParallelAttention):
    """TP-aware Qwen3 attention with QK norm and interleaved MRoPE."""

    def __init__(self, config, comm_group: CommGroup | None = None):
        text = config.text_config
        super().__init__(
            comm_group=comm_group,
            hidden_size=text.hidden_size,
            num_heads=text.num_attention_heads,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            qkv_bias=text.attention_bias,
            o_bias=text.attention_bias,
            qk_norm=True,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=text.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)
        q = q * cos.unsqueeze(1) + _rotate_half(q) * sin.unsqueeze(1)
        k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
        output = cache_handle.run_attention(q=q, k=k, v=v)
        return self.o_proj(output.reshape(tokens, self.num_heads * self.head_dim))


class QwenVLDecoderLayer(nn.Module):
    def __init__(self, config, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        text = config.text_config
        self.input_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.self_attn = QwenVLAttention(config, comm_group)
        self.post_attention_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.mlp = ParallelSparseMoeBlock(
            comm_group=comm_group,
            hidden_size=text.hidden_size,
            moe_intermediate_size=text.moe_intermediate_size,
            num_experts=text.num_experts,
            num_experts_per_tok=text.num_experts_per_tok,
            norm_topk_prob=text.norm_topk_prob,
        )

    def forward(self, hidden_states, cache_handle, cos, sin):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cache_handle, cos, sin)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class QwenVLForCausalLM(nn.Module):
    """Qwen3-VL MoE language backbone; vision stays a separate graph node."""

    def __init__(self, config, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        text = config.text_config
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            num_embeddings=text.vocab_size,
            embedding_dim=text.hidden_size,
            comm_group=comm_group,
        )
        self.model.layers = nn.ModuleList(
            [QwenVLDecoderLayer(config, comm_group) for _ in range(text.num_hidden_layers)]
        )
        self.model.norm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=text.hidden_size,
            output_size=text.vocab_size,
            bias=False,
            gather_output=True,
        )

    @staticmethod
    def _inject_deepstack(
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_token_mask = visual_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        if int(visual_token_mask.sum()) != visual_embeds.shape[0]:
            raise ValueError(
                "QwenVL DeepStack feature count must equal the visual-token count; "
                f"got {visual_embeds.shape[0]} features for {int(visual_token_mask.sum())} tokens."
            )
        result = hidden_states.clone()
        result[visual_token_mask] += visual_embeds.to(device=result.device, dtype=result.dtype)
        return result

    def forward(
        self,
        input_embeds,
        cache_handle,
        position_ids,
        position_advance: int | list[int] | None = None,
        cos=None,
        sin=None,
        visual_token_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ):
        if cos is None or sin is None:
            cos, sin = compute_mrope_cos_sin(
                position_ids,
                head_dim=self.config.text_config.head_dim,
                rope_theta=self.config.text_config.rope_theta,
                mrope_section=tuple(self.config.text_config.rope_scaling["mrope_section"]),
                dtype=input_embeds.dtype,
            )
        if deepstack_visual_embeds is not None:
            if visual_token_mask is None:
                raise ValueError("QwenVL DeepStack features require a visual-token mask.")
            if len(deepstack_visual_embeds) != len(self.config.vision_config.deepstack_visual_indexes):
                raise ValueError(
                    "QwenVL DeepStack layer count does not match the vision configuration; "
                    f"got {len(deepstack_visual_embeds)} feature sets."
                )
        hidden_states = input_embeds
        for index, layer in enumerate(self.model.layers):
            cache_handle.set_layer_idx(index)
            hidden_states = layer(hidden_states, cache_handle, cos, sin)
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                hidden_states = self._inject_deepstack(
                    hidden_states,
                    visual_token_mask,
                    deepstack_visual_embeds[index],
                )
        cache_handle.advance_seq_lens(pos_id_ns=position_advance)
        return self.model.norm(hidden_states)
