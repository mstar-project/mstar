# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from typing import Optional

import torch
from torch import nn
from transformers import ROPE_INIT_FUNCTIONS
from transformers.activations import ACT2FN

from mminf.engine.ar_engine import CacheHandle
from mminf.model.bagel.bagel_model import BagelModelConfig
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BagelRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[BagelModelConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BagelRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BagelMLP(nn.Module):
    def __init__(self, config: BagelModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class BagelAttention(nn.Module):
    def __init__(self,config: BagelModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
        if self.config.qk_norm:
            self.q_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        query_sequence: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        cache_handle: CacheHandle,
        write_cache=True,
        is_causal=True,
    ):
        query_states = self.q_proj(query_sequence).view(
            -1, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = query_position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)

        # Run paged attention
        attn_output = cache_handle.run_attention(
            q=query_states,
            k=key_states,
            v=value_states,
            layer_idx=self.layer_idx,
            is_causal=is_causal,
            write_cache=write_cache,
        )

        attn_output = attn_output.reshape(-1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class BagelAttentionMoT(nn.Module):
    def __init__(self, config: BagelModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.config.qk_norm:
            self.q_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        query_sequence: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        cache_handle: CacheHandle,
        write_cache=True,
        is_causal=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
    ):
        if mode == "und":
            query_states = self.q_proj(query_sequence).view(
                -1, self.num_heads, self.head_dim
            )
            key_states = self.k_proj(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            value_states = self.v_proj(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        elif mode == "gen":
            query_sequence = query_sequence.to(torch.bfloat16)

            query_states = query_sequence.new_zeros(
                (query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            key_states = query_sequence.new_zeros(
                (query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            value_states = query_sequence.new_zeros(
                (query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )

            text_query_sequence = query_sequence[text_indexes]
            vae_query_sequence = query_sequence[vae_token_indexes]

            query_states[text_indexes] = self.q_proj(
                text_query_sequence
            )
            query_states[vae_token_indexes] = self.q_proj_moe_gen(
                vae_query_sequence
            )

            key_states[text_indexes] = self.k_proj(
                text_query_sequence
            )
            key_states[vae_token_indexes] = self.k_proj_moe_gen(
                vae_query_sequence
            )

            value_states[text_indexes] = self.v_proj(
                text_query_sequence
            )
            value_states[vae_token_indexes] = self.v_proj_moe_gen(
                vae_query_sequence
            )

            query_states = query_states.view(
                -1, self.num_heads, self.head_dim
            )
            key_states = key_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )
            value_states = value_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )

            query_states = query_states.to(torch.float32)
            query_states[text_indexes] = self.q_norm(
                query_states[text_indexes]
            )
            query_states[vae_token_indexes] = self.q_norm_moe_gen(
                query_states[vae_token_indexes]
            )

            key_states = key_states.to(torch.float32)
            key_states[text_indexes] = self.k_norm(
                key_states[text_indexes]
            )
            key_states[vae_token_indexes] = self.k_norm_moe_gen(
                key_states[vae_token_indexes]
            )

        # rotary embeddings
        cos, sin = query_position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)

        # run paged attention
        attn_output = cache_handle.run_attention(
            q=query_states,
            k=key_states,
            v=value_states,
            layer_idx=self.layer_idx,
            is_causal=is_causal,
            write_cache=write_cache,
        )

        attn_output = attn_output.reshape(-1, self.hidden_size)

        if mode == "und":
            attn_output = self.o_proj(attn_output)

        elif mode == "gen":
            attn_output[text_indexes] = self.o_proj(
                attn_output[text_indexes]
            )
            attn_output[vae_token_indexes] = self.o_proj_moe_gen(
                attn_output[vae_token_indexes]
            )

        return attn_output


# class BagelDecoderLayer(nn.Module):
#     def __init__(self, config:BagelModelConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = BagelAttention(config, layer_idx)

#         self.mlp = Qwen2MLP(config)
#         self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         query_sequence: torch.Tensor,
#         query_position_embeddings: torch.Tensor,
#         cache_handle: CacheHandle,
#         write_cache=True,
#         is_causal=True,
#     ):

#         residual = query_sequence
#         query_sequence = self.input_layernorm(query_sequence)

#         # Self Attention
#         query_sequence = self.self_attn(
#             query_sequence=query_sequence,
#             query_position_embeddings=query_position_embeddings,
#             cache_handle=cache_handle,
#             write_cache=write_cache,
#             is_causal=is_causal,
#         )
#         query_sequence = residual + query_sequence

#         # Fully Connected
#         residual = query_sequence
#         query_sequence = self.post_attention_layernorm(query_sequence)
#         query_sequence = self.mlp(query_sequence)
#         query_sequence = residual + query_sequence

#         return query_sequence


class BagelMoTDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: BagelModelConfig, 
        layer_idx: Optional[int] = None, 
        attn_module = BagelAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = BagelMLP(config)
        self.mlp_moe_gen = BagelMLP(config)
        self.input_layernorm = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        cache_handle: CacheHandle,
        write_cache=True,
        is_causal=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
    ):
        residual = query_sequence
        if mode == "und":
            query_sequence = self.input_layernorm(query_sequence)
        elif mode == "gen":
            query_sequence_ = torch.zeros_like(query_sequence)
            query_sequence_[text_indexes] = self.input_layernorm(query_sequence[text_indexes])
            query_sequence_[vae_token_indexes] = self.input_layernorm_moe_gen(query_sequence[vae_token_indexes])
            query_sequence = query_sequence_

        # Self Attention
        query_sequence = self.self_attn(
            query_sequence=query_sequence,
            query_position_embeddings=query_position_embeddings,
            cache_handle=cache_handle,
            write_cache=write_cache,
            is_causal=is_causal,
            mode=mode,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
        )
        query_sequence = residual + query_sequence

        # Fully Connected
        residual = query_sequence
        if mode == "und":
            query_sequence = self.post_attention_layernorm(query_sequence)
            query_sequence = self.mlp(query_sequence)
        elif mode == "gen":
            text_query_sequence = query_sequence[text_indexes]
            vae_query_sequence = query_sequence[vae_token_indexes]
            text_query_sequence = self.post_attention_layernorm(text_query_sequence).to(torch.bfloat16)
            vae_query_sequence = self.post_attention_layernorm_moe_gen(vae_query_sequence).to(torch.bfloat16)

            query_sequence_ = torch.zeros_like(query_sequence).to(torch.bfloat16)
            query_sequence_[text_indexes] = self.mlp(text_query_sequence)
            query_sequence_[vae_token_indexes] = self.mlp_moe_gen(vae_query_sequence)
            query_sequence = query_sequence_

        query_sequence = residual + query_sequence

        return query_sequence


class BagelLanguageModel(nn.Module):
    def __init__(self, config: BagelModelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = BagelMoTDecoderLayer
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelRotaryEmbedding(config=config)

    def forward(
        self,
        query_sequence: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        cache_handle: CacheHandle,
        write_cache=True,
        is_causal=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
    ):

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert vae_token_indexes is not None
                assert text_indexes is not None
                extra_inputs.update(
                    vae_token_indexes=vae_token_indexes,
                    text_indexes=text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            query_sequence = decoder_layer(
                query_sequence=query_sequence,
                query_position_embeddings=query_position_embeddings,
                cache_handle=cache_handle,
                write_cache=write_cache,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                query_sequence = self.norm(query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(query_sequence)
                packed_query_sequence_[text_indexes] = self.norm(query_sequence[text_indexes])
                packed_query_sequence_[vae_token_indexes] = self.norm_moe_gen(query_sequence[vae_token_indexes])
                query_sequence = packed_query_sequence_
        else:
            query_sequence = self.norm(query_sequence)


class BagelForCausalLM(nn.Module):
    def __init__(self, config: BagelModelConfig):
        super().__init__()
        self.model = BagelLanguageModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.model

    def forward(
        self,
        query_sequence: torch.Tensor,
        query_position_ids: torch.Tensor,
        write_cache=True,
        is_causal=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
    ):
        outputs = self.model(
            query_sequence=query_sequence,
            query_position_ids=query_position_ids,
            write_cache=write_cache,
            is_causal=is_causal,
            mode=mode,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
        )

        return outputs
