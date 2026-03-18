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
from transformers.activations import ACT2FN

from mminf.engine.ar_engine import BatchedCacheManager
from mminf.model.bagel.config import BagelModelConfig
from mminf.utils.flashinfer_utils import run_rms_norm

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096


def _to_text_mask(
    text_indexes: torch.Tensor | None,
    vae_token_indexes: torch.Tensor | None,
    sequence_len: int,
    device: torch.device,
) -> torch.Tensor:
    if text_indexes is None and vae_token_indexes is None:
        return torch.ones(sequence_len, dtype=torch.bool, device=device)

    if text_indexes is not None and text_indexes.dtype == torch.bool:
        return text_indexes.to(device=device)

    seq_indexes = torch.arange(sequence_len, device=device)
    if text_indexes is not None:
        return torch.isin(seq_indexes, text_indexes.to(device=device, dtype=torch.long))

    if vae_token_indexes is None or vae_token_indexes.numel() == 0:
        return torch.ones(sequence_len, dtype=torch.bool, device=device)

    return ~torch.isin(seq_indexes, vae_token_indexes.to(device=device, dtype=torch.long))


class BagelRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


    def forward(self, hidden_states):
        pass
        # # NOTE: this is replaced by flashinfer rmsnorm
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

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


# class BagelAttention(nn.Module):
#     def __init__(self,config: BagelModelConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx

#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = config.is_causal
#         self.attention_dropout = config.attention_dropout

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )
#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

#         if self.config.qk_norm:
#             self.q_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
#             self.k_norm = BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         else:
#             self.q_norm = nn.Identity()
#             self.k_norm = nn.Identity()

#     def forward(
#         self,
#         query_sequence: torch.Tensor,
#         cache_handle: CacheHandle,
#         write_cache=True,
#         is_causal=True,
#     ):
#         query_states = self.q_proj(query_sequence).view(
#             -1, self.num_heads, self.head_dim
#         )
#         key_states = self.k_proj(query_sequence).view(
#             -1, self.num_key_value_heads, self.head_dim
#         )
#         value_states = self.v_proj(query_sequence).view(
#             -1, self.num_key_value_heads, self.head_dim
#         )

#         query_states = run_rms_norm(
#             query_states, self.q_norm.weight, eps=self.q_norm.variance_epsilon
#         )
#         key_states = run_rms_norm(
#             key_states, self.k_norm.weight, eps=self.k_norm.variance_epsilon
#         )

#         query_states, key_states = cache_handle.apply_rope_default(
#             query_states, key_states, rope_theta=self.rope_theta
#         )

#         query_states = query_states.to(torch.bfloat16)
#         key_states = key_states.to(torch.bfloat16)
#         value_states = value_states.to(torch.bfloat16)

#         # Run paged attention
#         attn_output = cache_handle.run_attention(
#             q=query_states,
#             k=key_states,
#             v=value_states,
#             layer_idx=self.layer_idx,
#             is_causal=is_causal,
#             write_cache=write_cache,
#         )

#         attn_output = attn_output.reshape(-1, self.hidden_size)
#         attn_output = self.o_proj(attn_output)

#         return attn_output


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
        cache_handle,
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

            query_states = run_rms_norm(
                query_states, self.q_norm.weight, eps=self.q_norm.variance_epsilon
            )
            key_states = run_rms_norm(
                key_states, self.k_norm.weight, eps=self.k_norm.variance_epsilon
            )

        elif mode == "gen":
            text_mask = _to_text_mask(
                text_indexes, vae_token_indexes, query_sequence.shape[0], query_sequence.device
            )
            query_sequence = query_sequence.to(torch.bfloat16)

            text_query_states = self.q_proj(query_sequence).view(
                -1, self.num_heads, self.head_dim
            )

            vae_query_states = self.q_proj_moe_gen(query_sequence).view(
                -1, self.num_heads, self.head_dim
            )

            text_key_states = self.k_proj(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            vae_key_states = self.k_proj_moe_gen(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            text_value_states = self.v_proj(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            vae_value_states = self.v_proj_moe_gen(query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            text_query_states = run_rms_norm(
                text_query_states,
                self.q_norm.weight,
                eps=self.q_norm.variance_epsilon
            )
            text_key_states = run_rms_norm(
                text_key_states,
                self.k_norm.weight,
                eps=self.k_norm.variance_epsilon
            )
            vae_query_states = run_rms_norm(
                vae_query_states,
                self.q_norm_moe_gen.weight,
                eps=self.q_norm_moe_gen.variance_epsilon
            )
            vae_key_states = run_rms_norm(
                vae_key_states,
                self.k_norm_moe_gen.weight,
                eps=self.k_norm_moe_gen.variance_epsilon
            )

            text_mask = text_mask[:, None, None]
            query_states = torch.where(
                text_mask, text_query_states, vae_query_states
            )
            key_states = torch.where(
                text_mask, text_key_states, vae_key_states
            )
            value_states = torch.where(
                text_mask, text_value_states, vae_value_states
            )

        # RoPE: pos_ids pre-computed by plan_rope before the LLM forward
        query_states, key_states = cache_handle.apply_rope(
            query_states, key_states, rope_theta=self.rope_theta
        )

        # Paged attention: plan (page alloc, FlashInfer index tensors) was
        # done by plan_attention before the LLM forward
        attn_output = cache_handle.run_attention(
            q=query_states,
            k=key_states,
            v=value_states,
            layer_idx=self.layer_idx,
        )

        attn_output = attn_output.reshape(-1, self.hidden_size)

        if mode == "und":
            attn_output = self.o_proj(attn_output)

        elif mode == "gen":
            text_mask = _to_text_mask(
                text_indexes,
                vae_token_indexes,
                query_sequence.shape[0],
                query_sequence.device,
            )
            attn_text = self.o_proj(attn_output)
            attn_vae = self.o_proj_moe_gen(attn_output)
            attn_output = torch.where(text_mask[:, None], attn_text, attn_vae)

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
        cache_handle,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
    ):
        residual = query_sequence
        if mode == "und":
            query_sequence = run_rms_norm(
                query_sequence, self.input_layernorm.weight, eps=self.input_layernorm.variance_epsilon
            )
        elif mode == "gen":
            text_mask = _to_text_mask(text_indexes, vae_token_indexes, query_sequence.shape[0], query_sequence.device)
            text_query = run_rms_norm(
                query_sequence,
                self.input_layernorm.weight,
                eps=self.input_layernorm.variance_epsilon,
            )
            vae_query = run_rms_norm(
                query_sequence,
                self.input_layernorm_moe_gen.weight,
                eps=self.input_layernorm_moe_gen.variance_epsilon,
            )
            query_sequence = torch.where(text_mask[:, None], text_query, vae_query)

        # Self Attention
        query_sequence = self.self_attn(
            query_sequence=query_sequence,
            cache_handle=cache_handle,
            mode=mode,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
        )
        query_sequence = residual + query_sequence

        # Fully Connected
        residual = query_sequence
        if mode == "und":
            query_sequence = run_rms_norm(
                query_sequence, self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon
            )
            query_sequence = self.mlp(query_sequence)
        elif mode == "gen":
            text_mask = _to_text_mask(text_indexes, vae_token_indexes, query_sequence.shape[0], query_sequence.device)
            text_query = run_rms_norm(
                query_sequence,
                self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
            )
            vae_query = run_rms_norm(
                query_sequence,
                self.post_attention_layernorm_moe_gen.weight,
                eps=self.post_attention_layernorm_moe_gen.variance_epsilon,
            )
            text_query = self.mlp(text_query)
            vae_query = self.mlp_moe_gen(vae_query)
            query_sequence = torch.where(text_mask[:, None], text_query, vae_query)

        query_sequence = residual + query_sequence

        return query_sequence


class BagelLanguageModel(nn.Module):
    def __init__(self, config: BagelModelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = True

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = BagelMoTDecoderLayer
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        write_cache=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
        custom_advance_pos_id=None,
    ):
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

        for _layer_idx, decoder_layer in enumerate(self.layers):
            query_sequence = decoder_layer(
                query_sequence=query_sequence,
                cache_handle=cache_handle,
                **extra_inputs,
            )

        if write_cache:
            cache_handle.advance_seq_lens(pos_id_ns=custom_advance_pos_id)

        if self.use_moe:
            if mode == "und":
                query_sequence = run_rms_norm(
                    query_sequence, self.norm.weight, eps=self.norm.variance_epsilon
                )
            elif mode == "gen":
                text_mask = _to_text_mask(
                    text_indexes,
                    vae_token_indexes,
                    query_sequence.shape[0],
                    query_sequence.device,
                )
                query_text = run_rms_norm(
                    query_sequence,
                    self.norm.weight,
                    eps=self.norm.variance_epsilon,
                )
                query_vae = run_rms_norm(
                    query_sequence,
                    self.norm_moe_gen.weight,
                    eps=self.norm_moe_gen.variance_epsilon,
                )
                query_sequence = torch.where(text_mask[:, None], query_text, query_vae)
        else:
            query_sequence = run_rms_norm(
                query_sequence, self.norm.weight, eps=self.norm.variance_epsilon
            )
        return query_sequence


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
        cache_handle,
        write_cache=True,
        mode="und",
        vae_token_indexes=None,
        text_indexes=None,
        custom_advance_pos_id=None,
        **kwargs
    ):
        assert mode in ["und", "gen"]
        outputs = self.model(
            query_sequence=query_sequence,
            cache_handle=cache_handle,
            write_cache=write_cache,
            mode=mode,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            custom_advance_pos_id=custom_advance_pos_id,
        )

        return outputs
