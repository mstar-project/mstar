from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from mminf.engine.ar_engine import BatchedCacheManager
from mminf.model.orpheus.config import OrpheusModelConfig
from mminf.utils.flashinfer_utils import run_rms_norm


class OrpheusRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No-op: FlashInfer's run_rms_norm is called directly instead
        pass


class OrpheusMLP(nn.Module):
    """SwiGLU MLP. Holds separate ``gate_proj`` / ``up_proj`` ``nn.Linear``
    layers for weight loading; ``consolidate_gate_up_weight()`` concatenates
    them into a single ``gate_up_proj_weight`` buffer (one fused GEMM instead
    of two) and nulls out the originals.
    """

    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def consolidate_gate_up_weight(self) -> None:
        if self.gate_proj is None:
            return
        gate_up_proj_weight = torch.cat(
            (self.gate_proj.weight, self.up_proj.weight), dim=0,
        ).contiguous()
        self.register_buffer("gate_up_proj_weight", gate_up_proj_weight, persistent=False)
        self.gate_proj = None
        self.up_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        else:
            gate_up = F.linear(x, self.gate_up_proj_weight)
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class OrpheusAttention(nn.Module):
    """GQA self-attention. Holds separate ``q_proj`` / ``k_proj`` / ``v_proj``
    ``nn.Linear`` layers for weight loading; ``consolidate_qkv_weight()``
    concatenates them into a single ``qkv_proj_weight`` buffer (one fused
    GEMM instead of three) and nulls out the originals.
    """

    def __init__(self, config: OrpheusModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scaling["factor"]
        self.high_freq = config.rope_scaling["high_freq_factor"]
        self.low_freq = config.rope_scaling["low_freq_factor"]
        self.old_context_len = config.rope_scaling["original_max_position_embeddings"]

        self._q_dim = self.num_heads * self.head_dim
        self._kv_dim = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self._q_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self._kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self._kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def consolidate_qkv_weight(self) -> None:
        if self.q_proj is None:
            return
        qkv_proj_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0,
        ).contiguous()
        self.register_buffer("qkv_proj_weight", qkv_proj_weight, persistent=False)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        if self.q_proj is not None:
            query_states = self.q_proj(query_sequence).view(-1, self.num_heads, self.head_dim)
            key_states = self.k_proj(query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            value_states = self.v_proj(query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        else:
            qkv = F.linear(query_sequence, self.qkv_proj_weight)
            q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)
            query_states = q.view(-1, self.num_heads, self.head_dim)
            key_states = k.view(-1, self.num_key_value_heads, self.head_dim)
            value_states = v.view(-1, self.num_key_value_heads, self.head_dim)

        query_states, key_states = cache_handle.apply_rope(
            query_states, key_states,
            rope_theta=self.rope_theta,
            rope_scale=self.rope_scale,
            low_freq_factor=self.low_freq,
            high_freq_factor=self.high_freq,
            old_context_len=self.old_context_len
        )

        attn_output = cache_handle.run_attention(q=query_states, k=key_states, v=value_states)
        attn_output = attn_output.reshape(-1, self.hidden_size)
        return self.o_proj(attn_output)


class OrpheusDecoderLayer(nn.Module):
    def __init__(self, config: OrpheusModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.self_attn = OrpheusAttention(config, layer_idx)
        self.mlp = OrpheusMLP(config)
        self.input_layernorm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        residual = query_sequence
        query_sequence = run_rms_norm(
            query_sequence, self.input_layernorm.weight, eps=self.input_layernorm.variance_epsilon
        )
        query_sequence = self.self_attn(query_sequence=query_sequence, cache_handle=cache_handle)
        query_sequence = residual + query_sequence

        residual = query_sequence
        query_sequence = run_rms_norm(
            query_sequence, self.post_attention_layernorm.weight, eps=self.post_attention_layernorm.variance_epsilon
        )
        query_sequence = self.mlp(query_sequence)
        query_sequence = residual + query_sequence

        return query_sequence


class OrpheusLanguageModel(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [OrpheusDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = decoder_layer(query_sequence=query_sequence, cache_handle=cache_handle)

        cache_handle.advance_seq_lens()

        query_sequence = run_rms_norm(query_sequence, self.norm.weight, eps=self.norm.variance_epsilon)
        return query_sequence

    def consolidate_fused_weights(self) -> None:
        for layer in self.layers:
            layer.self_attn.consolidate_qkv_weight()
            layer.mlp.consolidate_gate_up_weight()


class OrpheusForCausalLM(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.model = OrpheusLanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(query_sequence=query_sequence, cache_handle=cache_handle)

    def consolidate_fused_weights(self) -> None:
        self.model.consolidate_fused_weights()