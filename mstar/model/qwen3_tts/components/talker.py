"""Neural components for Qwen3-TTS Talker and residual CodePredictor.

The checkpoint has two coupled autoregressive axes:

* Across time, the 28-layer Talker predicts codec group 0 at 12 Hz and uses
  M*'s paged KV cache. Its attention/MLP projections are tensor-parallel.
* Within one time step, the 5-layer CodePredictor walks codec groups 1-15.
  This short depth axis uses a fixed local KV tensor and is kept replicated.

Class/module names intentionally follow Hugging Face checkpoint namespaces so
``load_hf_weights`` can stream parameters without a model-specific state-dict
rewrite beyond the standard fused Q/K/V and gate/up rules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig, Qwen3TTSTalkerConfig

# ---------------------------------------------------------------------------
# Talker backbone: autoregression across audio frames
# ---------------------------------------------------------------------------


class Qwen3TTSResizeMLP(nn.Module):
    """Two-layer text-embedding projection used by the official Talker."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(hidden_states)))


class Qwen3TTSTalkerLayer(nn.Module):
    """One pre-norm Talker block backed by M* tensor-parallel layers."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = ParallelAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            comm_group=comm_group,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
            comm_group=comm_group,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache_handle)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3TTSTalkerLanguageModel(nn.Module):
    """Backbone matching ``talker.model.*`` and the engine-managed KV cache."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Qwen3TTSTalkerLayer(config, comm_group=comm_group)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_idx, layer in enumerate(self.layers):
            # One cache handle owns every layer's paged K/V tensor. Selecting
            # the layer before attention keeps module code independent of the
            # physical page allocator.
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states, cache_handle)
        # Sequence lengths advance once per forward, after every layer wrote
        # K/V for the same packed token range.
        cache_handle.advance_seq_lens()
        return self.norm(hidden_states)


class Qwen3TTSTalkerModel(nn.Module):
    """Talker backbone, text projection, and codec-group-0 output head."""

    def __init__(
        self,
        config: Qwen3TTSModelConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        talker = config.talker
        self.model = Qwen3TTSTalkerLanguageModel(talker, comm_group=comm_group)
        self.text_projection = Qwen3TTSResizeMLP(
            talker.text_hidden_size,
            talker.text_hidden_size,
            talker.hidden_size,
        )
        self.codec_head = nn.Linear(
            talker.hidden_size, talker.vocab_size, bias=False
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        return self.model(input_embeds, cache_handle)


# ---------------------------------------------------------------------------
# CodePredictor: autoregression across codec groups within one frame
# ---------------------------------------------------------------------------


class Qwen3TTSCodePredictorLayer(nn.Module):
    """Parameter container for one replicated residual-code decoder block."""

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.input_layernorm = RMSNorm(cp.hidden_size, cp.rms_norm_eps)
        self.self_attn = ParallelAttention(
            hidden_size=cp.hidden_size,
            num_heads=cp.num_attention_heads,
            num_kv_heads=cp.num_key_value_heads,
            head_dim=cp.head_dim,
            rope_theta=cp.rope_theta,
            qk_norm=True,
            rms_norm_eps=cp.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            cp.hidden_size, cp.rms_norm_eps
        )
        self.mlp = ParallelGatedMLP(
            hidden_size=cp.hidden_size,
            intermediate_size=cp.intermediate_size,
            activation="silu",
        )


class Qwen3TTSCodePredictorInnerModel(nn.Module):
    """Depth decoder and one embedding table per residual codec group."""

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.layers = nn.ModuleList([
            Qwen3TTSCodePredictorLayer(config)
            for _ in range(cp.num_hidden_layers)
        ])
        self.norm = RMSNorm(cp.hidden_size, cp.rms_norm_eps)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(cp.vocab_size, config.talker.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ])


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the checkpoint's non-interleaved RoPE to grouped Q/K tensors."""
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(2)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(2)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)

    return (
        q * cos.to(q.dtype) + rotate_half(q) * sin.to(q.dtype),
        k * cos.to(k.dtype) + rotate_half(k) * sin.to(k.dtype),
    )


class Qwen3TTSCodePredictor(nn.Module):
    """Five-layer decoder that predicts codec groups 1 through 15.

    ``forward_depth_unrolled`` performs one position on the group-depth axis.
    Callers first write the Talker hidden state at position 0, then repeatedly
    feed the preceding codec embedding at positions 1-15. Keeping this as
    tensor-only code allows the complete depth loop to be CUDA-graph captured.
    """

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        if cp.hidden_size != config.talker.hidden_size:
            raise ValueError(
                "M* currently requires equal Talker and CodePredictor hidden "
                "sizes; the supported 0.6B checkpoint uses 1024 for both"
            )
        self.config = cp
        self.model = Qwen3TTSCodePredictorInnerModel(config)
        self.lm_head = nn.ModuleList([
            nn.Linear(cp.hidden_size, cp.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ])
        self.register_buffer(
            # Populated after weight loading. A single contiguous tensor lets
            # the piecewise loop select a residual head without traversing a
            # Python ModuleList during CUDA Graph replay.
            "lm_head_weight",
            torch.empty(
                config.num_code_groups - 1,
                cp.vocab_size,
                cp.hidden_size,
            ),
            persistent=False,
        )

    def consolidate_stacked_weights(self) -> None:
        """Pack the 15 loaded LM heads into graph-friendly contiguous storage."""
        self.lm_head_weight = torch.stack([
            head.weight.data for head in self.lm_head
        ]).contiguous()

    def forward_depth_unrolled(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        """Run all CodePredictor layers for one or more adjacent depth tokens.

        ``kv_cache`` layout is ``[layers, batch, K/V, groups, kv_heads,
        head_dim]``. Unlike Talker cache, it is frame-local scratch space:
        every generated frame starts at ``cache_pos=0`` and overwrites it.
        """
        hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape

        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            qkv = attn.qkv_proj(hidden_states)
            q_size = attn.num_heads * attn.head_dim
            kv_size = attn.num_kv_heads * attn.head_dim
            q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
            q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
            k = k.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
            v = v.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
            q, k = attn._apply_qk_norm(q, k)
            q, k = _apply_rope(
                q, k, position_ids, float(attn.rope_theta)
            )

            # Append this depth position, then attend over the prefix already
            # generated within the same codec frame.
            end = cache_pos + seq_len
            kv_cache[layer_idx, :, 0, cache_pos:end].copy_(k)
            kv_cache[layer_idx, :, 1, cache_pos:end].copy_(v)
            keys = kv_cache[layer_idx, :, 0, :end]
            values = kv_cache[layer_idx, :, 1, :end]
            repeats = attn.num_heads // attn.num_kv_heads
            keys = keys.repeat_interleave(repeats, dim=2).transpose(1, 2)
            values = values.repeat_interleave(repeats, dim=2).transpose(1, 2)
            queries = q.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                queries, keys, values, is_causal=False
            ).transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            hidden_states = residual + attn.o_proj(attn_output)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        return self.model.norm(hidden_states)
