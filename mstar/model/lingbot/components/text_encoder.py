from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LingBotQwen3VLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LingBotQwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.head_dim = head_dim
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        self.attention_scaling = 1.0

    def _make_inv_freq(self) -> torch.Tensor:
        theta = float(self.config.rope_theta)
        return 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

    def reset_inv_freq(self, device: torch.device | str | None = None) -> None:
        self.inv_freq = self._make_inv_freq().to(device=device or self.inv_freq.device)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos).transpose(2, 3)
            freqs = self._apply_interleaved_mrope(freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LingBotQwen3VLMLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotQwen3VLAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = LingBotQwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LingBotQwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary_pos_emb(q, k, *position_embeddings)
        if self.num_key_value_groups != 1:
            k = k[:, :, None, :, :].expand(
                bsz, self.num_key_value_heads, self.num_key_value_groups, seq_len, self.head_dim
            )
            v = v[:, :, None, :, :].expand(
                bsz, self.num_key_value_heads, self.num_key_value_groups, seq_len, self.head_dim
            )
            k = k.reshape(bsz, self.num_heads, seq_len, self.head_dim)
            v = v.reshape(bsz, self.num_heads, seq_len, self.head_dim)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(out)


class LingBotQwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.self_attn = LingBotQwen3VLAttention(config, layer_idx)
        self.mlp = LingBotQwen3VLMLP(config)
        self.input_layernorm = LingBotQwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LingBotQwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class LingBotQwen3VLTextModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LingBotQwen3VLDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = LingBotQwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LingBotQwen3VLRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        bsz, seq_len, _ = hidden_states.shape
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, bsz, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_attention_mask = self._prepare_causal_attention_mask(
            attention_mask,
            text_position_ids,
            bsz,
            seq_len,
        )
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, position_embeddings, causal_attention_mask)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            past_key_values=None,
        )

    @staticmethod
    def _prepare_causal_attention_mask(
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        if attention_mask is None or bool(torch.all(attention_mask == 1).item()):
            return None
        query_pos = position_ids[:, :, None]
        key_pos = position_ids[:, None, :]
        causal = key_pos <= query_pos
        key_allowed = attention_mask[:, None, :].to(torch.bool)
        query_allowed = attention_mask[:, :, None].to(torch.bool)
        allowed = causal & key_allowed & query_allowed
        return allowed[:, None, :, :].expand(batch_size, 1, seq_len, seq_len)


class LingBotQwen3VLModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.language_model = LingBotQwen3VLTextModel(config.text_config)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

    def forward(self, *args, **kwargs):
        if "position_ids" not in kwargs or kwargs["position_ids"] is None:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                kwargs["position_ids"], _ = self.get_rope_index(input_ids, attention_mask)
        return self.language_model(*args, **kwargs)


class LingBotTextEncoderModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = LingBotQwen3VLModel(config)

    def encode_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def reset_non_persistent_buffers(self, device: torch.device | str | None = None) -> None:
        self.model.language_model.rotary_emb.reset_inv_freq(device=device)
