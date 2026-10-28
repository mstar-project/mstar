"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear attention: the dense layer.

The math is the engine's torch reference (``kda_torch``), which the KDA resource's reference
bundle also runs against the recurrent pool where the fused Triton kernels do not.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.linear_attn.kda_torch import (
    causal_conv1d_prefill,
    causal_conv1d_update,
    chunk_kda,
    forget_gate,
    gated_rms_norm,
    recurrent_kda_step,
)


@dataclass
class Glm5NextKdaConfig:
    """Standalone slice of ``Glm5NextModelConfig`` that KDA needs."""

    hidden_size: int = 4096
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_size: int = 4  # HF short_conv_kernel_size
    gate_lower_bound: float = -5.0
    rms_norm_eps: float = 1e-5

    @property
    def linear_qkv_dim(self) -> int:
        """Per-projection KDA width: heads x head_dim (8192 full-size)."""
        return self.linear_num_heads * self.linear_head_dim

    @property
    def linear_conv_channels(self) -> int:
        """Fused depthwise conv channels over cat(q, k, v): 3 x qkv_dim."""
        return 3 * self.linear_qkv_dim

    @classmethod
    def reduced(cls) -> "Glm5NextKdaConfig":
        """Tiny-dim variant (H=4, D=32) for the CPU parity tests."""
        return cls(hidden_size=64, linear_num_heads=4, linear_head_dim=32)


def apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """Zero padded positions BEFORE any projection (2D boolean ``(B, L)`` mask)."""
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class Glm5NextForgetGate(nn.Module):
    """Per-head, per-key-channel log forget gate ``g (B, L, H, D)``, fp32."""

    def __init__(self, config: Glm5NextKdaConfig, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        if config.gate_lower_bound is None:
            raise ValueError(
                "gate_lower_bound is None: the softplus forget-gate branch is "
                "dead for GLM-5.3-Flash and not implemented"
            )
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.gate_lower_bound = float(config.gate_lower_bound)

        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, config.linear_qkv_dim, bias=False, dtype=dtype)
        # fp32 params (HF _keep_in_fp32_modules_strict); stored fp32 in the
        # checkpoint. Zero init keeps the fresh module runnable (loader
        # overwrites): decay_rate = 1, g = lower_bound * sigmoid(f(x)).
        self.dt_bias = nn.Parameter(torch.zeros(config.linear_qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(B, L, hidden) -> (B, L, H, D)`` fp32 log-decay in (lower_bound, 0)."""
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        f = self.f_b_proj(self.f_a_proj(hidden_states)).view(hidden_shape)
        return forget_gate(f, self.dt_bias, self.A_log, self.gate_lower_bound)


class Glm5NextRMSNormGated(nn.Module):
    """Gated RMSNorm over head_dim, weight shared across heads, strict fp32."""

    def __init__(self, head_dim: int, eps: float, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return gated_rms_norm(hidden_states, gate, self.weight, self.variance_epsilon)


class Glm5NextLinearAttention(nn.Module):
    """One KDA layer: projections + fused causal conv + delta rule + gated out."""

    def __init__(self, config: Glm5NextKdaConfig, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.conv_kernel_size = config.linear_conv_kernel_size
        # Kernel-internal chunking, not a config field; 64 is the reference
        # default, which the parity tests are written against.
        self.chunk_size = 64

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        # Fused depthwise conv over cat(q, k, v); fp32 weight, no bias
        # (loader fuses the checkpoint's three [8192, 1, 4] bf16 taps and
        # upcasts). padding is unused (both paths call F.conv1d directly)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
            dtype=torch.float32,
        )
        self.forget_gate = Glm5NextForgetGate(config, dtype=dtype)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, dtype=dtype)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False, dtype=dtype)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False, dtype=dtype)
        # q | k | v | b | f_a | g_a, built by process_weights_after_loading.
        self._in_proj_weight: torch.Tensor | None = None

    def process_weights_after_loading(self, device) -> None:
        """Concatenate the six input projections for the fused kernels, zero-padded to a
        multiple of 16 rows: Triton only vectorizes loads from the projected activations when
        their row stride is a multiple of 16 elements (3336 -> 3344 at one TP8 rank)."""
        del device
        weights = [
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.b_proj.weight,
            self.forget_gate.f_a_proj.weight, self.g_a_proj.weight,
        ]
        pad = -sum(w.shape[0] for w in weights) % 16
        weights.append(weights[0].new_zeros(pad, self.hidden_size))
        self._in_proj_weight = torch.cat(weights).detach()

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero per-request state: ``(S (B, H, D, D) fp32, conv (B, 3HD, K-1))``."""
        if device is None:
            device = self.o_proj.weight.device
        recurrent_state = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            dtype=torch.float32, device=device,
        )
        conv_state = torch.zeros(
            batch_size, self.conv_dim, self.conv_kernel_size - 1,
            dtype=self.q_proj.weight.dtype, device=device,
        )
        return recurrent_state, conv_state

    def _mixed_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Raw pre-conv, pre-SiLU projections: ``(B, 3*H*D, L)``, order q|k|v."""
        return torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

    def _split_heads(
        self, conv_out: torch.Tensor, hidden_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(
            conv_out.transpose(1, 2), [self.qkv_dim] * 3, dim=-1
        )
        return query.view(hidden_shape), key.view(hidden_shape), value.view(hidden_shape)

    def _finalize(
        self,
        core_attn_out: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Output gate (from the PRE-conv input) + gated RMSNorm + o_proj."""
        batch_size, seq_len = hidden_states.shape[:2]
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, self.qkv_dim)
        return self.o_proj(output)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full or continued prefill over ``(B, L, hidden)``; chunked kernel."""
        if (recurrent_state is None) != (conv_state is None):
            raise ValueError(
                "pass both recurrent_state and conv_state (continue) or neither "
                "(first prefill); got exactly one"
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        mixed_qkv = self._mixed_qkv(hidden_states)
        tail_len = self.conv_kernel_size - 1
        if conv_state is None:
            conv_out = causal_conv1d_prefill(mixed_qkv, self.conv1d.weight)
            # Raw pre-SiLU tail; the left zero-pad covers L < kernel - 1.
            conv_state = F.pad(mixed_qkv, (tail_len, 0))[:, :, -tail_len:].contiguous()
        else:
            extended = torch.cat([conv_state, mixed_qkv], dim=-1)
            conv_out = causal_conv1d_prefill(extended, self.conv1d.weight)[:, :, -seq_len:]
            conv_state.copy_(extended[:, :, -conv_state.shape[-1]:])

        query, key, value = self._split_heads(conv_out, hidden_shape)

        # Gates from the PRE-conv hidden states — never the conv output.
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        core_attn_out, final_state = chunk_kda(
            query, key, value, g, beta,
            chunk_size=self.chunk_size, initial_state=recurrent_state,
        )
        if recurrent_state is None:
            recurrent_state = final_state.to(torch.float32)
        else:
            recurrent_state.copy_(final_state)

        output = self._finalize(core_attn_out, hidden_states, hidden_shape)
        return output, recurrent_state, conv_state

    def decode_step(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """One decode token ``(B, 1, hidden)``; states mutated IN PLACE."""
        batch_size, seq_len = hidden_states.shape[:2]
        if seq_len != 1:
            raise ValueError(
                f"decode_step is single-token (got seq_len={seq_len}); "
                "multi-token continue goes through prefill()"
            )
        hidden_shape = (batch_size, 1, self.num_heads, self.head_dim)

        mixed_qkv = self._mixed_qkv(hidden_states)
        conv_out = causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight)
        query, key, value = self._split_heads(conv_out, hidden_shape)

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        core_attn_out = recurrent_kda_step(query, key, value, g, beta, recurrent_state)
        return self._finalize(core_attn_out, hidden_states, hidden_shape)
