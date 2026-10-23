"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear attention — pure-torch math.

Also the KDA resource's reference kernel bundle (``TorchKDAKernels``), which runs this math
against the recurrent pool's slots where the fused Triton kernels do not.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


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


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-form l2 normalization: ``x / sqrt(sum(x^2) + eps)``, fp32 inputs."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def causal_conv1d_prefill(mixed_qkv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv + SiLU over ``(B, channels, L)``, fp32 compute."""
    seq_len = mixed_qkv.shape[-1]
    num_channels = mixed_qkv.shape[1]
    out = F.conv1d(
        mixed_qkv.to(weight.dtype),
        weight=weight,
        bias=None,
        padding=weight.shape[-1] - 1,
        groups=num_channels,
    )[:, :, :seq_len]
    return F.silu(out).to(mixed_qkv.dtype)


def causal_conv1d_update(
    mixed_qkv: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Single/multi-token conv step against a rolled state, fp32 compute."""
    seq_len = mixed_qkv.shape[-1]
    num_channels = mixed_qkv.shape[1]
    state_len = conv_state.shape[-1]
    window = torch.cat([conv_state, mixed_qkv], dim=-1).to(weight.dtype)
    conv_state.copy_(window[:, :, -state_len:])
    out = F.conv1d(window, weight=weight, bias=None, padding=0, groups=num_channels)
    out = out[:, :, -seq_len:]
    return F.silu(out).to(mixed_qkv.dtype)


def chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule (reference ``chunk_kimi_delta_attention``)."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    # FLA computes these in fp32 — after the casts, before the scale.
    query = l2norm(query)
    key = l2norm(key)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    if initial_state is not None:
        # Clamp the chunk to a continue's own length: a short resume/verify
        # tail is never padded out. L >= chunk_size continues are unchanged.
        chunk_size = min(chunk_size, sequence_length)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # (B, H, num_chunks, chunk_size, D)
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Per-head AND per-key-channel decay — the KDA/GDN divergence.
    g = g.cumsum(dim=-2)
    upper_incl_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    # decay[i, j] = exp(g_i - g_j), (B, H, N, C, C, D), materialized for ALL
    # chunks at once as the reference does. It is the largest transient in
    # prefill, and recomputing it per chunk is how to shrink it.
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = (
        -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask)
        .sum(dim=-1)
        .masked_fill(upper_incl_diag, 0)
    )
    # T = (I - A)^-1 by forward substitution over the strictly-lower A.
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    t_mat = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    u = t_mat @ v_beta  # (B, H, N, C, D)
    w = t_mat @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=u.dtype, device=u.device,
        )
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    core_attn_out = torch.zeros_like(u)
    strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        g_i = g[:, :, i]

        # Inter chunk: decay-weighted read of the carry-in state.
        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        # Intra chunk: diagonal INCLUDED — token i reads its own write.
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
            .sum(dim=-1)
            .masked_fill(strict_upper, 0)
        )
        v_prime = w[:, :, i] @ last_recurrent_state
        v_new = u[:, :, i] - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        # exp(g_C) broadcasts over the v-dim: decay along the KEY dim of S.
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


def recurrent_kda_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """One recurrent delta-rule token (reference ``recurrent_kimi_delta_attention``)."""
    if state.dtype != torch.float32:
        raise ValueError(f"KDA recurrent state must be fp32, got {state.dtype}")
    initial_dtype = query.dtype
    query, key, value, g, beta = [
        x.to(torch.float32) for x in (query, key, value, g, beta)
    ]

    query = l2norm(query)
    key = l2norm(key)
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    q_t = query[:, 0]  # (B, H, D)
    k_t = key[:, 0]
    v_t = value[:, 0]
    decay_t = g[:, 0].unsqueeze(-1).exp()  # (B, H, D, 1) — key-dim decay
    beta_t = beta[:, 0].unsqueeze(-1)  # (B, H, 1)

    state.mul_(decay_t)
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, D_v)
    delta = (v_t - kv_mem) * beta_t
    state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2))
    core_attn_out = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.unsqueeze(1).to(initial_dtype)


def forget_gate(
    f: torch.Tensor, dt_bias: torch.Tensor, a_log: torch.Tensor, lower_bound: float,
) -> torch.Tensor:
    """Log forget gate ``(..., H, D)`` fp32 in (lower_bound, 0) from the up-projected
    ``f (..., H, D)``: ``lower_bound * sigmoid(exp(A) * (f + dt_bias))``."""
    g = f.float() + dt_bias.float().view(f.shape[-2:])
    decay_rate = torch.exp(a_log.float()).view(-1, 1)
    return lower_bound * torch.sigmoid(decay_rate * g)


def gated_rms_norm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    """RMSNorm over the last dim times ``sigmoid(gate)``, strict fp32, in ``x``'s dtype."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = weight.to(torch.float32) * x * torch.sigmoid(gate.to(torch.float32))
    return x.to(input_dtype)


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


class TorchKDAKernels:
    """The KDA resource's reference bundle: this module's torch math against the pool's
    slots. Decode and verify gather and scatter the rows' slots on device; prefill walks the
    spans on the host, so it cannot be captured. The pool's state is V-first, ``[H, V, K]``;
    this math runs on K-first copies of it.

    Takes what the fused bundle takes (the low-rank gates with ``params.f_b`` / ``g_b``) or
    the full-rank forget gate, and returns what it would.
    """

    def layout(self, spans, num_tokens, fixed):
        return None

    @torch.compiler.disable
    def run_paged(self, qkv, g, beta, plan, conv_state, rec_state, p, gate=None):
        h, d = p.num_heads, p.head_dim
        t = qkv.shape[0]
        f = F.linear(g, p.f_b) if p.f_b is not None else g
        g_log = forget_gate(f.reshape(t, h, d), p.dt_bias, p.A_log, p.lower_bound)
        beta = torch.sigmoid(beta)
        weight = p.conv_weight.unsqueeze(1)  # the depthwise conv's [3P, 1, W]
        if plan.is_decode:
            out = self._decode(qkv, g_log, beta, plan, conv_state, rec_state, weight, h, d)
        else:
            out = self._prefill(qkv, g_log, beta, plan, conv_state, rec_state, weight, h, d)
        if gate is None or p.g_b is None:
            return out
        gate = F.linear(gate, p.g_b).view(t, h, d)
        return gated_rms_norm(out, gate, p.norm_weight, p.norm_eps).reshape(t, h * d)

    @staticmethod
    def _decode(qkv, g_log, beta, plan, conv_state, rec_state, weight, h, d):
        slots = plan.slot_ids.long()
        conv = conv_state.index_select(0, slots)
        rec = rec_state.index_select(0, slots).transpose(-1, -2).contiguous()
        y = causal_conv1d_update(qkv.unsqueeze(-1), conv, weight).squeeze(-1)
        q, k, v = (x.reshape(-1, 1, h, d) for x in y.split(h * d, dim=-1))
        out = recurrent_kda_step(q, k, v, g_log.unsqueeze(1), beta.unsqueeze(1), rec)
        conv_state.index_copy_(0, slots, conv)
        rec_state.index_copy_(0, slots, rec.transpose(-1, -2))
        return out.squeeze(1)

    @staticmethod
    def _prefill(qkv, g_log, beta, plan, conv_state, rec_state, weight, h, d):
        out = torch.empty(qkv.shape[0], h, d, dtype=qkv.dtype, device=qkv.device)
        start = 0
        for n, slot in zip(plan.spans, plan.slot_ids.tolist(), strict=True):
            rows = slice(start, start + n)
            start += n
            if n == 0:
                continue
            # the slot's conv tail in place, its state K-first: a fresh slot is zeros
            conv = conv_state[slot].unsqueeze(0)
            rec = rec_state[slot].transpose(-1, -2).unsqueeze(0).contiguous()
            extended = torch.cat([conv, qkv[rows].t().unsqueeze(0)], dim=-1)
            y = causal_conv1d_prefill(extended, weight)[:, :, -n:].transpose(1, 2)
            conv.copy_(extended[:, :, -conv.shape[-1]:])
            q, k, v = (x.reshape(1, n, h, d) for x in y.split(h * d, dim=-1))
            core, final = chunk_kda(
                q, k, v, g_log[rows].unsqueeze(0), beta[rows].unsqueeze(0),
                initial_state=rec,
            )
            rec_state[slot].copy_(final[0].transpose(-1, -2))
            out[rows] = core[0]
        return out

    @torch.compiler.disable
    def run_verify(self, qkv, g, beta, plan, conv_state, rec_state, spec, p, gate=None):
        """``kda_triton.kda_verify``'s scheme with the decode math: replay each row's accepted
        prefix from the checkpoint, store the state there, run the block and cache its inputs
        on the side the step does not read."""
        h, d = p.num_heads, p.head_dim
        t = qkv.shape[0]
        f = F.linear(g, p.f_b) if p.f_b is not None else g.reshape(t, h * d)
        out = self._verify(qkv, f, beta, plan, conv_state, rec_state, spec, p)
        if gate is None or p.g_b is None:
            return out
        gate = F.linear(gate, p.g_b).view(t, h, d)
        return gated_rms_norm(out, gate, p.norm_weight, p.norm_eps).reshape(t, h * d)

    @staticmethod
    def _verify(qkv, f, beta, plan, conv_state, rec_state, spec, p):
        """All rows at once, a row advancing through its prefix only while the token was
        accepted (a mask where the kernel branches): no host reads, so a graph captures it."""
        h, d = p.num_heads, p.head_dim
        n, block = plan.num_rows, plan.block
        weight = p.conv_weight.unsqueeze(1)  # the depthwise conv's [3P, 1, W]
        slots = plan.slot_ids[:n].long()
        kmax = spec.prefix.shape[2]
        plen = spec.prefix_len[:, 0].index_select(0, slots).clamp(0, kmax)
        side = spec.side[:, 0].index_select(0, slots).long() & 1
        other = 1 - side

        def token(x, f_t, b_t, win, rec):
            """One token of every row, on copies of the window and the K-first state."""
            win, rec = win.clone(), rec.clone()
            y = causal_conv1d_update(x.unsqueeze(-1), win, weight).squeeze(-1)
            q, k, v = (z.reshape(n, 1, h, d) for z in y.split(h * d, dim=-1))
            g_log = forget_gate(f_t.reshape(n, h, d), p.dt_bias, p.A_log, p.lower_bound)
            b_t = torch.sigmoid(b_t)
            out = recurrent_kda_step(q, k, v, g_log.unsqueeze(1), b_t.unsqueeze(1), rec)
            return out.squeeze(1), win, rec

        # the conv window before the prefix: the pool's before a request's first verify
        win = torch.where((plen == 0)[:, None, None], conv_state.index_select(0, slots),
                          spec.conv[slots, side])
        rec = rec_state.index_select(0, slots).transpose(-1, -2).contiguous()
        for i in range(kmax):
            _, w, r = token(spec.prefix[slots, side, i], spec.g[slots, side, i],
                            spec.beta[slots, side, i], win, rec)
            live = plen > i
            win = torch.where(live[:, None, None], w, win)
            rec = torch.where(live[:, None, None, None], r, rec)
        rec_state.index_copy_(0, slots, rec.transpose(-1, -2))
        spec.conv[slots, other] = win.to(spec.conv.dtype)
        outs = []
        for j in range(block):
            x, f_j, b_j = qkv[j::block], f[j::block], beta[j::block]
            o, win, rec = token(x, f_j, b_j, win, rec)
            outs.append(o)
            spec.prefix[slots, other, j] = x.to(spec.prefix.dtype)
            spec.g[slots, other, j] = f_j.reshape(n, h, d).to(spec.g.dtype)
            spec.beta[slots, other, j] = b_j.to(spec.beta.dtype)
        return torch.stack(outs, 1).reshape(n * block, h, d)
