"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear attention — pure-torch math.

KDA is a gated delta rule: per head, a fixed-size fp32 matrix memory
``S (k_dim, v_dim)`` is decayed PER KEY-CHANNEL every token, written with a
beta-weighted error-correcting update, then read by the query. 34 of the 45
decoder layers are KDA; they carry no RoPE and no position input of any
kind — position enters only through the depthwise causal conv and the
recurrence order. Math and op order are transcribed 1:1 from the reference
(``raw/glm53-flash/transformers/modular_glm5_next.py`` —
``Glm5NextTextLinearAttention``, ``chunk_kimi_delta_attention``,
``recurrent_kimi_delta_attention``) so each path is bit-exact vs the HF
fallback math; derivation and pitfall list:
``drafts/2026-08-31-glm53-kda-spec.md``.

Import discipline: torch + stdlib ONLY (no flashinfer/triton/engine — lane
ground rule 4); the file must import and test on any machine.

Dtype discipline (each line here is a real parity trap):

- conv weight, ``dt_bias`` and ``A_log`` are fp32 (HF
  ``_keep_in_fp32_modules_strict``) and the conv computes in fp32; every
  other weight is bf16. Never blanket-``.to(bf16)`` this module.
- ``g``, the log forget gate, is fp32 from its ``.float()`` on and is a
  smooth parameterization bounded to ``(gate_lower_bound, 0)`` — not a
  clamp. ``beta``'s sigmoid runs in bf16 and is upcast inside the kernel.
- the delta-rule kernels cast q/k/v/g/beta to fp32, l2norm q and k with
  ``x / sqrt(sum(x^2) + 1e-6)`` (the FLA form — NOT ``F.normalize``), THEN
  scale q by ``head_dim**-0.5``; outputs downcast once at kernel exit.
- recurrent state ``S`` is fp32 end to end (a bf16 ``S`` drifts over long
  decodes); conv state stores RAW post-projection, pre-SiLU q|k|v columns
  (bf16-exact), oldest to newest — zero init IS the first-prefill zero pad.

Path selection is per-phase engine logic, never tensor-dependent control
flow: ``prefill`` covers first prefill (no state) and multi-token continue
(chunked-prefill resume, MTP verify) via carried state; ``decode_step`` is
the fixed-shape single-token step — no ``.item()``/``.cpu()``, no
data-dependent branching, CUDA-graph/compile friendly. The delta rule is
not rewindable: rejected speculative tokens must never be committed to
``S`` — the engine checkpoints or defers the state commit (M2).

Weight-loader contract (``weight_loader.py``'s job; shapes recorded here):
the checkpoint stores THREE conv weights ``{q,k,v}_conv1d.weight
[8192, 1, 4]`` bf16, fused as ``cat(dim=0)`` into ``conv1d.weight
[24576, 1, 4]`` (channel order q|k|v, matching the activation concat) and
cast fp32 at load; the ForgetGate params (``A_log [64]`` fp32, ``dt_bias
[8192]`` fp32, ``f_{a,b}_proj``) live FLAT under ``self_attn.`` in the
checkpoint; no KDA tensor is FP8 and none has a bias — exactly 15 tensors
per layer, identical across all 34 KDA layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Glm5NextKdaConfig:
    """Standalone slice of ``Glm5NextModelConfig`` that KDA needs.

    Attribute names match ``mstar/model/glm5_next/config.py`` exactly, so
    ``Glm5NextLinearAttention`` duck-types against either object; this
    dataclass only exists to keep ``kda.py`` importable with no package
    siblings (torch + stdlib only). Defaults are the real checkpoint values
    (config.json ``linear_attn_config``).
    """

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
        """Tiny-dim variant for CPU tests — the spec's parity-measurement
        geometry (H=4, D=32; measured chunk-vs-recurrent max diff 6.1e-5
        on bf16 outputs, ~1e-6 on fp32 states at L=130)."""
        return cls(hidden_size=64, linear_num_heads=4, linear_head_dim=32)


def apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """Zero padded positions BEFORE any projection (2D boolean ``(B, L)`` mask).

    A padded token must never enter a conv window or the state: with
    bias-free projections, x = 0 gives k = v = 0, so the delta rule neither
    decays a live carry (left pad sits ahead of any real token) nor writes.
    """
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-form l2 normalization: ``x / sqrt(sum(x^2) + eps)``, fp32 inputs.

    Intentionally sqrt-add-divide — ``F.normalize`` (max-eps) and rsqrt-mul
    variants are last-ulp different and break bit-exactness vs the
    reference triton kernel's fallback math.
    """
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def causal_conv1d_prefill(mixed_qkv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv + SiLU over ``(B, channels, L)``, fp32 compute.

    Left-pads ``kernel_size - 1`` zeros (F.conv1d pads both sides; keeping
    the first L outputs makes it left-pad-only), then SiLU on ALL of
    q, k, v. ``weight`` is the fused ``(channels, 1, kernel_size)`` fp32
    conv weight; output downcasts to the input dtype after the activation.
    """
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
    """Single/multi-token conv step against a rolled state, fp32 compute.

    ``conv_state (B, channels, W)`` holds the last W RAW post-projection,
    pre-SiLU columns (W >= kernel_size - 1; widths 3 and 4 are both
    bit-exact with the full prefill conv). Rolls the state in place, then
    emits the last ``seq_len`` valid conv outputs + SiLU. Fixed shapes, no
    branches — decode capture-safe.
    """
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
    """Chunked gated delta rule (reference ``chunk_kimi_delta_attention``).

    Inputs ``(B, L, H, D)`` (beta ``(B, L, H)``), any float dtype; all
    compute in fp32 with q/k l2norm and the ``D**-0.5`` q-scale applied
    in-kernel. Right-pads to a multiple of ``chunk_size`` (padded rows are
    provably inert), cumsums ``g`` within chunks, builds
    ``T = (I - A)^-1`` by forward substitution over the strictly-lower
    ``A``, and carries the recurrent state across chunks with per-channel
    decay ``exp(g_C)`` along the KEY dim.

    Returns ``(core_attn_out (B, L, H, D) in the input dtype,
    final_state (B, H, D, D) fp32)``. Multi-token continue (chunked-prefill
    resume, MTP verify) is THIS path with ``initial_state`` — never a
    recurrent python loop. Continue calls shrink the effective chunk to
    cover ``L`` (an M2 k+1-token verify runs a k-iteration substitution
    loop, not 63): chunk size changes rounding, not math, and chunk-vs-
    recurrent parity is tolerance-bar by spec. First prefill keeps the
    fixed reference chunking — it is the one path with an HF bitwise
    counterpart.
    """
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
        # Small-L continue path (docstring): never pad a verify/resume tail
        # past its own length. L >= chunk_size continues are unchanged.
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
    # decay[i, j] = exp(g_i - g_j), (B, H, N, C, C, D). Materialized for
    # ALL chunks at once (1:1 with the reference): at the L=2048 serving
    # cap this transient is ~4.3 GB/layer unsharded (~0.5 GB at TP8) —
    # the known first-OOM candidate as prefill batching grows. The M3
    # lever is a per-chunk recompute (bit-exact: elementwise ops + last-
    # dim reductions on identical slices), cutting it by N.
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
    """One recurrent delta-rule token (reference ``recurrent_kimi_delta_attention``).

    Inputs ``(B, 1, H, D)`` (beta ``(B, 1, H)``); ``state (B, H, D, D)``
    fp32, mutated IN PLACE (persistent per-request slot — no reallocation,
    capture-safe). The order is load-bearing:

    1. ``S *= exp(g_t)``        (decay first, per key-channel)
    2. ``kv_mem = k_t . S``     (read what memory returns for this key)
    3. ``S += k_t (x) ((v_t - kv_mem) * beta_t)``  (delta write)
    4. ``o_t = q_t . S``        (read AFTER the write — token sees itself)

    Returns the core output ``(B, 1, H, D)`` in the input dtype.
    """
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


class Glm5NextForgetGate(nn.Module):
    """Per-head, per-key-channel log forget gate ``g (B, L, H, D)``, fp32.

    ``g = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``
    — a smooth parameterization bounded to ``(gate_lower_bound, 0)``, NOT a
    clamp. The low-rank matmuls run bf16; the ``.float()`` happens BEFORE
    the fp32 ``dt_bias`` add, and everything after is fp32. The HF softplus
    branch (``gate_lower_bound is None``) is dead for this checkpoint and
    deliberately not implemented.
    """

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
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay_rate = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))
        return self.gate_lower_bound * torch.sigmoid(decay_rate * g)


class Glm5NextRMSNormGated(nn.Module):
    """Gated RMSNorm over head_dim, weight shared across heads, strict fp32.

    ``eps`` must be config ``rms_norm_eps = 1e-5`` — NOT the HF class
    default 1e-6. Gate activation is SIGMOID (GLM-5.3 overrides the
    Qwen3-family SiLU). Norm, weight and gating all run fp32; the result
    downcasts once to the input dtype.
    """

    def __init__(self, head_dim: int, eps: float, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Glm5NextLinearAttention(nn.Module):
    """One KDA layer: projections + fused causal conv + delta rule + gated out.

    ``config`` duck-types ``Glm5NextModelConfig`` (or the standalone
    ``Glm5NextKdaConfig``): ``hidden_size``, ``linear_num_heads``,
    ``linear_head_dim``, ``linear_conv_kernel_size``, ``gate_lower_bound``,
    ``rms_norm_eps``. Big trap baked into both paths: ``g``, ``beta`` and
    the output ``gate`` are computed from the PRE-conv layer input, never
    from the conv output.

    Per-request state (see ``init_state``): recurrent ``S (H, D, D)`` fp32
    mandatory (4 MiB/layer full-size; 136 MiB/request over the 34 KDA
    layers) + conv tail ``(3*H*D, kernel-1)`` in the module dtype
    (144 KiB/layer bf16). Fixed size — NOT paged KV.
    """

    def __init__(self, config: Glm5NextKdaConfig, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.conv_kernel_size = config.linear_conv_kernel_size
        # Kernel-internal chunking, not a config field; 64 matches the
        # reference default and the parity measurements.
        self.chunk_size = 64

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False, dtype=dtype)
        # Fused depthwise conv over cat(q, k, v); fp32 weight, no bias
        # (loader fuses the checkpoint's three [8192, 1, 4] bf16 taps and
        # upcasts). padding is unused (both paths call F.conv1d directly)
        # but kept so the module mirrors the reference geometry.
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

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero per-request state: ``(S (B, H, D, D) fp32, conv (B, 3HD, K-1))``.

        Zero conv init is exactly the zero left-padding of a first prefill —
        one convention shared by both paths.
        """
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
        """Full or continued prefill over ``(B, L, hidden)``; chunked kernel.

        With no state: first prefill (zero left-context conv, delta rule
        from zeros). With both states: multi-token continue — prepend the
        cached raw conv tail, keep the LAST L conv outputs (bit-exact with
        one long conv), and chunk with ``initial_state`` (MTP verify and
        chunked-prefill resume take this path, never a recurrent loop).
        Passed-in states are updated IN PLACE and also returned; fresh ones
        are allocated when none are passed. ``attention_mask`` is a 2D
        boolean padding mask applied to ``hidden_states`` before any
        projection.

        Returns ``(output (B, L, hidden), recurrent_state fp32, conv_state)``.
        """
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
        """One decode token ``(B, 1, hidden)``; states mutated IN PLACE.

        Fixed shapes, no host syncs, no data-dependent branching — the
        whole step is capture/compile friendly. Both states are required:
        the first generated token always follows a ``prefill``.
        """
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
