"""Kimi Delta Attention reference (spec C): short causal conv, lower-bounded gate, delta-rule
recurrence, gated output RMSNorm. Pure PyTorch, fp32 math, any device.

State conventions
-----------------
* Conv cache per projection: ``[D, W]`` = the last ``W`` *inputs* (oldest first), exactly
  fla's ``ShortConvolution`` cache ``[N, D, W]``. Only the last ``W - 1`` entries influence
  the next token; the full window is kept so that a decode step is a roll + dot.
* Recurrent state: ``[H, K, V]`` ("K-first") inside this module. The serving kernels
  (FlashKDA, fla with ``transpose_state_layout``/``state_v_first``) store ``[H, V, K]``;
  :func:`to_v_first` / :func:`from_v_first` convert. With K == V == 128 this is a plain
  transpose of the last two dims.

The module also provides drop-in CPU stand-ins for the three fla symbols the HF modeling
code uses (``ShortConvolution``, ``FusedRMSNormGated``, ``chunk_kda`` /
``fused_recurrent_kda``) so the HF reference model runs without Triton.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

L2NORM_EPS = 1e-6


# ----------------------------------------------------------------------------- pieces
def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv over one sequence.

    ``x [T, D]``, ``weight [D, W]`` (or ``[D, 1, W]``), ``cache [D, W]`` previous inputs.
    Returns ``(y [T, D], new_cache [D, W])``.
    """
    if weight.dim() == 3:
        weight = weight[:, 0, :]
    d, w = weight.shape
    t = x.shape[0]
    xf = x.float()
    if cache is None:
        hist = torch.zeros(w - 1, d, dtype=torch.float32, device=x.device)
    else:
        hist = cache.float().t()[1:]  # the last W-1 inputs, oldest first
    xp = torch.cat([hist, xf], dim=0)  # [W-1+T, D]
    y = torch.zeros(t, d, dtype=torch.float32, device=x.device)
    for j in range(w):
        y += xp[j : j + t] * weight[:, j].float()
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(activation)
    new_cache = xp[-w:].t().contiguous()  # [D, W], the last W inputs
    return y.to(x.dtype), new_cache.to(x.dtype)


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)


def kda_gate(
    g_raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None,
) -> torch.Tensor:
    """Log-decay ``[.., H, K]`` from the raw gate logits ``g_raw [.., H, K]``,
    ``A_log [H]`` and ``dt_bias [H*K]``."""
    h, k = g_raw.shape[-2], g_raw.shape[-1]
    z = g_raw.float() + dt_bias.float().view(h, k)
    a = torch.exp(A_log.float()).view(h, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * z)
    return -a * F.softplus(z)


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta-rule scan over one sequence, all inputs already prepared:
    ``q, k`` L2-normalized ``[T, H, K]``, ``v [T, H, V]``, ``g_log [T, H, K]`` (log-decay),
    ``beta [T, H]`` (post-sigmoid), ``state [H, K, V]`` fp32 or None.
    Returns ``(o [T, H, V] fp32, state [H, K, V] fp32)``."""
    t, h, kd = q.shape
    vd = v.shape[-1]
    if scale is None:
        scale = kd ** -0.5
    q = q.float() * scale
    k = k.float()
    v = v.float()
    g_log = g_log.float()
    beta = beta.float()
    s = torch.zeros(h, kd, vd, dtype=torch.float32, device=q.device) if state is None else state.float().clone()
    o = torch.zeros(t, h, vd, dtype=torch.float32, device=q.device)
    for i in range(t):
        s = s * torch.exp(g_log[i])[:, :, None]  # Diag(alpha) S: scale row k
        u = v[i] - torch.einsum("hk,hkv->hv", k[i], s)  # v - S^T k
        u = u * beta[i][:, None]
        s = s + torch.einsum("hk,hv->hkv", k[i], u)  # + k u^T
        o[i] = torch.einsum("hk,hkv->hv", q[i], s)  # S^T q
    return o, s


def gated_rms_norm(o: torch.Tensor, g_out: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``RMSNorm_headdim(o) * weight * sigmoid(g_out)`` per head, fp32 math."""
    of = o.float()
    y = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    y = y * torch.sigmoid(g_out.float())
    return y.to(o.dtype)


def to_v_first(state_kv: torch.Tensor) -> torch.Tensor:
    return state_kv.transpose(-1, -2).contiguous()


def from_v_first(state_vk: torch.Tensor) -> torch.Tensor:
    return state_vk.transpose(-1, -2).contiguous()


# ----------------------------------------------------------------------------- layer
@dataclass
class KDAWeights:
    q_proj: torch.Tensor  # [H*K, hidden]
    k_proj: torch.Tensor
    v_proj: torch.Tensor  # [H*V, hidden]
    q_conv: torch.Tensor  # [H*K, W] (or [H*K, 1, W])
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    f_a_proj: torch.Tensor  # [K, hidden]
    f_b_proj: torch.Tensor  # [H*K, K]
    b_proj: torch.Tensor  # [H, hidden]
    g_proj: torch.Tensor  # [H*V, hidden]
    A_log: torch.Tensor  # [H]
    dt_bias: torch.Tensor  # [H*K]
    o_norm: torch.Tensor  # [V]
    o_proj: torch.Tensor  # [hidden, H*V]
    num_heads: int
    head_dim: int
    lower_bound: float | None = -5.0
    norm_eps: float = 1e-5

    @classmethod
    def from_hf_module(cls, m) -> KDAWeights:
        return cls(
            q_proj=m.q_proj.weight, k_proj=m.k_proj.weight, v_proj=m.v_proj.weight,
            q_conv=m.q_conv1d.weight, k_conv=m.k_conv1d.weight, v_conv=m.v_conv1d.weight,
            f_a_proj=m.f_a_proj.weight, f_b_proj=m.f_b_proj.weight, b_proj=m.b_proj.weight,
            g_proj=m.g_proj.weight, A_log=m.A_log, dt_bias=m.dt_bias, o_norm=m.o_norm.weight,
            o_proj=m.o_proj.weight, num_heads=m.num_heads, head_dim=m.head_dim,
            lower_bound=m.gate_lower_bound, norm_eps=m.o_norm.eps,
        )


@dataclass
class KDAState:
    conv_q: torch.Tensor | None = None  # [D, W]
    conv_k: torch.Tensor | None = None
    conv_v: torch.Tensor | None = None
    recurrent: torch.Tensor | None = None  # [H, K, V] fp32


def kda_layer_forward(
    w: KDAWeights, x: torch.Tensor, state: KDAState | None = None,
) -> tuple[torch.Tensor, KDAState]:
    """One sequence ``x [T, hidden]`` (already input-layernormed) -> ``([T, hidden], state)``."""
    state = state or KDAState()
    h, d = w.num_heads, w.head_dim
    t = x.shape[0]
    q, cq = short_conv(F.linear(x, w.q_proj), w.q_conv, state.conv_q)
    k, ck = short_conv(F.linear(x, w.k_proj), w.k_conv, state.conv_k)
    v, cv = short_conv(F.linear(x, w.v_proj), w.v_conv, state.conv_v)
    g_raw = F.linear(F.linear(x, w.f_a_proj), w.f_b_proj).view(t, h, d)
    beta = torch.sigmoid(F.linear(x, w.b_proj).float())  # [T, H]
    g_log = kda_gate(g_raw, w.A_log, w.dt_bias, w.lower_bound)
    qn = l2norm(q.view(t, h, d))
    kn = l2norm(k.view(t, h, d))
    o, s = kda_recurrent(qn, kn, v.view(t, h, d), g_log, beta, state.recurrent)
    g_out = F.linear(x, w.g_proj).view(t, h, d)
    y = gated_rms_norm(o.to(x.dtype), g_out, w.o_norm, w.norm_eps)
    out = F.linear(y.reshape(t, h * d), w.o_proj)
    return out, KDAState(conv_q=cq, conv_k=ck, conv_v=cv, recurrent=s)


# ----------------------------------------------------------------- fla stand-ins (CPU)
class TorchShortConvolution(nn.Conv1d):
    """Same constructor/forward contract as ``fla.modules.ShortConvolution``."""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: str | None = "silu", backend=None, device=None, dtype=None, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias,
                         padding=kernel_size - 1, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs):
        assert residual is None and mask is None
        b, t, d = x.shape
        if cu_seqlens is not None:
            assert b == 1
            bounds = cu_seqlens.tolist()
        else:
            bounds = None
        ys, caches = [], []
        n = b if bounds is None else len(bounds) - 1
        for i in range(n):
            xi = x[i] if bounds is None else x[0, bounds[i]:bounds[i + 1]]
            ci = None if cache is None else cache[i]
            y, c = short_conv(xi, self.weight, ci, self.activation)
            ys.append(y)
            caches.append(c)
        y = torch.stack(ys) if bounds is None else torch.cat(ys)[None]
        new_cache = torch.stack(caches)
        if cache is not None:
            cache.copy_(new_cache.to(cache.dtype))
            new_cache = cache
        return y, (new_cache if output_final_state else None)


class TorchRMSNormGated(nn.Module):
    """Same contract as ``fla.modules.FusedRMSNormGated`` (norm, weight, then gate)."""

    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5,
                 activation: str = "swish", device=None, dtype=None):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        gf = g.float()
        if self.activation in ("swish", "silu"):
            y = y * gf * torch.sigmoid(gf)
        elif self.activation == "sigmoid":
            y = y * torch.sigmoid(gf)
        else:
            raise ValueError(self.activation)
        return y.to(x.dtype)


def _prep_inputs(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm_in_kernel,
                 use_gate_in_kernel, use_beta_sigmoid_in_kernel, lower_bound):
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    return q, k, v, g, beta


def torch_chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, A_log=None, dt_bias=None,
                    safe_gate=False, lower_bound=None, transpose_state_layout=False,
                    state_v_first=False, cu_seqlens=None, **kwargs):
    """CPU stand-in for ``fla.ops.kda.chunk_kda`` / ``fused_recurrent_kda`` (inference
    semantics: forward only). Layout ``[B, T, H, *]``; state ``[N, H, V, K]`` when
    ``transpose_state_layout`` or ``state_v_first`` else ``[N, H, K, V]``."""
    v_first = transpose_state_layout or state_v_first
    q, k, v, g, beta = _prep_inputs(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm_in_kernel,
                                    use_gate_in_kernel, use_beta_sigmoid_in_kernel, lower_bound)
    b = q.shape[0]
    bounds = None if cu_seqlens is None else cu_seqlens.tolist()
    n = b if bounds is None else len(bounds) - 1
    outs, states = [], []
    for i in range(n):
        sl = (i, slice(None)) if bounds is None else (0, slice(bounds[i], bounds[i + 1]))
        s0 = None
        if initial_state is not None:
            s0 = initial_state[i]
            if v_first:
                s0 = from_v_first(s0)
        o, s = kda_recurrent(q[sl], k[sl], v[sl], g[sl], beta[sl], s0, scale)
        outs.append(o.to(v.dtype))
        states.append(to_v_first(s) if v_first else s)
    o = torch.stack(outs) if bounds is None else torch.cat(outs)[None]
    return o, (torch.stack(states) if output_final_state else None)


torch_fused_recurrent_kda = torch_chunk_kda


def patch_hf_modeling_for_cpu(mod) -> None:
    """Replace the fla symbols in an imported HF Kimi modeling module with the torch
    stand-ins, so ``KimiDeltaAttention`` runs without Triton."""
    mod.ShortConvolution = TorchShortConvolution
    mod.FusedRMSNormGated = TorchRMSNormGated
    mod.chunk_kda = torch_chunk_kda
    mod.fused_recurrent_kda = torch_fused_recurrent_kda
