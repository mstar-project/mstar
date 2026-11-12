"""Torch stand-ins for the GDN kernels, so the linear-attention resource runs on
the CPU: FlashInfer's chunked, one-token and multi-token gated delta rule, and
the vendored Triton short conv (varlen, and one or more tokens a row).
Sequential and in the inputs' dtype (float32 or float64), they are also the
tests' reference recurrence.

The delta rule, per token, on a K-last state ``S [HV, V, K]`` (FlashInfer's
pool layout): decay ``S *= alpha``, then ``S += beta * (v - S k) k^T``, and the
output ``S q * scale``. A v-head reads k-head ``h // (HV / H)``. A closed gate
(``alpha = 1``, ``beta = 0``) leaves ``S`` unchanged.
"""
from __future__ import annotations

import itertools
import sys
import types

import torch
import torch.nn.functional as F


def delta_rule(q, k, v, alpha, beta, state, scale):
    """One sequence's recurrence from ``state [HV, V, K]``: outputs ``[T, HV, V]`` and the final state."""
    hv, h = v.shape[1], k.shape[1]
    group = hv // h
    q = q.repeat_interleave(group, dim=1)
    k = k.repeat_interleave(group, dim=1)
    s = state.clone()
    out = []
    for t in range(v.shape[0]):
        s = s * alpha[t][:, None, None]
        sk = torch.einsum("hvk,hk->hv", s, k[t])
        s = s + beta[t][:, None, None] * torch.einsum("hv,hk->hvk", v[t] - sk, k[t])
        out.append(torch.einsum("hvk,hk->hv", s, q[t]) * scale)
    return (torch.stack(out) if out else v.new_zeros(0, *v.shape[1:])), s


def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=True,
                           cu_seqlens=None, use_qk_l2norm_in_kernel=False, **kwargs):
    """``flashinfer.gdn_prefill.chunk_gated_delta_rule``: ``g`` is the decay,
    exponentiated, as the wrapper passes it. With ``state_checkpoints``, row
    ``r``'s state after each ``checkpoint_every_n_tokens`` of its tokens goes
    to ``state_checkpoints[checkpoint_cu_starts[r] + i]`` (the prompt
    snapshots a prefix cache restores)."""
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    checkpoints, every = kwargs.get("state_checkpoints"), kwargs.get("checkpoint_every_n_tokens") or 0
    starts = kwargs["checkpoint_cu_starts"].tolist() if checkpoints is not None and every else None
    cu = cu_seqlens.tolist()
    out = torch.zeros_like(v)
    finals = []
    for row in range(len(cu) - 1):
        s, e = cu[row], cu[row + 1]
        state = initial_state[row]
        # piece by piece at the checkpoints: the same sequential recurrence
        bounds = list(range(s, e, every)) + [e] if starts is not None else [s, e]
        for i, (a, b) in enumerate(itertools.pairwise(bounds)):
            o, state = delta_rule(q[a:b], k[a:b], v[a:b], g[a:b], beta[a:b], state, scale)
            out[a:b] = o.to(out.dtype)
            if starts is not None and b - a == every:
                checkpoints[starts[row] + i] = state.to(checkpoints.dtype)
        finals.append(state)
    return out, torch.stack(finals)


def causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc, cache_indices=None, has_initial_state=None,
                     activation="silu", null_block_id=-1, **kwargs):
    """The varlen depthwise conv: ``x [dim, tokens]``, each row continuing the
    window its slot holds (zeros without state), the slot left holding the
    row's last ``width - 1`` inputs."""
    out = torch.zeros_like(x)
    width = weight.shape[1] - 1
    cu = query_start_loc.tolist()
    for row in range(len(cu) - 1):
        s, e = cu[row], cu[row + 1]
        slot = int(cache_indices[row])
        if slot == null_block_id or e <= s:
            continue
        window = conv_states[slot].to(x.dtype) if bool(has_initial_state[row]) else x.new_zeros(x.shape[0], width)
        seq = torch.cat([window, x[:, s:e]], dim=1)
        y = sum(weight[:, j: j + 1] * seq[:, j: j + e - s] for j in range(width + 1))
        if bias is not None:
            y = y + bias[:, None]
        out[:, s:e] = F.silu(y) if activation in ("silu", "swish") else y
        conv_states[slot] = seq[:, -width:].to(conv_states.dtype)
    return out


def gated_delta_rule_decode_pretranspose(q, k, v, state, A_log, a, dt_bias, b, scale=None, output=None,
                                         initial_state=None, initial_state_indices=None, use_qk_l2norm=True):
    """``flashinfer.gdn_decode``'s one-token step, for the decode wrapper:
    ``q/k/v [rows, 1, heads, dim]``, raw gates, each row's state updated in
    place at its slot of the pool ``initial_state``."""
    assert state is None
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, a, b = q[:, 0], k[:, 0], v[:, 0], a[:, 0], b[:, 0]
    if use_qk_l2norm:
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    alpha = torch.exp(-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    out = v.new_zeros(v.shape, dtype=torch.float32)
    for row in range(v.shape[0]):
        slot = int(initial_state_indices[row])
        if slot < 0:
            continue
        s0 = initial_state[slot].float()
        o, final = delta_rule(q[row: row + 1].float(), k[row: row + 1].float(), v[row: row + 1].float(),
                              alpha[row: row + 1], beta[row: row + 1], s0, scale)
        out[row] = o[0]
        initial_state[slot] = final.to(initial_state.dtype)
    if output is not None:
        output.copy_(out.unsqueeze(1))
        return output, None
    return out.to(v.dtype).unsqueeze(1), None


def gated_delta_rule_mtp(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b, scale=None,
                         output=None, disable_state_update=None, use_qk_l2norm=True, **kwargs):
    """``flashinfer.gdn_decode``'s multi-token step, for the verify wrapper:
    ``q/k/v [rows, T, heads, dim]``, raw gates ``[rows, T, HV]``, each row from
    its slot of the pool ``initial_state``; a negative slot is skipped (its
    output left as ``output`` had it); the final state written back unless
    ``disable_state_update``."""
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    if use_qk_l2norm:
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    alpha = torch.exp(-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    out = output if output is not None else v.new_zeros(v.shape)
    for row in range(v.shape[0]):
        slot = int(initial_state_indices[row])
        if slot < 0:
            continue
        o, final = delta_rule(q[row].float(), k[row].float(), v[row].float(), alpha[row], beta[row],
                              initial_state[slot].float(), scale)
        out[row] = o.to(out.dtype)
        if not disable_state_update:
            initial_state[slot] = final.to(initial_state.dtype)
    return out, None


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                         null_block_id=-1, out=None, update_state=True, **kwargs):
    """The conv of each row's next tokens: ``x [rows, dim]`` (one token) or
    ``[rows, dim, T]``, each row's window at its slot of ``conv_state [slots,
    dim, width]`` slid by its tokens in place (left as it was with
    ``update_state=False``); a null-slot row is skipped, its output left as
    ``out`` had it."""
    one = x.dim() == 2
    xs = x[..., None] if one else x
    result = torch.zeros_like(xs) if out is None else (out[..., None] if one else out)
    width = conv_state.shape[-1]
    for row in range(xs.shape[0]):
        slot = int(conv_state_indices[row])
        if slot == null_block_id:
            continue
        seq = torch.cat([conv_state[slot].to(xs.dtype), xs[row]], dim=1)
        tokens = xs.shape[-1]
        y = sum(weight[:, j: j + 1] * seq[:, j: j + tokens] for j in range(width + 1))
        if bias is not None:
            y = y + bias[:, None]
        result[row] = (F.silu(y) if activation in ("silu", "swish", True) else y).to(result.dtype)
        if update_state:
            conv_state[slot] = seq[:, -width:].to(conv_state.dtype)
    return result[..., 0] if one else result


def install(monkeypatch) -> None:
    """Route the GDN resource's kernels to the stand-ins, and its compiled
    helpers to plain calls; its verify check sees CUDA, where the real kernels run."""
    import mstar.utils.causal_conv1d as conv
    from mstar.engine.resources.linear_attn import gdn, wrappers

    prefill = types.ModuleType("flashinfer.gdn_prefill")
    prefill.chunk_gated_delta_rule = chunk_gated_delta_rule
    decode = types.ModuleType("flashinfer.gdn_decode")
    decode.gated_delta_rule_decode_pretranspose = gated_delta_rule_decode_pretranspose
    decode.gated_delta_rule_mtp = gated_delta_rule_mtp
    package = sys.modules.get("flashinfer") or types.ModuleType("flashinfer")
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", prefill)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", decode)
    monkeypatch.setattr(conv, "causal_conv1d_fn", causal_conv1d_fn)
    monkeypatch.setattr(conv, "causal_conv1d_update", causal_conv1d_update)
    monkeypatch.setattr(wrappers, "compiled", lambda fn, **options: fn)
    monkeypatch.setattr(gdn, "compiled", lambda fn, **options: fn)
    monkeypatch.setattr(gdn, "current_platform", types.SimpleNamespace(is_cuda=lambda: True, name="cuda"))
