"""Nemotron-H hybrid Mamba-2 / attention / MLP backbone (the ``nano`` LLM).

M* port of ``NemotronHForCausalLM`` (HF ``model_type='nemotron_h'``). Layers are
heterogeneous, driven by ``NanoConfig.hybrid_override_pattern``:

    * ``M`` Mamba-2 mixer   — recurrent (conv + ssm) state per request
    * ``*`` self-attention  — NoPE, GQA, paged KV cache (only these get a cache)
    * ``-`` MLP             — squared-ReLU, non-gated

Module tree mirrors the checkpoint (``stt_model.`` prefix stripped by the
remapper) so weights load by name with no per-tensor renames:

    embed_tokens.weight              -> self.embed_tokens
    lm_head.weight                   -> self.lm_head
    function_head.weight             -> self.function_head
    llm.norm_f.weight                -> self.llm.norm_f
    llm.layers.{i}.norm.weight       -> self.llm.layers[i].norm
    llm.layers.{i}.mixer.*           -> self.llm.layers[i].mixer

Design notes:
  * NoPE — attention layers apply no RoPE; we subclass ``Attention`` and make
    ``_apply_rope`` a no-op. The layers bind the ``NANO_KV`` / ``NANO_ATTN``
    resources the model declares (no position resource).
  * Mamba state — the conv window and the SSM state of every Mamba-2 layer
    are slots in the engine's recurrent-state pool (``MAMBA_STATE``), stepped
    by the ``MAMBA`` resource: the mixer binds a ``Mamba2Callable`` and calls
    ``mix.conv`` then ``mix(...)`` per layer, like the attention layers call
    ``attend``. Fixed-address, so the decode step captures as a CUDA graph.

The Mamba-2 forward is a pure-PyTorch reference scan (no ``mamba_ssm`` kernel
dependency) — correct but not fast; the kernel fast path is a later swap.
TODO(verify): numeric parity of the scan against the HF NemotronHMamba2Mixer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.convenience import Mamba2Callable
from mstar.model.components.attention import Attention
from mstar.model.components.norm import RMSNorm
from mstar.model.nemotron_duplex.config import MAMBA, MAMBA_STATE, NANO_ATTN, NANO_KV, NanoConfig

# ---------------------------------------------------------------------------
# Cached decode state (engine-free O(T) path; additive to offline_forward)
# ---------------------------------------------------------------------------


@dataclass
class NemotronHCache:
    """Per-layer decode cache for the O(T) cached path (single sequence).

    Populated by :meth:`NemotronHLLM.prefill` and mutated in place by
    :meth:`NemotronHLLM.decode_step`. Only the layer kinds that carry state
    have entries:

        * ``attn[layer_idx]``  -> ``[k, v]``, each ``(Lh, num_kv_heads, head_dim)``
          — the running key/value history for a ``*`` attention layer (NoPE, so
          no per-position rotation to reconcile: entries are position-agnostic).
        * ``conv[layer_idx]``  -> ``(conv_dim, conv_kernel-1)`` — the rolling
          buffer of the last ``conv_kernel-1`` *raw* (pre-conv, pre-silu) xBC
          inputs for an ``M`` mamba layer. FIXED length, left-zero-padded when
          fewer than ``conv_kernel-1`` tokens have been seen.
        * ``ssm[layer_idx]``   -> ``(nheads, head_dim, d_state)`` — the SSM
          recurrent state for an ``M`` mamba layer (fp32).

    ``-`` MLP layers are stateless and appear in none of the dicts.
    """

    attn: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    conv: dict[int, torch.Tensor] = field(default_factory=dict)
    ssm: dict[int, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mixers
# ---------------------------------------------------------------------------


def _rms_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Llama-style RMSNorm in fp32 (offline path — avoids the bf16 fused kernel)."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


class NemotronHAttention(Attention):
    """GQA self-attention with NoPE (Nemotron-H applies no positional encoding)."""

    def _apply_rope(self, q, k, label):  # noqa: ARG002 - override to disable RoPE
        return q, k

    def offline_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Eager full-sequence causal self-attention for offline inference.

        ``hidden_states``: ``(L, H)`` (single sequence). Mirrors the reference
        ``NemotronHAttention`` (GQA, NoPE, SDPA default scale, causal).
        """
        L = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(L, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(L, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(L, self.num_kv_heads, self.head_dim).transpose(0, 1)
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=0)
        v = v.repeat_interleave(n_rep, dim=0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (heads, L, hd)
        out = out.transpose(0, 1).reshape(L, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def prefill(self, hidden_states: torch.Tensor):
        """Cached-path prefill: same math as :meth:`offline_forward`, but also
        returns the per-token ``k`` / ``v`` (pre-GQA-expansion) to seed the cache.

        Returns ``(out (L, H), k (L, num_kv_heads, head_dim), v (...))``.
        """
        L = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(L, self.num_kv_heads, self.head_dim)
        n_rep = self.num_heads // self.num_kv_heads
        kh = k.transpose(0, 1).repeat_interleave(n_rep, dim=0)
        vh = v.transpose(0, 1).repeat_interleave(n_rep, dim=0)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), kh, vh, is_causal=True)
        out = out.transpose(0, 1).reshape(L, self.num_heads * self.head_dim)
        return self.o_proj(out), k.detach(), v.detach()

    def decode_forward(self, x: torch.Tensor, kv_cache_for_layer):
        """Single-token cached decode.

        ``x``: ``(1, H)`` — the new token. ``kv_cache_for_layer``: ``[k, v]``,
        each ``(Lh, num_kv_heads, head_dim)`` — the cached history. Appends the
        new token's k/v and attends over all cached keys (NoPE; no causal mask
        is needed since the query is the last position). Returns
        ``(out (1, H), (k_all, v_all))``.
        """
        k_cache, v_cache = kv_cache_for_layer
        q = self.q_proj(x).view(1, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(1, self.num_kv_heads, self.head_dim)
        v_new = self.v_proj(x).view(1, self.num_kv_heads, self.head_dim)
        k_all = torch.cat([k_cache, k_new], dim=0)  # (Lh+1, num_kv_heads, hd)
        v_all = torch.cat([v_cache, v_new], dim=0)
        n_rep = self.num_heads // self.num_kv_heads
        kh = k_all.transpose(0, 1).repeat_interleave(n_rep, dim=0)  # (heads, Lh+1, hd)
        vh = v_all.transpose(0, 1).repeat_interleave(n_rep, dim=0)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), kh, vh, is_causal=False)
        out = out.transpose(0, 1).reshape(1, self.num_heads * self.head_dim)
        return self.o_proj(out), (k_all.detach(), v_all.detach())


class NemotronHMLP(nn.Module):
    """Non-gated squared-ReLU MLP: ``down_proj(relu(up_proj(x))**2)``."""

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.square(F.relu(self.up_proj(hidden_states))))


class _RMSNormGated(nn.Module):
    """Mamba-2 gated RMSNorm (grouped): gate, then per-group RMS-normalize.

    Matches the reference ``MambaRMSNormGated`` with ``norm_before_gate=False``:
    ``x = x * silu(z)`` first, then RMS-normalize *within each group* of
    ``group_size`` channels (group_size = intermediate / n_groups), then scale by
    the per-channel weight. Computed in fp32.
    """

    def __init__(self, dim: int, eps: float, group_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.group_size = group_size
        self.n_groups = dim // group_size

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = y.dtype
        y = y.float() * F.silu(z.float())
        shape = y.shape
        y = y.reshape(*shape[:-1], self.n_groups, self.group_size)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y.reshape(shape)
        return (y * self.weight.float()).to(dtype)


class Mamba2Mixer(nn.Module):
    """Mamba-2 mixer whose recurrent state lives in the engine's pool.

    Param layout matches HF ``NemotronHMamba2Mixer``:
        in_proj: [z (d_inner) | xBC (conv_dim) | dt (nheads)]
        conv1d: depthwise over conv_dim channels, kernel=conv_kernel (causal)
        A_log, D, dt_bias per head; gated RMSNorm(d_inner); out_proj
    """

    def __init__(self, config: NanoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.nheads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.d_inner = config.d_inner
        self.d_state = config.ssm_state_size
        self.n_groups = config.n_groups
        self.conv_kernel = config.conv_kernel
        self.conv_dim = config.conv_dim

        in_proj_out = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.nheads
        self.in_proj = nn.Linear(config.hidden_size, in_proj_out, bias=config.mamba_proj_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            bias=config.use_conv_bias,
        )
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.norm = _RMSNormGated(
            self.d_inner, eps=config.rms_norm_eps, group_size=self.d_inner // self.n_groups
        )
        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=config.mamba_proj_bias)
        # Resolved at load by ``NodeSubmodule.bind_node_resources``; None on
        # the standalone (engine-free) paths.
        self.mix: Mamba2Callable | None = None

    def bind_resources(self, resources: dict) -> None:
        """Resolve the recurrent pool and the Mamba-2 resource this layer steps
        through. See ``NodeSubmodule.bind_node_resources``."""
        pool = resources.get(MAMBA_STATE)
        attn = resources.get(MAMBA)
        self.mix = Mamba2Callable(pool=pool, attn=attn) if pool is not None and attn is not None else None

    # -- helpers ---------------------------------------------------------

    def _split_in_proj(self, projected: torch.Tensor):
        z, xBC, dt = torch.split(
            projected, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )
        return z, xBC, dt

    def _ssd_scan(self, x, dt, A, B, C, h0=None):
        """Sequential SSD reference scan.

        Shapes (L = tokens):
            x:  (L, nheads, head_dim)   dt: (L, nheads)   A: (nheads,)
            B:  (L, ngroups, d_state)   C:  (L, ngroups, d_state)
            h0: (nheads, head_dim, d_state) or None
        Returns y (L, nheads, head_dim) and final state h (nheads, head_dim, d_state).
        """
        L = x.shape[0]
        heads_per_group = self.nheads // self.n_groups
        # expand groups -> heads (contiguous: head i uses group i // heads_per_group).
        # This matches the mamba_ssm kernel / reference decode path (the deployed
        # behavior). NOTE: the reference's *naive prefill* uses .repeat (tile,
        # i % n_groups) instead — an HF fallback quirk we intentionally do NOT copy.
        B = B.repeat_interleave(heads_per_group, dim=1)  # (L, nheads, d_state)
        C = C.repeat_interleave(heads_per_group, dim=1)
        dA = torch.exp(dt * A)  # (L, nheads)
        h = (
            h0
            if h0 is not None
            else x.new_zeros(self.nheads, self.head_dim, self.d_state)
        )
        ys = []
        for t in range(L):
            # dBx: (nheads, head_dim, d_state)
            dBx = (dt[t].unsqueeze(-1) * x[t]).unsqueeze(-1) * B[t].unsqueeze(1)
            h = dA[t].view(-1, 1, 1) * h + dBx
            y_t = torch.einsum("hpn,hn->hp", h, C[t])  # (nheads, head_dim)
            ys.append(y_t)
        y = torch.stack(ys, dim=0)  # (L, nheads, head_dim)
        y = y + self.D.view(1, -1, 1) * x
        return y, h

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Engine forward over the step's packed rows ``[total_tokens, H]``.

        The label and the layer index are cursors on ``self.mix`` (bound once
        per stack, advanced per Mamba layer by ``NemotronHLLM.forward``). The
        conv window and the SSM state are read from and written to this
        layer's pool blocks by the resource; decode rows take the fused
        single-step kernels, prefill rows the varlen conv + sequential scan.
        With no resources bound (standalone use) this is the single-sequence
        math of ``_forward_seq`` from a fresh state.
        """
        if self.mix is None:
            out, _, _ = self._forward_seq(hidden_states, None, None)
            return out
        n = hidden_states.shape[0]
        z, xBC, dt = self._split_in_proj(self.in_proj(hidden_states))
        # [conv_dim, 1, k] -> [conv_dim, k] for the kernel; silu applied inside
        xBC = self.mix.conv(xBC, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias)
        x, B, C = torch.split(
            xBC, [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1
        )
        y = self.mix(
            x.view(n, self.nheads, self.head_dim), dt,
            B.view(n, self.n_groups, self.d_state), C.view(n, self.n_groups, self.d_state),
            self.A_log, self.D, self.dt_bias,
        )
        y = self.norm(y.reshape(n, self.d_inner).to(hidden_states.dtype), z)
        return self.out_proj(y)

    def _forward_seq(self, hidden_states, prev_conv, prev_ssm):
        """One request's full-sequence Mamba over ``hidden_states`` [L, H] with its
        incoming ``prev_conv`` [conv_dim, k-1] / ``prev_ssm`` [nheads, head_dim,
        d_state] (or ``None`` to seed fresh). Returns ``(out [L,H], new_conv, new_ssm)``,
        the conv state left-zero-padded to a fixed [conv_dim, k-1] for decode."""
        L = hidden_states.shape[0]
        z, xBC, dt = self._split_in_proj(self.in_proj(hidden_states))

        xBC_t = xBC.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, L)
        if prev_conv is not None:
            xBC_t = torch.cat([prev_conv.unsqueeze(0), xBC_t], dim=-1)
        conv_out = self.conv1d(xBC_t)[..., : xBC_t.shape[-1]]
        conv_out = conv_out[..., -L:]  # keep the L new positions
        xBC = F.silu(conv_out.squeeze(0).transpose(0, 1))  # (L, conv_dim)
        # fixed [conv_dim, k-1] rolling buffer of raw inputs (left-pad when short)
        raw = xBC_t.squeeze(0)  # (conv_dim, seen)
        k1 = self.conv_kernel - 1
        if raw.shape[-1] >= k1:
            new_conv = raw[:, -k1:].contiguous()
        else:
            pad = raw.new_zeros(self.conv_dim, k1 - raw.shape[-1])
            new_conv = torch.cat([pad, raw], dim=-1)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        x = x.view(L, self.nheads, self.head_dim)
        B = B.view(L, self.n_groups, self.d_state)
        C = C.view(L, self.n_groups, self.d_state)

        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt = F.softplus(dt.float() + self.dt_bias.float())  # (L, nheads)

        y, new_ssm = self._ssd_scan(x.float(), dt, A, B.float(), C.float(), h0=prev_ssm)
        y = y.reshape(L, self.d_inner).to(hidden_states.dtype)
        y = self.norm(y, z)
        return self.out_proj(y), new_conv.detach(), new_ssm.detach()

    # -- cached O(T) decode path -----------------------------------------

    def prefill(self, hidden_states: torch.Tensor):
        """Cached-path prefill: numerically identical to ``forward`` with no
        prior state, but also returns the recurrent state to seed the cache.

        Returns ``(out (L, H), conv_state (conv_dim, conv_kernel-1), ssm_state)``.
        The conv state is the last ``conv_kernel-1`` *raw* xBC inputs, a FIXED
        length buffer left-zero-padded when ``L < conv_kernel-1``.
        """
        L = hidden_states.shape[0]
        z, xBC, dt = self._split_in_proj(self.in_proj(hidden_states))

        # causal depthwise conv over (x|B|C) — no prior conv state at prefill
        xBC_t = xBC.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, L)
        conv_out = self.conv1d(xBC_t)[..., :L]
        xBC_c = F.silu(conv_out.squeeze(0).transpose(0, 1))  # (L, conv_dim)

        # fixed-length rolling conv buffer of raw (pre-conv) xBC inputs
        k1 = self.conv_kernel - 1
        raw = xBC_t.squeeze(0)  # (conv_dim, L)
        if L >= k1:
            conv_state = raw[:, -k1:].contiguous()
        else:
            pad = raw.new_zeros(self.conv_dim, k1 - L)
            conv_state = torch.cat([pad, raw], dim=-1)

        x, B, C = torch.split(
            xBC_c,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        x = x.view(L, self.nheads, self.head_dim)
        B = B.view(L, self.n_groups, self.d_state)
        C = C.view(L, self.n_groups, self.d_state)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt.float() + self.dt_bias.float())

        y, new_ssm = self._ssd_scan(x.float(), dt, A, B.float(), C.float(), h0=None)
        y = y.reshape(L, self.d_inner).to(hidden_states.dtype)
        y = self.norm(y, z)
        return self.out_proj(y), conv_state.detach(), new_ssm.detach()

    def decode_step(self, hidden_state: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """Single-token Mamba recurrence from cached conv/ssm state.

        ``hidden_state``: ``(1, H)``. ``conv_state``: ``(conv_dim, conv_kernel-1)``
        rolling buffer of raw xBC inputs. ``ssm_state``: ``(nheads, head_dim,
        d_state)``. Returns ``(out (1, H), new_conv_state, new_ssm_state)``.

        The conv step is an explicit dot product over a fixed window rather than
        the padded ``conv1d`` used at prefill: window = [buffer(k-1) | new(1)],
        which reproduces the causal conv at the new position exactly, and the
        buffer rolls forward by dropping its oldest column.
        """
        z, xBC, dt = self._split_in_proj(self.in_proj(hidden_state))

        # rolling conv: append the new raw xBC, take the last conv_kernel window
        xBC_new = xBC.transpose(0, 1)  # (conv_dim, 1)
        window = torch.cat([conv_state, xBC_new], dim=-1)  # (conv_dim, conv_kernel)
        w = self.conv1d.weight[:, 0, :]  # (conv_dim, conv_kernel)
        conv_out = (window * w).sum(-1)  # (conv_dim,)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias
        new_conv_state = window[:, 1:].contiguous()  # (conv_dim, conv_kernel-1)
        xBC_c = F.silu(conv_out).unsqueeze(0)  # (1, conv_dim)

        x, B, C = torch.split(
            xBC_c,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        x = x.view(1, self.nheads, self.head_dim)
        B = B.view(1, self.n_groups, self.d_state)
        C = C.view(1, self.n_groups, self.d_state)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt.float() + self.dt_bias.float())

        y, new_ssm = self._ssd_scan(x.float(), dt, A, B.float(), C.float(), h0=ssm_state)
        y = y.reshape(1, self.d_inner).to(hidden_state.dtype)
        y = self.norm(y, z)
        return self.out_proj(y), new_conv_state.detach(), new_ssm.detach()


# ---------------------------------------------------------------------------
# Block + model
# ---------------------------------------------------------------------------


def _build_mixer(config: NanoConfig, kind: str, layer_idx: int) -> nn.Module:
    if kind == "mamba":
        return Mamba2Mixer(config, layer_idx)
    if kind == "attention":
        return NemotronHAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
            qk_norm=False,
            rms_norm_eps=config.rms_norm_eps,
            attn_key=NANO_ATTN,
            kv_key=NANO_KV,
            pos_key=None,  # NoPE
        )
    if kind == "mlp":
        return NemotronHMLP(config)
    raise ValueError(f"Unknown layer kind: {kind!r}")


class NemotronHBlock(nn.Module):
    """Pre-norm residual block: ``h + mixer(norm(h))`` (single norm per block)."""

    def __init__(self, config: NanoConfig, kind: str, layer_idx: int):
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = _build_mixer(config, kind, layer_idx)

    def forward(self, hidden_states):
        # Every mixer reads its resources through bound cursors (attention:
        # ``attend``; Mamba-2: ``mix``); the MLP is stateless.
        return hidden_states + self.mixer(self.norm(hidden_states))

    def offline_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Engine-free block for offline inference; ``hidden_states`` is ``(L, H)``."""
        normed = _rms_fp32(hidden_states, self.norm.weight, self.norm.variance_epsilon)
        if self.kind == "attention":
            out = self.mixer.offline_forward(normed)
        else:  # mamba (full-seq scan) / mlp — both run engine-free with no state
            out = self.mixer(normed)
        return hidden_states + out

    def prefill(self, hidden_states: torch.Tensor, cache: NemotronHCache) -> torch.Tensor:
        """Cached-path prefill for one block; populates ``cache`` in place.

        Same residual/norm math as :meth:`offline_forward`; differs only in that
        stateful mixers hand back their state to seed the cache.
        """
        normed = _rms_fp32(hidden_states, self.norm.weight, self.norm.variance_epsilon)
        if self.kind == "attention":
            out, k, v = self.mixer.prefill(normed)
            cache.attn[self.layer_idx] = [k, v]
        elif self.kind == "mamba":
            out, conv_state, ssm_state = self.mixer.prefill(normed)
            cache.conv[self.layer_idx] = conv_state
            cache.ssm[self.layer_idx] = ssm_state
        else:  # mlp — stateless
            out = self.mixer(normed)
        return hidden_states + out

    def decode_step(self, hidden_state: torch.Tensor, cache: NemotronHCache) -> torch.Tensor:
        """Cached-path single-token decode for one block; mutates ``cache``."""
        normed = _rms_fp32(hidden_state, self.norm.weight, self.norm.variance_epsilon)
        if self.kind == "attention":
            out, new_kv = self.mixer.decode_forward(normed, cache.attn[self.layer_idx])
            cache.attn[self.layer_idx] = [new_kv[0], new_kv[1]]
        elif self.kind == "mamba":
            out, new_conv, new_ssm = self.mixer.decode_step(
                normed, cache.conv[self.layer_idx], cache.ssm[self.layer_idx]
            )
            cache.conv[self.layer_idx] = new_conv
            cache.ssm[self.layer_idx] = new_ssm
        else:  # mlp — stateless
            out = self.mixer(normed)
        return hidden_state + out


class NemotronHLLM(nn.Module):
    """The ``llm`` container: hybrid layer stack + final norm (no embeddings)."""

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        kinds = config.layer_types
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, kinds[i], i) for i in range(config.num_hidden_layers)]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # attention layer -> dense KV-cache index (only ``*`` layers have a
        # KV slot); mamba layer -> dense recurrent-pool index (only ``M``
        # layers have state). Each pool is sized by its own layer count.
        self._attn_cache_idx: dict[int, int] = {}
        self._mamba_idx: dict[int, int] = {}
        for i, kind in enumerate(kinds):
            if kind == "attention":
                self._attn_cache_idx[i] = len(self._attn_cache_idx)
            elif kind == "mamba":
                self._mamba_idx[i] = len(self._mamba_idx)

    def forward(self, input_embeds, *, label: str = "main"):
        """Engine forward over packed rows ``[sum(seq_lens), H]``.

        The label and the per-kind dense layer indices are cursors on the
        shared resources: bound once per step, advanced per layer (see
        ``Attention.forward`` / ``Mamba2Mixer.forward``). The runner commits
        the cache and state advance from the step declaration, so nothing is
        advanced here.
        """
        hidden = input_embeds
        first_attn = next((b for b in self.layers if b.kind == "attention"), None)
        if first_attn is not None:
            first_attn.mixer.attend.bind_step(label)
        first_mamba = next((b for b in self.layers if b.kind == "mamba" and b.mixer.mix is not None), None)
        if first_mamba is not None:
            first_mamba.mixer.mix.bind_step(label)
        for block in self.layers:
            if block.kind == "attention":
                block.mixer.attend.set_layer_idx(self._attn_cache_idx[block.layer_idx])
            elif block.kind == "mamba" and block.mixer.mix is not None:
                block.mixer.mix.set_layer_idx(self._mamba_idx[block.layer_idx])
            hidden = block(hidden)
        return self.norm_f(hidden)

    def offline_forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Engine-free full-sequence forward; ``input_embeds`` is ``(L, H)``."""
        hidden = input_embeds
        for block in self.layers:
            hidden = block.offline_forward(hidden)
        return _rms_fp32(hidden, self.norm_f.weight, self.norm_f.variance_epsilon)

    def prefill(self, input_embeds: torch.Tensor):
        """Cached O(T) prefill. ``input_embeds`` ``(L, H)`` -> ``(hidden (L, H),
        cache)``. Runs the full-sequence offline math while populating a fresh
        :class:`NemotronHCache` for subsequent single-token decode.
        """
        cache = NemotronHCache()
        hidden = input_embeds
        for block in self.layers:
            hidden = block.prefill(hidden, cache)
        hidden = _rms_fp32(hidden, self.norm_f.weight, self.norm_f.variance_epsilon)
        return hidden, cache

    def decode_step(self, input_embed: torch.Tensor, cache: NemotronHCache) -> torch.Tensor:
        """Cached O(T) decode of one new token. ``input_embed`` ``(1, H)`` ->
        ``hidden (1, H)``; mutates ``cache`` in place (appends attention k/v,
        rolls conv buffers, advances ssm state).
        """
        hidden = input_embed
        for block in self.layers:
            hidden = block.decode_step(hidden, cache)
        return _rms_fp32(hidden, self.norm_f.weight, self.norm_f.variance_epsilon)


class NemotronHForCausalLM(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.llm = NemotronHLLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Function-calling head (custom_outputs: function_tokens/function_logits).
        if config.has_function_head:
            self.function_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_embeds, *, label: str = "main") -> torch.Tensor:
        return self.llm(input_embeds, label=label)

    def offline_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence offline forward: ``input_ids`` (L,) -> logits (L, vocab)."""
        hidden = self.embed_tokens(input_ids)
        hidden = self.llm.offline_forward(hidden)
        return self.lm_head(hidden)

    def prefill(self, input_embeds: torch.Tensor):
        """Cached O(T) prefill over embeddings. ``input_embeds`` ``(L, H)`` ->
        ``(hidden (L, H), cache)`` (post final-norm hidden; apply ``lm_head`` for
        logits). Additive to :meth:`offline_forward` — leaves it unchanged.
        """
        return self.llm.prefill(input_embeds)

    def decode_step(self, input_embed: torch.Tensor, cache: NemotronHCache) -> torch.Tensor:
        """Cached O(T) decode of one token. ``input_embed`` ``(1, H)`` ->
        ``hidden (1, H)``; mutates ``cache`` in place.
        """
        return self.llm.decode_step(input_embed, cache)

    @staticmethod
    def remap(name: str) -> str | None:
        """Map a composite-checkpoint key to a nano param path, or ``None`` to skip.

        The checkpoint bundles every stage under ``stt_model.`` / ``tts_model.``;
        keep only the LLM tensors (embed_tokens / lm_head / function_head / llm.*)
        and drop the rest (perception, rnnt, tts) so the stream loads cleanly
        into this module alone.
        """
        if not name.startswith("stt_model."):
            return None
        rest = name[len("stt_model."):]
        if rest.startswith(("embed_tokens.", "lm_head.", "function_head.", "llm.")):
            return rest
        return None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)
