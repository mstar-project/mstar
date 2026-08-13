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
    ``_apply_rope`` a no-op.
  * Mamba state (Option A) — the engine has no SSM-state pool yet, so conv/ssm
    state lives in the submodule's per-request ``PerRequestState`` and is
    threaded here via ``MambaStateAccessor``. Eager-only; the Option-B pool
    (batching + CUDA graphs) is the Phase-10 follow-up.

The Mamba-2 forward is a pure-PyTorch reference scan (no ``mamba_ssm`` kernel
dependency) — correct but not fast; the kernel fast path is a later swap.
TODO(verify): numeric parity of the scan against the HF NemotronHMamba2Mixer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.attention import Attention
from mstar.model.components.norm import RMSNorm
from mstar.model.nemotron_duplex.config import NanoConfig

# ---------------------------------------------------------------------------
# Per-request Mamba state plumbing (Option A: submodule-owned state)
# ---------------------------------------------------------------------------


@dataclass
class MambaStateAccessor:
    """Read/write view over one batch's Mamba conv+ssm state.

    Built by the submodule from ``engine_inputs.per_request_states`` and the
    current walk. Correctness draft is bs=1 (one request); kept indexable so the
    batched Option-B pool can drop in later. State tensors live in each request's
    ``PerRequestState.tensors`` under ``mamba{layer}.conv`` / ``.ssm``.
    """

    request_states: dict
    request_ids: list
    is_prefill: bool

    def _keys(self, layer_idx: int):
        return f"mamba{layer_idx}.conv", f"mamba{layer_idx}.ssm"

    def read(self, layer_idx: int):
        if self.is_prefill or not self.request_ids or self.request_states is None:
            return None, None
        st = self.request_states.get(self.request_ids[0])
        if st is None:
            return None, None
        ck, sk = self._keys(layer_idx)
        return st.get(ck), st.get(sk)

    def write(self, layer_idx: int, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        if not self.request_ids or self.request_states is None:
            return
        st = self.request_states.get(self.request_ids[0])
        if st is None:
            return
        ck, sk = self._keys(layer_idx)
        st.add(ck, conv_state.detach())
        st.add(sk, ssm_state.detach())


# ---------------------------------------------------------------------------
# Mixers
# ---------------------------------------------------------------------------


class NemotronHAttention(Attention):
    """GQA self-attention with NoPE (Nemotron-H applies no positional encoding)."""

    def _apply_rope(self, q, k, cache_handle):  # noqa: ARG002 - override to disable RoPE
        return q, k


class NemotronHMLP(nn.Module):
    """Non-gated squared-ReLU MLP: ``down_proj(relu(up_proj(x))**2)``."""

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor, cache_handle=None, mamba_state=None) -> torch.Tensor:  # noqa: ARG002
        return self.down_proj(torch.square(F.relu(self.up_proj(hidden_states))))


class _RMSNormGated(nn.Module):
    """Mamba-2 gated RMSNorm: ``rmsnorm(y * silu(z)) * weight``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = y * F.silu(z)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight


class Mamba2Mixer(nn.Module):
    """Mamba-2 mixer with M*-owned per-request recurrent state.

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
        self.time_step_limit = (0.0, float("inf"))

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
        self.norm = _RMSNormGated(self.d_inner, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=config.mamba_proj_bias)

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
        # expand groups -> heads
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager | None = None,  # noqa: ARG002
        mamba_state: MambaStateAccessor | None = None,
    ) -> torch.Tensor:
        L = hidden_states.shape[0]
        z, xBC, dt = self._split_in_proj(self.in_proj(hidden_states))

        prev_conv, prev_ssm = (None, None)
        if mamba_state is not None:
            prev_conv, prev_ssm = mamba_state.read(self.layer_idx)

        # --- causal depthwise conv over (x|B|C) ---
        xBC_t = xBC.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, L)
        if prev_conv is not None:
            xBC_t = torch.cat([prev_conv, xBC_t], dim=-1)
        conv_out = self.conv1d(xBC_t)[..., : xBC_t.shape[-1]]
        conv_out = conv_out[..., -L:]  # keep the L new positions
        xBC = F.silu(conv_out.squeeze(0).transpose(0, 1))  # (L, conv_dim)
        # conv state = last (conv_kernel-1) inputs, for the next step
        new_conv_state = xBC_t[..., -(self.conv_kernel - 1):].detach()

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

        if mamba_state is not None:
            mamba_state.write(self.layer_idx, new_conv_state, new_ssm)

        y = self.norm(y, z)
        return self.out_proj(y)


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

    def forward(self, hidden_states, cache_handle, mamba_state):
        residual = hidden_states
        normed = self.norm(hidden_states)
        out = self.mixer(normed, cache_handle=cache_handle, mamba_state=mamba_state)
        return residual + out


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

        # attention layer -> dense KV-cache index (only ``*`` layers have a slot)
        self._attn_cache_idx: dict[int, int] = {}
        a = 0
        for i, kind in enumerate(kinds):
            if kind == "attention":
                self._attn_cache_idx[i] = a
                a += 1

    def forward(self, input_embeds, cache_handle, mamba_state=None):
        hidden = input_embeds
        for block in self.layers:
            if block.kind == "attention":
                cache_handle.set_layer_idx(self._attn_cache_idx[block.layer_idx])
            hidden = block(hidden, cache_handle=cache_handle, mamba_state=mamba_state)
        cache_handle.advance_seq_lens()
        return self.norm_f(hidden)


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

    def forward(self, input_embeds, cache_handle, mamba_state=None) -> torch.Tensor:
        return self.llm(input_embeds, cache_handle=cache_handle, mamba_state=mamba_state)

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
