"""Kimi Delta Attention layer (spec C), tensor-parallel over heads, with pluggable
kernels and two execution contexts:

* **paged**: the layer is bound to a ``RecurrentStatePool`` (``state_key``: the per-layer slot
  blocks) and to the ``KDAManager`` planned against it (``attn_key``: this step's ``KDAPlan`` with
  ``slot_ids``, ``has_state``, ``cu_seqlens``); ``attn.run`` hands the layer's blocks to the kernels.
* **dense** (tests / eager reference): ``forward_dense(x, state)`` carries an explicit
  ``KDAState`` for one sequence.

Parameter names mirror the checkpoint (``q_proj``, ``q_conv1d``, ``f_b_proj``, ...) except
the projections of the layer input, which are packed into two GEMMs: ``qkv_proj`` (stacked
rules ``q_proj -> 0, k_proj -> 1, v_proj -> 2``) and ``in_proj`` = ``g_proj | b_proj | f_a_proj``
(``KDA_IN_PROJ_PARAMS``; the output gate and beta are column-parallel, the low-rank decay
factor ``f_a`` is replicated). One 24 MB GEMM instead of three launches of 22, 0.2 and 1.8 MB
that each paid the same fixed ramp (see ``MergedParallelLinear``).

State layout in the pool (``DeltaNetGeometry``, per layer, per slot):
``conv``: ``[3 * P_local, W - 1]`` = the pre-activation inputs of q | k | v before the current token;
``state``: ``[H_local, D, D]`` fp32, **V-first** (``S[v, k]``), the kernels' layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.linear_attn.kda_kernels import KDAParams
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.distributed.merged_linear import COLUMN, REPLICATED, MergedParallelLinear
from mstar.model.kimi_k3.components.common import (
    attach_dim0_loader,
    replicated_loader,
    restore_kept_dtypes,
)
from mstar.model.kimi_k3.reference.kda import (
    KDAState,
    from_v_first,
    gated_rms_norm,
    kda_gate,
    kda_recurrent,
    l2norm,
    short_conv,
    to_v_first,
)

KDA_STACKED_PARAMS = [(".qkv_proj", ".q_proj", 0), (".qkv_proj", ".k_proj", 1), (".qkv_proj", ".v_proj", 2)]
# the checkpoint's g_proj / b_proj / f_a_proj land in the merged in_proj by segment name
KDA_IN_PROJ_PARAMS = [(".in_proj", ".g_proj", "g"), (".in_proj", ".b_proj", "b"), (".in_proj", ".f_a_proj", "f_a")]


class TorchKDAKernels:
    """Reference kernels: per-sequence Python loop over the fp32 recurrence. Correct on
    any device, but it addresses slots with host integers, so it must not be captured in
    a CUDA graph (``cuda_graph_safe = False``); ``FLAKDAKernels`` is the GPU path."""

    cuda_graph_safe = False

    @torch.compiler.disable
    def run_paged(self, qkv, g_raw, beta_raw, plan, conv_state, rec_state, p: KDAParams) -> torch.Tensor:
        return self.run_lists(
            qkv, g_raw, beta_raw, plan.cu_seqlens_cpu, plan.slot_ids_cpu, plan.has_state_cpu,
            conv_state, rec_state, p,
        )

    @torch.compiler.disable
    def run_lists(
        self, qkv: torch.Tensor, g_raw: torch.Tensor, beta_raw: torch.Tensor,
        cu_seqlens: list[int], slot_ids: list[int], has_state: list[bool],
        conv_state: torch.Tensor, rec_state: torch.Tensor, p: KDAParams,
    ) -> torch.Tensor:
        """``qkv [T, 3P]`` pre-conv projections, ``g_raw [T, H, D]``, ``beta_raw [T, H]``;
        ``conv_state [slots, 3P, W - 1]`` and ``rec_state [slots, H, D, D]`` (V-first) are the
        pool's layer blocks, updated in place. Returns ``o [T, H, D]`` in ``qkv.dtype``. The
        reference conv keeps a ``W``-wide window whose oldest column is never read, so the
        block's ``W - 1`` columns are padded in front and the new window's tail is kept."""
        t = qkv.shape[0]
        h, d = p.num_heads, p.head_dim
        assert cu_seqlens[-1] == t, f"plan covers {cu_seqlens[-1]} tokens, forward got {t}: stale plan?"
        out = torch.empty(t, h, d, dtype=qkv.dtype, device=qkv.device)
        for i in range(len(slot_ids)):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            if e <= s:
                continue
            slot = slot_ids[i]
            conv_prev = None
            if has_state[i]:
                kept = conv_state[slot]
                conv_prev = torch.cat([kept.new_zeros(kept.shape[0], 1), kept], dim=-1)
            rec_prev = from_v_first(rec_state[slot]) if has_state[i] else None
            x = qkv[s:e]
            y, conv_new = short_conv(x, p.conv_weight, conv_prev)
            q, k, v = y.split([h * d, h * d, h * d], dim=-1)
            g_log = kda_gate(g_raw[s:e], p.A_log, p.dt_bias, p.lower_bound)
            o, rec_new = kda_recurrent(
                l2norm(q.reshape(-1, h, d)), l2norm(k.reshape(-1, h, d)), v.reshape(-1, h, d),
                g_log, torch.sigmoid(beta_raw[s:e].float()), rec_prev, p.scale,
            )
            out[s:e] = o.to(out.dtype)
            conv_state[slot].copy_(conv_new[:, 1:].to(conv_state.dtype))
            rec_state[slot].copy_(to_v_first(rec_new))
        return out


class ParallelKDAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        gate_lower_bound: float | None = -5.0,
        norm_eps: float = 1e-5,
        comm_group: CommGroup | None = None,
        state_key: str = "kda_state",
        attn_key: str = "kda_attn",
        kernels=None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp_rank, tp_size = comm_group.rank, comm_group.world_size
        assert num_heads % tp_size == 0, (num_heads, tp_size)
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.conv_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound
        self.norm_eps = norm_eps
        self._state_key = state_key
        self._attn_key = attn_key
        self.pool = None  # RecurrentStatePool
        self.attn = None  # KDAManager
        # kernels for the dense/reference paths; the paged path runs the manager's
        self.kernels = kernels or TorchKDAKernels()

        p_full = num_heads * head_dim
        p_local = self.num_heads * head_dim
        self.projection_size = p_local
        self.qkv_proj = MergedColumnParallelLinear(
            comm_group=comm_group, input_size=hidden_size, output_sizes=[p_full, p_full, p_full],
            bias=False, gather_output=False,
        )
        # depthwise causal conv weights [P_local, 1, W], one per projection (checkpoint names)
        self.q_conv1d = nn.Module()
        self.q_conv1d.weight = nn.Parameter(torch.empty(p_local, 1, conv_kernel_size))
        self.k_conv1d = nn.Module()
        self.k_conv1d.weight = nn.Parameter(torch.empty(p_local, 1, conv_kernel_size))
        self.v_conv1d = nn.Module()
        self.v_conv1d.weight = nn.Parameter(torch.empty(p_local, 1, conv_kernel_size))
        # output gate g, beta and the decay factor f_a all project the layer input: one GEMM
        self.in_proj = MergedParallelLinear(
            comm_group, hidden_size,
            [("g", p_full, COLUMN), ("b", num_heads, COLUMN), ("f_a", head_dim, REPLICATED)],
        )
        self.f_b_proj = ColumnParallelLinear(comm_group, head_dim, p_full, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(p_local, dtype=torch.float32))
        self.o_norm = nn.Module()
        self.o_norm.weight = nn.Parameter(torch.ones(head_dim))
        self.o_proj = RowParallelLinear(
            comm_group, p_full, hidden_size, bias=False, input_is_parallel=True, reduce_results=True,
        )
        self._tp = (tp_rank, tp_size)
        self._attach_loaders()

    def _attach_loaders(self) -> None:
        tp_rank, tp_size = self._tp
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            attach_dim0_loader(conv.weight, tp_rank, tp_size)
        attach_dim0_loader(self.A_log, tp_rank, tp_size)
        attach_dim0_loader(self.dt_bias, tp_rank, tp_size)
        self.A_log._keep_dtype = torch.float32
        self.dt_bias._keep_dtype = torch.float32
        self.o_norm.weight.weight_loader = replicated_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_loaders()
        restore_kept_dtypes(self)
        self._params_cache = None
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self._params_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def set_kernels(self, kernels) -> None:
        self.kernels = kernels

    # ------------------------------------------------------------------ pieces
    def bind_resources(self, resources: dict) -> None:
        """Resolve the pool and the manager planned against it. Off-GPU the manager has no
        kernels of its own (fla and FlashKDA are CUDA); it gets the torch reference."""
        self.pool = resources.get(self._state_key)
        self.attn = resources.get(self._attn_key)
        if self.attn is not None and getattr(self.attn, "kernels", None) is None:
            self.attn.set_kernels(TorchKDAKernels())

    def params(self) -> KDAParams:
        """The kernels' parameter bundle; the q/k/v conv weights concatenated into one
        ``[3P, W]`` tensor. Cached (keyed by the weights' storage and version, cleared on
        ``_apply``/reload): rebuilding the concatenation was one launch per layer per step."""
        ws = (self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight)
        key = tuple((w.data_ptr(), w._version) for w in ws)
        cached = getattr(self, "_params_cache", None)
        if cached is None or cached[0] != key:
            conv_w = torch.cat([w[:, 0] for w in ws], dim=0)
            cached = (key, KDAParams(
                conv_weight=conv_w, A_log=self.A_log, dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound, num_heads=self.num_heads, head_dim=self.head_dim,
                scale=self.head_dim ** -0.5,
            ))
            self._params_cache = cached
        return cached[1]

    def _project(self, x: torch.Tensor):
        """``(qkv [T, 3 P_local], g_raw [T, H, D], beta_raw [T, H], g_out [T, H, D])``: two GEMMs on
        ``x`` plus the tiny ``f_b`` factor. The merged segments are views; the kernels need
        contiguous ``beta`` and ``g`` (a no-op for one row, a small copy otherwise)."""
        t = x.shape[0]
        qkv = self.qkv_proj(x)
        mixed = self.in_proj.project(x)
        g_raw = self.f_b_proj(mixed["f_a"]).view(t, self.num_heads, self.head_dim)
        beta_raw = mixed["b"].contiguous()
        g_out = mixed["g"].contiguous().view(t, self.num_heads, self.head_dim)
        return qkv, g_raw, beta_raw, g_out

    def _finish(self, g_out: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        t = o.shape[0]
        y = self._gated_norm(o, g_out)
        return self.o_proj(y.reshape(t, self.num_heads * self.head_dim))

    def _gated_norm(self, o: torch.Tensor, g_out: torch.Tensor) -> torch.Tensor:
        """``RMSNorm_headdim(o) * weight * sigmoid(g)``: fla's fused Triton kernel on CUDA
        (fp32 math, same operation order as the reference), the torch reference elsewhere."""
        if o.is_cuda and o.dtype in (torch.bfloat16, torch.float16):
            try:
                from fla.modules.fused_norm_gate import rms_norm_gated
            except ImportError:
                rms_norm_gated = None
            if rms_norm_gated is not None:
                d = self.head_dim
                y = rms_norm_gated(
                    o.reshape(-1, d).contiguous(), g_out.reshape(-1, d).to(o.dtype).contiguous(),
                    self.o_norm.weight, None, activation="sigmoid", eps=self.norm_eps,
                )
                return y.view_as(o)
        return gated_rms_norm(o, g_out, self.o_norm.weight, self.norm_eps)

    # ------------------------------------------------------------------ paths
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paged path: the plan comes from the manager, the state blocks from the pool (the
        model loop sets the manager's layer cursor to this layer's index among the KDA layers)."""
        assert self.attn is not None and self.pool is not None, "bind_resources first, or use forward_dense"
        layer = self.attn.default_layer_idx
        assert layer is not None, "set the KDA layer cursor (attn.set_default_layer_idx) before the forward"
        qkv, g_raw, beta_raw, g_out = self._project(x)
        o = self.attn.run(
            qkv, g_raw, beta_raw, self.pool.block("conv", layer), self.pool.block("state", layer), self.params(),
        )
        return self._finish(g_out, o)

    def forward_dense(self, x: torch.Tensor, state: KDAState | None = None) -> tuple[torch.Tensor, KDAState]:
        """One sequence with explicit state (K-first recurrent, like the reference)."""
        state = state or KDAState()
        p = self.params()
        t = x.shape[0]
        qkv, g_raw, beta_raw, g_out = self._project(x)
        conv_prev = None
        if state.conv_q is not None:
            conv_prev = torch.cat([state.conv_q, state.conv_k, state.conv_v], dim=0)
        y, conv_new = short_conv(qkv, p.conv_weight, conv_prev)
        pl = self.projection_size
        q, k, v = y.split([pl, pl, pl], dim=-1)
        g_log = kda_gate(g_raw, p.A_log, p.dt_bias, p.lower_bound)
        o, rec = kda_recurrent(
            l2norm(q.reshape(t, self.num_heads, self.head_dim)),
            l2norm(k.reshape(t, self.num_heads, self.head_dim)),
            v.reshape(t, self.num_heads, self.head_dim),
            g_log, torch.sigmoid(beta_raw.float()), state.recurrent, p.scale,
        )
        cq, ck, cv = conv_new.split([pl, pl, pl], dim=0)
        return self._finish(g_out, o.to(x.dtype)), KDAState(conv_q=cq, conv_k=ck, conv_v=cv, recurrent=rec)


__all__ = ["KDA_IN_PROJ_PARAMS", "KDA_STACKED_PARAMS", "KDAParams", "ParallelKDAAttention", "TorchKDAKernels", "F"]
