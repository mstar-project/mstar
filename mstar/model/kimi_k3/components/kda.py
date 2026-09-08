"""Kimi Delta Attention layer (spec C), tensor-parallel over heads, with pluggable
kernels and two execution contexts:

* **paged**: the layer is bound to a ``RecurrentStateManager`` (``state_key``) and reads
  this step's addressing (``slot_ids``, ``has_state``, ``cu_seqlens``) from the resource;
  conv and recurrent states live in the resource's per-layer slot tensors.
* **dense** (tests / eager reference): ``forward_dense(x, state)`` carries an explicit
  ``KDAState`` for one sequence.

Parameter names mirror the checkpoint (``q_proj``, ``q_conv1d``, ``f_a_proj``, ...) except
the three input projections, which are packed into ``qkv_proj`` (loaded through stacked
rules ``q_proj -> 0, k_proj -> 1, v_proj -> 2``).

State layout in the resource (per layer, per slot):
``conv``: ``[3 * P_local, W]`` = the last ``W`` pre-activation inputs of q | k | v;
``recurrent``: ``[H_local, D, D]`` fp32, **V-first** (``S[v, k]``), the kernels' layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.kimi_k3.components.common import (
    ReplicatedLinear,
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


@dataclass
class KDAParams:
    """What a kernel needs besides activations: per-rank parameters and constants."""
    conv_weight: torch.Tensor  # [3*P_local, W] (q | k | v)
    A_log: torch.Tensor  # [H_local] fp32
    dt_bias: torch.Tensor  # [P_local] fp32
    lower_bound: float | None
    num_heads: int
    head_dim: int
    scale: float


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
        ``conv_state [slots, 3P, W]`` and ``rec_state [slots, H, D, D]`` (V-first) are the
        resource's layer views, updated in place. Returns ``o [T, H, D]`` in ``qkv.dtype``."""
        t = qkv.shape[0]
        h, d = p.num_heads, p.head_dim
        assert cu_seqlens[-1] == t, f"plan covers {cu_seqlens[-1]} tokens, forward got {t}: stale plan?"
        out = torch.empty(t, h, d, dtype=qkv.dtype, device=qkv.device)
        for i in range(len(slot_ids)):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            if e <= s:
                continue
            slot = slot_ids[i]
            conv_prev = conv_state[slot] if has_state[i] else None
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
            conv_state[slot].copy_(conv_new.to(conv_state.dtype))
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
        self.state = None
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
        self.f_a_proj = ReplicatedLinear(hidden_size, head_dim)
        self.f_b_proj = ColumnParallelLinear(comm_group, head_dim, p_full, bias=False)
        self.b_proj = ColumnParallelLinear(comm_group, hidden_size, num_heads, bias=False)
        self.g_proj = ColumnParallelLinear(comm_group, hidden_size, p_full, bias=False)
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
        self.f_a_proj.weight.weight_loader = replicated_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_loaders()
        restore_kept_dtypes(self)
        return result

    def set_kernels(self, kernels) -> None:
        self.kernels = kernels

    # ------------------------------------------------------------------ pieces
    def bind_resources(self, resources: dict) -> None:
        self.state = resources.get(self._state_key)

    def params(self) -> KDAParams:
        conv_w = torch.cat(
            [self.q_conv1d.weight[:, 0], self.k_conv1d.weight[:, 0], self.v_conv1d.weight[:, 0]], dim=0,
        )
        return KDAParams(
            conv_weight=conv_w, A_log=self.A_log, dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound, num_heads=self.num_heads, head_dim=self.head_dim,
            scale=self.head_dim ** -0.5,
        )

    def _project(self, x: torch.Tensor):
        t = x.shape[0]
        qkv = self.qkv_proj(x)  # [T, 3 P_local]
        g_raw = self.f_b_proj(self.f_a_proj(x)).view(t, self.num_heads, self.head_dim)
        beta_raw = self.b_proj(x)  # [T, H_local]
        return qkv, g_raw, beta_raw

    def _finish(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        g_out = self.g_proj(x).view(t, self.num_heads, self.head_dim)
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
        """Paged path: addressing and state come from the bound resource (layer cursor
        set by the model loop)."""
        assert self.state is not None, "bind_resources first, or use forward_dense"
        plan = self.state.plan_output
        qkv, g_raw, beta_raw = self._project(x)
        o = self.kernels.run_paged(
            qkv, g_raw, beta_raw, plan,
            self.state.layer_view("conv"), self.state.layer_view("recurrent"), self.params(),
        )
        return self._finish(x, o)

    def forward_dense(self, x: torch.Tensor, state: KDAState | None = None) -> tuple[torch.Tensor, KDAState]:
        """One sequence with explicit state (K-first recurrent, like the reference)."""
        state = state or KDAState()
        p = self.params()
        t = x.shape[0]
        qkv, g_raw, beta_raw = self._project(x)
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
        return self._finish(x, o.to(x.dtype)), KDAState(conv_q=cq, conv_k=ck, conv_v=cv, recurrent=rec)


__all__ = ["KDA_STACKED_PARAMS", "KDAParams", "ParallelKDAAttention", "TorchKDAKernels", "F"]
