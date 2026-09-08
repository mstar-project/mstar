"""GPU kernels for the KDA layer, slot-indexed and CUDA-graph safe.

``FLAKDAKernels`` runs the layer's inner math with flash-linear-attention's Triton kernels:

* prefill (packed varlen): ``causal_conv1d`` with per-sequence initial conv states and
  ``chunk_kda`` with per-sequence initial recurrent states, both gathered from the resource
  slots by a *device* index tensor and scattered back with ``index_copy_``;
* decode (one token per row): ``causal_conv1d_update`` on the gathered conv windows and
  ``fused_recurrent_kda_fwd`` with ``ssm_state_indices`` writing the recurrent slots in place.

Every address the kernels touch comes from tensors (``slot_ids``, ``cu_seqlens``; ``has_state``
masks the prefill's initial states), so a captured decode graph replays correctly with new slot ids. The
recurrent state layout is V-first (``[slots, H, V, K]``), the kernels' native layout.
"""
from __future__ import annotations

import torch

from mstar.engine.resources.recurrent.config import RecurrentPlanOutput


class FLAKDAKernels:
    cuda_graph_safe = True

    def __init__(self):
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from fla.modules.conv.triton.ops import causal_conv1d_update
        from fla.ops.kda import chunk_kda
        from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

        self._conv = causal_conv1d
        self._conv_update = causal_conv1d_update
        self._chunk = chunk_kda
        self._recurrent = fused_recurrent_kda_fwd

    @torch.compiler.disable
    def run_paged(self, qkv, g_raw, beta_raw, plan: RecurrentPlanOutput, conv_state, rec_state, p) -> torch.Tensor:
        """``qkv [T, 3P]`` (pre-conv), ``g_raw [T, H, D]``, ``beta_raw [T, H]``; the resource's
        layer views ``conv_state [slots, 3P, W]`` and ``rec_state [slots, H, D, D]`` are
        updated in place. Returns ``o [T, H, D]``."""
        h, d = p.num_heads, p.head_dim
        rows = plan.num_rows
        slot_ids = plan.slot_ids[:rows].to(torch.long)
        has_state = plan.has_state[:rows]
        t = qkv.shape[0]
        conv_w = p.conv_weight.to(qkv.dtype)  # [3P, W]
        if plan.is_decode:
            # decode reads the slots as they are: the resource zeroes a slot when it hands it
            # out (and the scratch slot every padded step), so a row's first decode after its
            # prefill finds the written state and a fresh slot holds zeros -- no masking, no
            # gather/scatter of the recurrent states (tens of MB per layer at 64 rows)
            cache = conv_state.index_select(0, slot_ids)
            y, cache = self._conv_update(qkv.view(rows, 1, -1), cache, weight=conv_w, activation="silu")
            conv_state.index_copy_(0, slot_ids, cache.to(conv_state.dtype))
            # the low-level Triton kernel assumes contiguous [B, T, H, K]: a strided split of
            # the fused conv output reads the wrong memory for every row after the first, so
            # re-layout once to [3, rows, P] (one copy) and take contiguous leading slices
            y3 = y.view(rows, 3, h * d).transpose(0, 1).contiguous()
            q, k, v = y3[0], y3[1], y3[2]
            # every row is its own one-token sequence (cu_seqlens); without it the kernel
            # would chain the rows as one sequence and carry row i's state into row i+1
            o = self._recurrent(
                q=q.view(1, rows, h, d), k=k.view(1, rows, h, d), v=v.view(1, rows, h, d),
                g=g_raw.view(1, rows, h, d), beta=beta_raw.view(1, rows, h),
                A_log=p.A_log, dt_bias=p.dt_bias, initial_state=rec_state, scale=p.scale,
                output_final_state=True, inplace_final_state=True, state_v_first=True,
                cu_seqlens=plan.cu_seqlens[: rows + 1].to(torch.long),
                ssm_state_indices=slot_ids.to(torch.int32),
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                lower_bound=p.lower_bound,
            )[0]
            return o.view(rows, h, d)
        # prefill: varlen over the packed rows with gathered initial states
        cu = plan.cu_seqlens[: rows + 1].to(torch.long)
        conv_init = conv_state.index_select(0, slot_ids) * has_state[:, None, None].to(conv_state.dtype)
        y, conv_final = self._conv(
            qkv.view(1, t, -1), weight=conv_w, initial_state=conv_init.to(qkv.dtype),
            output_final_state=True, activation="silu", cu_seqlens=cu,
        )
        conv_state.index_copy_(0, slot_ids, conv_final.to(conv_state.dtype))
        y3 = y.view(t, 3, h * d).transpose(0, 1).contiguous()  # one copy, three contiguous views
        q, k, v = y3[0], y3[1], y3[2]
        rec_init = rec_state.index_select(0, slot_ids) * has_state[:, None, None, None].to(rec_state.dtype)
        o, rec_final = self._chunk(
            q=q.view(1, t, h, d), k=k.view(1, t, h, d), v=v.view(1, t, h, d),
            g=g_raw.view(1, t, h, d), beta=beta_raw.view(1, t, h),
            A_log=p.A_log, dt_bias=p.dt_bias, initial_state=rec_init.float(), output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            safe_gate=p.lower_bound is not None, lower_bound=p.lower_bound, state_v_first=True,
            cu_seqlens=cu,
        )
        rec_state.index_copy_(0, slot_ids, rec_final.to(rec_state.dtype))
        return o.view(t, h, d)


class FlashKDAKernels(FLAKDAKernels):
    """``FLAKDAKernels`` with the prefill recurrence on FlashKDA (CUTLASS, sm90a): the
    packed varlen chunked scan takes the gathered fp32 V-first initial states and writes
    the final states, which are scattered back to the slots. Decode stays on the fla
    recurrent kernel (FlashKDA has no decode path)."""

    def __init__(self):
        super().__init__()
        import flash_kda

        self._fwd = flash_kda.fwd
        self._workspace_size = flash_kda.get_workspace_size
        self._workspace: torch.Tensor | None = None

    def _get_workspace(self, t: int, h: int, n: int, device) -> torch.Tensor:
        size = int(self._workspace_size(t, h, n))
        if self._workspace is None or self._workspace.numel() < size or self._workspace.device != device:
            self._workspace = torch.empty(size, dtype=torch.uint8, device=device)
        return self._workspace

    @torch.compiler.disable
    def run_paged(self, qkv, g_raw, beta_raw, plan: RecurrentPlanOutput, conv_state, rec_state, p) -> torch.Tensor:
        if plan.is_decode:
            return super().run_paged(qkv, g_raw, beta_raw, plan, conv_state, rec_state, p)
        h, d = p.num_heads, p.head_dim
        rows = plan.num_rows
        slot_ids = plan.slot_ids[:rows].to(torch.long)
        has_state = plan.has_state[:rows]
        t = qkv.shape[0]
        conv_w = p.conv_weight.to(qkv.dtype)
        cu = plan.cu_seqlens[: rows + 1].to(torch.long)
        conv_init = conv_state.index_select(0, slot_ids) * has_state[:, None, None].to(conv_state.dtype)
        y, conv_final = self._conv(
            qkv.view(1, t, -1), weight=conv_w, initial_state=conv_init.to(qkv.dtype),
            output_final_state=True, activation="silu", cu_seqlens=cu,
        )
        conv_state.index_copy_(0, slot_ids, conv_final.to(conv_state.dtype))
        q, k, v = y.view(t, -1).split([h * d, h * d, h * d], dim=-1)
        rec_init = (rec_state.index_select(0, slot_ids) * has_state[:, None, None, None].to(rec_state.dtype))
        rec_init = rec_init.float().contiguous()
        rec_final = torch.empty_like(rec_init)
        out = torch.empty(1, t, h, d, dtype=torch.bfloat16, device=qkv.device)
        self._fwd(
            q.view(1, t, h, d).to(torch.bfloat16).contiguous(), k.view(1, t, h, d).to(torch.bfloat16).contiguous(),
            v.view(1, t, h, d).to(torch.bfloat16).contiguous(), g_raw.view(1, t, h, d).to(torch.bfloat16).contiguous(),
            beta_raw.view(1, t, h).to(torch.bfloat16).contiguous(), float(p.scale), out,
            p.A_log.float().contiguous(), p.dt_bias.float().view(h, d).contiguous(), float(p.lower_bound),
            initial_state=rec_init, final_state=rec_final,
            cu_seqlens=cu.to(torch.int32), workspace=self._get_workspace(t, h, rows, qkv.device),
        )
        rec_state.index_copy_(0, slot_ids, rec_final.to(rec_state.dtype))
        return out.view(t, h, d).to(qkv.dtype)


def default_kernels(device: torch.device | str | None, prefer: str = "auto"):
    """The best KDA kernels for ``device``: FlashKDA prefill + fla decode on CUDA when
    ``flash_kda`` is importable (and the shapes fit its D == 128 requirement, checked
    at first use), fla alone otherwise, the torch reference off-GPU."""
    from mstar.model.kimi_k3.components.kda import TorchKDAKernels

    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type != "cuda":
        return TorchKDAKernels()
    if prefer in ("auto", "flashkda"):
        try:
            return FlashKDAKernels()
        except Exception:
            pass
    try:
        return FLAKDAKernels()
    except Exception:  # fla missing or broken: fall back, uncaptured
        return TorchKDAKernels()
