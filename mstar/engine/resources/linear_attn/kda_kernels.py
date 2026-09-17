"""GPU kernels for Kimi Delta Attention against a recurrent state pool, slot-indexed and
CUDA-graph safe.

``FLAKDAKernels`` runs the layer's inner math with flash-linear-attention's Triton kernels:

* prefill (packed varlen): ``causal_conv1d`` with per-sequence initial conv states and
  ``chunk_kda`` with per-sequence initial recurrent states, both gathered from the pool's
  slots by a *device* index tensor and scattered back with ``index_copy_``;
* decode (one token per row): the slot-indexed conv update (``conv_update.py``, in place in the
  pool; fla's ``causal_conv1d_update`` on gathered windows as the fallback) and
  ``fused_recurrent_kda_fwd`` with ``ssm_state_indices`` writing the recurrent slots in place.

``FlashKDAKernels`` swaps the prefill recurrence for FlashKDA (CUTLASS, sm90a). Every address the
kernels touch comes from tensors (``slot_ids``, ``cu_seqlens``; ``has_state`` masks the prefill's
initial states), so a captured decode graph replays correctly with new slot ids.

Pool layout (``DeltaNetGeometry``): ``state [slots, H, V, K]`` fp32, V-first, the kernels' native
layout; ``conv [slots, 3P, W - 1]``, the inputs before the current token. fla's conv window is one
column wider with a dead oldest column, so the prefill path pads a zero column in front and
keeps the last ``W - 1`` of the final state.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple

import torch
import torch.nn.functional as F

from mstar.engine.resources.linear_attn.conv_update import conv_update_slots, conv_update_slots_supported


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


class SpecBlocks(NamedTuple):
    """A layer's per-slot speculative blocks (``DeltaNetGeometry.to_blocks(speculative_tokens=k)``):
    the pending prefix a verify step left behind. ``prefix [slots, k+1, 3P]`` pre-conv inputs,
    ``g [slots, k+1, H, D]`` raw gates, ``beta [slots, k+1, H]`` raw betas, ``length [slots, 1]``
    int32 (accepted + 1 of them are real; 0 after a prefill), shared by every layer (layer 0's)."""
    prefix: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    length: torch.Tensor


def _fla_conv_state(
    state: torch.Tensor, slot_ids: torch.Tensor, has_state: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    """The rows' conv windows as fla wants them, ``[N, D, W]``: a zero (dead) oldest column in front
    of the pool's ``W - 1`` kept inputs, zeroed where the row has no state yet."""
    kept = state.index_select(0, slot_ids) * has_state[:, None, None].to(state.dtype)
    n, d, s = kept.shape
    return torch.cat([kept.new_zeros(n, d, 1), kept], dim=-1).to(dtype)


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
    def run_paged(self, qkv, g_raw, beta_raw, plan, conv_state, rec_state, p: KDAParams) -> torch.Tensor:
        """``qkv [T, 3P]`` (pre-conv), ``g_raw [T, H, D]``, ``beta_raw [T, H]``; the pool's layer
        blocks ``conv_state [slots, 3P, W - 1]`` and ``rec_state [slots, H, D, D]`` are updated in
        place. Returns ``o [T, H, D]``."""
        h, d = p.num_heads, p.head_dim
        rows = plan.num_rows
        t = qkv.shape[0]
        conv_w = p.conv_weight.to(qkv.dtype)  # [3P, W]
        if plan.is_decode:
            # decode reads the slots as they are: the pool hands slots out zeroed and padding rows
            # address the sink, so a row's first decode after its prefill finds the written state
            # and a fresh slot holds zeros -- no masking, no gather/scatter of the recurrent states
            # (tens of MB per layer at 64 rows). The plan's int32 index buffers are used as they
            # are: index_select, advanced indexing and the fla kernels all take int32, and every
            # dtype conversion here would be one more captured launch per layer per step
            slot_ids = plan.slot_ids[:rows]
            if conv_update_slots_supported(qkv, conv_state):
                # one launch: the slot-indexed conv update reads and rewrites each row's window in
                # the pool itself (fla's kernel needs a gathered [rows, 3P, W] copy and a scatter back)
                y = conv_update_slots(qkv, conv_state, slot_ids, conv_w, activation="silu")
            else:
                cache = torch.cat([conv_state.new_zeros(rows, conv_state.shape[1], 1),
                                   conv_state.index_select(0, slot_ids)], dim=-1)
                y, cache = self._conv_update(qkv.view(rows, 1, -1), cache, weight=conv_w, activation="silu")
                conv_state[slot_ids] = cache[..., 1:].to(conv_state.dtype)
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
                cu_seqlens=plan.cu_seqlens[: rows + 1],
                ssm_state_indices=slot_ids,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                lower_bound=p.lower_bound,
            )[0]
            return o.view(rows, h, d)
        # prefill: varlen over the packed rows with gathered initial states
        slot_ids = plan.slot_ids[:rows].to(torch.long)
        has_state = plan.has_state[:rows]
        cu = plan.cu_seqlens[: rows + 1].to(torch.long)
        y, conv_final = self._conv(
            qkv.view(1, t, -1), weight=conv_w, output_final_state=True, activation="silu", cu_seqlens=cu,
            initial_state=_fla_conv_state(conv_state, slot_ids, has_state, qkv.dtype),
        )
        conv_state.index_copy_(0, slot_ids, conv_final[..., 1:].to(conv_state.dtype))
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


    def run_verify(self, qkv, g_raw, beta_raw, plan, conv_state, rec_state, spec: SpecBlocks, p: KDAParams):
        """The checkpoint recurrence of a verify step (plan section 8.3 item 6), static shapes and
        device tensors only (capturable). Rows gather their checkpoint window and pending prefix by
        slot; the prefix's conv (from that window) and the block's conv (from the window after the
        prefix, gathered at the prefix length) run as fp32 sequential taps like the reference;
        prefix positions past the accepted length become no-op tokens for the recurrence (k = v =
        0, raw gate and beta at -1e4: decay 1, beta 0, verified bit-identical); one launch of
        ``kda_recurrent_checkpoint`` runs prefix + block from the slot's state and writes the slot
        after the prefix; the block's outputs are returned, its raw inputs saved as the next prefix
        and the window after the prefix written to the pool. Same semantics as
        ``TorchKDAKernels.run_verify``."""
        from mstar.engine.resources.linear_attn.kda_spec_recurrent import kda_recurrent_checkpoint

        h, d = p.num_heads, p.head_dim
        rows = plan.num_rows
        k1 = plan.cu_seqlens_cpu[1] - plan.cu_seqlens_cpu[0]
        assert rows * k1 == qkv.shape[0], (rows, k1, qkv.shape)
        conv_w = p.conv_weight.float()  # [3P, W]
        w = conv_w.shape[1]
        slots = plan.slot_ids[:rows].to(torch.long)
        win = conv_state.index_select(0, slots).transpose(1, 2).float()  # [rows, W-1, 3P]: the checkpoint window
        pre = spec.prefix.index_select(0, slots).float()  # [rows, k1, 3P]
        # padding rows address the sink, whose length is garbage: clamp before it indexes anything
        plen = spec.length.index_select(0, slots)[:, 0].clamp_(0, k1)  # [rows] int32
        blk = qkv.view(rows, k1, -1).float()

        def conv(x):  # x [rows, W-1+T, 3P] -> silu(depthwise causal conv) [rows, T, 3P], fp32 taps in tap order
            t = x.shape[1] - (w - 1)
            y = x[:, 0:t] * conv_w[:, 0]
            for j in range(1, w):
                y = y + x[:, j:j + t] * conv_w[:, j]
            return F.silu(y)

        combined = torch.cat([win, pre], dim=1)  # [rows, W-1+k1, 3P]
        idx = plen.to(torch.long)[:, None] + torch.arange(w - 1, device=qkv.device)[None, :]  # last W-1 real inputs
        win_after = torch.gather(combined, 1, idx[:, :, None].expand(-1, -1, combined.shape[-1]))  # [rows, W-1, 3P]
        y_pre = conv(combined).to(qkv.dtype)  # positions >= plen are garbage, masked below
        y_blk = conv(torch.cat([win_after, blk], dim=1)).to(qkv.dtype)  # [rows, k1, 3P]
        conv_state.index_copy_(0, slots, win_after.transpose(1, 2).to(conv_state.dtype))

        def heads(y):  # [rows, T, 3P] -> q, k, v each [rows, T, H, D]
            q, k, v = y.split([h * d, h * d, h * d], dim=-1)
            return q.reshape(rows, -1, h, d), k.reshape(rows, -1, h, d), v.reshape(rows, -1, h, d)

        qP, kP, vP = heads(y_pre)
        qB, kB, vB = heads(y_blk)
        pad = (torch.arange(k1, device=qkv.device)[None, :] >= plen[:, None])  # [rows, k1]
        kP = torch.where(pad[:, :, None, None], torch.zeros_like(kP), kP)
        vP = torch.where(pad[:, :, None, None], torch.zeros_like(vP), vP)
        gP = torch.where(pad[:, :, None, None], torch.full_like(spec.g[:1, :1], -1e4).expand(rows, k1, h, d),
                         spec.g.index_select(0, slots))
        bP = torch.where(pad[:, :, None], torch.full_like(spec.beta[:1, :1], -1e4).expand(rows, k1, h),
                         spec.beta.index_select(0, slots))
        gB = g_raw.view(rows, k1, h, d)
        bB = beta_raw.view(rows, k1, h)
        cat = lambda a, b: torch.cat([a, b], dim=1).reshape(rows * 2 * k1, *a.shape[2:]).contiguous()
        cu = torch.arange(rows + 1, device=qkv.device, dtype=torch.int32) * (2 * k1)
        o = kda_recurrent_checkpoint(
            cat(qP, qB), cat(kP, kB), cat(vP, vB), cat(gP.to(qkv.dtype), gB.to(qkv.dtype)),
            cat(bP.to(qkv.dtype), bB.to(qkv.dtype)), p.A_log, p.dt_bias, rec_state, plan.slot_ids[:rows],
            (plen - 1).to(torch.int32), cu, p.scale, p.lower_bound,
        )
        o = o.view(rows, 2 * k1, h, d)[:, k1:].reshape(rows * k1, h, d)
        # this block is the next step's pending prefix (its length follows the verification)
        spec.prefix.index_copy_(0, slots, blk.to(spec.prefix.dtype))
        spec.g.index_copy_(0, slots, gB.to(spec.g.dtype))
        spec.beta.index_copy_(0, slots, bB.to(spec.beta.dtype))
        return o


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
    def run_paged(self, qkv, g_raw, beta_raw, plan, conv_state, rec_state, p: KDAParams) -> torch.Tensor:
        if plan.is_decode:
            return super().run_paged(qkv, g_raw, beta_raw, plan, conv_state, rec_state, p)
        h, d = p.num_heads, p.head_dim
        rows = plan.num_rows
        slot_ids = plan.slot_ids[:rows].to(torch.long)
        has_state = plan.has_state[:rows]
        t = qkv.shape[0]
        conv_w = p.conv_weight.to(qkv.dtype)
        cu = plan.cu_seqlens[: rows + 1].to(torch.long)
        y, conv_final = self._conv(
            qkv.view(1, t, -1), weight=conv_w, output_final_state=True, activation="silu", cu_seqlens=cu,
            initial_state=_fla_conv_state(conv_state, slot_ids, has_state, qkv.dtype),
        )
        conv_state.index_copy_(0, slot_ids, conv_final[..., 1:].to(conv_state.dtype))
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


def default_kernels(device: torch.device | str | None, backend: str = "auto", flashkda_fits: bool = True):
    """The KDA kernels for ``device`` and ``backend`` (``auto`` | ``flashkda`` | ``fla``): FlashKDA
    prefill + fla decode on CUDA when ``flash_kda`` imports and the shapes fit it
    (``flashkda_fits``: head_dim 128 with a bounded gate), fla alone otherwise; None off-GPU
    (a model then installs its torch reference kernels, see ``KDAManager.set_kernels``)."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type != "cuda":
        return None
    if backend in ("auto", "flashkda") and flashkda_fits:
        try:
            return FlashKDAKernels()
        except Exception:
            if backend == "flashkda":
                raise
    try:
        return FLAKDAKernels()
    except Exception:  # fla missing or broken: the model's reference kernels, uncaptured
        return None


__all__ = ["FLAKDAKernels", "FlashKDAKernels", "KDAParams", "default_kernels"]
