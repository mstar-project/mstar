"""Marlin MXFP4 MoE backend for Kimi K3's routed experts (SM80+; the kernel vLLM runs on H100).

The kernels are vLLM's ``marlin_moe_wna16`` (Apache-2.0), vendored under ``csrc/`` (see ``csrc/NOTICE``)
and JIT-built once per machine as ``_mstar_marlin_C`` (torch's extension cache), like the
``moe_align_block_size`` op. This module holds the weight/scale preparation (a per-expert repack into
Marlin's tile order plus the scale permutation, all from vLLM's ``marlin_utils``) and the fused
call: token/expert alignment (M*'s ``moe_align_block_size``), the gate/up GEMM, SiTU-GLU
(M*'s Triton kernel), the down GEMM with the routing weights applied, and the top-k reduction.

Same duck-typed interface as ``FlashInferMXFP4Experts`` (``convert(...)`` then
``__call__(z, idx, w)``), except that ``convert`` returns the converted tensors: Marlin's layouts
have other shapes and dtypes, so the owning module rebinds its parameters to them.

Expert parallelism: the backend takes the module's ``ExpertSharding``; ``__call__`` receives
global expert ids, maps them to this rank's local experts (assignments of other ranks get the
sharding's invalid id, which the alignment step drops) and zero-fills their top-k slots so the
summed result is this rank's partial.
"""
from __future__ import annotations

import glob
import os

import torch

# vllm.scalar_type.scalar_types.float4_e2m1f.id: the packed ScalarType the kernel dispatches on
FP4_E2M1F_ID = 562949953487106
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_MOE = os.path.join(_CSRC, "marlin_moe_wna16")


def _load_ops():
    if hasattr(torch.ops, "_mstar_marlin_C") and hasattr(torch.ops._mstar_marlin_C, "moe_wna16_marlin_gemm"):
        return torch.ops._mstar_marlin_C
    from torch.utils.cpp_extension import load

    from mstar.utils.fused_moe.align import _clear_stale_build_lock

    _clear_stale_build_lock("_mstar_marlin_C")
    sources = [os.path.join(_CSRC, "bindings.cpp"), os.path.join(_MOE, "ops.cu"),
               os.path.join(_CSRC, "marlin", "gptq_marlin_repack.cu")]
    sources += sorted(glob.glob(os.path.join(_MOE, "sm80_kernel_*.cu")))
    load(
        name="_mstar_marlin_C", sources=sources, is_python_module=False, verbose=False,
        extra_include_paths=[_CSRC, _MOE], extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode", "arch=compute_90,code=sm_90",
                           "-static-global-template-stub=false", "--expt-relaxed-constexpr", "-DUSE_CUDA"],
    )
    return torch.ops._mstar_marlin_C


# --- weight preparation (vLLM marlin_utils / marlin_utils_fp4, MXFP4 branch) ---------------------

def _scale_perms() -> tuple[list[int], list[int]]:
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return perm, single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    perm, single = _scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape(-1, len(perm))[:, perm]
    else:
        s = s.reshape(-1, len(single))[:, single]
    return s.reshape(-1, size_n).contiguous()


def mxfp4_process_scales(s: torch.Tensor) -> torch.Tensor:
    """bf16 activations: pair-swap columns then store as E8M0 (vLLM ``mxfp4_marlin_process_scales``)."""
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    return s.to(torch.float8_e8m0fnu)


def repack_experts(ops, packed: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    """``packed [E, N, K/2]`` uint8 (two fp4 per byte, low nibble first) -> Marlin tile order."""
    e = packed.shape[0]
    perm = torch.empty(0, dtype=torch.int, device=packed.device)
    out = None
    for i in range(e):
        q = packed[i].view(torch.int32).T.contiguous()  # [K/8, N] GPTQ order
        m = ops.gptq_marlin_repack(q, perm, size_k, size_n, 4, False)
        if out is None:
            out = torch.empty((e, *m.shape), dtype=m.dtype, device=m.device)
        out[i] = m
    return out


def prepare_scales(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    """``scale [E, N, K/32]`` E8M0-as-uint8 -> ``[E, K/32, N]`` E8M0 in Marlin order."""
    s = scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    return torch.stack([mxfp4_process_scales(marlin_permute_scales(s[i].T, size_k, size_n, 32)) for i in range(s.shape[0])])


class MarlinMXFP4Experts:
    """Routed-expert forward on the Marlin MXFP4 MoE kernel (bf16 activations)."""

    def __init__(self, situ_beta: float, situ_linear_beta: float | None, device: torch.device, sharding=None):
        self.ops = _load_ops()
        self.situ_beta, self.situ_linear_beta = float(situ_beta), situ_linear_beta
        self.device = torch.device(device)
        # ExpertSharding of the owning module (None: every expert is local)
        self.sharding = sharding
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.workspace = torch.zeros(sms * 4, dtype=torch.int, device=self.device)
        self.w13 = self.s13 = self.w2 = self.s2 = None
        self.num_experts = self.inter = self.latent = None

    def convert(self, gate_up_packed, gate_up_scale, down_packed, down_scale):
        """Repack ``gate_up_packed [E, 2I, K/2]`` / ``down_packed [E, K, I/2]`` (uint8 MXFP4) and
        their E8M0 scales into Marlin's layouts. Returns the four converted tensors in the same
        order (other shapes and dtypes: the caller rebinds its parameters to them); the backend
        keeps references. Peak memory during conversion is one expert's worth over the inputs."""
        e, n2, k_half = gate_up_packed.shape
        k, inter = k_half * 2, n2 // 2
        assert k % 256 == 0 and inter % 128 == 0, (k, inter)  # Marlin tile alignment
        self.num_experts, self.inter, self.latent = e, inter, k
        self.w13 = repack_experts(self.ops, gate_up_packed, n2, k)
        self.s13 = prepare_scales(gate_up_scale, n2, k)
        self.w2 = repack_experts(self.ops, down_packed, k, inter)
        self.s2 = prepare_scales(down_scale, k, inter)
        return self.w13, self.s13, self.w2, self.s2

    def rebind(self, gate_up_packed, gate_up_scale, down_packed, down_scale) -> None:
        """Point at the (converted) parameter tensors again after their storage was rebound."""
        self.w13, self.s13, self.w2, self.s2 = gate_up_packed, gate_up_scale, down_packed, down_scale

    @staticmethod
    def block_size_m(m: int, top_k: int, e: int) -> int:
        for bs in (8, 16, 32, 48, 64):
            if m * top_k / e / bs < 0.9:
                return bs
        return 64

    # longest prefill slice per kernel call: bounds the transient ``[m * top_k, latent]`` buffers
    # (2048 tokens x 16 experts x 3584 x bf16 = 235 MB) instead of a whole 16k-token prefill
    max_chunk_tokens = 2048

    @torch.compiler.disable
    def __call__(
        self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out [m, latent]`` (optional, ``z``'s dtype) receives the result -- the summed expert
        outputs land there directly, e.g. in the all-reduce buffer."""
        n = self.max_chunk_tokens
        if z.shape[0] <= n:
            return self._forward(z, topk_idx, topk_weight, out)
        # routing is per token, so slices along the token axis are independent
        if out is None:
            return torch.cat([self._forward(z[i:i + n], topk_idx[i:i + n], topk_weight[i:i + n])
                              for i in range(0, z.shape[0], n)])
        for i in range(0, z.shape[0], n):
            self._forward(z[i:i + n], topk_idx[i:i + n], topk_weight[i:i + n], out[i:i + n])
        return out

    def _forward(
        self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe.align import moe_align_block_size
        from mstar.utils.fused_moe.mxfp4 import situ_and_mul_triton

        m, k = z.shape
        top_k = topk_idx.shape[1]
        e, inter = self.num_experts, self.inter
        partial = self.sharding is not None and self.sharding.is_partial
        if partial:  # global -> local expert ids, other ranks' assignments -> the skipped id
            topk_idx = self.sharding.localize(topk_idx)
        idx = topk_idx.to(torch.int32).contiguous()
        w = topk_weight.to(torch.float32).contiguous()
        bs_m = self.block_size_m(m, top_k, e)
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(idx, bs_m, e)
        c1 = torch.empty(m * top_k, 2 * inter, dtype=z.dtype, device=z.device)
        c2 = torch.empty(m * top_k, inter, dtype=z.dtype, device=z.device)
        # slots of skipped assignments are never written: zero them so the top-k sum is the partial
        c3 = (torch.zeros if partial else torch.empty)(m * top_k, k, dtype=z.dtype, device=z.device)
        self.ops.moe_wna16_marlin_gemm(
            z, c1, self.w13, None, self.s13, None, None, None, None, None, self.workspace,
            sorted_ids, expert_ids, num_post_pad, w, bs_m, top_k, False, FP4_E2M1F_ID,
            m, 2 * inter, k, True, False, True, False, -1, -1, -1)
        situ_and_mul_triton(c1, c2, self.situ_beta, self.situ_linear_beta)
        self.ops.moe_wna16_marlin_gemm(
            c2, c3, self.w2, None, self.s2, None, None, None, None, None, self.workspace,
            sorted_ids, expert_ids, num_post_pad, w, bs_m, 1, True, FP4_E2M1F_ID,
            m * top_k, k, inter, True, False, True, False, -1, -1, -1)
        if out is None:
            return c3.view(m, top_k, k).sum(dim=1)
        torch.sum(c3.view(m, top_k, k), dim=1, out=out)
        return out
