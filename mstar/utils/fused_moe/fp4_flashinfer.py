"""Routed experts in MXFP4 or NVFP4 on B200/B300 (sm_100 / sm_103): FlashInfer's TRT-LLM FP4 MoE.

The checkpoint's packed experts (``moe.QuantExperts``: e2m1 bytes ``(E, rows, K/2)`` and row-major block scales)
are put in the kernel's layout once, in place, after the load: gate/up rows reordered to ``[up; gate]``, each
expert's bytes and scales permuted for the MMA (``_maybe_get_cached_w3_w1_permute_indices``,
``get_w2_permute_indices_with_cache``) and the scales interleaved (``block_scale_interleave``) — the steps of
FlashInfer 0.7.2's ``prepare_trtllm_fp4_weights`` minus its quantizer, which the checkpoint has already been
through. ``trtllm_fp4_block_scale_routed_moe`` then takes M*'s own routing (ids and weights).

- MXFP4 (Kimi K3/K3.1's experts): bf16 activations (``W4A16``, the H200 path's numerics) by default;
  ``activation="mxfp8"`` quantizes them per 32 to e4m3 (``W4A8``, the FP8 tensor-core rate; hypothesis: faster at
  some quality cost, which the B300 run measures).
- NVFP4 (modelopt, nvidia/*-NVFP4): activations quantized to NVFP4 with the layer's calibrated input scale (the
  largest of its experts'), dequant scalars per expert as vLLM's modelopt path sets them.

Checked on a B300 (sm_103): on nvidia/GLM-5.3-NVFP4's own experts it matches a reference that rounds the
activations as FlashInfer's quantizer does to 0.022 (rel. L2); the CPU reference rounds activation block scales up,
not to nearest, so the GPU tests bound the kernel by that reference's own error (``test_glm52_nvfp4.py``).
"""

from __future__ import annotations

import torch

_EPILOGUE_TILE_M = 128  # the TRT-LLM kernels' tile along a GEMM's rows
_SUPPORTED = ((10, 0), (10, 3))


def unavailable_reason(device: torch.device) -> str | None:
    from mstar.platforms import current_platform

    capability = current_platform.cuda_capability(device) if device.type == "cuda" else None
    if capability is None or tuple(capability) not in _SUPPORTED:
        return "FlashInfer's TRT-LLM FP4 MoE runs on sm_100 / sm_103 (B200, B300)"
    try:
        from flashinfer.fused_moe.core import trtllm_fp4_block_scale_routed_moe  # noqa: F401
    except ImportError as e:
        return f"FlashInfer's TRT-LLM FP4 MoE is not importable ({e})"
    return None


def _shuffle(w: torch.Tensor, s: torch.Tensor, gated: bool, sf_block: int, cache: dict) -> None:
    """One expert's ``w (rows, K/2)`` and ``s (rows, K/sf_block)`` bytes, permuted in place for the MMA."""
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    permute = _maybe_get_cached_w3_w1_permute_indices if gated else get_w2_permute_indices_with_cache
    kw = {"is_gated_act_gemm": True} if gated else {}
    p = permute(cache, w, _EPILOGUE_TILE_M, **kw)
    w.copy_(w[p.to(w.device)].contiguous())
    p_sf = permute(cache, s, _EPILOGUE_TILE_M, num_elts_per_sf=16, **kw)
    s.view(-1).copy_(block_scale_interleave(s[p_sf.to(s.device)].contiguous()).view(-1))


class TrtllmFp4Experts:
    """The converted experts of one MoE block and the call that runs them."""

    def __init__(self, block, fmt: str, activation: str = "bf16"):
        self.block, self.fmt, self.activation = block, fmt, activation

    @classmethod
    @torch.no_grad()
    def convert(cls, block, device: torch.device, activation: str = "bf16") -> TrtllmFp4Experts:
        fmt = block.experts_packed.name
        e, two_inter, hidden = block._experts_dims()
        inter = two_inter // 2
        if hidden % 128 or inter % 128:
            raise ValueError(f"the TRT-LLM FP4 MoE needs hidden and expert widths in 128s; got {hidden}, {inter}")
        experts = block.experts
        w1, w2 = ((experts.gate_up_proj_packed, experts.down_proj_packed) if fmt == "mxfp4"
                  else (experts.gate_up_proj, experts.down_proj))
        s1, s2 = experts.gate_up_proj_scale, experts.down_proj_scale
        sf_block = 32 if fmt == "mxfp4" else 16
        cache: dict = {}
        for i in range(e):
            # [gate; up] -> [up; gate], the kernel's order
            w1[i].copy_(torch.cat([w1[i, inter:], w1[i, :inter]]))
            s1[i].copy_(torch.cat([s1[i, inter:], s1[i, :inter]]))
            _shuffle(w1[i], s1[i], True, sf_block, cache)
            _shuffle(w2[i], s2[i], False, sf_block, cache)
        out = cls(block, fmt, activation)
        ones = torch.ones(e, dtype=torch.float32, device=device)
        if fmt == "mxfp4":
            out.scalars = (ones, ones, ones)
            out.input_global = None
        else:
            f32 = lambda t: t.view(torch.float32)  # noqa: E731
            in1 = f32(experts.gate_up_proj_input_scale).max()          # one activation scale for every expert
            ws1 = f32(experts.gate_up_proj_scale_2)[:, 0].contiguous()   # gate == up (checked at load)
            in2 = f32(experts.down_proj_input_scale)[:, 0].contiguous()
            ws2 = f32(experts.down_proj_scale_2)[:, 0].contiguous()
            alpha1 = in1 * ws1
            out.scalars = (alpha1 / in2, alpha1, in2 * ws2)
            out.input_global = (1.0 / in1).reshape(1)
        return out

    def _hidden(self, flat: torch.Tensor):
        if self.fmt == "nvfp4":
            from flashinfer import fp4_quantize

            q, sf = fp4_quantize(flat, global_scale=self.input_global, sf_vec_size=16, sf_use_ue8m0=False,
                                 is_sf_swizzled_layout=False)
            return q, sf.view(torch.float8_e4m3fn).reshape(flat.shape[0], -1)
        if self.activation == "mxfp8":
            from flashinfer.quantization.fp8_quantization import mxfp8_quantize

            q, sf = mxfp8_quantize(flat, is_sf_swizzled_layout=False)
            return q, sf.view(torch.float8_e4m3fn).reshape(flat.shape[0], -1)
        return flat, None

    # FlashInfer's launch wrappers stay outside dynamo; the kernels run inside a captured graph
    @torch.compiler.disable
    def __call__(self, flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
                 skip_invalid: bool) -> torch.Tensor:
        from flashinfer.fused_moe.core import trtllm_fp4_block_scale_routed_moe

        experts = self.block.experts
        e, two_inter, _ = self.block._experts_dims()
        w1, w2 = ((experts.gate_up_proj_packed, experts.down_proj_packed) if self.fmt == "mxfp4"
                  else (experts.gate_up_proj, experts.down_proj))
        hidden, hidden_scale = self._hidden(flat)
        out1, out1_gate, out2 = self.scalars
        # the sentinel id ``e`` (padding rows, another rank's experts) is one more expert no rank holds here
        return trtllm_fp4_block_scale_routed_moe(
            (selected_experts.to(torch.int32), routing_weights.to(torch.float32)), None, hidden, hidden_scale,
            w1, experts.gate_up_proj_scale.view(torch.float8_e4m3fn), None, None, None, None,
            w2, experts.down_proj_scale.view(torch.float8_e4m3fn), None, out1, out1_gate, out2,
            e + 1 if skip_invalid else e, selected_experts.shape[1], None, None, two_inter // 2, 0, e, None,
            do_finalize=True,
        )[0]
