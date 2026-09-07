"""Routed MXFP4 experts on FlashInfer's CUTLASS fused MoE (SM90 mixed-input GEMMs).

Two Hopper paths share one weight format (packed E2M1 bytes + E8M0 scales, permuted into the
GEMM's tile layout): ``w4a16`` keeps bf16 activations, ``humming`` quantizes activations to FP8
per token inside the kernel (FP8 x MXFP4 tensor cores, ~2x the W4A16 GEMM rate). SiTU-GLU runs
fused between the two GEMMs (``flashinfer_situ_patch`` adds it to the JIT sources).

The conversion runs **in place, one expert at a time**, so a rank holding 40+ GiB of experts
never needs a second copy: the packed parameters keep their logical shape but their bytes become
the FlashInfer layout (``fc1`` rows reordered from ``[gate | up]`` to ``[up | gate]`` first).
"""
from __future__ import annotations

import torch

from mstar.model.kimi_k3.reference.mxfp4 import MXFP4_GROUP


class FlashInferMXFP4Experts:
    """Owns the converted expert parameters and runs ``cutlass_fused_moe`` on them.

    ``gate_up_packed [E, 2*inter, latent/2]`` / ``gate_up_scale [E, 2*inter, latent/32]`` and
    ``down_packed [E, latent, inter/2]`` / ``down_scale [E, latent, inter/32]`` are the module's
    uint8 parameters (rows ``[gate | up]``); they are rewritten in place by ``convert``.
    """

    def __init__(self, *, mode: str, situ_beta: float, situ_linear_beta: float | None, device: torch.device):
        from flashinfer.tllm_enums import ActivationType

        assert mode in ("w4a16", "humming"), mode
        self.mode = mode
        self.activation = ActivationType.Situ
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta) if situ_linear_beta is not None else 0.0
        self.device = device
        self.converted = False
        self.w13: torch.Tensor | None = None
        self.w2: torch.Tensor | None = None
        self.quant_scales: list[torch.Tensor] = []
        self.alpha: torch.Tensor | None = None
        self.beta: torch.Tensor | None = None

    # ------------------------------------------------------------------ conversion
    @torch.no_grad()
    def convert(self, gate_up_packed, gate_up_scale, down_packed, down_scale) -> None:
        from flashinfer import fused_moe

        e, two_inter, half_latent = gate_up_packed.shape
        inter = two_inter // 2
        latent = half_latent * 2
        assert down_packed.shape == (e, latent, inter // 2), down_packed.shape
        residual13 = torch.empty(e, dtype=torch.float32, device=self.device)
        residual2 = torch.empty(e, dtype=torch.float32, device=self.device)
        scale13_shape = scale2_shape = None
        for i in range(e):
            # fc1 rows: [gate | up] -> [up | gate]
            w = torch.cat([gate_up_packed[i, inter:], gate_up_packed[i, :inter]]).unsqueeze(0).contiguous()
            s = torch.cat([gate_up_scale[i, inter:], gate_up_scale[i, :inter]]).unsqueeze(0).contiguous()
            w_il, s_il, r = self._convert_one(fused_moe, w, s)
            gate_up_packed[i].copy_(w_il.view(two_inter, half_latent))
            gate_up_scale[i].view(-1).copy_(s_il.reshape(-1))
            residual13[i] = r
            scale13_shape = s_il.shape[1:]
            w = down_packed[i].unsqueeze(0).contiguous()
            s = down_scale[i].unsqueeze(0).contiguous()
            w_il, s_il, r = self._convert_one(fused_moe, w, s)
            down_packed[i].copy_(w_il.view(latent, inter // 2))
            down_scale[i].view(-1).copy_(s_il.reshape(-1))
            residual2[i] = r
            scale2_shape = s_il.shape[1:]
        self.w13, self.w2 = gate_up_packed, down_packed
        s13 = gate_up_scale.view(e, *scale13_shape).view(torch.int32)
        s2 = down_scale.view(e, *scale2_shape).view(torch.int32)
        if self.mode == "humming":
            # slots: fc1 folded scales, fc1 residual (x 2^6 epilogue compensation), reserved
            # fc2 activation scale, fc2 folded scales, fc2 residual
            self.quant_scales = [
                s13, (residual13 * 64.0).contiguous(), torch.ones((), device=self.device, dtype=torch.float32),
                s2, (residual2 * 64.0).contiguous(),
            ]
        else:
            self.quant_scales = [s13, s2]
        self.alpha = torch.full((e,), self.situ_beta, device=self.device, dtype=torch.float32)
        self.beta = torch.full((e,), self.situ_linear_beta, device=self.device, dtype=torch.float32)
        self.converted = True

    def _convert_one(self, fused_moe, w: torch.Tensor, s: torch.Tensor):
        """``w [1, rows, K/2]``, ``s [1, rows, K/32]`` -> interleaved weight, folded scales,
        per-expert residual (1.0 for W4A16, Humming's factored E8M0 residual otherwise)."""
        if self.mode == "humming":
            w_il, s_il, r = fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming(w, s)
            return w_il, s_il, r.reshape(-1)[0]
        w_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w, "fp4")
        s_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(s, MXFP4_GROUP)
        return w_il, s_il, torch.ones((), device=w.device, dtype=torch.float32)

    # ------------------------------------------------------------------ tuning
    @torch.no_grad()
    def autotune(self, token_counts=(1, 2, 4, 8, 16, 32, 64, 128), top_k: int = 16) -> bool:
        """Run the routed GEMMs once per decode bucket under FlashInfer's autotuner so the
        CUTLASS grouped GEMM keeps the best tactic per shape for the process (the default
        tactic is ~35% slower at one token). Tactics are cached by shape, so one converted
        layer tunes every layer of the same shape. Returns False when the autotuner is absent."""
        try:
            from flashinfer.autotuner import autotune
        except Exception:
            return False
        assert self.converted
        e, _, half_latent = self.w13.shape
        latent = half_latent * 2
        k = min(top_k, e)
        with autotune(True):
            for t in token_counts:
                x = torch.randn(t, latent, device=self.device, dtype=torch.bfloat16)
                idx = torch.stack([torch.randperm(e, device=self.device)[:k] for _ in range(t)]).to(torch.int32)
                w = torch.softmax(torch.randn(t, k, device=self.device), -1)
                for _ in range(2):
                    self(x, idx, w)
        return True

    # ------------------------------------------------------------------ forward
    @torch.compiler.disable
    def __call__(self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        from flashinfer import fused_moe

        assert self.converted, "convert() the expert weights first"
        out = torch.empty(z.shape[0], z.shape[1], dtype=torch.bfloat16, device=z.device)
        fused_moe.cutlass_fused_moe(
            z.to(torch.bfloat16), topk_idx.to(torch.int32), topk_weight.to(torch.float32),
            self.w13, self.w2, torch.bfloat16, quant_scales=self.quant_scales,
            use_w4_group_scaling=True, use_wfp4afp8_humming=(self.mode == "humming"), output=out,
            activation_type=self.activation, swiglu_alpha=self.alpha, swiglu_beta=self.beta,
        )
        return out.to(z.dtype) if out.dtype != z.dtype else out
