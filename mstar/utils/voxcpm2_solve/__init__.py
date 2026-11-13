"""H200 B1 VoxCPM2 solve. The engine owns compilation, capture and replay."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)
CUTLASS_REVISION = "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"


@functools.lru_cache(maxsize=1)
def load_native():
    """Build once, before capture, from packaged sources and pinned CUTLASS headers."""
    from torch.utils.cpp_extension import load

    cutlass = Path(os.environ["CUTLASS_ROOT"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip()
    if revision != CUTLASS_REVISION:
        raise RuntimeError(f"VoxCPM2 solve requires CUTLASS {CUTLASS_REVISION}, got {revision}")
    subprocess.run(["git", "-C", str(cutlass), "diff", "--quiet", "HEAD", "--", "include"], check=True)
    source = Path(__file__).parent / "csrc"
    digest = hashlib.sha256((torch.__version__ + revision).encode())
    for name in ("bindings.cpp", "solve.cu", "gate.cuh"):
        digest.update((source / name).read_bytes())
    return load(
        name=f"_mstar_vox_solve_{digest.hexdigest()[:16]}",
        sources=[str(source / "bindings.cpp"), str(source / "solve.cu")],
        extra_include_paths=[str(cutlass / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a", "--expt-relaxed-constexpr",
                          "--expt-extended-lambda", "-lineinfo"],
        extra_ldflags=["-lcuda", "-lcublasLt"],
    )


@torch.library.custom_op("mstar::voxcpm2_solve", mutates_args=())
def solve(
    lm: torch.Tensor, residual: torch.Tensor, cond: torch.Tensor, noise: torch.Tensor,
    cfg: torch.Tensor, span: torch.Tensor, weights: list[torch.Tensor],
    biases: list[torch.Tensor], norms: list[torch.Tensor], cosine: torch.Tensor,
    sine: torch.Tensor, eps: float,
) -> torch.Tensor:
    return load_native().solve(lm, residual, cond, noise, cfg, span, weights, biases, norms, cosine, sine, eps)


@solve.register_fake
def _solve_fake(lm, residual, cond, noise, cfg, span, weights, biases, norms, cosine, sine, eps):
    return lm.new_empty((lm.shape[0], 4, 64))


def pack_gate(weight: torch.Tensor) -> torch.Tensor:
    """Interleave eight gate and eight up rows, matching the WGMMA register partners."""
    return weight.reshape(2, 512, 8, 1024).transpose(0, 1).reshape(8192, 1024).contiguous()


class PatchSolve(nn.Module):
    """Immutable inference weights; all invocation storage lives inside the op.

    Nonpersistent buffers keep device moves coherent without changing checkpoint
    names. The original weights remain available for unsupported shapes and fallback.
    """

    def __init__(self, model):
        super().__init__()
        dit = model.feat_decoder.estimator
        linears = [model.lm_to_dit_proj, model.res_to_dit_proj, dit.cond_proj, dit.in_proj,
                   dit.time_mlp.linear_1, dit.time_mlp.linear_2,
                   dit.delta_time_mlp.linear_1, dit.delta_time_mlp.linear_2, dit.out_proj]
        norms = []
        for layer in dit.decoder.layers:
            linears.extend([layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                            layer.mlp.gate_up_proj, layer.mlp.down_proj])
            norms.extend([layer.input_layernorm.weight, layer.post_attention_layernorm.weight])
        norms.append(dit.decoder.norm.weight)
        for i, module in enumerate(linears):
            weight = module.weight.detach()
            if i >= 9 and (i - 9) % 4 == 2:
                weight = pack_gate(weight)
            self.register_buffer(f"w{i}", weight, persistent=False)
            if i < 9:
                self.register_buffer(f"b{i}", module.bias.detach(), persistent=False)
        for i, weight in enumerate(norms):
            self.register_buffer(f"n{i}", weight.detach(), persistent=False)
        self.register_buffer("cosine", dit.decoder.rope.cos[:11].contiguous(), persistent=False)
        self.register_buffer("sine", dit.decoder.rope.sin[:11].contiguous(), persistent=False)
        self.eps = model.config.dit.rms_norm_eps

    def _apply(self, fn, recurse=True):
        # The worker casts the loaded submodule again. Like RopeTable, retain
        # the materialized rotary values in fp32 while following device moves.
        cosine, sine = self.cosine, self.sine
        super()._apply(fn, recurse)
        self.cosine = cosine.to(device=self.cosine.device)
        self.sine = sine.to(device=self.sine.device)
        return self

    def supports(self, lm, residual, cond, noise, cfg, span) -> bool:
        batch = lm.shape[0] if lm.ndim == 2 else 0
        return (
            batch == 1
            and lm.shape == (batch, 2048) and residual.shape == (batch, 2048)
            and cond.shape == (batch, 4, 64) and noise.shape == (batch, 64, 4)
            and cfg.shape == (batch,) and span.shape == (11,)
            and all(x.dtype == torch.bfloat16 for x in (lm, residual, cond, noise, span))
            and cfg.dtype == torch.float32 and self.w0.dtype == torch.bfloat16
            and lm.device.type == "cuda" and lm.device == self.w0.device
            and all(x.device == lm.device and x.is_contiguous() for x in (lm, residual, cond, noise, cfg, span))
        )

    def forward(self, lm, residual, cond, noise, cfg, span):
        return solve(lm, residual, cond, noise, cfg, span,
                     [getattr(self, f"w{i}") for i in range(57)],
                     [getattr(self, f"b{i}") for i in range(9)],
                     [getattr(self, f"n{i}") for i in range(25)], self.cosine, self.sine, self.eps)


def prepare(model) -> PatchSolve | None:
    """Select the measured specialization once, after checkpoint loading."""
    from mstar.platforms import current_platform

    c, d = model.config, model.config.dit
    weight = model.lm_to_dit_proj.weight
    if not (
        weight.device.type == "cuda" and weight.dtype == torch.bfloat16
        and current_platform.arch(weight.device) == "sm_90"
        and "H200" in torch.cuda.get_device_name(weight.device)
        and (c.patch_size, c.feat_dim, c.lm.hidden_size, c.residual.hidden_size) == (4, 64, 2048, 2048)
        and (d.hidden_size, d.intermediate_size, d.num_hidden_layers,
             d.num_attention_heads, d.num_key_value_heads, d.head_dim) == (1024, 4096, 12, 16, 2, 128)
        and not c.dit_mean_mode
    ):
        return None
    if not os.environ.get("CUTLASS_ROOT"):
        logger.info("VoxCPM2 H200 solve unavailable: set CUTLASS_ROOT as documented in installation.rst")
        return None
    load_native()
    logger.info("VoxCPM2: H200 B1 native solve; other shapes retain the standard path")
    return PatchSolve(model)
