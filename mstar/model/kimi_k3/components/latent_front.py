"""Kimi K3.1's latent-MoE front on FlashInfer's generated Kimi-K3 program (``kimi_k3_latent_moe_front``, the
experimental ``cake`` backend: SM100 / SM103 with 148 SMs, TP 1 or 8, up to ``MAX_TOKENS`` rows at decode).

One launch reads the block's packed input weight (``KimiLatentMoE._in_weight``: router | latent down projection |
this rank's shared gate_up) once and writes the router's fp32 logits, the bf16 latent and the shared experts'
bf16 SiTU activation (``4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`` in fp32 on the fp32 products, one rounding).
It replaces the packed fp32 GEMM (cuBLAS split-K and its reduce on B300: 13.0 + 3.2 us at c=1), the cast of its
columns and the SiTU kernel, and runs the shared experts' gate_up in the same weight stream, so the step needs no
side stream for them.

FlashInfer builds each program on first use (one per padded row count: 8, 16, 32, 64, 128) and keeps per-device
scratch; ``warm`` does both at load, outside any capture. A launch allocates nothing and does not synchronize, so
a CUDA graph captures it.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

MAX_TOKENS = 128
# the decode programs' padded row counts (cake_backend.SUPPORTED_N_PAD): warming each builds every program a
# decode step can use
WARM_TOKENS = (8, 16, 32, 64, 128)
EXPERTS, LATENT, HIDDEN, SHARED = 896, 3584, 7168, 6144
SITU = (4.0, 25.0)


def unavailable(weight: torch.Tensor | None, experts: int, latent: int, situ, tp: int) -> str | None:
    """Why the front can't serve a block with this packed input weight, or None where it can."""
    if weight is None:
        return "the block keeps no packed input weight (quantized or mixed-dtype projections)"
    if not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        return f"the packed input weight is {weight.dtype} on {weight.device}, not contiguous bf16 on CUDA"
    if (experts, latent, weight.shape[1]) != (EXPERTS, LATENT, HIDDEN) or tp not in (1, 8):
        return (f"the program is K3's geometry at TP 1 or 8; this block has {experts} experts, a {latent} latent, "
                f"TP {tp}")
    if weight.shape[0] != EXPERTS + LATENT + 2 * (SHARED // tp):
        return f"the packed input weight has {weight.shape[0]} rows, not router + latent + this rank's shared gate_up"
    if (situ.beta, situ.linear_beta) != SITU:
        return f"the program's SiTU is beta 4, linear beta 25; this block's is {situ}"
    try:
        from flashinfer.experimental.kimi_k3_latent_moe import cake_backend
    except ImportError as exc:
        return f"FlashInfer has no Kimi-K3 LatentMoE programs ({exc})"
    try:
        ok = all(cake_backend.generated_program_available(weight.device, "front", tp, n) for n in WARM_TOKENS)
    except Exception as exc:  # noqa: BLE001 - an unsupported GPU raises from the device check
        return f"{type(exc).__name__}: {exc}"
    if not ok:
        return f"FlashInfer registers no front program for {torch.cuda.get_device_name(weight.device)}"
    missing = missing_sources(tp)
    return f"FlashInfer's install ships no sources for its front programs ({missing[0]})" if missing else None


def missing_sources(tp: int = 8) -> list[str]:
    """The front programs' CUDA sources FlashInfer would JIT-build from and its install lacks: its 0.7.2 wheels
    register the programs and ship no sources (flashinfer-ai/flashinfer#4568), so a first call fails to build."""
    from pathlib import Path

    from flashinfer.experimental.kimi_k3_latent_moe import cake_backend, cake_jit

    root = Path(cake_jit.__file__).resolve().parent / "csrc"
    missing = []
    for n in WARM_TOKENS:
        for key in cake_backend.route_kernel_keys("front", tp, n):
            record = cake_jit.MODULES[cake_jit.KERNELS[key]]
            missing += [str(root / s) for s in record["sources"] if not (root / s).is_file()]
    return missing


def _launch(x: torch.Tensor, weight: torch.Tensor, experts: int, latent: int, out_logits, out_latent,
            out_act) -> None:
    """The program's launch into the outputs (the CPU tests put a torch reference here)."""
    from flashinfer.kimi_k3_latent_moe import kimi_k3_latent_moe_front

    kimi_k3_latent_moe_front(x, weight[:experts], weight[experts:experts + latent], weight[experts + latent:],
                             out_logits, out_latent, out_act)


def _shapes(x: torch.Tensor, weight: torch.Tensor, experts: int, latent: int):
    T, i_local = x.shape[0], (weight.shape[0] - experts - latent) // 2
    return (T, experts), (T, latent), (T, i_local)


@torch.library.custom_op("mstar::k31_latent_front", mutates_args=())
def front(x: torch.Tensor, weight: torch.Tensor, experts: int, latent: int
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logits fp32 (T, experts), latent (T, latent), shared act (T, I_local))`` in ``x``'s dtype for contiguous
    ``x`` (T, hidden), ``T <= MAX_TOKENS``, and the block's packed input ``weight`` (router | latent down |
    shared gate then up rows)."""
    s_logits, s_latent, s_act = _shapes(x, weight, experts, latent)
    logits = x.new_empty(s_logits, dtype=torch.float32)
    z, act = x.new_empty(s_latent), x.new_empty(s_act)
    if x.shape[0]:
        _launch(x, weight, experts, latent, logits, z, act)
    return logits, z, act


@front.register_fake
def _(x, weight, experts, latent):
    s_logits, s_latent, s_act = _shapes(x, weight, experts, latent)
    return x.new_empty(s_logits, dtype=torch.float32), x.new_empty(s_latent), x.new_empty(s_act)


def situ_spec():
    """The SiTU the program computes, as ``unavailable`` compares it."""
    from mstar.utils.situ import SiTU

    return SiTU(beta=SITU[0], linear_beta=SITU[1])


_WARMED: set[tuple[int, int]] = set()


def warm(weight: torch.Tensor) -> None:
    """Build and load every decode program and FlashInfer's per-device scratch for this rank's shapes, once per
    device and shape, by a launch at each padded row count."""
    i_local = (weight.shape[0] - EXPERTS - LATENT) // 2
    key = (weight.device.index or 0, i_local)
    if key in _WARMED:
        return
    for n in WARM_TOKENS:
        front(torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=weight.device), weight, EXPERTS, LATENT)
    torch.cuda.synchronize(weight.device)
    _WARMED.add(key)
    logger.info("Kimi-K3 LatentMoE front: FlashInfer's programs for %s rows built (shared I_local %d)",
                "/".join(map(str, WARM_TOKENS)), i_local)
