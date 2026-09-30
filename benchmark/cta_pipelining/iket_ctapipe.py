"""IKET (in-kernel event tracing) driver for the CuTe DSL CTA-pipelined MLP.

Runs one warm-up and one traced forward of ``CuteCTAPipelinedMLP`` on two GPUs
with the kernels compiled with IKET ranges (``CTA_PIPE_IKET=1``):

    consumer DMA warp : ``wait_row`` (spin on the row counter; payload = row block)
                        ``tma_tile`` (issuing one tile's TMA loads)
    MMA warps         : ``mma_tile``, ``epi_tile``
    producer store warp: ``signal`` (store completion wait + fences + remote red)
    local fc1 (GPU 1)  : the consumer GPU's own share of GEMM 1 (plain role, a third
                        kernel ``CTAPipePlainGemm``), launched before the consumer

Order matters for IKET 4.7.1: the cuBLAS reference is computed *after* the first
pipelined forward. When torch/cuBLAS modules are loaded before the two DSL kernels,
IKET files the second DSL kernel's GPU-1 module instance under an older module entry
and silently skips its launches ("LaunchShouldInstrument Returns false ...
ModuleHasInstrumentedDeviceFunction"), so only the producer shows up in the trace.
The traced (last) forward must run back-to-back after another forward: after >= ~20-50 ms
without GPU0->GPU1 NVLink traffic (here: the cuBLAS reference's first-call init after forward
0), the first peer stores of the next producer launch stall ~200 us on every CTA while the
links wake up (README, "IKET in-kernel profile"). With the default 3 forwards, forward 1 absorbs
that and forward 2 shows steady state, as in the benchmark loop. Kernels must JIT inside the
run-iket process; run-iket runs the target twice, so keep the launch count small:

    export CTA_PIPE_IKET=1 CUTE_DSL_COMPILER_OPT=iket CUTE_DSL_DISABLE_FILE_CACHING=1
    run-iket -o OUT --clobber profile --postprocess json -- \\
        python -m benchmark.cta_pipelining.iket_ctapipe --tokens 8192
    python -m benchmark.cta_pipelining.iket_analyze OUT/*.trace.json
"""

from __future__ import annotations

import argparse
import os

import torch

from benchmark.cta_pipelining.bench_mlp_2gpu import DTYPE, make_weights
from mstar.utils.cta_pipelining import mlp_reference
from mstar.utils.cta_pipelining.cute_mlp import CuteCTAPipelinedMLP


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--k", type=int, default=3072)
    ap.add_argument("--n", type=int, default=14336)
    ap.add_argument("--activation", default="gelu_tanh", choices=["gelu_tanh", "silu", "none"])
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--forwards", type=int, default=3, help="launches per kernel (last one is the one to read)")
    args = ap.parse_args(argv)
    if args.paper:
        args.k, args.n, args.activation, args.no_bias = 8192, 8192, "none", True
    if os.environ.get("CTA_PIPE_IKET") != "1":
        print("warning: CTA_PIPE_IKET != 1, kernels are compiled without IKET ranges")

    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)  # producer, consumer
    w1, b1, w2, b2 = make_weights(args.k, args.n, not args.no_bias, d0)
    x = torch.randn(args.tokens, args.k, device=d0, dtype=DTYPE, generator=torch.Generator(device=d0).manual_seed(1))
    mlp = CuteCTAPipelinedMLP(w1, b1, w2, b2, producer_device=d0, consumer_device=d1, activation=args.activation)
    ref = None
    for i in range(args.forwards):
        with torch.cuda.device(d0):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0 = torch.cuda.current_stream(d0)
        start.record(s0)
        y = mlp(x)
        end.record(s0)
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        if ref is None:  # after the DSL kernels are loaded, so cuBLAS/torch modules cannot confuse IKET's module table
            ref = mlp_reference(x, w1, b1, w2, b2, args.activation)
        err = (y.float() - ref.float()).abs().max().item()
        print(f"forward {i}: {start.elapsed_time(end):.3f} ms  max|err|={err:.4g}", flush=True)


if __name__ == "__main__":
    main()
