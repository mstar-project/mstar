"""The LoRA Triton kernels against cuBLAS, per layer of Phi-4-mini's projections.

Times ``mstar::lora_apply_`` (the Triton kernels) and the same delta in cuBLAS at each rank,
row count and adapter count, over Phi-4-mini's four LoRA targets (qkv, o, gate_up, down), and
prints a table per adapter count with their sum: one decoder layer's LoRA. With ``--adapters 8``
the rows are spread evenly over 8 slots, and cuBLAS runs one shrink and one expand per slot.

Each timing replays a CUDA graph of the call over enough copies of the adapter weights to
exceed twice the L2 cache, so the weights come from HBM, as in a decode step where every layer
has its own. Rows and outputs are shared between copies.

    python -m benchmark.lora.kernels                      # 32/64/128/2048 rows, ranks 16-320, 1 and 8 adapters
    python -m benchmark.lora.kernels --rows 32 --ranks 256,320 --adapters 1
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from typing import NamedTuple

import torch

from benchmark.kernels.timing import time_graph

# Phi-4-mini (and Phi-4-multimodal's language model): target -> (in features, output slices)
PHI4_MINI = {
    "qkv_proj": (3072, (3072, 1024, 1024)),
    "o_proj": (3072, (3072,)),
    "gate_up_proj": (3072, (8192, 8192)),
    "down_proj": (8192, (3072,)),
}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}



class Case(NamedTuple):
    """One target's rows and slot layout; ``bounds`` is ``slot_indptr`` on the host."""
    x: torch.Tensor
    y: torch.Tensor
    token_order: torch.Tensor
    slot_indptr: torch.Tensor
    ranks: torch.Tensor
    scalings: torch.Tensor
    out_slices: tuple[int, ...]
    bounds: list[int]
    scale: float


Apply = Callable[[Case, torch.Tensor, torch.Tensor], None]


def triton_apply(c: Case, a: torch.Tensor, b: torch.Tensor) -> None:
    torch.ops.mstar.lora_apply_(c.y, c.x, a, b, c.token_order, c.slot_indptr, c.ranks, c.scalings,
                                list(c.out_slices))


def cublas_apply(c: Case, a: torch.Tensor, b: torch.Tensor) -> None:
    """The same delta as one shrink GEMM and one expand GEMM per slice, per adapter slot."""
    rank = a.shape[1] // len(c.out_slices)
    for slot in range(1, len(c.bounds) - 1):
        lo, hi = c.bounds[slot], c.bounds[slot + 1]
        if lo == hi:
            continue
        h = c.x[lo:hi] @ a[slot].T
        col = 0
        for i, width in enumerate(c.out_slices):
            c.y[lo:hi, col:col + width].addmm_(h[:, i * rank:(i + 1) * rank], b[slot, col:col + width].T,
                                               alpha=c.scale)
            col += width


def time_target(apply: Apply, rows: int, rank: int, adapters: int, in_features: int, out_slices,
                dtype: torch.dtype, repeats: int = 3) -> float:
    """Microseconds per call of ``apply`` at one target's shape, the median of ``repeats``."""
    slots = adapters + 1  # slot 0 is the pool's zero adapter
    width = len(out_slices) * rank
    per_copy = slots * (width * in_features + sum(out_slices) * rank) * dtype.itemsize
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    copies = max(1, min(64, -(-2 * l2 // per_copy)))
    weights = [
        (torch.randn(slots, width, in_features, device="cuda", dtype=dtype) * in_features ** -0.5,
         torch.randn(slots, sum(out_slices), rank, device="cuda", dtype=dtype) * rank ** -0.5)
        for _ in range(copies)
    ]
    bounds = [0, 0] + [rows * (s + 1) // adapters for s in range(adapters)]
    case = Case(
        x=torch.randn(rows, in_features, device="cuda", dtype=dtype),
        y=torch.zeros(rows, sum(out_slices), device="cuda", dtype=dtype),
        token_order=torch.arange(rows, device="cuda", dtype=torch.int32),
        slot_indptr=torch.tensor(bounds, device="cuda", dtype=torch.int32),
        ranks=torch.tensor([0] + [rank] * adapters, device="cuda", dtype=torch.int32),
        scalings=torch.tensor([0.0] + [2.0] * adapters, device="cuda"),
        out_slices=tuple(out_slices), bounds=bounds, scale=2.0,
    )

    def launch():
        for a, b in weights:
            apply(case, a, b)

    return statistics.median(time_graph(launch, replays=20) / copies for _ in range(repeats))


def time_layer(apply: Apply, rows: int, rank: int, adapters: int, dtype: torch.dtype,
               targets=PHI4_MINI) -> dict[str, float]:
    """Per-target microseconds, and their sum as ``"layer"``."""
    out = {name: time_target(apply, rows, rank, adapters, k, slices, dtype) for name, (k, slices) in targets.items()}
    out["layer"] = sum(out.values())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", default="32,64,128,2048")
    p.add_argument("--ranks", default="16,64,256,320")
    p.add_argument("--adapters", default="1,8", help="adapters the rows are spread over")
    p.add_argument("--dtype", default="bf16", choices=DTYPES)
    p.add_argument("--per-target", action="store_true", help="also print each target's time")
    args = p.parse_args(argv)
    import mstar.engine.resources.lora.triton_backend  # noqa: F401  (registers the CUDA kernels)

    dtype = DTYPES[args.dtype]
    ints = lambda s: [int(v) for v in s.split(",")]  # noqa: E731
    print(f"{torch.cuda.get_device_name()}, {args.dtype}, torch {torch.__version__}; µs per layer "
          "(Phi-4-mini's qkv + o + gate_up + down)")
    for adapters in ints(args.adapters):
        print(f"\n{adapters} adapter(s)\n")
        print("| rank | rows | Triton | cuBLAS | Triton / cuBLAS |\n| --- | --- | --- | --- | --- |")
        for rank in ints(args.ranks):
            for rows in ints(args.rows):
                t = time_layer(triton_apply, rows, rank, adapters, dtype)
                c = time_layer(cublas_apply, rows, rank, adapters, dtype)
                print(f"| {rank} | {rows} | {t['layer']:.0f} | {c['layer']:.0f} | {t['layer'] / c['layer']:.2f} |")
                if args.per_target:
                    for name in PHI4_MINI:
                        print(f"|   {name} | | {t[name]:.1f} | {c[name]:.1f} | |")


if __name__ == "__main__":
    main()
