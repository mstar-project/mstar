"""GPU hardware ceilings for speed-of-light estimates.

``gpu_spec.json`` is a copy of VibeSim's ``gpu/spec.json`` catalog. Conventions
(inherited from that catalog, keep them when editing):
  - TFLOPS are DENSE (no 2:4 sparsity). If a datasheet number is exactly 2x the
    value here, you are reading the sparse figure.
  - ``mem_bandwidth_gbps`` is HBM bandwidth in GB/s.
  - ``interconnect_bandwidth_gbps`` is BIDIRECTIONAL.
  - ``fp32_tflops`` is CUDA-core throughput, not the GEMM roofline.
  - GB200 compute/memory fields are combined for its 2 GPUs.

Resolution is by exact (case-insensitive) match of ``name`` or ``aliases``, with
a substring fallback for ``torch.cuda.get_device_name()`` strings like
"NVIDIA B200". A GPU that cannot be resolved returns ``None`` — callers must
degrade (no speed-of-light column) rather than invent a ceiling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    mem_bandwidth_gbps: float
    bf16_tflops: float
    fp16_tflops: float | None = None
    fp8_tflops: float | None = None
    fp4_tflops: float | None = None
    fp32_tflops: float | None = None
    int8_tops: float | None = None

    def peak_tflops(self, dtype: str) -> float | None:
        """Dense peak for a torch-ish dtype string; bf16 is the fallback."""
        d = dtype.lower().replace("torch.", "")
        if "fp8" in d or "float8" in d or "e4m3" in d or "e5m2" in d:
            return self.fp8_tflops or self.bf16_tflops
        if "fp4" in d or "float4" in d:
            return self.fp4_tflops or self.bf16_tflops
        if d in ("int8", "char"):
            return self.int8_tops or self.bf16_tflops
        if d in ("float16", "half", "fp16"):
            return self.fp16_tflops or self.bf16_tflops
        if d in ("float32", "float", "fp32", "tf32", "float64", "double"):
            return self.fp32_tflops or self.bf16_tflops
        return self.bf16_tflops

    def ridge_flops_per_byte(self, dtype: str = "bfloat16") -> float | None:
        """Arithmetic intensity above which this GPU is compute-bound."""
        peak = self.peak_tflops(dtype)
        if peak is None or not self.mem_bandwidth_gbps:
            return None
        return peak * 1000.0 / self.mem_bandwidth_gbps


_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_spec.json")


def _load_catalog() -> list[dict]:
    with open(_SPEC_PATH) as f:
        return json.load(f)["gpus"]


def resolve_gpu(name: str) -> GpuSpec | None:
    """Resolve a device name (e.g. ``torch.cuda.get_device_name()``) to a spec."""
    want = name.strip().lower()
    entries = _load_catalog()

    def to_spec(g: dict) -> GpuSpec:
        return GpuSpec(
            name=g["name"],
            mem_bandwidth_gbps=g["mem_bandwidth_gbps"],
            bf16_tflops=g.get("bf16_tflops") or g.get("fp16_tflops"),
            fp16_tflops=g.get("fp16_tflops"),
            fp8_tflops=g.get("fp8_tflops"),
            fp4_tflops=g.get("fp4_tflops"),
            fp32_tflops=g.get("fp32_tflops"),
            int8_tops=g.get("int8_tops"),
        )

    for g in entries:
        names = [g["name"], *g.get("aliases", [])]
        if any(want == n.lower() for n in names):
            return to_spec(g)
    # Substring fallback: longest alias contained in the queried name wins, so
    # "NVIDIA B200" resolves to B200-SXM-180GB without an exact alias hit.
    best, best_len = None, 0
    for g in entries:
        for n in [g["name"], *g.get("aliases", [])]:
            nl = n.lower()
            if nl in want and len(nl) > best_len:
                best, best_len = g, len(nl)
    return to_spec(best) if best else None
