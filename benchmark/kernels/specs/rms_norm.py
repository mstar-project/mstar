"""``mstar::rms_norm``: the shipped op against torch, Inductor's fused codegen,
and Triton and HIP kernels over their knob spaces."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from benchmark.kernels import isa
from benchmark.kernels.candidates import Candidate, is_amd, search_order, space
from benchmark.kernels.spec import KernelSpec, shape_product
from mstar.kernels.hip.build import JitExtension
from mstar.platforms.facts import DeviceFacts

KERNELS = Path(__file__).resolve().parents[1] / "kernels"

# decode rows (one token per request) up to the largest decode bucket, and
# prompt-sized rows; the hidden sizes of Qwen3.5 0.8B, 2B and 9B
SHAPES = shape_product(rows=(1, 4, 16, 64, 128, 512, 2048), hidden=(1024, 2048, 4096))
TRITON_SPACE = {"num_warps": (1, 2, 4, 8, 16), "rows_per_program": (1, 2, 4)}
TRITON_AMD_SPACE = {**TRITON_SPACE, "waves_per_eu": (0, 1, 2)}
HIP_SPACE = {"threads": (64, 128, 256, 512, 1024), "vec": (1, 2, 4, 8)}


def _bytes(shape, dtype) -> float:
    size = torch.empty((), dtype=dtype).element_size()
    return (2 * shape["rows"] + 1) * shape["hidden"] * size


def _registry():
    return torch.ops.mstar.rms_norm


def _torch():
    from mstar.engine.resources.rms_norm.rms_norm import _rms_norm_reference

    return _rms_norm_reference


def _inductor():
    from mstar.engine.resources.rms_norm.rms_norm import _rms_norm_reference

    return torch.compile(_rms_norm_reference, dynamic=False)


def _triton(knobs):
    from benchmark.kernels.kernels import rms_norm_triton

    compiled = []
    fn = rms_norm_triton.make(
        knobs["num_warps"], knobs["rows_per_program"], knobs.get("waves_per_eu"), record=compiled.append,
    )
    fn.compiled = compiled
    return fn


def _triton_isa(fn, args) -> dict:
    return isa.triton_stats(fn.compiled[-1] if fn.compiled else None)


@functools.cache
def _hip_extension() -> JitExtension:
    extension = JitExtension("_mstar_harness_rms_norm", [KERNELS / "rms_norm_variants.cu"],
                             extra_cuda_cflags=["-O3"])
    extension.load()
    return extension


def _hip(knobs):
    _hip_extension()
    op = torch.ops.mstar_harness_rms_norm.rms_norm_variant
    threads, vec = knobs["threads"], knobs["vec"]

    def rms_norm(x, weight, eps, norm_dtype=None):
        return op(x, weight, eps, threads, vec)

    rms_norm.knobs = knobs
    return rms_norm


@functools.cache
def _library_stats(library: str, arch: str) -> dict:
    # one library holds every variant: unbundle it once
    return isa.hip_library_stats(library, arch)


def _hip_isa(fn, args) -> dict:
    from mstar.platforms import current_platform

    if not current_platform.is_rocm():
        return {"error": "code-object statistics are read on ROCm only"}
    stats = _library_stats(str(_hip_extension().library_path()), current_platform.arch())
    if "kernels" not in stats:
        return stats
    # the mangled name carries the instantiation: rms_norm_kernel<T, THREADS, VEC>
    ints = f"Li{fn.knobs['threads']}ELi{fn.knobs['vec']}E"
    dtype = "bfloat16" if args[0].dtype == torch.bfloat16 else "half"
    for name, kernel in stats["kernels"].items():
        if "rms_norm_kernel" in name and ints in name and dtype in name:
            return {"kernel": name, **kernel}
    return {"error": f"no rms_norm_kernel {ints} {dtype} in the code object"}


def candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    seeds = seeds or {}
    found = [
        Candidate("registry", {}, _registry),
        Candidate("torch", {}, _torch),
        Candidate("inductor", {}, _inductor),
    ]
    triton_points = space(TRITON_AMD_SPACE if is_amd(facts) else TRITON_SPACE, facts)
    for knobs in search_order(triton_points, seeds.get("triton", ()), budget):
        found.append(Candidate("triton", knobs, functools.partial(_triton, knobs), _triton_isa))
    for knobs in search_order(space(HIP_SPACE, facts), seeds.get("hip", ()), budget):
        found.append(Candidate("hip", knobs, functools.partial(_hip, knobs), _hip_isa))
    return found


def spec(shapes=None) -> KernelSpec:
    return KernelSpec.for_op(
        "rms_norm",
        shapes=tuple(shapes or SHAPES),
        dtypes=(torch.bfloat16,),
        bytes_moved=_bytes,
        flops=lambda shape, dtype: 0.0,
        candidates=candidates,
    )
