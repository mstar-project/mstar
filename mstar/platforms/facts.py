"""What a GPU is, for kernel heuristics and the kernel harness.

``Platform.device_facts`` reads what torch reports about the device and fills
in the rest (memory bandwidth, peak matrix throughput, XCDs, the memory-side
cache) from the table of known products below. A field nobody knows is None,
so a new product degrades to "no roofline" rather than to a wrong one. Model
code never reads this; which kernel runs stays the resource's choice.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.platforms.interface import Platform

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30


@dataclasses.dataclass(frozen=True)
class DeviceFacts:
    platform: str
    # as torch reports it, e.g. "AMD Instinct MI300X"
    name: str | None = None
    # the name normalized, e.g. "mi300x"; tuned tables are keyed by it, since
    # products of one arch (MI300X, MI325X) share binaries but not timings
    product: str | None = None
    arch: str | None = None
    # CUs on AMD, SMs on NVIDIA
    compute_units: int | None = None
    # dies a grid is spread over, round-robin by workgroup id (AMD)
    xcds: int | None = None
    # LDS (shared memory) per CU or SM
    lds_bytes: int | None = None
    # L2 as one CU sees it: per XCD on AMD
    l2_bytes: int | None = None
    # memory-side cache behind L2 (AMD Infinity Cache); a microbenchmark has to
    # rotate operands past it to measure HBM
    mall_bytes: int | None = None
    memory_bytes: int | None = None
    # peak DRAM bandwidth, GB/s
    hbm_gbps: float | None = None
    # dense matrix peak by dtype ("bf16", "fp16", "fp8", "fp32"), TFLOPS
    peak_tflops: Mapping[str, float] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    warp_size: int | None = None

    def floor_us(self, bytes_moved: float, flops: float = 0.0, dtype: str = "bf16") -> float | None:
        """The roofline floor of a kernel that moves ``bytes_moved`` bytes of
        HBM traffic and does ``flops`` matrix FLOPs: whichever bound is higher,
        in microseconds. None when the bandwidth is unknown, or the throughput
        is and FLOPs were given."""
        if self.hbm_gbps is None:
            return None
        floor = bytes_moved / (self.hbm_gbps * 1e3)
        if flops:
            peak = self.peak_tflops.get(dtype)
            if peak is None:
                return None
            floor = max(floor, flops / (peak * 1e6))
        return floor


@dataclasses.dataclass(frozen=True)
class _Product:
    """Datasheet values for one product; what the device reports wins."""
    pattern: str
    arch: str | None
    compute_units: int | None
    xcds: int | None
    lds_bytes: int | None
    l2_bytes: int | None
    mall_bytes: int | None
    hbm_gbps: float
    peak_tflops: Mapping[str, float]


# Dense peaks from AMD's and NVIDIA's datasheets. The MI300X/MI325X and
# MI350X/MI355X cache and LDS sizes are from AMD's CDNA3/CDNA4 architecture
# pages (rocm.docs.amd.com gpu-arch); an MI210 or MI250 device is one GCD.
_PRODUCTS: tuple[_Product, ...] = (
    _Product(r"^mi300x", "gfx942", 304, 8, 64 * KiB, 4 * MiB, 256 * MiB, 5300.0,
             {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "fp32": 163.4}),
    _Product(r"^mi325x", "gfx942", 304, 8, 64 * KiB, 4 * MiB, 256 * MiB, 6000.0,
             {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "fp32": 163.4}),
    _Product(r"^mi350x", "gfx950", 256, 8, 160 * KiB, 4 * MiB, 256 * MiB, 8000.0,
             {"bf16": 2306.9, "fp16": 2306.9, "fp8": 4613.7, "fp32": 144.2}),
    _Product(r"^mi355x", "gfx950", 256, 8, 160 * KiB, 4 * MiB, 256 * MiB, 8000.0,
             {"bf16": 2516.6, "fp16": 2516.6, "fp8": 5033.2, "fp32": 157.3}),
    _Product(r"^mi210", "gfx90a", 104, 1, 64 * KiB, 8 * MiB, None, 1638.4,
             {"bf16": 181.0, "fp16": 181.0, "fp32": 45.3}),
    _Product(r"^mi250", "gfx90a", 104, 1, 64 * KiB, 8 * MiB, None, 1638.4,
             {"bf16": 181.0, "fp16": 181.0, "fp32": 45.3}),
    _Product(r"^h100.*(sxm|hbm3)", "sm_90", 132, None, 228 * KiB, 50 * MiB, None, 3350.0,
             {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "fp32": 494.7}),
    _Product(r"^h100.*pcie", "sm_90", 114, None, 228 * KiB, 50 * MiB, None, 2000.0,
             {"bf16": 756.5, "fp16": 756.5, "fp8": 1513.0, "fp32": 378.0}),
    _Product(r"^h200", "sm_90", 132, None, 228 * KiB, 50 * MiB, None, 4800.0,
             {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "fp32": 494.7}),
    _Product(r"^a100.*80gb", "sm_80", 108, None, 164 * KiB, 40 * MiB, None, 2039.0,
             {"bf16": 312.0, "fp16": 312.0, "fp32": 156.0}),
)


def normalize_product(name: str | None) -> str | None:
    """``"AMD Instinct MI300X"`` -> ``"mi300x"``, ``"NVIDIA H100 80GB HBM3"`` ->
    ``"h100-80gb-hbm3"``. A virtual function's ``VF`` suffix is dropped: the
    slice runs the same silicon."""
    if not name:
        return None
    product = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    for prefix in ("amd-instinct-", "amd-", "nvidia-"):
        product = product.removeprefix(prefix)
    return product.removesuffix("-vf") or None


def _known(product: str | None, arch: str | None) -> _Product | None:
    if product is None:
        return None
    for known in _PRODUCTS:
        if re.search(known.pattern, product) and (arch is None or known.arch == arch):
            return known
    return None


def facts_for(
    platform: str,
    *,
    name: str | None = None,
    arch: str | None = None,
    compute_units: int | None = None,
    lds_bytes: int | None = None,
    l2_bytes: int | None = None,
    memory_bytes: int | None = None,
    warp_size: int | None = None,
) -> DeviceFacts:
    """Facts from what a device reported, completed from the product table."""
    product = normalize_product(name)
    known = _known(product, arch)
    if known is None:
        return DeviceFacts(
            platform=platform, name=name, product=product, arch=arch, compute_units=compute_units,
            lds_bytes=lds_bytes, l2_bytes=l2_bytes, memory_bytes=memory_bytes, warp_size=warp_size,
        )
    return DeviceFacts(
        platform=platform,
        name=name,
        product=product,
        arch=arch or known.arch,
        compute_units=compute_units or known.compute_units,
        xcds=known.xcds,
        lds_bytes=known.lds_bytes or lds_bytes,
        l2_bytes=l2_bytes or known.l2_bytes,
        mall_bytes=known.mall_bytes,
        memory_bytes=memory_bytes,
        hbm_gbps=known.hbm_gbps,
        peak_tflops=MappingProxyType(dict(known.peak_tflops)),
        warp_size=warp_size,
    )


@functools.cache
def _gpu_facts(platform_name: str, arch: str | None, index: int) -> DeviceFacts:
    props = torch.cuda.get_device_properties(index)
    return facts_for(
        platform_name,
        name=props.name,
        arch=arch,
        compute_units=props.multi_processor_count,
        lds_bytes=getattr(props, "shared_memory_per_multiprocessor", None) or None,
        l2_bytes=getattr(props, "L2_cache_size", None) or None,
        memory_bytes=props.total_memory,
        warp_size=getattr(props, "warp_size", None),
    )


def device_facts(platform: Platform, device=None) -> DeviceFacts:
    """``platform``'s facts about ``device`` (an index, a device, or None for
    the current one), cached per device."""
    if not (platform.is_cuda() or platform.is_rocm()):
        return DeviceFacts(platform=platform.name)
    index = device if isinstance(device, int) else (torch.device(device).index if device is not None else None)
    if index is None:
        index = torch.cuda.current_device()
    return _gpu_facts(platform.name, platform.arch(index), index)
