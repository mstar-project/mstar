"""The roofline floor of one shape on one device.

A kernel can go no faster than its HBM traffic at the device's peak bandwidth
or its matrix FLOPs at peak throughput, whichever is higher
(``DeviceFacts.floor_us``). Inside a captured graph a launch costs a few
microseconds more whatever it does, which ``bench.node_floor_us`` measures, so
a small decode kernel's honest floor is the larger of the two.
"""

from __future__ import annotations

import dataclasses

import torch

from mstar.platforms.facts import DeviceFacts

_PEAK_KEY = {
    torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32",
    getattr(torch, "float8_e4m3fnuz", None): "fp8", getattr(torch, "float8_e4m3fn", None): "fp8",
}


def peak_key(dtype: torch.dtype) -> str:
    return _PEAK_KEY.get(dtype, str(dtype).removeprefix("torch."))


@dataclasses.dataclass(frozen=True)
class Roofline:
    bytes_moved: float
    flops: float
    # None where the device's peaks are unknown
    floor_us: float | None
    bound: str | None

    def pct_of_floor(self, us: float) -> float | None:
        """How close ``us`` is to the floor, in percent (100 is at the floor)."""
        if self.floor_us is None or us <= 0:
            return None
        return 100.0 * self.floor_us / us


def roofline(bytes_moved: float, flops: float, dtype: torch.dtype, facts: DeviceFacts) -> Roofline:
    floor = facts.floor_us(bytes_moved, flops, peak_key(dtype))
    if floor is None:
        return Roofline(bytes_moved, flops, None, None)
    memory = facts.floor_us(bytes_moved)
    return Roofline(bytes_moved, flops, floor, "memory" if memory is not None and memory >= floor else "compute")
