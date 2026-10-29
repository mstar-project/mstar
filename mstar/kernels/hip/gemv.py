"""The HIP decode GEMV (``decode_gemv.cu``): a weight stream for up to 16 rows
on AMD GPUs, with the same signatures as the Triton kernels in
``mstar.kernels.triton.gemv``. Knobs: ``ytile`` output columns per wave,
``unrl`` 16-byte loads per column in flight, ``waves`` per workgroup (1-4) and
``groups`` workgroups per CU. Built on first use (``JitExtension``)."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mstar.kernels.hip.build import JitExtension

DEFAULT_KNOBS: Mapping[str, int] = {"ytile": 2, "unrl": 2, "waves": 4, "groups": 2}
# the (ytile, unrl) pairs decode_gemv.cu instantiates
TILES = ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2))

EXTENSION = JitExtension(
    "_mstar_hip_decode_gemv", [Path(__file__).resolve().parent / "decode_gemv.cu"], extra_cuda_cflags=["-O3"],
)


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _ops():
    EXTENSION.load()
    return torch.ops.mstar_hip_gemv


def _launch_args(x: torch.Tensor, knobs: Mapping[str, Any]) -> tuple[int, int, int, int]:
    groups = int(knobs["groups"]) * _cu_count(x.device.index or 0)
    return int(knobs["ytile"]), int(knobs["unrl"]), int(knobs["waves"]), groups


def decode_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    knobs: Mapping[str, Any] = DEFAULT_KNOBS,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` for bf16 x ``[rows <= 16, K]``, weight ``[N, K]``."""
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    return _ops().decode_gemv(x, weight, bias, *_launch_args(x, knobs))


def decode_gemv_silu_mul(
    x: torch.Tensor, weight: torch.Tensor, knobs: Mapping[str, Any] = DEFAULT_KNOBS,
) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` for ``weight = [gate; up]``."""
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    return _ops().decode_gemv_silu_mul(x, weight, *_launch_args(x, knobs))
