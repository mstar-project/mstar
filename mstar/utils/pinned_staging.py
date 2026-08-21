"""Pinned host staging for the small per-step index tensors a plan sends to the GPU.

Why this exists
---------------
Every attention plan ships a handful of tiny int tensors to the device —
indptrs, page indices, per-request lengths, RoPE positions, scatter maps.
The obvious ways to do that both stall the CPU:

- ``torch.tensor(values, device="cuda")`` and ``dev_buf.copy_(pageable_cpu)``
  are *pageable* host-to-device copies. CUDA performs a stream sync before a
  pageable memcpy, so the call does not return until every kernel already
  queued on the stream has finished. The CPU can never run ahead of the GPU.
- Handing FlashInfer *device* tensors makes its ``plan()`` do ``.to("cpu")``
  on them — a device-to-host copy that drains the stream the same way.

From **pinned** host memory ``copy_(..., non_blocking=True)`` is a real
asynchronous DMA, and FlashInfer's ``.to("cpu")`` becomes a no-op. That is
what lets a decode step queue its whole draft chain (sync pass + k-1
iterations, each with its own plan) without the host waiting on the device
between iterations — the difference between a captured chain iteration
costing ~GPU time and costing GPU time *plus* the host's plan, serialized.

Reuse safety comes for free: pinned tensors are served by PyTorch's caching
host allocator, which records the stream event of every ``non_blocking``
copy that reads a block and never hands the block out again before that
event completes. So a fresh ``pinned(...)`` per plan is safe to drop on the
floor as soon as the copies are enqueued. (Buffers that are *not* torch
tensors — e.g. FlashInfer's own C++ pinned int workspace — need an explicit
event fence; ``FlashInferMLAWrapper.plan`` keeps one.)
"""
from __future__ import annotations

import torch

_PIN = torch.cuda.is_available()


def pinned(values, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """A pinned CPU tensor holding ``values`` (list/tuple or a CPU tensor).

    Falls back to pageable memory when CUDA is not available (CPU-only
    tests), where the semantics are identical and nothing is asynchronous.
    """
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError("pinned() takes host values, got a device tensor")
        src = values.to(dtype)
        if _PIN and src.is_pinned():
            return src
    else:
        src = torch.as_tensor(values, dtype=dtype)
    if not _PIN:
        return src
    out = torch.empty(src.shape, dtype=dtype, pin_memory=True)
    out.copy_(src)
    return out


def to_device_async(
    values, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """``torch.tensor(values, dtype, device)`` without the stream drain: stage
    through pinned memory and copy ``non_blocking``."""
    host = pinned(values, dtype)
    if host.device == device:
        return host
    return host.to(device, non_blocking=True)
