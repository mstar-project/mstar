"""Pinned host staging for the small per-step index tensors a plan sends to the GPU."""
from __future__ import annotations

import torch

_PIN = torch.cuda.is_available()


def pinned(values, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """A pinned CPU tensor holding ``values`` (list/tuple or a CPU tensor)."""
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
