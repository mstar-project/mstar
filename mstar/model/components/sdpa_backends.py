"""SDPA backend selection for attention whose shapes are not fixed."""

from contextlib import AbstractContextManager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# SDPA sends a dense attn_mask to cuDNN when it can, and cuDNN builds a graph for
# each new shape: about 1 s of host time on an H200. Under capture the shapes
# are fixed and the builds happen once at startup, so only eager calls avoid it.
_ENABLED = {
    SDPBackend.FLASH_ATTENTION: torch._C._get_flash_sdp_enabled,
    SDPBackend.EFFICIENT_ATTENTION: torch._C._get_mem_efficient_sdp_enabled,
    SDPBackend.MATH: torch._C._get_math_sdp_enabled,
    SDPBackend.OVERRIDEABLE: torch._C._get_overrideable_sdp_enabled,
}


def no_cudnn_sdpa() -> AbstractContextManager:
    """Turns ``CUDNN_ATTENTION`` off for the enclosed calls; every other backend
    stays as the process had it."""
    return sdpa_kernel([backend for backend, enabled in _ENABLED.items() if enabled()])
