"""SDPA backend selection for attention whose shapes are not fixed."""

from contextlib import AbstractContextManager

from torch.nn.attention import SDPBackend, sdpa_kernel

# SDPA sends a dense attn_mask to cuDNN when it can, and cuDNN builds a graph for
# each new shape: about 1 s of host time on an H200. Under capture the shapes
# are fixed and the builds happen once at startup, so only eager calls avoid it.
NO_CUDNN_BACKENDS = [
    b for b in SDPBackend.__members__.values() if b not in (SDPBackend.ERROR, SDPBackend.CUDNN_ATTENTION)
]


def no_cudnn_sdpa() -> AbstractContextManager:
    """Enables every SDPA backend except ``CUDNN_ATTENTION`` for the enclosed calls."""
    return sdpa_kernel(NO_CUDNN_BACKENDS)
