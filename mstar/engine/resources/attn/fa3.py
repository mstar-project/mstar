"""Whether the FlashAttention-3 wheels import and have a kernel for the GPU.

Every FA3 wheel's probe lives here. ``fa3-fwd`` (``fa3_fwd_interface``) runs the
dense varlen pass; on ROCm, AITER's varlen FMHA stands in for it.
"""

import functools

import torch

from mstar.platforms import current_platform

# imported here, not per `run`: the wheel may be absent, so keep the failure
try:
    from fa3_fwd_interface import flash_attn_varlen_func
except Exception as _fa3_exc:  # noqa: BLE001
    flash_attn_varlen_func = None
    _VARLEN_IMPORT_ERROR: str | None = f"{type(_fa3_exc).__name__}: {_fa3_exc}"
else:
    _VARLEN_IMPORT_ERROR = None

# ROCm has no FA3; AITER's varlen FMHA takes the same call, so dense runs on it
# there instead of degrading to the paged backend
if flash_attn_varlen_func is None and current_platform.is_rocm():
    try:
        from aiter import flash_attn_varlen_func
    except Exception as _aiter_exc:  # noqa: BLE001
        _VARLEN_IMPORT_ERROR = f"{_VARLEN_IMPORT_ERROR}; aiter: {type(_aiter_exc).__name__}: {_aiter_exc}"
    else:
        _VARLEN_IMPORT_ERROR = None


def _fa3_has_kernel(capability: tuple[int, int]) -> bool:
    # fa3-fwd ships sm_90a and sm_80 kernels and no PTX, so a GPU of any other
    # architecture, Blackwell's among them, has no image to launch
    return capability == (9, 0) or capability[0] == 8


@functools.cache
def _fa3_unavailable_reason(device: torch.device | None = None) -> str | None:
    """why no varlen kernel can run on ``device`` (the current GPU if None): none
    imported (fa3-fwd, and AITER on ROCm), or fa3-fwd has none for its capability"""
    if _VARLEN_IMPORT_ERROR is not None:
        return _VARLEN_IMPORT_ERROR
    capability = current_platform.cuda_capability(device)
    if capability is not None and not _fa3_has_kernel(capability):
        return f"fa3-fwd has no kernel for compute capability {capability[0]}.{capability[1]}"
    return None
