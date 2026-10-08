"""CUDA stream state a worker has to manage.

What runs on a side stream matters: work on the SMs there (a kernel, or a
device-to-device copy) while a CUDA graph replays slows every graph node in
the process from then on. Side-stream work beside a live graph should be
H2D / D2H copies only -- see mstar.utils.h2d.
"""
from __future__ import annotations

import torch


def reset_device_scheduling(device) -> None:
    """Create and destroy a throwaway CUDA context on ``device``.

    Side-stream kernels or D2D copies beside a graph replay can leave graph
    nodes launching late until the device's scheduling is rebuilt, which
    another context arriving on the GPU does.
    """
    import ctypes

    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(index)
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cu.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_int()
    ctx = ctypes.c_void_p()
    if cu.cuDeviceGet(ctypes.byref(handle), index) == 0 and \
            cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, handle.value) == 0:
        cu.cuCtxDestroy_v2(ctx)
    # back onto torch's primary context for this thread
    torch.cuda.set_device(index)
