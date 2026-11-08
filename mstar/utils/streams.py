"""CUDA stream state a worker has to manage.

What runs on a side stream can matter: work on the SMs there (a kernel, or a
device-to-device copy) while a CUDA graph replays has slowed every graph node
in the process from then on, so side-stream work beside a live graph is H2D /
D2H copies (see mstar.utils.h2d) unless measured. Measured on a B300 with
Qwen3-TTS, whose Codec runs its windows on a side stream beside the Talker's
captured decode (``side_stream`` in its configs), the decode steps took no
longer with the codec there than with it on the chain: per-step GPU time p50
6.10 and 6.16 ms against 6.09 and 6.09 ms at c=8, 4.2 against 5.0 ms at c=1,
two servers per arm (``benchmark/worker_phases``, ``gpu_exec.talker_decode``).
"""
from __future__ import annotations

import torch

from mstar.platforms import current_platform


def reset_device_scheduling(device) -> None:
    """Create and destroy a throwaway CUDA context on ``device``.

    Side-stream kernels or D2D copies beside a graph replay can leave graph
    nodes launching late until the device's scheduling is rebuilt, which
    another context arriving on the GPU does.
    """
    import ctypes

    # goes through the CUDA driver API: NVIDIA only
    if not current_platform.is_cuda():
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
