"""Whole-page copies between the device KV cache and the host page pool.

Both caches are laid out ``[layers, pages, ...]``, so a page is one contiguous
block per layer, a layer stride apart. ``copy_`` of that strided view between
device and pinned host memory blocks the host until the copy stream drains and
runs at a fraction of the link's bandwidth. On CUDA and ROCm each page is
instead one ``cudaMemcpy2DAsync`` / ``hipMemcpy2DAsync`` with the layer stride
as its pitch: async, and run on the copy engines.

`make_page_copier` picks the first copier that loads and passes a round trip,
once per process: on CUDA `CudaBindingsCopier`, then `CudartCopier`; on ROCm
`HipCopier`; then `TorchCopier`.
"""

from __future__ import annotations

import ctypes
import functools
import logging

import torch

from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

_MEMCPY_DEFAULT = 4  # cudaMemcpyDefault / hipMemcpyDefault: direction from the pointers


class PageCopier:
    """Copies every layer of ``src_pages[i]`` of ``src`` onto ``dst_pages[i]``
    of ``dst``, queued on ``stream``."""

    name = "base"

    def copy(
        self,
        dst: torch.Tensor,
        dst_pages: list[int],
        src: torch.Tensor,
        src_pages: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        raise NotImplementedError


class TorchCopier(PageCopier):
    """Per-page ``copy_``: works everywhere, but on CUDA blocks the host."""

    name = "torch"

    def copy(self, dst, dst_pages, src, src_pages, stream):
        with torch.cuda.stream(stream):
            for dst_idx, src_idx in zip(dst_pages, src_pages, strict=True):
                dst[:, dst_idx].copy_(src[:, src_idx], non_blocking=True)


class _Memcpy2DCopier(PageCopier):
    """One ``cudaMemcpy2DAsync`` per page; subclasses supply the call."""

    def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream) -> int:
        raise NotImplementedError

    def copy(self, dst, dst_pages, src, src_pages, stream):
        assert dst[0, 0].is_contiguous() and src[0, 0].is_contiguous()
        assert dst.dtype == src.dtype and dst.shape[0] == src.shape[0]
        assert dst.shape[2:] == src.shape[2:]
        elt = dst.element_size()
        width = dst[0, 0].numel() * elt
        dst_pitch, dst_page = dst.stride(0) * elt, dst.stride(1) * elt
        src_pitch, src_page = src.stride(0) * elt, src.stride(1) * elt
        dst_ptr, src_ptr = dst.data_ptr(), src.data_ptr()
        handle = stream.cuda_stream
        for dst_idx, src_idx in zip(dst_pages, src_pages, strict=True):
            err = self._memcpy_2d(
                dst_ptr + dst_idx * dst_page, dst_pitch,
                src_ptr + src_idx * src_page, src_pitch,
                width, dst.shape[0], handle,
            )
            if err != 0:
                raise RuntimeError(f"cudaMemcpy2DAsync failed with error {err}")


class CudaBindingsCopier(_Memcpy2DCopier):
    """Through ``cuda.bindings``, which Linux CUDA builds of torch 2.10+ require."""

    name = "cuda.bindings"

    def __init__(self):
        from cuda.bindings import runtime

        self._runtime = runtime
        self._default = runtime.cudaMemcpyKind.cudaMemcpyDefault

    def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream):
        (err,) = self._runtime.cudaMemcpy2DAsync(
            dst, dpitch, src, spitch, width, height, self._default, stream
        )
        return int(err)


class _MappedRuntimeCopier(_Memcpy2DCopier):
    """Through ctypes, on the runtime library torch itself loaded. Found by its
    mapping rather than by name, so it is never some other toolkit's runtime that
    happens to be on the library path."""

    _lib = _symbol = ""

    def __init__(self):
        torch.cuda.init()
        with open("/proc/self/maps") as maps:
            paths = {
                line.split()[-1] for line in maps if f"/{self._lib}.so" in line
            }
        if len(paths) != 1:
            raise OSError(f"expected one {self._lib} mapped by torch, found {sorted(paths)}")
        fn = getattr(ctypes.CDLL(paths.pop()), self._symbol)
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
        ]
        self._fn = fn

    def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream):
        return self._fn(
            dst, dpitch, src, spitch, width, height, _MEMCPY_DEFAULT, stream
        )


class CudartCopier(_MappedRuntimeCopier):
    """libcudart, for torch 2.9, which ships no ``cuda.bindings``."""

    name = "libcudart"
    _lib, _symbol = "libcudart", "cudaMemcpy2DAsync"


class HipCopier(_MappedRuntimeCopier):
    """libamdhip64 on ROCm, where ``copy_`` of a strided page blocks the host too."""

    name = "libamdhip64"
    _lib, _symbol = "libamdhip64", "hipMemcpy2DAsync"


def _round_trips(copier: PageCopier) -> bool:
    """Two strided pages to pinned host memory and back, bit for bit."""
    device = torch.arange(3 * 4 * 6, dtype=torch.float32, device="cuda").view(3, 4, 6)
    host = torch.zeros(3, 3, 6).pin_memory()
    back = torch.zeros_like(device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    copier.copy(host, [2, 0], device, [1, 3], stream)
    copier.copy(back, [0, 2], host, [2, 0], stream)
    stream.synchronize()
    return torch.equal(back[:, 0], device[:, 1]) and torch.equal(back[:, 2], device[:, 3])


@functools.cache
def make_page_copier() -> PageCopier:
    """The fastest copier that works here; `TorchCopier` off CUDA and ROCm."""
    if current_platform.is_cuda():
        candidates = (CudaBindingsCopier, CudartCopier)
    elif current_platform.is_rocm():
        candidates = (HipCopier,)
    else:
        return TorchCopier()
    for cls in candidates:
        try:
            copier = cls()
            if _round_trips(copier):
                logger.info("KV page copies: one 2D async memcpy a page, via %s", cls.name)
                return copier
            logger.warning("KV page copies: %s returned wrong data; not using it", cls.name)
        except (ImportError, OSError, AttributeError, RuntimeError) as e:
            logger.info("KV page copies: %s unavailable (%s)", cls.name, e)
    logger.warning(
        "KV page copies: no 2D async memcpy binding works here, so offload "
        "and reload fall back to copy_, which blocks the host and runs several "
        "times slower"
    )
    return TorchCopier()
