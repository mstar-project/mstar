"""Whole-page copies between the device KV cache and the host page pool.

Both caches are laid out ``[layers, pages, ...]``, so a page is one contiguous
block per layer, a layer stride apart. ``copy_`` of that strided view between
device and pinned host memory blocks the host until the copy stream drains and
runs at a fraction of the link's bandwidth. On CUDA and ROCm each run of pages
adjacent in both caches is instead one ``cudaMemcpy2DAsync`` / ``hipMemcpy2DAsync``
with the layer stride as its pitch: async, and run on the copy engines.

`make_page_copier` picks the first copier that loads and passes a round trip,
once per process: on CUDA `CudartCopier`, then `CudaBindingsCopier`, whose calls
drop the GIL; on ROCm `HipCopier`; then `TorchCopier`.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import weakref
from typing import NamedTuple

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


class _Layout(NamedTuple):
    dst: weakref.ref
    src: weakref.ref
    width: int
    height: int
    dst_ptr: int
    dst_pitch: int
    dst_page: int
    src_ptr: int
    src_pitch: int
    src_page: int


class _Memcpy2DCopier(PageCopier):
    """One ``cudaMemcpy2DAsync`` per run of adjacent pages; subclasses supply the call."""

    def __init__(self):
        # Each pair of caches is checked once, keyed by the two tensors' ids. The tensors are held weakly, so
        # this process-wide copier doesn't keep a freed cache alive, and a later tensor with the same id is
        # checked again.
        self._layouts: dict[tuple[int, int], _Layout] = {}

    def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream) -> int:
        raise NotImplementedError

    def _check(self, dst: torch.Tensor, src: torch.Tensor) -> _Layout:
        assert dst[0, 0].is_contiguous() and src[0, 0].is_contiguous()
        assert dst.dtype == src.dtype and dst.shape[0] == src.shape[0]
        assert dst.shape[2:] == src.shape[2:]
        assert dst.stride(1) == dst[0, 0].numel() and src.stride(1) == src[0, 0].numel(), (
            "pages must sit back to back in each layer for a run of adjacent pages to be one copy"
        )
        elt = dst.element_size()
        return _Layout(
            weakref.ref(dst), weakref.ref(src), dst[0, 0].numel() * elt, dst.shape[0],
            dst.data_ptr(), dst.stride(0) * elt, dst.stride(1) * elt,
            src.data_ptr(), src.stride(0) * elt, src.stride(1) * elt,
        )

    def copy(self, dst, dst_pages, src, src_pages, stream):
        layout = self._layouts.get((id(dst), id(src)))
        if layout is None or layout.dst() is not dst or layout.src() is not src:
            layout = self._layouts[id(dst), id(src)] = self._check(dst, src)
        _, _, width, height, dst_ptr, dst_pitch, dst_page, src_ptr, src_pitch, src_page = layout
        runs: list[list[int]] = []
        for dst_idx, src_idx in zip(dst_pages, src_pages, strict=True):
            if runs and dst_idx == runs[-1][0] + runs[-1][2] and src_idx == runs[-1][1] + runs[-1][2]:
                runs[-1][2] += 1
            else:
                runs.append([dst_idx, src_idx, 1])
        handle = stream.cuda_stream
        for dst_idx, src_idx, pages in runs:
            err = self._memcpy_2d(
                dst_ptr + dst_idx * dst_page, dst_pitch,
                src_ptr + src_idx * src_page, src_pitch,
                pages * width, height, handle,
            )
            if err != 0:
                raise RuntimeError(f"cudaMemcpy2DAsync failed with error {err}")


class CudaBindingsCopier(_Memcpy2DCopier):
    """Through ``cuda.bindings``, which Linux CUDA builds of torch 2.10+ require."""

    name = "cuda.bindings"

    def __init__(self):
        from cuda.bindings import runtime

        super().__init__()
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
    _dll = ctypes.CDLL

    def __init__(self):
        super().__init__()
        torch.cuda.init()
        with open("/proc/self/maps") as maps:
            paths = {
                line.split()[-1] for line in maps if f"/{self._lib}.so" in line
            }
        if len(paths) != 1:
            raise OSError(f"expected one {self._lib} mapped by torch, found {sorted(paths)}")
        fn = getattr(self._dll(paths.pop()), self._symbol)
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
    """libcudart, the runtime torch loaded, through ``ctypes.PyDLL`` so each call keeps the GIL. Dropping it
    costs up to a switch interval per call while another thread runs Python. A call that waits on the
    driver stalls every Python thread."""

    name = "libcudart"
    _lib, _symbol = "libcudart", "cudaMemcpy2DAsync"
    _dll = ctypes.PyDLL


class HipCopier(_MappedRuntimeCopier):
    """libamdhip64 on ROCm, where ``copy_`` of a strided page blocks the host too."""

    name = "libamdhip64"
    _lib, _symbol = "libamdhip64", "hipMemcpy2DAsync"


def _round_trips(copier: PageCopier) -> bool:
    """Three strided pages, two of them adjacent, to pinned host memory and back, bit for bit."""
    device = torch.arange(3 * 4 * 6, dtype=torch.float32, device="cuda").view(3, 4, 6)
    host = torch.zeros(3, 3, 6).pin_memory()
    back = torch.zeros_like(device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    copier.copy(host, [0, 1, 2], device, [1, 2, 0], stream)
    copier.copy(back, [2, 3, 1], host, [0, 1, 2], stream)
    stream.synchronize()
    return torch.equal(back[:, 1:], device[:, :3])


@functools.cache
def make_page_copier() -> PageCopier:
    """The fastest copier that works here; `TorchCopier` off CUDA and ROCm."""
    if current_platform.is_cuda():
        candidates = (CudartCopier, CudaBindingsCopier)
    elif current_platform.is_rocm():
        candidates = (HipCopier,)
    else:
        return TorchCopier()
    for cls in candidates:
        try:
            copier = cls()
            if _round_trips(copier):
                logger.info("KV page copies: one 2D async memcpy a run of pages, via %s", cls.name)
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
