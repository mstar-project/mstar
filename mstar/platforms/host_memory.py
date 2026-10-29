"""Page-locking host memory that is already mapped, through the device runtime
torch loaded.

A registered range is copied to and from the device directly and
asynchronously; a pageable one is bounced through the runtime's own staging
buffer, and the copy blocks the calling thread. ``Platform.host_register`` is
the entry point.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# cudaErrorHostMemoryAlreadyRegistered; HIP's hipErrorHostMemoryAlreadyRegistered is the same number
_ALREADY_REGISTERED = 712


class HostRegistrar:
    """``<api>HostRegister`` / ``<api>HostUnregister`` of one runtime library.

    Called through ctypes, which releases the GIL for the call, so registering
    a large mapping (tens of ms for 256 MiB) does not stall the process's other
    Python threads. The library is loaded on first use; if it cannot be, that
    is logged once and every call returns False.
    """

    def __init__(self, api: str, load: Callable[[], ctypes.CDLL]):
        self._api = api
        self._load = load
        self._lock = threading.Lock()
        self._loaded = False
        self._register = self._unregister = self._get_last_error = None

    def _bind(self) -> bool:
        with self._lock:
            if self._loaded:
                return self._register is not None
            self._loaded = True
            try:
                lib = self._load()
            except OSError as e:
                logger.warning(
                    "cannot load the %s runtime to page-lock host memory (%s); copies through such memory stay "
                    "pageable", self._api, e)
                return False
            register = getattr(lib, f"{self._api}HostRegister")
            register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            register.restype = ctypes.c_int
            unregister = getattr(lib, f"{self._api}HostUnregister")
            unregister.argtypes = [ctypes.c_void_p]
            unregister.restype = ctypes.c_int
            get_last_error = getattr(lib, f"{self._api}GetLastError")
            get_last_error.restype = ctypes.c_int
            self._register, self._unregister, self._get_last_error = register, unregister, get_last_error
            return True

    def register(self, ptr: int, nbytes: int) -> bool:
        """Register ``[ptr, ptr + nbytes)``; a range already registered counts."""
        if self._register is None and not self._bind():
            return False
        rc = self._register(ptr, nbytes, 0)
        if rc != 0:
            # The error stays set until read, and this thread's next torch
            # kernel launch would raise it (an already-registered range too).
            self._get_last_error()
        ok = rc in (0, _ALREADY_REGISTERED)
        if not ok:
            logger.warning("%sHostRegister(%#x, %d) failed rc=%d; copies through this memory stay pageable",
                           self._api, ptr, nbytes, rc)
        return ok

    def unregister(self, ptr: int) -> bool:
        """Unregister the range ``register`` started at ``ptr``."""
        if self._register is None and not self._bind():
            return False
        rc = self._unregister(ptr)
        if rc != 0:
            self._get_last_error()
        return rc == 0


def _load_cudart() -> ctypes.CDLL:
    if not torch.version.cuda:
        raise OSError("torch is not built with CUDA")
    # by the soname of torch's CUDA major, which reaches the copy torch loaded
    # even where it is not on the loader path (a pip install of torch)
    return ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")


def _load_amdhip() -> ctypes.CDLL:
    if not torch.version.hip:
        raise OSError("torch is not built with ROCm")
    # Only the copy torch already loaded (RTLD_NOLOAD): a ROCm wheel bundles its
    # own, and a second HIP runtime in the process (the system ROCm's, say)
    # would hold registrations torch's copies never consult.
    names = ("libamdhip64.so", f"libamdhip64.so.{torch.version.hip.split('.')[0]}")
    for name in names:
        try:
            return ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue
    raise OSError(f"none of {', '.join(names)} is loaded in this process")


CUDA_HOST = HostRegistrar("cuda", _load_cudart)
HIP_HOST = HostRegistrar("hip", _load_amdhip)
