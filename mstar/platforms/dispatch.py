"""One callable with a different implementation per platform.

A ``PlatformDispatch`` holds a factory per platform and picks one on its first
call, so declaring one at import time loads no kernel library and touches no
device.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from mstar.platforms import current_platform

Factory = Callable[[], Callable[..., Any]]


class PlatformDispatch:
    """Forwards each call to the implementation for ``current_platform``.

    A factory takes no arguments, does its own imports and returns the
    implementation. ``default`` covers platforms without their own factory.
    The first call picks and caches the implementation. After that the
    overhead is one attribute read, and none inside a captured graph.
    """

    def __init__(
        self,
        name: str,
        *,
        cuda: Factory | None = None,
        rocm: Factory | None = None,
        xpu: Factory | None = None,
        cpu: Factory | None = None,
        default: Factory | None = None,
    ):
        self.name = name
        self._factories = {"cuda": cuda, "rocm": rocm, "xpu": xpu, "cpu": cpu}
        self._default = default
        self._impl: Callable[..., Any] | None = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        impl = self._impl
        if impl is None:
            impl = self._resolve()
        return impl(*args, **kwargs)

    def _resolve(self) -> Callable[..., Any]:
        with self._lock:
            if self._impl is None:
                platform = current_platform.name
                factory = self._factories.get(platform) or self._default
                if factory is None:
                    raise NotImplementedError(
                        f"{self.name} has no implementation for platform {platform}"
                    )
                self._impl = factory()
            return self._impl

    def reset(self) -> None:
        """Drop the cached implementation so the next call picks again."""
        with self._lock:
            self._impl = None

    def __repr__(self) -> str:
        return f"PlatformDispatch({self.name!r})"
