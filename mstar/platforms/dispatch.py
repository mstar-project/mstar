"""One callable with a different implementation per platform.

A ``PlatformDispatch`` holds a factory per platform and picks one on its first
call, so declaring one at import time loads no kernel library and touches no
device.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from mstar.platforms import current_platform

Factory = Callable[[], Callable[..., Any]]


class PlatformDispatch:
    """Forwards each call to the implementation for ``current_platform``.

    A factory takes no arguments, does its own imports and returns the
    implementation. ``default`` covers platforms without their own factory.
    ``archs`` maps ``"<platform>:<arch>"`` keys (``"rocm:gfx950"``,
    ``"cuda:sm_100"``) to factories that win over their platform's on that
    architecture; the device is only asked for its architecture when the
    platform has such a key.
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
        archs: Mapping[str, Factory] | None = None,
    ):
        self.name = name
        self._factories = {"cuda": cuda, "rocm": rocm, "xpu": xpu, "cpu": cpu}
        self._default = default
        self._archs = dict(archs or {})
        for key in self._archs:
            platform, _, arch = key.partition(":")
            if platform not in self._factories or not arch:
                raise ValueError(f"{name}: {key!r} is not a '<platform>:<arch>' key")
        self._impl: Callable[..., Any] | None = None
        self._lock = threading.Lock()
        _instances.add(self)

    def __call__(self, *args, **kwargs):
        impl = self._impl
        if impl is None:
            impl = self._resolve()
        return impl(*args, **kwargs)

    def selected_key(self) -> str | None:
        """The key of the factory this process uses: ``"<platform>:<arch>"``,
        the platform's name or ``"default"``; None when there is none. Runs no
        factory."""
        platform = current_platform.name
        if any(key.partition(":")[0] == platform for key in self._archs):
            key = f"{platform}:{current_platform.arch()}"
            if key in self._archs:
                return key
        if self._factories.get(platform) is not None:
            return platform
        return "default" if self._default is not None else None

    def resolve(self) -> Callable[..., Any]:
        """The implementation, picked (running its factory) if no call has yet."""
        return self._impl or self._resolve()

    def _resolve(self) -> Callable[..., Any]:
        with self._lock:
            if self._impl is None:
                key = self.selected_key()
                if key is None:
                    raise NotImplementedError(
                        f"{self.name} has no implementation for platform {current_platform.name}"
                    )
                factory = self._default if key == "default" else (
                    self._archs.get(key) or self._factories[key]
                )
                self._impl = factory()
            return self._impl

    @staticmethod
    def reset_all() -> None:
        """Drop every instance's cached implementation, for tests that switch
        the platform."""
        for dispatch in list(_instances):
            dispatch.reset()

    def reset(self) -> None:
        """Drop the cached implementation so the next call picks again."""
        with self._lock:
            self._impl = None

    def __repr__(self) -> str:
        return f"PlatformDispatch({self.name!r})"


_instances: weakref.WeakSet[PlatformDispatch] = weakref.WeakSet()
