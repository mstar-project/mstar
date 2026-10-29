"""Build C++/CUDA/HIP sources into a torch extension on first use.

``JitExtension`` wraps ``torch.utils.cpp_extension.load`` with what every
kernel built this way needs:

- The extension is named by a hash of its sources, flags, target
  architectures and torch build, and its sources are staged under the
  extensions root by that hash. cpp_extension's ninja file names sources by
  absolute path, so building each checkout's own copy rebuilt the op whenever
  another worktree booted; staged, every checkout of one source shares one
  build, and a changed source, flag or architecture gets its own.
- On ROCm, CUDA sources are hipified by cpp_extension, and the code objects
  target the architectures named (one ``--offload-arch`` each, so several make
  a fat binary) or else the visible device's, rather than every architecture
  torch was built for. The link points at the HIP runtime this process
  already loaded: torch links ``-lamdhip64`` from ``ROCM_HOME``, and a second,
  different runtime breaks device state.
- A lock left behind by a killed build is removed before building, since
  cpp_extension waits on it forever.
- The build runs once per process, retried once (ranks building together
  race on that sweep), and a failure is kept as the reason the extension is
  unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

# build name -> the loaded module (or True), or the exception that stopped it
_BUILT: dict[str, Any] = {}
_FAILED: dict[str, BaseException] = {}
_BUILD_LOCK = threading.Lock()


def hip_runtime_ldflags(maps_path: str = "/proc/self/maps") -> list[str]:
    """Link flags pointing at the ``libamdhip64`` this process already loaded."""
    try:
        with open(maps_path) as maps:
            for line in maps:
                path = line.rsplit(None, 1)[-1]
                if os.path.basename(path).startswith("libamdhip64.so"):
                    lib_dir = os.path.dirname(path)
                    return [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
    except OSError:
        pass
    return []


def clear_stale_build_lock(build_dir: Path) -> None:
    """Remove a ``FileBaton`` lock no process holds open.

    cpp_extension serialises builds with a lock file and waits on it in an
    untimed loop. A build process killed while holding it (SIGKILL, OOM, a
    preemption) leaves every later process hanging in warmup with no log line.
    The baton keeps the file open for the build's duration, so a lock no
    process has open is abandoned, which mtime alone cannot tell.
    """
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return
    try:
        target = os.path.realpath(lock_path)
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                        return  # a live build holds it
            except OSError:
                continue  # exited, or not ours to read
    except OSError:
        return
    logger.warning(
        "removing the abandoned JIT build lock %s (no process holds it); a build was killed mid-compile",
        lock_path,
    )
    for stale in (lock_path, build_dir / ".ninja_lock"):
        try:
            os.remove(stale)
        except OSError:
            pass


class JitExtension:
    """C++/CUDA/HIP ``sources`` (plus ``headers`` they include by relative
    path) built with ``torch.utils.cpp_extension.load`` on first use.

    ``archs`` names the GPU architectures to build for (``("gfx942",
    "gfx950")`` for one fat binary); None builds for the visible device on
    ROCm and leaves cpp_extension's own choice on CUDA. ``check`` runs after a
    load (with the module, or None) and raises if the build is unusable, e.g.
    by touching an op the source registers.
    """

    def __init__(
        self,
        name: str,
        sources: Sequence[str | os.PathLike],
        *,
        headers: Sequence[str | os.PathLike] = (),
        archs: Sequence[str] | None = None,
        extra_cflags: Sequence[str] = (),
        extra_cuda_cflags: Sequence[str] = (),
        extra_ldflags: Sequence[str] = (),
        is_python_module: bool = False,
        check: Callable[[Any], None] | None = None,
    ):
        self.name = name
        self.sources = [Path(s) for s in sources]
        self.headers = [Path(h) for h in headers]
        self.archs = tuple(archs) if archs else None
        self.extra_cflags = list(extra_cflags)
        self.extra_cuda_cflags = list(extra_cuda_cflags)
        self.extra_ldflags = list(extra_ldflags)
        self.is_python_module = is_python_module
        self.check = check

    def target_archs(self) -> tuple[str, ...]:
        """The architectures the code objects are built for; empty when the
        toolchain picks (CUDA without ``archs``, or no GPU)."""
        if self.archs is not None:
            return self.archs
        if current_platform.is_rocm() or current_platform.is_cuda():
            return (current_platform.arch(),)
        return ()

    def _arch_flags(self) -> list[str]:
        if self.archs is None:
            if not current_platform.is_rocm():
                return []
            archs = self.target_archs()
        else:
            archs = self.archs
        if torch.version.hip is not None:
            return [f"--offload-arch={arch}" for arch in archs]
        flags = []
        for arch in archs:
            sm = arch.removeprefix("sm_")
            flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
        return flags

    def _digest(self) -> str:
        h = hashlib.sha256()
        for path in [*self.sources, *self.headers]:
            h.update(path.name.encode())
            h.update(path.read_bytes())
        toolkit = torch.version.hip or torch.version.cuda or "cpu"
        h.update(json.dumps([
            self.extra_cflags, self.extra_cuda_cflags, self.extra_ldflags, list(self.target_archs()),
            torch.__version__, toolkit,
        ]).encode())
        return h.hexdigest()[:12]

    def build_name(self) -> str:
        """The extension's name: its prefix and the hash of what it is built from."""
        return f"{self.name}_{self._digest()}"

    def stage(self) -> list[str]:
        """Copy the sources and headers under the extensions root, keyed by the
        hash, and return the staged sources' paths. Writes are atomic, so a
        racing rank sees a whole file or none."""
        digest = self._digest()
        # beside the build directories, under TORCH_EXTENSIONS_DIR or its default
        src_dir = self.build_dir().parent / f"{self.name}_src" / digest
        staged = []
        for path in [*self.sources, *self.headers]:
            dest = src_dir / path.name
            if not dest.exists():
                src_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
                tmp.write_bytes(path.read_bytes())
                os.replace(tmp, dest)
            if path in self.sources:
                staged.append(str(dest))
        return staged

    def build_dir(self) -> Path:
        from torch.utils.cpp_extension import _get_build_directory

        return Path(_get_build_directory(self.build_name(), verbose=False))

    def library_path(self) -> Path:
        """The shared library a build writes (where to read its code objects)."""
        return self.build_dir() / f"{self.build_name()}.so"

    def load(self, verbose: bool = False) -> Any:
        """Build if needed and load: the module with ``is_python_module``, else
        the library's path. Runs once per process per build name, and raises
        a RuntimeError caused by the build's error."""
        name = self.build_name()
        with _BUILD_LOCK:
            if name in _BUILT:
                return _BUILT[name]
            if name not in _FAILED:
                try:
                    _BUILT[name] = self._build(name, verbose)
                    return _BUILT[name]
                except Exception as exc:  # noqa: BLE001  kept as the reason
                    _FAILED[name] = exc
            raise RuntimeError(f"extension {name} did not build") from _FAILED[name]

    def _build(self, name: str, verbose: bool) -> Any:
        from torch.utils.cpp_extension import load

        ldflags = list(self.extra_ldflags)
        if torch.version.hip is not None:
            ldflags += hip_runtime_ldflags()
        cuda_cflags = [*self.extra_cuda_cflags, *self._arch_flags()]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                sources = self.stage()
                if attempt == 0:
                    clear_stale_build_lock(self.build_dir())
                else:
                    time.sleep(2.0)
                module = load(
                    name=name,
                    sources=sources,
                    extra_cflags=self.extra_cflags,
                    extra_cuda_cflags=cuda_cflags,
                    extra_ldflags=ldflags,
                    is_python_module=self.is_python_module,
                    verbose=verbose,
                )
                if self.check is not None:
                    self.check(module)
                return module
            except Exception as exc:  # noqa: BLE001  retried once, then raised
                last_error = exc
        assert last_error is not None
        raise last_error

    def unavailable_reason(self) -> str | None:
        """None once the extension loads; else why it cannot (building it if
        that has not been tried in this process)."""
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001
            cause = exc.__cause__ or exc
            return f"{type(cause).__name__}: {cause}"
        return None

    def __repr__(self) -> str:
        return f"JitExtension({self.name!r}, {[s.name for s in self.sources]})"
