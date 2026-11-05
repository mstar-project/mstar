"""Put this process on the emulated GPU. Idempotent; call before mstar is imported."""

from __future__ import annotations

import os
import sys

_done = False
FAKE_LIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_libs")


def install() -> None:
    global _done
    if _done:
        return
    _done = True
    import signal
    import threading

    if threading.current_thread() is threading.main_thread():
        # before torch starts its native threads: they inherit the mask, so the
        # signal can only reach thread_dumps' waiter
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    # Triton launches go through the interpreter's entry point, which triton_launch prices
    os.environ.setdefault("TRITON_INTERPRET", "1")
    # stand-ins for CUDA-only kernel libraries (flashinfer, ...) shadow the real ones
    if os.path.isdir(FAKE_LIBS) and FAKE_LIBS not in sys.path:
        sys.path.insert(0, FAKE_LIBS)

    import torch

    from mstar_sim import cost, cuda, device

    cost.configure(os.environ.get("MSTAR_SIM_GPU", "h100"), os.environ.get("MSTAR_SIM_PROFILE_DB"))
    if os.environ.get("MSTAR_SIM_TRACE"):
        cost.start_trace(os.environ["MSTAR_SIM_TRACE"])
    cuda.install()
    device.install()
    from mstar_sim import dist

    dist.install()
    if os.environ.get("MSTAR_SIM_REPORT"):
        _report_at_exit(os.environ["MSTAR_SIM_REPORT"])
    _no_compile(torch)
    from mstar_sim import triton_launch

    triton_launch.install()
    from mstar_sim import cpp_extension

    cpp_extension.install()
    from mstar_sim import patches

    patches.install()
    thread_dumps()
    if os.environ.get("MSTAR_SIM_SHM_MB"):
        _small_dev_shm(int(os.environ["MSTAR_SIM_SHM_MB"]))


def thread_dumps() -> None:
    """``kill -USR1 <pid>`` prints every thread's Python stack to stderr and passes
    the signal on to the process's children, so one signal to the server dumps
    its workers too: for a hung repro. A thread of its own waits for the signal
    (blocked everywhere else, and inherited blocked by every thread started
    after), so the dump runs even while the main thread sits in a C call."""
    import signal
    import threading
    import traceback

    if threading.current_thread() is not threading.main_thread():
        return

    def wait():
        while True:
            signal.sigwait({signal.SIGUSR1})
            names = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"mstar_sim: thread dump of pid {os.getpid()}\n"]
            for ident, frame in sys._current_frames().items():
                lines.append(f"--- thread {names.get(ident, ident)}\n")
                lines.extend(traceback.format_stack(frame))
            sys.stderr.write("".join(lines))
            sys.stderr.flush()
            for child in _children(os.getpid()):
                try:
                    os.kill(child, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

    threading.Thread(target=wait, name="mstar_sim-thread-dump", daemon=True).start()


def _small_dev_shm(mib: int) -> None:
    """``--shm-mb N``: ``os.statvfs`` of ``/dev/shm`` answers as an N MiB tmpfs, its free
    space N MiB less what every ``mstar_*`` file in it really holds (a dead run's
    leftovers included; other mstar runs on the box count too). Only what mstar's
    Python reads to size its arena and files changes: writes past N still succeed,
    and the Rust arena's own growth check (``fstatvfs`` on a segment) sees the real
    tmpfs."""
    real = os.statvfs
    total = mib << 20

    def used() -> int:
        held = 0
        for entry in os.scandir("/dev/shm"):
            if entry.name.startswith("mstar_"):
                files = [entry.path] if not entry.is_dir(follow_symlinks=False) else [
                    os.path.join(d, f) for d, _, names in os.walk(entry.path) for f in names]
                for path in files:
                    try:
                        held += os.lstat(path).st_blocks * 512
                    except OSError:
                        pass
        return held

    def statvfs(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).rstrip("/") == "/dev/shm":
            free = max(0, total - used()) // 4096
            return os.statvfs_result((4096, 4096, total // 4096, free, free, 1 << 20, 1 << 20, 1 << 20, 0, 255))
        return real(path, *args, **kwargs)

    os.statvfs = statvfs


def _children(pid: int) -> list[int]:
    out = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                        out.append(int(entry))
            except (OSError, IndexError, ValueError):
                pass
    return out


def _no_compile(torch) -> None:
    """torch.compile can't trace the emulated tensors; run eagerly (graph capture still
    emulated). Its arguments are still checked against the real signature, so a keyword
    this torch doesn't take fails as it would on a GPU."""
    import inspect

    signature = inspect.signature(torch.compile)

    real = torch.compile

    def outside_the_mode(fn):
        """Calls ``fn`` with the emulator's torch-function mode off: Dynamo would
        trace the mode too, and recompile on its state."""
        import functools

        from mstar_sim import device

        @functools.wraps(fn)
        def call(*args, **kw):
            stack = torch.overrides._get_current_function_mode_stack()
            if not stack or type(stack[-1]) is not device.SimMode:
                return fn(*args, **kw)
            with torch.overrides._pop_mode_temporarily():
                return fn(*args, **kw)

        return call

    def compile(model=None, **kwargs):  # noqa: A001
        signature.bind(model, **kwargs)
        if callable(kwargs.get("backend")):
            # a test's own backend (a frame counter, over CPU tensors) compiles for real
            if model is None:
                return lambda fn: compile(fn, **kwargs)
            compiled = real(model, **kwargs)
            if isinstance(compiled, torch.nn.Module):
                compiled.forward = outside_the_mode(compiled.forward)
                return compiled
            return outside_the_mode(compiled)
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = compile


def _report_at_exit(path: str) -> None:
    """At exit, append one JSON line per process: simulated device busy time, the
    host's waits on the device, and the emulator's own host time. Waits near
    zero mean the run was host-bound (mstar's own CPU work plus the emulator's)."""
    import atexit
    import json
    import time

    from mstar_sim import clock

    start = time.perf_counter()

    def write():
        if clock.DEFAULT.kernels == 0:
            return  # a process that drove no GPU (API server, conductor)
        wall = time.perf_counter() - start
        row = {
            "pid": os.getpid(), "gpu": os.environ.get("MSTAR_SIM_GPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_s": round(wall, 3),
            "device_busy_s": round(clock.DEFAULT.busy, 3),
            "kernels": clock.DEFAULT.kernels,
            "graph_replays": clock.STATS["replays"],
            # torch calls on cuda tensors outside graphs: per replay, a decode step's eager work
            "eager_calls": clock.STATS["calls"],
            "host_wait_s": round(clock.STATS["host_wait"], 3),
            "emulator_host_s": round(clock.STATS["host_compute"], 3),
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    atexit.register(write)
