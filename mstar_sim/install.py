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
    from mstar_sim import patches

    patches.install()
    thread_dumps()


def thread_dumps() -> None:
    """``kill -USR1 <pid>`` prints every thread's Python stack to stderr: for a
    hung repro. A thread of its own waits for the signal (blocked everywhere
    else, and inherited blocked by every thread started after), so the dump runs
    even while the main thread sits in a C call."""
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

    threading.Thread(target=wait, name="mstar_sim-thread-dump", daemon=True).start()


def _no_compile(torch) -> None:
    """torch.compile can't trace the emulated tensors; run eagerly (graph capture still emulated)."""

    def compile(model=None, **kwargs):  # noqa: A001
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
            "host_wait_s": round(clock.STATS["host_wait"], 3),
            "emulator_host_s": round(clock.STATS["host_compute"], 3),
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    atexit.register(write)
