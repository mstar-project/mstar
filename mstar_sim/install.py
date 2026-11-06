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
    from mstar_sim import waits

    waits.install()  # every clock and wait of the process, before anything binds them
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

    threading.Thread(target=wait, name="mstar_sim-thread-dump", daemon=True).start()  # (not counted: its own)


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
    """At exit, append one JSON line per process: its device's busy time (if it drove one), and
    how its clock was made (``mstar_sim.clock``, ``mstar_sim.waits``). ``clock_s`` is the time
    mstar saw pass; ``wall_s`` the box's. ``device_waited_s`` near zero means the run was
    host-bound (mstar's own CPU work, the emulator's is not in it). ``slept_s`` is device waits
    that ended in real time because another thread ran, with ``jump_blockers`` the call sites
    that ran; ``late_resumes`` those ended over a millisecond late; ``thread_time_faster`` reads
    of ``time.thread_time`` that ran ahead of the clock (an op that released the GIL inside the
    emulator); ``run_delay_s`` the emulator's time the box descheduled; ``ahead_stalls`` the times
    a thread waited for the deployment's bound; ``late_deliveries`` messages polled within 1 ms
    of a move past W, so maybe moved past; ``meet_skew_s`` the widest gap between ranks' streams
    at a collective; ``parks`` the times the process counted as blocked with threads in waits
    unknown to the emulator, and ``unparks_by_look`` those a look ended, its work having run
    hidden for up to that look; ``gate_waits`` looks that found its own tasks still on a CPU."""
    import atexit
    import json

    from mstar_sim import clock, waits

    start, real = clock.now(), clock._real()

    def write():
        if clock.FAST and clock.DEFAULT.kernels == 0:
            return  # nothing to say of a process that drove no GPU on the real clock
        import multiprocessing

        role = (multiprocessing.current_process().name if "spawn_main" in " ".join(sys.orig_argv)
                else " ".join(a for a in sys.orig_argv[1:4] if not a.startswith("/"))[:80])
        row = {
            "pid": os.getpid(), "process": role, "gpu": os.environ.get("MSTAR_SIM_GPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_s": round(clock._real() - real, 3), "clock_s": round(clock.now() - start, 3),
            "device_busy_s": round(clock.DEFAULT.busy, 3),
            "kernels": clock.DEFAULT.kernels,
            "graph_replays": clock.STATS["replays"],
            # torch calls on cuda tensors outside graphs: per replay, a decode step's eager work
            "eager_calls": clock.STATS["calls"],
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in clock.STATS.items()
               if k not in ("replays", "calls")},
            "slept_s": round(waits.REPORT["slept_s"], 3), "late_resumes": waits.REPORT["late_resumes"],
            "ahead_stalls": waits.REPORT["ahead_stalls"], "late_deliveries": waits.REPORT["late_deliveries"],
            "meet_skew_s": round(waits.REPORT["meet_skew_s"], 6), "mover_s": round(waits.REPORT["mover_s"], 3),
            "scan_s": round(waits.REPORT["scan_s"], 3), "scans": waits.REPORT["scans"],
            "settle_deferrals": waits.REPORT["settle_deferrals"], "gate_waits": waits.REPORT["gate_waits"],
            "parks": waits.REPORT["parks"], "unparks_by_look": waits.REPORT["unparks_by_look"],
            "jump_blockers": dict(sorted(waits.REPORT["jump_blockers"].items(), key=lambda kv: -kv[1])[:8]),
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    atexit.register(write)
