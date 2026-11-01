"""A torch.profiler window in a running mstar worker, opened from outside.

Put this directory on the worker's ``PYTHONPATH`` and set ``KTRACE_TRIGGER``
to a path prefix. Once the module of ``KTRACE_TARGET`` (``module:Class.method``,
default the worker's ``_postprocess_batch``, which its main thread runs once a
step) is imported, that method is wrapped: when ``<prefix>.on`` appears, or
is touched again (its content: the window in ms), the wrapper starts a
profiler on that thread (kineto refuses other threads), stops it after the
window, and writes ``<prefix>.<pid>.json`` (a chrome trace; kernels launched
by graph replays included) and ``<prefix>.done.<pid>``. ``KTRACE_SHAPES=1``
also records CPU ops with their input shapes, which eager steps carry and a
replayed graph does not. ``benchmark.kernels.trace.ktrace`` reads the trace.
"""

import os

_PREFIX = os.environ.get("KTRACE_TRIGGER")
_TARGET = os.environ.get("KTRACE_TARGET", "mstar.worker.worker:Worker._postprocess_batch")

if _PREFIX:
    import sys
    import threading
    import time

    _MODULE, _ATTR = _TARGET.split(":")
    _CLASS, _METHOD = _ATTR.split(".")

    def _start():
        from torch.profiler import ProfilerActivity, profile

        if os.environ.get("KTRACE_SHAPES") == "1":
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
        else:
            prof = profile(activities=[ProfilerActivity.CUDA])
        prof.start()
        return prof

    def _install(module) -> None:
        cls = getattr(module, _CLASS)
        inner = getattr(cls, _METHOD)
        state = {"prof": None, "until": 0.0, "start": 0.0, "seen": 0.0, "next_check": 0.0}

        def wrapped(self, *args, **kwargs):
            now = time.monotonic()
            prof = state["prof"]
            if prof is None and now >= state["next_check"]:
                state["next_check"] = now + 0.2
                try:
                    st = os.stat(_PREFIX + ".on")
                except OSError:
                    st = None
                if st is not None and st.st_mtime > state["seen"]:
                    state["seen"] = st.st_mtime
                    try:
                        with open(_PREFIX + ".on") as f:
                            window_ms = float(f.read().strip() or "1500")
                    except (OSError, ValueError):
                        window_ms = 1500.0
                    prof = _start()
                    state.update(prof=prof, start=now, until=now + window_ms / 1000.0)
            out = inner(self, *args, **kwargs)
            if prof is not None and time.monotonic() >= state["until"]:
                state["prof"] = None
                prof.stop()
                pid = os.getpid()
                prof.export_chrome_trace(f"{_PREFIX}.{pid}.json")
                with open(f"{_PREFIX}.done.{pid}", "w") as f:
                    f.write(f"{state['start']:.3f} {time.monotonic():.3f}\n")
            return out

        setattr(cls, _METHOD, wrapped)

    def _wait_for_target() -> None:
        # the worker imports its module some time after interpreter start
        for _ in range(36000):
            module = sys.modules.get(_MODULE)
            if module is not None and hasattr(module, _CLASS):
                _install(module)
                return
            time.sleep(0.1)

    threading.Thread(target=_wait_for_target, name="ktrace-hook", daemon=True).start()
