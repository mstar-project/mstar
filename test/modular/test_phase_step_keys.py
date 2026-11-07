"""Phase samples recorded while a thread has a step key set are also recorded
under ``<key>.<name>``; other threads and ``unkeyed`` samples are not."""
import threading

from mstar.utils import profiler


def test_a_threads_phases_are_also_recorded_under_its_step_key(monkeypatch):
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    buf = profiler._PhaseBuffer(list)
    monkeypatch.setattr(profiler, "_PHASE_BUF", buf)

    profiler.set_step_key("step.decoder.decode")
    try:
        profiler.phase_record("engine.forward", 1.0)
        laps = profiler.PhaseLaps("engine.")
        laps.lap("plan")
        buf.unkeyed("gpu_exec").append(2.0)
        buf["step.decoder.decode.gpu_exec"].append(3.0)

        other = threading.Thread(target=lambda: profiler.phase_record("engine.forward", 4.0))
        other.start()
        other.join()
    finally:
        profiler.set_step_key(None)
    profiler.phase_record("engine.forward", 5.0)

    assert buf["engine.forward"] == [1.0, 4.0, 5.0]
    assert buf["step.decoder.decode.engine.forward"] == [1.0]
    assert len(buf["step.decoder.decode.engine.plan"]) == 1
    assert len(buf["step.decoder.decode.engine.plan.cpu"]) == 1
    # a sample about another step, and a name already keyed, are left alone
    assert buf["gpu_exec"] == [2.0]
    assert "step.decoder.decode.gpu_exec" in buf and buf["step.decoder.decode.gpu_exec"] == [3.0]
    assert not any(name.startswith("step.decoder.decode.step.") for name in buf)


def test_the_device_idle_before_a_step_is_its_gap():
    """What a phase table needs to tell a step whose host work is exposed from
    one whose GPU work is long: the device's idle time before it."""
    from collections import deque

    from mstar.utils.profiler import _PhaseBuffer
    from mstar.worker.worker import Worker

    class Mark:
        def __init__(self, t):
            self.t = t

        def query(self):
            return True

        def elapsed_time(self, later):
            return later.t - self.t

    worker = Worker.__new__(Worker)
    worker._phase_buf = _PhaseBuffer(list)  # the process-wide one holds what other tests recorded
    worker._gpu_spans = deque([
        (Mark(0.0), Mark(5.0), "step.n.decode"), (Mark(7.0), Mark(15.0), "step.n.mixed"),
    ])
    worker._drain_gpu_spans()
    assert worker._phase_buf["step.n.mixed.gpu_gap"] == [0.002]
    assert "step.n.decode.gpu_gap" not in worker._phase_buf
