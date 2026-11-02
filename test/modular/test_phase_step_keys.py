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
