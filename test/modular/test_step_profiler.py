"""The MSTAR_TORCH_PROFILE step windows: each opens at its absolute start step of the chosen
graph walk, covers ``count`` steps, reports once with the batch sizes seen, writes a trace, and
the hook is a no-op otherwise."""
import torch

from mstar.utils.step_profiler import StepProfiler


def test_from_env_parses_windows_and_rank_filter(tmp_path):
    env = {"MSTAR_TORCH_PROFILE": "10:5,300:40", "MSTAR_TORCH_PROFILE_DIR": str(tmp_path)}
    p = StepProfiler.from_env("worker_0", env)
    assert (p.windows, p.graph_walk, p.out_dir) == ([(10, 5), (300, 40)], "decode", str(tmp_path))
    assert StepProfiler.from_env("worker_3", env) is None  # default: rank 0 only
    env["MSTAR_TORCH_PROFILE_RANKS"] = "1,3"
    assert StepProfiler.from_env("worker_3", env) is not None and StepProfiler.from_env("worker_0", env) is None
    env["MSTAR_TORCH_PROFILE_RANKS"] = "all"
    assert StepProfiler.from_env("worker_7", env).tag == "worker_7"
    assert StepProfiler.from_env("worker_0", {"MSTAR_TORCH_PROFILE": "2:3", "MSTAR_TORCH_PROFILE_WALK": "prefill"}).graph_walk == "prefill"
    assert StepProfiler.from_env("worker_0", {}) is None


def test_windows_open_at_their_start_steps_and_report_once_each(tmp_path, caplog):
    prof = StepProfiler(windows=[(1, 2), (5, 1)], graph_walk="decode", tag="worker_0", out_dir=str(tmp_path))
    x = torch.randn(64, 64)
    walks = ["decode", "prefill", "decode", "decode", "decode", "decode", "decode", "decode"]
    #         step0     ignored   step1(w1) step2(w1) step3    step4    step5(w2) step6
    with caplog.at_level("INFO"):
        for i, walk in enumerate(walks):
            with prof.step(walk, batch_size=i + 1):
                x = x @ x * 0.5
            assert prof.done == (i >= 6)
        with prof.step("decode", 9):  # after the last window: a no-op
            x = x @ x
    reports = [r.message for r in caplog.records if "steps profiled" in r.message]
    assert len(reports) == 2
    assert "window 1: 2 decode steps profiled (batch sizes 3..4)" in reports[0] and "aten::" in reports[0]
    assert "window 2: 1 decode steps profiled (batch sizes 7..7)" in reports[1]
    assert (tmp_path / "mstar_worker_0_decode_w1_trace.json").exists()
    assert (tmp_path / "mstar_worker_0_decode_w2_trace.json").exists()
