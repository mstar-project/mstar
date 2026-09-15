"""The MSTAR_TORCH_PROFILE step window: opens after ``skip`` steps of the chosen graph walk,
covers ``count`` steps, reports once, writes a trace, and stays a no-op otherwise."""
import torch

from mstar.utils.step_profiler import StepProfiler


def test_from_env_parses_spec_and_rank_filter(tmp_path):
    env = {"MSTAR_TORCH_PROFILE": "10:5", "MSTAR_TORCH_PROFILE_DIR": str(tmp_path)}
    p = StepProfiler.from_env("worker_0", env)
    assert (p.skip, p.count, p.graph_walk, p.out_dir) == (10, 5, "decode", str(tmp_path))
    assert StepProfiler.from_env("worker_3", env) is None  # default: rank 0 only
    env["MSTAR_TORCH_PROFILE_RANKS"] = "1,3"
    assert StepProfiler.from_env("worker_3", env) is not None and StepProfiler.from_env("worker_0", env) is None
    env["MSTAR_TORCH_PROFILE_RANKS"] = "all"
    assert StepProfiler.from_env("worker_7", env).tag == "worker_7"
    assert StepProfiler.from_env("worker_0", {"MSTAR_TORCH_PROFILE": "2:3:prefill"}, ).graph_walk == "prefill"
    assert StepProfiler.from_env("worker_0", {}) is None


def test_window_covers_count_steps_after_skip_and_reports_once(tmp_path, caplog):
    prof = StepProfiler(skip=1, count=2, graph_walk="decode", tag="worker_0", out_dir=str(tmp_path))
    x = torch.randn(64, 64)
    with caplog.at_level("INFO"):
        for i, walk in enumerate(["decode", "prefill", "decode", "decode", "decode"]):
            with prof.step(walk):
                x = x @ x * 0.5
            if i < 3:
                assert not prof.done  # step 0 skipped, 'prefill' ignored, step 2 is the first profiled
        assert prof.done
        with prof.step("decode"):  # after the window: a no-op
            x = x @ x
    reports = [r.message for r in caplog.records if "steps profiled" in r.message]
    assert len(reports) == 1 and "2 decode steps profiled" in reports[0] and "aten::" in reports[0]
    assert (tmp_path / "mstar_worker_0_decode_trace.json").exists()
