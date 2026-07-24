"""CPU-only tests for MSTAR_BURST_CAP thread-cap coordination.

Runs each scenario in a FRESH subprocess: torch's thread-pool size is a
process-global set at startup, so an in-process test would leak state between
cases and give false results.
"""
import os
import subprocess
import sys
import textwrap

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(snippet: str, env_extra: dict) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = WT + os.pathsep + env.get("PYTHONPATH", "")
    env.update({k: str(v) for k, v in env_extra.items()})
    out = subprocess.run(
        [PY, "-c", textwrap.dedent(snippet)],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    assert out.returncode == 0, f"child failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip()


def test_off_is_untouched():
    """Flag off: apply returns None and torch threads keep the default (>1 on
    a multicore box) — i.e. we did not touch anything."""
    out = _run(
        """
        import torch
        from mstar.utils.burst_cap import apply_process_thread_cap, enabled
        default = torch.get_num_threads()
        r = apply_process_thread_cap("worker")
        print(enabled(), r, default, torch.get_num_threads())
        """,
        {"MSTAR_BURST_CAP": "0"},
    )
    en, ret, before, after = out.split()
    assert en == "False"
    assert ret == "None"
    assert before == after  # threads unchanged


def test_on_caps_threads_and_env():
    """Flag on: torch intra-op threads == budget and native env exported."""
    out = _run(
        """
        import os, torch
        from mstar.utils.burst_cap import apply_process_thread_cap
        r = apply_process_thread_cap("worker")
        print(r, torch.get_num_threads(), os.environ["OMP_NUM_THREADS"],
              os.environ["MKL_NUM_THREADS"])
        """,
        {"MSTAR_BURST_CAP": "1", "MSTAR_BURST_THREADS": "6"},
    )
    ret, nthreads, omp, mkl = out.split()
    assert ret == "6"
    assert nthreads == "6"
    assert omp == "6"
    assert mkl == "6"


def test_per_role_override_wins():
    """MSTAR_BURST_THREADS_<ROLE> overrides the global budget for that role."""
    out = _run(
        """
        import torch
        from mstar.utils.burst_cap import apply_process_thread_cap
        print(apply_process_thread_cap("worker"), torch.get_num_threads())
        """,
        {"MSTAR_BURST_CAP": "1", "MSTAR_BURST_THREADS": "8",
         "MSTAR_BURST_THREADS_WORKER": "3"},
    )
    ret, nthreads = out.split()
    assert ret == "3"
    assert nthreads == "3"


def test_capped_workers_only_narrows():
    """capped_workers never widens an existing pool and no-ops when off."""
    out = _run(
        """
        from mstar.utils.burst_cap import capped_workers
        # off -> unchanged
        import os
        print(capped_workers(3, "worker"))
        """,
        {"MSTAR_BURST_CAP": "0"},
    )
    assert out == "3"
    out = _run(
        """
        from mstar.utils.burst_cap import capped_workers
        # budget 8, pool default 3 -> stays 3 (never widened)
        print(capped_workers(3, "worker"))
        """,
        {"MSTAR_BURST_CAP": "1", "MSTAR_BURST_THREADS": "8"},
    )
    assert out == "3"
    out = _run(
        """
        from mstar.utils.burst_cap import capped_workers
        # budget 2, pool default 3 -> narrowed to 2
        print(capped_workers(3, "worker"))
        """,
        {"MSTAR_BURST_CAP": "1", "MSTAR_BURST_THREADS": "2"},
    )
    assert out == "2"


def test_preprocess_result_identical_across_cap():
    """The load-bearing correctness guarantee: capping threads changes only how
    many cores a CPU op uses, never the RESULT. Run the same torchvision resize
    (the preprocess hot op) with the cap off and on; bytes must match."""
    snippet = """
        import torch
        from mstar.utils.burst_cap import apply_process_thread_cap
        apply_process_thread_cap("api_server")
        import torchvision.transforms.functional as F
        g = torch.Generator().manual_seed(0)
        img = torch.rand(3, 512, 384, generator=g)
        out = F.resize(img, (256, 256),
                       interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        import hashlib
        print(hashlib.sha256(out.numpy().tobytes()).hexdigest())
    """
    h_off = _run(snippet, {"MSTAR_BURST_CAP": "0"})
    h_on = _run(snippet, {"MSTAR_BURST_CAP": "1", "MSTAR_BURST_THREADS": "2"})
    assert h_off == h_on, f"resize result differs under cap: {h_off} vs {h_on}"


if __name__ == "__main__":
    test_off_is_untouched()
    test_on_caps_threads_and_env()
    test_per_role_override_wins()
    test_capped_workers_only_narrows()
    test_preprocess_result_identical_across_cap()
    print("ALL BURST_CAP TESTS PASSED")
