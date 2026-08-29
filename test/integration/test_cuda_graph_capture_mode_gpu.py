"""One GPU, seconds: a foreign thread's CUDA calls must not break a capture.

Reproduces the 2026-08-20 failure shape directly. The glm52 load-liveness
heartbeat is a thread doing ``torch.mm`` on its own stream; under the default
``cudaStreamCaptureModeGlobal`` any "potentially unsafe" CUDA call from ANY
thread (the allocator's event queries, a ``synchronize``) fails and
invalidates whatever capture is open in the process. The runner now captures
in ``thread_local`` mode, which polices only the capturing thread.

The foreign thread here hammers ``torch.cuda.synchronize()`` — always unsafe
under a global-mode capture — while the main thread captures a graph long
enough for the race to be certain. With ``global`` the capture dies; with
``thread_local`` it records and replays the right numbers.
"""
import threading

import pytest
import torch

from mstar.engine.cuda_graph_runner import _CAPTURE_ERROR_MODE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)


def _capture_under_foreign_syncs(mode: str) -> torch.Tensor:
    device = torch.device("cuda")
    stop = threading.Event()
    errors: list[BaseException] = []

    def _foreign():
        s = torch.cuda.Stream()
        a = torch.ones(2048, 2048, device=device, dtype=torch.bfloat16)
        while not stop.is_set():
            try:
                with torch.cuda.stream(s):
                    torch.mm(a, a)
                torch.cuda.synchronize()
            except BaseException as e:  # noqa: BLE001 — recorded, asserted below
                errors.append(e)

    x = torch.ones(1024, 1024, device=device)
    static_in = torch.zeros_like(x)
    t = threading.Thread(target=_foreign, daemon=True)
    t.start()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm-up outside the capture
                y = static_in @ x
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode=mode):
            y = static_in
            for _ in range(2000):  # a long enough capture for the race to be certain
                y = y @ x + 1.0
        static_in.copy_(torch.full_like(x, 0.5))
        g.replay()
        torch.cuda.synchronize()
        return y.sum()
    finally:
        stop.set()
        t.join(timeout=10)
        # Under thread_local the foreign thread must have seen no errors at all.
        if mode == "thread_local":
            assert not errors, f"foreign thread hit {errors[:1]}"


def test_runner_captures_in_thread_local_mode():
    assert _CAPTURE_ERROR_MODE == "thread_local"


def test_capture_survives_foreign_thread_cuda_calls():
    out = _capture_under_foreign_syncs("thread_local")
    assert torch.isfinite(out).all()


def test_global_mode_is_broken_by_foreign_thread_cuda_calls():
    """Documents WHY thread_local: the same capture under global mode dies.
    If this ever passes, the capture got short enough to win the race —
    lengthen it, do not delete the test."""
    with pytest.raises(RuntimeError):
        _capture_under_foreign_syncs("global")
