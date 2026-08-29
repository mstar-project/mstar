"""One GPU, seconds: a foreign thread's CUDA calls must not break a capture.

Reproduces the 2026-08-20 failure shape directly. The glm52 load-liveness
heartbeat is a thread doing ``torch.mm`` on its own stream; under the default
``cudaStreamCaptureModeGlobal`` any "potentially unsafe" CUDA call from ANY
thread (the allocator's event queries, a ``synchronize``) fails and
invalidates whatever capture is open in the process. The runner now captures
in ``thread_local`` mode, which polices only the capturing thread.

The foreign thread here does what the heartbeat does — ``torch.mm`` on its own
non-blocking stream — plus a ``stream.synchronize()`` on that stream, a
"potentially unsafe" call that global-mode capture forbids from every thread
and thread_local mode allows from every thread but the capturing one. (A
``torch.cuda.synchronize()`` is NOT a valid stand-in: a device-wide sync
touches the legacy stream and is illegal during any capture, in any mode.)
With ``global`` the capture dies; with ``thread_local`` it records and replays
the right numbers and the foreign thread sees no error.
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
        with torch.cuda.stream(s):
            torch.mm(a, a)  # cuBLAS workspace / handle for this stream, outside the race
        s.synchronize()
        while not stop.is_set():
            try:
                with torch.cuda.stream(s):
                    torch.mm(a, a)
                s.synchronize()
            except BaseException as e:  # noqa: BLE001 — recorded, asserted below
                errors.append(e)

    # Contraction x = 0.5*I: y <- y@x + 1 converges to 2.0 from any start, so a
    # 2000-iteration captured body stays finite and its replay is checkable.
    x = torch.eye(1024, device=device) * 0.5
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
        return y
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
    # replayed with the static input rewritten to 0.5: the fixed point is 2.0
    assert torch.allclose(out, torch.full_like(out, 2.0)), out.flatten()[:4]


def test_global_mode_is_broken_by_foreign_thread_cuda_calls():
    """Documents WHY thread_local: the same capture under global mode dies.
    If this ever passes, the capture got short enough to win the race —
    lengthen it, do not delete the test."""
    with pytest.raises((RuntimeError, AssertionError)):
        _capture_under_foreign_syncs("global")
