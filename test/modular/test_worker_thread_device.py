"""Worker executor threads must be pinned to the worker's CUDA device.

The CUDA current device is per-thread and defaults to 0.
``Worker.__init__`` pins only the main thread; the GPU/plan executor
threads run engine forwards and attention pre-planning. PyTorch ops are
covered by per-tensor device guards, but raw Triton launches (mstar's
fused-MoE and sampling kernels) and bare ``torch.cuda`` stream/sync
calls resolve against the THREAD device — on a worker driving a
non-zero device, an unpinned thread issues that work against device 0,
unordered with the real compute stream (silent corruption or illegal
accesses on the eager fallback path, which launches Triton at request
time).

These tests exercise ``Worker._init_cuda_executor_thread`` exactly as
``Worker.run`` wires it into its executors.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

from mstar.worker.worker import Worker

torch = pytest.importorskip("torch")


def _worker_with_device(device: torch.device) -> Worker:
    w = object.__new__(Worker)
    w.device = device
    return w


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_executor_thread_pinned_to_worker_device():
    w = _worker_with_device(torch.device("cuda", 1))
    ex = ThreadPoolExecutor(
        max_workers=1, initializer=w._init_cuda_executor_thread
    )
    try:
        assert ex.submit(torch.cuda.current_device).result() == 1
    finally:
        ex.shutdown()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_triton_launch_from_pinned_thread_is_ordered():
    """A Triton launch from the pinned executor thread lands on the
    worker device's stream: it must observe a producer queued earlier on
    that stream (the eager-fallback shape of the bug)."""
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    dev1 = torch.device("cuda", 1)
    w = _worker_with_device(dev1)

    with torch.cuda.device(dev1):
        a = torch.zeros((512, 512), dtype=torch.bfloat16, device=dev1)
        out = torch.full((512, 256), -1.0, dtype=torch.bfloat16, device=dev1)
        torch.cuda._sleep(500_000_000)  # keep the stream busy
        a.fill_(1.0)                    # produced late

    ex = ThreadPoolExecutor(
        max_workers=1, initializer=w._init_cuda_executor_thread
    )
    try:
        ex.submit(act_and_mul_triton, a, out, "silu").result()
    finally:
        ex.shutdown()
    torch.cuda.synchronize(dev1)

    expected = torch.nn.functional.silu(torch.tensor(1.0)).item()
    got = out.float().mean().item()
    assert abs(got - expected) < 0.01, (
        f"Triton launch did not order behind the device-1 producer "
        f"(mean {got}, expected ~{expected})"
    )
