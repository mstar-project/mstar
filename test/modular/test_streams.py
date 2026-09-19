"""``Fork.run``: in order without a capture (CPU or GPU), and under a CUDA graph capture the two
branches land on two streams with the same result."""
import pytest
import torch

from mstar.utils.streams import Fork


def test_fork_runs_in_order_without_a_capture():
    calls = []
    f = Fork()
    a, b = f.run(lambda: calls.append("a") or 1, lambda: calls.append("b") or 2)
    assert (a, b) == (1, 2) and calls == ["a", "b"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fork_under_capture_matches_the_sequential_result():
    x = torch.randn(64, 256, device="cuda")
    w0 = torch.randn(256, 128, device="cuda")
    w1 = torch.randn(256, 96, device="cuda")
    want = (x @ w0).sum(1) + (x @ w1).sum(1)
    f = Fork()
    out = torch.empty(64, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            a, b = f.run(lambda: x @ w0, lambda: x @ w1)
            out.copy_(a.sum(1) + b.sum(1))
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, want, atol=1e-3, rtol=1e-4)
    assert f._stream is not None  # the capture took the two-stream path
