"""``mstar.engine.side_stream``: a forward's independent branch on a second stream, joined before it is read.
On the CPU the branch runs inline; on a GPU it runs on the side stream, eager and inside a captured graph,
and gives what the same work does on one stream."""
import pytest
import torch

from mstar.engine import side_stream


def test_off_the_gpu_the_branch_runs_inline():
    x = torch.randn(4, 8)
    out, token = side_stream.fork(lambda t: t * 2, x)
    assert token is None
    side_stream.join(token)
    torch.testing.assert_close(out, x * 2, rtol=0, atol=0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _branches(x, w1, w2):
    """Main: a chain of GEMMs; side: another chain reading the same input, joined into the main result."""
    side, token = side_stream.fork(lambda t: torch.tanh(t @ w2) @ w2, x)
    main = torch.relu(x @ w1) @ w1
    side_stream.join(token)
    return main + side


@cuda
def test_a_branch_on_the_side_stream_matches_one_stream_eager_and_captured():
    g = torch.Generator(device="cuda").manual_seed(0)
    x, w1, w2 = (torch.randn(64, 64, device="cuda", generator=g) for _ in range(3))
    want = torch.relu(x @ w1) @ w1 + torch.tanh(x @ w2) @ w2
    torch.testing.assert_close(_branches(x, w1, w2), want, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _branches(x, w1, w2)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _branches(x, w1, w2)
    x.copy_(torch.randn(64, 64, device="cuda", generator=g))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.relu(x @ w1) @ w1 + torch.tanh(x @ w2) @ w2, rtol=0, atol=0)


@cuda
def test_each_main_stream_has_its_own_side_stream():
    """A node capturing on its own stream must not share a side stream with another node's graphs."""
    x = torch.randn(4, 4, device="cuda")
    _, token = side_stream.fork(lambda t: t + 1, x)
    side_stream.join(token)
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        _, token = side_stream.fork(lambda t: t + 1, x)
        side_stream.join(token)
    mains = {main for (dev, main) in side_stream._STREAMS if dev == x.device.index}
    assert {torch.cuda.default_stream().cuda_stream, other.cuda_stream} <= mains
    torch.cuda.synchronize()
