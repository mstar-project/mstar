"""StreamManager layouts and Fork, from mstar/utils/streams.py."""
import pytest
import torch

from mstar.utils import streams
from mstar.utils.streams import (
    CHECK_STOP,
    DEFAULT,
    FORK,
    KV_OFFLOAD,
    PLAN,
    RECV,
    SEND,
    Fork,
    StreamManager,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_layout_parsing_without_cuda():
    split, aux = StreamManager("split"), StreamManager("aux")
    assert split.physical_name(PLAN) == PLAN
    assert aux.physical_name(PLAN) == aux.physical_name(SEND) == "aux"
    assert aux.physical_name(KV_OFFLOAD) == "offload"
    assert StreamManager("shared").physical_name(KV_OFFLOAD) == "offload"
    explicit = StreamManager("plan=aux, send=aux")
    assert explicit.physical_name(PLAN) == explicit.physical_name(SEND) == "aux"
    assert explicit.physical_name(RECV) == RECV
    for layout in ("split", "aux", "shared", "plan=aux"):
        assert StreamManager(layout).physical_name(DEFAULT) == DEFAULT


@pytest.mark.parametrize("bad", ["default=aux", "plan", "plan=", "=aux"])
def test_bad_layouts_rejected(bad):
    with pytest.raises(ValueError):
        StreamManager(bad)


def test_cpu_device_gets_no_stream():
    assert StreamManager("split").get(PLAN, "cpu") is None


@cuda
def test_roles_share_by_name_not_by_caller():
    m = StreamManager("split")
    # every caller asking for a role gets the same stream: one per role
    assert m.get(PLAN) is m.get(PLAN)
    assert m.get(PLAN) is not m.get(SEND)
    assert m.get(DEFAULT) == torch.cuda.default_stream()
    assert m.num_streams() == 2


@cuda
def test_aux_layout_is_main_aux_and_lazy_offload():
    m = StreamManager("aux")
    assert m.get(PLAN) is m.get(SEND) is m.get(RECV) is m.get(CHECK_STOP)
    assert m.get(PLAN) != torch.cuda.default_stream()
    # nothing has offloaded yet, so there is no offload stream
    assert m.num_streams() == 1
    assert m.get(KV_OFFLOAD) is not m.get(PLAN)
    assert m.num_streams() == 2


def test_default_layout_is_aux(monkeypatch):
    monkeypatch.delenv("MSTAR_STREAMS", raising=False)
    assert StreamManager().physical_name(KV_OFFLOAD) == "offload"
    assert StreamManager().physical_name(PLAN) == "aux"


@cuda
def test_fork_matches_eager_and_captures_two_branches(monkeypatch):
    monkeypatch.setattr(streams, "_MANAGER", StreamManager("split"))
    x = torch.randn(64, 64, device="cuda")
    fork = Fork(enabled=True)

    def branch0():
        return x @ x

    def branch1():
        return (x * 2).sum()

    def step():
        return fork.run(branch0, branch1)

    eager = step()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = step()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured[0], eager[0])
    torch.testing.assert_close(captured[1], eager[1])
    # the branch went through the shared FORK role, not a stream of its own
    assert streams.stream_manager().num_streams() == 1
    assert streams.stream_manager().get(FORK) is not None


@cuda
def test_reset_device_scheduling_keeps_cuda_state_usable():
    from mstar.utils.streams import reset_device_scheduling

    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.arange(8, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.mul(x, 2, out=y)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.mul(x, 2, out=y)

    reset_device_scheduling(dev)

    assert torch.cuda.current_device() == dev.index
    x.add_(1)  # tensors allocated before the reset are still valid
    g.replay()  # and so is a graph captured before it
    torch.cuda.synchronize()
    assert y.tolist() == [2.0 * (i + 1) for i in range(8)]


def test_reset_device_scheduling_is_a_noop_off_cuda():
    from mstar.utils.streams import reset_device_scheduling

    reset_device_scheduling("cpu")
