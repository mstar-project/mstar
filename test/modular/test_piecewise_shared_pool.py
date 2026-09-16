"""Piecewise regions of one node capture into one shared graph memory pool."""

import inspect

import pytest
import torch

from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner


def test_runner_accepts_a_shared_pool_and_keeps_it():
    assert "memory_pool" in inspect.signature(PiecewiseAcceleratorGraphRunner.__init__).parameters
    runner = object.__new__(PiecewiseAcceleratorGraphRunner)
    runner._memory_pool = None
    # a runner built without a pool allocates its own at capture time (CUDA only)
    src = inspect.getsource(PiecewiseAcceleratorGraphRunner.warmup_and_capture)
    assert "if self._memory_pool is None" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph pools need CUDA")
def test_two_runners_share_the_pool_handle():
    from unittest import mock

    pool = torch.cuda.graphs.graph_pool_handle()
    make = lambda: PiecewiseAcceleratorGraphRunner(  # noqa: E731
        label="r", config=mock.Mock(capture_batch_sizes=[1], declare_step=None), resources={},
        step_runner=mock.Mock(), device=torch.device("cuda"), autocast_dtype=None, num_slots=1, memory_pool=pool,
    )
    a, b = make(), make()
    assert a._memory_pool is pool and b._memory_pool is pool


def test_region_outputs_are_copied_into_buffers_outside_the_pool(monkeypatch):
    """The captured callable's own tensors stay behind in the pool; what the
    runner keeps (and ``get_view`` aliases) is a copy allocated before the
    capture, so another graph's replay can never land on it."""
    import torch

    from mstar.engine import accelerator_graph_runner as cgr

    produced = {}

    def run():
        produced["x"] = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        produced["y"] = torch.ones(2, dtype=torch.int64)
        return dict(produced)

    captured = {}

    def fake_capture(fn, pool, device, autocast_dtype):
        captured["pool"] = pool
        return "graph", fn()

    monkeypatch.setattr(cgr, "capture_into_graph", fake_capture)
    warm = run()
    graph, static = cgr.capture_with_static_outputs(run, warm, pool="shared", device="cpu", autocast_dtype=None)

    assert graph == "graph" and captured["pool"] == "shared"
    assert set(static) == {"x", "y"}
    for name, value in static.items():
        assert torch.equal(value, produced[name])
        assert value.data_ptr() != produced[name].data_ptr()
        assert value.data_ptr() != warm[name].data_ptr()
