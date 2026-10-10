"""Piecewise regions of one node capture into one shared graph memory pool."""

import inspect

import pytest
import torch

from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner


def test_runner_accepts_a_shared_pool_and_keeps_it():
    assert "memory_pool" in inspect.signature(PiecewiseCudaGraphRunner.__init__).parameters
    runner = object.__new__(PiecewiseCudaGraphRunner)
    runner._memory_pool = None
    # a runner built without a pool allocates its own at capture time (CUDA only)
    src = inspect.getsource(PiecewiseCudaGraphRunner.warmup_and_capture)
    assert "if self._memory_pool is None" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph pools need CUDA")
def test_two_runners_share_the_pool_handle():
    from unittest import mock

    pool = torch.cuda.graphs.graph_pool_handle()
    make = lambda: PiecewiseCudaGraphRunner(  # noqa: E731
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

    from mstar.engine import cuda_graph_runner as cgr

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


def test_run_without_a_graph_runs_the_region_eagerly_through_its_step():
    """No captured graph for the shape (here an eager-only region, ``[]`` batch sizes):
    ``run`` declares, admits, plans and commits the region's step over the real rows,
    calls the region on the inputs as given, and returns its output."""
    from unittest import mock

    from mstar.engine.cuda_graph_config import PiecewiseBatchedConfig
    from mstar.engine.resources.step import ADMIT_OK

    seen = {}

    def region(call):
        seen["ids"] = call.engine_inputs.request_ids
        return {"y": call.static_inputs["x"] * 2}

    def declare(request_ids, seq_lens):
        seen["declared"] = (list(request_ids), list(seq_lens))
        return mock.Mock()

    step_runner = mock.Mock()
    step_runner.admit.return_value = ADMIT_OK
    runner = PiecewiseCudaGraphRunner(
        label="r", config=PiecewiseBatchedConfig(
            capture_fn=region, make_static_inputs=lambda shape: {}, declare_step=declare, seq_len=4,
            capture_batch_sizes=[], eager_fallback=True,
        ),
        resources={}, step_runner=step_runner, device=torch.device("cpu"), autocast_dtype=None, num_slots=1,
    )
    assert not runner.can_run(3)
    out = runner.run(static_inputs={"x": torch.ones(3, 4)}, request_ids=[7, 8, 9], seq_lens=[2, 2, 2])
    assert torch.equal(out.get_view("y"), torch.full((3, 4), 2.0))
    assert seen == {"declared": ([7, 8, 9], [2, 2, 2]), "ids": [7, 8, 9]}
    step = step_runner.admit.call_args.args[0]
    ctx = step.set_ctx.call_args.args[0]
    assert ctx.slot_lease is None and not ctx.capture and tuple(ctx.request_ids) == (7, 8, 9)
    assert step_runner.plan.called and step_runner.commit.called


def test_run_without_a_graph_raises_unless_the_region_falls_back():
    from unittest import mock

    from mstar.engine.cuda_graph_config import PiecewiseBatchedConfig

    runner = PiecewiseCudaGraphRunner(
        label="r", config=PiecewiseBatchedConfig(
            capture_fn=lambda call: {}, make_static_inputs=lambda shape: {}, seq_len=4, capture_batch_sizes=[],
        ),
        resources={}, step_runner=mock.Mock(), device=torch.device("cpu"), autocast_dtype=None, num_slots=1,
    )
    with pytest.raises(RuntimeError, match="no captured graph"):
        runner.run(static_inputs={}, request_ids=[1])
