"""Unsupported resources reject recipes before capture buffers are allocated."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
)
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner,
    PiecewiseAcceleratorGraphRunner,
    fail_if_graphs_required,
)
from mstar.model.submodule_base import NodeInputs


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
def test_resource_owns_kernel_capture_support(device_type):
    from mstar.engine.resources.attn.cross import FlashInferCrossManager
    from mstar.engine.resources.attn.dense import DenseAttentionManager
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager
    from mstar.engine.resources.linear_attn.gdn import GDNManager
    from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager
    from mstar.engine.resources.sampler.resource import SamplerResource

    if device_type == "cuda":
        for cls in (FlashInferManager, FlashInferRaggedManager,
                    FlashInferCrossManager, GDNManager, Mamba2Manager):
            resource = object.__new__(cls)
            resource._device = torch.device(device_type)
            assert resource.can_capture()
    sampler = object.__new__(SamplerResource)
    sampler._device = torch.device(device_type)
    assert sampler.can_capture() is (device_type != "cpu")
    assert not object.__new__(DenseAttentionManager).can_capture()


@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("can_capture", [False, True])
def test_resource_capability_filters_recipes_and_preserves_strict_mode(
    monkeypatch, caplog, piecewise, can_capture,
):
    resource = SimpleNamespace(can_capture=lambda: can_capture, force_double_buffer=False)
    step = SimpleNamespace(steps={"attention": object()})
    declare = Mock(return_value=step)
    step_runner = Mock()
    if piecewise:
        config = PiecewiseBatchedConfig(
            seq_len=4, capture_batch_sizes=[1, 2], declare_step=declare,
            capture_fn=lambda **kwargs: None, make_static_inputs=lambda shape: {},
        )
        runner = PiecewiseAcceleratorGraphRunner(
            label="blocks", config=config, resources={"attention": resource},
            step_runner=step_runner, device=torch.device("cpu"),
            autocast_dtype=None, num_slots=2,
        )
    else:
        config = BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode", capture_batch_sizes=[1, 2],
            single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
        )
        submodule = SimpleNamespace(
            get_accelerator_graph_configs=lambda *args: [config], declare_step=declare,
        )
        runner = AcceleratorGraphRunner(
            submodule_name="llm", submodule=submodule, resources={"attention": resource},
            step_runner=step_runner, device=torch.device("cpu"), autocast_dtype=None,
            joint_comm_group=SimpleNamespace(world_size=1), num_slots=2,
        )

    specs = runner.prepare_for_capture()
    assert bool(specs) is can_capture
    assert runner.prepare_for_capture() is specs
    if can_capture:
        step_runner.build_accelerator_graph_buffers.assert_called_once()
        missing = runner.dropped_shapes if piecewise else runner.dropped_buckets
        assert not missing
    else:
        step_runner.build_accelerator_graph_buffers.assert_not_called()
        step_runner.ingest_capture_row.assert_not_called()
        assert declare.call_count == 1, "capability is checked once per recipe"
        assert "resources cannot capture: ['attention']" in caplog.text
        missing = runner.dropped_shapes if piecewise else runner.dropped_buckets
        assert len(missing) == 2
        monkeypatch.setenv("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", "1")
        with pytest.raises(RuntimeError, match="accelerator graphs are required"):
            fail_if_graphs_required([str(bucket) for bucket in missing])
