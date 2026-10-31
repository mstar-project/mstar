
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources.sampler.utils import sample_accelerator_graphable_gpu
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.device import DeviceCapability
from mstar.worker.worker import Worker


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "xpu:1"])
@pytest.mark.parametrize("capability_source", ["early_query", "matching", "other_device"])
def test_engine_load_binds_capability_to_loaded_device(device, capability_source):
    device = torch.device(device)
    supplied = (
        DeviceCapability(device if capability_source == "matching" else torch.device("cpu"))
        if capability_source != "early_query" else None
    )
    engine = Engine(graph_runtime=None, device_capability=supplied)
    if capability_source == "early_query":
        assert engine.device_capability.device == torch.device("cpu")

    engine.load_model(
        {}, [], WorkerParallelGroups(num_workers=1, global_rank=0),
        device, transfer_engine_info=None,
    )

    assert engine.device_capability.device == device
    if supplied is not None and supplied.device == device:
        assert engine.device_capability is supplied
    else:
        assert engine.device_capability is DeviceCapability.of(device)
    assert (engine.device_capability.graph_backend is not None) is (device.type != "cpu")


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_output_copy_uses_generic_stream_on_worker_device(monkeypatch, device_type):
    worker = object.__new__(Worker)
    worker.device = torch.device(device_type, 1)
    worker._d2h_stream = None
    side = MagicMock()
    runtime = SimpleNamespace(
        is_available=lambda: True,
        Stream=Mock(return_value=side),
        Event=Mock(),
        stream=Mock(),
    )
    monkeypatch.setattr(torch, device_type, runtime)
    worker.device_capability = DeviceCapability(worker.device)
    make_stream = Mock(return_value=side)
    monkeypatch.setattr(torch, "Stream", make_stream)
    tensor = SimpleNamespace(
        device=worker.device, dtype=torch.int64, shape=(1,), numel=lambda: 1,
    )
    monkeypatch.setattr(torch, "is_tensor", lambda value: value is tensor)
    cpu_tensor = Mock()
    worker._get_pinned_d2h_buffer = Mock(return_value=cpu_tensor)
    host_value = object()
    outputs = BatchedModelOutput(
        per_rid_outputs={7: {"token": [tensor, host_value], "metadata": "keep"}},
    )
    event = object()

    copied, host_rows = worker._prematerialize_for_check_stop(outputs, event)

    make_stream.assert_called_once_with(device=worker.device)
    side.wait_event.assert_called_once_with(event)
    # nothing is copied until the stop check reads a name
    side.__enter__.assert_not_called()
    assert copied[7]["token"] == [cpu_tensor, host_value]
    assert copied[7]["metadata"] == "keep"
    side.__enter__.assert_called_once_with()
    side.__exit__.assert_called_once()
    runtime.Stream.assert_not_called()
    runtime.stream.assert_not_called()
    cpu_tensor.copy_.assert_called_once_with(tensor, non_blocking=True)
    side.synchronize.assert_called_once_with()
    assert host_rows is None
    assert outputs.per_rid_outputs[7]["token"][0] is tensor


def test_cpu_output_copy_needs_no_accelerator_runtime():
    worker = object.__new__(Worker)
    worker.device = torch.device("cpu")
    outputs = BatchedModelOutput(per_rid_outputs={7: {"token": [torch.tensor([3])]}})
    copied, host_rows = worker._prematerialize_for_check_stop(outputs, None)
    assert copied is outputs.per_rid_outputs
    assert host_rows is None


@pytest.mark.parametrize(
    ("device_type", "available", "compiled"),
    [("cpu", True, False), ("cuda", False, False), ("xpu", False, False),
     ("cuda", True, True), ("xpu", True, True)],
)
def test_engine_compile_checks_its_own_device(monkeypatch, device_type, available, compiled):
    engine = object.__new__(Engine)
    engine._device = torch.device(device_type)
    engine._device_capability = SimpleNamespace(
        is_available=available and device_type != "cpu",
    )
    submodule = SimpleNamespace(forward=Mock(), forward_batched=Mock())
    engine._submodules = {"node": SimpleNamespace(submodule=submodule)}
    engine._item_encoders = set()
    compile_fn = Mock(side_effect=lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch, "compile", compile_fn)

    engine._compile_submodules()

    assert compile_fn.call_count == (2 if compiled else 0)


def test_cpu_graph_sampler_does_not_enter_cuda_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", Mock(side_effect=AssertionError("CUDA entered")))
    logits = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="unsupported on 'cpu'"):
        sample_accelerator_graphable_gpu(logits, None, None, None, None, None)
