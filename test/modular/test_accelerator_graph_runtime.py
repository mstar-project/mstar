"""Device dispatch for graph execution and its worker output copies."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.sampler.utils import sample_cuda_graphable_gpu
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.worker import Worker


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_output_copy_uses_worker_device_runtime(monkeypatch, device_type):
    worker = object.__new__(Worker)
    worker.device = torch.device(device_type, 1)
    worker._d2h_stream = None
    side = Mock()
    runtime = SimpleNamespace(
        is_available=lambda: True,
        Stream=Mock(return_value=side),
        stream=Mock(return_value=nullcontext()),
    )
    monkeypatch.setattr(torch, device_type, runtime)
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

    runtime.Stream.assert_called_once_with(device=worker.device)
    side.wait_event.assert_called_once_with(event)
    runtime.stream.assert_called_once_with(side)
    cpu_tensor.copy_.assert_called_once_with(tensor, non_blocking=True)
    side.synchronize.assert_called_once_with()
    assert copied == {7: {"token": [cpu_tensor, host_value], "metadata": "keep"}}
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
    engine._device_module = SimpleNamespace(is_available=lambda: available)
    submodule = SimpleNamespace(forward=Mock(), forward_batched=Mock())
    engine._submodules = {"node": SimpleNamespace(submodule=submodule)}
    compile_fn = Mock(side_effect=lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch, "compile", compile_fn)

    engine._compile_submodules()

    assert compile_fn.call_count == (2 if compiled else 0)


def test_xpu_graph_sampler_does_not_enter_cuda_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", Mock(side_effect=AssertionError("CUDA entered")))
    logits = SimpleNamespace(device=torch.device("xpu"))
    with pytest.raises(NotImplementedError, match="use eager sampling on xpu"):
        sample_cuda_graphable_gpu(logits, None, None, None, None, None)
