"""Runtime capability selection must not initialize accelerator hardware."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.accelerator_graph_backend import _GRAPH_BACKENDS, AcceleratorGraphPool
from mstar.utils.device import DeviceCapability


def test_cpu_capability_is_cached_and_does_not_touch_accelerators(monkeypatch):
    fail = Mock(side_effect=AssertionError("CPU queried an accelerator"))
    monkeypatch.setattr(torch.accelerator, "is_available", fail)
    monkeypatch.setattr(torch.accelerator, "synchronize", fail)
    capability = DeviceCapability.of("cpu")
    assert capability is DeviceCapability.of(torch.device("cpu"))
    assert not capability.is_accelerator
    assert not capability.pin_memory
    assert capability.graph_backend is None
    assert not capability.is_capturing()
    fail.assert_not_called()


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_unavailable_runtime_selection_does_not_initialize_hardware(monkeypatch, device_type):
    runtime = getattr(torch, device_type)
    monkeypatch.setattr(runtime, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "_lazy_init", Mock(side_effect=AssertionError("initialized")))
    capability = DeviceCapability(torch.device(device_type, 0))
    assert capability.is_accelerator
    assert not capability.pin_memory
    assert capability.graph_backend is not None
    assert not capability.is_capturing()


def test_another_runtime_reports_capabilities_without_creating_streams_or_events(monkeypatch):
    device = torch.device("privateuseone:2")
    runtime = SimpleNamespace(
        is_available=lambda: True, Stream=Mock(),
        Event=Mock(), stream=Mock(),
    )
    backend = SimpleNamespace(is_current_stream_capturing=Mock(return_value=True))
    monkeypatch.setattr(torch, "privateuseone", runtime, raising=False)
    monkeypatch.setitem(_GRAPH_BACKENDS, "privateuseone", lambda device: backend)

    capability = DeviceCapability(device)
    assert capability.is_accelerator and capability.pin_memory
    assert capability.graph_backend is backend and capability.is_capturing()
    runtime.Stream.assert_not_called()
    runtime.Event.assert_not_called()
    runtime.stream.assert_not_called()


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_graph_backend_unwraps_pool_only_at_runtime_boundary(monkeypatch, device_type):
    capability = DeviceCapability(torch.device(device_type, 0))
    runtime = getattr(torch, device_type)
    capture = Mock(return_value=nullcontext())
    monkeypatch.setattr(runtime, "graph", capture, raising=False)
    pool, graph = AcceleratorGraphPool(object()), object()
    with capability.graph_backend.capture(graph, pool=pool):
        pass
    capture.assert_called_once_with(graph, pool=pool.handle)
