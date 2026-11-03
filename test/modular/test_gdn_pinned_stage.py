"""GDN's plan staging reuses its host buffers, so each stage must record an
event on the destination's accelerator stream and the next must wait on it
before rewriting them, on any accelerator rather than CUDA only."""
import numpy as np
import torch

from mstar.engine.resources.linear_attn.wrappers import _PinnedIndexStage


class _Event:
    log: list = []

    def __init__(self, device=None):
        self.device = device

    def record(self, stream=None):
        _Event.log.append(("record", stream))

    def synchronize(self):
        _Event.log.append(("synchronize",))


def _stage(monkeypatch):
    _Event.log = []
    monkeypatch.setattr(torch, "Event", _Event)
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda device: f"stream:{device.type}")
    return _PinnedIndexStage()


def test_each_stage_waits_on_the_previous_ones_copies(monkeypatch):
    # a meta tensor stands in for an accelerator destination (xpu, cuda, ...)
    stage = _stage(monkeypatch)
    dst = torch.empty(64, dtype=torch.int32, device="meta")
    for _ in range(2):
        stage.begin()
        try:
            stage.upload("cu", np.arange(5, dtype=np.int32), dst)
        finally:
            stage.end()
    assert _Event.log == [("record", "stream:meta"), ("synchronize",), ("record", "stream:meta")]
    assert stage._done.device == dst.device


def test_a_cpu_destination_needs_no_event(monkeypatch):
    stage = _stage(monkeypatch)
    dst = torch.zeros(64, dtype=torch.int32)
    stage.begin()
    stage.upload("cu", np.arange(5, dtype=np.int32), dst)
    stage.end()
    assert _Event.log == [] and dst[:5].tolist() == [0, 1, 2, 3, 4]
