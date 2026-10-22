"""A piecewise region's capture-time padding rows give their pages back.

A capture hands each padding row a share of the bucket's token budget, and a
packed replay pads with zero-length rows, so what the capture allocated is
never read again. ``AcceleratorGraphRunner`` releases it after warmup; the piecewise
runner has to do the same, or the largest buckets of every region stay out of
the pool for the server's lifetime.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from unittest import mock

import pytest
import torch

from mstar.engine import accelerator_graph_runner as cgr
from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool
from mstar.engine.accelerator_graph_config import PiecewiseBatchedConfig, PiecewisePackedConfig
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.resources import (
    KVStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager

PAGE_SIZE = 4
CPU = torch.device("cpu")


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


class _CpuGraphBackend:
    """The graph backend calls the piecewise runner makes, as no-ops on the CPU."""

    device_type = "cpu"

    def is_available(self):
        return True

    def set_device(self):
        pass

    def synchronize(self):
        pass

    def graph_pool_handle(self):
        return AcceleratorGraphPool("cpu")


_CPU_GRAPHS = SimpleNamespace(graph_backend=_CpuGraphBackend())


@pytest.fixture(autouse=True)
def _cpu_capture(monkeypatch):
    """The runner's capture path on the CPU: every CUDA call it makes is a
    no-op, and the graph capture just runs the region once."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    # warm-up reads the device's memory through torch.accelerator, which has no CPU device
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda device=None: 0, raising=False)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda device=None: 0, raising=False)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda device=None: None, raising=False)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(
        cgr, "capture_into_graph",
        lambda run, pool, device, dtype, keep_graph=False: (SimpleNamespace(replay=lambda: None), run()),
    )


def _kv(max_num_pages: int = 64) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE, max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
    )
    # a CPU cache: the fixture's faked CUDA must not reach its pinned host buffers
    with mock.patch.object(torch.cuda, "is_available", lambda: False):
        return KVManager(
            cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=CPU, dtype=torch.float32,
        )


def _declare(request_ids, seq_lens):
    return SubmoduleStep(
        segments=[Segment(rid, "main", n) for rid, n in zip(request_ids, seq_lens, strict=True)],
        steps={"kv": KVStep()},
    )


def _runner(kv: KVManager) -> PiecewiseAcceleratorGraphRunner:
    step_runner = StepRunner({"kv": kv})
    config = PiecewisePackedConfig(
        capture_fn=lambda call: {"x": call.static_inputs["x"] + 1},
        make_static_inputs=lambda shape: {"x": torch.zeros(shape.total_tokens)},
        declare_step=_declare,
        total_tokens=[16, 32],
        capture_batch_sizes=[1, 2],
    )
    return PiecewiseAcceleratorGraphRunner(
        label="region", config=config, resources={"kv": kv},
        step_runner=step_runner, device=CPU, autocast_dtype=None, num_slots=1,
        device_capability=_CPU_GRAPHS,
    )


def test_capture_pages_come_back_to_the_pool():
    kv = _kv()
    free = kv._arena.num_free
    runner = _runner(kv)

    runner.warmup_and_capture(None)

    assert runner.any_graphs
    assert kv._arena.num_free == free, (
        f"{free - kv._arena.num_free} pages held by padding rows after capture"
    )


def test_a_replay_after_release_allocates_only_the_real_rows():
    kv = _kv()
    runner = _runner(kv)
    runner.warmup_and_capture(None)
    kv.ingest_request("real")
    free = kv._arena.num_free

    runner.run({"x": torch.ones(5)}, request_ids=["real"], seq_lens=[5])

    assert kv._streams["real"]["main"].stored_len == 5
    assert kv._arena.num_free == free - 2, "padding rows took pages at replay"


def test_the_whole_pool_is_usable_after_capture():
    kv = _kv()
    runner = _runner(kv)
    runner.warmup_and_capture(None)
    kv.ingest_request("real")
    step = _declare(["real"], [63 * PAGE_SIZE])
    step.set_ctx(StepContext(request_ids=("real",), graph_walk="w", slot=0, capture=False))

    # one page is the sink: a request needing every other page fits only if
    # the capture gave its pages back
    assert StepRunner({"kv": kv}).admit(step).ok


def test_a_batched_region_keeps_its_padding_rows():
    # batched padding rows replay real spans, so their pages stay with them
    kv = _kv(max_num_pages=16)
    config = PiecewiseBatchedConfig(
        capture_fn=lambda call: {"x": call.static_inputs["x"] + 1},
        make_static_inputs=lambda shape: {"x": torch.zeros(shape.bs, 8)},
        declare_step=_declare,
        seq_len=8,
        capture_batch_sizes=[2],
    )
    runner = PiecewiseAcceleratorGraphRunner(
        label="region", config=config, resources={"kv": kv},
        step_runner=StepRunner({"kv": kv}), device=CPU, autocast_dtype=None, num_slots=1,
        device_capability=_CPU_GRAPHS,
    )
    runner.warmup_and_capture(None)
    kv.ingest_request("big")
    big = _declare(["big"], [kv._arena.num_free * PAGE_SIZE - 2 * PAGE_SIZE])
    big.set_ctx(StepContext(request_ids=("big",), graph_walk="w", slot=0, capture=False))
    step_runner = StepRunner({"kv": kv})
    assert step_runner.admit(big).ok
    kv.ingest_request("real")

    runner.run({"x": torch.ones(1, 8)}, request_ids=["real"], real_bs=1)

    assert kv._streams["real"]["main"].stored_len == 8
