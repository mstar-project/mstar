"""The worker reads MSTAR_LAUNCH_WAIT_MS and MSTAR_MAX_CONSECUTIVE_SPEC_STEPS
when it is built.

``Worker.run`` used to parse both after it had sent SETUP_DONE, so a value
such as ``5ms`` raised only once weight load, warmup and CUDA-graph capture
were done and the conductor had counted the worker ready, and then the worker
died. Now a bad value is reported, with a warning, before anything loads and
the default is used, as ``MSTAR_PY_SWITCH_INTERVAL_SEC`` already does.

These run the real ``Worker.__init__`` over stand-ins for the process groups,
the transports, the graph runtime and the engines, and read the values where
they are used: on the ``Speculator`` and the ``StepPipeline``.
"""

from __future__ import annotations

import inspect
import logging
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.worker import worker as worker_mod
from mstar.worker.engine_manager import InlineStreams
from mstar.worker.worker import Worker

LAUNCH = "MSTAR_LAUNCH_WAIT_MS"
CAP = "MSTAR_MAX_CONSECUTIVE_SPEC_STEPS"


def _build(monkeypatch) -> Worker:
    """The real ``Worker.__init__``, with no sockets, process groups or engines."""
    monkeypatch.setattr(worker_mod, "make_communicator", lambda **kw: SimpleNamespace(
        register_event_for_poll=lambda event: None,
    ))
    monkeypatch.setattr(worker_mod, "create_tensor_communication_manager", lambda **kw: SimpleNamespace(
        transfer_engine=None, my_session_id="session", get_tensor=None,
    ))
    monkeypatch.setattr(worker_mod, "_make_graph_runtime", lambda **kw: SimpleNamespace(
        set_node_metadata=lambda **metadata: None, set_inline_streams=lambda keys: None,
    ))
    # an engine with no paged KV cache to preempt in
    monkeypatch.setattr(worker_mod.EngineManager, "build", lambda *a, **kw: SimpleNamespace(
        side_keys=(), inline_streams=InlineStreams(frozenset(), {}, {}),
        engine=SimpleNamespace(paged_kv_caches=list, host_allowance=None, node_dtype=lambda node: None),
    ))
    monkeypatch.setattr(Worker, "_plan_step_cpus", lambda self, *a: (None, frozenset()))
    return Worker(
        worker_id="w0", worker_ids=["w0"], model=None, my_worker_graphs=[],
        model_config={}, all_worker_graph_ids_to_graph_walks={},
        all_worker_graph_ids_to_nodes={}, all_worker_graph_ids_to_dyn_loops={},
        sharding_config=None, parallel_groups=SimpleNamespace(init_dist=lambda **kw: None),
        device=torch.device("cpu"),
    )


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(LAUNCH, raising=False)
    monkeypatch.delenv(CAP, raising=False)


def test_defaults(monkeypatch):
    worker = _build(monkeypatch)
    assert worker.pipeline._launch_wait_s == pytest.approx(0.005)
    assert worker.speculator._max_consecutive == 1024


def test_valid_values_are_used(monkeypatch):
    monkeypatch.setenv(LAUNCH, "0.5")
    monkeypatch.setenv(CAP, "8")
    worker = _build(monkeypatch)
    assert worker.pipeline._launch_wait_s == pytest.approx(0.0005)
    assert worker.speculator._max_consecutive == 8


@pytest.mark.parametrize(("var", "raw"), [
    (LAUNCH, "5ms"), (LAUNCH, "nan"), (LAUNCH, "inf"), (LAUNCH, "1e20"),
    (CAP, "5ms"), (CAP, "1e3"), (CAP, "16.0"),
])
def test_a_malformed_value_is_reported_when_the_worker_is_built(monkeypatch, caplog, var, raw):
    monkeypatch.setenv(var, raw)
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert worker.pipeline._launch_wait_s == pytest.approx(0.005)
    assert worker.speculator._max_consecutive == 1024
    assert any(var in r.getMessage() and raw in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(("var", "raw", "read", "expected"), [
    (LAUNCH, "-3", lambda worker: worker.pipeline._launch_wait_s, 0.0),
    (CAP, "-1", lambda worker: worker.speculator._max_consecutive, 0),
])
def test_a_negative_value_is_raised_to_the_minimum(monkeypatch, caplog, var, raw, read, expected):
    monkeypatch.setenv(var, raw)
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert read(worker) == expected
    assert any(var in r.getMessage() and "below" in r.getMessage() for r in caplog.records)


def test_a_cap_of_zero_says_speculation_is_off(monkeypatch, caplog):
    monkeypatch.setenv(CAP, "0")
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert worker.speculator._max_consecutive == 0
    assert any("speculation off" in r.getMessage() for r in caplog.records)


def test_run_does_not_parse_them_after_setup_done():
    source = inspect.getsource(Worker.run)
    assert LAUNCH not in source and CAP not in source
