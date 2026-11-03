"""The worker reads MSTAR_LAUNCH_WAIT_MS and MSTAR_MAX_CONSECUTIVE_SPEC_STEPS
when it is built.

``Worker.run`` used to parse both after it had sent SETUP_DONE, so a value
such as ``5ms`` raised only once weight load, warmup and accelerator graph capture
were done and the conductor had counted the worker ready, and then the worker
died. Now a bad value is reported, with a warning, before anything loads and
the default is used, as ``MSTAR_PY_SWITCH_INTERVAL_SEC`` already does.

These run the real ``Worker.__init__`` up to the first call after the knobs
(``phase_buffer``), which is before the communicator, the graph runtime and
the engines are built.
"""

from __future__ import annotations

import inspect
import logging
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.worker import worker as worker_mod
from mstar.worker.worker import Worker

LAUNCH = "MSTAR_LAUNCH_WAIT_MS"
CAP = "MSTAR_MAX_CONSECUTIVE_SPEC_STEPS"


class _StopError(Exception):
    pass


def _build(monkeypatch) -> Worker:
    """Run ``Worker.__init__`` until just after it reads the knobs."""
    def stop():
        raise _StopError

    monkeypatch.setattr(worker_mod, "phase_buffer", stop)
    worker = Worker.__new__(Worker)
    with pytest.raises(_StopError):
        worker.__init__(
            worker_id="w0", worker_ids=["w0"], model=None, my_worker_graphs=[],
            model_config={}, all_worker_graph_ids_to_graph_walks={},
            all_worker_graph_ids_to_nodes={}, all_worker_graph_ids_to_dyn_loops={},
            sharding_config=None, parallel_groups=None, device=torch.device("cpu"),
        )
    return worker


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(LAUNCH, raising=False)
    monkeypatch.delenv(CAP, raising=False)


def test_defaults(monkeypatch):
    worker = _build(monkeypatch)
    assert worker._launch_wait_s == pytest.approx(0.005)
    assert worker._max_consecutive_spec == 1024


def test_valid_values_are_used(monkeypatch):
    monkeypatch.setenv(LAUNCH, "0.5")
    monkeypatch.setenv(CAP, "8")
    worker = _build(monkeypatch)
    assert worker._launch_wait_s == pytest.approx(0.0005)
    assert worker._max_consecutive_spec == 8


@pytest.mark.parametrize(("var", "raw"), [
    (LAUNCH, "5ms"), (LAUNCH, "nan"), (LAUNCH, "inf"), (LAUNCH, "1e20"),
    (CAP, "5ms"), (CAP, "1e3"), (CAP, "16.0"),
])
def test_a_malformed_value_is_reported_when_the_worker_is_built(monkeypatch, caplog, var, raw):
    monkeypatch.setenv(var, raw)
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert worker._launch_wait_s == pytest.approx(0.005)
    assert worker._max_consecutive_spec == 1024
    assert any(var in r.getMessage() and raw in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(("var", "raw", "attr", "expected"), [
    (LAUNCH, "-3", "_launch_wait_s", 0.0),
    (CAP, "-1", "_max_consecutive_spec", 0),
])
def test_a_negative_value_is_raised_to_the_minimum(monkeypatch, caplog, var, raw, attr, expected):
    monkeypatch.setenv(var, raw)
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert getattr(worker, attr) == expected
    assert any(var in r.getMessage() and "below" in r.getMessage() for r in caplog.records)


def test_a_cap_of_zero_says_speculation_is_off(monkeypatch, caplog):
    monkeypatch.setenv(CAP, "0")
    with caplog.at_level(logging.WARNING, logger=worker_mod.logger.name):
        worker = _build(monkeypatch)
    assert worker._max_consecutive_spec == 0
    assert any("speculation off" in r.getMessage() for r in caplog.records)


def test_run_does_not_parse_them_after_setup_done():
    source = inspect.getsource(Worker.run)
    assert LAUNCH not in source and CAP not in source
