"""``resolve_dist_timeout`` / ``MSTAR_DIST_TIMEOUT_S``: the process-group timeout knob.

Torch's default (10 min) is shorter than a large checkpoint load at TP8, so a
deployment opts in via ``dist_timeout_s`` in its config; the env var overrides
it per process. Unset means ``{}`` — torch's default, untouched. Serving
collectives go back to torch's default once warmup is done.
"""

from __future__ import annotations

import socket
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout

sys.path.insert(0, ".")

import mstar.distributed.communication as comm_mod
from mstar.distributed.communication import (
    DIST_TIMEOUT_ENV,
    CommGroup,
    WorkerParallelGroups,
    resolve_dist_timeout,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(DIST_TIMEOUT_ENV, raising=False)


def test_neither_set_is_torch_default():
    assert resolve_dist_timeout() == {}
    assert resolve_dist_timeout(None) == {}


def test_config_alone():
    assert resolve_dist_timeout(7200) == {"timeout": timedelta(seconds=7200)}


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv(DIST_TIMEOUT_ENV, "90")
    assert resolve_dist_timeout(7200) == {"timeout": timedelta(seconds=90)}
    assert resolve_dist_timeout() == {"timeout": timedelta(seconds=90)}


def test_blank_env_falls_back_to_config(monkeypatch):
    monkeypatch.setenv(DIST_TIMEOUT_ENV, "  ")
    assert resolve_dist_timeout(30) == {"timeout": timedelta(seconds=30)}


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_config_raises(bad):
    with pytest.raises(ValueError, match="positive"):
        resolve_dist_timeout(bad)


def test_non_positive_env_raises(monkeypatch):
    monkeypatch.setenv(DIST_TIMEOUT_ENV, "0")
    with pytest.raises(ValueError, match="positive"):
        resolve_dist_timeout(7200)


def test_non_numeric_env_raises(monkeypatch):
    monkeypatch.setenv(DIST_TIMEOUT_ENV, "soon")
    with pytest.raises(ValueError, match=DIST_TIMEOUT_ENV):
        resolve_dist_timeout()


def _capture_process_group_calls(monkeypatch):
    """Stub the two collectives ``init_dist`` issues and record their kwargs."""
    calls: dict[str, list[dict]] = {"init": [], "new_group": []}
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: calls["init"].append(kw))
    monkeypatch.setattr(
        dist, "new_group", lambda **kw: calls["new_group"].append(kw) or object()
    )
    monkeypatch.setattr(dist, "get_default_backend_for_device", lambda _t: "gloo")
    return calls


def _two_tp_groups() -> WorkerParallelGroups:
    groups = WorkerParallelGroups(
        num_workers=4, global_rank=0, any_parallelism=True,
        world_parallel_groups=[(0, 1), (2, 3)],
    )
    groups.add("llm", CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1]))
    return groups


def test_init_dist_threads_timeout_into_world_and_every_subgroup(monkeypatch):
    calls = _capture_process_group_calls(monkeypatch)
    groups = _two_tp_groups()
    groups.dist_timeout_s = 3600
    groups.init_dist(device=torch.device("cpu"))

    expected = timedelta(seconds=3600)
    assert [c["timeout"] for c in calls["init"]] == [expected]
    # ``new_group`` runs once per rank tuple on every rank, members or not,
    # for the device group and its gloo twin.
    assert [c["timeout"] for c in calls["new_group"]] == [expected] * 4


def test_init_dist_without_timeout_leaves_torch_default(monkeypatch):
    calls = _capture_process_group_calls(monkeypatch)
    _two_tp_groups().init_dist(device=torch.device("cpu"))

    assert all("timeout" not in c for c in calls["init"] + calls["new_group"])


@pytest.fixture
def one_rank_gloo_world():
    """A real one-rank gloo world, made with a long timeout as ``init_dist`` makes it."""
    assert not dist.is_initialized()
    long = timedelta(seconds=3600)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0,
        timeout=long,
    )
    sub = dist.new_group(ranks=[0], backend="gloo", timeout=long)
    yield sub
    dist.destroy_process_group()


def _timeout(pg) -> timedelta:
    return pg._get_backend(torch.device("cpu")).options._timeout


def test_serving_returns_every_group_to_torch_default(one_rank_gloo_world):
    # load and warmup need the long timeout; a wedged rank in serving must
    # still abort at torch's default rather than hang for hours
    sub = one_rank_gloo_world
    groups = _two_tp_groups()
    groups.dist_timeout_s = 3600
    comm = groups.node_to_tp_group["llm"]
    comm.device_group = comm.cpu_group = sub
    assert _timeout(dist.group.WORLD) == _timeout(sub) == timedelta(seconds=3600)

    groups.restore_default_timeout()

    assert _timeout(dist.group.WORLD) == _timeout(sub) == default_pg_timeout


def test_without_load_timeout_nothing_is_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(comm_mod, "_set_pg_timeout", lambda *a: calls.append(a))
    _two_tp_groups().restore_default_timeout()
    assert calls == []
