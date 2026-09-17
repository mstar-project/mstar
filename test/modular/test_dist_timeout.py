"""``resolve_dist_timeout`` / ``MSTAR_DIST_TIMEOUT_S``: the process-group timeout knob."""

from __future__ import annotations

import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, ".")

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
    # ``new_group`` runs once per rank tuple on every rank, members or not.
    assert [c["timeout"] for c in calls["new_group"]] == [expected, expected]


def test_init_dist_without_timeout_leaves_torch_default(monkeypatch):
    calls = _capture_process_group_calls(monkeypatch)
    _two_tp_groups().init_dist(device=torch.device("cpu"))

    assert all("timeout" not in c for c in calls["init"] + calls["new_group"])
