"""``resolve_dist_timeout``: the process-group timeout knob.

Torch's default (10 min) is shorter than a large checkpoint load at TP8, so a
deployment opts in via ``dist_timeout_s`` in its config. Unset means ``{}`` —
torch's default, untouched. Serving collectives go back to torch's default
once warmup is done.
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
from mstar.distributed.communication import CommGroup, WorkerParallelGroups
from mstar.distributed.utils import resolve_dist_timeout


def test_neither_set_is_torch_default():
    assert resolve_dist_timeout() == {}
    assert resolve_dist_timeout(None) == {}


def test_config_alone():
    assert resolve_dist_timeout(7200) == {"timeout": timedelta(seconds=7200)}


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_config_raises(bad):
    with pytest.raises(ValueError, match="positive"):
        resolve_dist_timeout(bad)


def _capture_process_group_calls(monkeypatch):
    """Stub the two collectives ``init_dist`` issues and record their kwargs,
    and the rendezvous store's timeout."""
    calls: dict[str, list] = {"init": [], "new_group": [], "store": []}
    monkeypatch.setattr(
        comm_mod, "rendezvous_store",
        lambda init_method, rank, world_size, timeout: calls["store"].append(timeout),
    )
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
    groups.add("llm", CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1]), "tp")
    return groups


def test_init_dist_threads_timeout_into_world_and_every_subgroup(monkeypatch):
    calls = _capture_process_group_calls(monkeypatch)
    groups = _two_tp_groups()
    # a lockstep instance no axis group spans gets a gloo group of its own,
    # which every warmup capture barrier waits on
    groups.node_to_instance_comm["llm"] = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1, 2, 3])
    groups.node_to_instance_groups["llm"] = frozenset({(0, 1, 2, 3)})
    groups.dist_timeout_s = 3600
    groups.init_dist(device=torch.device("cpu"))

    expected = timedelta(seconds=3600)
    assert calls["store"] == [expected]
    assert [c["timeout"] for c in calls["init"]] == [expected]
    # ``new_group`` runs once per gloo group (each axis tuple's twin, a
    # non-member's stand-in included, and the instance's), then once per axis
    # tuple for its device group.
    assert [c["timeout"] for c in calls["new_group"]] == [expected] * 5
    assert [0, 1, 2, 3] in [c["ranks"] for c in calls["new_group"]]


def test_init_dist_without_timeout_leaves_torch_default(monkeypatch):
    calls = _capture_process_group_calls(monkeypatch)
    _two_tp_groups().init_dist(device=torch.device("cpu"))

    # the world group and its store get torch's default, which init_dist now names
    assert calls["store"] == [default_pg_timeout]
    assert [c["timeout"] for c in calls["init"]] == [default_pg_timeout]
    assert all("timeout" not in c for c in calls["new_group"])


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
    comm = groups.axis_groups["tp"]["llm"]
    comm.device_group = comm.cpu_group = sub
    assert _timeout(dist.group.WORLD) == _timeout(sub) == timedelta(seconds=3600)

    groups.restore_default_timeout()

    assert _timeout(dist.group.WORLD) == _timeout(sub) == default_pg_timeout


def test_without_load_timeout_nothing_is_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(comm_mod, "_set_pg_timeout", lambda *a: calls.append(a))
    _two_tp_groups().restore_default_timeout()
    assert calls == []
