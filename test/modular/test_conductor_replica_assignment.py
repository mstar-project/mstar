"""Data-parallel replica choice: a new request goes to the replica serving the
fewest live requests, so concurrent requests spread over the replicas instead of
landing on one at random. The load is per-(node group, replica) counters the
conductor keeps as requests register and drop."""
from collections import Counter
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402


def _wg(group_id, ranks, sp_size=1):
    return WorkerGraph(section=None, graph_walks=set(), ranks=list(ranks), sp_size=sp_size, _group_id=group_id)


def _conductor(worker_graphs):
    conductor = object.__new__(Conductor)
    conductor.worker_graphs = worker_graphs
    conductor.requests = {}
    conductor._replica_load = Counter()
    return conductor


def _admit(conductor, rid):
    replicas = conductor._pick_replicas()
    conductor._register_request(rid, SimpleNamespace(dp_replicas=replicas))
    return conductor._assign_worker_graphs_to_workers(replicas)


def test_requests_alternate_over_dp_replicas():
    # two node groups replicated on ranks 0 and 1, one wg each
    conductor = _conductor({0: _wg(0, [0, 1]), 1: _wg(0, [0, 1])})
    first = _admit(conductor, "a")
    second = _admit(conductor, "b")
    # a group's wgs stay on one replica, and the second request takes the idle one
    assert first[0] == first[1] and second[0] == second[1]
    assert first[0] != second[0]
    # a request leaving frees its replica's count, once
    conductor._drop_request("a")
    conductor._drop_request("a")
    assert sorted(conductor._replica_load.values()) == [0, 1]
    assert _admit(conductor, "c")[0] == first[0]


def test_lockstep_instances_are_balanced_whole():
    # one sp2 node replicated as two instances: ranks (0, 1) and (2, 3)
    conductor = _conductor({0: _wg(0, [0, 1, 2, 3], sp_size=2)})
    picks = {tuple(_admit(conductor, rid)[0]) for rid in ("a", "b")}
    assert picks == {("worker_0", "worker_1"), ("worker_2", "worker_3")}


def test_groups_sharing_a_worker_balance_independently():
    # group 1 is DP2 on ranks 0 and 1; group 0 has one replica, on rank 0. Rank
    # 0's group-0 requests must not push every group-1 request onto rank 1.
    conductor = _conductor({0: _wg(0, [0]), 1: _wg(1, [0, 1])})
    picks = [_admit(conductor, rid)[1][0] for rid in "abcd"]
    assert sorted(picks) == ["worker_0", "worker_0", "worker_1", "worker_1"]
