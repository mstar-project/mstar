"""Tests for replica assignment (masking + locality) and cross-host cut detection."""

from types import SimpleNamespace

import pytest

from mstar.cluster.spec import ClusterSpec
from mstar.conductor.conductor import (
    NoLiveReplicaError,
    assign_worker_graphs,
    find_cross_host_cuts,
)
from mstar.graph.base import GraphEdge
from mstar.model.base import WorkerGraph


def _wg(nodes, ranks, tp_size=1, group_id=0, walks=("decode",), edges=()):
    section = SimpleNamespace(
        get_nodes=lambda: {
            n: SimpleNamespace(outputs=[
                GraphEdge(next_node=dst, name=name) for (src, dst, name) in edges if src == n
            ])
            for n in nodes
        },
    )
    return WorkerGraph(
        section=section, graph_walks=set(walks), ranks=list(ranks),
        tp_size=tp_size, _group_id=group_id,
    )


def _two_host_spec():
    return ClusterSpec.from_config({
        "cluster": {"hosts": [
            {"addr": "a", "gpus": [0, 1]},              # ranks 0,1 on host 0
            {"addr": "b", "gpus": [0, 1], "zmq_port_base": 19600},  # ranks 2,3 on host 1
        ]}
    })


class TestAssignment:
    def test_group_coordination(self):
        spec = ClusterSpec.single_host()
        wg1 = _wg(["A"], ranks=[0, 1], group_id=0)
        wg2 = _wg(["A"], ranks=[0, 1], group_id=0, walks=("prefill",))
        for _ in range(10):
            out = assign_worker_graphs(
                {wg1.worker_graph_id: wg1, wg2.worker_graph_id: wg2}, spec, set()
            )
            assert out[wg1.worker_graph_id] == out[wg2.worker_graph_id]

    def test_dead_replica_masked(self):
        spec = ClusterSpec.single_host()
        wg = _wg(["A"], ranks=[0, 1], group_id=0)
        for _ in range(10):
            out = assign_worker_graphs({wg.worker_graph_id: wg}, spec, {"worker_0"})
            assert out[wg.worker_graph_id] == ["worker_1"]

    def test_no_live_replica_raises(self):
        spec = ClusterSpec.single_host()
        wg = _wg(["A"], ranks=[0], group_id=0)
        with pytest.raises(NoLiveReplicaError):
            assign_worker_graphs({wg.worker_graph_id: wg}, spec, {"worker_0"})

    def test_tp_replicas(self):
        spec = ClusterSpec.single_host()
        wg = _wg(["A"], ranks=[0, 1, 2, 3], tp_size=2, group_id=0)
        out = assign_worker_graphs({wg.worker_graph_id: wg}, spec, {"worker_0"})
        # replica [0,1] contains the dead worker; only [2,3] is live
        assert out[wg.worker_graph_id] == ["worker_2", "worker_3"]

    def test_locality_prefers_used_host(self):
        spec = _two_host_spec()
        anchored = _wg(["A"], ranks=[2], group_id=0)          # host 1 only
        flexible = _wg(["B"], ranks=[0, 3], group_id=1)       # replicas on host 0 and host 1
        for _ in range(20):
            out = assign_worker_graphs(
                {anchored.worker_graph_id: anchored, flexible.worker_graph_id: flexible},
                spec, set(),
            )
            assert out[flexible.worker_graph_id] == ["worker_3"]  # host 1, co-located

    def test_rank_tiebreak_prefers_shared_worker(self):
        spec = ClusterSpec.single_host()
        anchored = _wg(["A"], ranks=[1], group_id=0)
        flexible = _wg(["B"], ranks=[0, 1], group_id=1)       # host scores tie
        for _ in range(20):
            out = assign_worker_graphs(
                {anchored.worker_graph_id: anchored, flexible.worker_graph_id: flexible},
                spec, set(),
            )
            assert out[flexible.worker_graph_id] == ["worker_1"]  # same worker as A

    def test_rank_tiebreak_within_host_ties(self):
        spec = ClusterSpec.from_config({
            "cluster": {"hosts": [
                {"addr": "a", "gpus": [0, 1, 2]},             # ranks 0-2 on host 0
                {"addr": "b", "gpus": [0], "zmq_port_base": 19600},  # rank 3 on host 1
            ]}
        })
        anchored = _wg(["A"], ranks=[1], group_id=0)          # host 0, worker_1
        flexible = _wg(["B"], ranks=[0, 1], group_id=1)       # both replicas on host 0
        for _ in range(20):
            out = assign_worker_graphs(
                {anchored.worker_graph_id: anchored, flexible.worker_graph_id: flexible},
                spec, set(),
            )
            assert out[flexible.worker_graph_id] == ["worker_1"]

    def test_first_pick_uniform(self):
        spec = ClusterSpec.single_host()
        wg = _wg(["A"], ranks=[0, 1], group_id=0)
        seen = set()
        for _ in range(40):
            out = assign_worker_graphs({wg.worker_graph_id: wg}, spec, set())
            seen.add(out[wg.worker_graph_id][0])
        assert seen == {"worker_0", "worker_1"}


class TestCrossHostCuts:
    def _maps(self, wgs):
        node_walk_to_wg = {}
        for wg in wgs:
            for walk in wg.graph_walks:
                for n in wg.section.get_nodes():
                    node_walk_to_wg[(n, walk)] = wg
        return {wg.worker_graph_id: wg for wg in wgs}, node_walk_to_wg

    def test_cut_detected_across_hosts(self):
        spec = _two_host_spec()
        prod = _wg(["A"], ranks=[0], edges=[("A", "B", "hidden")])
        cons = _wg(["B"], ranks=[2])
        wgs, nw = self._maps([prod, cons])
        cuts = find_cross_host_cuts(wgs, nw, spec)
        assert cuts == {("decode", "A", "B", "hidden", (0,), (1,))}

    def test_same_host_edge_not_reported(self):
        spec = _two_host_spec()
        prod = _wg(["A"], ranks=[0], edges=[("A", "B", "hidden")])
        cons = _wg(["B"], ranks=[1])  # also host 0
        wgs, nw = self._maps([prod, cons])
        assert find_cross_host_cuts(wgs, nw, spec) == set()

    def test_special_destinations_ignored(self):
        spec = _two_host_spec()
        prod = _wg(["A"], ranks=[0], edges=[("A", "EMIT_TO_CLIENT", "out")])
        wgs, nw = self._maps([prod])
        assert find_cross_host_cuts(wgs, nw, spec) == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
