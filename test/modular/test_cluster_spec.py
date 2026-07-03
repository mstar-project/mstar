"""Tests for the cluster topology spec (hosts x GPUs -> global ranks)."""

from types import SimpleNamespace

import pytest

from mstar.cluster.spec import DEFAULT_ZMQ_PORT_BASE, ClusterSpec, HostSpec
from mstar.communication.communicator import CommProtocol


def _two_host_config():
    return {
        "cluster": {
            "hosts": [
                {"addr": "nodeA", "gpus": [0, 1, 2, 3]},
                {"addr": "nodeB", "gpus": [0, 1], "zmq_port_base": 19500},
            ]
        }
    }


class TestSynthesizedSingleHost:
    def test_absent_section_gives_identity_mapping(self):
        spec = ClusterSpec.from_config({"model": "x", "node_groups": []})
        assert not spec.is_multi_host()
        assert spec.head_addr == "localhost"
        for rank in (0, 3, 7, 42):
            ws = spec.worker_spec(rank)
            assert ws.worker_id == f"worker_{rank}"
            assert ws.host_index == 0
            assert ws.local_device == rank
            assert ws.addr == "localhost"

    def test_none_config(self):
        spec = ClusterSpec.from_config(None)
        assert spec.worker_spec(5).local_device == 5

    def test_accepts_any_nonnegative_ranks(self):
        spec = ClusterSpec.single_host()
        spec.validate_ranks([0, 2, 5])  # non-contiguous is fine
        with pytest.raises(ValueError):
            spec.worker_spec(-1)

    def test_protocols_all_allowed(self):
        spec = ClusterSpec.single_host()
        for proto in (CommProtocol.SHM, CommProtocol.RDMA, CommProtocol.TCP):
            spec.validate_protocol(proto)


class TestExplicitClusters:
    def test_single_host_scrambled_gpus(self):
        spec = ClusterSpec.from_config(
            {"cluster": {"hosts": [{"addr": "nodeA", "gpus": [3, 1, 0]}]}}
        )
        assert not spec.is_multi_host()
        assert [spec.worker_spec(r).local_device for r in range(3)] == [3, 1, 0]
        assert all(spec.worker_spec(r).addr == "nodeA" for r in range(3))

    def test_two_hosts_rank_assignment(self):
        spec = ClusterSpec.from_config(_two_host_config())
        assert spec.is_multi_host()
        assert spec.head_addr == "nodeA"
        assert spec.hosts[1].zmq_port_base == 19500
        assert spec.hosts[0].zmq_port_base == DEFAULT_ZMQ_PORT_BASE

        ws4 = spec.worker_spec(4)
        assert (ws4.host_index, ws4.local_device, ws4.addr) == (1, 0, "nodeB")
        ws5 = spec.worker_spec(5)
        assert (ws5.host_index, ws5.local_device, ws5.addr) == (1, 1, "nodeB")
        assert spec.worker_spec(3).host_index == 0
        assert spec.host_of_rank(5) == 1

    def test_same_addr_hosts_allowed(self):
        # Two logical hosts on one machine (loopback), disjoint GPU sets.
        spec = ClusterSpec.from_config({
            "cluster": {"hosts": [
                {"addr": "127.0.0.1", "gpus": [0, 1]},
                {"addr": "127.0.0.1", "gpus": [2, 3], "zmq_port_base": 19500},
            ]}
        })
        assert spec.is_multi_host()
        assert spec.worker_spec(2).local_device == 2
        assert spec.worker_spec(2).host_index == 1

    def test_optional_host_fields(self):
        spec = ClusterSpec.from_config({
            "cluster": {"hosts": [{
                "addr": "nodeA", "gpus": [0],
                "bind_addr": "10.0.0.1",
                "env": {"NCCL_SOCKET_IFNAME": "eth1"},
                "rdma_device": "mlx5_0",
            }]}
        })
        host = spec.head
        assert host.bind_addr == "10.0.0.1"
        assert host.env == {"NCCL_SOCKET_IFNAME": "eth1"}
        assert host.rdma_device == "mlx5_0"


class TestValidation:
    def test_rank_out_of_range(self):
        spec = ClusterSpec.from_config(_two_host_config())
        with pytest.raises(ValueError, match="global rank 6"):
            spec.validate_ranks([0, 6])

    def test_shm_rejected_multi_host(self):
        spec = ClusterSpec.from_config(_two_host_config())
        with pytest.raises(ValueError, match="SHM"):
            spec.validate_protocol(CommProtocol.SHM)
        with pytest.raises(ValueError, match="SHM"):
            spec.validate_protocol("SHM")
        spec.validate_protocol(CommProtocol.RDMA)
        spec.validate_protocol(CommProtocol.TCP)

    @pytest.mark.parametrize("hosts, match", [
        ([], "at least one host"),
        ([{"addr": "", "gpus": [0]}], "empty `addr`"),
        ([{"addr": "a", "gpus": []}], "empty `gpus`"),
        ([{"addr": "a", "gpus": [0, 0]}], "duplicate GPU"),
        ([{"addr": "a", "gpus": [-1]}], "non-negative"),
        ([{"addr": "a", "gpus": [0], "port": 1}], "unknown key"),
        ([{"gpus": [0]}], "must define"),
        ([{"addr": "a"}], "must define"),
    ])
    def test_bad_host_entries(self, hosts, match):
        with pytest.raises(ValueError, match=match):
            ClusterSpec.from_config({"cluster": {"hosts": hosts}})

    def test_bad_section_shape(self):
        with pytest.raises(ValueError, match="hosts"):
            ClusterSpec.from_config({"cluster": {}})
        with pytest.raises(ValueError, match="mapping"):
            ClusterSpec.from_config({"cluster": {"hosts": ["nodeA"]}})

    def test_direct_construction_checks(self):
        with pytest.raises(ValueError):
            ClusterSpec([])
        with pytest.raises(ValueError):
            ClusterSpec(
                [HostSpec(addr="a", gpus=(0,))], identity_single_host=True
            )


class TestTPConfigLocalDevices:
    def _fake_worker_graphs(self):
        from mstar.model.base import WorkerGraph

        section = SimpleNamespace(get_nodes=lambda: {"LLM": None})
        wg = WorkerGraph(
            section=section, graph_walks={"decode"}, ranks=[0, 1], tp_size=2
        )
        return {wg.worker_graph_id: wg}

    def test_local_devices_threaded(self):
        from mstar.distributed.communication import GlobalTPConfig

        tp = GlobalTPConfig(
            worker_graphs=self._fake_worker_graphs(),
            worker_ids=["worker_0", "worker_1"],
            local_devices=[5, 3],
        )
        assert tp.per_worker_config["worker_0"].local_device == 5
        assert tp.per_worker_config["worker_1"].local_device == 3
        assert tp.per_worker_config["worker_1"].global_rank == 1

    def test_local_devices_default_none(self):
        from mstar.distributed.communication import GlobalTPConfig

        tp = GlobalTPConfig(
            worker_graphs=self._fake_worker_graphs(),
            worker_ids=["worker_0", "worker_1"],
        )
        assert tp.per_worker_config["worker_0"].local_device is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
