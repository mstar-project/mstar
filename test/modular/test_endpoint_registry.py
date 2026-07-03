"""Tests for control-plane endpoint resolution and the ZMQ communicator modes."""

import os
import time

import pytest

from mstar.cluster.endpoints import ControlPlaneEndpoints
from mstar.cluster.spec import ClusterSpec
from mstar.communication.communicator import CommProtocol, ZMQCommunicator


def _spec(hosts):
    return ClusterSpec.from_config({"cluster": {"hosts": hosts}})


def _two_host_endpoints(base_a=None, base_b=None, addr="127.0.0.1"):
    pid_slot = (os.getpid() % 500) * 4
    base_a = base_a if base_a is not None else 21000 + pid_slot
    base_b = base_b if base_b is not None else base_a + 600
    spec = _spec([
        {"addr": addr, "gpus": [0], "zmq_port_base": base_a},
        {"addr": addr, "gpus": [1], "zmq_port_base": base_b},
    ])
    return ControlPlaneEndpoints(spec), base_a, base_b


class TestResolution:
    def test_head_entities_on_head_host(self):
        eps, base_a, _ = _two_host_endpoints()
        assert eps.connect_endpoint("api_server") == f"tcp://127.0.0.1:{base_a}"
        assert eps.connect_endpoint("conductor") == f"tcp://127.0.0.1:{base_a + 1}"
        assert eps.connect_endpoint("api_server_preprocess_worker") == f"tcp://127.0.0.1:{base_a + 2}"

    def test_workers_use_their_hosts(self):
        eps, base_a, base_b = _two_host_endpoints()
        # global rank 0 lives on host A, global rank 1 on host B; worker ports
        # are keyed by global rank on the owning host's base.
        assert eps.connect_endpoint("worker_0") == f"tcp://127.0.0.1:{base_a + 100}"
        assert eps.connect_endpoint("worker_1") == f"tcp://127.0.0.1:{base_b + 101}"

    def test_agents(self):
        eps, base_a, base_b = _two_host_endpoints()
        assert eps.connect_endpoint("node_agent_0") == f"tcp://127.0.0.1:{base_a + 50}"
        assert eps.connect_endpoint("node_agent_1") == f"tcp://127.0.0.1:{base_b + 51}"

    def test_bind_uses_bind_addr(self):
        spec = _spec([
            {"addr": "nodeA", "gpus": [0], "bind_addr": "10.0.0.5"},
            {"addr": "nodeB", "gpus": [0]},
        ])
        eps = ControlPlaneEndpoints(spec)
        assert eps.bind_endpoint("conductor").startswith("tcp://10.0.0.5:")
        assert eps.connect_endpoint("conductor").startswith("tcp://nodeA:")
        assert eps.bind_endpoint("worker_1").startswith("tcp://0.0.0.0:")

    def test_unknown_entity_rejected(self):
        eps, _, _ = _two_host_endpoints()
        with pytest.raises(ValueError, match="no control-plane endpoint"):
            eps.connect_endpoint("mystery_service")

    def test_single_host_never_tcp(self):
        assert not ControlPlaneEndpoints(ClusterSpec.single_host()).use_tcp()
        assert not ControlPlaneEndpoints(_spec([{"addr": "a", "gpus": [0, 1]}])).use_tcp()

    def test_pickle_round_trip(self):
        # The resolver rides mp-spawn kwargs into worker processes.
        import pickle

        eps, base_a, _ = _two_host_endpoints()
        restored = pickle.loads(pickle.dumps(eps))
        assert restored.use_tcp()
        assert restored.connect_endpoint("conductor") == f"tcp://127.0.0.1:{base_a + 1}"


class TestPortValidation:
    def test_same_addr_same_base_is_fine(self):
        # Worker ports are keyed by global rank and agent ports by host index,
        # so two loopback hosts sharing one base still get disjoint ports.
        spec = _spec([
            {"addr": "127.0.0.1", "gpus": [0]},
            {"addr": "127.0.0.1", "gpus": [1]},
        ])
        ControlPlaneEndpoints(spec).validate_ports([0, 1])

    def test_overlapping_ranges_collide(self):
        # Host B's agent port (base+51) lands exactly on host A's preprocess
        # port (base+2) when the bases differ by 49 on the same address.
        spec = _spec([
            {"addr": "127.0.0.1", "gpus": [0], "zmq_port_base": 20000},
            {"addr": "127.0.0.1", "gpus": [1], "zmq_port_base": 19951},
        ])
        with pytest.raises(ValueError, match="port collision"):
            ControlPlaneEndpoints(spec).validate_ports([0, 1])

    def test_disjoint_bases_pass(self):
        eps, _, _ = _two_host_endpoints()
        eps.validate_ports([0, 1])

    def test_different_addrs_never_collide(self):
        spec = _spec([
            {"addr": "nodeA", "gpus": [0]},
            {"addr": "nodeB", "gpus": [1]},  # same base, different addr
        ])
        ControlPlaneEndpoints(spec).validate_ports([0, 1])


def _drain_until(comm, want, timeout_s=5.0):
    got = []
    deadline = time.monotonic() + timeout_s
    while len(got) < want and time.monotonic() < deadline:
        got.extend(comm.get_all_new_messages())
        time.sleep(0.01)
    return got


class TestTcpMesh:
    def test_multi_host_message_exchange(self):
        eps, _, _ = _two_host_endpoints()
        conductor = ZMQCommunicator("conductor", ["worker_0", "worker_1"], endpoints=eps)
        worker0 = ZMQCommunicator("worker_0", ["conductor", "worker_1"], endpoints=eps)
        worker1 = ZMQCommunicator("worker_1", ["conductor", "worker_0"], endpoints=eps)
        try:
            assert conductor.protocol == CommProtocol.TCP
            conductor.send("worker_1", {"kind": "work", "n": 1})
            worker1.send("conductor", {"kind": "done", "n": 2})
            worker0.send("worker_1", {"kind": "peer", "n": 3})

            w1_msgs = _drain_until(worker1, want=2)
            assert sorted(m["kind"] for m in w1_msgs) == ["peer", "work"]
            c_msgs = _drain_until(conductor, want=1)
            assert c_msgs[0] == {"kind": "done", "n": 2}
        finally:
            conductor.close()
            worker0.close()
            worker1.close()

    def test_lazy_send_to_unlisted_entity(self):
        eps, _, _ = _two_host_endpoints()
        a = ZMQCommunicator("conductor", [], endpoints=eps)
        b = ZMQCommunicator("api_server", [], endpoints=eps)
        try:
            a.send("api_server", "hello")  # push socket created on demand
            assert _drain_until(b, want=1) == ["hello"]
        finally:
            a.close()
            b.close()

    def test_resolver_overrides_env_transport(self, monkeypatch):
        monkeypatch.setenv("MSTAR_ZMQ_TRANSPORT", "IPC")
        eps, base_a, _ = _two_host_endpoints()
        comm = ZMQCommunicator("conductor", [], endpoints=eps)
        try:
            assert comm.protocol == CommProtocol.TCP
            assert comm._endpoint("api_server") == f"tcp://127.0.0.1:{base_a}"
        finally:
            comm.close()


class TestIpcMode:
    def test_single_host_resolver_stays_ipc(self, tmp_path):
        eps = ControlPlaneEndpoints(ClusterSpec.single_host())
        comm = ZMQCommunicator(
            "conductor", [], ipc_socket_path_prefix=str(tmp_path), endpoints=eps
        )
        try:
            assert comm.protocol == CommProtocol.IPC
            assert comm._endpoint("conductor").startswith("ipc://")
        finally:
            comm.close()

    def test_stale_socket_file_is_replaced(self, tmp_path):
        stale = tmp_path / "conductor.ipc"
        stale.write_bytes(b"")  # plain file where a dead instance left its socket
        comm = ZMQCommunicator("conductor", [], ipc_socket_path_prefix=str(tmp_path))
        try:
            sender = ZMQCommunicator(
                "api_server", ["conductor"], ipc_socket_path_prefix=str(tmp_path)
            )
            try:
                sender.send("conductor", "ping")
                assert _drain_until(comm, want=1) == ["ping"]
            finally:
                sender.close()
        finally:
            comm.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
