"""Deterministic control-plane endpoints for every entity in a deployment.

Each entity binds one PULL socket at a well-known address derived purely from
the cluster spec, so peers can dial each other with zero discovery traffic:

    api_server                     head host, zmq_port_base + 0
    conductor                      head host, zmq_port_base + 1
    api_server_preprocess_worker   head host, zmq_port_base + 2
    node_agent_{k}                 hosts[k],  zmq_port_base + 50 + k
    worker_{r}                     r's host,  zmq_port_base + 100 + r

Ports come from each host's own ``zmq_port_base``, so two logical hosts that
share one machine address (loopback test rigs) stay disjoint by using
different bases. Single-host deployments never consult this: their control
plane stays on ``ipc://`` sockets.
"""

from __future__ import annotations

from dataclasses import dataclass

from mstar.cluster.spec import ClusterSpec

_HEAD_OFFSETS = {"api_server": 0, "conductor": 1, "api_server_preprocess_worker": 2}
_AGENT_OFFSET = 50
_WORKER_OFFSET = 100


@dataclass(frozen=True)
class ControlPlaneEndpoints:
    """Resolves entity ids to TCP endpoints. Pickleable; shipped to workers."""

    cluster: ClusterSpec

    def use_tcp(self) -> bool:
        return self.cluster.is_multi_host()

    def _host_and_port(self, entity_id: str):
        if entity_id in _HEAD_OFFSETS:
            host = self.cluster.head
            return host, host.zmq_port_base + _HEAD_OFFSETS[entity_id]
        if entity_id.startswith("node_agent_"):
            idx_str = entity_id.removeprefix("node_agent_")
            if idx_str.isdigit() and int(idx_str) < len(self.cluster.hosts):
                idx = int(idx_str)
                host = self.cluster.hosts[idx]
                return host, host.zmq_port_base + _AGENT_OFFSET + idx
        if entity_id.startswith("worker_"):
            rank_str = entity_id.removeprefix("worker_")
            if rank_str.isdigit():
                spec = self.cluster.worker_spec(int(rank_str))
                host = self.cluster.hosts[spec.host_index]
                return host, host.zmq_port_base + _WORKER_OFFSET + spec.global_rank
        raise ValueError(f"no control-plane endpoint defined for entity {entity_id!r}")

    def connect_endpoint(self, entity_id: str) -> str:
        host, port = self._host_and_port(entity_id)
        return f"tcp://{host.addr}:{port}"

    def bind_endpoint(self, entity_id: str) -> str:
        host, port = self._host_and_port(entity_id)
        return f"tcp://{host.bind_addr}:{port}"

    def validate_ports(self, ranks) -> None:
        """Fail if two entities would bind the same (addr, port).

        Only bites when logical hosts share one machine address (loopback
        rigs); their ``zmq_port_base`` values must keep every derived port
        disjoint.
        """
        used: dict[tuple[str, int], str] = {}
        entities = (
            list(_HEAD_OFFSETS)
            + [f"node_agent_{k}" for k in range(len(self.cluster.hosts))]
            + [f"worker_{r}" for r in ranks]
        )
        for entity in entities:
            host, port = self._host_and_port(entity)
            key = (host.addr, port)
            if key in used:
                raise ValueError(
                    f"control-plane port collision: {entity!r} and {used[key]!r} both "
                    f"resolve to {host.addr}:{port}; give same-addr hosts disjoint "
                    f"zmq_port_base values"
                )
            used[key] = entity
