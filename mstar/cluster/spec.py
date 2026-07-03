"""Cluster topology: mapping global worker ranks onto hosts and local GPUs.

A deployment is described by an optional ``cluster:`` section in the model
config YAML:

.. code-block:: yaml

    cluster:
      hosts:
        - addr: nodeA            # routable hostname/IP for control + data planes
          gpus: [0, 1, 2, 3]     # local CUDA indices, in global-rank order
        - addr: nodeB
          gpus: [0, 1]
          zmq_port_base: 19500   # optional; per-host base for TCP control sockets

Global ranks are assigned by concatenating ``gpus`` across hosts in order:
nodeA's four GPUs are global ranks 0-3, nodeB's two GPUs are ranks 4-5. The
``ranks`` lists in ``node_groups`` continue to use these global ranks, so the
placement vocabulary is unchanged.

When the section is absent the deployment is a single host and every global
rank maps to the same-numbered local CUDA device (``rank r`` -> ``cuda:r``),
exactly matching the historical single-host behavior — including configs whose
rank sets are non-contiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_ZMQ_PORT_BASE = 19000
LOCALHOST = "localhost"

_HOST_KEYS = {"addr", "gpus", "zmq_port_base", "bind_addr", "env", "rdma_device"}


@dataclass(frozen=True)
class HostSpec:
    """One machine in the deployment."""

    addr: str  # routable hostname or IP; used for control sockets, transfer-engine sessions, and NCCL rendezvous
    gpus: tuple[int, ...]  # local CUDA indices, in global-rank order
    zmq_port_base: int = DEFAULT_ZMQ_PORT_BASE
    bind_addr: str = "0.0.0.0"  # interface to bind listening sockets on (multi-NIC hosts)
    env: dict[str, str] = field(default_factory=dict)  # extra env for this host's workers (e.g. NCCL_SOCKET_IFNAME)
    rdma_device: str = ""  # optional device filter for the RDMA transfer engine


@dataclass(frozen=True)
class WorkerSpec:
    """Where one worker (one global rank) runs."""

    worker_id: str
    global_rank: int
    host_index: int
    local_device: int  # CUDA index on its host: the worker's device is cuda:{local_device}
    addr: str  # the host's routable address


class ClusterSpec:
    """Parsed, validated view of the ``cluster:`` config section.

    ``identity_single_host=True`` is the synthesized default used when the
    config has no ``cluster:`` section: one host at ``localhost`` where any
    global rank maps to the identically-numbered CUDA device. Explicit specs
    instead enumerate every GPU and reject ranks outside the enumeration.
    """

    def __init__(self, hosts: list[HostSpec], identity_single_host: bool = False):
        if not hosts:
            raise ValueError("cluster: `hosts` must contain at least one host")
        if identity_single_host and (len(hosts) != 1 or hosts[0].gpus):
            raise ValueError("identity_single_host requires exactly one host with no explicit gpus")
        for i, host in enumerate(hosts):
            if not host.addr:
                raise ValueError(f"cluster: hosts[{i}] has an empty `addr`")
            if not identity_single_host and not host.gpus:
                raise ValueError(f"cluster: hosts[{i}] ({host.addr}) has an empty `gpus` list")
            if any((not isinstance(g, int)) or g < 0 for g in host.gpus):
                raise ValueError(f"cluster: hosts[{i}] ({host.addr}) gpus must be non-negative integers: {host.gpus}")
            if len(set(host.gpus)) != len(host.gpus):
                raise ValueError(f"cluster: hosts[{i}] ({host.addr}) lists a duplicate GPU index: {host.gpus}")

        self.hosts = list(hosts)
        self._identity = identity_single_host

        # global rank -> (host index, local CUDA index), in enumeration order.
        self._rank_map: dict[int, tuple[int, int]] = {}
        if not self._identity:
            rank = 0
            for host_idx, host in enumerate(self.hosts):
                for local_device in host.gpus:
                    self._rank_map[rank] = (host_idx, local_device)
                    rank += 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def single_host(cls) -> "ClusterSpec":
        return cls([HostSpec(addr=LOCALHOST, gpus=())], identity_single_host=True)

    @classmethod
    def from_config(cls, config: dict | None) -> "ClusterSpec":
        """Build from a loaded model-config dict; absent section => single host."""
        cluster = (config or {}).get("cluster")
        if cluster is None:
            return cls.single_host()
        if not isinstance(cluster, dict) or "hosts" not in cluster:
            raise ValueError("cluster: section must be a mapping with a `hosts` list")
        hosts = []
        for i, entry in enumerate(cluster["hosts"]):
            if not isinstance(entry, dict):
                raise ValueError(f"cluster: hosts[{i}] must be a mapping, got {type(entry).__name__}")
            unknown = set(entry) - _HOST_KEYS
            if unknown:
                raise ValueError(
                    f"cluster: hosts[{i}] has unknown key(s) {sorted(unknown)}; known keys: {sorted(_HOST_KEYS)}"
                )
            if "addr" not in entry or "gpus" not in entry:
                raise ValueError(f"cluster: hosts[{i}] must define `addr` and `gpus`")
            hosts.append(HostSpec(
                addr=str(entry["addr"]),
                gpus=tuple(entry["gpus"]),
                zmq_port_base=int(entry.get("zmq_port_base", DEFAULT_ZMQ_PORT_BASE)),
                bind_addr=str(entry.get("bind_addr", "0.0.0.0")),
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                rdma_device=str(entry.get("rdma_device", "")),
            ))
        return cls(hosts)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_multi_host(self) -> bool:
        return len(self.hosts) > 1

    @property
    def head(self) -> HostSpec:
        return self.hosts[0]

    @property
    def head_addr(self) -> str:
        return self.head.addr

    def worker_spec(self, global_rank: int) -> WorkerSpec:
        if self._identity:
            if global_rank < 0:
                raise ValueError(f"invalid global rank {global_rank}")
            return WorkerSpec(
                worker_id=f"worker_{global_rank}",
                global_rank=global_rank,
                host_index=0,
                local_device=global_rank,
                addr=self.head_addr,
            )
        if global_rank not in self._rank_map:
            raise ValueError(
                f"global rank {global_rank} is not in the cluster "
                f"(cluster enumerates ranks 0..{len(self._rank_map) - 1})"
            )
        host_idx, local_device = self._rank_map[global_rank]
        return WorkerSpec(
            worker_id=f"worker_{global_rank}",
            global_rank=global_rank,
            host_index=host_idx,
            local_device=local_device,
            addr=self.hosts[host_idx].addr,
        )

    def host_of_rank(self, global_rank: int) -> int:
        return self.worker_spec(global_rank).host_index

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_ranks(self, ranks) -> None:
        """Every rank referenced by the placement must exist in the cluster."""
        for rank in ranks:
            self.worker_spec(rank)  # raises with a precise message

    def validate_protocol(self, protocol) -> None:
        """Reject transports that cannot cross hosts.

        The SHM tensor transport moves bytes through host-local files, so it
        is only valid when every worker shares one machine.
        """
        name = getattr(protocol, "value", protocol)
        if self.is_multi_host() and str(name).upper() == "SHM":
            raise ValueError(
                "tensor-comm-protocol SHM cannot be used with a multi-host cluster; "
                "use RDMA or TCP"
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self._identity:
            return "single host (localhost), global rank == local CUDA device"
        per_host = ", ".join(f"{h.addr}: {len(h.gpus)} gpu(s)" for h in self.hosts)
        return f"{len(self.hosts)} host(s), {len(self._rank_map)} gpu(s) [{per_host}]"
