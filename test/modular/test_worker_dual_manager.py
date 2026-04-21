"""Worker dual-manager construction smoke tests (no GPU required).

Verifies the manager registry is set up correctly under both Mooncake-only
and NVSHMEM modes WITHOUT actually doing any transport work.
"""

import pytest
import torch

from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.communication.tensors import (
    NVSHMEMCommunicationManager,
    SharedMemoryCommunicationManager,
)
from mminf.graph.base import GraphEdge


class _NullCommunicator(BaseCommunicator):
    def send(self, entity_id, msg): pass
    def get_all_new_messages(self): return []


# ---------------------------------------------------------------------------
# resolve_transport unit tests (no Worker.__init__ needed)
# ---------------------------------------------------------------------------

def _make_resolver_with_managers(my_rank: int, world: int, has_nvshmem: bool):
    """Build a minimal stand-in for Worker.resolve_transport without
    invoking the full Worker.__init__ (which requires a Model + engines)."""
    from mminf.communication.communicator import MOONCAKE_PROTOCOLS
    from mminf.graph.special_destinations import SPECIAL_DESTINATIONS

    class _Stub:
        managers: dict
        rank: int
        _mooncake_protocol = CommProtocol.RDMA

        def resolve_transport(self, edge):
            # Verbatim copy of Worker.resolve_transport for unit testing
            # without real construction.
            from mminf.worker.worker import Worker
            return Worker.resolve_transport(self, edge)

    stub = _Stub()
    stub.managers = {p: object() for p in MOONCAKE_PROTOCOLS}
    if has_nvshmem:
        # Minimal NVSHMEM mgr stub with the attrs resolve_transport reads.
        nvshmem_mgr = type("_StubNVSHMEM", (), {})()
        nvshmem_mgr.entity_id_to_rank = {f"worker_{i}": i for i in range(world)}
        nvshmem_mgr.rank = my_rank
        stub.managers[CommProtocol.NVSHMEM] = nvshmem_mgr
    stub.rank = my_rank
    return stub


def test_resolve_transport_nvshmem_for_cross_rank():
    """AUTO edge to a different rank that's an NVSHMEM peer → NVSHMEM."""
    stub = _make_resolver_with_managers(my_rank=0, world=2, has_nvshmem=True)
    edge = GraphEdge(next_node="worker_1", name="x", transport=CommProtocol.AUTO)
    assert stub.resolve_transport(edge) == CommProtocol.NVSHMEM


def test_resolve_transport_mooncake_for_same_rank():
    """AUTO edge to same rank (self-edge) → falls through to Mooncake (Rule 4)."""
    stub = _make_resolver_with_managers(my_rank=0, world=2, has_nvshmem=True)
    edge = GraphEdge(next_node="worker_0", name="x", transport=CommProtocol.AUTO)
    assert stub.resolve_transport(edge) == CommProtocol.RDMA


def test_resolve_transport_mooncake_for_emit_to_client():
    """EMIT_TO_CLIENT (api_server) → Mooncake even with NVSHMEM installed."""
    from mminf.graph.special_destinations import EMIT_TO_CLIENT
    stub = _make_resolver_with_managers(my_rank=0, world=2, has_nvshmem=True)
    edge = GraphEdge(next_node=EMIT_TO_CLIENT, name="x", transport=CommProtocol.AUTO)
    assert stub.resolve_transport(edge) == CommProtocol.RDMA


def test_resolve_transport_explicit_nvshmem_to_special_dest_raises():
    """Explicit NVSHMEM transport to EMIT_TO_CLIENT raises (api_server isn't
    an NVSHMEM peer)."""
    from mminf.graph.special_destinations import EMIT_TO_CLIENT
    stub = _make_resolver_with_managers(my_rank=0, world=2, has_nvshmem=True)
    edge = GraphEdge(next_node=EMIT_TO_CLIENT, name="x", transport=CommProtocol.NVSHMEM)
    with pytest.raises(ValueError, match="NVSHMEM.*EMIT_TO_CLIENT|special destination"):
        stub.resolve_transport(edge)


def test_resolve_transport_explicit_nvshmem_without_manager_raises():
    """Explicit NVSHMEM on a worker without the NVSHMEM manager installed raises."""
    stub = _make_resolver_with_managers(my_rank=0, world=2, has_nvshmem=False)
    edge = GraphEdge(next_node="worker_1", name="x", transport=CommProtocol.NVSHMEM)
    with pytest.raises(ValueError, match="NVSHMEM.*manager"):
        stub.resolve_transport(edge)
