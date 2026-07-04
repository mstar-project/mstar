"""Tests for per-tensor transport dispatch in multi-host deployments."""

import os
from types import SimpleNamespace
from unittest import mock

import torch

from mstar.cluster.spec import ClusterSpec
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensors import (
    HybridCommunicationManager,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    create_tensor_communication_manager,
)
from mstar.conductor.conductor import Conductor
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.worker.node_manager_utils import (
    PREPROCESS_WORKER_ENTITY,
    NodeOutputRouting,
    mark_intra_host_uuids,
)


def _info(uuid, source="worker_0"):
    return TensorPointerInfo(
        dims=[2], dtype=torch.float32, nbytes=8, address=0, stride=[1],
        uuid=uuid, source_session_id="sess", source_entity=source,
    )


def _edge(name, next_node, infos):
    return GraphEdge(next_node=next_node, name=name, tensor_info=infos)


def _routing(**kwargs):
    base = dict(
        routed_to_this_worker_graph=[], is_first_tp_rank=True,
        persist=[], to_workers={},
    )
    base.update(kwargs)
    return NodeOutputRouting(**base)


# ---------------------------------------------------------------------------
# mark_intra_host_uuids
# ---------------------------------------------------------------------------

class TestMarkIntraHostUuids:
    def test_all_consumers_co_hosted(self):
        info = _info("u1")
        routing = _routing(to_workers={"worker_1": [_edge("e", "n", [info])]})
        got = mark_intra_host_uuids(routing, {"worker_0", "worker_1"})
        assert got == {"u1"}
        assert info.via_shm is True

    def test_remote_consumer_excluded(self):
        info = _info("u1")
        routing = _routing(to_workers={"worker_2": [_edge("e", "n", [info])]})
        got = mark_intra_host_uuids(routing, {"worker_0", "worker_1"})
        assert got == set()
        assert info.via_shm is False

    def test_mixed_fanout_stays_on_engine(self):
        local_copy, remote_copy = _info("u1"), _info("u1")
        routing = _routing(to_workers={
            "worker_1": [_edge("e", "n", [local_copy])],
            "worker_2": [_edge("e", "n", [remote_copy])],
        })
        got = mark_intra_host_uuids(routing, {"worker_0", "worker_1"})
        assert got == set()
        assert local_copy.via_shm is False and remote_copy.via_shm is False

    def test_persisted_tensor_excluded(self):
        # Persisted outputs can be re-routed to any host in later walks, even
        # when this walk's only consumer is co-hosted.
        routed, persisted = _info("u1"), _info("u1")
        routing = _routing(
            persist=[_edge("e", "conductor", [persisted])],
            to_workers={"worker_1": [_edge("e", "n", [routed])]},
        )
        got = mark_intra_host_uuids(routing, {"worker_0", "worker_1"})
        assert got == set()
        assert routed.via_shm is False

    def test_emit_to_client_follows_preprocess_locality(self):
        info = _info("u1")
        routing = _routing(emit_to_client=[_edge("e", "EMIT_TO_CLIENT", [info])])
        assert mark_intra_host_uuids(routing, {"worker_0"}) == set()
        assert info.via_shm is False
        got = mark_intra_host_uuids(
            routing, {"worker_0", PREPROCESS_WORKER_ENTITY}
        )
        assert got == {"u1"}
        assert info.via_shm is True

    def test_streaming_edges_classified(self):
        info = _info("u1")
        routing = _routing(
            streaming_to_workers={"worker_1": [_edge("e", "n", [info])]}
        )
        assert mark_intra_host_uuids(routing, {"worker_0", "worker_1"}) == {"u1"}
        assert info.via_shm is True

    def test_multiple_uuids_split(self):
        near, far = _info("u1"), _info("u2")
        routing = _routing(to_workers={
            "worker_1": [_edge("e1", "n", [near])],
            "worker_9": [_edge("e2", "n", [far])],
        })
        got = mark_intra_host_uuids(routing, {"worker_0", "worker_1"})
        assert got == {"u1"}
        assert near.via_shm is True and far.via_shm is False


# ---------------------------------------------------------------------------
# TensorPointerInfo wire format
# ---------------------------------------------------------------------------

def test_via_shm_default_and_clone():
    info = _info("u1")
    assert info.via_shm is False
    assert info.clone().via_shm is False
    info.via_shm = True
    assert info.clone().via_shm is True


# ---------------------------------------------------------------------------
# Factory selection + hybrid data path (transfer engine stubbed out)
# ---------------------------------------------------------------------------

class _StubEngine:
    def __init__(self, *args, **kwargs):
        self._engine = None

    def get_session_id(self):
        return "stub:0"

    def register_memory(self, ptr, nbytes):
        return 0

    def unregister_memory(self, ptr):
        return 0

    def get_async_reader(self, device):
        return None


class _StubReader:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, read_info):
        raise AssertionError("engine read submitted for an shm-only transfer")


def _patched_engine():
    return mock.patch.multiple(
        "mstar.communication.tensors",
        MooncakeTransferEngine=_StubEngine,
        AsyncMooncakeReader=_StubReader,
    )


def _make(protocol, same_host_entities, entity="worker_0", tmpdir=None):
    return create_tensor_communication_manager(
        protocol=protocol,
        my_entity_id=entity,
        hostname="127.0.0.1",
        device="cpu",
        communicator=mock.Mock(),
        shm_dir=tmpdir,
        same_host_entities=same_host_entities,
    )


class TestFactorySelection:
    def test_shm_protocol_unchanged(self, tmp_path):
        mgr = _make(CommProtocol.SHM, {"worker_1"}, tmpdir=str(tmp_path))
        assert type(mgr) is SharedMemoryCommunicationManager

    def test_no_locality_info_keeps_mooncake(self, tmp_path):
        with _patched_engine():
            mgr = _make(CommProtocol.TCP, None, tmpdir=str(tmp_path))
        assert type(mgr) is MooncakeCommunicationManager

    def test_locality_info_upgrades_to_hybrid(self, tmp_path):
        with _patched_engine():
            for protocol in (CommProtocol.TCP, CommProtocol.RDMA):
                mgr = _make(protocol, {"worker_1"}, tmpdir=str(tmp_path))
                assert type(mgr) is HybridCommunicationManager


class TestHybridDataPath:
    def _pair(self, tmp_path):
        with _patched_engine():
            producer = _make(
                CommProtocol.TCP, {"worker_1"}, "worker_0", str(tmp_path)
            )
            consumer = _make(
                CommProtocol.TCP, {"worker_0"}, "worker_1", str(tmp_path)
            )
        return producer, consumer

    def test_shm_round_trip(self, tmp_path):
        producer, consumer = self._pair(tmp_path)
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        infos = producer.store_and_return_tensor_info("r1", {"edge": [tensor]})
        info = infos["edge"][0]
        producer.register_for_send("r1", [info.uuid], shm_uuids={info.uuid})
        path = os.path.join(str(tmp_path), f"mstar_worker_0_{info.uuid}")
        assert os.path.exists(path)

        info.via_shm = True
        consumer.start_read_tensors(
            "r1", [_edge_with(info, name="edge", next_node="n")]
        )
        ready = consumer.get_ready_tensors()
        assert [e.name for e in ready["r1"]] == ["edge"]
        assert torch.equal(consumer.get_tensor("r1", info.uuid), tensor)

    def test_send_dedup_and_cleanup(self, tmp_path):
        producer, _ = self._pair(tmp_path)
        tensor = torch.ones(4)
        infos = producer.store_and_return_tensor_info("r1", {"edge": [tensor]})
        uuid = infos["edge"][0].uuid
        producer.register_for_send("r1", [uuid], shm_uuids={uuid})
        producer.register_for_send("r1", [uuid], shm_uuids={uuid})
        path = producer._shm_files[uuid]
        # shm-sent uuids are not engine-registered, so the engine cleanup
        # path must not try to unregister them.
        assert not producer.tensor_store.is_registered("r1", uuid)
        producer._cleanup_by_uuid("r1", uuid)
        assert not os.path.exists(path)
        assert uuid not in producer._shm_files
        assert not producer.tensor_store.check_uuid_presence("r1", uuid)

    def test_engine_uuids_still_registered(self, tmp_path):
        producer, _ = self._pair(tmp_path)
        infos = producer.store_and_return_tensor_info(
            "r1", {"edge": [torch.ones(4), torch.ones(4)]}
        )
        near, far = (i.uuid for i in infos["edge"])
        producer.register_for_send("r1", [near, far], shm_uuids={near})
        assert near in producer._shm_files
        assert far not in producer._shm_files
        # Engine-path uuids take the normal registration bookkeeping.
        assert producer.tensor_store.is_registered("r1", far)


def _edge_with(info, name, next_node):
    return GraphEdge(next_node=next_node, name=name, tensor_info=[info])


# ---------------------------------------------------------------------------
# conductor gate: when workers get a same-host entity set at all
# ---------------------------------------------------------------------------

class TestSameHostEntitiesGate:
    def _conductor_like(self, spec, intra_protocol):
        return SimpleNamespace(
            cluster_spec=spec,
            intra_host_tensor_protocol=intra_protocol,
            worker_specs={
                "worker_0": SimpleNamespace(host_index=0),
                "worker_1": SimpleNamespace(host_index=1),
            },
        )

    def _two_host_spec(self):
        return ClusterSpec.from_config({
            "cluster": {"hosts": [
                {"addr": "a", "gpus": [0]},
                {"addr": "b", "gpus": [0], "zmq_port_base": 19600},
            ]}
        })

    def test_single_host_returns_none(self):
        fake = self._conductor_like(ClusterSpec.single_host(), CommProtocol.SHM)
        assert Conductor._same_host_entities(fake, 0) is None

    def test_non_shm_intra_protocol_returns_none(self):
        fake = self._conductor_like(self._two_host_spec(), CommProtocol.RDMA)
        assert Conductor._same_host_entities(fake, 0) is None
        assert Conductor._same_host_entities(fake, 1) is None

    def test_shm_intra_protocol_returns_cohosted_entities(self):
        fake = self._conductor_like(self._two_host_spec(), CommProtocol.SHM)
        assert Conductor._same_host_entities(fake, 0) == {
            "worker_0", "conductor", "api_server", "api_server_preprocess_worker",
        }
        assert Conductor._same_host_entities(fake, 1) == {"worker_1"}
