"""ArenaShmCommunicationManager (RFC #130 Step 2): producer stages tensors
into the Rust shared-memory arena, the location rides the TensorPointerInfo,
a separate consumer manager reads them zero-copy, and reclaim frees the
arena slots. Skipped unless the ``mstar_rust`` extension is installed."""
import os

import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar.communication.arena import ArenaShmCommunicationManager
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensors import (
    SharedMemoryCommunicationManager,
    create_tensor_communication_manager,
)
from mstar.graph.base import GraphEdge


class _NullCommunicator:
    """The store/register/read path under test never touches the mesh."""

    def send(self, entity_id, msg):
        raise AssertionError("unexpected control-mesh send")

    def get_all_new_messages(self):
        return []


def _manager(entity, tmp_path):
    os.environ["MSTAR_SHM_ARENA_SEGMENT_MB"] = "1"
    os.environ["MSTAR_SHM_ARENA_MAX_SEGMENTS"] = "4"
    return ArenaShmCommunicationManager(
        my_entity_id=entity, hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path),
    )


def test_producer_to_consumer_roundtrip(tmp_path):
    prod = _manager("w0", tmp_path)
    cons = _manager("w1", tmp_path)

    tensors = {
        "hidden": [torch.randn(4, 8), torch.arange(32, dtype=torch.int64)],
        "empty": [torch.empty(0, 3)],
    }
    infos = prod.store_and_return_tensor_info("r1", tensors)
    uuids = [i.uuid for il in infos.values() for i in il]
    prod.register_for_send("r1", uuids)

    # The location was stamped onto the shipped descriptors.
    for il in infos.values():
        for info in il:
            assert info.shm_segment is not None and info.shm_segment.startswith(
                "mstar_arena_w0")

    edges = [
        GraphEdge(next_node="B", name=name, tensor_info=il)
        for name, il in infos.items()
    ]
    cons.start_read_tensors("r1", edges)
    for name, originals in tensors.items():
        for original, info in zip(originals, infos[name], strict=True):
            got = cons.tensor_store.get_tensor("r1", info.uuid)
            assert torch.equal(got, original), name


def test_reclaim_frees_arena_slots(tmp_path):
    prod = _manager("w2", tmp_path)
    infos = prod.store_and_return_tensor_info(
        "r2", {"x": [torch.randn(16)]})
    (info,) = infos["x"]
    prod.register_for_send("r2", [info.uuid])
    assert prod._arena_locs
    prod._cleanup_by_uuid("r2", info.uuid)
    assert not prod._arena_locs
    assert not prod._infos_by_uuid


def test_arena_grows_then_backpressures(tmp_path):
    os.environ["MSTAR_SHM_ARENA_FULL_TIMEOUT_S"] = "0.2"
    try:
        prod = _manager("w3", tmp_path)
        # 1 MiB segments, cap 4: filling ~3.5 MiB grows the arena...
        infos = prod.store_and_return_tensor_info(
            "r3", {"big": [torch.zeros(300_000, dtype=torch.uint8)
                           for _ in range(12)]})
        uuids = [i.uuid for i in infos["big"]]
        prod.register_for_send("r3", uuids)
        assert prod._arena.num_segments > 1
        # ...and past the cap, register_for_send backpressures then fails
        # loudly instead of hanging.
        more = prod.store_and_return_tensor_info(
            "r3", {"more": [torch.zeros(300_000, dtype=torch.uint8)
                            for _ in range(6)]})
        with pytest.raises(RuntimeError, match="arena full"):
            prod.register_for_send("r3", [i.uuid for i in more["more"]])
    finally:
        del os.environ["MSTAR_SHM_ARENA_FULL_TIMEOUT_S"]


def test_transport_mismatch_fails_loudly(tmp_path):
    """A mixed deployment (arena producer + file consumer, or the reverse)
    must fail with an explicit MSTAR_SHM_ARENA message in BOTH directions,
    not a FileNotFoundError or a hang."""
    arena_prod = _manager("mx0", tmp_path)
    file_cons = SharedMemoryCommunicationManager(
        my_entity_id="mx1", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path))
    infos = arena_prod.store_and_return_tensor_info(
        "rm", {"x": [torch.randn(4)]})
    arena_prod.register_for_send("rm", [infos["x"][0].uuid])
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    with pytest.raises(RuntimeError, match="MSTAR_SHM_ARENA"):
        file_cons.start_read_tensors("rm", [edge])

    file_prod = SharedMemoryCommunicationManager(
        my_entity_id="mx2", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path))
    arena_cons = _manager("mx3", tmp_path)
    infos = file_prod.store_and_return_tensor_info(
        "rm2", {"y": [torch.randn(4)]})
    file_prod.register_for_send("rm2", [infos["y"][0].uuid])
    edge = GraphEdge(next_node="B", name="y", tensor_info=infos["y"])
    with pytest.raises(RuntimeError, match="MSTAR_SHM_ARENA"):
        arena_cons.start_read_tensors("rm2", [edge])


def test_factory_flag(tmp_path, monkeypatch):
    def make(value):
        monkeypatch.setenv("MSTAR_SHM_ARENA", value)
        return create_tensor_communication_manager(
            protocol=CommProtocol.SHM, my_entity_id=f"f_{value}",
            hostname="localhost", device="cpu",
            communicator=_NullCommunicator(), shm_dir=str(tmp_path),
        )

    assert type(make("0")) is SharedMemoryCommunicationManager
    assert type(make("1")) is ArenaShmCommunicationManager
    assert type(make("AUTO")) is ArenaShmCommunicationManager
    with pytest.raises(ValueError):
        make("yes")
