"""ArenaShmCommunicationManager: producer stages tensors
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
    prod.register_for_send("r1", [i for il in infos.values() for i in il])

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
    prod.register_for_send("r2", [info])
    assert prod._arena_locs
    prod._cleanup_by_uuid("r2", info.uuid)
    assert not prod._arena_locs
    assert not prod._infos_by_uuid


def test_arena_grows_then_spills(tmp_path):
    """Past the segment cap the producer SPILLS to the per-uuid file
    protocol (slower, never fails — the old manager's saturation behavior);
    the consumer reads spilled tensors through the file fallback; reclaim
    unlinks the files. Stats expose the fragmentation gauge."""
    os.environ["MSTAR_SHM_ARENA_SPILL_AFTER_S"] = "0.05"
    try:
        prod = _manager("w3", tmp_path)
        # 1 MiB segments, cap 4: filling ~3.5 MiB grows the arena...
        infos = prod.store_and_return_tensor_info(
            "r3", {"big": [torch.zeros(300_000, dtype=torch.uint8)
                           for _ in range(12)]})
        prod.register_for_send("r3", list(infos["big"]))
        assert prod._arena.num_segments > 1
        st = prod.stats_summary()
        assert st["segments"] == prod._arena.num_segments
        assert 0 < st["largest_free_block"] <= st["free_bytes"]
        # ...and past the cap, further tensors spill to files instead of
        # failing: shm_segment stays None and a per-uuid file appears.
        vals = [torch.full((300_000,), i, dtype=torch.uint8)
                for i in range(6)]
        more = prod.store_and_return_tensor_info("r3", {"more": vals})
        prod.register_for_send("r3", list(more["more"]))
        spilled = [i for i in more["more"] if i.shm_segment is None]
        assert spilled, "expected at least one spill past the cap"
        assert all(u in prod._shm_files for u in
                   (i.uuid for i in spilled))
        # The consumer round-trips spilled tensors via the file fallback.
        cons = _manager("w4", tmp_path)
        edge = GraphEdge(next_node="B", name="more",
                         tensor_info=more["more"])
        cons.start_read_tensors("r3", [edge])
        for val, info in zip(vals, more["more"], strict=True):
            got = cons.tensor_store.get_tensor("r3", info.uuid)
            assert torch.equal(got, val)
        # Reclaim unlinks the spilled files.
        for info in spilled:
            path = prod._shm_files[info.uuid]
            prod._cleanup_by_uuid("r3", info.uuid)
            assert not os.path.exists(path)
    finally:
        del os.environ["MSTAR_SHM_ARENA_SPILL_AFTER_S"]


def test_mixed_edge_and_fragmentation_signature(tmp_path, caplog):
    """One edge can mix arena-staged and spilled tensors (the consumer
    dispatches per descriptor), and a reserve that fails while TOTAL free
    space covers it logs the fragmentation signature (largest free block
    collapsed) before spilling."""
    import logging

    os.environ["MSTAR_SHM_ARENA_SPILL_AFTER_S"] = "0.05"
    try:
        prod = _manager("w6", tmp_path)
        # Fill to the 4-segment cap with 12 x 300 KB...
        fill = prod.store_and_return_tensor_info(
            "r6", {"fill": [torch.zeros(300_000, dtype=torch.uint8)
                            for _ in range(12)]})
        prod.register_for_send("r6", list(fill["fill"]))
        # ...then free ALTERNATE allocations: ~1.2 MB total free, but no
        # contiguous block larger than ~300 KB.
        for info in fill["fill"][::2]:
            prod._cleanup_by_uuid("r6", info.uuid)
        st = prod.stats_summary()
        assert st["free_bytes"] > 500_000 > st["largest_free_block"]

        # A small tensor fits a hole (arena); a 500 KB one has the total
        # free space but no block -> fragmentation warning, then spill.
        small = torch.arange(1000, dtype=torch.uint8)
        big = torch.full((500_000,), 7, dtype=torch.uint8)
        mixed = prod.store_and_return_tensor_info(
            "r6", {"mixed": [small, big]})
        with caplog.at_level(logging.WARNING,
                             logger="mstar.communication.arena"):
            prod.register_for_send("r6", list(mixed["mixed"]))
        s_info, b_info = mixed["mixed"]
        assert s_info.shm_segment is not None      # staged in a hole
        assert b_info.shm_segment is None          # spilled
        assert any("fragmentation" in r.message for r in caplog.records)

        # The consumer reads the MIXED edge: one from the arena, one from
        # the spill file, in a single start_read_tensors call.
        cons = _manager("w7", tmp_path)
        edge = GraphEdge(next_node="B", name="mixed",
                         tensor_info=mixed["mixed"])
        cons.start_read_tensors("r6", [edge])
        assert torch.equal(
            cons.tensor_store.get_tensor("r6", s_info.uuid), small)
        assert torch.equal(
            cons.tensor_store.get_tensor("r6", b_info.uuid), big)
    finally:
        del os.environ["MSTAR_SHM_ARENA_SPILL_AFTER_S"]


def test_strict_mode_backpressures_then_fails(tmp_path):
    """MSTAR_SHM_ARENA_SPILL=0 restores the strict contract: backpressure
    at the cap, then a loud arena-full error."""
    os.environ["MSTAR_SHM_ARENA_SPILL"] = "0"
    os.environ["MSTAR_SHM_ARENA_FULL_TIMEOUT_S"] = "0.2"
    try:
        prod = _manager("w5", tmp_path)
        infos = prod.store_and_return_tensor_info(
            "r5", {"big": [torch.zeros(300_000, dtype=torch.uint8)
                           for _ in range(12)]})
        prod.register_for_send("r5", list(infos["big"]))
        more = prod.store_and_return_tensor_info(
            "r5", {"more": [torch.zeros(300_000, dtype=torch.uint8)
                            for _ in range(6)]})
        with pytest.raises(RuntimeError, match="arena full"):
            prod.register_for_send("r5", list(more["more"]))
    finally:
        del os.environ["MSTAR_SHM_ARENA_SPILL"]
        del os.environ["MSTAR_SHM_ARENA_FULL_TIMEOUT_S"]


def test_transport_mismatch_fails_loudly(tmp_path):
    """A mixed deployment (arena producer + file consumer, or the reverse)
    fails with an explicit MSTAR_SHM_ARENA message where data would be
    unreachable (arena producer -> file consumer); the reverse direction
    interops via the arena consumer's file fallback."""
    arena_prod = _manager("mx0", tmp_path)
    file_cons = SharedMemoryCommunicationManager(
        my_entity_id="mx1", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path))
    infos = arena_prod.store_and_return_tensor_info(
        "rm", {"x": [torch.randn(4)]})
    arena_prod.register_for_send("rm", [infos["x"][0]])
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    with pytest.raises(RuntimeError, match="MSTAR_SHM_ARENA"):
        file_cons.start_read_tensors("rm", [edge])

    # The reverse direction now INTEROPS: a file-producer's tensors carry
    # no arena location, which is exactly the spill wire shape — the arena
    # consumer reads them through its file fallback.
    file_prod = SharedMemoryCommunicationManager(
        my_entity_id="mx2", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path))
    arena_cons = _manager("mx3", tmp_path)
    y = torch.randn(4)
    infos = file_prod.store_and_return_tensor_info("rm2", {"y": [y]})
    file_prod.register_for_send("rm2", [infos["y"][0]])
    edge = GraphEdge(next_node="B", name="y", tensor_info=infos["y"])
    arena_cons.start_read_tensors("rm2", [edge])
    got = arena_cons.tensor_store.get_tensor("rm2", infos["y"][0].uuid)
    assert torch.equal(got, y)


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
