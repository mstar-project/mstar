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


def test_instance_unique_names_no_collision(tmp_path):
    """Two servers with the SAME entity id must not share /dev/shm names —
    a fixed name would let the second create() truncate the first's live
    segments (silent corruption, observed on a shared cluster)."""
    a = _manager("dup", tmp_path)
    x = torch.arange(64, dtype=torch.uint8)
    infos = a.store_and_return_tensor_info("r", {"x": [x]})
    a.register_for_send("r", list(infos["x"]))
    b = _manager("dup", tmp_path)   # same entity id, second instance
    assert b._arena.segment_name(0) != a._arena.segment_name(0)
    # a's staged data survives b's creation; the descriptor still resolves.
    cons = _manager("dupc", tmp_path)
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    cons.start_read_tensors("r", [edge])
    assert torch.equal(
        cons.tensor_store.get_tensor("r", infos["x"][0].uuid), x)


def test_orphan_sweep(tmp_path):
    """A SIGKILLed server's segments (owner pid gone) are reclaimed by the
    next construction's sweep; live owners' files are left alone."""
    dead = "/dev/shm/mstar_arena_zombie_999999999_deadbeef.seg0"
    keep = f"/dev/shm/mstar_arena_alive_{os.getpid()}_cafebabe.seg0"
    with open(dead, "wb") as f:
        f.write(b"x" * 64)
    with open(keep, "wb") as f:
        f.write(b"x")
    try:
        _manager("sweeper", tmp_path)
        assert not os.path.exists(dead), "dead-owner orphan not swept"
        assert os.path.exists(keep), "live-owner file wrongly swept"
    finally:
        for f in (dead, keep):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass


def test_dead_peer_segments_evicted(tmp_path):
    """A consumer must not accumulate mappings for peer segments whose
    backing file is gone (instance-unique names mean every producer restart
    mints NEW names — a never-evicting cache leaks a generation of mappings
    per restart)."""
    prod = _manager("evp", tmp_path)
    x = torch.arange(64, dtype=torch.uint8)
    infos = prod.store_and_return_tensor_info("re", {"x": [x]})
    prod.register_for_send("re", list(infos["x"]))
    cons = _manager("evc", tmp_path)
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    cons.start_read_tensors("re", [edge])
    seg = infos["x"][0].shm_segment
    assert seg in cons._peer_segments
    cons.pending.clear()               # no in-flight reads
    # Producer goes away gracefully: Drop unlinks its segments.
    prod._cleanup_by_uuid("re", infos["x"][0].uuid)
    del prod
    import gc

    gc.collect()
    assert not os.path.exists(f"/dev/shm/{seg}")
    cons._peer_evict_last = 0.0        # bypass the time gate
    cons.start_read_tensors("re", [])  # triggers the eviction sweep
    assert seg not in cons._peer_segments


def test_ttl_backstop_reclaims_abort_orphans(tmp_path):
    """A slot staged but never ACKed (abort) is force-freed once older
    than MSTAR_SHM_ARENA_SLOT_TTL_S, letting a full arena recover instead
    of spilling forever. Off by default."""
    os.environ["MSTAR_SHM_ARENA_SLOT_TTL_S"] = "0.05"
    try:
        prod = _manager("wt", tmp_path)
        infos = prod.store_and_return_tensor_info(
            "rt", {"x": [torch.zeros(300_000, dtype=torch.uint8)]})
        prod.register_for_send("rt", list(infos["x"]))
        assert prod._arena_locs
        import time as _t

        _t.sleep(0.06)
        assert prod._reclaim_expired() >= 1
        assert not prod._arena_locs
    finally:
        del os.environ["MSTAR_SHM_ARENA_SLOT_TTL_S"]


class _AckCapture(_NullCommunicator):
    """Collects TENSOR_RECEIVED messages instead of raising."""

    def __init__(self):
        self.acks = []

    def send(self, entity_id, msg):
        self.acks.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


def _ack_count(comm, uuid):
    return sum(msg.body.successful_tensors.get(uuid, 0)
               for _e, msg in comm.acks)


def test_shared_uuid_across_edges_acks_fanout(tmp_path):
    """One tensor consumed by TWO nodes (staggered readiness): the producer
    counted fanout 2, so both references must ACK — a bare per-uuid gate
    suppresses the second and leaks the slot forever (the soak's
    one-slot-per-request signature). A re-DELIVERY of the same edge must
    still be suppressed."""
    from mstar.communication.tensors import FutureAndPointers

    comm = _AckCapture()
    cons = ArenaShmCommunicationManager(
        my_entity_id="ack1", hostname="localhost", device="cpu",
        communicator=comm, shm_dir=str(tmp_path))
    prod = _manager("ack0", tmp_path)
    x = torch.arange(64, dtype=torch.uint8)
    infos = prod.store_and_return_tensor_info("ra", {"x": [x]})
    (info,) = infos["x"]
    prod.register_for_send("ra", [info])

    e1 = GraphEdge(next_node="B", name="x", tensor_info=[info])
    cons.start_read_tensors("ra", [e1])
    cons.get_ready_tensors()
    assert _ack_count(comm, info.uuid) == 1

    e2 = GraphEdge(next_node="C", name="x", tensor_info=[info])
    cons.start_read_tensors("ra", [e2])
    cons.get_ready_tensors()
    assert _ack_count(comm, info.uuid) == 2, "second reference not ACKed"

    cons.pending.append(FutureAndPointers(
        future=None, graph_edges=[e1], request_id="ra", rx_time=0.0))
    cons.get_ready_tensors()
    assert _ack_count(comm, info.uuid) == 2, "re-delivered edge re-ACKed"


def test_segments_unlinked_at_interpreter_exit(tmp_path):
    """A worker exits with its manager still referenced (no explicit
    cleanup path), so the Rust Drop never runs — the exit finalizer must
    unlink the segments anyway."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[2])
    code = f"""
import sys
sys.path.insert(0, {repo_root!r})
import os
os.environ["MSTAR_SHM_ARENA_SEGMENT_MB"] = "1"
os.environ["MSTAR_SHM_ARENA_MAX_SEGMENTS"] = "2"
import torch
from mstar.communication.arena import ArenaShmCommunicationManager

class _C:
    def send(self, *a): pass
    def get_all_new_messages(self): return []

m = ArenaShmCommunicationManager(
    my_entity_id="exitcase", hostname="localhost", device="cpu",
    communicator=_C(), shm_dir={repr(str(tmp_path))})
infos = m.store_and_return_tensor_info(
    "r", {{"x": [torch.arange(64, dtype=torch.uint8)]}})
m.register_for_send("r", [infos["x"][0]])
print(m._own_segment_paths[0])
KEEP_ALIVE = m   # global reference survives to interpreter exit
"""
    out = subprocess.run([sys.executable, "-c", code], check=False,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    seg_path = out.stdout.strip().splitlines()[-1]
    assert seg_path.startswith("/dev/shm/mstar_arena_exitcase_")
    assert not os.path.exists(seg_path), "segment survived interpreter exit"


def test_persisted_tensor_reclaims_at_request_end(tmp_path):
    """A tensor that is both emitted and PERSISTED must reclaim when the
    request ends. Each persist edge contributes a reference to the fanout
    (set_output_ref_counts counts routing.persist), and nothing released
    them: clearing the flag left the count above zero, so the slot leaked
    once per persisted tensor — one per request in the BAGEL soak."""
    prod = _manager("pst", tmp_path)
    rid = "rp"
    emit_edge = GraphEdge(next_node="EMIT", name="tok", tensor_info=[])
    persist_edge = GraphEdge(next_node="PERSIST", name="tok", tensor_info=[])
    gni = prod.store_and_populate_graph_edges(
        request_id=rid, tensors={"tok": [torch.arange(8)]},
        graph_edges=[emit_edge, persist_edge], node_name="LLM",
        graph_walk="w", skip_cuda_sync=True, skip_ref_count=True)
    (info,) = gni["tok"]
    prod.register_for_send(rid, [info])
    # Worker order: fanout counts first, then the persist flag is set.
    prod.set_output_ref_counts(rid, {info.uuid}, [emit_edge, persist_edge])
    prod.set_persist(rid, info.uuid, True)
    assert prod._arena_locs
    prod.dereference(rid, info.uuid, n=1)      # the emit consumer ACKs
    assert prod._arena_locs, "freed while still persisted"
    prod.cleanup_request(rid)
    assert not prod._arena_locs, "persisted tensor leaked its arena slot"


def test_unpersist_then_consume_reclaims(tmp_path):
    """The re-route path: unpersist hands the tensor to N new consumers
    (increment_ref(N) + flag cleared); once they ACK it must reclaim."""
    prod = _manager("pst2", tmp_path)
    rid = "ru"
    persist_edge = GraphEdge(next_node="PERSIST", name="kv", tensor_info=[])
    gni = prod.store_and_populate_graph_edges(
        request_id=rid, tensors={"kv": [torch.arange(8)]},
        graph_edges=[persist_edge], node_name="LLM", graph_walk="w",
        skip_cuda_sync=True, skip_ref_count=True)
    (info,) = gni["kv"]
    prod.register_for_send(rid, [info])
    prod.set_output_ref_counts(rid, {info.uuid}, [persist_edge])
    prod.set_persist(rid, info.uuid, True)
    # Conductor routes it to 2 consumers later (UNPERSIST_TENSORS).
    prod.increment_ref(rid, info.uuid, n=2)
    prod.set_persist(rid, info.uuid, False)
    assert prod._arena_locs, "freed before the new consumers read"
    prod.dereference(rid, info.uuid, n=2)      # both ACK
    assert not prod._arena_locs, "slot leaked after unpersist consumers ACKed"
