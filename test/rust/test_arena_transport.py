"""ArenaShmCommunicationManager: producer stages tensors
into the Rust shared-memory arena, the location rides the TensorPointerInfo,
a separate consumer manager reads them zero-copy, and reclaim frees the
arena slots. Skipped unless the ``mstar_rust`` extension is installed."""
import os
import sys

import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar.communication.arena import ArenaShmCommunicationManager
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
from mstar.communication.tensors import (
    SharedMemoryCommunicationManager,
    create_tensor_communication_manager,
)
from mstar.distributed.base import (
    NodeAndGraphWalk,
    ShardingConfig,
    ShardingGroup,
)
from mstar.graph.base import GraphEdge
from mstar.utils.containers import ParallelList


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
            got = cons.tensor_store.get_tensor(info.uuid)
            assert torch.equal(got, original), name


def test_reclaim_frees_arena_slots(tmp_path):
    prod = _manager("w2", tmp_path)
    infos = prod.store_and_return_tensor_info(
        "r2", {"x": [torch.randn(16)]})
    (info,) = infos["x"]
    prod.register_for_send("r2", [info])
    assert prod._arena_locs
    prod._cleanup_by_uuid(info.uuid)
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
            got = cons.tensor_store.get_tensor(info.uuid)
            assert torch.equal(got, val)
        # Reclaim unlinks the spilled files.
        for info in spilled:
            path = prod._shm_files[info.uuid]
            prod._cleanup_by_uuid(info.uuid)
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
            prod._cleanup_by_uuid(info.uuid)
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
            cons.tensor_store.get_tensor(s_info.uuid), small)
        assert torch.equal(
            cons.tensor_store.get_tensor(b_info.uuid), big)
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
    got = arena_cons.tensor_store.get_tensor(infos["y"][0].uuid)
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
        cons.tensor_store.get_tensor(infos["x"][0].uuid), x)


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


class _IpcCommunicator(_NullCommunicator):
    """A deployment's communicator: over IPC, so its manager has a namespace."""

    protocol = CommProtocol.IPC
    ipc_socket_path_prefix = "/tmp/mstar/"

    def _endpoint(self, entity_id):
        return f"ipc://{self.ipc_socket_path_prefix}/{entity_id}.ipc"


def _deployment_manager(entity, shm_dir, token, monkeypatch):
    monkeypatch.setenv("MSTAR_DEPLOYMENT_TOKEN", token)
    os.environ["MSTAR_SHM_ARENA_SEGMENT_MB"] = "1"
    os.environ["MSTAR_SHM_ARENA_MAX_SEGMENTS"] = "4"
    return ArenaShmCommunicationManager(
        my_entity_id=entity, hostname="localhost", device="cpu",
        communicator=_IpcCommunicator(), shm_dir=str(shm_dir),
    )


def test_a_start_in_another_pid_namespace_keeps_live_segments(
        monkeypatch, tmp_path):
    """Containers sharing /dev/shm cannot see each other's pids: a second
    deployment's startup used to unlink every live segment of the first."""
    a = _deployment_manager("w0", tmp_path, "a" * 16, monkeypatch)
    x = torch.arange(64, dtype=torch.uint8)
    infos = a.store_and_return_tensor_info("r", {"x": [x]})
    a.register_for_send("r", list(infos["x"]))
    seg_path = a._own_segment_paths[0]

    real_exists, hidden = os.path.exists, f"/proc/{os.getpid()}"
    monkeypatch.setattr(
        os.path, "exists", lambda p: p != hidden and real_exists(p))
    _deployment_manager("w0", tmp_path, "b" * 16, monkeypatch)
    monkeypatch.setattr(os.path, "exists", real_exists)

    assert os.path.exists(seg_path), "a live neighbour's segment was swept"
    cons = _deployment_manager("w1", tmp_path, "a" * 16, monkeypatch)
    cons.start_read_tensors(
        "r", [GraphEdge(next_node="B", name="x", tensor_info=infos["x"])])
    assert torch.equal(cons.tensor_store.get_tensor(infos["x"][0].uuid), x)


def test_a_dead_namespaces_segments_are_swept(monkeypatch, tmp_path):
    dead = "d" * 12
    lock = f"/dev/shm/mstar_{dead}.lock"
    seg = f"/dev/shm/mstar_{dead}_arena_w0_deadbeef.seg0"
    open(lock, "wb").close()
    with open(seg, "wb") as f:
        f.write(b"x" * 64)
    try:
        _deployment_manager("sweeper", tmp_path, "c" * 16, monkeypatch)
        assert not os.path.exists(seg), "a dead namespace's segment was kept"
        assert not os.path.exists(lock)
    finally:
        for p in (seg, lock):
            if os.path.exists(p):
                os.unlink(p)


def test_a_dead_process_in_another_shm_dir_keeps_live_segments(
        monkeypatch, tmp_path):
    """One deployment's processes can use different shm_dirs: the unheld lock
    a dead one leaves in its shm_dir must not let a sweep take the live
    segments in /dev/shm."""
    live_dir, dead_dir = tmp_path / "live", tmp_path / "dead"
    live_dir.mkdir()
    dead_dir.mkdir()
    a = _deployment_manager("w0", live_dir, "e" * 16, monkeypatch)
    infos = a.store_and_return_tensor_info(
        "r", {"x": [torch.arange(64, dtype=torch.uint8)]})
    a.register_for_send("r", list(infos["x"]))
    seg_path = a._own_segment_paths[0]
    (dead_dir / f"mstar_{a._shm_namespace}.lock").touch()

    _deployment_manager("w0", dead_dir, "f" * 16, monkeypatch)
    assert os.path.exists(seg_path), "a live process's segment was swept"


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
    prod._cleanup_by_uuid(infos["x"][0].uuid)
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


def test_register_for_send_uuids_matches_the_per_request_loop(tmp_path):
    """The uuid-driven form stages a whole batch in one pass and syncs its D2H
    stream once instead of once per request. It must stage exactly what B
    separate register_for_send calls would."""
    looped = _manager("w0", tmp_path)
    batched = _manager("w1", tmp_path)
    cons = _manager("w2", tmp_path)

    def _stage(mgr, use_batch):
        per_request = {}
        for rid in (10, 11, 12):
            infos = mgr.store_and_return_tensor_info(
                rid, {"h": [torch.randn(4, 8)], "e": [torch.empty(0, 3)]},
            )
            per_request[rid] = [i for il in infos.values() for i in il]
        if use_batch:
            mgr.register_for_send_uuids(ParallelList(
                list(per_request),
                [[i.uuid for i in infos] for infos in per_request.values()],
            ))
        else:
            for rid, infos in per_request.items():
                mgr.register_for_send(rid, infos)
        return per_request

    torch.manual_seed(0)
    loop_out = _stage(looped, use_batch=False)
    torch.manual_seed(0)
    batch_out = _stage(batched, use_batch=True)

    for rid in loop_out:
        for a, b in zip(loop_out[rid], batch_out[rid], strict=True):
            # Read back from the store: the uuid form stamps placement there,
            # and a Rust bookkeeper holds a copy, not the object returned above.
            a = looped.tensor_store.get_info(a.uuid)
            b = batched.tensor_store.get_info(b.uuid)
            # Staged into the arena (not spilled) in both, at the same offset.
            assert (a.shm_segment is None) == (b.shm_segment is None)
            assert a.shm_offset == b.shm_offset
            assert a.dims == b.dims and a.nbytes == b.nbytes
            assert looped.tensor_store.is_registered(a.uuid)
            assert batched.tensor_store.is_registered(b.uuid)

    # And the bytes actually survive a read by a third party.
    edges = [GraphEdge(name="h", next_node="n", tensor_info=[
        batched.tensor_store.get_info(batch_out[10][0].uuid)
    ])]
    cons.start_read_tensors(10, edges)
    assert torch.equal(
        cons.tensor_store.get_tensor(batch_out[10][0].uuid),
        batched.tensor_store.get_tensor(batch_out[10][0].uuid),
    )


# --- the stamp has to reach the STORE, not just the caller's object ---------

@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("batched", [False, True])
def test_the_arena_location_is_readable_back_out_of_the_store(
    tmp_path, backend, batched
):
    """Staging stamps ``shm_segment`` in place; the store has to see it.

    The wire descriptor is rebuilt from the store by uuid (RouteOutput carries
    indices, not objects), so a stamp that only lands on the caller's object
    ships ``shm_segment=None`` -- which a consumer reads as "spilled to a file"
    and goes looking for a file nobody wrote. The Python bookkeeper hides this
    by keeping the caller's object; the Rust one copies it in.
    """
    prod = _manager(f"wb_{backend}_{int(batched)}", tmp_path)
    if backend == "rust":
        prod.tensor_store = TensorStore(RustTensorBookkeeping())

    infos = prod.store_and_return_tensor_info(7, {"h": [torch.randn(4, 8)]})
    flat = [i for il in infos.values() for i in il]
    if batched:
        prod.register_for_send_uuids(ParallelList([7], [[i.uuid for i in flat]]))
    else:
        prod.register_for_send(7, flat)

    stored = prod.tensor_store.get_info(flat[0].uuid)
    assert stored.shm_segment is not None, "staged, so it must carry a segment"
    if not batched:
        # Only the descriptor form hands the caller's object in to be stamped;
        # the uuid form has none, and under the Rust bookkeeper ``flat`` is a
        # copy the store never sees again.
        assert stored.shm_segment == flat[0].shm_segment
        assert stored.shm_offset == flat[0].shm_offset


# --- placement crosses as an index, not a segment name ----------------------

@pytest.mark.parametrize("backend", ["python", "rust"])
def test_placement_crosses_as_an_index_once_per_segment(tmp_path, backend):
    """``_reserve`` already returns the segment INDEX, so rendering its name
    per tensor only for the store to intern it straight back down to an id is
    a Python str alloc, a String alloc, a memcpy and a hash per tensor for an
    integer the caller already had.
    """
    prod = _manager(f"idx_{backend}", tmp_path)
    if backend == "rust":
        prod.tensor_store = TensorStore(RustTensorBookkeeping())

    store = prod.tensor_store
    named: list[int] = []
    placed: list[list[int]] = []
    real_name, real_place = store.register_shm_segment, store.set_shm_placement

    def spy_name(index, name):
        named.append(index)
        return real_name(index, name)

    def spy_place(uuids, segment_idxs, offsets):
        placed.append(list(segment_idxs))
        return real_place(uuids, segment_idxs, offsets)

    store.register_shm_segment = spy_name
    store.set_shm_placement = spy_place

    infos = prod.store_and_return_tensor_info(
        7, {"h": [torch.randn(4, 8) for _ in range(16)]},
    )
    flat = [i for il in infos.values() for i in il]
    assert len(flat) == 16
    prod.register_for_send_uuids(ParallelList([7], [[i.uuid for i in flat]]))

    [idxs] = placed
    assert len(idxs) == 16
    assert all(isinstance(i, int) for i in idxs), \
        "the segment has to cross as an index, not a name"
    assert len(named) <= prod._arena.num_segments, (
        f"named a segment {len(named)} times for 16 tensors; naming is once "
        "per segment, at the grow"
    )
    # And the placement still landed: the store resolved every index.
    for info in flat:
        assert store.get_info(info.uuid).shm_segment is not None


def test_growth_names_only_new_segments_under_their_own_index(tmp_path):
    """Two things the test above cannot see, because its 128-byte tensors
    never grow the arena past one segment:

    1. a grow must name only what it added -- re-naming the accumulated list
       each time is quadratic in segment count;
    2. each index must carry ITS OWN name. Renumbering the whole list from
       the new start files every existing segment under the wrong index, so
       a tensor placed in segment N is later read out of segment 0's file.
    """
    prod = _manager("w_grow", tmp_path)
    store = prod.tensor_store
    named: list[tuple[int, str]] = []
    real_name = store.register_shm_segment

    def spy_name(index, name):
        named.append((index, name))
        return real_name(index, name)

    store.register_shm_segment = spy_name

    # 1 MiB segments: 12 x 300 KB crosses several of them.
    infos = prod.store_and_return_tensor_info(
        "rg", {"big": [torch.zeros(300_000, dtype=torch.uint8)
                       for _ in range(12)]},
    )
    prod.register_for_send("rg", list(infos["big"]))
    assert prod._arena.num_segments > 1, "need a grown arena to test this"

    idxs = [i for i, _ in named]
    assert len(idxs) == len(set(idxs)), (
        f"re-named an already-named segment: {idxs}; naming is once per "
        "segment, at the grow"
    )
    for i, name in named:
        assert name == prod._seg_names[i], (
            f"segment {i} was registered as {name!r}, but it is "
            f"{prod._seg_names[i]!r}"
        )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_an_unregistered_segment_index_is_a_loud_error(backend):
    """Silently stamping ``shm_segment=None`` would read as "spilled to a
    file" at the consumer, which then hunts for a file nobody wrote -- a
    failure that surfaces far from here.

    Checked before the uuid lookup, so both backends raise on a bad index
    whether or not the tensor is still tracked.
    """
    store = TensorStore(
        RustTensorBookkeeping() if backend == "rust" else None
    )
    with pytest.raises(ValueError, match="never registered"):
        store.set_shm_placement([12345], [99], [0])


# --- a sharded edge keeps the placement its clone was made before ------------


class _RecordingCommunicator:
    """Keeps the TENSOR_RECEIVED acks a consumer sends on read."""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


def _tp1_to_tp2_config() -> ShardingConfig:
    """A sharded signal from a single-rank producer into a two-rank consumer,
    so the fanout slices and therefore CLONES the descriptor."""
    cfg = ShardingConfig(
        groups=[
            ShardingGroup(
                nodes=["A"], tp_size=1, graph_walks=["decode"], _tp_rank=0,
            ),
            ShardingGroup(nodes=["B"], tp_size=2, graph_walks=["decode"]),
        ],
        tp_enabled_nodes={"A", "B"},
        shard_dim={"x": 0},
    )
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["worker_0"],
        NodeAndGraphWalk("B", "decode"): ["worker_1", "worker_2"],
    })
    return cfg


def test_a_sharded_edge_still_names_its_arena_segment(tmp_path):
    """The wire descriptor for a sharded edge has to carry the placement.

    ``fanout_graph_edges`` clones the descriptor per destination to rewrite
    dims/nbytes/offset, and it clones during ROUTING -- before staging stamps
    ``shm_segment`` onto the copy the store holds, which is the only copy
    ``register_for_send_uuids`` can reach. Left alone the clone ships
    ``shm_segment=None``, which a consumer reads as "spilled to a per-uuid
    file" and then opens a file the arena producer never wrote.
    """
    prod = _manager("worker_0", tmp_path)
    cfg = _tp1_to_tp2_config()
    prod.register_request(7, cfg)

    original = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    infos = prod.store_and_return_tensor_info(
        7, {"x": [original]}, node_name="A", graph_walk="decode",
    )
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    fanout = cfg.fanout_graph_edges(
        edge, source_node="A", source_graph_walk="decode",
        dest_graph_walk="decode",
    )
    assert sorted(fanout) == ["worker_1", "worker_2"]

    uuids = sorted({i.uuid for e in fanout.values() for i in e.tensor_info})
    prod.register_for_send_uuids(ParallelList([7], [uuids]))
    # What the runtime does at each point an edge leaves the process.
    for wire_edge in fanout.values():
        prod.refresh_shm_placement([wire_edge])

    for worker, wire_edge in sorted(fanout.items()):
        for info in wire_edge.tensor_info:
            assert info.shm_segment is not None, (
                f"{worker}'s descriptor has no segment, so its consumer would "
                "look for a per-uuid file that was never written"
            )
    # The refresh must touch ONLY the placement: everything else on the clone
    # is this destination's slice.
    assert [
        (i.offset, i.nbytes, tuple(i.dims))
        for _w, e in sorted(fanout.items()) for i in e.tensor_info
    ] == [(0, 64, (2, 8)), (64, 64, (2, 8))]


def test_a_peer_reads_its_own_half_of_a_sharded_edge(tmp_path):
    """End to end: the descriptor the refresh produced is one a peer can
    actually read its shard out of."""
    prod = _manager("worker_0", tmp_path)
    cfg = _tp1_to_tp2_config()
    prod.register_request(7, cfg)

    original = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    infos = prod.store_and_return_tensor_info(
        7, {"x": [original]}, node_name="A", graph_walk="decode",
    )
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos["x"])
    fanout = cfg.fanout_graph_edges(
        edge, source_node="A", source_graph_walk="decode",
        dest_graph_walk="decode",
    )
    uuids = sorted({i.uuid for e in fanout.values() for i in e.tensor_info})
    prod.register_for_send_uuids(ParallelList([7], [uuids]))
    for wire_edge in fanout.values():
        prod.refresh_shm_placement([wire_edge])

    # Each rank of B reads the half the fanout cut for it. `shm_offset` is the
    # tensor's base in the segment and `offset` the shard's cut within it, so
    # rank 1 only lands on the right rows if both survived the refresh.
    for rank, (worker, expected) in enumerate(
        [("worker_1", original[:2]), ("worker_2", original[2:])]
    ):
        cons = _manager(worker, tmp_path)
        # get_ready_tensors acks the producer; _NullCommunicator refuses sends.
        cons.communicator = _RecordingCommunicator()
        wire_edge = fanout[worker]
        for f in cons.start_read_tensors(7, [wire_edge], graph_walk="decode") or []:
            f.result()
        ready = cons.get_ready_tensors(graph_walk="decode")
        assert 7 in ready and len(ready[7]) == 1, f"rank {rank} read nothing"
        got = cons.get_tensor(ready[7][0].tensor_info[0].uuid)
        assert torch.equal(got, expected), f"rank {rank} read the wrong rows"


@pytest.mark.parametrize("shape", [
    pytest.param((4, 3), id="rows"),
    # Zero-byte rows: both slices are offset 0, nbytes 0; only dims differ.
    pytest.param((4, 0), id="empty-rows"),
])
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_two_slices_of_one_tensor_on_one_worker_do_not_alias(tmp_path, backend, shape):
    """``x`` goes from a TP=1 producer to a TP=2 ``B`` and a TP=4 ``C``, both
    with rank 0 on worker_1, which so receives rows [0, 2) and row 0 under one
    producer uuid. Keyed by the uuid alone, ``C`` got ``B``'s rows."""
    cfg = ShardingConfig(
        groups=[
            ShardingGroup(nodes=["A"], tp_size=1, graph_walks=["decode"], _tp_rank=0),
            ShardingGroup(nodes=["B"], tp_size=2, graph_walks=["decode"]),
            ShardingGroup(nodes=["C"], tp_size=4, graph_walks=["decode"]),
        ],
        tp_enabled_nodes={"A", "B", "C"},
        shard_dim={"x": 0},
    )
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["worker_0"],
        NodeAndGraphWalk("B", "decode"): ["worker_1", "worker_2"],
        NodeAndGraphWalk("C", "decode"): [f"worker_{i}" for i in range(1, 5)],
    })
    prod = _manager("worker_0", tmp_path)
    cons = _manager("worker_1", tmp_path)
    cons.communicator = _RecordingCommunicator()
    if backend == "rust":
        cons.tensor_store = TensorStore(RustTensorBookkeeping())
    for mgr in (prod, cons):
        mgr.register_request(7, cfg)

    x = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    (info,) = prod.store_and_return_tensor_info(
        7, {"x": [x]}, node_name="A", graph_walk="decode",
    )["x"]
    edges = [
        cfg.fanout_graph_edges(
            GraphEdge(next_node=node, name="x", tensor_info=[info]),
            source_node="A", source_graph_walk="decode", dest_graph_walk="decode",
        )["worker_1"]
        for node in ("B", "C")
    ]
    prod.register_for_send_uuids(ParallelList([7], [[info.uuid]]))
    prod.refresh_shm_placement(edges)

    for fut in cons.start_read_tensors(7, edges, graph_walk="decode") or []:
        fut.result()
    ready = cons.get_ready_tensors(graph_walk="decode")[7]
    got = {e.next_node: cons.get_tensor(e.tensor_info[0].uuid) for e in ready}
    assert torch.equal(got["B"], x[:2])
    assert torch.equal(got["C"], x[:1]), "C got the slice stored for B"
    assert [msg.body.successful_tensors for _, msg in cons.communicator.sent] == [
        {info.uuid: 2}
    ]
    for edge in ready:
        cons.dereference(edge.tensor_info[0].uuid)
    assert cons.tensor_store.get_all_uuids(7) == []


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_an_edge_mixing_a_reused_and_a_new_slice_is_read_in_full(tmp_path, backend):
    """``x`` holds a tensor with rows and one with none. ``C``'s slice of the
    empty one is ``B``'s, so it is reused while its other slice is new: the
    new one still drops its transit ref."""
    cfg = ShardingConfig(
        groups=[
            ShardingGroup(nodes=["A"], tp_size=1, graph_walks=["decode"], _tp_rank=0),
            ShardingGroup(nodes=["B"], tp_size=2, graph_walks=["decode"]),
            ShardingGroup(nodes=["C"], tp_size=4, graph_walks=["decode"]),
        ],
        tp_enabled_nodes={"A", "B", "C"},
        shard_dim={"x": 0},
    )
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["worker_0"],
        NodeAndGraphWalk("B", "decode"): ["worker_1", "worker_2"],
        NodeAndGraphWalk("C", "decode"): [f"worker_{i}" for i in range(1, 5)],
    })
    prod = _manager("worker_0", tmp_path)
    cons = _manager("worker_1", tmp_path)
    cons.communicator = _RecordingCommunicator()
    if backend == "rust":
        cons.tensor_store = TensorStore(RustTensorBookkeeping())
    for mgr in (prod, cons):
        mgr.register_request(7, cfg)

    full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    infos = prod.store_and_return_tensor_info(
        7, {"x": [full, torch.zeros(0, 3)]}, node_name="A", graph_walk="decode",
    )["x"]
    edges = [
        cfg.fanout_graph_edges(
            GraphEdge(next_node=node, name="x", tensor_info=infos),
            source_node="A", source_graph_walk="decode", dest_graph_walk="decode",
        )["worker_1"]
        for node in ("B", "C")
    ]
    prod.register_for_send_uuids(ParallelList([7], [[i.uuid for i in infos]]))
    prod.refresh_shm_placement(edges)

    for fut in cons.start_read_tensors(7, edges, graph_walk="decode") or []:
        fut.result()
    ready = cons.get_ready_tensors(graph_walk="decode")[7]
    got = {e.next_node: cons.get_tensor(e.tensor_info[0].uuid) for e in ready}
    assert torch.equal(got["B"], full[:2])
    assert torch.equal(got["C"], full[:1])
    for edge in ready:
        for info in edge.tensor_info:
            cons.dereference(info.uuid)
    assert cons.tensor_store.get_all_uuids(7) == [], "a slice kept its transit ref"


def test_arena_stages_the_stored_host_copy(tmp_path):
    """A stored host copy is what lands in the arena (a host memcpy, no D2H);
    it differs from the stored tensor only so the test can tell."""
    prod = _manager("w0", tmp_path)
    cons = _manager("w1", tmp_path)
    assert prod.needs_cpu_tensor

    host = torch.arange(4, dtype=torch.int64)
    infos = prod.store_and_return_tensor_info(
        "r1", {"tok": [torch.zeros(4, dtype=torch.int64)]},
        cpu_tensors={"tok": [host]},
    )
    prod.register_for_send("r1", infos["tok"])
    cons.start_read_tensors(
        "r1", [GraphEdge(next_node="B", name="tok", tensor_info=infos["tok"])],
    )
    assert torch.equal(cons.tensor_store.get_tensor(infos["tok"][0].uuid), host)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32, torch.bool])
def test_a_cpu_read_owns_its_bytes_past_reclaim(tmp_path, dtype):
    """A CPU consumer's tensor is a copy: the producer reclaiming the slot and
    staging another tensor into it must not change what was read."""
    prod = _manager("w0", tmp_path)
    cons = _manager("w1", tmp_path)
    original = (torch.arange(3 * 16 * 24) % 7 == 3) if dtype == torch.bool else (
        torch.arange(3 * 16 * 24) % 251).to(dtype)
    original = original.reshape(1, 3, 16, 24)
    infos = prod.store_and_return_tensor_info("r1", {"img": [original]})
    prod.register_for_send("r1", infos["img"])
    (info,) = infos["img"]
    cons.start_read_tensors("r1", [GraphEdge(next_node="B", name="img", tensor_info=infos["img"])])
    got = cons.tensor_store.get_tensor(info.uuid)
    assert got.dtype == dtype and got.shape == original.shape and torch.equal(got, original)

    location = prod._arena_locs[info.uuid]
    prod._cleanup_by_uuid(info.uuid)
    again = prod.store_and_return_tensor_info("r2", {"img": [torch.zeros_like(original)]})
    prod.register_for_send("r2", again["img"])
    assert prod._arena_locs[again["img"][0].uuid] == location, "the test needs the slot reused"
    assert torch.equal(got, original), "the read aliased the producer's slot"


def _is_mapped(ptr: int) -> bool:
    with open("/proc/self/maps") as f:
        for line in f:
            lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
            if lo <= ptr < hi:
                return True
    return False


@pytest.fixture
def fake_pins(monkeypatch):
    """Stand-in for cudaHostRegister/Unregister on a CPU host: records each
    unregister and whether the range was still mapped at that moment."""
    from mstar.communication import arena as arena_mod

    pinned, unpinned = [], []

    def pin(ptr, nbytes):
        pinned.append(ptr)
        return True

    def unpin(ptr):
        unpinned.append((ptr, _is_mapped(ptr)))
        return True

    monkeypatch.setattr(arena_mod, "_pin", pin)
    monkeypatch.setattr(arena_mod, "_unpin", unpin)
    return pinned, unpinned


def test_own_pinned_segments_are_unregistered_before_the_arena_unmaps(
        tmp_path, fake_pins):
    import gc

    pinned, unpinned = fake_pins
    prod = _manager("pin_own", tmp_path)
    prod._pin_segments = True
    for i in range(2):  # the second tensor grows a pinned segment
        infos = prod.store_and_return_tensor_info(
            f"r{i}", {"x": [torch.zeros(700_000, dtype=torch.uint8)]})
        prod.register_for_send(f"r{i}", list(infos["x"]))
    assert prod._arena.num_segments == 2 and len(pinned) == 1
    del prod, infos
    gc.collect()
    assert unpinned == [(pinned[0], True)]


def test_close_unregisters_own_and_peer_segments_once(tmp_path, fake_pins):
    import gc

    pinned, unpinned = fake_pins
    prod = _manager("pin_prod", tmp_path)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [torch.arange(64, dtype=torch.uint8)]})
    prod.register_for_send("r", list(infos["x"]))
    cons = _manager("pin_cons", tmp_path)
    cons._pin_segments = True
    cons.start_read_tensors(
        "r", [GraphEdge(next_node="B", name="x", tensor_info=infos["x"])])
    assert len(pinned) == 1 and cons._pinned_bytes > 0

    cons.close()
    assert unpinned == [(pinned[0], True)]
    assert cons._pinned_bytes == 0
    assert all(p == 0 for _, _, p in cons._peer_segments.values())
    del cons
    gc.collect()
    assert len(unpinned) == 1


def test_close_during_a_segment_registration_still_unregisters_it(
        tmp_path, monkeypatch):
    import threading

    from mstar.communication import arena as arena_mod

    prod = _manager("pin_race", tmp_path)
    prod._pin_segments = True
    unpinned, closer = [], []

    def pin(ptr, nbytes):
        t = threading.Thread(target=prod.close)
        t.start()
        t.join(0.2)
        assert t.is_alive(), "close() did not wait for the registration"
        closer.append(t)
        return True

    monkeypatch.setattr(arena_mod, "_pin", pin)
    monkeypatch.setattr(arena_mod, "_unpin",
                        lambda ptr: unpinned.append(ptr) or True)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [torch.zeros(1_500_000, dtype=torch.uint8)]})
    prod.register_for_send("r", list(infos["x"]))
    closer[0].join(5)
    assert unpinned == [prod._arena.segment(1).ptr_len()[0]]
    assert prod._pinned_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_host_registration_outlives_a_dropped_arena(tmp_path):
    import gc

    from mstar.platforms import current_platform

    os.environ["MSTAR_SHM_ARENA_SEGMENT_MB"] = "1"
    os.environ["MSTAR_SHM_ARENA_MAX_SEGMENTS"] = "4"
    m = ArenaShmCommunicationManager(
        my_entity_id="pin_gpu", hostname="localhost", device="cuda",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path),
    )
    (ptr,) = m._own_pinned
    del m
    gc.collect()
    # succeeds only if the registration was left behind
    assert not current_platform.host_unregister(ptr)


def _bigger_than_dev_shm() -> int:
    """A size no segment in /dev/shm can have: preallocating it fails with
    ENOSPC up front, without filling the real /dev/shm."""
    if not sys.platform.startswith("linux") or not os.path.isdir("/dev/shm"):
        pytest.skip("Linux /dev/shm only")
    st = os.statvfs("/dev/shm")
    if not st.f_blocks:
        pytest.skip("/dev/shm has no size limit")
    return st.f_frsize * st.f_blocks + (1 << 20)


class _NoRoomToGrow:
    """The producer's arena, except that every reserve asks for a segment
    /dev/shm cannot hold: what growth meets once /dev/shm is full."""

    def __init__(self, arena, too_big):
        self._arena, self._too_big = arena, too_big

    def reserve(self, nbytes):
        return self._arena.reserve(self._too_big)

    def __getattr__(self, name):
        return getattr(self._arena, name)


@pytest.mark.parametrize("spill", ["1", "0"])
def test_no_room_to_grow_spills_or_fails_clearly(tmp_path, monkeypatch, spill):
    monkeypatch.setenv("MSTAR_SHM_ARENA_SPILL", spill)
    monkeypatch.setenv("MSTAR_SHM_ARENA_FULL_TIMEOUT_S", "0.05")
    prod = _manager("w0", tmp_path)
    cons = _manager("w1", tmp_path)
    prod._arena = _NoRoomToGrow(prod._arena, _bigger_than_dev_shm())
    t = torch.arange(64, dtype=torch.int64)
    infos = prod.store_and_return_tensor_info("r1", {"x": [t]})
    if spill == "0":
        with pytest.raises(RuntimeError, match="arena full.*no space"):
            prod.register_for_send("r1", infos["x"])
        return
    prod.register_for_send("r1", infos["x"])
    (info,) = infos["x"]
    assert info.shm_segment is None and info.uuid in prod._shm_files
    cons.start_read_tensors(
        "r1", [GraphEdge(next_node="B", name="x", tensor_info=infos["x"])])
    assert torch.equal(cons.tensor_store.get_tensor(info.uuid), t)


# Two producers and a consumer whose per-entity ceilings (3 x 4 MiB) each fit
# a 16 MiB /dev/shm, but whose staged tensors together do not. Spills go to
# argv[1], off the small tmpfs, so they can be read back; the last step spills
# into the full tmpfs itself, the default spill directory.
_SMALL_DEV_SHM_SCRIPT = """
import errno, os, sys
os.environ.update(MSTAR_SHM_ARENA_SEGMENT_MB="4", MSTAR_SHM_ARENA_MAX_SEGMENTS="3")
import torch
from mstar.communication.arena import ArenaShmCommunicationManager
from mstar.graph.base import GraphEdge

class Comm:
    def send(self, *a):
        raise AssertionError
    def get_all_new_messages(self):
        return []

def manager(entity):
    return ArenaShmCommunicationManager(
        my_entity_id=entity, hostname="localhost", device="cpu",
        communicator=Comm(), shm_dir=sys.argv[1])

prods, cons = [manager("p0"), manager("p1")], manager("c0")
sent, spilled = {}, 0
for p in prods:
    for i in range(3):
        t = torch.full((3_500_000,), i + 1, dtype=torch.uint8)
        (info,) = p.store_and_return_tensor_info("r", {"x": [t]})["x"]
        p.register_for_send("r", [info])
        spilled += info.shm_segment is None
        sent[info.uuid] = (t, info)
edge = GraphEdge(next_node="B", name="x", tensor_info=[i for _, i in sent.values()])
cons.start_read_tensors("r", [edge])
assert all(torch.equal(cons.tensor_store.get_tensor(u), t) for u, (t, _) in sent.items())
try:
    manager("late")
except RuntimeError as e:
    assert "no space" in str(e), e
    assert not [n for n in os.listdir("/dev/shm") if "_late_" in n]
else:
    raise AssertionError("an arena was created in a full /dev/shm")
prods[1].shm_dir = "/dev/shm"
(info,) = prods[1].store_and_return_tensor_info("r", {"x": [torch.ones(3_500_000)]})["x"]
try:
    prods[1].register_for_send("r", [info])
except OSError as e:
    assert e.errno == errno.ENOSPC, e
else:
    raise AssertionError("a spill into a full /dev/shm succeeded")
print(f"spilled={spilled}")
"""


def _private_small_dev_shm_cmd(inner: str) -> list[str]:
    """Run ``inner`` in a private mount namespace whose /dev/shm is a 16 MiB
    tmpfs, so the host's /dev/shm is never touched. Needs passwordless sudo."""
    return ["sudo", "-n", "unshare", "-m", "sh", "-c",
            "mount -t tmpfs -o size=16m,mode=1777 tmpfs /dev/shm && " + inner]


def _run_in_small_dev_shm(tmp_path, source: str, *args: str):
    """Run ``source`` as this user, in a private 16 MiB /dev/shm; skip where
    that mount is not possible."""
    import shlex
    import shutil
    import subprocess

    if not sys.platform.startswith("linux") or not all(
            shutil.which(b) for b in ("sudo", "unshare", "setpriv")):
        pytest.skip("needs Linux with sudo, unshare and setpriv")
    probe = subprocess.run(_private_small_dev_shm_cmd("true"), check=False,
                           capture_output=True, timeout=30)
    if probe.returncode:
        pytest.skip("cannot mount a private tmpfs: "
                    f"{probe.stderr.decode(errors='replace').strip()}")
    script = tmp_path / "small_dev_shm.py"
    script.write_text(source)
    env = [f"{k}={v}" for k, v in os.environ.items()
           if not k.startswith("MSTAR_SHM_ARENA")]
    inner = shlex.join([
        "setpriv", f"--reuid={os.getuid()}", f"--regid={os.getgid()}",
        "--clear-groups", "env", "-i", *env,
        sys.executable, str(script), *args])
    proc = subprocess.run(_private_small_dev_shm_cmd("exec " + inner),
                          check=False, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc


def test_a_full_dev_shm_spills_instead_of_crashing(tmp_path):
    """Real tmpfs exhaustion: before segments were preallocated, the second
    producer's copy into a sparse segment died with SIGBUS."""
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    proc = _run_in_small_dev_shm(tmp_path, _SMALL_DEV_SHM_SCRIPT, str(spill_dir))
    spilled = int(proc.stdout.strip().rsplit("=", 1)[1])
    assert spilled > 0, proc.stdout


# /dev/shm keeps 3 MiB free, less than one 4 MiB segment. Preallocating there
# would fill the tmpfs and then fail; the arena must refuse the growth up
# front, and grow as soon as room is back.
_LESS_THAN_A_SEGMENT_FREE_SCRIPT = """
import os
from mstar_rust import SegmentedShmArena

MiB = 1 << 20
filler = f"/dev/shm/filler_{os.getpid()}"
with open(filler, "wb") as f:
    f.write(b"\\1" * (9 * MiB))
a = SegmentedShmArena.create(f"mstar_test_lowroom_{os.getpid()}", 4 * MiB, 4)
assert a.reserve(4 * MiB) == (0, 0)
st = os.statvfs("/dev/shm")
free = st.f_bavail * st.f_frsize
assert 0 < free < 4 * MiB, free
for _ in range(2):
    try:
        a.reserve(256)
    except RuntimeError as e:
        assert str(e).endswith(f"only {free} B free"), e
    else:
        raise AssertionError("grew a segment /dev/shm cannot hold")
os.remove(filler)
assert a.reserve(256) == (1, 0) and a.num_segments == 2
print("ok")
"""


def test_less_than_a_segment_free_fails_growth_up_front(tmp_path):
    proc = _run_in_small_dev_shm(tmp_path, _LESS_THAN_A_SEGMENT_FREE_SCRIPT)
    assert proc.stdout.strip() == "ok", proc.stdout


# RLIMIT_FSIZE makes posix_fallocate itself fail (EFBIG) while the
# free-space check passes: that growth is not repeated within the window.
_FALLOCATE_FAILS_SCRIPT = """
import os, resource, signal, time
from mstar_rust import SegmentedShmArena

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, resource.RLIM_INFINITY))
a = SegmentedShmArena.create(f"mstar_test_fsize_{os.getpid()}", 4096, 4)
failed_at, errors = time.monotonic(), []
for _ in range(2):
    try:
        a.reserve(2 << 20)
    except RuntimeError as e:
        errors.append(str(e))
first, again = errors
assert "arena full: no space" in first and "not retried" not in first, first
assert "not retried" in again or time.monotonic() - failed_at >= 1, again
assert a.num_segments == 1
del a
print("ok")
"""


def test_a_failed_fallocate_is_not_repeated_within_the_window():
    import subprocess

    if not sys.platform.startswith("linux"):
        pytest.skip("Linux only")
    proc = subprocess.run([sys.executable, "-c", _FALLOCATE_FAILS_SCRIPT],
                          check=False, capture_output=True, text=True,
                          timeout=60)
    assert proc.stdout.strip() == "ok", proc.stdout + proc.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_failed_pin_leaves_cuda_usable():
    from mstar.communication import arena

    assert not arena._pin(4096, 4096)  # not a mapping of this process
    assert (torch.ones(4, device="cuda") + 1).sum().item() == 8


def test_segments_are_pinned_through_the_platform(monkeypatch):
    """hipHostRegister on ROCm, cudaHostRegister on CUDA: the arena only asks."""
    from mstar.communication import arena as arena_mod

    calls = []

    class _Platform:
        def host_register(self, ptr, nbytes):
            calls.append(("register", ptr, nbytes))
            return True

        def host_unregister(self, ptr):
            calls.append(("unregister", ptr))
            return True

    monkeypatch.setattr(arena_mod, "current_platform", _Platform())
    monkeypatch.setattr(arena_mod.torch.cuda, "is_available", lambda: True)
    assert arena_mod._pin(0x1000, 4096)
    assert arena_mod._unpin(0x1000)
    assert calls == [("register", 0x1000, 4096), ("unregister", 0x1000)]
