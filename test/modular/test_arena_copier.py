"""The arena's Rust hot paths (``rust/src/arena_copy.rs``), driven through the
manager: what they must keep from the Python paths they replace, and that a row
they cannot do comes back to Python."""
import ctypes
import os
import threading
import time

import pytest
import torch

from mstar.communication import arena as arena_mod
from mstar.communication.arena import ArenaShmCommunicationManager
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge
from mstar.utils.containers import ParallelList

rs = pytest.importorskip("mstar_rust")

if not hasattr(rs, "ArenaTransport") or not rs.torch_abi_embedded():
    pytest.skip("this mstar_rust has no arena copier", allow_module_level=True)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class _Comm:
    def send(self, entity_id, msg):
        raise AssertionError("unexpected control-mesh send")

    def get_all_new_messages(self):
        return []


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "2")
    monkeypatch.setenv("MSTAR_SHM_ARENA_MAX_SEGMENTS", "4")
    made = []

    def _make(entity, device="cpu", **kwargs):
        m = ArenaShmCommunicationManager(
            my_entity_id=entity, hostname="localhost", device=device,
            communicator=_Comm(), shm_dir=str(tmp_path), **kwargs,
        )
        made.append(m)
        return m

    yield _make
    for m in made:
        m.close()


def _stage(prod, rid, tensors):
    infos = prod.store_and_return_tensor_info(rid, {"x": tensors})["x"]
    prod.register_for_send(rid, infos)
    return infos


def _edge(infos):
    return GraphEdge(next_node="B", name="x", tensor_info=infos)


def _pin_callbacks(pin, unpin):
    """Addresses of (register, unregister, get_last_error) that call ``pin(ptr,
    nbytes)`` and ``unpin(ptr)``, and the ctypes objects to keep alive."""
    callbacks = (
        ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)(
            lambda ptr, nbytes, _flags: 0 if pin(ptr, nbytes) else 1),
        ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(lambda ptr: 0 if unpin(ptr) else 1),
        ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0),
    )
    return tuple(ctypes.cast(c, ctypes.c_void_p).value for c in callbacks), callbacks


def _free_bytes(m):
    return m.stats_summary()["free_bytes"]


def _no_torch_copy(monkeypatch):
    """The Python paths' tell: every one of them goes through a frombuffer."""
    def refuse(*a, **k):
        raise AssertionError("took the Python path")

    monkeypatch.setattr(arena_mod.torch, "frombuffer", refuse)


SHAPES = [
    pytest.param(torch.arange(16, dtype=torch.float32), id="float32"),
    pytest.param(torch.arange(24, dtype=torch.int64).reshape(2, 3, 4), id="3d"),
    pytest.param(torch.tensor(1.5), id="0-dim"),
    pytest.param(torch.empty(0, 3), id="zero-bytes"),
    pytest.param(torch.empty(4, 0), id="zero-byte-rows"),
    pytest.param(torch.arange(9) % 2 == 0, id="bool"),
    pytest.param(torch.arange(32.0)[8:16], id="a-view-with-an-offset"),
]


@pytest.mark.parametrize("original", SHAPES)
def test_a_roundtrip_touches_no_python_copy(make, monkeypatch, original):
    prod, cons = make("w0"), make("w1")
    _no_torch_copy(monkeypatch)
    infos = _stage(prod, "r", [original])
    assert infos[0].shm_segment is not None, "the caller's descriptor is stamped in place"
    assert prod.tensor_store.get_info(infos[0].uuid).shm_segment == infos[0].shm_segment
    assert prod._transport.has_slot(infos[0].uuid)
    assert prod.tensor_store.is_registered(infos[0].uuid)
    cons.start_read_tensors("r", [_edge(infos)])
    got = cons.tensor_store.get_tensor(infos[0].uuid)
    assert got.dtype == original.dtype and got.shape == original.shape
    assert torch.equal(got, original)
    assert cons.tensor_store.bookkeeping.dereference_batch_uniform([infos[0].uuid], 1)[0] == []
    assert cons.tensor_store.bookkeeping.dereference_batch_uniform([infos[0].uuid], 1)[0] == [infos[0].uuid]


def test_register_for_send_uuids_publishes_to_the_store_alone(make, monkeypatch):
    prod, cons = make("w0"), make("w1")
    _no_torch_copy(monkeypatch)
    tensors = [torch.full((8,), float(i)) for i in range(5)]
    infos = prod.store_and_return_tensor_info("r", {"x": tensors})["x"]
    uuids = [i.uuid for i in infos]
    prod.register_for_send_uuids(ParallelList(["r", "s"], [uuids[:2], uuids[2:]]))
    assert all(i.shm_segment is None for i in infos), "nothing asked for the stamp in place"
    read = [prod.tensor_store.get_info(u) for u in uuids]
    assert all(i.shm_segment == prod._arena.segment_name(0) for i in read)
    cons.start_read_tensors("r", [_edge(read)])
    for u, t in zip(uuids, tensors, strict=True):
        assert torch.equal(cons.tensor_store.get_tensor(u), t)


@pytest.mark.parametrize("source", [
    pytest.param(lambda: torch.arange(24.0).reshape(4, 6).t(), id="non-contiguous"),
    pytest.param(lambda: torch.arange(8.0)[::2], id="strided"),
])
def test_a_source_rust_cannot_read_goes_to_the_python_path(make, source):
    prod, cons = make("w0"), make("w1")
    t = source()
    infos = _stage(prod, "r", [t, torch.ones(3)])
    assert all(i.shm_segment is not None for i in infos)
    cons.start_read_tensors("r", [_edge(infos)])
    assert torch.equal(cons.tensor_store.get_tensor(infos[0].uuid), t)
    assert torch.equal(cons.tensor_store.get_tensor(infos[1].uuid), torch.ones(3))


def test_a_tensor_whose_bytes_are_not_its_values_never_reaches_a_raw_copy(make):
    """A conj view's memory holds the unconjugated values, and the store can hold one."""
    prod = make("w0")
    t = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64).conj()
    with pytest.raises(RuntimeError):
        # The Python path's refusal, unchanged: it never copied raw bytes either.
        _stage(prod, "r", [t])
    assert prod._transport.live()[0] == 0


def test_growth_is_pinned_and_named_before_rust_uses_the_segment(make, monkeypatch):
    prod, cons = make("w0"), make("w1")
    pins = []
    fns, keep = _pin_callbacks(lambda p, n: pins.append(p) or True, lambda p: True)
    prod._transport.set_pins(fns, 1 << 30)
    pins.clear()  # the first segment
    tensors = [torch.full((700_000,), i, dtype=torch.uint8) for i in range(3)]
    infos = _stage(prod, "r", tensors)
    assert prod._arena.num_segments == 2 and len(pins) == 1, "the arena grew once, pinned"
    names = [prod._arena.segment_name(i) for i in range(2)]
    assert {i.shm_segment for i in infos} == set(names)
    more = _stage(prod, "s", [torch.full((700_000,), 9, dtype=torch.uint8)])
    assert more[0].shm_segment == names[1], "Rust reserves in the grown segment now"
    cons.start_read_tensors("r", [_edge(infos), _edge(more)])
    for i, t in zip([*infos, *more], [*tensors, torch.full((700_000,), 9, dtype=torch.uint8)], strict=True):
        assert torch.equal(cons.tensor_store.get_tensor(i.uuid), t)
    del keep


def test_a_segment_grown_behind_the_transport_is_adopted_before_use(make):
    """A reserve keeps to adopted segments; the next growth path adopts this one,
    names it to the store and lands there."""
    prod = make("w0")
    prod._arena.reserve(prod._segment_bytes)  # the first segment is full
    prod._arena.reserve(16)  # grows, with nothing told about it
    assert prod._arena.num_segments == 2
    infos = prod.store_and_return_tensor_info("r", {"x": [torch.ones(4)]})["x"]
    prod.register_for_send("r", infos)
    assert infos[0].shm_segment == prod._arena.segment_name(1)


def test_arena_saturation_spills_the_rest(make):
    prod, cons = make("w0"), make("w1")
    tensors = [torch.full((1_500_000,), i, dtype=torch.uint8) for i in range(6)]
    infos = _stage(prod, "r", tensors)
    spilled = [i for i in infos if i.shm_segment is None]
    assert spilled and all(prod._transport.spilled_path("w0", i.uuid) for i in spilled)
    cons.start_read_tensors("r", [_edge(infos)])
    for i, t in zip(infos, tensors, strict=True):
        assert torch.equal(cons.tensor_store.get_tensor(i.uuid), t)


def test_a_uuid_met_twice_in_one_batch_gets_one_slot(make):
    prod = make("w0")
    infos = prod.store_and_return_tensor_info("r", {"x": [torch.ones(8)]})["x"]
    before = _free_bytes(prod)
    uuid = infos[0].uuid
    prod.register_for_send_uuids(ParallelList(["r"], [[uuid, uuid]]))
    assert before - _free_bytes(prod) == 256
    prod.register_for_send_uuids(ParallelList(["r"], [[uuid]]))
    assert before - _free_bytes(prod) == 256, "an already registered uuid takes no slot"


def test_concurrent_duplicates_leave_no_orphaned_slot(make):
    """However two threads staging one uuid interleave, one slot ends up held, and
    cleanup returns it."""
    prod = make("w0")
    before = _free_bytes(prod)
    for _round in range(50):
        infos = prod.store_and_return_tensor_info("r", {"x": [torch.ones(50_000)]})["x"]
        uuid = infos[0].uuid
        barrier = threading.Barrier(4)

        def stage(barrier=barrier, uuid=uuid):
            barrier.wait()
            prod.register_for_send_uuids(ParallelList(["r"], [[uuid]]))

        threads = [threading.Thread(target=stage) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert prod.tensor_store.is_registered(uuid)
        assert before - _free_bytes(prod) == 200_192, "one 256-aligned slot of 200 KB"
        prod._cleanup_by_uuid(uuid)
        assert _free_bytes(prod) == before, "a slot was orphaned"
    assert prod._transport.live()[0] == 0


def _mixed_reads(prod, cons):
    """A batch with every kind of row a read meets: a plain one, one met twice in
    the batch, one the store already holds, two slices of one tensor, and one of
    this entity's own. Returns the tensors the descriptors should read."""
    a, b, c, d = torch.arange(4.0), torch.arange(4.0) + 10, torch.arange(4.0) + 20, torch.arange(8.0).reshape(4, 2)
    ia, ib, ic, id_ = _stage(prod, "r", [a, b, c, d])
    cons.start_read_tensors("r", [_edge([ic.clone()])])  # already in the store below
    first, second = id_.clone(), id_.clone()
    first.dims, first.nbytes = [1, 2], 8
    second.dims, second.nbytes, second.offset = [3, 2], 24, 8
    (own,) = cons.store_and_return_tensor_info("r", {"y": [torch.ones(2)]})["y"]
    cons.increment_ref(own.uuid, 1)
    cons.register_request("r", ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={}))
    rows = [ia.clone(), ib.clone(), ib.clone(), ic.clone(), first, second, own.clone()]
    want = [a, b, b, c, d[:1], d[1:], torch.ones(2)]
    return rows, want


def test_a_batch_reads_the_same_through_rust_as_through_the_python_walk(make):
    """Transfer metrics keep the Python walk over the descriptors, so one manager
    of each compares them: same tensors, same references, same pending edges."""
    prod = make("w0")
    results = []
    for name, metrics in (("c0", False), ("c1", True)):
        cons = make(name)
        if metrics:
            cons.enable_transfer_metrics()
        rows, want = _mixed_reads(prod, cons)
        edges = [_edge(rows[:2]), _edge(rows[2:4]), _edge(rows[4:6]), _edge(rows[6:])]
        cons.start_read_tensors("r", edges)
        for info, tensor in zip(rows, want, strict=True):
            assert torch.equal(cons.tensor_store.get_tensor(info.uuid), tensor)
        rust = cons.tensor_store._rust
        relabel = {u: i for i, u in enumerate(dict.fromkeys(i.uuid for i in rows))}
        results.append((
            [relabel[i.uuid] for i in rows],
            [rust.ref_count(i.uuid) for i in rows],
            [len(fp.graph_edges[0].tensor_info) for fp in cons.pending],
        ))
    assert results[0] == results[1]


def test_a_large_host_copy_releases_the_gil(make, monkeypatch):
    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "64")
    prod = make("big")
    tensor = torch.ones(48 << 20, dtype=torch.uint8)
    infos = prod.store_and_return_tensor_info("r", {"x": [tensor]})["x"]
    ticks, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        store = prod.tensor_store
        before = ticks[0]
        prod._transport.stage(store._rust, [infos[0].uuid], False)
        assert ticks[0] != before, "no other thread ran during the copy"
    finally:
        stop.set()
        thread.join()


def test_a_descriptor_past_its_segment_is_refused(make):
    prod, cons = make("w0"), make("w1")
    infos = _stage(prod, "r", [torch.ones(4)])
    infos[0].shm_offset = prod._segment_bytes - 8
    with pytest.raises(ValueError, match="run past"):
        cons.start_read_tensors("r", [_edge(infos)])
    assert not cons.tensor_store.check_uuid_presence(infos[0].uuid)


def test_a_descriptor_whose_dims_disagree_with_its_bytes_is_refused(make):
    prod, cons = make("w0"), make("w1")
    infos = _stage(prod, "r", [torch.ones(4)])
    infos[0].dims = (5,)
    with pytest.raises(RuntimeError, match="dims and dtype"):
        cons.start_read_tensors("r", [_edge(infos)])
    assert not cons.tensor_store.check_uuid_presence(infos[0].uuid)


def test_a_dead_peer_is_forgotten_by_rust_too(make, tmp_path):
    prod, cons = make("w0"), make("w1")
    infos = _stage(prod, "r", [torch.ones(4)])
    cons.start_read_tensors("r", [_edge(infos)])
    seg = infos[0].shm_segment
    assert cons._transport.has_peer(seg)
    os.unlink(f"/dev/shm/{seg}")
    cons._peer_evict_last = 0.0
    cons.pending.clear()
    cons._evict_dead_peers()
    assert not cons._transport.has_peer(seg)
    info = infos[0].clone()
    info.uuid += 1
    with pytest.raises(RuntimeError, match=seg):
        cons._transport.read(cons.tensor_store._rust, ["r"], [info], 2, 0, {})


def test_the_ttl_backstop_reclaims_rust_staged_slots(make, monkeypatch):
    monkeypatch.setenv("MSTAR_SHM_ARENA_SLOT_TTL_S", "0.001")
    prod = make("w0")
    before = _free_bytes(prod)
    infos = _stage(prod, "r", [torch.ones(4)])
    assert prod._transport.has_slot(infos[0].uuid)
    time.sleep(0.01)
    assert prod._transport.reclaim_expired() == 1
    assert not prod._transport.has_slot(infos[0].uuid) and _free_bytes(prod) == before


def test_profiling_records_what_rust_staged(make):
    prod = make("w0", enable_prof=True)
    t = torch.ones(16)
    infos = prod.store_and_return_tensor_info("r", {"x": [t]})["x"]
    prod.register_for_send_uuids(ParallelList(["r"], [[infos[0].uuid]]))
    tx = prod.get_tx_info("r")
    assert len(tx) == 1 and tx[0].num_bytes == t.nbytes


@pytest.mark.parametrize("backend", ["no-wrapper", "python-bookkeeper"])
def test_the_python_paths_still_work_alone(make, monkeypatch, backend):
    """A platform with no wrapper, and a Python bookkeeper, both get no Rust path."""
    if backend == "no-wrapper":
        monkeypatch.setattr(arena_mod, "require_torch_abi", lambda device: False)
    prod, cons = make("w0"), make("w1")
    if backend == "no-wrapper":
        assert prod._read_device_type is None
    else:
        prod.tensor_store = TensorStore(PythonTensorBookkeeping())
        cons.tensor_store = TensorStore(PythonTensorBookkeeping())
    t = torch.arange(16.0)
    infos = _stage(prod, "r", [t])
    assert prod._transport.has_slot(infos[0].uuid)
    cons.start_read_tensors("r", [_edge(infos)])
    assert torch.equal(cons.tensor_store.get_tensor(infos[0].uuid), t)
    prod._cleanup_by_uuid(infos[0].uuid)
    assert prod._transport.live()[0] == 0


# ---- on a device -------------------------------------------------------------


def _device():
    return f"cuda:{torch.cuda.current_device()}"


@needs_cuda
@pytest.mark.parametrize("nbytes", [64, 65536, 1 << 21])
def test_a_device_source_is_in_the_segment_when_registration_returns(make, monkeypatch, nbytes):
    """The trailing sync: no descriptor can ship ahead of its bytes."""
    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "8")
    prod = make("w0", device=_device())
    _no_torch_copy(monkeypatch)
    t = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device=_device())
    torch.cuda.default_stream().synchronize()
    infos = prod.store_and_return_tensor_info("r", {"x": [t]})["x"]
    prod.register_for_send_uuids(ParallelList(["r"], [[infos[0].uuid]]), skip_cuda_sync=True)
    seg, off = prod._transport.slot(infos[0].uuid)
    in_segment = bytes(prod._segment_view(seg)[off:off + nbytes])
    assert in_segment == bytes(t.cpu().numpy().tobytes())


@needs_cuda
def test_a_device_read_lands_behind_the_event_with_two_references(make, monkeypatch):
    """The ACK gate: an edge reports ready only once its copies have landed."""
    prod, cons = make("w0"), make("w1", device=_device())
    t = torch.arange(1 << 18, dtype=torch.int32)
    infos = _stage(prod, "r", [t])
    _no_torch_copy(monkeypatch)
    futures = cons.start_read_tensors("r", [_edge(infos)])
    assert len(futures) == 1
    assert isinstance(cons.pending[0].future, arena_mod._CudaEventFuture)
    futures[0].result(timeout=30)
    assert cons.pending[0].future.done()
    got = cons.tensor_store.get_tensor(infos[0].uuid)
    assert got.device == torch.device(_device()) and torch.equal(got.cpu(), t)
    bk = cons.tensor_store.bookkeeping
    assert bk.dereference_batch_uniform([infos[0].uuid], 1)[0] == []
    assert bk.dereference_batch_uniform([infos[0].uuid], 1)[0] == [infos[0].uuid]


@needs_cuda
def test_a_zero_byte_read_on_a_device_needs_no_event(make, monkeypatch):
    prod, cons = make("w0"), make("w1", device=_device())
    infos = _stage(prod, "r", [torch.empty(0, 3)])
    _no_torch_copy(monkeypatch)
    assert cons.start_read_tensors("r", [_edge(infos)]) == []
    assert cons.pending[0].future is None
    got = cons.tensor_store.get_tensor(infos[0].uuid)
    assert got.device == torch.device(_device()) and got.shape == (0, 3)


@needs_cuda
def test_a_device_read_is_allocated_under_the_copy_stream(make):
    """So a block is tagged as torch's own copy would tag it, and the allocator
    orders its reuse against the copy."""
    prod, cons = make("w0"), make("w1", device=_device())
    # A size no earlier allocation left a block for, so this makes its own segment.
    t = torch.ones(2_718_281, dtype=torch.uint8)
    infos = _stage(prod, "r", [t])
    with torch.cuda.stream(cons._h2d_stream):
        cons._read_staged([("r", infos[0])], [])
    got = cons.tensor_store.get_tensor(infos[0].uuid)
    ptr = got.data_ptr()
    streams = [
        seg["stream"] for seg in torch.cuda.memory._snapshot()["segments"]
        if seg["address"] <= ptr < seg["address"] + seg["total_size"]
    ]
    assert streams == [cons._h2d_stream.cuda_stream]


@needs_cuda
def test_a_gathered_buffer_is_held_until_the_event(make, monkeypatch):
    prod, cons = make("w0"), make("w1", device=_device())
    tensors = [torch.full((8,), float(i)) for i in range(6)]
    infos = _stage(prod, "r", tensors)
    _no_torch_copy(monkeypatch)
    futures = cons.start_read_tensors_multi(
        [("r", [_edge(infos[:3])]), ("r", [_edge(infos[3:])])])
    assert cons.pending[0].future is cons.pending[1].future
    held = [k for k in cons.pending[0].future._keep if torch.is_tensor(k)]
    assert len(held) == 1 and held[0].is_pinned() and held[0].numel() == 48 * 4
    futures[0].result(timeout=30)
    got = [cons.tensor_store.get_tensor(i.uuid) for i in infos]
    assert all(torch.equal(g.cpu(), t) for g, t in zip(got, tensors, strict=True))
    assert len({g.untyped_storage().data_ptr() for g in got}) == 1


def _failing_memcpy(prod, on_copy=None):
    """A copier whose device copies fail, or run ``on_copy`` first."""
    proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_int, ctypes.c_void_p)
    real = proto(arena_mod._cudart_copy_fns()[0])

    def copy(dst, src, n, kind, stream):
        if on_copy is None:
            return 1  # cudaErrorInvalidValue
        on_copy()
        return real(dst, src, n, kind, stream)

    callback = proto(copy)
    fns = list(arena_mod._cudart_copy_fns())
    fns[0] = ctypes.cast(callback, ctypes.c_void_p).value
    prod._transport = rs.ArenaTransport(
        prod._arena, prod.shm_dir, "mstar_", prod.my_entity_id, True, 0.0, 30.0, 0.0,
        device_index=torch.cuda.current_device(), gather_item_bytes=16384,
        cuda=tuple(fns), d2h_stream=prod._d2h_stream.cuda_stream,
        h2d_stream=prod._h2d_stream.cuda_stream,
    )
    return callback


@needs_cuda
@pytest.mark.parametrize("nbytes", [64, 1 << 20])
def test_a_device_tensor_spills_and_reads_back_onto_a_device(make, monkeypatch, nbytes):
    """No arena has room: the D2H lands in its own file before the send returns,
    and the read queues its H2D behind the event, from a buffer held until then."""
    prod = make("w0", device=_device(), use_arena=False)
    cons = make("w1", device=_device())
    t = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device=_device())
    torch.cuda.default_stream().synchronize()
    infos = prod.store_and_return_tensor_info("r", {"x": [t]}, skip_cuda_sync=True)["x"]
    prod.register_for_send("r", infos, skip_cuda_sync=True)
    assert infos[0].shm_segment is None
    path = prod._transport.spilled_path("w0", infos[0].uuid)
    with open(path, "rb") as f:
        assert f.read() == bytes(t.cpu().numpy().tobytes())
    futures = cons.start_read_tensors("r", [_edge(infos)])
    futures[0].result(timeout=30)
    assert torch.equal(cons.tensor_store.get_tensor(infos[0].uuid), t)
    prod._cleanup_by_uuid(infos[0].uuid)
    assert not os.path.exists(path)


@needs_cuda
def test_a_failed_device_copy_frees_its_slot_and_stays_quiet_for_torch(make):
    prod = make("w0", device=_device())
    keep = _failing_memcpy(prod)
    before = _free_bytes(prod)
    t = torch.ones(16, device=_device())
    infos = prod.store_and_return_tensor_info("r", {"x": [t, t.clone()]}, skip_cuda_sync=True)["x"]
    with pytest.raises(RuntimeError, match="cudaMemcpyAsync"):
        prod.register_for_send_uuids(ParallelList(["r"], [[i.uuid for i in infos]]))
    assert _free_bytes(prod) == before and prod._transport.live()[0] == 0
    assert not any(prod.tensor_store.is_registered(i.uuid) for i in infos)
    # A kernel launch would raise an error left set on the thread.
    assert torch.equal((t + 1).cpu(), torch.full((16,), 2.0))
    del keep


@needs_cuda
def test_a_teardown_during_the_copy_returns_the_slot(make):
    """A cleanup that meets a uuid mid-staging finds only the claim, so the stager
    has to free what it reserved."""
    prod = make("w0", device=_device())
    infos = []
    keep = _failing_memcpy(prod, lambda: prod._cleanup_by_uuid(infos[0].uuid))
    before = _free_bytes(prod)
    t = torch.ones(16, device=_device())
    infos.extend(prod.store_and_return_tensor_info("r", {"x": [t]}, skip_cuda_sync=True)["x"])
    prod.register_for_send_uuids(ParallelList(["r"], [[infos[0].uuid]]))
    assert _free_bytes(prod) == before and prod._transport.live()[0] == 0
    del keep


@needs_cuda
def test_a_store_that_frees_in_place_during_the_copy_returns_the_slot(make):
    """A release the store settles itself never reaches the slot table, so the
    stager finds its claim still there and has to notice the tensor is gone."""
    prod = make("w0", device=_device())
    keep = _failing_memcpy(prod, lambda: prod.tensor_store.release_request("r", False))
    before = _free_bytes(prod)
    t = torch.ones(16, device=_device())
    infos = prod.store_and_return_tensor_info("r", {"x": [t]}, skip_cuda_sync=True)["x"]
    prod.register_for_send_uuids(ParallelList(["r"], [[infos[0].uuid]]))
    assert _free_bytes(prod) == before and prod._transport.live()[0] == 0
    del keep


@needs_cuda
def test_a_row_that_fails_after_copies_are_queued_waits_for_them(make, monkeypatch):
    """Rust queues what it can before the rest are settled, so an error in those
    must not drop the host buffers under the copies."""
    prod, cons = make("w0"), make("w1", device=_device())
    infos = _stage(prod, "r", [torch.ones(4), torch.zeros(4)])
    (own,) = cons.store_and_return_tensor_info("r", {"y": [torch.ones(2)]})["y"]
    monkeypatch.setattr(cons, "_slice_existing_tensor", lambda **_: 1 / 0)
    synced, real = [], torch.cuda.Stream.synchronize

    def synchronize(stream):
        synced.append(stream)
        real(stream)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", synchronize)
    with pytest.raises(ZeroDivisionError):
        cons.start_read_tensors("r", [_edge([*infos, own])])
    assert synced == [cons._h2d_stream]


@needs_cuda
def test_a_device_read_holds_its_mapping_until_the_event(make):
    prod, cons = make("w0"), make("w1", device=_device())
    infos = _stage(prod, "r", [torch.ones(4)])
    futures = cons.start_read_tensors("r", [_edge(infos)])
    assert any(isinstance(k, rs.ShmArena) for k in cons.pending[0].future._keep)
    futures[0].result(timeout=30)


@needs_cuda
def test_a_device_on_another_platform_keeps_its_torch_paths(make, monkeypatch):
    """ROCm and XPU run this manager with device-agnostic torch calls."""
    class NotNvidia:
        name = "other"

        @staticmethod
        def is_cuda():
            return False

    monkeypatch.setattr(arena_mod, "current_platform", NotNvidia)
    monkeypatch.setattr(arena_mod, "_cudart_copy_fns",
                        lambda: pytest.fail("cudart on a platform that is not CUDA"))
    prod, cons = make("w0", device=_device()), make("w1", device=_device())
    assert prod._read_device_type is None and cons._read_device_type is None
    t = torch.arange(32.0, device=_device())
    infos = prod.store_and_return_tensor_info("r", {"x": [t]})["x"]
    prod.register_for_send("r", infos)
    futures = cons.start_read_tensors("r", [_edge(infos)])
    futures[0].result(timeout=30)
    assert torch.equal(cons.tensor_store.get_tensor(infos[0].uuid), t)
