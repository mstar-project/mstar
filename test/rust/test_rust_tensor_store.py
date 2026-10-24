"""``RustTensorStore`` keeps the tensors in the Rust bookkeeper and frees, inside
a release, the ones that need nothing from the transport.

Two things have to hold. What is stored, what it resolves to, its refcounts and
when it goes away must match the Python store the Python graph runtime uses
(``MSTAR_RUST_GRAPH=0``) -- a divergence is a tensor freed under a reader, or
one held until its request ends. And a tensor the transport still has state
for (registered memory, an arena slot, a per-uuid file, a shard dim to undo)
must still reach the transport's teardown, since only a reported uuid does.
"""
import random
import sys
import weakref
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

pytest.importorskip("mstar_rust", reason="mstar_rust not built")

import torch

from mstar.communication.tensor_store import (
    PythonTensorBookkeeping,
    RustTensorBookkeeping,
    RustTensorStore,
    TensorStore,
)
from mstar.communication.tensors import TensorCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import TensorPointerInfo
from mstar.utils.containers import ParallelList


class _Manager(TensorCommunicationManager):
    """The real manager's store, dereference and teardown paths, with a
    transport whose only state is ``transport``: a uuid is in it from its
    registration until its teardown runs, which is what an arena slot or a
    per-uuid file is."""

    def __init__(self, store: TensorStore):
        super().__init__(
            my_entity_id="w0", my_session_id="s0", device="cpu",
            communicator=None, transfer_engine=None,
        )
        self.tensor_store = store
        self.transport: set[int] = set()
        self.torn_down: list[int] = []

    def register(self, uuids):
        for uuid in uuids:
            self.tensor_store.set_metadata(uuid, mem_registered=True)
            self.transport.add(uuid)

    def _cleanup_by_uuid(self, uuid, registered=None):
        super()._cleanup_by_uuid(uuid, registered)
        if not self.tensor_store.check_uuid_presence(uuid):
            return
        if registered is None:
            registered = self.tensor_store.is_registered(uuid)
        if registered:
            self.transport.remove(uuid)
            self.torn_down.append(uuid)
        self.tensor_store.remove_tensor(uuid)

    def register_for_send(self, request_id, tensor_infos, skip_cuda_sync=False):
        raise NotImplementedError

    def register_for_send_uuids(self, per_request, skip_cuda_sync=False):
        raise NotImplementedError

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        raise NotImplementedError


def _pair():
    py = _Manager(TensorStore(PythonTensorBookkeeping()))
    rs = _Manager(TensorStore(RustTensorBookkeeping()))
    assert type(rs.tensor_store) is RustTensorStore
    assert type(py.tensor_store) is TensorStore
    return py, rs


def _info(uuid: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=(2,), dtype=torch.float32, nbytes=8, address=0, stride=(1,),
        uuid=uuid, source_session_id="s0", source_entity="w0",
    )


def _ref_count(store, uuid):
    bk = store.bookkeeping
    if isinstance(bk, RustTensorBookkeeping):
        return bk._rust.ref_count(uuid)
    info = bk._ref_info.get(uuid)
    return None if info is None else info.ref_cnt


def _state(mgr, uuids, rids):
    s = mgr.tensor_store
    return {
        "refs": {u: _ref_count(s, u) for u in uuids},
        "present": {u for u in uuids if s.check_uuid_presence(u)},
        "objects": {u: id(s.get_tensor(u)) for u in uuids if s.check_uuid_presence(u)},
        "gc": {u: s.can_gc(u) for u in uuids},
        "info": {u: s.get_info(u) is not None for u in uuids},
        "owned": {r: sorted(s.get_all_uuids(r)) for r in rids},
        "transport": set(mgr.transport),
        "torn_down": sorted(mgr.torn_down),
    }


@pytest.mark.parametrize("seed", range(20))
def test_random_sequences_match_the_python_store(seed):
    """Store, reference, dereference (single, batched, by the runtime's
    release), persist, register and tear down at random, and compare
    everything observable after every step."""
    rng = random.Random(seed)
    py, rs = _pair()
    rids = [1, 2, 3, "api-x"]
    for mgr in (py, rs):
        cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
        for rid in rids:
            mgr.register_request(rid, cfg)
    minted: list[int] = []

    for step in range(120):
        op = rng.choice([
            "store", "store", "ref", "deref_batch", "deref_batch", "deref_one",
            "release", "persist", "register", "teardown",
        ])
        live = [u for u in minted if py.tensor_store.check_uuid_presence(u)]
        if op == "store":
            rid = rng.choice(rids)
            n = rng.randint(1, 4)
            tensors = [torch.full((2,), float(len(minted) + i)) for i in range(n)]
            outs = {}
            for mgr in (py, rs):
                outs[mgr is rs] = mgr.store_and_return_tensor_info_batch(
                    [rid], {rid: {"x": list(tensors)}}, ["x"],
                    node_name="N", graph_walk="w", skip_cuda_sync=True,
                )
            # Both minted from their own counters, from the same start.
            assert outs[False].flat_uuids == outs[True].flat_uuids
            minted.extend(outs[True].flat_uuids)
            if rng.random() < 0.7:
                for mgr in (py, rs):
                    mgr.increment_ref_batch_uniform(outs[True].flat_uuids, 1)
        elif not live:
            continue
        elif op == "ref":
            picks = rng.sample(live, min(len(live), 3))
            n = rng.randint(1, 2)
            for mgr in (py, rs):
                mgr.increment_ref_batch_uniform(picks, n)
        elif op == "deref_batch":
            picks = rng.sample(live, min(len(live), rng.randint(1, 4)))
            counts = [1] * len(picks)
            for mgr in (py, rs):
                mgr.dereference_batch(picks, counts)
        elif op == "deref_one":
            u = rng.choice(live)
            for mgr in (py, rs):
                mgr.dereference(u, 1)
        elif op == "release":
            # What the Rust runtime does with a node's consumed inputs: the
            # bookkeeper releases, the manager tears down what is reported.
            picks = rng.sample(live, min(len(live), rng.randint(1, 4)))
            for mgr in (py, rs):
                mgr.cleanup_collectable(
                    *mgr.tensor_store.bookkeeping.dereference_batch_uniform(
                        picks, 1, cleanup=True,
                    )
                )
        elif op == "persist":
            u = rng.choice(live)
            flag = rng.random() < 0.5
            for mgr in (py, rs):
                mgr.set_persist(u, flag)
        elif op == "register":
            picks = rng.sample(live, min(len(live), 2))
            for mgr in (py, rs):
                mgr.register(picks)
        elif op == "teardown":
            rid = rng.choice(rids)
            force = rng.random() < 0.5
            for mgr in (py, rs):
                mgr.cleanup_request(rid, force=force)
                mgr.register_request(
                    rid, ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
                )
        assert _state(py, minted, rids) == _state(rs, minted, rids), (seed, step, op)

    for rid in rids:
        for mgr in (py, rs):
            mgr.force_cleanup_request(rid)
    assert _state(py, minted, rids) == _state(rs, minted, rids)
    assert rs.tensor_store.stored_counts() == (0, 0)
    assert not rs.transport and not py.transport


def test_a_collected_tensor_is_gone_from_rust():
    """Freed in the release itself: the store held the last reference."""
    _, mgr = _pair()
    t = torch.ones(3)
    ref = weakref.ref(t)
    stored = mgr.store_and_return_tensor_info_batch(
        [1], {1: {"x": [t]}}, ["x"], skip_cuda_sync=True,
    )
    del t
    (uuid,) = stored.flat_uuids
    mgr.increment_ref_batch_uniform([uuid], 1)
    assert ref() is not None

    reported = mgr.tensor_store.bookkeeping.dereference_batch_uniform(
        [uuid], 1, cleanup=True,
    )
    assert reported == ([], []), "nothing for the transport, nothing reported"
    assert ref() is None
    assert not mgr.tensor_store.check_uuid_presence(uuid)
    assert mgr.tensor_store.get_info(uuid) is None
    assert mgr.tensor_store.stored_counts() == (0, 0)
    with pytest.raises(KeyError):
        mgr.tensor_store.get_tensor(uuid)


def test_a_registered_tensor_still_reaches_the_transport():
    """Reported, and held until the teardown: unregistering memory reads it."""
    _, mgr = _pair()
    t = torch.ones(3)
    ref = weakref.ref(t)
    stored = mgr.store_and_return_tensor_info_batch(
        [1], {1: {"x": [t]}}, ["x"], skip_cuda_sync=True,
    )
    del t
    (uuid,) = stored.flat_uuids
    mgr.increment_ref_batch_uniform([uuid], 1)
    mgr.register([uuid])

    uuids, registered = mgr.tensor_store.bookkeeping.dereference_batch_uniform(
        [uuid], 1, cleanup=True,
    )
    assert (uuids, registered) == ([uuid], [True])
    assert ref() is not None and mgr.tensor_store.check_uuid_presence(uuid)
    mgr.cleanup_collectable(uuids, registered)
    assert mgr.torn_down == [uuid] and not mgr.transport
    assert ref() is None


def test_an_arena_slot_is_freed_on_release(tmp_path, monkeypatch):
    """The real arena manager over the Rust store: a staged tensor's slot is
    returned by the release the runtime does, an unstaged one is freed with
    no trip through the arena's teardown at all."""
    from mstar.communication.arena import ArenaShmCommunicationManager

    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "1")
    monkeypatch.setenv("MSTAR_SHM_ARENA_MAX_SEGMENTS", "2")
    mgr = ArenaShmCommunicationManager(
        my_entity_id="w_store", hostname="localhost", device="cpu",
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        shm_dir=str(tmp_path),
    )
    mgr.tensor_store = TensorStore(RustTensorBookkeeping())
    staged, local = torch.randn(16), torch.randn(16)
    stored = mgr.store_and_return_tensor_info_batch(
        ["r"], {"r": {"a": [staged], "b": [local]}}, ["a", "b"],
        skip_cuda_sync=True,
    )
    a, b = stored.flat_uuids
    mgr.increment_ref_batch_uniform([a, b], 1)
    mgr.register_for_send_uuids(ParallelList(["r"], [[a]]))
    assert a in mgr._arena_locs

    seen = []
    real = mgr.cleanup_collectable
    mgr.cleanup_collectable = lambda u, r: (seen.extend(u), real(u, r))[1]
    mgr.cleanup_collectable(
        *mgr.tensor_store.bookkeeping.dereference_batch_uniform([a, b], 1, cleanup=True)
    )
    assert seen == [a], "only the staged one needs the arena's teardown"
    assert not mgr._arena_locs, "its slot went back to the arena"
    assert mgr.tensor_store.stored_counts() == (0, 0)
    mgr.close()


def test_a_shard_dim_keeps_its_uuid_reported():
    """A uuid with a shard dim to undo on read has Python state a silent
    free would leak, so it is flagged and comes back like a registered one."""
    _, mgr = _pair()
    mgr.register_request(
        1, ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={"x": 1}),
    )
    stored = mgr.store_and_return_tensor_info_batch(
        [1], {1: {"x": [torch.zeros(2, 3)]}}, ["x"], skip_cuda_sync=True,
    )
    (uuid,) = stored.flat_uuids
    assert mgr.uuid_to_shard_dim == {uuid: 1}
    assert mgr.get_tensor(uuid).shape == (2, 3), "the shard dim is undone on read"
    assert mgr.get_tensors([uuid])[0].shape == (2, 3)
    mgr.increment_ref_batch_uniform([uuid], 1)
    mgr.dereference_batch_uniform([uuid])
    assert mgr.uuid_to_shard_dim == {}
    assert not mgr.tensor_store.check_uuid_presence(uuid)


def test_teardown_frees_what_a_request_owns_and_purges_its_handle():
    """Handles are recycled, so a request's ownership must not outlive it --
    except for what is deferred, which the soft teardown keeps by design."""
    _, mgr = _pair()
    store = mgr.tensor_store
    tensors = [torch.ones(2) for _ in range(4)]
    refs = [weakref.ref(t) for t in tensors]
    stored = mgr.store_and_return_tensor_info_batch(
        [7], {7: {"x": tensors}}, ["x"], skip_cuda_sync=True,
    )
    del tensors
    u0, u1, u2, u3 = stored.flat_uuids
    mgr.increment_ref_batch_uniform([u1], 1)       # still read by someone
    mgr.register([u2])                              # transport state

    mgr.cleanup_request(7)
    assert sorted(store.get_all_uuids(7)) == [u1], "the referenced one is deferred"
    assert mgr.torn_down == [u2]
    assert [r() is None for r in refs] == [True, False, True, True]

    mgr.force_cleanup_request(7)
    assert store.get_all_uuids(7) == []
    assert store.stored_counts() == (0, 0)
    assert all(r() is None for r in refs)


def test_store_round_trip_and_host_copies():
    s = TensorStore(RustTensorBookkeeping())
    t, cpu = torch.arange(3), torch.arange(3)
    s.put_tensor(5, 11, t, _info(11), cpu_tensor=cpu)
    assert s.get_tensor(11) is t and s.get_cpu_tensor(11) is cpu
    assert s.get_tensors([11, 12]) == [t, None]
    # A re-put under another request moves the ownership with it.
    s.put_tensor("other", 11, t, _info(11))
    assert s.get_all_uuids(5) == [] and s.get_all_uuids("other") == [11]
    assert s.get_cpu_tensor(11) is None
    s.remove_tensor(11)
    assert s.stored_counts() == (0, 0) and s.get_info(11) is None
    with pytest.raises(ValueError):
        s.put_tensor_batch_multi([1], [1, 2], [t, t], [_info(1), _info(2)])


# --- through the Rust graph runtime ----------------------------------------

def _decode_runtime(sock_dir, bookkeeping):
    from mstar_rust import ZmqCommunicator

    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.graph.runtime.rust import RustGraphRuntime
    from mstar.model.base import WorkerGraph

    section = Loop(
        name="loop",
        section=GraphNode(
            name="LLM", input_names=["tok"],
            outputs=[GraphEdge(next_node="LLM", name="tok")],
        ),
        max_iters=8, outputs=[],
    )
    wg = WorkerGraph(section=section, graph_walks={"decode"}, ranks=[0], worker_graph_id=0)
    return RustGraphRuntime(
        my_worker_id="w0", my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {"decode"}}, all_wg_ids_to_dyn_loops={0: {"loop"}},
        all_wg_ids_to_nodes={0: {"LLM"}}, node_to_partition={"LLM": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=bookkeeping,
        communicator=SimpleNamespace(_inner=ZmqCommunicator("w0", sock_dir)),
    )


def _ingest_stored(rt, store, registered: bool):
    from mstar.graph.runtime.base import ColumnarEdgeSpecs

    rid = rt.add_request(
        request_id="r0", partition="default", graph_walk="decode",
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["w0"]]), loop_caps={},
    )
    t = torch.ones(2)
    store.put_tensor(rid, 100, t, _info(100))
    store.increment_ref(100, 1)
    if registered:
        store.set_metadata(100, mem_registered=True)
    block = ColumnarEdgeSpecs.empty()
    block.add(rid, "tok", [100], False, next_node="LLM")
    rt.ingest_inputs_batch(block)
    return rid, weakref.ref(t)


@pytest.mark.parametrize("registered", [False, True])
def test_the_runtime_frees_a_consumed_input_in_place(tmp_path, registered):
    """``cleanup_consumed_inputs`` reports only what the transport must tear
    down; the rest is gone from the store when it returns."""
    bk = RustTensorBookkeeping()
    store = TensorStore(bk)
    rt = _decode_runtime(str(tmp_path), bk)
    rid, ref = _ingest_stored(rt, store, registered)
    rt.pop_rids("LLM", "decode", [rid])

    freed = rt.cleanup_consumed_inputs("LLM", [rid], [0])
    if registered:
        assert (list(freed.uuids), list(freed.registered)) == ([100], [True])
        assert ref() is not None, "held for the transport's teardown"
        store.remove_tensor(100)
    else:
        assert list(freed.uuids) == []
    assert ref() is None
    assert store.stored_counts() == (0, 0)


def test_a_tensor_freed_with_the_gil_released_is_gone_on_return(tmp_path):
    """``route_and_send_inline`` routes inside ``allow_threads``, so a consumed
    input its completion frees is dropped without the GIL. pyo3 defers that
    decref to its pool and drains the pool when the GIL comes back at the end
    of the call -- so the tensor is gone by the time the call returns, and the
    step needs no Python fallback for it."""
    bk = RustTensorBookkeeping()
    store = TensorStore(bk)
    rt = _decode_runtime(str(tmp_path), bk)
    rid, ref = _ingest_stored(rt, store, registered=False)
    rt.pop_rids("LLM", "decode", [rid])

    column = torch.tensor([5], dtype=torch.int64)
    status, _cid, _sent, route = rt.route_and_send_inline(
        "default", "decode", "LLM", ["tok"], [rid], "tok", column, [rid],
    )
    assert status == 1, "nothing left for Python to tear down"
    assert route is None
    assert ref() is None
    assert store.stored_counts() == (0, 0)


def test_dropping_host_copies_keeps_the_tensors():
    """``drop_cpu_tensors`` forgets the host copies a step stored (they are
    views of buffers the producer reuses) and leaves the tensors held."""
    store = TensorStore(RustTensorBookkeeping())
    a, b = torch.tensor([1]), torch.tensor([2])
    store.put_tensor(1, 11, a, _info(11), cpu_tensor=torch.tensor([10]))
    store.put_tensor(1, 12, b, _info(12))
    assert store.get_cpu_tensor(11) is not None
    store.drop_cpu_tensors()
    assert store.get_cpu_tensor(11) is None
    assert store.get_tensor(11) is a and store.get_tensor(12) is b
    store.put_tensor(1, 13, torch.tensor([3]), _info(13), cpu_tensor=torch.tensor([30]))
    store.drop_cpu_tensors()
    assert store.get_cpu_tensor(13) is None
