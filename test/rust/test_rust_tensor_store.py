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


def _state(mgr, uuids, rids, alive):
    """Everything observable, plus which stored tensor objects are still alive
    (``alive``: uuid -> weakref to the object this store was given), so a
    store that forgets a tensor but keeps a reference to it differs."""
    s = mgr.tensor_store
    return {
        "alive": {u for u, ref in alive.items() if ref() is not None},
        "cpu": {u for u in uuids if s.get_cpu_tensor(u) is not None},
        "refs": {u: _ref_count(s, u) for u in uuids},
        "present": {u for u in uuids if s.check_uuid_presence(u)},
        # a held tensor is the object the store was given
        "objects": {
            u for u in uuids if s.check_uuid_presence(u) and s.get_tensor(u) is alive[u]()
        },
        "gc": {u: s.can_gc(u) for u in uuids},
        "info": {u: s.get_info(u) is not None for u in uuids},
        "owned": {r: sorted(s.get_all_uuids(r)) for r in rids},
        "transport": set(mgr.transport),
        "torn_down": sorted(mgr.torn_down),
    }


@pytest.mark.parametrize("seed", range(20))
def test_random_sequences_match_the_python_store(seed):
    """Store (with and without a host copy), reference, dereference (single,
    batched, by the runtime's release), persist, register, drop host copies
    and tear down at random, and compare everything observable after every
    step, including which tensors each store still keeps alive."""
    rng = random.Random(seed)
    py, rs = _pair()
    rids = [1, 2, 3, "api-x"]
    for mgr in (py, rs):
        cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
        for rid in rids:
            mgr.register_request(rid, cfg)
    minted: list[int] = []
    alive = {False: {}, True: {}}
    # uuids for host-copy puts, past anything the managers mint
    host_uuid = 1 << 40

    def state(mgr):
        return _state(mgr, minted, rids, alive[mgr is rs])

    for step in range(120):
        op = rng.choice([
            "store", "store", "store_host", "ref", "deref_batch", "deref_batch",
            "deref_one", "release", "persist", "register", "drop_host", "teardown",
        ])
        live = [u for u in minted if py.tensor_store.check_uuid_presence(u)]
        if op == "store":
            rid = rng.choice(rids)
            n = rng.randint(1, 4)
            outs = {}
            for mgr in (py, rs):
                # each store its own objects, so liveness is per store
                tensors = [torch.full((2,), float(len(minted) + i)) for i in range(n)]
                outs[mgr is rs] = mgr.store_and_return_tensor_info_batch(
                    [rid], {rid: {"x": tensors}}, ["x"],
                    node_name="N", graph_walk="w", skip_cuda_sync=True,
                )
                alive[mgr is rs].update(zip(
                    outs[mgr is rs].flat_uuids, map(weakref.ref, tensors), strict=True,
                ))
                del tensors
            # Both minted from their own counters, from the same start.
            assert outs[False].flat_uuids == outs[True].flat_uuids
            minted.extend(outs[True].flat_uuids)
            if rng.random() < 0.7:
                for mgr in (py, rs):
                    mgr.increment_ref_batch_uniform(outs[True].flat_uuids, 1)
        elif op == "store_host":
            host_uuid += 1
            rid = rng.choice(rids)
            for mgr in (py, rs):
                t = torch.full((2,), float(host_uuid))
                mgr.tensor_store.put_tensor(rid, host_uuid, t, _info(host_uuid), cpu_tensor=t.clone())
                mgr.increment_ref_batch_uniform([host_uuid], 1)
                alive[mgr is rs][host_uuid] = weakref.ref(t)
                del t
            minted.append(host_uuid)
        elif op == "drop_host":
            for mgr in (py, rs):
                mgr.drop_host_copies()
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
        assert state(py) == state(rs), (seed, step, op)

    for rid in rids:
        for mgr in (py, rs):
            mgr.force_cleanup_request(rid)
    assert state(py) == state(rs)
    assert not state(rs)["alive"], "every tensor freed once its request is gone"
    assert rs.tensor_store.stored_counts() == (0, 0)
    assert not rs.transport and not py.transport


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


def test_the_runtime_reports_a_registered_input_and_keeps_it_for_the_transport(tmp_path):
    """``cleanup_consumed_inputs`` reports what the transport must tear down,
    and the store holds it until that teardown removes it. (An input with
    nothing to tear down is freed in place: see the GIL-released test.)"""
    bk = RustTensorBookkeeping()
    store = TensorStore(bk)
    rt = _decode_runtime(str(tmp_path), bk)
    rid, ref = _ingest_stored(rt, store, registered=True)
    rt.pop_rids("LLM", "decode", [rid])

    freed = rt.cleanup_consumed_inputs("LLM", [rid], [0])
    assert (list(freed.uuids), list(freed.registered)) == ([100], [True])
    assert ref() is not None, "held for the transport's teardown"
    store.remove_tensor(100)
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
