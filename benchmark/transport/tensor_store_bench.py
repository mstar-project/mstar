"""Host cost of the worker's tensor store: storing a step's outputs, releasing
consumed tensors, reading inputs back, tearing a request down.

CPU only, single process, no transport traffic: what is timed is the Python
and the Rust crossings a worker's main thread pays per step (the
``worker.postprocess.store_tensors`` and ``cleanup_inputs`` phases), with the
GIL held throughout.

Two stores over the same Rust bookkeeper (the ``MSTAR_RUST_GRAPH=1`` setup):

- ``py-store``: the Python ``TensorStore`` -- tensors and ownership in Python
  dicts, every collectable uuid reported back and torn down per uuid in
  Python. This is the store as it was before ``RustTensorStore``.
- ``rust-store``: ``RustTensorStore`` -- tensors and ownership in the
  bookkeeper, a collectable uuid that needs no transport teardown freed inside
  the Rust release.

    python -m benchmark.transport.tensor_store_bench
    python -m benchmark.transport.tensor_store_bench --sizes 28,64,256 --reps 3000
"""
from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time

import torch

from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.model.submodule_base import BatchedModelOutput


class _NullCommunicator:
    def send(self, *args, **kwargs):
        pass


def _store(kind: str) -> TensorStore:
    bk = RustTensorBookkeeping()
    if kind == "rust-store":
        return TensorStore(bk)
    # The Python store over the Rust bookkeeper, bypassing the dispatch in
    # TensorStore.__new__ (which would pick RustTensorStore).
    store = object.__new__(TensorStore)
    store.bookkeeping = bk
    TensorStore.__init__(store)
    return store


def _manager(kind: str, shm_dir: str, arena_like: bool):
    cls = SharedMemoryCommunicationManager
    if arena_like:
        from mstar.communication.arena import ArenaShmCommunicationManager

        # The arena's inlined batch teardown, without mapping an arena: the
        # uuids timed here are never staged, so it only walks its dicts.
        class _ArenaLike(SharedMemoryCommunicationManager):
            cleanup_collectable = ArenaShmCommunicationManager.cleanup_collectable

        cls = _ArenaLike
    mgr = cls(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=shm_dir,
    )
    mgr._arena_ts, mgr._arena_locs = {}, {}
    mgr.tensor_store = _store(kind)
    return mgr


def _register(mgr, rids):
    cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
    for rid in rids:
        mgr.register_request(rid, cfg)


def _row_batch(n: int):
    clone = torch.arange(n, dtype=torch.long).view(n, 1).clone()
    views = clone.split(1)
    rids = list(range(n))
    per_rid = {rid: {"new_token": [views[i]]} for i, rid in enumerate(rids)}
    return rids, BatchedModelOutput(
        per_rid_outputs=per_rid, row_views={"new_token": views},
        row_request_ids=tuple(rids),
    )


def _time(setup, op, after, reps: int) -> tuple[float, float]:
    """(median, mean) microseconds of ``op(setup())``; ``setup`` and
    ``after`` (which undoes what ``op`` left, so the store stays the same
    size) untimed."""
    samples = []
    gc.collect()
    for _ in range(reps):
        arg = setup()
        t0 = time.perf_counter()
        op(arg)
        samples.append((time.perf_counter() - t0) * 1e6)
        if after is not None:
            after()
    return statistics.median(samples), statistics.fmean(samples)


def _cases(mgr, n: int):
    rids, outputs = _row_batch(n)
    _register(mgr, rids)
    store = mgr.tensor_store
    bk = store.bookkeeping
    signals = ["new_token"]

    def drain(uuids):
        mgr.cleanup_collectable(*bk.dereference_batch_uniform(uuids, 1, cleanup=True))

    # -- store: the decode row path and the general path, each with the
    # safety hold the worker puts on right after (worker.postprocess.store_tensors)
    def store_rows(_):
        stored = mgr.store_row_outputs_batch(
            rids, outputs, signals, outputs.row_views, outputs.row_request_ids,
            node_name="LLM", graph_walk="decode",
        )
        mgr.increment_ref_batch_uniform(stored.flat_uuids, 1)
        store_rows.last = stored.flat_uuids

    def store_general(_):
        stored = mgr.store_and_return_tensor_info_batch(
            rids, outputs.per_rid_outputs, signals, node_name="LLM",
            graph_walk="decode", skip_cuda_sync=True,
        )
        mgr.increment_ref_batch_uniform(stored.flat_uuids, 1)
        store_general.last = stored.flat_uuids

    def stored_held(_=None):
        stored = mgr.store_row_outputs_batch(
            rids, outputs, signals, outputs.row_views, outputs.row_request_ids,
            node_name="LLM", graph_walk="decode",
        )
        mgr.increment_ref_batch_uniform(stored.flat_uuids, 1)
        return stored.flat_uuids

    yield "store_rows", lambda: None, store_rows, lambda: drain(store_rows.last)
    yield "store_general", lambda: None, store_general, lambda: drain(store_general.last)
    # -- release: what cleanup_consumed_inputs hands back, torn down
    # (worker.postprocess.cleanup_inputs)
    yield "release", stored_held, drain, None
    # -- read a batch's inputs back: per uuid, and the column form
    held = stored_held()
    get = mgr.get_tensor
    yield "get_per_uuid", lambda: held, lambda us: [get(u) for u in us], None
    yield "get_tensors", lambda: held, mgr.get_tensors, None
    drain(held)

    # -- teardown of a request that owns n tensors
    def owned_by_one(_=None):
        stored = mgr.store_and_return_tensor_info_batch(
            [0], {0: {"x": list(outputs.row_views["new_token"])}}, ["x"],
            node_name="LLM", graph_walk="decode", skip_cuda_sync=True,
        )
        return stored.flat_uuids

    def teardown(_):
        mgr.force_cleanup_request(0)
        _register(mgr, [0])

    yield "force_teardown", owned_by_one, teardown, None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", default="28,64,256",
                   help="uuids per batch (28 is one Orpheus stream chunk)")
    p.add_argument("--reps", type=int, default=2000)
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    kinds = ["py-store", "rust-store"]
    rows = []
    with tempfile.TemporaryDirectory() as shm_dir:
        for arena_like in (False, True):
            for n in sizes:
                results = {}
                for kind in kinds:
                    mgr = _manager(kind, shm_dir, arena_like)
                    for name, setup, op, after in _cases(mgr, n):
                        results[(name, kind)] = _time(setup, op, after, args.reps)
                names = dict.fromkeys(name for name, _ in results)
                for name in names:
                    a, b = results[(name, "py-store")], results[(name, "rust-store")]
                    rows.append((
                        "arena" if arena_like else "shm-file", n, name,
                        a[0], b[0], a[0] / b[0] if b[0] else float("nan"),
                    ))
    print(f"{'manager':9} {'n':>4} {'op':15} {'py-store us':>12} "
          f"{'rust-store us':>14} {'speedup':>8}   (medians)")
    for mgr_name, n, name, a, b, x in rows:
        print(f"{mgr_name:9} {n:>4} {name:15} {a:12.1f} {b:14.1f} {x:7.2f}x")


if __name__ == "__main__":
    main()
