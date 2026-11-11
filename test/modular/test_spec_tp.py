"""Speculative decoding on a two-rank lockstep instance: CPU processes over gloo,
each running the engine's step path on the same requests (`spec_toy.py`).

Rank 1 drafts differently and reads its logits differently, as a drafter or a
reduction that isn't bitwise alike across ranks would. Both ranks must still
emit the same tokens, take back to the same lengths and hold the same KV,
because they keep rank 0's drafts (``agree``) and rank 0's result (the
sampler's ``verify``). Without the broadcasts the ranks come apart, which
shows the test can tell.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))

WORLD = 2
PROMPTS = {1: [5, 9, 2], 2: [17], 3: [1, 2, 3, 4, 5, 6]}


def _comm(rank: int):
    from mstar.distributed.communication import CommGroup, JointGroups

    tp = CommGroup(rank, rank, list(range(WORLD)))
    tp.device_group = dist.group.WORLD
    tp.initialized = True
    return JointGroups(tp_group=tp, sp_group=CommGroup.trivial(), instance=tp)


def _run(rank: int, mode: str) -> dict:
    # a spawned rank imports what pytest's conftest would have stood in for
    import conftest  # noqa: F401
    import spec_toy as toy

    from mstar.engine.resources.kv import manager as manager_mod

    class _NoTransfer:
        def __init__(self, *a, **k):
            pass

        def get_kv_transfer_info(self, **k):
            pass

        def remove_request(self, rid):
            pass

        def cleanup(self):
            pass

    manager_mod.KVTransferManager = _NoTransfer
    k = 3
    # rank 1 spoils other drafts than rank 0
    drafter = toy.every(2) if rank == 0 else toy.every(3)
    eng = toy.engine(k, drafter)
    sub = eng._submodules[toy.NODE].submodule
    target = sub._target

    def skewed(engine_inputs, ids):
        out = target(engine_inputs, ids)
        logits = out.logits
        if rank == 1 and engine_inputs.step.ctx.graph_walk == "decode":
            # rank 1's argmax moves on every third row of a verify step (the toy's
            # prompt step takes the argmax without the sampler's broadcast)
            logits[::3] = logits[::3].roll(1, dims=-1)
        return out

    sub._target = skewed
    if mode != "no_broadcast":
        eng._resources["sampler"]._comm_group = _comm(rank)
    driver = toy.Driver(eng)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, 20)
    lengths = []
    for _ in range(4):
        driver.step(list(PROMPTS))
        lengths.append({rid: eng._resources["kv"]._streams[rid]["main"].stored_len for rid in PROMPTS})
    kv = eng._resources["kv"]
    held = {
        rid: toy._read_ids(kv, toy._views_of(kv, rid), kv._streams[rid]["main"].stored_len).tolist()
        for rid in PROMPTS
    }
    return {"out": driver.out, "lengths": lengths, "kv": held}


def _entry(rank, store, out, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=120),
    )
    torch.set_num_threads(1)
    try:
        torch.save(_run(rank, mode), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, mode):
    store, out = tmp_path / "store", tmp_path / "out"
    mp.spawn(_entry, args=(str(store), str(out), mode), nprocs=WORLD, join=True)
    return [torch.load(f"{out}.{rank}", weights_only=False) for rank in range(WORLD)]


def test_ranks_emit_and_hold_the_same_with_rank_0s_drafts_and_result(tmp_path):
    a, b = _spawn(tmp_path, "both")
    assert a["out"] == b["out"]
    assert a["lengths"] == b["lengths"]
    assert a["kv"] == b["kv"]


def test_without_the_broadcasts_the_ranks_come_apart(tmp_path):
    a, b = _spawn(tmp_path, "no_broadcast")
    assert a["out"] != b["out"] or a["lengths"] != b["lengths"]
