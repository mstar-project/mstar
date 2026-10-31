"""A DSpark drafter on a two-rank lockstep instance: CPU processes over gloo, each running the
engine's step path on the same requests, the drafter's weights skewed on rank 1 (as a sharded
drafter whose reductions aren't bitwise alike would draft apart).

Both ranks must emit the same tokens, rewind to the same lengths and hold the same target KV:
they keep rank 0's drafts (``agree``) and rank 0's verdict. The same when each verdict is taken
back on the device in the step after it (``device_rewind``): the ranks stage it from the same
host state at the same point, and take it back on the host at the same later one.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402
import dspark_toy as ds  # noqa: E402
import test_draft_verify_tp as tp  # noqa: E402

PROMPTS = tp.PROMPTS
WORLD = tp.WORLD


def _entry(rank, store, out, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=120),
    )
    torch.set_num_threads(1)
    try:
        torch.save(_run(rank, mode), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _run(rank: int, mode: str) -> dict:
    cpu = torch.device("cpu")
    sub = ds.ToyDSparkSubmodule(
        k=4, drafter_seed=5 + rank, kv_heads=toy.HEADS * WORLD, device_rewind=mode.endswith("device"),
    )
    sub.rank = rank
    sub.logit_skew = 0.5
    eng = ds.engine(sub, cpu, comm=tp._comm(rank), kv_heads=toy.HEADS * WORLD)
    kv, pos, dv = (eng._resources[key] for key in (toy.KV, toy.POS, toy.DV))
    drv = ds.Driver(eng, sub)
    drv.add_batch({rid: p for rid, p in enumerate(PROMPTS)}, 16)
    rids = list(range(len(PROMPTS)))
    lengths, live = [], list(rids)
    while live:
        drv.decode(live)
        lengths.append([(kv._streams[r]["main"].stored_len, pos.position(r, "main")) for r in live])
        live = [r for r in live if r not in drv.stops]
    ds.settle_all(eng, rids)
    rows = {}
    for rid in rids:
        stream = kv._streams[rid]["main"]
        pages = torch.as_tensor(stream.page_indices)
        rows[rid] = torch.stack([
            kv.layer_view(layer)[pages].transpose(0, 1).reshape(2, -1, toy.HEADS, toy.HEAD_DIM)[:, :stream.stored_len]
            for layer in range(toy.LAYERS)
        ])
    return {
        "emitted": dict(drv.emitted), "lengths": lengths, "rows": rows, "accepted": dv.drafts_accepted,
    }


def _both(tmp_path, mode: str):
    store, out = tmp_path / "store", tmp_path / "out"
    mp.spawn(_entry, args=(str(store), str(out), mode), nprocs=WORLD)
    return [torch.load(f"{out}.{rank}") for rank in range(WORLD)]


def test_the_ranks_keep_rank_zeros_dspark_drafts_and_verdicts(tmp_path):
    r0, r1 = _both(tmp_path, "every_step")
    ref = {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(PROMPTS)}
    assert r0["emitted"] == r1["emitted"] == ref
    assert r0["lengths"] == r1["lengths"]
    assert tp._rows_equal(r0["rows"], r1["rows"])


def test_ranks_take_verdicts_back_on_the_device_alike(tmp_path):
    r0, r1 = _both(tmp_path, "every_step_device")
    ref = {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(PROMPTS)}
    assert r0["emitted"] == r1["emitted"] == ref
    assert r0["lengths"] == r1["lengths"]
    assert tp._rows_equal(r0["rows"], r1["rows"])

