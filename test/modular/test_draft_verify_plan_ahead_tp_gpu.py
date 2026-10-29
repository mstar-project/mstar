"""Verify steps planned ahead of the verdict on a two-rank lockstep instance: one process per GPU
over NCCL, each running the toy's captured verify steps with every decode step pre-planned
ahead of the previous verdict and corrected for its rewind (``StepRunner.settle_preplan``).

Rank 1 drafts differently at some positions and reads its logits slightly differently. Both
ranks must pre-plan the same steps, emit the same tokens, rewind to the same lengths at every
step and hold the same KV, bitwise, as they do when each step is planned after the rewind: the
reservation, the rewind and the corrections are made from state every rank shares and from
rank 0's broadcast verdict. A rank that decided otherwise would hang the instance on its next
collective. Skips without two GPUs.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if torch.cuda.device_count() < 2:
    pytest.skip("needs two GPUs", allow_module_level=True)
pytest.importorskip("flashinfer")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402

PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
WORLD = 2


def _comm(rank: int) -> JointGroups:
    tp = CommGroup(rank, rank, list(range(WORLD)))
    tp.device_group = dist.group.WORLD
    # the rank votes (capture flags, the KV check after warmup) run over the gloo twin
    tp.cpu_group = dist.new_group(ranks=list(range(WORLD)), backend="gloo", use_local_synchronization=True)
    tp.initialized = True
    return JointGroups(tp_group=tp, sp_group=CommGroup.trivial())


def _entry(rank, store, out, plan_ahead):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{store}", rank=rank, world_size=WORLD,
        timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank),
    )
    try:
        torch.save(_run(rank, plan_ahead), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _run(rank: int, plan_ahead: bool) -> dict:
    gpu = torch.device("cuda", rank)
    sub = toy.ToySubmodule(k=3, corrupt=4, skew=2)
    sub.rank = rank
    sub.logit_skew = 0.5
    eng = toy.engine(
        sub, gpu, attn="flashinfer", comm=_comm(rank), kv_heads=toy.HEADS * WORLD, warmup=True,
        dtype=torch.float32, kv_dtype=torch.float16,
    )
    kv, pos, dv = (eng._resources[key] for key in (toy.KV, toy.POS, toy.DV))
    drv = toy.Driver(eng, sub, plan_ahead=plan_ahead)
    rids = list(range(len(PROMPTS)))
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, 16)
    lengths = []
    live = list(rids)
    while live:
        drv.decode(live)
        lengths.append([(kv._streams[r]["main"].stored_len, pos.position(r, "main")) for r in live])
        live = [r for r in live if r not in drv.stops]
    torch.cuda.synchronize()
    verdicts = dv.settle(rids)
    if verdicts is not None:
        kv.rewind(verdicts)
        pos.rewind(verdicts)
    rows = {}
    for rid in rids:
        stream = kv._streams[rid]["main"]
        pages = torch.as_tensor(stream.page_indices[: -(-stream.stored_len // toy.PAGE)], device=gpu)
        rows[rid] = torch.stack([
            kv.layer_view(layer)[pages].transpose(0, 1).reshape(2, -1, toy.HEADS, toy.HEAD_DIM)[:, :stream.stored_len]
            for layer in range(toy.LAYERS)
        ]).cpu()
    return {
        "emitted": dict(drv.emitted), "lengths": lengths, "rows": rows,
        "accepted": dv.drafts_accepted, "ahead": drv.planned_ahead, "steps": len(lengths),
    }


def _both(tmp_path, plan_ahead: bool):
    store, out = tmp_path / f"store{int(plan_ahead)}", tmp_path / f"out{int(plan_ahead)}"
    mp.spawn(_entry, args=(str(store), str(out), plan_ahead), nprocs=WORLD)
    return [torch.load(f"{out}.{rank}") for rank in range(WORLD)]


def _rows_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[rid].shape == b[rid].shape and torch.equal(a[rid], b[rid]) for rid in a
    )


def test_ranks_planning_ahead_stay_alike(tmp_path):
    r0, r1 = _both(tmp_path, plan_ahead=True)
    assert r0["ahead"] == r1["ahead"] == r0["steps"] > 0
    ref = {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(PROMPTS)}
    assert r0["emitted"] == r1["emitted"] == ref
    assert r0["lengths"] == r1["lengths"]
    assert _rows_equal(r0["rows"], r1["rows"])
    assert r0["accepted"] == r1["accepted"] > 0
    b0, b1 = _both(tmp_path, plan_ahead=False)
    assert b0["emitted"] == r0["emitted"] and b0["lengths"] == r0["lengths"]
    assert _rows_equal(b0["rows"], r0["rows"]) and _rows_equal(b1["rows"], r1["rows"])
