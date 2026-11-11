"""Qwen3 with a DSpark drafter on a two-rank lockstep instance: CPU processes over
gloo, each running the real engine on the same requests (`qwen3_cpu`).

Rank 1's target head and drafter projection are perturbed, as a sharded model
whose reductions aren't bitwise alike across ranks would read its logits and
draft differently. Both ranks must still emit the same tokens, take back to
the same lengths and hold the same target KV, because they keep rank 0's
drafts (``agree``) and rank 0's result (the sampler's ``verify``). Without the
broadcasts the ranks come apart, which shows the test can tell.
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
PROMPTS = {1: [5, 9, 2, 7], 2: [17], 3: [1, 2, 3, 4, 5, 6]}
MAX = 16


def _comm(rank: int):
    from mstar.distributed.communication import CommGroup, JointGroups

    tp = CommGroup(rank, rank, list(range(WORLD)))
    tp.device_group = dist.group.WORLD
    tp.initialized = True
    return JointGroups(tp_group=tp, sp_group=CommGroup.trivial(), instance=tp)


@torch.no_grad()
def _perturb(module: torch.nn.Module, scale: float) -> None:
    gen = torch.Generator().manual_seed(11)
    module.weight.add_(torch.randn(module.weight.shape, generator=gen, dtype=module.weight.dtype) * scale)


def _run(rank: int, checkpoints: tuple[str, str], mode: str) -> dict:
    # a spawned rank stands in for what pytest's conftest and fixtures would have
    import conftest  # noqa: F401
    import qwen3_cpu as q

    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.sampler import utils as sampler_utils

    q.register_cpu_rope()
    sampler_utils.sample_tokens = q._cpu_draw
    manager_mod.KVTransferManager = q._StubTransfer
    manager_mod._DEBUG_ASSERTS = True
    target, dspark = (Path(p) for p in checkpoints)
    model = q.make_model(target, q.draft_block(dspark, 3))
    eng = q.engine(model, prepare=(lambda sub: _perturb(sub.drafter.fc, 1.0)) if rank == 1 else None)
    if mode == "broadcast":
        eng._resources["sampler"]._comm_group = _comm(rank)
    driver = q.Driver(eng, model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    driver.prefill(list(PROMPTS), PROMPTS)
    if rank == 1:
        # rank 1 reads its logits differently from here on (the prompt step's draw is
        # the CPU stand-in's, which no broadcast covers)
        _perturb(eng._submodules[q.NODE].submodule.model.lm_head, 0.5)
    kv = eng._resources["kv_cache"]
    lengths, live = [], list(PROMPTS)
    for _ in range(6):
        if not live:
            break
        stopped = driver.step(live)
        lengths.append({rid: kv._streams[rid]["main"].stored_len for rid in live})
        live = [rid for rid in live if not stopped[rid]]
    held = {}
    for rid in PROMPTS:
        stream = kv._streams[rid]["main"]
        n = stream.stored_len
        pages = torch.tensor([stream.page_indices[p // q.PAGE] for p in range(n)])
        offsets = torch.tensor([p % q.PAGE for p in range(n)])
        # the target's layers: the drafter's hold rank 1's perturbed context
        held[rid] = torch.stack([
            kv.kv_cache.read_tokens(layer_idx=layer, page_idx=pages, cache_idx=offsets)
            for layer in range(model.config.num_hidden_layers)
        ])
    return {"out": driver.out, "lengths": lengths, "kv": held}


def _entry(rank, store, out, checkpoints, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=120),
    )
    torch.set_num_threads(1)
    try:
        torch.save(_run(rank, checkpoints, mode), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, mode):
    from qwen3_cpu import set_context, write_checkpoints

    target, dspark = write_checkpoints(tmp_path / "ckpt")
    # once, before the ranks share the checkpoint (`make_model` then finds it set)
    set_context(target)
    checkpoints = (str(target), str(dspark))
    store, out = tmp_path / "store", tmp_path / "out"
    mp.spawn(_entry, args=(str(store), str(out), checkpoints, mode), nprocs=WORLD, join=True)
    return [torch.load(f"{out}.{rank}", weights_only=False) for rank in range(WORLD)]


def test_ranks_emit_and_hold_the_same_with_rank_0s_drafts_and_result(tmp_path):
    a, b = _spawn(tmp_path, "broadcast")
    assert a["out"] == b["out"]
    assert a["lengths"] == b["lengths"]
    assert all(torch.equal(a["kv"][rid], b["kv"][rid]) for rid in PROMPTS)


def test_without_the_broadcasts_the_ranks_come_apart(tmp_path):
    a, b = _spawn(tmp_path, "alone")
    assert a["out"] != b["out"] or a["lengths"] != b["lengths"]
