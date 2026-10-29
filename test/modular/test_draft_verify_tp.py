"""Draft-verify on a two-rank lockstep instance: CPU processes over gloo, each running the
engine's step path on the same requests.

Rank 1 drafts differently at some positions and reads its logits slightly differently, as a
drafter or a reduction that isn't bitwise alike across ranks would. Both ranks must still
emit the same tokens, rewind to the same lengths at every step, and hold the same KV, because
they keep rank 0's drafts (``agree``) and rank 0's verdict (the sampler's ``verify``). Without
either, the ranks come apart, which shows the test can tell. The same holds at temperature
> 0, with drafts sampled from distributions rank 1 reads differently: each rank draws its own
verdict, and rank 0's is the one kept. With a KDA layer, both ranks settle their recurrent state
from rank 0's verdict and hold the same state, checkpoint and verify blocks.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402

PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25]]
WORLD = 2


def _comm(rank: int) -> JointGroups:
    tp = CommGroup(rank, rank, list(range(WORLD)))
    tp.device_group = dist.group.WORLD
    tp.initialized = True
    return JointGroups(tp_group=tp, sp_group=CommGroup.trivial())


def _entry(rank, store, out, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    torch.set_num_threads(1)
    try:
        torch.save(_run(rank, mode), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _run(rank: int, mode: str) -> dict:
    cpu = torch.device("cpu")
    sampled = mode.startswith("sampled")
    if sampled:
        # rank 1 reads both the draft head's logits and the target's differently, so it draws
        # other drafts from other distributions, and verifies them differently
        sub = toy.ToySubmodule(k=3, sampled_drafts=True)
    else:
        sub = toy.ToySubmodule(k=3, corrupt=4, skew=2, linear="kda" if "kda" in mode else None)
    sub.rank = rank
    sub.logit_skew = 0.5
    eng = toy.engine(sub, cpu, comm=_comm(rank), kv_heads=toy.HEADS * WORLD, sampled_verify=sampled)
    kv, pos, dv = (eng._resources[key] for key in (toy.KV, toy.POS, toy.DV))
    if mode == "no_agree":
        dv._group = None
    elif mode in ("no_verdict_broadcast", "sampled_no_verdict_broadcast", "kda_no_verdict_broadcast"):
        eng._resources[toy.SAMPLER]._comm_group = None
    drv = toy.Driver(eng, sub)
    rids = list(range(len(PROMPTS)))
    stopping = mode in ("both", "sampled", "kda")
    sampling = {"temperature": 0.9, "top_k": 12} if sampled else {}
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, 16, ignore_eos=not stopping, seed=7 + rid, **sampling)
    lengths = []
    if stopping:
        # the stops come from the verdicts, so both ranks run the same steps
        live = list(rids)
        while live:
            drv.decode(live)
            lengths.append([(kv._streams[r]["main"].stored_len, pos.position(r, "main")) for r in live])
            live = [r for r in live if r not in drv.stops]
    else:
        # a fixed number of steps: ranks that disagree must not wait on each other's stops
        for _ in range(4):
            drv.decode(rids)
            lengths.append([(kv._streams[r]["main"].stored_len, pos.position(r, "main")) for r in rids])
    # take the last verdict back as the next step would, then read what each stream keeps
    verdicts = dv.settle(rids)
    if verdicts is not None:
        kv.rewind(verdicts)
        pos.rewind(verdicts)
    rows = {}
    for rid in rids:
        stream = kv._streams[rid]["main"]
        pages = torch.as_tensor(stream.page_indices)
        rows[rid] = torch.stack([
            kv.layer_view(layer)[pages].transpose(0, 1).reshape(2, -1, toy.HEADS, toy.HEAD_DIM)[:, :stream.stored_len]
            for layer in range(toy.LAYERS)
        ])
    state = {}
    if sub.kda is not None:
        # every block of each request's slot: the checkpoint, the conv window, the verify blocks
        pool = eng._resources[toy.STATE]
        for rid in rids:
            slot = pool._slots[rid]["main"].index
            state[rid] = {name: (t[:, slot] if pool.config.blocks[name].per_layer else t[slot]).clone()
                          for name, t in pool._blocks.items()}
    return {"emitted": dict(drv.emitted), "lengths": lengths, "rows": rows, "accepted": dv.drafts_accepted,
            "state": state}


def _both(tmp_path, mode: str):
    store, out = tmp_path / "store", tmp_path / "out"
    mp.spawn(_entry, args=(str(store), str(out), mode), nprocs=WORLD)
    return [torch.load(f"{out}.{rank}") for rank in range(WORLD)]


def _rows_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[rid].shape == b[rid].shape and torch.equal(a[rid], b[rid]) for rid in a
    )


def test_the_ranks_keep_rank_zeros_drafts_and_verdicts(tmp_path):
    r0, r1 = _both(tmp_path, "both")
    ref = {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(PROMPTS)}
    assert r0["emitted"] == r1["emitted"] == ref
    assert r0["lengths"] == r1["lengths"]
    assert _rows_equal(r0["rows"], r1["rows"])
    assert r0["accepted"] == r1["accepted"] > 0


def _states_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[rid].keys() == b[rid].keys() and all(torch.equal(a[rid][n], b[rid][n]) for n in a[rid])
        for rid in a
    )


def test_the_ranks_settle_their_recurrent_state_alike(tmp_path):
    """Rank 1 drafts and reads its logits apart, so it would keep other drafts than rank 0;
    each rank settles its KDA state from rank 0's verdict inside the forward, so their
    checkpoints, conv windows, kept counts and sides stay bitwise alike."""
    r0, r1 = _both(tmp_path, "kda")
    assert r0["emitted"] == r1["emitted"]
    assert r0["lengths"] == r1["lengths"]
    assert _rows_equal(r0["rows"], r1["rows"])
    assert r0["accepted"] == r1["accepted"] > 0
    assert r0["state"] and _states_equal(r0["state"], r1["state"])
    # each request's last block kept a count the verdicts gave it
    assert all(0 < int(s["dv_len"][0]) <= 4 for s in r0["state"].values())


def test_at_temperature_the_ranks_keep_rank_zeros_sampled_verdicts(tmp_path):
    r0, r1 = _both(tmp_path, "sampled")
    assert r0["emitted"] == r1["emitted"]
    assert r0["lengths"] == r1["lengths"]
    assert _rows_equal(r0["rows"], r1["rows"])
    assert r0["accepted"] == r1["accepted"] > 0
    # sampled, not the greedy stream
    assert r0["emitted"] != {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(PROMPTS)}


@pytest.mark.parametrize(
    "mode", ["no_agree", "no_verdict_broadcast", "sampled_no_verdict_broadcast", "kda_no_verdict_broadcast"],
)
def test_without_the_broadcasts_the_ranks_come_apart(tmp_path, mode):
    r0, r1 = _both(tmp_path, mode)
    if mode == "no_agree":
        # the verdict is still rank 0's, so the tokens and lengths agree; rank 1's KV holds
        # its own drafts where rank 0's were accepted
        assert r0["emitted"] == r1["emitted"]
        assert not _rows_equal(r0["rows"], r1["rows"])
    else:
        assert r0["lengths"] != r1["lengths"] or r0["emitted"] != r1["emitted"]
    if mode.startswith("kda"):
        # the settles followed each rank's own verdict
        assert not _states_equal(r0["state"], r1["state"])
