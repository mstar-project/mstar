"""Microbenchmark of a decode step's KV ``admit`` / ``plan`` / ``commit``, on a
real ``KVManager``.

No server and no model. Streams are built through the real lifecycle (each
request's prompt prefilled, then decoded), and then ``--steps`` decode steps of
the whole batch run admit -> plan -> commit, each timed: a leased step (real
rows, then padding rows on SINK_PAGE), planned as a captured decode step is
when it is not pre-planned. On ``--device cuda`` the plan stages its write
locations through the pinned H2D stager as a server's does.

    python -m benchmark.kv_plan.bench_kv_step --device cuda

Run it against two checkouts by putting each first on ``PYTHONPATH``. It is
quieter than a server: the plan runs warm and alone, not beside a GPU thread
and a main thread, so per-call costs here are a lower bound; compare
checkouts, and confirm with ``benchmark/worker_phases`` on a served model.

Regimes (rows real / padded, prompt length range, page size 128):

* ``c128``: 124 / 128 rows, prompts of 96-768 tokens (Qwen3.5 0.8B at c=128).
* ``c32``: 30 / 32 rows, the same prompts.
* ``long``: 30 / 32 rows, 16k-32k tokens each.
* ``whisper``: 24 rows, up to 300 tokens (Whisper's decoder).
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics
import time

import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

REGIMES = {
    # real rows, bucket, prompt length range, page size, max_seq_len
    "c128": (124, 128, 96, 768, 128, 32768),
    "c32": (30, 32, 96, 768, 128, 32768),
    "long": (30, 32, 16384, 32000, 128, 32768),
    "whisper": (24, 24, 8, 300, 128, 448),
}


class _NoTransfer:
    """No remote transfers: the plan never reaches them."""

    def __init__(self, *a, **k):
        pass

    def get_kv_transfer_info(self, **k):
        pass

    def owns_transfer_info(self, *a, **k):
        return False

    def cleanup(self):
        pass

    def remove_request(self, rid):
        pass


def _step(rids, padded, bucket, slot):
    step = KVStep(segments=tuple(Segment(r, "main", 1) for r in padded))
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=bucket, num_tokens=bucket)),
    )
    ctx.set_padded_rids(padded)
    return step, ctx


def run(regime: str, steps: int, device: str, seed: int) -> dict[str, list[float]]:
    n_real, bucket, lo, hi, page_size, max_seq_len = REGIMES[regime]
    rng = random.Random(seed)
    lens = [rng.randint(lo, hi) for _ in range(n_real)]
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
            max_num_pages=sum(-(-(n + steps + 8) // page_size) for n in lens) + 64,
            page_size=page_size,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device(device), dtype=torch.float32,
    )
    kv._cg_max_seq_len = max(bucket, 64)
    rids = [f"req{i}" for i in range(n_real)]
    for rid, n in zip(rids, lens, strict=True):
        kv.ingest_request(rid, KVReqConfig())
        step = KVStep(segments=(Segment(rid, "main", n),))
        ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    padded = tuple(rids) + tuple(f"pad{i}" for i in range(bucket - n_real))
    times: dict[str, list[float]] = {"admit": [], "plan": [], "commit": []}
    gc.collect()
    for i in range(steps):
        step, ctx = _step(rids, padded, bucket, slot=i % 2)
        t0 = time.perf_counter()
        assert kv.admit(step, ctx).ok
        t1 = time.perf_counter()
        kv.plan(step, ctx)
        t2 = time.perf_counter()
        kv.commit(step, ctx)
        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        times["admit"].append(t1 - t0)
        times["plan"].append(t2 - t1)
        times["commit"].append(t3 - t2)
    return times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5, help="fresh managers per regime (seeds)")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    args = ap.parse_args()
    manager_mod.KVTransferManager = _NoTransfer
    print(f"checkout: {manager_mod.__file__}  device: {args.device}")
    for regime in args.regimes.split(","):
        runs = [run(regime, args.steps, args.device, seed) for seed in range(args.repeats)]
        cells = []
        for phase in ("admit", "plan", "commit"):
            med = statistics.median(statistics.median(r[phase]) for r in runs) * 1e6
            mean = statistics.median(statistics.mean(r[phase]) for r in runs) * 1e6
            cells.append(f"{phase} {med:7.1f} (mean {mean:7.1f})")
        print(f"{regime:8s} " + "  ".join(cells) + "  us/step")


if __name__ == "__main__":
    main()
