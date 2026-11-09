"""Kimi K3.1's prefix reuse against a cold prefill of the same tokens, on a truncated release
checkpoint (``python -m mstar.model.kimi_k3.truncate``), on GPU in bf16 or on CPU in fp32.

One engine with the prefix cache open (pages of 128 tokens, KDA snapshots) prefills ``p``, then
``q``, which shares p's first 1024 tokens: q restores the KDA state p kept at 896 (the last page
boundary of p's second 512-token chunk) and reads the latent pages and DSA index keys p wrote, and
runs on from there in chunks. ``q`` again then restores the snapshot q's first run kept at 1280.
A second engine, cache shut, prefills ``q`` cold in the same chunks and whole, decoding
teacher-forced on the first engine's tokens. A run passes when the hits skip what the rules say,
their logits stay within twice the cold run's own chunking spread (chunked against whole, at least
0.05), and their greedy tokens are the cold run's wherever its top-1 leads by more than that.

Cases: ``dense`` (under the release's index_topk of 2048, where dense MLA is K3.1's DSA) and
``dsa`` (dsa_long_context, index_topk cut to 256: past it, every row picks its keys from the
index keys on the pages, prefill rows too).

    python test/kimi_k3/gpu_prefix_parity.py --ckpt <dir> --out prefix.json
    python test/kimi_k3/gpu_prefix_parity.py --ckpt <bf16 dir> --out dsa.json --cases dsa  # a process a case
    python test/kimi_k3/gpu_prefix_parity.py --ckpt <tiny dir> --out p.json --device cpu --dtype float32 --cpu
    python test/kimi_k3/gpu_prefix_parity.py --ckpt <dir> --out tp2.json --tp 2  # two GPUs

``--tp 2`` runs both engines at TP2 with ``prefix_cache_tp``, stepped as the worker steps a TP group: rank 0
measures each prompt's cached prefix and, at each step it sends, leases the hits
(``Engine.decide_cached_prefix``); the follower takes rank 0's measure and adopts the leases
(``adopt_cached_prefix``) after checking its page allocations against rank 0's, and never looks a prefix
up for itself. Both ranks' runs are checked as TP1's, and must agree token for token.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_truncated_parity import _cpu_stubs, _step, build  # noqa: E402


class _TP:
    """One rank of a TP2 run: rank 0 decides what the worker's TP scheduling ships in a step's head
    (``ScheduleTPNode.prefix_hits`` and ``page_trails``), the follower adopts it."""

    def __init__(self, rank: int, cpu_group):
        self.rank, self.cpu_group = rank, cpu_group

    def bcast(self, obj):
        import torch.distributed as dist

        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.cpu_group)
        return box[0]

    def lockstep(self, engine, walk: str, rid: str, step_context) -> None:
        from mstar.utils.ipc_format import ScheduleTPNode

        head = None
        if self.rank == 0:
            hits = engine.decide_cached_prefix(SimpleNamespace(
                node_name="LLM", request_ids=(rid,), step_context=step_context))
            head = ScheduleTPNode("LLM", walk, [f"wire-{rid}"], prefix_hits=hits,
                                  page_trails=engine.page_trails("LLM"))
        head = self.bcast(head)
        if self.rank:
            engine.check_page_trails("LLM", walk, 0, head.page_trails)
            if not engine.adopt_cached_prefix(head, caught_up=True):
                raise RuntimeError(f"rank {self.rank}: {walk} {rid}: rank 0's leases {head.prefix_hits} not taken")


def _wire(engine) -> None:
    """Wire ids for the leases a TP head carries, as the worker's graph runtime names requests."""
    if engine._graph_runtime is None:
        engine._graph_runtime = SimpleNamespace(get_rid_string=lambda rid: f"wire-{rid}",
                                                get_rid_handle=lambda wire: wire.removeprefix("wire-"))


def _tp_step(engine, walk: str, rid: str, feed: torch.Tensor, tp: _TP):
    """``_step`` for one row, with the TP head exchanged before the step is built."""
    from mstar.model.kimi_k3.tests import engine_helpers as H

    batch = H._batch(walk, {rid: feed})
    tp.lockstep(engine, walk, rid, batch.step_context)
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    row = outputs.per_rid_outputs[rid]
    return {rid: row["new_token"][0]}, {rid: row["logits"][0].reshape(-1).float().cpu()}

PAGE, CHUNK, SHARED, TAIL = 128, 512, 1024, 512
# the snapshots q restores: p's second chunk's last boundary, then q's own first chunk's
EXPECT_CACHED = {"p": 0, "q": 896, "q_again": 1280}


def prefill(engine, model, rid: str, prompt: torch.Tensor, chunk: int, keyed: bool, tp: _TP | None = None):
    """``rid``'s prompt as the worker runs it: measured (a keyed row past its cached prefix), then
    in chunks of ``chunk`` from there. ``(tokens skipped, the first token, its logits, whether each
    chunk replayed a captured prefill)``."""
    from mstar.engine.chunking import ChunkRange
    from mstar.model.kimi_k3.tests import engine_helpers as H

    if keyed:
        H.add_keyed_request(engine, model, rid, prompt.cpu(), page_size=PAGE)
    else:
        H.add_request(engine, rid)
    total = len(prompt)
    def measure():
        return engine.input_sequence_len("LLM", "prefill", H._info(rid, "prefill"), {"text_inputs": [prompt]},
                                         rid=rid)

    # Above one rank the scheduler measures on rank 0, which leases the row's cached prefix there; the
    # row's first head carries that lease and the follower adopts it (its own lookup would see this
    # rank's later commits), then measures the row past it
    if tp is None or tp.rank == 0:
        measured = measure()
    if tp is not None:
        tp.lockstep(engine, "prefill", rid, H._batch("prefill", {rid: prompt}).step_context)
        if tp.rank:
            measured = measure()
    cached = start = measured.cached_prefix
    if tp is not None and tp.bcast(cached) != cached:
        raise RuntimeError(f"rank {tp.rank} measured {cached} cached tokens of {rid}, rank 0 {tp.bcast(None)}")
    first, replayed = True, []
    while True:
        end = min(start + chunk, total)
        batch = H._batch("prefill", {rid: prompt})
        # a first chunk's stride counts from 0, past a cached prefix too
        batch.chunk_ranges = {rid: ChunkRange(start, end, total, end if start == cached else end - start)}
        if tp is not None and not first:  # the first step's head went out with the measure
            tp.lockstep(engine, "prefill", rid, batch.step_context)
        first = False
        engine.prepare_inputs(batch)
        assert not batch.failed_requests, batch.failed_requests
        outputs = engine.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        replayed.append(batch.step_context.slot_lease is not None)
        engine.finalize_batch(batch)
        if end >= total:
            break
        start = end
    engine.release_chunk_inputs(rid, "LLM")
    row = outputs.per_rid_outputs[rid]
    return cached, row["new_token"][0].reshape(1), row["logits"][0].reshape(-1).float().cpu(), replayed


def serve(engine, model, rid: str, prompt: torch.Tensor, chunk: int, keyed: bool, steps: int,
          forced: list[int] | None = None, tp: _TP | None = None) -> dict:
    """Prefill, then ``steps`` greedy decodes, fed ``forced`` (teacher-forced) when given."""
    cached, tok, logits, replayed = prefill(engine, model, rid, prompt, chunk, keyed, tp)
    tokens, rows = [int(tok)], [logits]
    for k in range(steps):
        feed = tok if forced is None else torch.tensor([forced[k]], device=tok.device)
        if tp is None:
            got, step_logits = _step(engine, "decode", {rid: feed.reshape(1)})
        else:
            got, step_logits = _tp_step(engine, "decode", rid, feed.reshape(1), tp)
        tok = got[rid].reshape(1)
        tokens.append(int(tok))
        rows.append(step_logits[rid])
    engine.remove_request(rid)
    return {"cached": cached, "tokens": tokens, "logits": torch.stack(rows), "replayed": replayed}


def _prompts(vocab: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    high = min(vocab, 160000)
    low = min(1000, high // 4)
    shared, p_tail, q_tail = (torch.randint(low, high, (n,), generator=g) for n in (SHARED, TAIL, TAIL))
    return torch.cat([shared, p_tail]).to(device), torch.cat([shared, q_tail]).to(device)


def _free() -> None:
    """What of an engine's memory comes back before the next one loads. Not all of it: on an H200 the
    bf16 cut's second case runs out loading its first engine, so that cut takes a process per case."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare(name: str, run: dict, cold: dict, tol: float) -> dict:
    """``run``'s logits against the cold run's, teacher-forced on ``run``'s tokens: the largest gap per
    step, and the steps whose greedy token differs where the cold top-1 leads by more than ``tol``."""
    diff = (run["logits"] - cold["logits"]).abs().amax(-1)
    top2 = cold["logits"].topk(2, dim=-1).values
    lead = top2[:, 0] - top2[:, 1]
    picks = cold["logits"].argmax(-1).tolist()
    clear_misses = [k for k, (a, b) in enumerate(zip(run["tokens"], picks, strict=True)) if a != b and lead[k] > tol]
    return {"case": name, "cached": run["cached"], "logits_max_diff": round(diff.max().item(), 5),
            "logits_mean_step_diff": round(diff.mean().item(), 5), "prefill_diff": round(diff[0].item(), 5),
            "cold_logits_std": round(cold["logits"].std().item(), 4),
            "argmax_agree": sum(a == b for a, b in zip(run["tokens"], picks, strict=True)), "steps": len(picks),
            "clear_misses": clear_misses}


def run_case(args, name: str, overrides: dict, failures: list, tp: _TP | None = None, group=None,
             groups=None, device: str | None = None, keep: dict | None = None) -> list[dict]:
    device = device or args.device
    warm, model = build(args.ckpt, device, overrides=overrides, dtype=args.dtype, page_size=PAGE,
                        prefix_slots=8, group=group, groups=groups, prefill_buckets=args.prefill_buckets)
    _wire(warm)
    # as the worker sets it: a chunked walk's row is measured past its cached prefix
    warm.set_chunking_policies(model.chunking_policy_of())
    p, q = _prompts(model.config.vocab_size, device)
    runs = {"p": serve(warm, model, "p", p, CHUNK, True, 4, tp=tp)}
    runs["q"] = serve(warm, model, "q", q, CHUNK, True, args.steps, tp=tp)
    runs["q_again"] = serve(warm, model, "q_again", q, CHUNK, True, args.steps, forced=runs["q"]["tokens"], tp=tp)
    del warm
    _free()
    cold_engine, cold_model = build(args.ckpt, device, overrides=overrides, dtype=args.dtype, page_size=PAGE,
                                    group=group, groups=groups, prefill_buckets=args.prefill_buckets)
    _wire(cold_engine)
    cold_engine.set_chunking_policies(cold_model.chunking_policy_of())
    forced = runs["q"]["tokens"]
    cold = serve(cold_engine, None, "cold", q, CHUNK, False, args.steps, forced=forced, tp=tp)
    whole = serve(cold_engine, None, "whole", q, len(q), False, args.steps, forced=forced, tp=tp)
    # chunks of 320 tokens: on 64-token boundaries, as the scheduler cuts K3.1's (chunk_align), off the buckets;
    # and of 300: off them, as only a hand-cut chunk can be, which must run eager (can_replay) and still agree
    aligned = serve(cold_engine, None, "cold320", q, 320, False, args.steps, forced=forced, tp=tp)
    off = serve(cold_engine, None, "cold300", q, 300, False, args.steps, forced=forced, tp=tp)
    del cold_engine
    _free()
    if keep is not None:
        keep[name] = {rid: {"cached": r["cached"], "tokens": r["tokens"]} for rid, r in
                      {**runs, "cold": cold, "whole": whole}.items()}
    floor = compare(f"{name}/whole-vs-cold", {**whole, "tokens": forced}, cold, 0.0)
    tol = max(2 * floor["logits_max_diff"], 0.05)
    rows = [floor]
    for rid, want in EXPECT_CACHED.items():
        if runs[rid]["cached"] != want:
            failures.append(f"{name}/{rid}: skipped {runs[rid]['cached']} tokens, the rules say {want}")
    for rid, run in (("q", runs["q"]), ("q_again", runs["q_again"]), ("cold320", aligned), ("cold300", off)):
        row = compare(f"{name}/{rid}-vs-cold", run, cold, tol)
        row["tol"] = round(tol, 5)
        rows.append(row)
        if row["logits_max_diff"] > tol or row["clear_misses"]:
            failures.append(f"{name}/{rid}: off the cold run by more than its chunking spread ({row})")
    replays = {rid: r["replayed"] for rid, r in {**runs, "cold": cold, "cold320": aligned, "cold300": off}.items()}
    rows.append({"case": f"{name}/replayed_chunks", **replays})
    if args.prefill_buckets and not overrides.get("dsa_long_context"):
        # every chunk up to the largest bucket replays where it starts on a 64-token boundary: the prompts'
        # first chunks, the chunks past a prefix hit (896, 1280), every 320-token chunk; of the 300-token
        # chunks only the first does
        fits = max(args.prefill_buckets)
        for rid in ("p", "q", "q_again", "cold"):
            if CHUNK <= fits and not all(replays[rid]):
                failures.append(f"{name}/{rid}: a chunk on a 64-token boundary ran eager ({replays[rid]})")
        if not all(replays["cold320"]):
            failures.append(f"{name}/cold320: an aligned 320-token chunk ran eager ({replays['cold320']})")
        if replays["cold300"][:2] != [True, False]:
            failures.append(f"{name}/cold300: a chunk off a 64-token boundary replayed ({replays['cold300']})")
    for row in rows:
        print(json.dumps(row), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--cases", default="dense,dsa")
    ap.add_argument("--prefill-buckets", type=lambda v: [int(n) for n in v.split(",")], default=None,
                    help="capture prefill at these packed token counts: the hits and the cold run replay them")
    ap.add_argument("--tp", type=int, choices=(1, 2), default=1, help="2: both engines at TP2 on cuda:0 and cuda:1")
    # CPU: the FlashInfer stub and no CUDA (a tiny checkpoint's dry run)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.cpu:
        _cpu_stubs()
    length = SHARED + TAIL + args.steps + 8
    args.case_overrides = {"dense": {"max_seq_len": 2048, "index_topk": 2048},
                           "dsa": {"index_topk": 256, "dsa_long_context": True, "max_seq_len": length}}
    if args.tp == 2:
        return _tp2(args)
    failures: list[str] = []
    rows = []
    for name in args.cases.split(","):
        rows += run_case(args, name, args.case_overrides[name], failures)
    return _report(args.out, rows, failures)


def _report(out: str, rows: list, failures: list) -> int:
    Path(out).write_text(json.dumps({"rows": rows, "failures": failures}, indent=1))
    for f in failures:
        print("FAIL", f, flush=True)
    print("PASS" if not failures else f"{len(failures)} check(s) failed", flush=True)
    return 1 if failures else 0


def _tp2_rank(rank: int, args, port: int) -> None:
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True,
                                  world_parallel_groups=[(0, 1)], dist_timeout_s=1800)
    groups.add("LLM", group, "tp")
    groups.node_to_instance_comm["LLM"] = group
    if args.cpu:  # gloo, as gpu_truncated_parity's CPU TP2 (a tiny checkpoint's dry run)
        from mstar.distributed.all_reduce import AllReduce

        _cpu_stubs()
        device = "cpu"
        torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                             world_size=2, timeout=datetime.timedelta(seconds=1800))
        group.device_group = group.cpu_group = torch.distributed.new_group([0, 1])
        group.initialized = True
        group._all_reduce = AllReduce()
    else:
        device = f"cuda:{rank}"
        groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cuda", rank))
    tp = _TP(rank, group.cpu_group)
    code, failures, rows, keep = 0, [], [], {}
    try:
        for name in args.cases.split(","):
            rows += run_case(args, name, args.case_overrides[name], failures, tp=tp, group=group,
                             groups=groups, device=device, keep=keep)
    except BaseException:
        import traceback

        traceback.print_exc()
        code, failures = 1, failures + [f"rank {rank} raised"]
    Path(f"{args.out}.rank{rank}").write_text(json.dumps({"rows": rows, "failures": failures, "runs": keep}))
    sys.stdout.flush()
    sys.stderr.flush()
    # as gpu_truncated_parity's TP2 ranks: exit without the NCCL teardown (job 1461's ranks hung in it)
    os._exit(code)


def _tp2(args) -> int:
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        mp.spawn(_tp2_rank, args=(args, port), nprocs=2)
    except Exception as e:  # a rank's non-zero exit: its file says why
        print(f"a rank failed: {e}", flush=True)
    ranks = []
    for r in (0, 1):
        path = Path(f"{args.out}.rank{r}")
        missing = {"rows": [], "failures": [f"rank {r}: no results"], "runs": {}}
        ranks.append(json.loads(path.read_text()) if path.exists() else missing)
    failures = [f"rank {r}: {f}" for r, got in enumerate(ranks) for f in got["failures"]]
    for name, runs in ranks[0]["runs"].items():
        for rid, run in runs.items():
            peer = ranks[1]["runs"].get(name, {}).get(rid)
            same = peer is not None and peer == run
            print(json.dumps({"case": f"tp2-ranks/{name}/{rid}", "cached": run["cached"],
                              "follower_cached": None if peer is None else peer["cached"], "tokens_equal": same}),
                  flush=True)
            if not same:
                failures.append(f"tp2 {name}/{rid}: the ranks disagree (rank 0 {run}, rank 1 {peer})")
    return _report(args.out, ranks[0]["rows"], failures)


if __name__ == "__main__":
    os.environ.setdefault("MSTAR_PRE_PLAN_SPEC", "0")
    sys.exit(main())
