"""Kimi K3.1's captured decode on the release cut, traced: does the shared-experts branch run beside the routed
experts? torch.profiler (CUPTI) records the replayed graphs' kernels with their streams; or under nsys:

    python benchmark/kimi_k3/moe_overlap_trace.py --ckpt <mxfp4 cut> --bs 1 --overlap routed --out trace.json
    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -o trace \\
        python benchmark/kimi_k3/moe_overlap_trace.py --ckpt <mxfp4 cut> --bs 1 --nsys
    python benchmark/kimi_k3/moe_overlap_trace.py --analyze trace.sqlite

The analysis takes one decode replay: its span, the kernels' sum, how long two or more kernels ran at once,
per stream the kernels and their time, and which side-stream kernels overlapped which main-stream kernels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def trace(args) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "kimi_k3"))
    import gpu_decode_profile as P

    P.OVERLAP = args.overlap
    engine, sub, _ = P.build(args.ckpt, "mlp", capture=True, sizes=(args.bs,))
    inputs = P.start(engine, args.bs, sub.config.vocab_size)
    for _ in range(5):
        inputs = P.decode(engine, inputs)
    torch.cuda.synchronize()
    if args.nsys:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.steps):
            inputs = P.decode(engine, inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                inputs = P.decode(engine, inputs)
            torch.cuda.synchronize()
        kernels = [{"name": e["name"], "ts": e["ts"], "dur": e["dur"], "stream": e.get("args", {}).get("stream"),
                    "graph": e.get("args", {}).get("graph id"), "corr": e.get("args", {}).get("correlation")}
                   for e in _chrome_events(prof) if e.get("cat") == "kernel"]
        Path(args.out).write_text(json.dumps(kernels))
    print(json.dumps({"traced_steps": args.steps, "bs": args.bs, "overlap": args.overlap}), flush=True)


def _chrome_events(prof) -> list[dict]:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        return json.loads(Path(f.name).read_text())["traceEvents"]


def _replays(kernels: list[tuple]) -> list[list[tuple]]:
    """Split ``(start, end, stream, name)`` kernels (sorted) into decode steps at gaps over 30 us with no
    kernel running (between two steps the host prepares the next one)."""
    steps, cur, busy_until = [], [], None
    for k in kernels:
        if cur and k[0] - busy_until > 30_000:
            steps.append(cur)
            cur = []
        cur.append(k)
        busy_until = k[1] if busy_until is None else max(busy_until, k[1])
    if cur:
        steps.append(cur)
    return steps


def analyze(path: str, top: int) -> dict:
    """One decode step's concurrency, from a torch.profiler kernel list (.json) or an nsys export (.sqlite)."""
    if path.endswith(".json"):
        ks = sorted((int(e["ts"] * 1e3), int((e["ts"] + e["dur"]) * 1e3), e["stream"], e["name"])
                    for e in json.loads(Path(path).read_text()))
    else:
        import sqlite3

        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        names = dict(db.execute("select id, value from StringIds"))
        ks = sorted((s, e, st, names.get(n, "?")) for s, e, st, n in
                    db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"))
    steps = _replays(ks)
    big = [st for st in steps if len(st) >= max(len(x) for x in steps) * 0.9]
    ks = big[len(big) // 2]
    t0, t1 = min(k[0] for k in ks), max(k[1] for k in ks)
    events = sorted([(k[0], 1) for k in ks] + [(k[1], -1) for k in ks])
    depth, last, concurrent = 0, t0, 0
    for t, d in events:
        if depth >= 2:
            concurrent += t - last
        depth += d
        last = t
    streams: dict = {}
    for k in ks:
        streams.setdefault(k[2], []).append(k)
    main = max(streams, key=lambda s: len(streams[s]))
    side = [k for k in ks if k[2] != main]
    pairs = []
    for k in side:
        hits = [m for m in streams[main] if m[0] < k[1] and m[1] > k[0]]
        shared_ns = sum(min(k[1], m[1]) - max(k[0], m[0]) for m in hits)
        pairs.append({"side_kernel": k[3][:60], "us": round((k[1] - k[0]) / 1e3, 2),
                      "overlapped_us": round(shared_ns / 1e3, 2), "beside": sorted({m[3][:40] for m in hits})[:3]})
    report = {"steps_found": len(steps), "step_kernels": len(ks), "span_us": round((t1 - t0) / 1e3, 1),
              "kernel_sum_us": round(sum(k[1] - k[0] for k in ks) / 1e3, 1),
              "two_or_more_at_once_us": round(concurrent / 1e3, 1),
              "streams": {str(s): {"kernels": len(v), "us": round(sum(k[1] - k[0] for k in v) / 1e3, 1)}
                          for s, v in sorted(streams.items(), key=lambda kv: -len(kv[1]))},
              "side_stream_kernel_us": round(sum(k[1] - k[0] for k in side) / 1e3, 1),
              "side_stream_overlapped_us": round(sum(p["overlapped_us"] for p in pairs), 1),
              "side_kernels": pairs[:top]}
    print(json.dumps(report, indent=1), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--overlap", choices=("routed", "shared", "off"), default="routed")
    ap.add_argument("--out", default="trace.json", help="the torch.profiler kernel list")
    ap.add_argument("--nsys", action="store_true", help="mark the range for nsys instead of torch.profiler")
    ap.add_argument("--analyze", help="a trace run's kernel list (.json) or an nsys export (.sqlite)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze, args.top)
    else:
        trace(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
