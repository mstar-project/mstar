#!/usr/bin/env python3
"""Summarise a StepKernelTrace chrome trace: kernels, gaps, collectives.

    python env/kernel_trace_summary.py step-trace-worker_0.json [--top 25]

Reads the JSON that ``MSTAR_PROFILE_STEPS`` writes (see
``mstar/utils/profiler.py``), keeps the GPU-side events (kernels, memcpy,
memset), and prints what the phase timers cannot: how many kernels a step is,
how long they run, how much of the wall is gaps between them, which names
dominate by time and by count, and how much of it is collectives. Gaps are
computed on the union of all streams, so overlapping streams do not count as
idle.
"""
import argparse
import collections
import json
import re

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
COLLECTIVE = re.compile(r"nccl|multimem|all_?reduce|allgather|all_?gather|symm", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--launches", action="store_true",
                    help="also count cudaGraphLaunch / cudaLaunchKernel runtime calls")
    args = ap.parse_args()

    with open(args.trace) as f:
        events = json.load(f)["traceEvents"]

    gpu = [e for e in events if e.get("cat") in GPU_CATS and "dur" in e]
    gpu.sort(key=lambda e: e["ts"])
    if not gpu:
        print("no GPU events in trace"); return

    span = gpu[-1]["ts"] + gpu[-1]["dur"] - gpu[0]["ts"]
    # union of intervals across streams -> busy; the rest is gap
    busy = 0.0
    cur_s, cur_e = gpu[0]["ts"], gpu[0]["ts"] + gpu[0]["dur"]
    gaps = []
    for e in gpu[1:]:
        s, t = e["ts"], e["ts"] + e["dur"]
        if s > cur_e:
            gaps.append(s - cur_e)
            busy += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    busy += cur_e - cur_s

    by_name = collections.defaultdict(lambda: [0, 0.0])
    for e in gpu:
        r = by_name[e["name"]]
        r[0] += 1; r[1] += e["dur"]
    coll = [(n, c, d) for n, (c, d) in by_name.items() if COLLECTIVE.search(n)]
    coll_n = sum(c for _, c, _ in coll); coll_t = sum(d for _, _, d in coll)
    durs = sorted(e["dur"] for e in gpu)
    def pct(p): return durs[min(len(durs) - 1, int(p * len(durs)))]
    launches = {}
    if args.launches:
        for e in events:
            if e.get("cat") == "cuda_runtime" and ("Launch" in e.get("name", "")):
                launches[e["name"]] = launches.get(e["name"], 0) + 1

    print(f"window     : {span/1000:.2f} ms wall, {len(gpu)} GPU events")
    print(f"busy       : {busy/1000:.2f} ms ({100*busy/span:.1f} %)   gaps: {(span-busy)/1000:.2f} ms in {len(gaps)} gaps"
          f" (median {sorted(gaps)[len(gaps)//2] if gaps else 0:.1f} us)")
    print(f"kernel dur : p50 {pct(.5):.1f} us  p90 {pct(.9):.1f}  p99 {pct(.99):.1f}  max {durs[-1]:.1f};"
          f" <3us: {sum(d < 3 for d in durs)}  3-10: {sum(3 <= d < 10 for d in durs)}  >=10: {sum(d >= 10 for d in durs)}")
    print(f"collectives: {coll_n} events, {coll_t/1000:.2f} ms ({100*coll_t/max(busy,1e-9):.1f} % of busy)")
    if launches:
        print("runtime    : " + ", ".join(f"{k} x{v}" for k, v in sorted(launches.items())))
    print(f"\ntop {args.top} by total time (count, total ms, mean us):")
    for n, (c, d) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"  {c:6d}  {d/1000:8.3f}  {d/c:7.1f}  {n[:110]}")
    print(f"\ntop {args.top} by count:")
    for n, (c, d) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {c:6d}  {d/1000:8.3f}  {d/c:7.1f}  {n[:110]}")


if __name__ == "__main__":
    main()
