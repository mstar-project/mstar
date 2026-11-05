"""Per-step kernel table from a kernel trace.

    python -m benchmark.kernels.trace.ktrace TRACE_DIR [--anchor REGEX] [--top N] [--json OUT]

Reads every torch.profiler chrome trace (``*.json``, its "kernel" events) and
rocprofv3 ``*kernel_trace.csv`` under TRACE_DIR, takes the process with the
most kernels (the GPU worker) and cuts its kernels into steps at each match of
``--anchor``, a kernel that runs once per forward. For the most common step
kinds (same kernel count) it reports the wall time per step (anchor to
anchor), the GPU busy time per step (the union of kernel intervals), and time
per kernel category, per launch kind (replayed graph, eager, copy) and per
kernel name. A tracer leaves kernel durations close to real but inflates the
gaps between kernels, so read the busy/wall split as a bound on idle time.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import re
import statistics
import sys

CATEGORIES = [
    ("gemm", r"Cijk_|triton_tem_|wvSplitK|wv_splitk|LLGemm|LLMM|gemm|Gemm|_mm_|matmul"),
    ("gdn_conv", r"causal_conv1d"),
    ("gdn_rule", r"gdr_decode|gated_delta|recurrent|chunk_gated|fused_sigmoid_gating|delta_rule"),
    ("sampler", (r"TopK|TopP|Sampling|sampling|_split_partials|_split_combine|_split_write|argmax|softmax|"
                 r"gumbel|temperature|compare_scalar|FillFunctor|CUDAFunctorOnSelf_add|scatter_fill|direct_copy")),
    ("add_norm", r"RMSNorm|rmsnorm|rms_norm|add_mean_mul_pow_rsqrt_0|layernorm"),
    ("gated_norm", r"rsqrt_silu|norm_gated|gated_rms|RMSNormGated|layer_norm_fwd"),
    ("rope_qknorm", r"neg_pow_rsqrt|rotary|rope|qk_norm|mean_pow_split"),
    ("kv_write", r"kv_scatter|reshape_and_cache|cache_kernel"),
    ("attention", r"unified_attention|reduce_segments|paged_attention|ll4mi|attn_fwd|fmha|flash_attn|mha_"),
    ("act", r"silu|gelu|act_and_mul|SiluAndMul"),
    ("embed", r"embedding|index_select|indexSelect"),
    ("allreduce", r"allreduce|all_reduce|cross_device|rccl|nccl"),
]
_CATEGORY_RE = [(name, re.compile(rx)) for name, rx in CATEGORIES]
# (start ns, end ns, kernel name, launch kind)
Row = tuple[int, int, str, str]


def category(name: str) -> str:
    for cat, rx in _CATEGORY_RE:
        if rx.search(name):
            return cat
    return "other"


def launch_kind(api: str | None) -> str:
    """How a kernel was launched, from the runtime call sharing its correlation id."""
    if api in ("hipGraphLaunch", "cudaGraphLaunch"):
        return "graph"
    if api and "Memcpy" in api:
        return "copy"
    return "eager" if api else "?"


def load_chrome(path: str) -> list[Row]:
    with open(path) as f:
        events = json.load(f).get("traceEvents", [])
    api = {e["args"]["correlation"]: e["name"] for e in events
           if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    rows = []
    for e in events:
        if e.get("cat") != "kernel" or "dur" not in e:
            continue
        t0 = int(float(e["ts"]) * 1e3)
        kind = launch_kind(api.get(e.get("args", {}).get("correlation")))
        rows.append((t0, t0 + int(float(e["dur"]) * 1e3), e.get("name", ""), kind))
    rows.sort()
    return rows


def load_csv(path: str) -> list[Row]:
    """rocprofv3's kernel trace; it does not say how a kernel was launched."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Kernel_Name") or r.get("KernelName") or ""
            try:
                t0 = int(r.get("Start_Timestamp") or r.get("BeginNs"))
                t1 = int(r.get("End_Timestamp") or r.get("EndNs"))
            except (TypeError, ValueError):
                continue
            rows.append((t0, t1, name, "?"))
    rows.sort()
    return rows


def load(path: str) -> list[Row]:
    return load_chrome(path) if path.endswith(".json") else load_csv(path)


def union(intervals) -> int:
    """Total length covered by ``intervals``."""
    total, cur0, cur1 = 0, None, None
    for a, b in sorted(intervals):
        if cur1 is None or a > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        total += cur1 - cur0
    return total


def short(name: str, n: int = 70) -> str:
    name = re.sub(r"\(.*", "", name)
    return name if len(name) <= n else name[: n - 3] + "..."


def trace_files(trace_dir: str) -> list[str]:
    files = glob.glob(os.path.join(trace_dir, "**", "*kernel_trace.csv"), recursive=True)
    files += [f for f in glob.glob(os.path.join(trace_dir, "**", "*.json"), recursive=True)
              if not f.endswith("ktrace.json")]
    return files


def cut_steps(rows: list[Row], anchor: str) -> list[dict]:
    """The kernels between consecutive matches of ``anchor``, one dict per step."""
    rx = re.compile(anchor)
    starts = [i for i, (_, _, name, _) in enumerate(rows) if rx.search(name)]
    steps = []
    for i, j in itertools.pairwise(starts):
        kernels = rows[i:j]
        steps.append({"n": len(kernels), "wall": rows[j][0] - rows[i][0],
                      "busy": union([(a, b) for a, b, _, _ in kernels]), "kernels": kernels})
    return steps


def kind_table(steps: list[dict], top: int) -> dict:
    """Medians over steps of one kind (one kernel count), in microseconds: per
    category, per launch kind, per (category, launch kind) and per kernel name."""
    med = statistics.median
    first = steps[0]["kernels"]
    cat_us, cat_launch_us, name_us, launch_us = (collections.defaultdict(list) for _ in range(4))
    for step in steps:
        per_cat, per_launch = collections.Counter(), collections.Counter()
        per_cat_launch, per_name = collections.Counter(), collections.Counter()
        for a, b, name, kind in step["kernels"]:
            per_cat[category(name)] += b - a
            per_launch[kind] += b - a
            per_cat_launch[(category(name), kind)] += b - a
            per_name[short(name)] += b - a
        for cat in [c for c, _ in CATEGORIES] + ["other"]:
            cat_us[cat].append(per_cat.get(cat, 0) / 1e3)
        for kind in {k for _, _, _, k in first}:
            launch_us[kind].append(per_launch.get(kind, 0) / 1e3)
        for key in {(category(n), k) for _, _, n, k in first}:
            cat_launch_us[key].append(per_cat_launch.get(key, 0) / 1e3)
        for name, ns in per_name.items():
            name_us[name].append(ns / 1e3)
    cat_count = collections.Counter(category(n) for _, _, n, _ in first)
    launch_count = collections.Counter(k for _, _, _, k in first)
    cat_launch_count = collections.Counter((category(n), k) for _, _, n, k in first)
    name_count = collections.Counter(short(n) for _, _, n, _ in first)
    return {
        "steps": len(steps), "kernels": len(first),
        "wall_us": med(s["wall"] / 1e3 for s in steps), "busy_us": med(s["busy"] / 1e3 for s in steps),
        "walls_us": [s["wall"] / 1e3 for s in steps],
        "categories": sorted(({"category": c, "us": med(v), "kernels": cat_count[c]} for c, v in cat_us.items()
                              if med(v) > 0), key=lambda r: -r["us"]),
        "launches": {k: {"kernels": launch_count[k], "us": med(v)} for k, v in sorted(launch_us.items())},
        "category_launch": sorted(({"category": c, "launch": k, "us": med(v), "kernels": cat_launch_count[(c, k)]}
                                   for (c, k), v in cat_launch_us.items()), key=lambda r: -r["us"]),
        "names": sorted(({"kernel": n, "calls": name_count[n], "us": med(v)} for n, v in name_us.items()),
                        key=lambda r: -r["us"])[:top],
    }


def _kind_text(n_kernels: int, table: dict) -> list[str]:
    w, b = table["wall_us"], table["busy_us"]
    out = ["", f"== step kind: {n_kernels} kernels, {table['steps']} steps ==",
           f"wall/step {w:9.1f} us   GPU busy/step {b:9.1f} us   kernels-sum/step "
           f"{sum(r['us'] for r in table['categories']):9.1f} us   idle {1 - b / w:.1%}"]
    walls = table["walls_us"]
    if len(walls) >= 6:
        q = sorted(walls)
        out += [f"wall per step, first 16: {' '.join(f'{x:.0f}' for x in walls[:16])}",
                f"wall even/odd steps: {statistics.median(walls[0::2]):.1f} / {statistics.median(walls[1::2]):.1f}"
                f" us; p10/p50/p90 {q[len(q) // 10]:.1f}/{statistics.median(q):.1f}/{q[len(q) * 9 // 10]:.1f}"]
    out.append(f"{'category':12s} {'us/step':>9s} {'share':>6s} {'kernels':>7s}")
    out += [f"{r['category']:12s} {r['us']:9.1f} {r['us'] / b:6.1%} {r['kernels']:7d}" for r in table["categories"]]
    out.append("by launch: " + "  ".join(f"{k}: {v['kernels']} kernels {v['us']:.1f} us"
                                         for k, v in table["launches"].items()))
    out.append(f"{'category/launch':22s} {'us/step':>9s} {'kernels':>7s}")
    out += [f"{r['category'] + '/' + r['launch']:22s} {r['us']:9.1f} {r['kernels']:7d}"
            for r in table["category_launch"]]
    out += ["", f"{'kernel':70s} {'calls':>5s} {'us/step':>9s} {'us/call':>8s}"]
    out += [f"{r['kernel']:70s} {r['calls']:5d} {r['us']:9.1f} {r['us'] / max(r['calls'], 1):8.2f}"
            for r in table["names"]]
    return out


def report(trace_dir: str, anchor: str = "embedding", top: int = 30) -> tuple[str, dict]:
    """The text report and its JSON form, for the busiest process under ``trace_dir``."""
    files = trace_files(trace_dir)
    if not files:
        raise SystemExit(f"no kernel trace under {trace_dir}")
    per_file = sorted(((len(rows), f, rows) for f in files for rows in [load(f)]), reverse=True)
    out = [f"# {n:8d} kernels  {os.path.relpath(f, trace_dir)}" for n, f, _ in per_file]
    _, path, rows = per_file[0]
    if not rows:
        raise SystemExit("empty trace")
    wall = rows[-1][1] - rows[0][0]
    busy = union([(a, b) for a, b, _, _ in rows])
    out += ["", f"process {os.path.relpath(path, trace_dir)}: {len(rows)} kernels over {wall / 1e6:.1f} ms, "
                f"GPU busy {busy / wall:.1%}",
            "", "most frequent kernels (pick an anchor that runs once per forward):"]
    out += [f"  {c:7d}  {name}" for name, c in collections.Counter(short(n) for _, _, n, _ in rows).most_common(12)]
    steps = cut_steps(rows, anchor)
    if len(steps) < 2:
        raise SystemExit(f"anchor {anchor!r} cut {len(steps)} steps; pass --anchor")
    kinds = collections.Counter(s["n"] for s in steps)
    out += ["", f"{len(steps)} steps; kernel counts per step (count: steps): "
            + ", ".join(f"{n}: {c}" for n, c in kinds.most_common(8))]
    result = {"trace": path, "wall_ms": wall / 1e6, "busy_frac": busy / wall, "kinds": {}}
    for n_kernels, n_steps in kinds.most_common(3):
        if n_steps < 3:
            continue
        table = kind_table([s for s in steps if s["n"] == n_kernels], top)
        result["kinds"][str(n_kernels)] = table
        out += _kind_text(n_kernels, table)
    return "\n".join(out), result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_dir")
    parser.add_argument("--anchor", default=r"embedding")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    text, result = report(args.trace_dir, args.anchor, args.top)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
