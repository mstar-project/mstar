"""Summarize an IKET ``trace.json`` from ``iket_ctapipe.py``.

For every kernel launch (the producer and the consumer are distinct kernels
and run in different CUDA contexts = GPUs) prints, per range name, the count
and the µs distribution, then the per-CTA time budget: how much of a CTA's
lifetime the DMA warp spent spinning in ``wait_row`` (consumer) or the store
warp spent in ``signal`` (producer). Timestamps are ns from each GPU's
globaltimer; do not compare across launches on different GPUs.

    python -m benchmark.cta_pipelining.iket_analyze OUT/*.trace.json [more traces...] [--all] [--by-tile]

Also prints the CTA start / end skew and the end time by tiles per CTA.
``--by-tile`` adds, per range name, the duration by tile index within a warp
(the i-th ``epi_tile`` of a warp is its CTA's i-th tile) for the first 6 and
last 3 indices.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def _pct(vals, q):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def _fmt_stats(vals_us):
    if not vals_us:
        return "n=0"
    return (
        f"n={len(vals_us):6d} sum={sum(vals_us) / 1e3:9.3f} ms mean={statistics.fmean(vals_us):8.2f} "
        f"p50={_pct(vals_us, 0.5):8.2f} p90={_pct(vals_us, 0.9):8.2f} max={max(vals_us):8.2f} us"
    )


def print_by_tile(ranges, names, locs, first=6, last=3):
    pw = defaultdict(list)  # (cta, warp) -> [(start, end, name)]
    for r in ranges:
        loc = locs[r["warpLocIdxs"][0]]
        pw[(tuple(loc["ctaId"]), loc["warpId"])].append((r["startTs"], r["endTs"], names[r["rangeNameIdx"]]))
    by_idx = defaultdict(lambda: defaultdict(list))  # name -> tile idx -> [us]
    for rs in pw.values():
        rs.sort()
        idx = defaultdict(int)
        for s, e, n in rs:
            by_idx[n][idx[n]].append((e - s) / 1e3)
            idx[n] += 1
    for name in ("wait_row", "mma_tile", "epi_tile", "signal"):
        if name not in by_idx:
            continue
        n_idx = max(by_idx[name]) + 1
        shown = sorted(set(range(min(first, n_idx))) | set(range(max(0, n_idx - last), n_idx)))
        print(f"    by tile index: {name}")
        for i in shown:
            v = by_idx[name][i]
            print(f"      #{i:3d} n={len(v):5d} mean={statistics.fmean(v):8.2f} p50={_pct(v, 0.5):8.2f} "
                  f"max={max(v):8.2f} us")


def summarize_launch(trace, launch, by_tile=False):
    names = trace["stringTable"]
    locs = trace["locationTable"]
    ranges = launch.get("ranges", [])
    lifetimes = launch.get("warpLifetimes", [])
    kname = launch["kernelName"]
    short = "producer" if "producer" in kname.lower() else "consumer" if "consumer" in kname.lower() else kname[:60]
    grid = launch["gridDimX"] * launch["gridDimY"] * launch["gridDimZ"]
    print(f"\n=== launch gridId={launch.get('gridId')} ctx={launch.get('contextId')} grid={grid} "
          f"block={launch['blockDimX']} kernel={short}")
    print(f"    {kname[:110]}")
    if not ranges and not lifetimes:
        print("    (no ranges / lifetimes recorded)")
        return

    t0 = min([r["startTs"] for r in ranges] + [w["startTs"] for w in lifetimes])
    t1 = max([r["endTs"] for r in ranges] + [w["endTs"] for w in lifetimes])
    print(f"    span (first event -> last event): {(t1 - t0) / 1e3:.1f} us")

    by_name = defaultdict(list)
    per_warp = defaultdict(lambda: defaultdict(float))  # (cta, warp) -> name -> ns
    mma_count = defaultdict(int)  # (cta, warp) -> mma_tile ranges = tiles done by that CTA
    first_wait_end = None
    for r in ranges:
        name = names[r["rangeNameIdx"]]
        dur = r["endTs"] - r["startTs"]
        by_name[name].append(dur / 1e3)
        loc = locs[r["warpLocIdxs"][0]]
        per_warp[(tuple(loc["ctaId"]), loc["warpId"])][name] += dur
        if name == "mma_tile":
            mma_count[(tuple(loc["ctaId"]), loc["warpId"])] += 1
        if name == "wait_row":
            first_wait_end = r["endTs"] if first_wait_end is None else min(first_wait_end, r["endTs"])
    for name in sorted(by_name):
        print(f"    {name:10s} {_fmt_stats(by_name[name])}")

    # Per-CTA lifetime from warpLifetimes if present, else from range extent.
    cta_life = defaultdict(lambda: [None, None])
    src = lifetimes if lifetimes else ranges
    for w in src:
        loc = locs[w["locIdx"]] if "locIdx" in w else locs[w["warpLocIdxs"][0]]
        cta = tuple(loc["ctaId"])
        s, e = w["startTs"], w["endTs"]
        cur = cta_life[cta]
        cur[0] = s if cur[0] is None else min(cur[0], s)
        cur[1] = e if cur[1] is None else max(cur[1], e)
    lives_us = [(e - s) / 1e3 for s, e in cta_life.values() if s is not None]
    if lives_us:
        print(f"    CTA lifetime    {_fmt_stats(lives_us)}")
        starts = [s for s, _ in cta_life.values() if s is not None]
        ends = [e for s, e in cta_life.values() if s is not None]
        print(f"    CTA start skew {(max(starts) - min(starts)) / 1e3:.1f} us, "
              f"end skew (max end - min end) {(max(ends) - min(ends)) / 1e3:.1f} us")
        tiles = {}  # cta -> tiles (max over its MMA warps)
        for (cta, _warp), c in mma_count.items():
            tiles[cta] = max(tiles.get(cta, 0), c)
        by_tiles = defaultdict(list)
        for cta, c in tiles.items():
            by_tiles[c].append((cta_life[cta][1] - t0) / 1e3)
        for c in sorted(by_tiles):
            e = by_tiles[c]
            print(f"    CTAs with {c:3d} tiles: {len(e):4d}, end at +{min(e):.1f} .. +{max(e):.1f} us "
                  f"(mean +{statistics.fmean(e):.1f})")
    # Share of the CTA's lifetime a *single warp* spends inside each range (ranges are
    # recorded per warp, so summing over the 8 MMA warps would over-count 8x).
    for name in ("wait_row", "tma_tile", "mma_tile", "epi_tile", "signal"):
        fr = []
        for (cta, _warp), sums in per_warp.items():
            s, e = cta_life[cta]
            if s is not None and e > s and name in sums:
                fr.append(100.0 * sums[name] / (e - s))
        if fr:
            print(f"    {name:10s} share of CTA lifetime per warp: mean {statistics.fmean(fr):5.1f}%  "
                  f"p50 {_pct(fr, 0.5):5.1f}%  max {max(fr):5.1f}%   ({len(fr)} warps)")
    if first_wait_end is not None:
        fill_us = (first_wait_end - t0) / 1e3
        print(f"    first row block ready (first wait_row end) at +{fill_us:.1f} us after kernel start")
    n_mma = len(by_name.get("mma_tile", []))
    if n_mma and grid:
        print(f"    mma_tile ranges per CTA (all MMA warps): {n_mma / grid:.1f}")
    if by_tile:
        print_by_tile(ranges, names, locs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="one or more run-iket trace.json files")
    ap.add_argument("--all", action="store_true", help="summarize every launch, not only the last per kernel")
    ap.add_argument("--by-tile", action="store_true", help="per tile-index breakdown (first 6 / last 3 tiles)")
    args = ap.parse_args(argv)
    for path in args.traces:
        trace = json.load(open(path))
        launches = trace["launches"]
        print(f"\n##### {path}")
        print(f"{len(launches)} launches, range names: {trace['stringTable']}")
        if args.all:
            chosen = launches
        else:  # last launch of each distinct kernel (the warm-up ones are skipped)
            last = {}
            for i, l in enumerate(launches):
                last[l["kernelName"]] = i
            chosen = [launches[i] for i in sorted(last.values())]
        for l in chosen:
            summarize_launch(trace, l, args.by_tile)


if __name__ == "__main__":
    main()
