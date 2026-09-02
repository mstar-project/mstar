#!/usr/bin/env python3
"""Analysis for the resource_pools_2 reproduction cells.

Reads every cell directory under a cells root, pairs `base` and `head` runs of the
same (config, run) and prints the table the findings are stated in.

  python3 stress-test/itl_ratio.py [cells_root]   default: ./stress-run/cells

Metrics
  chat cells   ITL = per-request median inter-chunk gap, then the cell median
               over requests. Also mean gap, median gap, sub-1 ms gap share,
               p99 gap (all pooled over every gap in the cell) and throughput.
  image cells  L = e2e_latency, cell median. The image endpoint is not streamed,
               so ITL does not apply.

dmon columns (`nvidia-smi dmon -s u -o DT`) are
  0 date, 1 time, 2 gpu, 3 sm, 4 mem, 5 enc, 6 dec.
Column 3 is SM-busy; column 4 is memory-controller busy, which is easy to
misread as SM.
"""
import json
import os
import re
import statistics as st
import sys
from datetime import datetime, timezone

DEFAULT_ROOT = os.path.join(os.getcwd(), "stress-run", "cells")
SUB_MS = 0.001


def pct(h, b):
    return float("nan") if not b else (h / b - 1) * 100


def q(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def meta(d):
    """(tag, cfg) from the boot record, e.g. run=head_bagel_single_gpu_2026...Z"""
    try:
        txt = open(os.path.join(d, "boot.txt")).read()
    except OSError:
        return None, None
    m = re.search(r"run=(base|head)_(\S+?)_\d{8}T", txt)
    return (m.group(1), m.group(2)) if m else (None, None)


def _dmon_rows(d):
    try:
        return [l.split() for l in open(os.path.join(d, "dmon.txt")) if not l.startswith("#")]
    except OSError:
        return []


def dmon(d, s, e, col):
    """(mean, share-of-samples-reading-zero) for one dmon column over [s, e]."""
    v = []
    for r in _dmon_rows(d):
        if len(r) <= col or not r[col].isdigit():
            continue
        try:
            t = datetime.strptime(r[0] + r[1], "%Y%m%d%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if s <= t <= e:
            v.append(float(r[col]))
    if not v:
        return None, None
    return round(st.mean(v), 1), round(100.0 * sum(1 for x in v if x == 0) / len(v), 1)


def longest_zero_run(d, s, e, col=3):
    """Longest consecutive stretch of 0 in one dmon column, in samples (= seconds
    at the default 1 Hz). The c=32 stall shows up here as a multi-hundred-second
    run; a healthy cell reads 0 only at the span edges."""
    v = []
    for r in _dmon_rows(d):
        if len(r) <= col or not r[col].isdigit():
            continue
        try:
            t = datetime.strptime(r[0] + r[1], "%Y%m%d%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if s <= t <= e:
            v.append(int(r[col]))
    best = cur = 0
    for x in v:
        cur = cur + 1 if x == 0 else 0
        best = max(best, cur)
    return best


def load(root):
    cells = {}
    if not os.path.isdir(root):
        sys.exit(f"no cells root at {root}")
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "boot.txt")):
            continue
        tag, cfg = meta(d)
        if not tag:
            continue
        win = {}
        wf = os.path.join(d, "spans.txt")
        if os.path.exists(wf):
            for l in open(wf, errors="ignore"):
                p = l.split()
                if len(p) >= 3:
                    win[p[0]] = (datetime.fromisoformat(p[1].replace("Z", "+00:00")),
                                 datetime.fromisoformat(p[2].replace("Z", "+00:00")))
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.startswith("warmup_"):
                continue
            run = f[:-5]
            j = json.load(open(os.path.join(d, f)))
            ok = [r for r in j["per_request"]
                  if r["error"] is None and r["e2e_latency"] is not None]
            img = "text_to_image" in " ".join(j["argv"])
            vals = ([r["e2e_latency"] for r in ok] if img
                    else [v for r in ok for k, v in r["itl_median"].items() if k == "text"])
            gaps = [] if img else [g for r in ok for k, gs in r["itl_gaps"].items()
                                   if k == "text" for g in gs]
            ttft = [t for r in ok for t in r["ttft"].values()]
            sm = mem = smz = zrun = None
            if run in win:
                sm, _ = dmon(d, *win[run], col=3)
                mem, _ = dmon(d, *win[run], col=4)
                _, smz = dmon(d, *win[run], col=3)
                zrun = longest_zero_run(d, *win[run], col=3)
            cells[(cfg, tag, run)] = dict(
                cell=name, kind="L" if img else "ITL",
                n=len(ok), nfail=j["n_failed"], nreq=j["n_requests"],
                p50=q(vals, .50), p90=q(vals, .90), p99=q(vals, .99),
                gap_mean=st.mean(gaps) if gaps else None,
                gap_med=st.median(gaps) if gaps else None,
                gap_p99=q(gaps, .99), n_gaps=len(gaps),
                sub_ms=(100.0 * sum(1 for g in gaps if g < SUB_MS) / len(gaps)) if gaps else None,
                ttft50=q(ttft, .50), tput=j["agg"]["request_throughput"],
                wall=j["wall_time_s"], sm=sm, mem=mem, smzero=smz, zero_run=zrun)
    return cells


def fmt(x, w=9, p=4):
    return f"{x:{w}.{p}f}" if isinstance(x, float) else f"{str(x):>{w}}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    C = load(root)
    if not C:
        sys.exit(f"no cells found under {root}")

    print(f"cells root: {root}\n")
    hdr = (f"{'config':24}{'run':10}{'m':4}{'base p50':>10}{'head p50':>10}{'d%':>8}"
           f"{'b p99':>10}{'h p99':>10}{'b tput':>9}{'h tput':>9}{'d%':>8}"
           f"{'bSM':>6}{'hSM':>6}{'b0run':>7}{'h0run':>7}{'ok b/h':>12}")
    print(hdr)
    print("-" * len(hdr))
    for cfg, run in sorted({(c, r) for c, t, r in C}):
        b, h = C.get((cfg, "base", run)), C.get((cfg, "head", run))
        if not b or not h:
            print(f"{cfg:24}{run:10}INCOMPLETE  base={bool(b)} head={bool(h)}")
            continue
        print(f"{cfg:24}{run:10}{b['kind']:4}{fmt(b['p50'],10)}{fmt(h['p50'],10)}"
              f"{pct(h['p50'], b['p50']):8.2f}{fmt(b['p99'],10)}{fmt(h['p99'],10)}"
              f"{b['tput']:9.2f}{h['tput']:9.2f}{pct(h['tput'], b['tput']):8.2f}"
              f"{str(b['sm']):>6}{str(h['sm']):>6}{str(b['zero_run']):>7}{str(h['zero_run']):>7}"
              f"{b['n']:>5}/{b['nreq']:<3}{h['n']:>3}/{h['nreq']:<3}")

    print("\n=== chat gap distribution (the bursty-delivery finding) ===")
    hdr2 = (f"{'config':24}{'run':10}{'arm':6}{'mean gap':>11}{'med gap':>11}"
            f"{'p99 gap':>11}{'sub-1ms %':>11}{'n gaps':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for cfg, run in sorted({(c, r) for c, t, r in C}):
        for tag in ("base", "head"):
            c = C.get((cfg, tag, run))
            if not c or c["kind"] != "ITL" or not c["n_gaps"]:
                continue
            print(f"{cfg:24}{run:10}{tag:6}{c['gap_mean']*1000:10.3f}m{c['gap_med']*1000:10.3f}m"
                  f"{c['gap_p99']*1000:10.3f}m{c['sub_ms']:11.1f}{c['n_gaps']:9}")

    # A span in spans.txt brackets the client process, not the load, so it
    # always carries a few idle seconds of client start-up and exit. A real
    # stall is hundreds of seconds (the c=32 stall idles for ~285 s), so the
    # flag threshold sits well clear of that edge effect.
    STALL_SECONDS = 15
    print(f"\n=== failures and stalls (0%-SM stretch > {STALL_SECONDS}s) ===")
    any_flag = False
    for k in sorted(C):
        c = C[k]
        flag = []
        if c["nfail"]:
            flag.append(f"{c['nfail']} FAILED of {c['nreq']}")
        if c["zero_run"] and c["zero_run"] > STALL_SECONDS:
            flag.append(f"0%-SM run {c['zero_run']}s")
        if flag:
            any_flag = True
            print(f"  {k[0]} {k[1]} {k[2]:10} ({c['cell']}): " + ", ".join(flag))
    if not any_flag:
        print("  none - every request completed and no run held the GPU at "
              f"0% SM for more than {STALL_SECONDS}s")


if __name__ == "__main__":
    main()
