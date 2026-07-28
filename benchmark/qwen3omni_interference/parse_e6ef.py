"""Parse E6EF rep dirs + decisions into a rung-level CSV and frontier table.

Rung rows re-derive pooled metrics from the rep results.json files (the
decisions.jsonl verdicts are cross-checked but the CSV is source-derived).
"""

import csv
import glob
import json
import math
import os
import re

EXPF = "/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E/frontier"
TTFA_SLO, ITL_SLO, GROWTH_MAX, DRAIN_MAX, WINDOW_S = 2000.0, 1500.0, 1.5, 45.0, 120.0
BUCKET_P50 = {"libri_len15": 20.4, "libri_len60": 65.6,
              "libri_len120": 126.7, "libri_len240": 245.8}


def pct(v, p):
    s = sorted(v)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def pool(rep_dirs):
    ttfa, gaps, ratios, drains, a2t = [], [], [], [], []
    clean = True
    for d in rep_dirs:
        r = json.load(open(f"{d}/outputs/results.json"))
        clean &= (r["completed"] == r["num_requests"] and r["failed"] == 0)
        t2s = sorted((q for q in r["per_request"] if q["type"] == "text_to_speech"),
                     key=lambda q: int(q["request_id"]))
        ttfa += [q["ttft_ms"]["audio"] for q in t2s if "audio" in q.get("ttft_ms", {})]
        gaps += [g for q in t2s for g in q.get("itl_ms", {}).get("audio", [])]
        a2t += [q["ttft_ms"]["text"] for q in r["per_request"]
                if q["type"] == "audio_to_text" and "text" in q.get("ttft_ms", {})]
        third = len(t2s) // 3
        if third >= 3:
            def med(v):
                return sorted(v)[len(v) // 2]
            f = [q["ttft_ms"]["audio"] for q in t2s[:third] if "audio" in q["ttft_ms"]]
            l = [q["ttft_ms"]["audio"] for q in t2s[-third:] if "audio" in q["ttft_ms"]]
            if f and l and med(f) > 0:
                ratios.append(med(l) / med(f))
        drains.append(max(0.0, r["wall_time_s"] - WINDOW_S))
    growth = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else 1.0
    return {
        "clean": clean, "n_reps": len(rep_dirs),
        "ttfa_p50": pct(ttfa, 50), "ttfa_p99": pct(ttfa, 99),
        "itl_p50": pct(gaps, 50), "itl_p99": pct(gaps, 99), "itl_max": max(gaps) if gaps else None,
        "a2t_ttft_p99": pct(a2t, 99),
        "growth": growth, "drain_max": max(drains) if drains else None,
    }


def main() -> None:
    rows = []
    frontiers = {}
    censored = set()
    for dec in glob.glob(os.path.join(EXPF, "*", "decisions.jsonl")):
        for line in open(dec):
            j = json.loads(line)
            if j.get("censored"):
                censored.add((j["arm"], j["bucket"], j["rate"]))

    for arm_dir in sorted(glob.glob(os.path.join(EXPF, "*"))):
        arm = os.path.basename(arm_dir)
        if not os.path.isdir(arm_dir) or arm.startswith("_"):
            continue
        for bdir in sorted(glob.glob(os.path.join(arm_dir, "libri_len*"))):
            bucket = os.path.basename(bdir)
            by_rate = {}
            for rd in glob.glob(os.path.join(bdir, "r*_rep*")):
                if "_failed" in rd:
                    continue
                m = re.match(r"r([\d.]+)_rep(\d+)", os.path.basename(rd))
                if m and os.path.isfile(os.path.join(rd, "outputs", "results.json")):
                    by_rate.setdefault(float(m.group(1)), []).append(rd)
            for rate, reps in sorted(by_rate.items()):
                p = pool(sorted(reps))
                stationary = p["growth"] <= GROWTH_MAX and (p["drain_max"] or 0) <= DRAIN_MAX
                # Pre-registered protocol: a rung passes only with ALL 3 reps
                # clean. Rungs that bug-censored after 1-2 clean reps stay in
                # the CSV (n_reps column) as partial evidence but cannot pass.
                full = p["n_reps"] >= 3
                ttfa_pass = full and p["clean"] and stationary and p["ttfa_p99"] is not None and p["ttfa_p99"] <= TTFA_SLO
                itl_pass = full and p["clean"] and stationary and p["itl_max"] is not None and p["itl_max"] <= ITL_SLO
                rows.append({
                    "arm": arm, "bucket": bucket, "audio_p50_s": BUCKET_P50.get(bucket, ""),
                    "rate_req_s": rate, "n_reps": p["n_reps"], "clean": int(p["clean"]),
                    "ttfa_p50_ms": f"{p['ttfa_p50']:.0f}", "ttfa_p99_ms": f"{p['ttfa_p99']:.0f}",
                    "itl_p50_ms": f"{p['itl_p50']:.1f}", "itl_p99_ms": f"{p['itl_p99']:.0f}",
                    "itl_max_ms": f"{p['itl_max']:.0f}", "a2t_ttft_p99_ms": f"{p['a2t_ttft_p99']:.0f}",
                    "growth": f"{p['growth']:.2f}", "drain_max_s": f"{p['drain_max']:.0f}",
                    "stationary": int(stationary), "ttfa_pass": int(ttfa_pass), "itl_pass": int(itl_pass),
                })
                key = (arm, bucket)
                fr = frontiers.setdefault(key, {"ttfa": None, "itl": None, "max_eval": 0.0, "cens_rates": []})
                fr["max_eval"] = max(fr["max_eval"], rate)
                if ttfa_pass:
                    fr["ttfa"] = max(fr["ttfa"] or 0.0, rate)
                if itl_pass:
                    fr["itl"] = max(fr["itl"] or 0.0, rate)
            for (a, b, r) in censored:
                if a == arm and b == bucket:
                    frontiers.setdefault((arm, bucket), {"ttfa": None, "itl": None, "max_eval": 0.0,
                                                         "cens_rates": []})["cens_rates"].append(r)

    out = os.path.join(EXPF, "results_e6e_frontier.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rung rows)")

    print("\nFRONTIERS  (λ* = highest passing rung; > = grid-censored; ^ = bug-censored lower bound)")
    print(f"{'arm':8} {'bucket':14} {'λ*_TTFA':>8} {'λ*_ITL':>8}")
    for (arm, bucket), fr in sorted(frontiers.items()):
        def fmt(v):
            if v is None:
                return "<min*" if fr["cens_rates"] else "<min"
            s = f"{v:.1f}"
            if v >= 2.4:
                s = ">" + s
            elif any(c > v for c in fr["cens_rates"]):
                s = "^" + s
            return s
        print(f"{arm:8} {bucket:14} {fmt(fr['ttfa']):>8} {fmt(fr['itl']):>8}")
    print("(* = every evaluated rung bug-censored before 3 clean reps)")


if __name__ == "__main__":
    main()
