"""Pool E6EF reps for one (arm, bucket, rate) rung and print the verdict.

Usage: e6ef_eval.py <rep_dir> [<rep_dir> ...]

PRE-REGISTERED SLOs (frozen before any run):
  TTFA SLO:  T2S first-audio-chunk p99 <= 2000 ms (pooled over reps)
  ITL SLO:   T2S max inter-chunk gap <= 1500 ms (pooled max over reps)
  Stationarity gate (applies to both): geomean over reps of
  (TTFA p50 last third / first third, arrival order) <= 1.5 AND max
  post-window drain (wall - 120 s) <= 45 s.
"""

import json
import math
import sys

TTFA_SLO = 2000.0
ITL_SLO = 1500.0
GROWTH_MAX = 1.5
DRAIN_MAX = 45.0
WINDOW_S = 120.0


def pct(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def main() -> None:
    ttfa_all, gaps_all, ratios, drains = [], [], [], []
    a2t_ttft = []
    for d in sys.argv[1:]:
        r = json.load(open(f"{d}/outputs/results.json"))
        t2s = sorted((q for q in r["per_request"] if q["type"] == "text_to_speech"),
                     key=lambda q: int(q["request_id"]))
        tt = [q["ttft_ms"]["audio"] for q in t2s if "audio" in q.get("ttft_ms", {})]
        ttfa_all += tt
        gaps_all += [g for q in t2s for g in q.get("itl_ms", {}).get("audio", [])]
        a2t_ttft += [q["ttft_ms"]["text"] for q in r["per_request"]
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

    ttfa_p99 = pct(ttfa_all, 99)
    itl_max = max(gaps_all)
    growth = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else 1.0
    drain_max = max(drains)
    stationary = growth <= GROWTH_MAX and drain_max <= DRAIN_MAX
    print(f"VERDICT ttfa_p99={ttfa_p99:.0f} itl_max={itl_max:.0f} "
          f"ttfa_pass={int(ttfa_p99 <= TTFA_SLO)} itl_pass={int(itl_max <= ITL_SLO)} "
          f"stationary={int(stationary)} growth={growth:.2f} drain_max={drain_max:.0f} "
          f"itl_p99={pct(gaps_all, 99):.0f} a2t_ttft_p99={pct(a2t_ttft, 99):.0f} "
          f"n_t2s={len(ttfa_all)} n_gaps={len(gaps_all)}")


if __name__ == "__main__":
    main()
