"""Parse E6E run dirs into results_e6e_pd_interference.csv.

One row per run dir (arm x kind). T2S columns come from the audio modality
(TTFA = first audio chunk, ITL = client-side inter-chunk gaps, pooled across
requests); A2T columns from the text modality of audio_to_text requests.
"""

import csv
import glob
import json
import os
import sys

EXP = "/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E"

ARM_DESC = {
    "colo_3g": "M* colocated (qwen3omni.yaml: thinker prefill+decode share rank1)",
    "pd_3g": "M* PD-disaggregated (qwen3omni_pd_disaggregated.yaml: prefill rank0, decode rank1)",
    "pdv_3g": "M* PD variant (pd yaml + Code2Wav moved to rank2 beside Talker)",
}


def pct(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (k - lo)) + v[hi] * (k - lo)


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def main() -> None:
    rows = []
    for rdir in sorted(glob.glob(os.path.join(EXP, "*_*"))):
        rj = os.path.join(rdir, "outputs", "results.json")
        if not os.path.isfile(rj):
            continue
        run = os.path.basename(rdir)
        arm = next((a for a in ("colo_3g", "pdv_3g", "pd_3g") if run.startswith(a)), None)
        if arm is None:
            continue
        kind = run[len(arm) + 1:]
        r = json.load(open(rj))

        per_req = r.get("per_request", [])
        t2s = [q for q in per_req if q["type"] == "text_to_speech"]
        a2t = [q for q in per_req if q["type"] == "audio_to_text"]

        t2s_ttfa = [q["ttft_ms"]["audio"] for q in t2s if "audio" in q.get("ttft_ms", {})]
        t2s_itl = [g for q in t2s for g in q.get("itl_ms", {}).get("audio", [])]
        t2s_jct = [q["jct_ms"] for q in t2s]
        a2t_ttft = [q["ttft_ms"]["text"] for q in a2t if "text" in q.get("ttft_ms", {})]
        a2t_jct = [q["jct_ms"] for q in a2t]

        pt = r.get("per_type", {})
        failed_t2s = pt.get("text_to_speech", {}).get("failed", 0)
        failed_a2t = pt.get("audio_to_text", {}).get("failed", 0)

        rate = r.get("arrival", {}).get("rate") if r.get("arrival") else None
        rows.append({
            "run": run,
            "arm": arm,
            "deployment": ARM_DESC[arm],
            "kind": kind,
            "rate_req_s": rate if rate is not None else "",
            "num_requests": r["num_requests"],
            "completed": r["completed"],
            "failed": r["failed"],
            "wall_s": f"{r['wall_time_s']:.1f}",
            "t2s_n": len(t2s),
            "t2s_failed": failed_t2s,
            "t2s_ttfa_p50_ms": fmt(pct(t2s_ttfa, 50)),
            "t2s_ttfa_p99_ms": fmt(pct(t2s_ttfa, 99)),
            "t2s_itl_p50_ms": fmt(pct(t2s_itl, 50)),
            "t2s_itl_p99_ms": fmt(pct(t2s_itl, 99)),
            "t2s_itl_max_ms": fmt(max(t2s_itl) if t2s_itl else None),
            "t2s_itl_n_gaps": len(t2s_itl),
            "t2s_itl_frac_gt500ms": fmt(sum(g > 500 for g in t2s_itl) / len(t2s_itl), 4) if t2s_itl else "",
            "t2s_jct_p50_ms": fmt(pct(t2s_jct, 50)),
            "t2s_jct_p99_ms": fmt(pct(t2s_jct, 99)),
            "a2t_n": len(a2t),
            "a2t_failed": failed_a2t,
            "a2t_ttft_p50_ms": fmt(pct(a2t_ttft, 50)),
            "a2t_ttft_p99_ms": fmt(pct(a2t_ttft, 99)),
            "a2t_jct_p50_ms": fmt(pct(a2t_jct, 50)),
            "a2t_jct_p99_ms": fmt(pct(a2t_jct, 99)),
        })

    out = os.path.join(EXP, "results_e6e_pd_interference.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for row in rows:
        print(f"  {row['run']}: t2s ttfa p99={row['t2s_ttfa_p99_ms']} "
              f"itl p99={row['t2s_itl_p99_ms']} | a2t ttft p99={row['a2t_ttft_p99_ms']}")


if __name__ == "__main__":
    main()
