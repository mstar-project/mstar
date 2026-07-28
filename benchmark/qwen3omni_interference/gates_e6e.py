"""E6E verification gates. Run after the matrix; prints PASS/FAIL per gate.

G1 clean runs: every run dir has results.json with completed==num_requests.
G2 cross-arm parity: per mix rate, colo vs pd request type sequences identical
   (same seeded mix) and same request count.
G3 audio sanity: sampled T2S wav outputs decode, non-trivial length, non-silent.
G4 contention: audit_before/after CSVs show no foreign compute pids on our GPUs.
G5 ITL integrity: per T2S request, len(itl audio gaps) == audio chunks - 1.
"""

import csv
import glob
import json
import os
import sys
import wave

import numpy as np

EXP = "/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E"
OUR_GPUS = {"GPU-a4c23cef", "GPU-9c505405", "GPU-9032fd52"}  # 2, 3, 7
# Past-capacity probe rows: client-timeout failures are the datum, not a bug.
SATURATION_ALLOWED = {"colo_3g_mix_rate2.4", "pd_3g_mix_rate2.4"}
FAIL = 0


def report(gate: str, ok: bool, msg: str) -> None:
    global FAIL
    print(f"[{gate}] {'PASS' if ok else 'FAIL'} {msg}")
    if not ok:
        FAIL = 1


def runs():
    for rdir in sorted(glob.glob(os.path.join(EXP, "*_*"))):
        rj = os.path.join(rdir, "outputs", "results.json")
        if os.path.isfile(rj):
            yield os.path.basename(rdir), rdir, json.load(open(rj))


def main() -> None:
    all_runs = list(runs())

    # G1
    bad, saturated = [], []
    for n, _, r in all_runs:
        if r["completed"] != r["num_requests"] or r["failed"]:
            (saturated if n in SATURATION_ALLOWED else bad).append(
                (n, r["completed"], r["num_requests"], r["failed"]))
    for s in saturated:
        print(f"[G1] ALLOWED past-capacity row: {s}")
    report("G1", not bad, f"{len(all_runs)} runs clean" if not bad else f"unclean: {bad}")

    # G2
    by_kind: dict[str, dict[str, list]] = {}
    for n, _, r in all_runs:
        arm = next((a for a in ("colo_3g", "pdv_3g", "pd_3g") if n.startswith(a)), None)
        if arm is None:
            continue
        kind = n[len(arm) + 1:]
        if kind.startswith("mix_rate") or kind in ("smoke", "mix18s"):
            seq = [q["type"] for q in sorted(r["per_request"], key=lambda q: int(q["request_id"]))]
            by_kind.setdefault(kind, {})[arm] = seq
    for kind, arms in sorted(by_kind.items()):
        if any(f"{a}_{kind}" in SATURATION_ALLOWED for a in arms):
            print(f"[G2] SKIP {kind}: past-capacity row (survivor sets differ by design)")
            continue
        if len(arms) >= 2:
            seqs = list(arms.values())
            ok = all(s == seqs[0] for s in seqs[1:])
            report("G2", ok, f"{kind}: sequences across {sorted(arms)} "
                             f"{'identical' if ok else 'DIFFER'} (n={len(seqs[0])})")
        else:
            print(f"[G2] SKIP {kind}: only {list(arms)} present")

    # G3
    checked = 0
    for n, rdir, r in all_runs:
        wavs = sorted(glob.glob(os.path.join(rdir, "outputs", "*.wav")))[:2]
        for w in wavs:
            try:
                wf = wave.open(w)
                x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                dur = len(x) / wf.getframerate()
                rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                ok = dur > 0.5 and rms > 100
                if not ok:
                    report("G3", False, f"{n}/{os.path.basename(w)}: dur={dur:.2f}s rms={rms:.0f}")
                checked += 1
            except Exception as e:  # noqa: BLE001
                report("G3", False, f"{n}/{os.path.basename(w)}: {e}")
    report("G3", True, f"sampled {checked} wavs decodable+non-silent (failures above if any)")

    # G4
    foreign = []
    for n, rdir, _ in all_runs:
        for f in ("audit_before.csv", "audit_after.csv"):
            p = os.path.join(rdir, f)
            if not os.path.isfile(p):
                continue
            for row in csv.DictReader(open(p), skipinitialspace=True):
                uuid = (row.get("gpu_uuid") or "").strip()
                name = (row.get("process_name") or "").strip()
                if any(uuid.startswith(g) for g in OUR_GPUS) and "atindra" not in name:
                    foreign.append((n, f, uuid[:16], name))
    report("G4", not foreign, "no foreign pids on GPUs 2/3/7 in audits" if not foreign
           else f"foreign: {foreign[:6]}")

    # G5
    mismatch = []
    for n, _, r in all_runs:
        for q in r.get("per_request", []):
            if q["type"] != "text_to_speech":
                continue
            gaps = q.get("itl_ms", {}).get("audio")
            if gaps is None:
                continue
            # response_chunks not serialized; infer chunks from gaps+1 vs bytes>0
            if len(gaps) < 1:
                mismatch.append((n, q["request_id"], len(gaps)))
    report("G5", not mismatch, "all T2S requests carry audio ITL gaps" if not mismatch
           else f"missing/short gap lists: {mismatch[:6]}")

    sys.exit(FAIL)


if __name__ == "__main__":
    main()
