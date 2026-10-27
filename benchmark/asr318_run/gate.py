"""Correctness gate for one PR #318 ASR model: PASS/FAIL per check, exit 1 on any FAIL.
Adapted from mstar-correctness templates/gate.py. Every threshold below was fixed before
the run; never move one after seeing a result (a regression is a finding).

Inputs (all written by run.py / oracle.py on the box):
  --pytest-log   real-checkpoint GPU pytest (test/asr/...) for this model
  --report       oracle.py tf report.json: served transcripts teacher-forced through the fp32
                 reference (transformers Whisper / qwen-asr SDK), TF32 off
  --facts        run.py facts.json: WER per run, per-request content checks of every timed
                 mstar cell, stress, long-form, language detection, word timestamps, realtime

Checks (the run gated is --tag, default mstar_a; yardsticks: --yardsticks, default
vllm,sgl_omni,ref_bf16 = the other engines and the reference's own bf16 greedy output;
control: --control mstar_b = same code, second server process):

  pytest             summary line, nothing failed/errored, nothing skipped
  parity.harness     ref_fp32 (the reference's own fp32 greedy) agrees >= 0.999 with the fp32
                     reference: proves the teacher-forcing harness itself
  parity.agreement   mean argmax agreement >= 0.95 and >= (worst yardstick) - 0.02
  parity.confident   confident disagreements (oracle top-2 margin > 1, #302) summed over the
                     200 utterances <= the most any yardstick or the control shows
  parity.near-ties   max first-disagreement gap <= max(1.0, worst yardstick gap); max rank
                     <= max(4, worst yardstick rank) (vLLM's top-5 rule)
  parity.eos         share of transcripts whose next oracle token is end-of-text >=
                     (worst yardstick share) - 0.01: catches truncated / run-on output
  parity.complete    all 200 transcripts present, none empty
  wer.reference      WER(tag) <= WER(ref_bf16) + tol, tol = max(0.15 pt, 1.5 x |WER(ref_bf16)
                     - WER(ref_fp32)|); and WER(tag) <= 5% (test-clean sanity)
  wer.under_load     WER of the mstar c=64 cell <= WER of its c=1 cell + tol
  content            every request of every timed mstar cell: HTTP 200, non-empty, not
                     truncated (< 50% of the reference's words), utterance WER <= 50%
  stress.load        c=128 (400 req) and c=65 (130 req): 0 errors, 0 bad transcripts
  stress.bad_kwargs  max_output_tokens=abc -> HTTP 4xx within 60 s, server healthy after
  stress.long_prompt (Whisper) a 400-word prompt -> no 5xx/hang, server healthy after
  stress.temp1       language detection at temperature 1: 10/10 English
  long_form          sequential (and Whisper parallel): HTTP 200, WER <= 10%; Whisper
                     segments monotonic and ending <= duration + 1 s
  detect             detected language == the reference's on >= 90% of clips, and on the zh clip
  words              (Whisper) served word starts vs the reference's (HF DTW word timestamps):
                     median |d start| <= 0.1 s and >= 90% within 0.5 s
  realtime           (qwen3_asr_realtime) every session completes, none empty, WER <= 5%
Informational (printed, never gating): the PR's own claims (WER within 0.1 pt of the
reference, >90% of word starts within 0.1 s), the 8/16 x 20-minute Qwen3-ASR files.

  python gate.py --model turbo --family whisper --pytest-log P --report R --facts F
"""
import argparse
import json
import re
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text()) if path and Path(path).exists() else None
    except (OSError, ValueError):
        return None


def pytest_check(path):
    text = Path(path).read_text() if path and Path(path).exists() else ""
    summary = (re.findall(r"^=*\s*(\d+ (?:passed|failed|error|skipped|deselected)[^=\n]*? in [\d.]+s)",
                          text, re.M) or [""])[-1]
    ok = bool(summary) and "passed" in summary and not re.search(r"\b(failed|error|errors|skipped)\b", summary)
    return ok, summary or f"no pytest summary in {path}"


def stats(rows: dict) -> dict:
    v = list(rows.values())
    gaps = [x["logit_gap_at_first"] for x in v if x.get("logit_gap_at_first") is not None]
    ranks = [x["rank_at_first"] for x in v if x.get("rank_at_first") is not None]
    return {"n": len(v), "mean": sum(x["argmax_agreement"] for x in v) / max(1, len(v)),
            "max_gap": max(gaps, default=0.0), "max_rank": max(ranks, default=0),
            "conf_dis": sum(x.get("confident_disagreements") or 0 for x in v),
            "eos": sum(bool(x.get("eos_ok")) for x in v) / max(1, len(v)),
            "empty": sum(bool(x.get("empty")) for x in v),
            "full": sum(x.get("first_disagreement") is None and not x.get("empty") for x in v)}


def parity_checks(a, report):
    out = []
    if not report or "served" not in report:
        return [("parity", False, f"no oracle report at {a.report}")]
    served = report["served"]
    n_expected = len(report.get("uids") or []) or 200
    if "ref_fp32" in served:
        s = stats(served["ref_fp32"])
        out.append(("parity.harness", s["mean"] >= 0.999,
                    f"ref_fp32 agreement {s['mean']:.4f} (need >= 0.999; proves the teacher-forcing harness)"))
    else:
        out.append(("parity.harness", False, "no ref_fp32 run in the report"))
    if a.tag not in served:
        return out + [("parity", False, f"no '{a.tag}' run in the report (have {sorted(served)})")]
    me = stats(served[a.tag])
    yard = {t: stats(served[t]) for t in a.yardsticks.split(",") if t and t in served}
    missing = [t for t in a.yardsticks.split(",") if t and t not in served]
    ctrl = stats(served[a.control]) if a.control in served else None
    ys = "; ".join(f"{t} {s['mean']:.4f} conf_dis {s['conf_dis']} gap {s['max_gap']:.2f} eos {s['eos']:.3f}"
                   for t, s in yard.items())
    if ctrl:
        ys = f"control {a.control} {ctrl['mean']:.4f} conf_dis {ctrl['conf_dis']} eos {ctrl['eos']:.3f}; " + ys
    if missing:
        ys += f"; MISSING {missing}"
    floor = min([s["mean"] for s in yard.values()], default=1.0) - 0.02
    out.append(("parity.agreement", me["mean"] >= 0.95 and me["mean"] >= floor,
                f"{a.tag} {me['mean']:.4f}, {me['full']}/{me['n']} fully argmax "
                f"(need >= 0.95 and >= {floor:.4f}); {ys}"))
    conf_cap = max([s["conf_dis"] for s in yard.values()] + ([ctrl["conf_dis"]] if ctrl else []) + [0])
    out.append(("parity.confident", me["conf_dis"] <= conf_cap,
                f"{me['conf_dis']} confident disagreements (cap {conf_cap} = most of any yardstick/control)"))
    gap_cap = max([1.0] + [s["max_gap"] for s in yard.values()])
    rank_cap = max([4] + [s["max_rank"] for s in yard.values()])
    out.append(("parity.near-ties", me["max_gap"] <= gap_cap and me["max_rank"] <= rank_cap,
                f"max gap {me['max_gap']:.2f} (cap {gap_cap:.2f}), max rank {me['max_rank']} (cap {rank_cap})"))
    eos_floor = min([s["eos"] for s in yard.values()], default=1.0) - 0.01
    out.append(("parity.eos", me["eos"] >= eos_floor,
                f"{me['eos']:.3f} end at the oracle's end-of-text (need >= {eos_floor:.3f})"))
    out.append(("parity.complete", me["n"] >= n_expected and me["empty"] == 0,
                f"{me['n']}/{n_expected} transcripts, {me['empty']} empty"))
    return out


def wer_tol(w):
    return max(0.0015, 1.5 * abs(w["ref_bf16"] - w["ref_fp32"]))


def fact_checks(a, f):
    out, info = [], []
    if not f:
        return [("facts", False, f"no facts at {a.facts}")], info
    w = f.get("wer") or {}
    if all(k in w and w[k] is not None for k in (a.tag, "ref_bf16", "ref_fp32")):
        tol = wer_tol(w)
        out.append(("wer.reference", w[a.tag] <= w["ref_bf16"] + tol and w[a.tag] <= 0.05,
                    f"{a.tag} {100 * w[a.tag]:.2f}% vs reference bf16 {100 * w['ref_bf16']:.2f}% "
                    f"(fp32 {100 * w['ref_fp32']:.2f}%), tol {100 * tol:.2f} pt; others: "
                    + ", ".join(f"{k} {100 * v:.2f}%" for k, v in w.items() if v is not None and k not in (a.tag,))))
        info.append(("PR claim: WER within 0.1 pt of the reference", abs(w[a.tag] - w["ref_bf16"]) <= 0.001,
                     f"delta {100 * (w[a.tag] - w['ref_bf16']):+.2f} pt"))
        wc = f.get("wer_cells") or {}
        if wc.get("c1") is not None and wc.get("c64") is not None:
            out.append(("wer.under_load", wc["c64"] <= wc["c1"] + tol,
                        f"mstar c=64 {100 * wc['c64']:.2f}% vs c=1 {100 * wc['c1']:.2f}% (tol {100 * tol:.2f} pt)"))
        else:
            out.append(("wer.under_load", False, f"missing mstar c1/c64 cells: {wc}"))
    else:
        out.append(("wer.reference", False, f"missing WER inputs: {w}"))
    c = f.get("content")
    out.append(("content", bool(c) and c.get("cells", 0) > 0 and c.get("bad", 1) == 0 and c.get("errors", 1) == 0,
                f"{c}" if c else "no timed mstar cells"))
    s = f.get("stress") or {}
    ld = [s.get("c128"), s.get("c65")]
    out.append(("stress.load", all(x and x.get("n", 0) > 0 and x.get("bad") == 0 and x.get("errors") == 0 for x in ld),
                f"c128 {s.get('c128')}; c65 {s.get('c65')}"))
    bk = s.get("bad_kwargs") or {}
    out.append(("stress.bad_kwargs", 400 <= (bk.get("status") or 0) < 500 and bk.get("seconds", 1e9) <= 60
                and bk.get("healthy_after") is True, f"{bk or 'missing'}"))
    if a.family == "whisper":
        lp = s.get("long_prompt") or {}
        out.append(("stress.long_prompt", (lp.get("status") or 599) < 500 and lp.get("healthy_after") is True,
                    f"{lp or 'missing'}"))
    t1 = s.get("temp1") or {}
    out.append(("stress.temp1", t1.get("n", 0) >= 10 and t1.get("en") == t1.get("n"), f"{t1 or 'missing'}"))
    for k in ("qwen_20min_8", "qwen_20min_16"):
        if k in s:
            info.append((f"stress {k}", bool(s[k].get("completed") == s[k].get("n")), f"{s[k]}"))
    lf = f.get("long_form") or {}
    modes = ["mstar_seq"] + (["mstar_par"] if a.family == "whisper" else [])
    for m in modes:
        x = lf.get(m) or {}
        ok = x.get("status") == 200 and x.get("wer") is not None and x["wer"] <= 0.10
        if a.family == "whisper" and m == "mstar_seq":
            ok = ok and x.get("segments_ok") is True
        out.append((f"long_form.{m}", ok, f"{x or 'missing'}; reference {lf.get('ref')}"))
    d = f.get("detect") or {}
    out.append(("detect", d.get("n", 0) > 0 and d.get("agree", 0) >= 0.9 * d["n"] and d.get("zh_ok") is True,
                f"{d.get('agree')}/{d.get('n')} agree with the reference, zh_ok={d.get('zh_ok')}"))
    if a.family == "whisper":
        wd = f.get("words") or {}
        ok = (wd.get("matched", 0) > 0 and wd.get("median_abs_start", 9) <= 0.1 and wd.get("within_0.5", 0) >= 0.9)
        out.append(("words", ok, f"{wd or 'missing'}"))
        if wd.get("matched"):
            info.append(("PR claim: >90% of word starts within 0.1 s", wd.get("within_0.1", 0) > 0.9,
                         f"{100 * wd.get('within_0.1', 0):.1f}%"))
    if a.model == "qasr_rt":
        rt = f.get("realtime") or {}
        out.append(("realtime", rt.get("sessions", 0) > 0 and rt.get("errors") == 0 and rt.get("empty") == 0
                    and rt.get("wer") is not None and rt["wer"] <= 0.05, f"{rt or 'missing'}"))
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=("whisper", "qwen"))
    ap.add_argument("--pytest-log")
    ap.add_argument("--report")
    ap.add_argument("--facts")
    ap.add_argument("--tag", default="mstar_a")
    ap.add_argument("--control", default="mstar_b")
    ap.add_argument("--yardsticks", default="vllm,sgl_omni,ref_bf16")
    ap.add_argument("--skip", default="", help="comma list of check prefixes not applicable to this gate run")
    a = ap.parse_args()
    checks = []
    if a.pytest_log is not None:
        ok, detail = pytest_check(a.pytest_log)
        checks.append(("pytest (real checkpoint)", ok, detail))
    checks += parity_checks(a, load(a.report))
    fc, info = fact_checks(a, load(a.facts))
    checks += fc
    skips = [s for s in a.skip.split(",") if s]
    print(f"== gate {a.model} ({a.tag})")
    passed = True
    for name, ok, detail in checks:
        if any(name.startswith(s) for s in skips):
            print(f"SKIP  {name}: {detail}")
            continue
        passed &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    for name, ok, detail in info:
        print(f"INFO  {name}: {'holds' if ok else 'does not hold'} ({detail})")
    print(f"GATE {a.model}", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
