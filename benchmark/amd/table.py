"""table.md and table.json from a sweep run directory.

    python -m benchmark.amd.table RUN_DIR

Cells show the median over repeats, and the primary metric also its [min-max]. Flags are rerun
(spread over the limit once), noisy (still over it after the re-run), error (failed requests or
runs), client-bound (the benchmark client was busy most of the run) and HEALTH-FAIL (thermal
throttling or new ECC errors during the arm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.amd import catalog, tables

# (summary key, header, scale, decimals) per family; the workload's primary metric also gets [min-max].
COLUMNS = {
    "text": [("tok_s", "out tok/s", 1, 0), ("ttft_p50_s", "TTFT p50 ms", 1e3, 1), ("ttft_p95_s", "TTFT p95 ms", 1e3, 1),
             ("itl_p50_s", "ITL p50 ms", 1e3, 2), ("tpot_p50_s", "TPOT p50 ms", 1e3, 2),
             ("e2e_p50_s", "E2E p50 s", 1, 2), ("out_tokens_per_req", "out tok/req", 1, 1),
             ("in_tokens_per_req", "in tok/req", 1, 0)],
    "tts": [("audio_s_per_s", "audio s/s", 1, 1), ("ttfa_p50_s", "TTFA p50 ms", 1e3, 1),
            ("ttfa_p95_s", "TTFA p95 ms", 1e3, 1), ("rtf_p50", "RTF p50", 1, 3),
            ("audio_s_per_req", "audio s/req", 1, 2), ("req_s", "req/s", 1, 2)],
    "asr": [("rtfx", "RTFx", 1, 1), ("req_s", "req/s", 1, 2), ("e2e_p50_s", "E2E p50 s", 1, 3),
            ("e2e_p95_s", "E2E p95 s", 1, 3)],
    "omni": [("audio_s_per_s", "audio s/s", 1, 1), ("ttfa_p50_s", "TTFA p50 ms", 1e3, 0),
             ("ttfa_p95_s", "TTFA p95 ms", 1e3, 0), ("rtf_p50", "RTF p50", 1, 3), ("e2e_p50_s", "E2E p50 s", 1, 2),
             ("e2e_p95_s", "E2E p95 s", 1, 2), ("text_ttft_p50_s", "text TTFT p50 ms", 1e3, 0),
             ("audio_s_per_req", "audio s/req", 1, 2)],
    "image": [("images_per_s", "images/s", 1, 3), ("e2e_p50_s", "latency p50 s", 1, 3),
              ("e2e_p95_s", "latency p95 s", 1, 3)],
    "video": [("clips_per_min", "clips/min", 1, 2), ("e2e_p50_s", "E2E p50 s", 1, 1), ("e2e_p95_s", "E2E p95 s", 1, 1),
              ("wall_s", "makespan s", 1, 1)],
}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _fmt(v, scale=1.0, dec=1) -> str:
    if v is None:
        return "-"
    return f"{v * scale:.{dec}f}"


def collect(run_dir: Path) -> dict:
    meta = _load(run_dir / "sweep.json") or {}
    corr = _load(run_dir / "correctness.json") or {}
    data = {"meta": meta, "health": {}, "arms": {}, "cells": [], "correctness": corr}
    for name in ("check_before", "check_after"):
        h = _load(run_dir / "health" / f"{name}.json")
        if h:
            data["health"][name] = h
    order = {a: i for i, a in enumerate(meta.get("arms", []))}
    rows = {r: i for i, r in enumerate(meta.get("rows", []))}
    arm_jsons = sorted(run_dir.glob("*/*/arm.json"),
                       key=lambda p: (rows.get(p.parent.parent.name, 99), order.get(p.parent.name, 99)))
    for arm_json in arm_jsons:
        info = _load(arm_json) or {}
        d = arm_json.parent
        versions = _load(d / "server" / "versions.json") or {}
        arm = catalog.ARMS.get(info.get("arm"))
        key = f"{info.get('row')}/{info.get('arm')}"
        data["arms"][key] = {
            "row": info.get("row"), "arm": info.get("arm"), "status": info.get("status"), "gpus": info.get("gpus"),
            "error": (info.get("error") or "").strip().splitlines()[-1:] if info.get("error") else None,
            "startup": info.get("startup"), "health": (info.get("health") or {}).get("verdict"),
            "correctness_pass": {k: v for k, v in (info.get("correctness_pass") or {}).items() if k != "error"},
            "correctness_pass_error": bool((info.get("correctness_pass") or {}).get("error")),
            "image": versions.get("image"), "stack": versions.get("stack"), "env": versions.get("env"),
            "command": versions.get("command"), "git": versions.get("git"),
            "recipe": arm.recipe if arm else versions.get("recipe"),
            "differs": arm.differs if arm else versions.get("differs"),
            "host": (versions.get("host") or {}).get("host"),
        }
        for cell_json in sorted(d.glob("cells/*/c*/cell.json")):
            cell = _load(cell_json)
            if cell:
                cell = {k: v for k, v in cell.items() if k != "attempts"}
                cell["dir"] = str(cell_json.parent)
                data["cells"].append(cell)
    return data


def _comparability(data: dict, row: str, arm: str, wl: str, fam: str) -> list[str]:
    """Reasons an arm's cells don't compute the same thing as mstar's, beyond the arm's own note."""
    notes = []
    a = data["arms"].get(f"{row}/{arm}", {})
    spec = catalog.ARMS.get(arm)
    differs = spec.differs_for.get(fam, spec.differs) if spec else a.get("differs")
    if differs:
        notes.append(differs)
    if fam == "text" and arm != "mstar":
        ours = [c for c in data["cells"] if c["row"] == row and c["arm"] == "mstar" and c["workload"] == wl]
        theirs = [c for c in data["cells"] if c["row"] == row and c["arm"] == arm and c["workload"] == wl]
        for key, label in (("out_tokens_per_req", "output"), ("in_tokens_per_req", "prompt")):
            o = [c["summary"].get(key, {}).get("median") for c in ours]
            t = [c["summary"].get(key, {}).get("median") for c in theirs]
            o, t = [x for x in o if x], [x for x in t if x]
            if o and t and abs(max(o) - max(t)) / max(o) > 0.02:
                notes.append(f"{label} tokens/request {max(t):.0f} vs mstar {max(o):.0f}")
    if fam in ("tts", "omni") and arm != "mstar":
        # Length-sensitive metrics (audio s/s, RTF) compare only if both say about as much.
        def audio(name):
            vals = [c["summary"].get("audio_s_per_req", {}).get("median") for c in data["cells"]
                    if c["row"] == row and c["arm"] == name and c["workload"] == wl]
            vals = [v for v in vals if v]
            return sum(vals) / len(vals) if vals else None
        o, t = audio("mstar"), audio(arm)
        if o and t and not 0.85 <= t / o <= 1.15:
            notes.append(f"{t / o - 1:+.0%} audio per request vs mstar")
    return notes


def render(data: dict) -> str:
    meta = data["meta"]
    node = meta.get("node") or {}
    names = sorted({str(v) for v in (node.get("gpu_names") or {}).values()})
    chip = ", ".join(names) or meta.get("node", {}).get("node_type", "?")
    title = f"# {meta.get('table')}: {chip}"
    lines = [title, ""]
    if meta.get("dry"):
        lines += ["**DRY RUN on a shared or partial node. Functional check only; none of these numbers are "
                  "timings.**", ""]
    lines += [f"- run `{meta.get('run_id')}`, {meta.get('started')} to {meta.get('finished')}",
              f"- node {node.get('host')}, partition {node.get('partition')}, job {meta.get('job')}, "
              f"GPUs {meta.get('gpus')}, lanes {meta.get('lanes')}, NUMA binding {meta.get('numa')}",
              f"- timed runs per cell after warmup: {meta.get('repeats')}. A cell over "
              f"{(meta.get('spread_limit') or 0):.0%} spread is re-run once. tok/s counts at most the requested "
              "output length per request."]
    if meta.get("note"):
        lines.append(f"- {meta['note']}")
    for name, h in data["health"].items():
        v = h["verdict"]
        lines.append(f"- node health {name}: {v['status']}" + (f" ({'; '.join(v['reasons'])[:300]})"
                                                               if v["reasons"] else ""))
    lines.append("")

    lines += ["## Arms", "", "| row | arm | status | engine / image | versions | env | attention (server log) | recipe "
              "| cold start: ready s, first request s |", "|---|---|---|---|---|---|---|---|---|"]
    for a in data["arms"].values():
        im = a.get("image") or {}
        src = im.get("source") or {}
        stack = a.get("stack") or {}
        if a["arm"] == "mstar" or not im:
            git = a.get("git") or {}
            what = f"mstar {git.get('branch', '')} {str(git.get('head', ''))[:9]}" if git else a["arm"]
            ver = " ".join(f"{k} {stack[k]}" for k in ("torch", "torch_hip", "amd-aiter", "amd-flashinfer", "triton",
                                                        "triton-rocm") if stack.get(k))
        else:
            what = f"{src.get('repo')}:{src.get('tag')} {str(src.get('digest', ''))[:19]}"
            ver = " ".join(f"{k} {stack[k]}" for k in ("vllm", "sglang", "vllm-omni", "torch", "torch_hip",
                                                        "amd-aiter") if stack.get(k))
        env = " ".join(f"{k}={v}" for k, v in (a.get("env") or {}).items()
                       if k.startswith(("VLLM_", "SGLANG_", "MSTAR_", "NCCL_", "AITER_", "MIOPEN_FIND"))
                       and not k.endswith("_DIR"))
        st = a.get("startup") or {}
        cold = f"{st.get('ready_s', '-')}, {st.get('first_request_s', '-')}" if st else "-"
        status = a["status"] + (f": {a['error'][0][:80]}" if a.get("error") else "")
        backends = ", ".join(st.get("backends") or []) or ("AUTO (platform default)" if a["arm"] == "mstar" else "")
        lines.append(f"| {a['row']} | {a['arm']} | {status} | {what} | {ver} | {env} | {backends} | "
                     f"{a.get('recipe') or ''} | {cold} |")
    lines.append("")

    for wl_name in meta.get("workloads", []):
        wl = tables.WORKLOADS.get(wl_name)
        if wl is None:
            continue
        cells = [c for c in data["cells"] if c["workload"] == wl_name]
        if not cells:
            continue
        cols = COLUMNS[wl.family]
        lines += [f"## {wl_name} ({wl.family})", ""]
        conc = (meta.get("concurrency") or {}).get(wl_name, list(wl.concurrency))
        lines.append(f"Prompts `{wl.prompts}`, " + (f"{wl.output_len} output tokens (ignore_eos), " if wl.output_len
                                                   else "") + f"closed loop at concurrency {conc}.")
        lines.append("")
        for row in meta.get("rows", []):
            rcells = [c for c in cells if c["row"] == row]
            if not rcells:
                continue
            r = catalog.ROWS.get(row)
            lines += [f"### {row}" + (f" ({r.gpus} GPU{'s' if r.gpus > 1 else ''})" if r else ""), ""]
            lines.append("| arm | c | " + " | ".join(h for _, h, _, _ in cols) + " | flags | correctness | "
                         "not comparable because |")
            lines.append("|---|---|" + "---|" * len(cols) + "---|---|---|")
            for c in sorted(rcells, key=lambda x: (meta.get("arms", []).index(x["arm"])
                                                   if x["arm"] in meta.get("arms", []) else 99, x["concurrency"])):
                vals = []
                for key, _, scale, dec in cols:
                    s = c["summary"].get(key)
                    if not s or (key == "in_tokens_per_req" and not s["median"]):
                        vals.append("-")
                    elif key == wl.primary[0]:
                        vals.append(f"{_fmt(s['median'], scale, dec)} [{_fmt(s['min'], scale, dec)}-"
                                    f"{_fmt(s['max'], scale, dec)}]")
                    else:
                        vals.append(_fmt(s["median"], scale, dec))
                h = (data["arms"].get(f"{row}/{c['arm']}") or {}).get("health") or {}
                # Thermal throttling or new ECC errors during the arm void its cells.
                flags = " ".join(c.get("flags", []) + (["error"] if c.get("errors") else [])
                                 + (["HEALTH-FAIL"] if h.get("status") == "FAIL" else []))
                corr = data["correctness"].get(f"{row}/{c['arm']}") or {}
                corr_txt = _correctness_text(corr)
                why = "; ".join(_comparability(data, row, c["arm"], wl_name, wl.family))
                lines.append(f"| {c['arm']} | {c['concurrency']} | " + " | ".join(vals) + f" | {flags} | {corr_txt} | "
                             f"{why} |")
            lines.append("")

    if data["correctness"]:
        lines += ["## Correctness (benchmark/correctness sample sets and references)", "",
                  "| row/arm | verdict | details |",
                  "|---|---|---|"]
        for k, v in data["correctness"].items():
            lines.append(f"| {k} | {v.get('verdict')} | {_correctness_details(v)} |")
        lines.append("")
    lines += ["## Health per arm", "", "| row/arm | verdict | reasons and notes |", "|---|---|---|"]
    for k, a in data["arms"].items():
        h = a.get("health") or {}
        txt = "; ".join((h.get("reasons") or []) + (h.get("notes") or []))[:400]
        lines.append(f"| {k} | {h.get('status', '-')} | {txt} |")
    lines.append("")
    return "\n".join(lines)


def _correctness_details(c: dict) -> str:
    parts = []
    for key, label in (("exact_vs_reference", "exact vs reference"), ("unexplained_divergences",
                       "divergences not at near ties"), ("wer_percent", "WER %"), ("reference_wer_percent",
                       "reference WER %"), ("wer_minus_reference_pt", "WER - reference (pt)")):
        if c.get(key) is not None:
            v = c[key]
            parts.append(f"{label} {v[0] if isinstance(v, list) and len(v) == 1 else v}")
    for key, label in (("duration_ratio", "duration ratio median"), ("log_mel_distance_db", "log-mel dB median")):
        if isinstance(c.get(key), dict) and c[key].get("median") is not None:
            parts.append(f"{label} {c[key]['median']:.3f}")
    if c.get("reference") == "missing":
        parts.append("no reference")
    if c.get("prompt_tokens_differ"):
        parts.append(f"prompt tokenized differently from the reference on {', '.join(c['prompt_tokens_differ'])}")
    if c.get("error"):
        parts.append(f"error: {c['error'][:120]}")
    return "; ".join(parts)


def _correctness_text(c: dict) -> str:
    if not c:
        return "-"
    parts = [c.get("verdict", "?")]
    if "exact_vs_reference" in c:
        parts.append(f"exact {c['exact_vs_reference']}")
    if c.get("unexplained_divergences") is not None:
        parts.append(f"{c['unexplained_divergences']} non-tie div")
    if c.get("wer_percent"):
        w = c["wer_percent"]
        parts.append(f"WER {w[0] if isinstance(w, list) else w}%")
    if c.get("wer_minus_reference_pt") is not None:
        parts.append(f"({c['wer_minus_reference_pt']:+} pt vs ref)")
    if c.get("reference") == "missing":
        parts.append("no reference")
    if c.get("prompt_tokens_differ"):
        parts.append(f"(prompt tokenized differently: {', '.join(c['prompt_tokens_differ'][:3])})")
    return " ".join(str(p) for p in parts)


def write(run_dir: Path) -> Path:
    data = collect(run_dir)
    (run_dir / "table.json").write_text(json.dumps(data, indent=1, default=str))
    out = run_dir / "table.md"
    out.write_text(render(data))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    print(write(Path(args.run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
