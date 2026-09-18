#!/usr/bin/env python3
"""Turn a results directory produced by ``bench_all.sh`` into the protocol table.

Reads, per system and concurrency, the benchmark runner's report
(``<run>/runner.log``: TTFT (audio) = time to first audio, RTF, audio seconds
per second), the offline chatterbox-vllm summary (``summary.json``) and the
WER guard (``wer.json``), and prints a markdown table plus a JSON dump::

    python benchmark/chatterbox/aggregate_results.py --results results/2026-09-18 [--out table.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_NAME = re.compile(r"^(mstar|tts_server|chatterbox_vllm)_?(.*)_c(\d+)$")
SYSTEMS = {"mstar": "M*", "tts_server": "Chatterbox-TTS-Server", "chatterbox_vllm": "chatterbox-vllm"}


def parse_run_name(name: str) -> tuple[str, str, str, int] | None:
    """``mstar_chatterbox_ctx25_c8`` -> (system, variant, options, concurrency).

    The tag after the system is the config name (``chatterbox``,
    ``chatterbox_turbo``, ``original``, ``turbo``) plus the RUN_TAG of a
    deployment variant (``ctx25``, ``chunk50``, ``compile_ctx25`` ...)."""
    m = RUN_NAME.match(name)
    if not m:
        return None
    system, tag, conc = SYSTEMS[m.group(1)], m.group(2), int(m.group(3))
    parts = [p for p in tag.split("_") if p]
    variant = "turbo" if "turbo" in parts else "chatterbox"
    options = "_".join(p for p in parts if p not in ("chatterbox", "turbo", "original"))
    return system, variant, options, conc


def _num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_runner_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    ttft = re.search(r"TTFT \(audio\)\s*:\s*(.*)", text)
    rtf = re.search(r"RTF\s*:\s*(.*)", text)
    req = re.search(r"Requests\s*:\s*(\d+)/(\d+) succeeded", text)
    return {
        "ttfa_p50_s": _num(r"p50=([0-9.]+)", ttft.group(1)) if ttft else None,
        "ttfa_p95_s": _num(r"p95=([0-9.]+)", ttft.group(1)) if ttft else None,
        "rtf_p50": _num(r"p50=([0-9.]+)", rtf.group(1)) if rtf else None,
        "rtf_p95": _num(r"p95=([0-9.]+)", rtf.group(1)) if rtf else None,
        "audio_s_per_s": _num(r"Throughput: ([0-9.]+) audio sec/s", text),
        "req_per_s": _num(r"Throughput: ([0-9.]+) req/s", text),
        "succeeded": int(req.group(1)) if req else None,
        "requested": int(req.group(2)) if req else None,
        "wall_s": _num(r"Total wall time: ([0-9.]+)s", text),
    }


def parse_vllm_summary(path: Path) -> dict:
    s = json.loads(path.read_text())
    return {
        "ttfa_p50_s": s.get("ttfa_p50_s"), "ttfa_p95_s": s.get("ttfa_p95_s"),
        "rtf_p50": s.get("rtf"), "rtf_p95": None,
        "audio_s_per_s": s.get("audio_seconds_per_second"),
        "t3_tokens_per_s": s.get("t3_tokens_per_second"),
        "succeeded": s.get("num_sentences"), "requested": s.get("num_sentences"),
        "sampling": s.get("sampling"),
    }


def collect(results: Path) -> list[dict]:
    rows = []
    for run in sorted(p for p in results.iterdir() if p.is_dir()):
        parsed = parse_run_name(run.name)
        if parsed is None:
            continue
        system, variant, options, conc = parsed
        if (run / "runner.log").exists():
            row = parse_runner_log(run / "runner.log")
        elif (run / "summary.json").exists():
            row = parse_vllm_summary(run / "summary.json")
        else:
            continue
        wer_path = run / "wer.json"
        if wer_path.exists():
            wer = json.loads(wer_path.read_text())
            row["wer"] = wer["wer"]
            row["mean_audio_s"] = wer.get("mean_audio_s")
        else:
            row["wer"] = None
            row["mean_audio_s"] = None
        sha_path = run / "sha.txt"
        row["sha"] = sha_path.read_text().split()[0][:8] if sha_path.exists() and sha_path.read_text().strip() else None
        rows.append({"system": system, "variant": variant, "options": options, "concurrency": conc,
                     "run": run.name, **row})
    return rows


def fmt(v, digits=2, suffix=""):
    return "n/a" if v is None else f"{v:.{digits}f}{suffix}"


def table(rows: list[dict], variant: str, env: dict) -> str:
    lines = [
        f"**{variant}** (H100 80GB, {env.get('gpu_clocks', 'clocks: see env.txt')}; 200 shared sentences, "
        "preset voice Abigail.wav, warmup excluded)",
        "",
        "| System (version) | c | TTFA p50 / p95 (s) | RTF p50 / p95 | audio s / s | WER | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted((r for r in rows if r["variant"] == variant),
                    key=lambda r: (r["system"] != "M*", r["system"], r["options"], r["concurrency"])):
        version = env.get(r["system"], "")
        if r["system"] == "M*" and r.get("sha"):
            version = f"(`{r['sha']}`)" + version.split(")", 1)[1] if ")" in version else f"(`{r['sha']}`)"
        label = r["system"] + (f" [{r['options']}]" if r["options"] else "")
        notes = []
        if r.get("succeeded") is not None and r.get("requested") and r["succeeded"] != r["requested"]:
            notes.append(f"{r['succeeded']}/{r['requested']} ok")
        if r.get("mean_audio_s"):
            notes.append(f"{r['mean_audio_s']:.1f} s audio/utt")
        if r["system"] == "chatterbox-vllm":
            notes.append("offline batch API, TTFA = batch wall time, RTF = batch compute/audio")
            if r.get("t3_tokens_per_s"):
                notes.append(f"T3 {r['t3_tokens_per_s']:.0f} tok/s")
        lines.append(
            f"| {label} {version} | {r['concurrency']} | {fmt(r['ttfa_p50_s'])} / {fmt(r['ttfa_p95_s'])} "
            f"| {fmt(r['rtf_p50'], 3)} / {fmt(r['rtf_p95'], 3)} | {fmt(r['audio_s_per_s'], 1)} "
            f"| {fmt(r['wer'] * 100 if r['wer'] is not None else None, 1, '%')} | {'; '.join(notes)} |"
        )
    return "\n".join(lines)


def read_env(results: Path) -> dict:
    env: dict = {}
    path = results / "env.txt"
    if not path.exists():
        return env
    text = path.read_text()
    sha = re.search(r"^([0-9a-f]{40})$", text, re.M)
    if sha:
        env["M*"] = f"(`{sha.group(1)[:8]}`)"
    m = re.search(r"torch ([\d.+a-z]+) flashinfer ([\d.a-z]+)", text)
    if m:
        env["M*"] += f" torch {m.group(1)}, flashinfer {m.group(2)}"
    m = re.search(r"tts-server torch ([\d.+a-z]+) transformers ([\d.]+)", text)
    if m:
        env["Chatterbox-TTS-Server"] = f"(torch {m.group(1)}, transformers {m.group(2)})"
    m = re.search(r"chatterbox-vllm torch ([\d.+a-z]+) vllm ([\d.]+)", text)
    if m:
        env["chatterbox-vllm"] = f"(vllm {m.group(2)}, torch {m.group(1)})"
    m = re.search(r"NVIDIA H100[^\n]*", text)
    if m:
        env["gpu_clocks"] = m.group(0)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None, help="markdown file (default: <results>/TABLE.md)")
    parser.add_argument(
        "--curated", action="store_true",
        help="only the rows the PR table needs: baselines, the shipped defaults ('final') and the "
             "offline decoder ('fix_chunk0' / 'chunk0' without lost requests)",
    )
    args = parser.parse_args()
    results = Path(args.results)
    rows = collect(results)
    env = read_env(results)
    if args.curated:
        # the PR rows: shipped defaults and the offline decoder per node, the earlier default,
        # and every baseline run (n04 = first allocation, n08 = second; the nodes differ)
        labels = {
            "final": "default (n04)", "final_n08": "default (n08)",
            "fix_chunk0": "offline decoder, stream_chunk_tokens 0 (n08)",
            "": "earlier default: fixed 25-token chunks (n04)",
        }
        rows = [r for r in rows if r["system"] != "M*" or r["options"] in labels]
        for r in rows:
            if r["system"] == "M*":
                r["options"] = labels[r["options"]]
            elif r["system"] == "Chatterbox-TTS-Server":
                r["options"] = "n08" if r["options"] == "n08" else "n04"
            elif r["system"] == "chatterbox-vllm":
                r["options"] = ("reference sampling, " if r["options"] == "refsampling" else "its defaults, ") + "n04"
    md = "\n\n".join(table(rows, v, env) for v in ("chatterbox", "turbo") if any(r["variant"] == v for r in rows))
    out = Path(args.out) if args.out else results / ("TABLE_curated.md" if args.curated else "TABLE.md")
    out.write_text(md + "\n")
    (results / "TABLE.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
