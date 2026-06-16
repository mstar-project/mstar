#!/usr/bin/env python3
"""Roll native-sweep results.json files into a SUMMARY.md table.

Matches the column layout of results/ming_t2t_sweep/SUMMARY.md (the
vllm-omni run) so the native numbers drop straight into the comparison:

  mode | concurrency | reqs | wall (s) | E2E p50 (ms) | E2E p95 (ms) | req/s | tok/s

tok/s is recomputed by tokenizing the saved req_*.txt outputs with Ming's
own tokenizer (same basis as the vllm-omni SUMMARY), summed over completed
requests and divided by wall time. If the tokenizer can't be loaded, the
column falls back to a whitespace-word count and is flagged.
"""
import glob
import json
import os
import sys

POINTS = [
    ("offline_b1", "OFFLINE", 1),
    ("closed_loop_c2", "CLOSED_LOOP", 2),
    ("closed_loop_c4", "CLOSED_LOOP", 4),
    ("closed_loop_c8", "CLOSED_LOOP", 8),
    ("closed_loop_c16", "CLOSED_LOOP", 16),
    ("closed_loop_c32", "CLOSED_LOOP", 32),
]


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "inclusionAI/Ming-flash-omni-2.0", trust_remote_code=True
        )
        return tok, True
    except Exception as e:  # noqa: BLE001
        print(f"[summarize] tokenizer load failed ({e!r}); "
              f"falling back to word-count tok/s", file=sys.stderr)
        return None, False


def _count_tokens(text, tok):
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())


def main(out_dir):
    tok, exact = _load_tokenizer()
    rows = []
    for name, mode, conc in POINTS:
        d = os.path.join(out_dir, name)
        rj = os.path.join(d, "results.json")
        if not os.path.exists(rj):
            print(f"[summarize] skip {name}: no results.json", file=sys.stderr)
            continue
        with open(rj) as f:
            r = json.load(f)

        total_tokens = 0
        for tf in glob.glob(os.path.join(d, "req_*.txt")):
            with open(tf) as f:
                total_tokens += _count_tokens(f.read(), tok)
        wall = r.get("wall_time_s", 0.0) or 0.0
        toks = (total_tokens / wall) if wall else 0.0

        rows.append({
            "mode": mode,
            "conc": conc,
            "reqs": r.get("completed", 0),
            "wall": wall,
            "p50": r.get("jct_median_ms", 0.0),
            "p95": r.get("jct_p95_ms", 0.0),
            "reqs_per_s": r.get("request_throughput", 0.0),
            "toks_per_s": toks,
            "failed": r.get("failed", 0),
        })

    lines = []
    lines.append("# Ming-flash-omni-2.0 T2T scaling sweep — NATIVE M* (mstar), 4×H100 80GB")
    lines.append("")
    lines.append("Native Walk-graph path (mstar/model/ming_omni_flash/), thinker-only TP=4,")
    lines.append("GPUs 4-7, `configs/ming_flash_omni_thinker_only_tp4.yaml`. Benchmarked with")
    lines.append("`--inference-system ours` (native /generate). Same grid + prompts as the")
    lines.append("vllm-omni sweep in `../ming_t2t_sweep/SUMMARY.md` for a 1:1 comparison.")
    lines.append("")
    if not exact:
        lines.append("> NOTE: tok/s below is a WHITESPACE WORD COUNT (tokenizer unavailable),")
        lines.append("> not directly comparable to the vllm-omni SUMMARY's tokenizer-based tok/s.")
        lines.append("")
    lines.append("| mode | concurrency | reqs | wall (s) | E2E p50 (ms) | E2E p95 (ms) | req/s | tok/s |")
    lines.append("|------|-------------|------|----------|--------------|--------------|-------|-------|")
    for r in rows:
        lines.append(
            f"| {r['mode']:<11} | {r['conc']:>11} | {r['reqs']:>4} | "
            f"{r['wall']:>8.2f} | {r['p50']:>12.0f} | {r['p95']:>12.0f} | "
            f"{r['reqs_per_s']:>5.2f} | {r['toks_per_s']:>6.1f} |"
        )
    total_failed = sum(r["failed"] for r in rows)
    lines.append("")
    lines.append(f"Total failed requests across sweep: {total_failed}")
    lines.append("")

    summary_path = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[summarize] wrote {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
