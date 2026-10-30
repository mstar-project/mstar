"""Re-count a finished table's text cells in tokens from what the run saved, for runs from before the runner did.

The runner used to count one token per streamed chunk and the scorer to read chunk i as reference token i, so an arm
that streams several tokens a chunk (speculative decoding) came out low in tok/s and failed correctness. This rebuilds
each timed run's numbers from its results.json and req_<i>.txt, scores a copy of the correctness pass, and writes
table-tokens.md. What the run saved is left alone. The chunk gaps of a multi-token stream were not saved, only split
over each chunk's tokens, so its ITL and chunk gap columns stay empty.

    python -m benchmark.amd.recount RUN_DIR [RUN_DIR ...]   # add --out DIR to write elsewhere than RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark.amd import catalog, correctness, sweep, table, tables
from benchmark.amd.launch import REPO
from benchmark.utils import clearly_multi_token

KEEP_FROM_REPEAT = ("dir", "client_wall_s", "client_cpu_s", "client_cpu_frac", "client_loop_frac")


def recount_results(results: dict, run_dir: Path, tokenizer) -> dict:
    """A run's results.json as the runner would write it now. A count that differs from the chunk count came from the
    server's usage block and stands."""
    per = [dict(r) for r in results.get("per_request") or []]
    multi = 0
    for r in per:
        chunks = (r.get("response_chunks") or {}).get("text", 0)
        out = run_dir / f"req_{r['request_id']}.txt"
        if chunks and r.get("output_text_tokens") == chunks and out.is_file():
            n = len(tokenizer.encode(out.read_text(encoding="utf-8"), add_special_tokens=False))
            r["output_text_tokens"] = n if clearly_multi_token(n, chunks) else chunks
        multi += bool(chunks) and clearly_multi_token(r.get("output_text_tokens") or 0, chunks)
    agg = dict(results.get("aggregate") or {})
    total, wall = sum(r.get("output_text_tokens") or 0 for r in per), results.get("wall_time_s")
    agg.update(total_text_tokens=total, text_token_throughput=total / wall if wall and total else None)
    itl = dict(agg.get("itl_s") or {})
    # A server that holds back partial characters has a few such requests, which leave the column meaningful.
    if multi > len(per) / 2:
        itl.pop("text", None)
    else:
        agg["text_chunk_gap_s"] = itl.get("text")
    agg["itl_s"] = itl
    return {**results, "per_request": per, "aggregate": agg}


def recount_cell(cell_dir: Path, tokenizer, output_len: int | None) -> dict | None:
    """The summary of a text cell's last attempt, recounted. None if it has no finished timed run."""
    cell = json.loads((cell_dir / "cell.json").read_text())
    good = []
    for rep in cell["attempts"][-1]["repeats"]:
        if "error" in rep:
            continue
        run_dir = Path(rep["dir"])
        res = recount_results(json.loads((run_dir / "results.json").read_text()), run_dir, tokenizer)
        st = sweep.repeat_stats(res, "text", None, output_len)
        good.append({**st, **{k: rep[k] for k in KEEP_FROM_REPEAT if k in rep}})
    return sweep.summarize_repeats(good) if good else None


def score_copy(C, arm_dir: Path, dest: Path, tree: Path) -> dict | None:
    """Score a copy of an arm's correctness pass, so the run's own report stays what the run produced."""
    corr = arm_dir / "correctness"
    if not (corr / "pass.json").is_file():
        return None
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corr, dest, ignore=shutil.ignore_patterns("scoring", "report.*", "summary.json"))
    return correctness.score_dir(C, tree, dest, "cpu")


def recount_run(run_dir: Path, out_dir: Path | None = None) -> Path:
    from transformers import AutoTokenizer

    out_dir = out_dir or run_dir
    data = table.collect(run_dir)
    tree = Path(data["meta"]["worktree"])
    C = correctness.load(REPO)
    tokenizers = {}
    for c in data["cells"]:
        wl, row = tables.WORKLOADS.get(c["workload"]), catalog.ROWS.get(c["row"])
        if wl is None or row is None or wl.family != "text":
            continue
        hf_id = row.hf_id
        if hf_id not in tokenizers:
            tokenizers[hf_id] = AutoTokenizer.from_pretrained(hf_id)
        summary = recount_cell(Path(c["dir"]), tokenizers[hf_id], wl.output_len)
        if summary:
            c["summary"] = summary
    scores = {}
    for key, arm in data["arms"].items():
        row = catalog.ROWS.get(arm["row"])
        if row is None or row.family != "text":
            continue
        arm_dir = run_dir / arm["row"] / arm["arm"]
        scored = score_copy(C, arm_dir, out_dir / "tokens" / arm["row"] / arm["arm"] / "correctness",
                            correctness._arm_tree(arm_dir, tree))
        if scored:
            scores[key] = scored
    data["correctness"] = {**data["correctness"], **scores}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip()
    data["notes"] = [f"Re-counted by benchmark.amd.recount (harness {head}) from the saved req_<i>.txt and correctness "
                     "chunks: this table's tokens, TPOT and correctness replace table.md's for arms that stream "
                     "several tokens a chunk. Timing is the original run's. A multi-token arm's ITL and chunk gap "
                     "were not saved.", *data.get("notes", [])]
    (out_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (out_dir / "tokens" / "table.json").write_text(json.dumps(data, indent=1, default=str))
    out = out_dir / "table-tokens.md"
    out.write_text(table.render(data))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="write here rather than in the run directory (one run only)")
    args = ap.parse_args(argv)
    if args.out and len(args.run_dir) > 1:
        ap.error("--out takes one run")
    for run_dir in args.run_dir:
        print(recount_run(run_dir, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
