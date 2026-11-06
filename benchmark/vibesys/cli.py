"""``python -m benchmark.vibesys``: record rounds, decide a round, tabulate a sweep, chart a loop.

Examples::

    # append one round of one arm to the results file, and print the row as written
    python -m benchmark.vibesys report --results results.jsonl '{"batch": "rp2-loop", ...}'

    # decide a round from its rows, exiting 1 when it isn't kept
    python -m benchmark.vibesys keep results.jsonl --hypothesis "fused GDN input stage, expect 1.04x" \\
        --model Qwen3.5-0.8B --workload "Random 1024/256" --c 64

    # a checkpoint sweep, one line per cell
    python -m benchmark.vibesys table results.jsonl --batch h200_1106 --hypothesis "checkpoint: run-base vs 61b95ae05"

    # the loop's best-so-far line on its headline cell, needs matplotlib (the eval extra)
    python -m benchmark.vibesys chart results.jsonl --model Qwen3.5-0.8B --workload ShareGPT --c 64 --out progress.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.vibesys import progress
from benchmark.vibesys.rows import METRICS, Cell, Row, keep, load, table


def _report(args: argparse.Namespace) -> int:
    try:
        row = Row.parse(json.loads(args.row))
    except ValueError as e:
        print(f"not recorded: {e}", file=sys.stderr)
        return 2
    with Path(args.results).open("a") as f:
        f.write(row.to_json() + "\n")
    print(row.to_json())
    return 0


def _keep(args: argparse.Namespace) -> int:
    rows = [r for r in load(args.file) if r.hypothesis == args.hypothesis and args.batch in (None, r.batch)]
    decision = keep(rows, Cell(args.model, args.workload, args.gpus, args.c), args.metric, args.rounds)
    print(("kept: " if decision.kept else "not kept: ") + decision.reason)
    return 0 if decision.kept else 1


def _table(args: argparse.Namespace) -> int:
    rows = [r for r in load(args.file) if r.batch == args.batch and args.hypothesis in (None, r.hypothesis)]
    print(table(rows, args.metric))
    return 0


def _chart(args: argparse.Namespace) -> int:
    title = args.title or f"{args.model}, {args.workload}, c={args.c}"
    progress.chart(load(args.file), Cell(args.model, args.workload, args.gpus, args.c), args.out, title, args.metric)
    return 0


def _add_cell(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--c", type=int, required=True)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--metric", default="throughput", choices=METRICS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.vibesys",
        allow_abbrev=False,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", allow_abbrev=False, help="append one row to a results file")
    p.add_argument("--results", required=True, help="the results file to append to")
    p.add_argument("row", help="the row as a JSON object")
    p.set_defaults(run=_report)

    p = sub.add_parser("keep", allow_abbrev=False, help="decide a round from its rows, and exit 1 when not kept")
    p.add_argument("file", help="a results file")
    p.add_argument("--hypothesis", required=True, help="the round's rows are the ones with this hypothesis")
    p.add_argument("--batch", help="and this batch, when another batch has a round with the same hypothesis")
    p.add_argument("--rounds", type=int, default=3, help="succeeded rounds each arm needs")
    _add_cell(p)
    p.set_defaults(run=_keep)

    p = sub.add_parser("table", allow_abbrev=False, help="one line per cell of a batch")
    p.add_argument("file", help="a results file")
    p.add_argument("--batch", required=True)
    p.add_argument("--hypothesis", help="only the rows with this hypothesis, when the batch holds several sweeps")
    p.add_argument("--metric", default="throughput", choices=METRICS)
    p.set_defaults(run=_table)

    p = sub.add_parser("chart", allow_abbrev=False, help="draw the best-so-far line of the loop's headline cell")
    p.add_argument("file", help="a results file")
    p.add_argument("--out", required=True, help="image file to write")
    p.add_argument("--title")
    _add_cell(p)
    p.set_defaults(run=_chart)

    args = parser.parse_args(argv)
    return args.run(args)
