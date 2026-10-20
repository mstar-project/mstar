"""Start every (row, arm) server of a table once, in parallel over the given GPUs, then stop it.

The first start of a server in a job pays its JIT and compile work (AITER builds, Inductor autotune,
torch.compile), so these starts are the cold-start numbers, and the table's own starts are warm.

    python -m benchmark.amd.prewarm --table qwen35-dense --gpus 0,1,2,3 --out DIR --worktree MSTAR_TREE
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

from benchmark.amd import catalog, tables
from benchmark.amd.launch import REPO, Launch, arm_worktree, gpu_list, tree_list


def plan_jobs(rows: list[str], arms: list[str]) -> list[tuple[catalog.Row, catalog.Arm]]:
    jobs = []
    for r in rows:
        row = catalog.ROWS[r]
        jobs += [(row, a) for a in catalog.arms_for(row, arms) if a.engine != "reference"]
    # Wide rows first, so they don't wait behind a tail of one-GPU starts.
    return sorted(jobs, key=lambda j: -j[0].gpus)


def run(jobs, gpus: list[int], out: Path, worktree: Path, port_base: int, log=print,
        trees: dict[str, Path] | None = None) -> dict:
    free = list(gpus)
    cond = threading.Condition()
    results: dict = {}

    def worker(row, arm, mine, port):
        key = f"{row.name}/{arm.name}"
        launch = Launch(arm=arm, row=row, gpus=mine, port=port, run_dir=out / row.name / arm.name,
                        worktree=arm_worktree(arm, worktree, trees or {}))
        t0 = time.monotonic()
        try:
            res = {"gpus": mine, **launch.start()}
        except Exception:
            res = {"gpus": mine, "error": traceback.format_exc()[-2000:]}
        finally:
            launch.stop()
            with cond:
                free.extend(mine)
                cond.notify_all()
        res["total_s"] = round(time.monotonic() - t0, 1)
        results[key] = res
        log(f"{key} on {mine}: ready {res.get('ready_s')} s, first request {res.get('first_request_s')} s"
            + (" ERROR" if "error" in res else ""))

    threads = []
    for i, (row, arm) in enumerate(jobs):
        with cond:
            while len(free) < row.gpus:
                cond.wait()
            mine = sorted(free)[: row.gpus]
            for g in mine:
                free.remove(g)
        t = threading.Thread(target=worker, args=(row, arm, mine, port_base + i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    (out / "prewarm.json").write_text(json.dumps(results, indent=1, default=str))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, choices=sorted(tables.TABLES))
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worktree", default=str(REPO))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--port-base", type=int, default=8600)
    ap.add_argument("--tree", action="append", default=[], metavar="NAME=PATH",
                    help="a named mstar tree for the arms that serve one (as in the sweep)")
    args = ap.parse_args(argv)
    t = tables.TABLES[args.table]
    trees = tree_list(args.tree)
    jobs = [(row, arm) for row, arm in plan_jobs(args.rows.split(",") if args.rows else list(t.rows),
                                                 args.arms.split(",") if args.arms else list(t.arms))
            if not arm.tree or arm.tree in trees]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(jobs, gpu_list(args.gpus), out, Path(args.worktree).resolve(), args.port_base,
              log=lambda m: print(time.strftime("[%H:%M:%S] ") + m, flush=True), trees=trees)
    return 1 if any("error" in r for r in res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
