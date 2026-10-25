"""What dry runs of new arms say to leave out of a timed table: --skip patterns for the sweep, with reasons.

A start failure leaves the arm out on that row, and on every row when it failed wherever it was tried. A failed
first request or a cell with errors leaves that workload out on that row (image-to-video on every row, since the
failure comes from the engine's video route), and so does a first request that never finished. A start that lost
its port to another server says nothing about the arm, so it leaves nothing out.

    python -m benchmark.amd.dry DRY_RUN_DIR...    # the --skip arguments on stdout, the reasons on stderr
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PORT_LOST = re.compile(r"EADDRINUSE|[Aa]ddress already in use|is answered by pid|port \d+ is (?:busy|held)"
                       r"|no free port among")
HUNG = re.compile(r"the first (\S+) request at \S+ did not finish")


def skips(run_dirs) -> tuple[list[str], list[str]]:
    """The --skip patterns (ROW/ARM/WORKLOAD) the dry runs under ``run_dirs`` call for, and a reason per finding."""
    out: list[str] = []
    why: list[str] = []
    tried, failed = defaultdict(set), defaultdict(set)
    for run in map(Path, run_dirs):
        for f in sorted(run.glob("*/*/arm.json")):
            info = json.loads(f.read_text())
            if f.parent.name != info.get("arm"):
                continue  # an arm run set aside beside the live one
            arm, row = info["arm"], info.get("row")
            err = (info.get("error") or "").strip()
            last = (err.splitlines() or [""])[-1][:200]
            if info.get("status") != "done":
                hung = HUNG.search(last)
                if hung and info.get("startup"):
                    wl = hung.group(1)
                    tried[arm].add(row)
                    out.append(f"{'*' if wl == 'video-i2v' else row}/{arm}/{wl}")
                    why.append(f"{row}/{arm} {wl}: {last}")
                elif PORT_LOST.search(err):
                    why.append(f"{row}/{arm} lost its port to another server, not left out: {last}")
                else:
                    tried[arm].add(row)
                    failed[arm].add(row)
                    out.append(f"{row}/{arm}/*")
                    why.append(f"{row}/{arm} did not start: {last or 'status ' + str(info.get('status'))}")
                continue
            tried[arm].add(row)
            bad: dict = {}
            for wl, sizes in (info.get("first_size") or {}).items():
                for size, v in sizes.items():
                    if isinstance(v, str):
                        bad.setdefault(wl, f"first request at {size}: {v}")
            for c in sorted(f.parent.glob("cells/*/c*/cell.json")):
                cell = json.loads(c.read_text())
                if cell.get("status") != "done" or cell.get("errors"):
                    bad.setdefault(cell.get("workload"), "; ".join(cell.get("errors") or [])[:200])
            for wl, reason in sorted(bad.items()):
                out.append(f"{'*' if wl == 'video-i2v' else row}/{arm}/{wl}")
                why.append(f"{row}/{arm} {wl} failed: {reason}")
    out += [f"*/{a}/*" for a, rows in failed.items() if rows == tried[a]]
    return list(dict.fromkeys(out)), why


def main(argv=None) -> int:
    out, why = skips(sys.argv[1:] if argv is None else argv)
    for w in why:
        print(f"dry run: {w}", file=sys.stderr)
    print(" ".join(f"--skip {x}" for x in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
