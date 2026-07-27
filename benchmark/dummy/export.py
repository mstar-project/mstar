"""Turn a directory of rendered aggregate summaries into paste-ready TSV.

``mstar.profile.aggregate`` renders one human-readable summary per sweep cell
(e.g. ``benchmark_results/dummy_loop/dummy_loop_K_10_BS_4.txt``). This reads
those back and emits tab-separated output, which Google Sheets splits into
columns on paste.

    # wide: one column per sweep cell, one row per metric  (the usual one)
    python -m benchmark.dummy.export benchmark_results/dummy_loop --stat mean

    # just the numbers you actually want to chart
    python -m benchmark.dummy.export benchmark_results/dummy_loop \\
        --select "execs/req" --select "all (ms/exec)" --select total

    # long/tidy: one row per (cell, metric, stat) — for pivot tables
    python -m benchmark.dummy.export benchmark_results/dummy_loop --long

Cell labels come from the filename: any ``<key>_<number>`` pairs become
columns, so ``dummy_loop_K_10_BS_4.txt`` yields ``K=10, BS=4`` and the sweep
sorts numerically rather than lexically (K=100 after K=10, not before).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STAT_COLUMNS = ("n", "mean", "p50", "p95", "p99", "min", "max")

_RE_RULE = re.compile(r"^[=\-\s]+$")
_RE_HEADING = re.compile(r"^ (\S.*?)\s*$")  # exactly one leading space
_RE_AGG_COUNT = re.compile(r"Aggregate over (\d+) requests")
# "dummy_loop_K_10_BS_4" -> [("K", 10), ("BS", 4)]
_RE_LABEL_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9]*)_(\d+(?:\.\d+)?)")


@dataclass
class Row:
    section: str
    group: tuple[str, ...]
    metric: str
    stats: dict[str, str]

    @property
    def key(self) -> tuple[str, tuple[str, ...], str]:
        return (self.section, self.group, self.metric)

    def label(self) -> str:
        parts = [*self.group, self.metric]
        return " / ".join(parts)


@dataclass
class Cell:
    """One sweep point: the parsed contents of a single summary file."""

    path: Path
    name: str
    params: dict[str, float] = field(default_factory=dict)
    num_requests: int | None = None
    rows: list[Row] = field(default_factory=list)

    def label(self) -> str:
        if not self.params:
            return self.name
        return " ".join(f"{k}={_num(v)}" for k, v in self.params.items())

    def by_key(self) -> dict[tuple[str, tuple[str, ...], str], Row]:
        return {r.key: r for r in self.rows}


def _num(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _split(line: str) -> list[str]:
    """Split a rendered table line on 2+ spaces.

    Safe because the renderer joins columns with exactly ``"  "`` and pads
    cells to width, while values that contain spaces ("recv → preprocess
    done", "worker_0 → api_server_preprocess_worker") only ever use single
    spaces internally.
    """
    return [c for c in re.split(r"\s{2,}", line.strip()) if c]


def parse_summary(path: Path) -> Cell:
    """Parse one rendered ``Aggregator.render()`` output."""
    cell = Cell(path=path, name=path.stem, params=parse_params(path.stem))
    section = ""
    headers: list[str] = []

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        count = _RE_AGG_COUNT.search(line)
        if count:
            cell.num_requests = int(count.group(1))
            continue
        if _RE_RULE.match(line):
            continue

        heading = _RE_HEADING.match(line)
        if heading and not line.startswith("  "):
            section = heading.group(1)
            headers = []
            continue

        cols = _split(line)
        if not cols:
            continue

        # The first table line after a heading is the column header; it always
        # ends with the fixed stat columns.
        if not headers:
            if cols[-len(STAT_COLUMNS):] == list(STAT_COLUMNS):
                headers = cols
            continue

        if len(cols) != len(headers):
            continue  # footnote or wrapped text, not a data row
        n_group = len(headers) - len(STAT_COLUMNS) - 1  # minus the "metric" col
        cell.rows.append(
            Row(
                section=section,
                group=tuple(cols[:n_group]),
                metric=cols[n_group],
                stats=dict(zip(STAT_COLUMNS, cols[n_group + 1:], strict=True)),
            )
        )
    return cell


def parse_params(stem: str) -> dict[str, float]:
    """Pull ``<key>_<number>`` pairs out of a filename for sorting/labelling."""
    return {k: float(v) for k, v in _RE_LABEL_PAIR.findall(stem)}


def sort_key(cell: Cell) -> tuple:
    # Sort by the numeric params in the order they appear in the filename, so
    # K=100 lands after K=10.
    return (tuple(cell.params.values()), cell.name)


def matches(row: Row, selects: list[str]) -> bool:
    if not selects:
        return True
    haystack = f"{row.section} {' '.join(row.group)} {row.metric}".lower()
    return any(s.lower() in haystack for s in selects)


def render_wide(cells: list[Cell], stat: str, selects: list[str]) -> str:
    """Rows = metrics, columns = sweep cells. One statistic per sheet."""
    maps = [c.by_key() for c in cells]

    # Preserve first-seen order across all cells so the sheet reads like the
    # original summaries.
    keys: list[tuple[str, tuple[str, ...], str]] = []
    seen = set()
    for cell in cells:
        for row in cell.rows:
            if row.key not in seen and matches(row, selects):
                seen.add(row.key)
                keys.append(row.key)

    param_names: list[str] = []
    for cell in cells:
        for name in cell.params:
            if name not in param_names:
                param_names.append(name)

    lines: list[str] = []
    header = ["section", "group", "metric", *[c.label() for c in cells]]
    # Parameter rows above the data make the sheet chart-friendly: select a
    # metric row plus the K row and you have an x/y series.
    for name in param_names:
        lines.append(
            "\t".join(
                ["", "", name, *[_num(c.params.get(name, float("nan")))
                                 if name in c.params else "" for c in cells]]
            )
        )
    lines.append("\t".join(["", "", "requests",
                            *[str(c.num_requests or "") for c in cells]]))
    lines.insert(0, "\t".join(header))

    for key in keys:
        section, group, metric = key
        values = [m[key].stats.get(stat, "") if key in m else "" for m in maps]
        lines.append("\t".join([section, " / ".join(group), metric, *values]))
    return "\n".join(lines) + "\n"


def render_long(cells: list[Cell], selects: list[str]) -> str:
    """One row per (cell, metric, stat) — feed this to a pivot table."""
    param_names: list[str] = []
    for cell in cells:
        for name in cell.params:
            if name not in param_names:
                param_names.append(name)

    lines = ["\t".join([
        "file", *param_names, "requests", "section", "group", "metric",
        *STAT_COLUMNS,
    ])]
    for cell in cells:
        params = [
            _num(cell.params[p]) if p in cell.params else "" for p in param_names
        ]
        for row in cell.rows:
            if not matches(row, selects):
                continue
            lines.append(
                "\t".join([
                    cell.name,
                    *params,
                    str(cell.num_requests or ""),
                    row.section,
                    " / ".join(row.group),
                    row.metric,
                    *[row.stats.get(s, "") for s in STAT_COLUMNS],
                ])
            )
    return "\n".join(lines) + "\n"


def collect(paths: list[str], pattern: str) -> list[Cell]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob(pattern)))
        else:
            files.append(p)
    if not files:
        raise SystemExit(f"no summary files found in {paths} (pattern {pattern!r})")
    cells = [parse_summary(f) for f in files]
    return sorted(cells, key=sort_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.dummy.export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="+",
        help="summary files, or directories of them",
    )
    parser.add_argument(
        "--pattern", default="*.txt", help="glob used when a path is a directory",
    )
    parser.add_argument(
        "--stat", default="mean", choices=STAT_COLUMNS,
        help="which statistic fills the wide table (default: mean)",
    )
    parser.add_argument(
        "--long", action="store_true",
        help="emit tidy rows (every stat) instead of the wide pivot",
    )
    parser.add_argument(
        "--select", action="append", default=[],
        help="keep only metrics whose section/group/name contains this "
             "substring; repeatable",
    )
    parser.add_argument("-o", "--output", default=None, help="write here (default stdout)")
    args = parser.parse_args(argv)

    cells = collect(args.paths, args.pattern)
    text = (
        render_long(cells, args.select)
        if args.long
        else render_wide(cells, args.stat, args.select)
    )

    if args.output:
        Path(args.output).write_text(text)
        print(
            f"wrote {args.output}  ({len(cells)} cells, "
            f"{len(text.splitlines()) - 1} rows)",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
