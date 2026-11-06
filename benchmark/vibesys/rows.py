"""Result rows from optimization rounds, and the rule that keeps a round.

A row is one timed round of one arm on one cell, a line of JSON in a results file. The
baseline arm is the current best commit and the candidate arm is that commit plus the
round's change.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

ROLES = ("baseline", "candidate")
# `probe` rows come from loop rounds, `counted` rows from checkpoint sweeps.
STATUSES = ("probe", "counted")
METRICS = ("throughput", "p50_ms", "p99_ms")
LOWER_IS_BETTER = frozenset({"p50_ms", "p99_ms"})

BETTER, WORSE, OVERLAP = "better", "worse", "overlap"

# keyed by the annotations as strings, which is how `from __future__ import annotations` keeps them
_ACCEPTS = {"str": (str,), "int": (int,), "float | None": (int, float, type(None))}


class Cell(NamedTuple):
    """What the two arms are compared on."""

    model: str
    workload: str
    gpus: int
    c: int


class Arms(NamedTuple):
    base: list[float]
    cand: list[float]


class Decision(NamedTuple):
    kept: bool
    reason: str


@dataclass(frozen=True)
class Row:
    batch: str
    family: str
    model: str
    workload: str
    gpus: int
    engine: str
    role: str
    config: str
    c: int
    round: int
    throughput: float | None
    unit: str
    latency: str
    p50_ms: float | None
    p99_ms: float | None
    requests: int
    failed: int
    status: str
    started: str
    error: str
    commit: str
    hypothesis: str

    @classmethod
    def parse(cls, obj: object) -> Row:
        if not isinstance(obj, dict):
            raise ValueError(f"a row is a JSON object, not {type(obj).__name__}")
        names = [f.name for f in fields(cls)]
        missing = [name for name in names if name not in obj]
        unknown = [key for key in obj if key not in names]
        if missing or unknown:
            raise ValueError(f"row keys don't match the schema: missing {missing}, unknown {unknown}")
        # isinstance counts True as an int, and no field is a bool
        wrong = [
            f"{f.name}={obj[f.name]!r}"
            for f in fields(cls)
            if isinstance(obj[f.name], bool) or not isinstance(obj[f.name], _ACCEPTS[f.type])
        ]
        if wrong:
            raise ValueError(f"wrong types: {', '.join(wrong)}")
        if obj["role"] not in ROLES:
            raise ValueError(f"role is one of {ROLES}, not {obj['role']!r}")
        if obj["status"] not in STATUSES:
            raise ValueError(f"status is one of {STATUSES}, not {obj['status']!r}")
        if datetime.fromisoformat(obj["started"]).tzinfo is None:
            raise ValueError(f"started needs its UTC offset, as in 2026-11-03T14:05-05:00, not {obj['started']!r}")
        return cls(**obj)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @property
    def cell(self) -> Cell:
        return Cell(self.model, self.workload, self.gpus, self.c)

    @property
    def succeeded(self) -> bool:
        return self.failed == 0 and not self.error


def load(path: str | Path) -> list[Row]:
    """Read every row of a results file, or raise naming the first line that isn't one."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(Row.parse(json.loads(line)))
        except ValueError as e:
            raise ValueError(f"{path}:{number}: {e}") from e
    return rows


def sweeps(rows: Iterable[Row]) -> dict[tuple[str, str], list[Row]]:
    """Group rows by batch and hypothesis, which together name one loop round or one checkpoint sweep."""
    out: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        out.setdefault((row.batch, row.hypothesis), []).append(row)
    return out


def arms(rows: Iterable[Row], metric: str) -> dict[Cell, Arms]:
    """Split ``metric`` from the succeeded rounds by cell and role."""
    out: dict[Cell, Arms] = {}
    for row in rows:
        value = getattr(row, metric)
        if not row.succeeded or value is None:
            continue
        arm = out.setdefault(row.cell, Arms([], []))
        (arm.base if row.role == "baseline" else arm.cand).append(value)
    return out


def separation(arm: Arms, metric: str) -> str:
    """`BETTER` when every candidate round beats every baseline round, `WORSE` when every
    baseline round beats every candidate round, and `OVERLAP` otherwise."""
    sign = -1 if metric in LOWER_IS_BETTER else 1
    base = [sign * v for v in arm.base]
    cand = [sign * v for v in arm.cand]
    if min(cand) > max(base):
        return BETTER
    if max(cand) < min(base):
        return WORSE
    return OVERLAP


def ratio(arm: Arms) -> float:
    return statistics.fmean(arm.cand) / statistics.fmean(arm.base)


def keep(rows: Iterable[Row], headline: Cell, metric: str = "throughput", rounds: int = 3) -> Decision:
    """Decide one round from its rows.

    The round is kept when ``headline`` has at least ``rounds`` succeeded rounds per arm, every
    candidate round beats every baseline round on ``metric``, and no other cell measured in the
    round is `WORSE` on ``metric``.
    """
    by_cell = arms(rows, metric)
    arm = by_cell.get(headline, Arms([], []))
    if len(arm.base) < rounds or len(arm.cand) < rounds:
        return Decision(
            False,
            f"{_name(headline)} has {len(arm.base)} baseline and {len(arm.cand)} candidate "
            f"rounds that succeeded, and needs {rounds} of each",
        )
    verdict = separation(arm, metric)
    if verdict != BETTER:
        return Decision(False, f"{_name(headline)} {metric} {ratio(arm):.3f}x, rounds {verdict}")
    worse = [
        _name(cell)
        for cell, other in by_cell.items()
        if cell != headline and other.base and other.cand and separation(other, metric) == WORSE
    ]
    if worse:
        return Decision(False, f"{metric} worse with no overlap on {', '.join(worse)}")
    return Decision(True, f"{_name(headline)} {metric} {ratio(arm):.3f}x, every candidate round ahead")


def table(rows: Iterable[Row], metric: str = "throughput") -> str:
    """A markdown table with one line per cell that has both arms: mean and range per arm,
    the ratio and the separation."""
    rows = list(rows)
    engines = {role: sorted({r.engine for r in rows if r.role == role}) for role in ROLES}
    lines = [
        f"| model | workload | c | {' / '.join(engines['baseline'])} | {' / '.join(engines['candidate'])} "
        f"| ratio | rounds |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell, arm in sorted(arms(rows, metric).items()):
        if not arm.base or not arm.cand:
            continue
        lines.append(
            f"| {cell.model} | {cell.workload} | {cell.c} | {_spread(arm.base)} | {_spread(arm.cand)} "
            f"| {ratio(arm):.3f} | {separation(arm, metric)} |"
        )
    return "\n".join(lines)


def _name(cell: Cell) -> str:
    gpus = f" on {cell.gpus} GPUs" if cell.gpus != 1 else ""
    return f"{cell.model} {cell.workload} c={cell.c}{gpus}"


def _spread(values: list[float]) -> str:
    return f"{statistics.fmean(values):,.1f} ({min(values):,.1f}-{max(values):,.1f}, n={len(values)})"
