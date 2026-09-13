#!/usr/bin/env python3
"""Validate steady Waypoint DiT CUDA replay in an Nsight SQLite export.

Export a report before running this check::

    nsys export -t sqlite -f true -o trace.sqlite trace.nsys-rep
    python3 test/waypoint/check_nsys_replay.py trace.sqlite \
        --expected-forwards 16

Only CUDA runtime calls nested inside the rollout ``engine.forward`` ranges are
examined. Synchronization in startup, output transfer, and postprocessing is
outside the steady DiT replay contract.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROLLOUT_RANGE = "worker[worker_0].node[dit].graph_walk[rollout]"


@dataclass(frozen=True)
class ReplayInspection:
    forwards: int
    graph_replays: int
    sync_or_blocking_calls: int
    offending_apis: tuple[tuple[str, int], ...]


_RANGE_CTE = """
WITH nvtx AS (
    SELECT n.rowid AS id, n.start, n.end, n.globalTid,
           coalesce(s.value, n.text) AS name
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.end IS NOT NULL
), rollout AS (
    SELECT * FROM nvtx WHERE name = :rollout_range
), forwards AS (
    SELECT DISTINCT f.*
    FROM nvtx AS f
    JOIN rollout AS r
      ON f.globalTid = r.globalTid
     AND f.start >= r.start
     AND f.end <= r.end
    WHERE f.name = 'engine.forward'
), calls AS (
    SELECT f.id AS forward_id, s.value AS api
    FROM forwards AS f
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS c
      ON c.globalTid = f.globalTid
     AND c.start >= f.start
     AND c.end <= f.end
    JOIN StringIds AS s ON s.id = c.nameId
), graph_launch_counts AS (
    SELECT forward_id, count(*) AS launches
    FROM calls
    WHERE api LIKE 'cudaGraphLaunch%'
    GROUP BY forward_id
)
"""

_BLOCKING_PREDICATE = """
api LIKE '%Synchronize%'
OR (api LIKE 'cudaMemcpy%' AND api NOT LIKE '%Async%')
OR (
    (api LIKE 'cudaMalloc%' OR api LIKE 'cudaFree%')
    AND api NOT LIKE '%Async%'
)
"""


def inspect_replay(
    database: Path,
    rollout_range: str = DEFAULT_ROLLOUT_RANGE,
) -> ReplayInspection:
    if not database.is_file():
        raise ValueError(f"Nsight SQLite export does not exist: {database}")

    summary_query = (
        _RANGE_CTE
        + """
SELECT count(*) AS forwards,
       sum(CASE WHEN coalesce(g.launches, 0) = 1 THEN 1 ELSE 0 END) AS graph_replays,
       sum(EXISTS(
           SELECT 1 FROM calls AS c
           WHERE c.forward_id = f.id AND (
"""
        + _BLOCKING_PREDICATE
        + """
           )
       )) AS sync_or_blocking_calls
FROM forwards AS f
LEFT JOIN graph_launch_counts AS g ON g.forward_id = f.id
"""
    )
    offenders_query = (
        _RANGE_CTE
        + """
SELECT api, count(*) AS occurrences
FROM calls
WHERE
"""
        + _BLOCKING_PREDICATE
        + """
GROUP BY api
ORDER BY api
"""
    )

    try:
        with sqlite3.connect(database) as connection:
            row = connection.execute(summary_query, {"rollout_range": rollout_range}).fetchone()
            offenders = tuple(
                (str(api), int(count))
                for api, count in connection.execute(
                    offenders_query,
                    {"rollout_range": rollout_range},
                )
            )
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"could not inspect Nsight SQLite export {database}: {exc}") from exc

    assert row is not None
    return ReplayInspection(
        forwards=int(row[0]),
        graph_replays=int(row[1] or 0),
        sync_or_blocking_calls=int(row[2] or 0),
        offending_apis=offenders,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite file produced by `nsys export -t sqlite`")
    parser.add_argument("--expected-forwards", type=int, required=True)
    parser.add_argument("--rollout-range", default=DEFAULT_ROLLOUT_RANGE)
    args = parser.parse_args()

    if args.expected_forwards < 1:
        parser.error("--expected-forwards must be positive")
    try:
        result = inspect_replay(args.database, args.rollout_range)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"forwards={result.forwards} graph_replays={result.graph_replays} "
        f"sync_or_blocking_calls={result.sync_or_blocking_calls}"
    )
    failures = []
    if result.forwards != args.expected_forwards:
        failures.append(f"expected {args.expected_forwards} forwards, found {result.forwards}")
    if result.graph_replays != result.forwards:
        failures.append(
            f"only {result.graph_replays}/{result.forwards} forwards launched exactly one CUDA graph"
        )
    if result.sync_or_blocking_calls:
        detail = ", ".join(f"{api}={count}" for api, count in result.offending_apis)
        failures.append(
            f"{result.sync_or_blocking_calls} forwards contain blocking CUDA calls ({detail})"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
