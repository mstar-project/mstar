from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

_CHECK_PATH = Path(__file__).parents[1] / "waypoint/check_nsys_replay.py"
_SPEC = importlib.util.spec_from_file_location("waypoint_nsys_check", _CHECK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHECK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CHECK
_SPEC.loader.exec_module(_CHECK)
DEFAULT_ROLLOUT_RANGE = _CHECK.DEFAULT_ROLLOUT_RANGE
inspect_replay = _CHECK.inspect_replay


def _profile_database(path: Path, apis: list[list[str]]) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                textId INTEGER, text TEXT
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER
            );
            """
        )
        strings = {DEFAULT_ROLLOUT_RANGE, "engine.forward", *(api for row in apis for api in row)}
        ids = {value: index for index, value in enumerate(sorted(strings), start=1)}
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", ((ids[v], v) for v in ids))
        for index, forward_apis in enumerate(apis):
            base = index * 100
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)",
                (base, base + 90, 7, ids[DEFAULT_ROLLOUT_RANGE]),
            )
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)",
                (base + 10, base + 80, 7, ids["engine.forward"]),
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                (
                    (base + 20 + offset, base + 21 + offset, 7, ids[api])
                    for offset, api in enumerate(forward_apis)
                ),
            )
    return path


def test_inspect_replay_accepts_graph_only_forwards(tmp_path: Path):
    database = _profile_database(
        tmp_path / "clean.sqlite",
        [["cudaMemcpyAsync", "cudaGraphLaunch_v10000"] for _ in range(3)],
    )

    result = inspect_replay(database)

    assert result.forwards == 3
    assert result.graph_replays == 3
    assert result.sync_or_blocking_calls == 0
    assert result.offending_apis == ()


def test_inspect_replay_reports_each_blocking_api(tmp_path: Path):
    database = _profile_database(
        tmp_path / "blocking.sqlite",
        [
            ["cudaGraphLaunch_v10000", "cudaDeviceSynchronize"],
            ["cudaGraphLaunch_v10000", "cudaMemcpy"],
            ["cudaGraphLaunch_v10000", "cudaMalloc"],
            ["cudaGraphLaunch_v10000", "cudaFree"],
        ],
    )

    result = inspect_replay(database)

    assert result.forwards == 4
    assert result.graph_replays == 4
    assert result.sync_or_blocking_calls == 4
    assert result.offending_apis == (
        ("cudaDeviceSynchronize", 1),
        ("cudaFree", 1),
        ("cudaMalloc", 1),
        ("cudaMemcpy", 1),
    )


def test_inspect_replay_reports_missing_graph_launch(tmp_path: Path):
    database = _profile_database(
        tmp_path / "missing-graph.sqlite",
        [["cudaGraphLaunch_v10000"], ["cudaMemcpyAsync"]],
    )

    result = inspect_replay(database)

    assert result.forwards == 2
    assert result.graph_replays == 1


def test_inspect_replay_rejects_multiple_graph_launches_in_one_forward(tmp_path: Path):
    database = _profile_database(
        tmp_path / "multiple-graphs.sqlite",
        [
            ["cudaGraphLaunch_v10000"],
            ["cudaGraphLaunch_v10000", "cudaGraphLaunch_v10000"],
        ],
    )

    result = inspect_replay(database)

    assert result.forwards == 2
    assert result.graph_replays == 1
