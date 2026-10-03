from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def live_server_benchmark():
    return runpy.run_path(
        str(Path(__file__).parents[2] / "benchmark" / "waypoint" / "benchmark_live_server.py")
    )


def test_live_server_benchmark_parses_minimal_argv_without_a_server_launch_helper(
    live_server_benchmark, tmp_path
):
    """The whole point of this script is to talk to a server someone else
    started, so it must not import ``subprocess`` or define its own
    server-launch helpers (``_server_command``/``_run_config`` stay in
    serve_rollout.py, reused rather than duplicated)."""
    module = live_server_benchmark
    assert "subprocess" not in module
    assert "_server_command" not in module
    assert "_run_config" not in module

    args = module["_parse_args"](
        [
            "--port",
            "8123",
            "--variant",
            "360p",
            "--seed-image",
            str(tmp_path / "seed.jpg"),
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8123
    assert args.streams == 1
    assert args.warmup_steps == 1
    assert args.request_timeout == 900.0
