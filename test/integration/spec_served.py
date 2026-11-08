"""What the served real-weights modules of speculative decoding share: their prompts, a checkpoint's cached snapshot,
the e2e server harness and the counters at ``/metrics``. Nothing here knows a model.

The integration tests are no package: load it by path, as ``fp32_screen.py`` is.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

E2E = Path(__file__).resolve().parents[1] / "e2e"

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May? Think step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three sentences why the sky is blue.",
    "List the first ten prime numbers, separated by commas, then sum them.",
    "Translate into French: The library opens at nine and closes at five on weekdays.",
    "A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip? Show your work.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What is 17 * 23? Answer with the computation.",
]


def _snapshot(repo: str) -> Path | None:
    """A cached snapshot holding every shard its index names (or its one shard)."""
    roots = [Path(os.environ["HF_HUB_CACHE"])] if os.environ.get("HF_HUB_CACHE") else []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / f"models--{repo.replace('/', '--')}" / "snapshots"
        for snap in sorted(snapshots.iterdir()) if snapshots.is_dir() else ():
            index = snap / "model.safetensors.index.json"
            if index.is_file():
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if all((snap / shard).is_file() for shard in shards):
                    return snap
            elif (snap / "model.safetensors").is_file() and (snap / "config.json").is_file():
                return snap
    return None


def _server_module():
    key = "_e2e_server"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, E2E / "server.py")
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


def _totals(url: str, *names: str) -> tuple[float, ...]:
    """Each counter's total at ``/metrics``, once the workers' 1 s reports have settled."""
    def total(text: str, name: str) -> float:
        return sum(float(v) for v in re.findall(rf"^{name}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M))

    last = None
    for _ in range(20):
        text = requests.get(url + "/metrics", timeout=30).text
        now = tuple(total(text, name) for name in names)
        if now == last:
            return now
        last = now
        time.sleep(1.5)
    return last


# each server's last reading, by url
_LAST: dict[str, tuple[float, ...]] = {}


def _counters(url: str, wait: bool = True) -> tuple[float, float]:
    """(verify rows, accepted drafts). A worker publishes its counters once a
    status interval, so the last requests' rows can trail their replies: read
    until the rows pass this server's last reading (10 s at most), unless
    ``wait`` is off, as at a phase's first reading."""
    since = _LAST.get(url, (0.0,))[0] if wait else None
    deadline = time.monotonic() + 10
    now = _totals(url, "mstar_spec_verify_rows_total", "mstar_spec_accepted_tokens_total")
    while since is not None and now[0] <= since and time.monotonic() < deadline:
        time.sleep(0.25)
        now = _totals(url, "mstar_spec_verify_rows_total", "mstar_spec_accepted_tokens_total")
    _LAST[url] = now
    return now


def _tau(before: tuple[float, float], after: tuple[float, float]) -> float:
    rows, accepted = after[0] - before[0], after[1] - before[1]
    assert rows > 0, "no verify rows reported"
    return (accepted + rows) / rows
