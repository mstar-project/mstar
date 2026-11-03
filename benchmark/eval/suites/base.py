"""What a suite provides to the harness.

A suite turns dataset items into HTTP requests, measures each request, and
scores a finished repeat. Per-request rows are plain dicts; keys that start
with ``_`` (decoded audio, ...) are kept in memory for scoring and dropped
from the results file.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.eval.datasets import Item, check_id


class Suite:
    name = ""
    help = ""
    # Item fields every item must have for this suite.
    requires: tuple[str, ...] = ()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        return None

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def config(self) -> dict[str, Any]:
        """Suite settings that decide the numbers; recorded in the results file."""
        return {}

    def default_items(self) -> list[Item] | None:
        """Items for a run without ``--data`` (a suite that builds its own requests), else None."""
        return None

    def check_items(self, items: list[Item]) -> None:
        for it in items:
            check_id(it.id)
            missing = [f for f in self.requires if not getattr(it, f)]
            if missing:
                raise ValueError(f"suite {self.name!r}: item {it.id!r} lacks {missing}")

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        raise NotImplementedError

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        """Metrics of one repeat (one number each)."""
        raise NotImplementedError


def error_row(item: Item, error: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    """A failed request. ``error`` is a fixed category (``HTTP 503``, an exception type, ...)
    and is all a results file keeps; ``detail`` (server bodies, exception text, which can
    carry addresses, paths or reflected input) goes only to the run's local error log."""
    return {"id": item.id, "ok": False, "error": error, "_detail": detail, **extra}


def exception_row(item: Item, exc: BaseException) -> dict[str, Any]:
    return error_row(item, type(exc).__name__, f"{type(exc).__name__}: {exc}")


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def parse_extra(pairs: list[str]) -> dict[str, Any]:
    """``KEY=JSON`` pairs (a bare string when the value is not JSON)."""
    out: dict[str, Any] = {}
    for item in pairs or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--extra expects KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out
