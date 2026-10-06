"""An early release (RELEASE_KV) and the plan kept as rows, together.

`release_kv` takes a request out of the reserved set ahead of its removal; the
plan cache keeps one row per reserved request. The rows must follow the release
(and ignore the removal after it), so a pool that plans decides as it would with
every stream read afresh, which `_DEBUG_ASSERTS` checks at every rebuild.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest

from mstar.engine.resources.kv import manager as manager_mod
from test_kv_release_at_completion import (  # noqa: F401  (the autouse fixture)
    _isolated,
    _manager,
    _prefill,
    _ready,
    _ingest,
    _step,
)


def _run(fit: str, order: str, cache: bool, monkeypatch) -> list:
    """Admit, decode, release and remove on a tight pool; what was decided, in order."""
    monkeypatch.setattr(manager_mod, "_PLAN_CACHE", cache)
    kv = _manager(max_num_pages=40, fit=fit, order=order)
    assert kv._plan_cache == (cache and kv._planned)
    seen, running, finished = [], [], set()
    for i in range(10):
        rid = f"r{i}"
        _ingest(kv, rid, prompt=48 + 16 * (i % 3), max_tokens=24 + 8 * (i % 4), base=1000 * i)
    for tick in range(60):
        for i in range(10):
            rid = f"r{i}"
            if rid in running or rid in finished:
                continue
            ok = _ready(kv, rid).ready
            seen.append((tick, rid, ok))
            if ok:
                assert _prefill(kv, rid, 48 + 16 * (i % 3)).ok
                running.append(rid)
        for rid in list(running):
            seen.append((tick, rid, "step", _step(kv, rid, 1).ok))
        if tick % 4 == 3 and running:
            done = running.pop(0)
            finished.add(done)
            kv.release_kv(done)          # the conductor says it is finished
            seen.append((tick, done, "released"))
            if tick % 8 == 7:
                kv.remove_request(done)  # the client read it; nothing left to free
        kv.assert_pages_conserved()
    for rid in list(kv._released):
        kv.remove_request(rid)
    return seen


@pytest.mark.parametrize("fit,order", [("peak", "backfill"), ("sum", "backfill"), ("peak", "fifo")])
def test_the_plan_kept_as_rows_follows_a_release_and_decides_as_reading_every_stream(fit, order, monkeypatch):
    kept = _run(fit, order, True, monkeypatch)
    fresh = _run(fit, order, False, monkeypatch)
    assert any(e[2] == "released" for e in kept)
    assert kept == fresh
