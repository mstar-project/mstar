"""What it costs a pool that backfills to be asked about a long queue.

The scheduler asks readiness of every request still waiting, on every pass. Under
``backfill`` each one used to be worked out in full, and a pool with a thousand
requests waiting spent more time answering that than running them. The pool now
looks only at the first ``backfill_window`` of them.
"""

from __future__ import annotations

import random
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_admission_peak import (
    DECODE,
    NODE,
    PAGE_SIZE,
    PREFILL,
    _drive,
    _finish,
    _ingest,
    _prefill,
    _ready,
    _request,
    _run,
    _StubTransfer,
)

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager

SEED = 20261006
FIT_AND_ORDER = [("sum", "backfill"), ("peak", "backfill")]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    for name in (
        "MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW",
        "MSTAR_STEP_TELEMETRY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _manager(
    max_num_pages: int = 16, fit: str = "peak", order: str = "backfill", **config,
) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            admission_fit=fit, admission_order=order, **config,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", {"main": (PREFILL, DECODE)})
    return kv


class _Counted:
    """Counts the calls to what a pool does in full for a request."""

    def __init__(self, kv: KVManager):
        self.full = 0
        admissible = kv._admissible

        def counting_admissible(*args, **kwargs):
            self.full += 1
            return admissible(*args, **kwargs)

        kv._admissible = counting_admissible


# ── the pool's own answers ─────────────────────────────────────────────


def _queue_run(kv: KVManager, seed: int, ops: int) -> tuple[list, dict]:
    """Requests arriving into a small pool and asked about by passes over the
    whole queue (in arrival order, or shuffled), run at their own pace, now and
    then aborted. Everything that decides who is admitted is in the log it returns.
    """
    rng = random.Random(seed)
    families = [list(range(base, base + rng.randrange(20, 70))) for base in (0, 1000, 2000)]
    queue: list[str] = []
    waiting: dict[str, tuple[int, int]] = {}
    admitted: dict[str, int] = {}
    running: dict[str, int] = {}
    log: list = []
    seen = dict(passes=0, admitted=0, finished=0, aborted=0)

    def step(rid: str) -> None:
        if rid in admitted:
            outcome = _prefill(kv, rid, admitted[rid])
            if outcome.ok:
                running[rid] = waiting.pop(rid)[1]
                del admitted[rid]
                log.append(("prefill", rid))
            return
        span = min(running[rid], rng.randrange(1, 2 * PAGE_SIZE))
        outcome = _run(kv, {rid: span}, DECODE)
        log.append(("decode", rid, span, outcome.ok))
        if outcome.ok:
            running[rid] -= span
            if not running[rid]:
                del running[rid]
                kv.remove_request(rid)
                seen["finished"] += 1

    for i in range(ops):
        roll = rng.random()
        if roll < 0.28 and len(queue) < 60:
            rid = f"r{i}"
            prompt = rng.choice(families) + [7] * rng.randrange(0, 30)
            max_tokens = rng.randrange(1, 10 * PAGE_SIZE)
            kv.ingest_request(rid, _request(prompt, max_tokens))
            queue.append(rid)
            waiting[rid] = (len(prompt), max_tokens)
        elif roll < 0.50 and queue:
            asked = list(queue)
            if rng.random() < 0.3:
                rng.shuffle(asked)
            seen["passes"] += 1
            let_in = []
            for rid in asked:
                if _ready(kv, rid).ready:
                    let_in.append(rid)
                    queue.remove(rid)
                    admitted[rid] = waiting[rid][0]
            seen["admitted"] += len(let_in)
            log.append(("pass", tuple(asked), tuple(let_in), tuple(kv._waiting), tuple(kv._reserved)))
        elif roll < 0.975 and (admitted or running):
            step(rng.choice(list(admitted) + list(running)))
        elif roll >= 0.975 and (queue or admitted or running):
            rid = rng.choice(queue + list(admitted) + list(running))
            for group in (waiting, admitted, running):
                group.pop(rid, None)
            if rid in queue:
                queue.remove(rid)
            kv.remove_request(rid)
            seen["aborted"] += 1
            log.append(("abort", rid))
        if manager_mod._DEBUG_ASSERTS or i % 8 == 0:
            kv.assert_pages_conserved()
        if kv._peak:
            assert kv._admitted_are_safe()
    return log, seen


# ── configuration ───────────────────────────────────────────────────────


def _config(**fields) -> PagedKVConfig:
    return PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, **fields)


def test_the_window_is_128_unless_told_otherwise():
    assert _config().resolved_backfill_window() == 128 == _config().backfill_window


def test_the_window_can_be_set_in_the_yaml_and_the_environment_overrules_it(monkeypatch):
    cfg = _config()
    cfg.apply_yaml_overrides(backfill_window=32)
    assert cfg.resolved_backfill_window() == 32

    monkeypatch.setenv("MSTAR_KV_BACKFILL_WINDOW", "7")
    assert cfg.resolved_backfill_window() == 7
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv._window_size == 7, "the environment did not reach the pool"


def test_the_window_in_the_yaml_reaches_the_pool_the_engine_builds_from_a_models_spec():
    from mstar.engine.resources.kv.config import KVSpec

    spec = KVSpec(resource_key="kv", nodes={"LLM"}, leader="LLM", config=_config())
    spec.apply_yaml_overrides(backfill_window=5, admission_order="backfill")
    info = SimpleNamespace(
        device=torch.device("cpu"), joint_comm_group=None, transfer_engine_info=None,
        kv_dtype=torch.float32, needs_remote_transfer=False, nodes=None,
    )

    assert KVManager.build(spec, info)._window_size == 5


@pytest.mark.parametrize("value", [0, -3, "many", "1.5", "", True, 2.0])
def test_a_window_that_is_not_a_positive_whole_number_is_an_error_not_a_default(value):
    with pytest.raises(ValueError, match="backfill_window"):
        _config().apply_yaml_overrides(backfill_window=value)
    with pytest.raises(ValueError, match="backfill_window"):
        _config(backfill_window=value).resolved_backfill_window()


@pytest.mark.parametrize("value", ["0", "-1", "many", "2.5"])
def test_a_window_in_the_environment_that_is_not_one_is_an_error(monkeypatch, value):
    monkeypatch.setenv("MSTAR_KV_BACKFILL_WINDOW", value)

    with pytest.raises(ValueError, match="MSTAR_KV_BACKFILL_WINDOW"):
        _config().resolved_backfill_window()


# ── the window ──────────────────────────────────────────────────────────


def _hog_and_queue(window: int, queue: str, fit: str = "sum") -> KVManager:
    """One admitted request holding 10 of the 15 pages, and ``queue`` (one letter
    a rid) asking, each for 10 pages that outlast it: five more than is left."""
    kv = _manager(16, fit=fit, order="backfill", backfill_window=window)
    kv.ingest_request("hog", _request(list(range(3000, 3100)), 60))
    assert _ready(kv, "hog").ready
    for rid in queue:
        kv.ingest_request(rid, _request(list(range(ord(rid) * 1000, ord(rid) * 1000 + 100)), 60))
    return kv


@pytest.mark.parametrize("fit", ["sum", "peak"])
def test_a_request_behind_the_window_is_not_asked_until_it_is_in_it(fit):
    kv = _hog_and_queue(2, "ab", fit)
    # one page each: they fit, but are behind the two that do not
    for rid in "cd":
        kv.ingest_request(rid, _request(list(range(ord(rid) * 1000, ord(rid) * 1000 + 10)), 4))
    counted = _Counted(kv)

    assert [_ready(kv, rid).ready for rid in "abcd"] == [False] * 4
    assert list(kv._waiting) == ["a", "b", "c", "d"]
    assert list(kv._window) == ["a", "b"]
    full = counted.full
    assert [_ready(kv, rid).ready for rid in "cd"] == [False, False]
    assert counted.full == full, "a request behind the window was worked out"

    kv.remove_request("a")
    assert list(kv._eligible()) == ["b", "c"], "the next in line did not take its place"
    assert _ready(kv, "c").ready, "a request that fits was not admitted once it was in the window"
    assert not _ready(kv, "d").ready or "d" in kv._reserved


def test_the_window_is_kept_not_read_from_the_queue_on_every_ask():
    kv = _hog_and_queue(3, "abcde")
    for rid in "abcde":
        kv._gate(rid, NODE, PREFILL)
    window = kv._window
    assert list(window) == ["a", "b", "c"]

    for _ in range(3):
        for rid in "abcde":
            assert kv._gate(rid, NODE, PREFILL) is not None
    kv.remove_request("e")
    assert kv._window is window, "the window was dropped when a request behind it left"

    kv.remove_request("b")
    assert kv._window is None, "the window was kept after one of its members left"
    assert list(kv._eligible()) == ["a", "c", "d"]


def test_the_window_holds_the_oldest_when_it_is_not_full():
    kv = _hog_and_queue(4, "abcd")
    for rid in "abc":
        kv._gate(rid, NODE, PREFILL)
    assert list(kv._eligible()) == ["a", "b", "c"]

    kv.remove_request("a")
    kv._gate("d", NODE, PREFILL)
    assert list(kv._eligible()) == ["b", "c", "d"] == list(kv._waiting)


@pytest.mark.parametrize("window", [1, 2, 5])
@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_small_window_never_leaves_requests_stuck(monkeypatch, fit, order, window):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(16, fit=fit, order=order, backfill_window=window)

    seen = _drive(kv, random.Random(SEED), ops=2000)

    assert seen["finished"] > 40, f"window {window}: too few requests ran to the end: {seen}"
    _finish(kv)


# ── the log ─────────────────────────────────────────────────────────────


def _log_rows(directory) -> list[dict]:
    import json

    from mstar.engine.resources.kv import admission_log

    admission_log.flush_all()
    return [
        json.loads(line)
        for path in sorted(directory.glob("admission_kv_pid*.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_pool_that_logs_writes_a_row_when_a_wait_changes_and_not_for_each_refusal(
    monkeypatch, tmp_path, fit, order,
):
    monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
    kv = _hog_and_queue(128, "abcdef", fit)
    for _ in range(4):
        for rid in "abcdef":
            _ready(kv, rid)
        # a page granted between passes moves the pages free, which is what makes a refusal asked again
        assert _run(kv, {"hog": PAGE_SIZE}, DECODE).ok
    rows = _log_rows(tmp_path)

    waits: dict[str, list[tuple]] = {}
    for row in rows:
        if row["event"] == "wait":
            waits.setdefault(row["rid"], []).append((row["is_head"], row["why"]))
    assert sorted(waits) == list("abcdef"), "a request that waited has no row of why"
    for rid, states in waits.items():
        again = [a == b for a, b in zip(states, states[1:], strict=False)]
        assert not any(again), f"{rid}: a row for a wait that had not changed"
    assert all(len(states) == 1 for states in waits.values())


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_request_is_timed_from_when_it_asked_and_not_from_when_the_window_reached_it(
    monkeypatch, tmp_path, fit, order,
):
    monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
    kv = _hog_and_queue(2, "ab", fit)
    kv.ingest_request("c", _request(list(range(9000, 9010)), 4))
    assert [_ready(kv, rid).ready for rid in "abc"] == [False] * 3
    time.sleep(0.05)
    kv.remove_request("a")

    assert _ready(kv, "c").ready
    reserve = next(r for r in _log_rows(tmp_path) if r["event"] == "reserve" and r["rid"] == "c")
    assert reserve["waited_s"] >= 0.05


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_pool_that_logs_decides_as_one_that_does_not(monkeypatch, tmp_path, fit, order):
    answers = {}
    for logged in (False, True):
        if logged:
            monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
        kv = _manager(24, fit=fit, order=order, backfill_window=16)
        answers[logged] = _queue_run(kv, SEED + 3, ops=1500)[0]

    assert answers[True] == answers[False]


# ── the padding rows ────────────────────────────────────────────────────


def test_what_the_padding_rows_keep_is_counted_out_of_the_capacity(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(32, fit="sum", order="fifo")
    assert kv._capacity() == 31

    kv.ingest_request(-1, _request(list(range(100)), 4))
    assert kv._alloc(-1, "main", 3 * PAGE_SIZE).success
    _ingest(kv, "real", 40, 4)
    assert kv._alloc("real", "main", 2 * PAGE_SIZE).success
    assert kv._capacity() == 31 - 3, "the padding row's pages are not the pool's to promise"

    kv.remove_request(-1)
    assert kv._capacity() == 31
