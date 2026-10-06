"""What it costs a pool that backfills to be asked about a long queue.

The scheduler asks readiness of every request still waiting, on every pass. Under
``backfill`` each one used to be worked out in full, and a pool with a thousand
requests waiting spent more time answering that than running them. The pool now
looks only at the first ``backfill_window`` of them, and says no to one the room
left rules out without working out its plan.

Here the cost is measured as the time of one pass, and the answers are checked
against the pool working every request in the window out in full: the cheap
refusals are a shortcut, and must never change who is admitted, or when.

``MSTAR_KV_BACKFILL_BENCH=1 pytest -s -k microbenchmark`` prints the per-pass cost
at a few queue lengths; it asserts nothing.
"""

from __future__ import annotations

import os
import random
import statistics
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
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig
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


# ── a pool under pressure ───────────────────────────────────────────────

POOL = 2048
HEADROOM = 8


def _pressed_pool(fit: str, reserved: int, waiting: int, seed: int = 7, **config):
    """``reserved`` requests admitted and prefilled, whose reservations leave
    ``HEADROOM`` pages of the pool and are only at the start of what they will
    decode, and ``waiting`` that run too long to fit in what is left, by the
    peak or by the sum.

    Returns the pool and the reserved rids.
    """
    rng = random.Random(seed)
    kv = _manager(POOL, fit=fit, order="backfill", **config)
    claim, extra = divmod(POOL - 1 - HEADROOM, reserved)
    held = []
    for i in range(reserved):
        # a prompt of two pages (three, for the few that take up what does not divide
        # evenly), and the rest of the reservation is decode, still to come, as long for all
        prompt = (2 + (i < extra)) * PAGE_SIZE - rng.randrange(0, PAGE_SIZE // 2)
        _ingest(kv, f"r{i}", prompt, (claim - 2) * PAGE_SIZE)
        assert _ready(kv, f"r{i}").ready, f"r{i} was not admitted into an empty pool"
        held.append((f"r{i}", prompt))
    for rid, prompt in held:
        assert _prefill(kv, rid, prompt).ok

    # a request that runs as long as the reserved ones, and so is there when they have all grown
    longest = (claim + 1) * PAGE_SIZE

    def arrive(j: int) -> str:
        _ingest(kv, f"w{j}", rng.randrange(80, 200), rng.randrange(longest, longest + 200))
        return f"w{j}"

    queue = [arrive(j) for j in range(waiting)]
    # let the pool settle: whatever fits is let in, and put back, so the queue is of what does not
    for settle in range(100):
        let_in = [rid for rid in queue if _ready(kv, rid).ready]
        if not let_in:
            break
        for rid in let_in:
            kv.remove_request(rid)
            queue.remove(rid)
        queue += [arrive(waiting + 1000 * (settle + 1) + k) for k in range(len(let_in))]
    else:
        raise AssertionError("the pool kept letting the queue in")
    assert len(kv._reserved) == reserved and len(kv._waiting) == waiting
    return kv, [rid for rid, _ in held]


def _one_pass(kv: KVManager, queue: list[str]) -> float:
    """The scheduler's pass over the queue: readiness for each, in arrival order. Microseconds."""
    started = time.perf_counter_ns()
    for rid in queue:
        kv._gate(rid, NODE, PREFILL)
    return (time.perf_counter_ns() - started) / 1e3


def _timed_passes(kv: KVManager, reserved: list[str], passes: int, seed: int = 1) -> list[float]:
    """Each pass follows a step that grants a page to one reserved request, as a busy pool's do."""
    rng = random.Random(seed)
    times = []
    for _ in range(passes):
        assert _run(kv, {rng.choice(reserved): PAGE_SIZE}).ok
        times.append(_one_pass(kv, list(kv._waiting)))
    return times


class _Counted:
    """Counts the calls to what a pool does in full for a request."""

    def __init__(self, kv: KVManager):
        self.full = 0
        self.ruled_out = 0
        admissible, ruled_out = kv._admissible, kv._ruled_out

        def counting_admissible(*args, **kwargs):
            self.full += 1
            return admissible(*args, **kwargs)

        def counting_ruled_out(*args, **kwargs):
            answer = ruled_out(*args, **kwargs)
            self.ruled_out += answer
            return answer

        kv._admissible, kv._ruled_out = counting_admissible, counting_ruled_out


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_pass_over_a_long_queue_costs_what_the_window_does(monkeypatch, fit, order):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    kv, reserved = _pressed_pool(fit, reserved=170, waiting=1500)
    window = kv._window_size
    counted = _Counted(kv)

    times = _timed_passes(kv, reserved, passes=5)

    median_ms = statistics.median(times) / 1e3
    print(
        f"\n{fit}+{order}: {len(kv._reserved)} reserved, {len(kv._waiting)} waiting, window {window}: "
        f"{median_ms:.2f} ms per pass (worst {max(times) / 1e3:.2f}); "
        f"{counted.full / 5:.0f} full evaluations and {counted.ruled_out / 5:.0f} refused cheaply per pass"
    )
    assert median_ms < 20, f"a pass over {len(kv._waiting)} waiting took {median_ms:.1f} ms"
    assert counted.full <= 5 * (window + 4), f"{counted.full} full evaluations in 5 passes"
    assert counted.ruled_out > 0, "nothing was refused from the room, so what is kept was not used"


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_pass_costs_no_more_for_a_longer_queue_than_the_window_covers(monkeypatch, fit, order):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    costs = {}
    for waiting in (300, 1500):
        kv, reserved = _pressed_pool(fit, reserved=64, waiting=waiting)
        counted = _Counted(kv)
        costs[waiting] = (statistics.median(_timed_passes(kv, reserved, passes=5)), counted.full)

    print(f"\n{fit}+{order}: per pass, by queue length: {costs}")
    # 5x the queue, and 128 of it in full either way: the rest costs a lookup each
    assert costs[1500][1] <= costs[300][1] + 4
    assert costs[1500][0] < 6 * costs[300][0] and costs[1500][0] < 20_000


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_default_window_looks_at_128(fit, order):
    assert _manager(16, fit=fit, order=order)._window_size == 128


# ── the answers are the pool's own ─────────────────────────────────────


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


def _same_answers(fit: str, order: str, window: int, seed: int) -> None:
    """The same run with the cheap refusals on and off: the same requests admitted at the same passes."""
    answers = {}
    for cheap in (True, False):
        kv = _manager(24, fit=fit, order=order, backfill_window=window)
        kv._cheap_refusals = cheap
        counted = _Counted(kv)
        log, seen = _queue_run(kv, SEED + seed, ops=2400)
        answers[cheap] = (log, seen, counted)

    (log_on, seen, on), (log_off, seen_off, off) = answers[True], answers[False]
    print(f"\n{fit}+{order} window {window} seed {seed}: {seen}; "
          f"full evaluations {on.full} with the cheap refusals, {off.full} without; "
          f"{on.ruled_out} requests refused cheaply")
    assert log_on == log_off, "the cheap refusals changed who was admitted, or when"
    assert seen == seen_off and seen["admitted"] > 40 and seen["finished"] > 20
    assert off.ruled_out == 0 and on.ruled_out > 0, "the run never refused a request cheaply"
    assert on.full < off.full


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
@pytest.mark.parametrize("window", [3, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_cheap_refusals_change_nobody_s_admission(monkeypatch, fit, order, window, seed):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    _same_answers(fit, order, window, seed)


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
@pytest.mark.parametrize("window", [3, 128])
def test_the_kept_counts_are_what_counting_afresh_gives_all_through_a_run(monkeypatch, fit, order, window):
    """With the debug assertions on, each use of what is kept is checked against counting afresh."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    _same_answers(fit, order, window, seed=7)


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_cheap_refusal_is_never_of_a_request_the_full_test_admits(monkeypatch, fit, order):
    """Each time it refuses, ask the full test too, on a pool that is left as it was."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(24, fit=fit, order=order, backfill_window=16)
    checked = []
    ruled_out = kv._ruled_out

    def checking(rid, need, hit):
        answer = ruled_out(rid, need, hit)
        if answer:
            kept = {
                name: dict(getattr(kv, name))
                for name in ("_refused", "_wait_state", "_asked_at")
            }
            head = kv._head()
            assert not kv._admissible(rid, head, need, hit), f"{rid} was refused, and fits"
            for name, was in kept.items():
                setattr(kv, name, was)
            checked.append(rid)
        return answer

    kv._ruled_out = checking
    _queue_run(kv, SEED + 9, ops=2400)

    assert len(checked) > 50


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


# ── the room ────────────────────────────────────────────────────────────


def test_the_room_is_kept_through_grants_and_taken_again_when_pages_come_free():
    kv = _manager(32, fit="sum", order="backfill")
    _ingest(kv, "a", prompt=100, max_tokens=60)
    assert _ready(kv, "a").ready and kv._reserved["a"].pages == 10
    assert kv._room() == 31 - 10

    # pages granted to a reserved request come off the supply and off what it is owed
    assert _prefill(kv, "a", 100).ok
    assert kv._room() == 31 - 10
    kept = kv._room_at
    assert kv._alloc("a", "main", 7 * PAGE_SIZE).success
    assert kv._room() == 31 - 10 and kv._room_at is kept, "a grant inside a reservation moved the room"

    # pages taken by something not counted are not: what is kept is above what there is, which
    # is safe, and a request it fails to rule out is refused by the full test
    kv.ingest_request("u", KVReqConfig())
    assert kv._alloc("u", "main", 3 * PAGE_SIZE).success
    assert kv._room() == 31 - 10 and kv._supply() - kv._outstanding() == 31 - 10 - 3
    assert kv._room_at is kept
    _ingest(kv, "big", prompt=100, max_tokens=60 + 12 * PAGE_SIZE)
    assert not _ready(kv, "big").ready

    # and taken again once they are given back, which is when it can have risen
    kv.remove_request("u")
    assert kv._room_at is kept and kv._room() == 31 - 10 and kv._room_at is not kept
    kv.remove_request("a")
    assert kv._room() == 31, "a released reservation did not return its room"


def test_the_room_falls_when_a_request_is_reserved():
    kv = _manager(32, fit="sum", order="backfill")
    _ingest(kv, "a", prompt=100, max_tokens=60)
    _ingest(kv, "b", prompt=40, max_tokens=24)
    assert kv._room() == 31
    assert _ready(kv, "a").ready and kv._room() == 21
    assert _ready(kv, "b").ready and kv._room() == 21 - 4


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


# ── the microbenchmark ──────────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("MSTAR_KV_BACKFILL_BENCH") != "1",
    reason="prints a table; MSTAR_KV_BACKFILL_BENCH=1 to run",
)
def test_microbenchmark_of_a_pass_over_the_queue(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    print("\nper pass over the queue (median of 7, each after a page grant):")
    for fit in ("sum", "peak"):
        for reserved in (64, 170):
            for waiting in (100, 500, 1500):
                kv, held = _pressed_pool(fit, reserved, waiting)
                times = _timed_passes(kv, held, passes=7)
                print(
                    f"  {fit:4s} reserved={len(kv._reserved):4d} waiting={len(kv._waiting):5d}: "
                    f"{statistics.median(times):10.0f} us"
                )
