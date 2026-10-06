"""The plan kept as arrays (`PlanTable`) is the plan made from every stream, answer for answer.

A plan used to be made afresh from each admitted request's streams whenever the pool's
counters moved, which is nearly every decision under load. It is now made from a row per
request that is read again only for the requests something touched since (`_plan_touch`),
and from the committed lengths, which are read every time. That is a shortcut, and must
never change who is admitted, or when: the same run with it on and off is the same log, and
with the debug assertions on, each rebuild is checked against counting every request afresh.

``MSTAR_KV_PLAN_CACHE=0`` is the off switch, read once when the manager module is loaded.
"""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
from test_kv_admission_peak import (
    DECODE,
    NODE,
    PAGE_SIZE,
    PREFILL,
    _finish,
    _prefill,
    _ready,
    _request,
    _run,
    _StubTransfer,
)
from test_kv_admission_peak import (
    _manager as _peak_manager,
)
from test_kv_backfill_cost import _manager, _queue_run

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep
from mstar.engine.resources.kv.cpu_page_pool import OffloadedStream
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PublishedKVInfo
from mstar.engine.resources.kv.plan_table import ABSENT, PlanTable
from mstar.engine.resources.step import Segment, StepContext

SEED = 20261006
FITS_AND_ORDERS = [("peak", "backfill"), ("sum", "fifo"), ("sum", "backfill"), ("peak", "fifo")]
# the ones that plan: the summed test in arrival order has no plan to keep
PLANNED = {("peak", "backfill"), ("sum", "backfill"), ("peak", "fifo")}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    for name in (
        "MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW",
        "MSTAR_STEP_TELEMETRY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


# ── the answers are the pool's own ─────────────────────────────────────


class _Decisions:
    """Everything a pool answers that a plan is behind, in the order it was asked."""

    def __init__(self, kv: KVManager):
        self.log: list = []
        self.rebuilds = 0
        for name in ("_ruled_out", "_admissible", "_defer_grant", "_head_shadow"):
            kv.__dict__[name] = self._recording(name, getattr(kv, name))
        plan_state = kv._plan_state

        def counting():
            before = kv._plan
            state = plan_state()
            self.rebuilds += state is not before
            return state

        kv._plan_state = counting

    def _recording(self, name, method):
        def recording(*args, **kwargs):
            answer = method(*args, **kwargs)
            if name == "_defer_grant":
                answer = None if answer is None else (answer.request_id, answer.label)
            self.log.append((name, args[0], answer))
            return answer

        return recording


def _same_answers(fit: str, order: str, window: int, seed: int) -> dict:
    """The same run with the plan kept and without: the same decisions, all the way through."""
    runs = {}
    for cached in (True, False):
        saved = manager_mod._PLAN_CACHE
        manager_mod._PLAN_CACHE = cached
        try:
            kv = _manager(24, fit=fit, order=order, backfill_window=window)
        finally:
            manager_mod._PLAN_CACHE = saved
        assert kv._plan_cache == (cached and (fit, order) in PLANNED)
        decisions = _Decisions(kv)
        log, seen = _queue_run(kv, SEED + seed, ops=2400)
        runs[cached] = (log, seen, decisions)

    (log_on, seen, on), (log_off, seen_off, off) = runs[True], runs[False]
    print(f"\n{fit}+{order} window {window} seed {seed}: {seen}; {len(on.log)} decisions, "
          f"{on.rebuilds} plans built")
    assert log_on == log_off, "keeping the plan changed who was admitted, or when"
    assert on.log == off.log, "keeping the plan changed an answer on the way"
    assert seen == seen_off and seen["admitted"] > 40 and seen["finished"] > 20
    assert on.rebuilds == off.rebuilds
    return seen


@pytest.mark.parametrize(("fit", "order"), FITS_AND_ORDERS)
@pytest.mark.parametrize("window", [3, 128])
@pytest.mark.parametrize("seed", [0, 1])
def test_keeping_the_plan_changes_nobody_s_admission(monkeypatch, fit, order, window, seed):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    _same_answers(fit, order, window, seed)


@pytest.mark.parametrize(("fit", "order"), FITS_AND_ORDERS)
def test_every_plan_built_is_what_counting_afresh_gives_all_through_a_run(monkeypatch, fit, order):
    """With the debug assertions on, each rebuild checks every row, and the order of the rows."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    _same_answers(fit, order, window=128, seed=7)


@pytest.mark.parametrize(("fit", "order"), sorted(PLANNED))
@pytest.mark.parametrize("seed", [2, 3])
def test_every_row_is_what_counting_afresh_gives_after_every_call_of_a_run(monkeypatch, fit, order, seed):
    """Not only where a plan is built: after each call, with nothing left touched."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    kv = _manager(24, fit=fit, order=order, backfill_window=128)
    assert kv._plan_cache
    _audited(kv)
    _, seen = _queue_run(kv, SEED + seed, ops=2400)
    print(f"\n{fit}+{order} seed {seed}: {seen}")
    assert seen["admitted"] > 40 and seen["finished"] > 20


def test_the_plan_is_kept_only_where_there_is_one_to_keep(monkeypatch):
    kv = _manager(24, fit="sum", order="fifo")
    assert not kv._plan_cache
    monkeypatch.setattr(manager_mod, "_PLAN_CACHE", False)
    assert not _manager(24, fit="peak", order="backfill")._plan_cache
    monkeypatch.setattr(manager_mod, "_PLAN_CACHE", True)
    assert _manager(24, fit="peak", order="backfill")._plan_cache


# ── the table, against rows kept in a dict ────────────────────────────


def _model_fields(rows: dict) -> tuple[list[list[int]], int, int]:
    """What `PlanTable.fields` is, from the definition of `_plan_entry`: a column per field."""
    columns: list[list[int]] = [[], [], [], [], []]
    for held, claim, now, decoding in rows.values():
        growth = rounds = 0
        for stream, prompt, decode in decoding:
            left = decode - max(0, stream.stored_len - prompt)
            if left > 0:
                growth += 1
                rounds = max(rounds, left)
        for column, value in zip(columns, (held, claim, now, growth, rounds), strict=True):
            column.append(value)
    held_total = sum(held for held, *_ in rows.values())
    need_total = sum(max(0, claim - held) for held, claim, *_ in rows.values())
    return columns, held_total, need_total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_table_is_its_rows_through_any_run_of_rows_added_set_and_dropped(seed):
    rng = random.Random(SEED + seed)
    table = PlanTable()
    rows: dict[str, tuple] = {}
    streams = [SimpleNamespace(stored_len=0) for _ in range(40)]
    widest = most = 0
    for i in range(1500):
        roll = rng.random()
        if roll < 0.35 or not rows:
            rid = f"r{i}"
            table.append(rid)
            rows[rid] = (0, 0, 0, [])
            assert len(table.with_spare()[0]) == len(table) + 1 == len(table.with_spare()[1])
            most = max(most, len(table))
        elif roll < 0.55:
            rid = rng.choice(list(rows))
            table.drop(rid)
            del rows[rid]
        elif roll < 0.8:
            rid = rng.choice(list(rows))
            held = rng.randrange(0, 20)
            decoding = [
                (rng.choice([ABSENT, *streams]), rng.randrange(0, 60), rng.randrange(1, 60))
                for _ in range(rng.choice([0, 1, 1, 2, 4]))
            ]
            widest = max(widest, len(decoding))
            rows[rid] = (held, held + rng.randrange(-2, 20), rng.randrange(0, 9), decoding)
            table.set(rid, *rows[rid])
        else:
            rng.choice(streams).stored_len = rng.randrange(0, 150)
        assert table.rids == list(rows)
        columns, held_total, need_total = _model_fields(rows)
        assert [column.tolist() for column in table.fields()] == columns, f"op {i}"
        assert (table.held_total, table.need_total) == (held_total, need_total), f"op {i}"
        held, need = table.with_spare()
        assert held[:len(rows)].tolist() == columns[0]
        assert need[:len(rows)].tolist() == [max(0, c - h) for h, c in zip(columns[0], columns[1], strict=True)]
        # a candidate is put in the slot past the rows, and nothing of theirs moves
        held[len(rows)], need[len(rows)] = 99, 99
        assert [column.tolist() for column in table.fields()] == columns, f"op {i}"
    assert most > 130 and widest == 4


# ── what each row is made of moves only where it is touched ───────────


def _fresh(kv: KVManager) -> None:
    """Rebuild as a plan does, then say what counting every request afresh says."""
    kv._plan_fields()
    kv._assert_plan_table(kv._plan_table)
    assert not kv._plan_dirty


# what the scheduler and the worker call: after each, the rows must be as counting afresh has them
CALLS = (
    "ingest_request", "admit_retrieve", "resolve_cached_prefix", "apply_cached_prefix", "admit",
    "plan", "commit", "protect_prefix", "release_oldest", "reset_request", "clear_preplan",
    "remove_request", "offload", "reload",
)


def _audited(kv: KVManager) -> KVManager:
    """``kv``, checking its rows against counting afresh after every call that moves a page.

    Nothing is left touched when it looks, so a row is right only if the call that moved it said
    so, and not because another call before the next plan did.
    """
    def checking(method):
        def call(*args, **kwargs):
            answer = method(*args, **kwargs)
            _fresh(kv)
            return answer

        return call

    for name in CALLS:
        kv.__dict__[name] = checking(getattr(kv, name))
    return kv


def _pool() -> KVManager:
    kv = _manager(64, fit="peak", order="backfill")
    assert kv._plan_cache
    return _audited(kv)


def _admit(kv: KVManager, rid: str, tokens: list[int], max_tokens: int) -> None:
    kv.ingest_request(rid, _request(tokens, max_tokens))
    assert _ready(kv, rid).ready, f"{rid} was not admitted into a pool with room"
    _fresh(kv)


def _ctx(*rids: str, walk: str = DECODE, preplan: bool = False) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False, is_preplan=preplan,
    )


def test_a_row_is_made_as_a_request_reserves_and_goes_as_it_is_removed():
    kv = _pool()
    for i in range(4):
        _admit(kv, f"r{i}", list(range(1000 * i, 1000 * i + 30)), 40)
        assert kv._plan_table.rids == list(kv._reserved)

    kv.remove_request("r1")
    assert kv._plan_table.rids == ["r0", "r2", "r3"] == list(kv._reserved)
    _fresh(kv)
    _admit(kv, "r4", list(range(7000, 7020)), 10)
    assert kv._plan_table.rids == ["r0", "r2", "r3", "r4"] == list(kv._reserved)
    for rid in list(kv._reserved):
        kv.remove_request(rid)
        _fresh(kv)
    assert kv._plan_table.rids == [] and kv._plan_table.held_total == 0 == kv._plan_table.need_total


def test_a_grant_moves_the_row():
    kv = _pool()
    _admit(kv, "a", list(range(0, 40)), 200)
    assert _prefill(kv, "a", 40).ok
    _fresh(kv)
    for _ in range(8):
        assert _run(kv, {"a": PAGE_SIZE}).ok
        _fresh(kv)
    assert _run(kv, {"a": 3}).ok
    _fresh(kv)


def _leased(kv: KVManager, tokens: list[int], rid: str = "second") -> str:
    """``rid``, behind a request that ran on the same tokens, and so leased its pages at the gate."""
    _admit(kv, "first", tokens, 20)
    assert _prefill(kv, "first", len(tokens)).ok
    _finish(kv)
    kv.ingest_request(rid, _request(tokens, 20))
    assert _ready(kv, rid).ready
    assert kv._streams[rid]["main"].lease, "the second request found nothing in the index"
    return rid


def test_a_lease_from_a_request_that_came_first_moves_the_row():
    """The second request finds the first one's pages in the index and leases them."""
    kv = _pool()
    tokens = list(range(0, 5 * PAGE_SIZE + 3))
    rid = _leased(kv, tokens)
    stream = kv._streams[rid]["main"]

    assert _prefill(kv, rid, len(tokens)).ok

    assert stream.hits > 0
    # the pages it leased are not pages it has to take
    assert kv._reserved[rid].pages == 7 - stream.hits
    assert _run(kv, {rid: 4}).ok


def test_a_lease_the_model_takes_less_of_moves_the_row():
    kv = _pool()
    tokens = list(range(0, 5 * PAGE_SIZE + 3))
    rid = _leased(kv, tokens)
    stream = kv._streams[rid]["main"]
    claim = kv._reserved[rid].pages
    leased = len(stream.lease)

    kv.apply_cached_prefix(rid, NODE, PREFILL, None, 2 * PAGE_SIZE)

    assert len(stream.lease) == 2 and kv._reserved[rid].pages == claim + leased - 2
    assert _prefill(kv, rid, len(tokens) - 2 * PAGE_SIZE).ok


def test_a_lease_a_step_was_not_cut_to_moves_the_row():
    """The gate took a lease and prepare never cut the inputs to it: the step writes from 0."""
    kv = _pool()
    tokens = list(range(0, 5 * PAGE_SIZE + 3))
    rid = _leased(kv, tokens)
    assert kv._streams[rid]["main"].gate_lease

    assert _run(kv, {rid: len(tokens)}, PREFILL).ok

    assert kv._streams[rid]["main"].hits == 0 and not kv._streams[rid]["main"].lease


def _published(generation: int) -> PublishedKVInfo:
    """Another worker's 100-token ``main``, at its reset ``generation``."""
    return PublishedKVInfo.build_for_rank(rank=0, world_size=1, seq_info={
        "main": KVSequenceInfo(
            seq_len=100, latest_kv_transfer_info="remote",
            page_indices=list(range(7)), reset_generation=generation,
        ),
    })


def test_a_stream_read_from_another_worker_moves_the_row():
    """It takes what the index has of the stream, and gives that back when the producer rewinds."""
    kv = _pool()
    tokens = list(range(100))
    _admit(kv, "a", tokens, 8)
    assert _prefill(kv, "a", 100).ok
    kv.remove_request("a")
    kv.ingest_request("b", _request(tokens, 8))

    assert kv.admit_retrieve("b", NODE, DECODE, _published(1)).ok
    assert kv._streams["b"]["main"].hits, "the read took nothing from the index"
    kv.admit_retrieve("b", NODE, DECODE, _published(2))


class _HostPool:
    """What the manager asks of the host pool, with no copies: the streams it was given."""

    def __init__(self):
        self.offloaded = {}

    def is_offloaded(self, rid):
        return bool(self.offloaded.get(rid))

    def labels(self, rid):
        return list(self.offloaded.get(rid, {}))

    def num_pages(self, rid, label):
        return len(self.offloaded[rid][label].cpu_page_indices)

    def offload_stream(
        self, rid, label, gpu_kv_cache, gpu_page_indices, stored_len, position, released,
        protected_prefix,
    ):
        del gpu_kv_cache
        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            list(gpu_page_indices), stored_len, position, released, protected_prefix,
        )
        return True

    def reload_stream(self, rid, label, gpu_kv_cache, gpu_page_indices):
        del gpu_kv_cache, gpu_page_indices
        state = self.offloaded[rid].pop(label)
        if not self.offloaded[rid]:
            del self.offloaded[rid]
        return state

    def discard(self, rid, label):
        self.offloaded.get(rid, {}).pop(label, None)

    def sync(self):
        pass

    def remove_request(self, rid):
        self.offloaded.pop(rid, None)


def test_an_offload_and_a_reload_move_the_row():
    kv = _pool()
    kv._cpu_pool = _HostPool()
    _admit(kv, "a", list(range(0, 50)), 40)
    assert _prefill(kv, "a", 50).ok
    assert _run(kv, {"a": 5}).ok
    held = kv._held_fresh("a")
    assert held > 0

    assert kv.offload("a") == held
    assert kv._held_fresh("a") == 0

    assert kv.reload("a")
    assert kv._held_fresh("a") == held
    assert _run(kv, {"a": 5}).ok


def test_a_reset_moves_the_row():
    kv = _pool()
    _admit(kv, "a", list(range(0, 50)), 30)
    assert _prefill(kv, "a", 50).ok
    assert _run(kv, {"a": 5}).ok
    _fresh(kv)
    kv.reset_request("a")
    _fresh(kv)
    kv.reset_request("a", free=True)
    _fresh(kv)


def test_a_release_moves_the_row():
    kv = _pool()
    _admit(kv, "a", list(range(0, 8 * PAGE_SIZE)), 30)
    assert _prefill(kv, "a", 8 * PAGE_SIZE).ok
    _fresh(kv)
    kv.protect_prefix("a", 2 * PAGE_SIZE, label="main")
    assert kv.release_oldest("a", 3 * PAGE_SIZE, label="main") == 3 * PAGE_SIZE
    _fresh(kv)
    assert _run(kv, {"a": 3}).ok
    _fresh(kv)


def test_a_retention_policy_moves_the_row():
    from mstar.engine.resources.kv.config import RetentionPolicy

    kv = _pool()
    _admit(kv, "a", list(range(0, 4 * PAGE_SIZE)), 200)
    assert _prefill(kv, "a", 4 * PAGE_SIZE).ok
    policy = RetentionPolicy(context_budget=2 * PAGE_SIZE, protected_prefix=PAGE_SIZE)
    for _ in range(6):
        step = KVStep(segments=(Segment("a", "main", PAGE_SIZE),), retention={("a", "main"): policy})
        ctx = _ctx("a")
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)
        _fresh(kv)
    assert kv._streams["a"]["main"].released > 0, "the policy released nothing"


def test_a_step_planned_ahead_and_abandoned_moves_the_row():
    """A pre-plan opens a label the request had not got; clearing it takes the label away again."""
    kv = _pool()
    _admit(kv, "a", list(range(0, 40)), 60)
    assert _prefill(kv, "a", 40).ok
    _fresh(kv)
    step = KVStep(segments=(Segment("a", "side", PAGE_SIZE),))
    ctx = _ctx("a", preplan=True)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    _fresh(kv)
    kv.clear_preplan()
    _fresh(kv)


def test_a_row_that_was_not_touched_is_caught_by_the_check(monkeypatch):
    """The check is on the touches: a page that moves without saying so is a row that is wrong."""
    kv = _pool()
    _admit(kv, "a", list(range(0, 40)), 60)
    assert _prefill(kv, "a", 40).ok
    _fresh(kv)

    kv._arena.retain(kv._streams["a"]["main"].page_indices[:1])
    kv._streams["a"]["main"].page_indices.append(kv._arena.acquire(1)[0])

    with pytest.raises(AssertionError, match="did not say so"):
        _fresh(kv)


def test_what_follows_the_committed_length_is_read_every_time():
    """No touch is made as a request decodes a token: the rounds it has left are not kept."""
    kv = _pool()
    _admit(kv, "a", list(range(0, 40)), 60)
    assert _prefill(kv, "a", 40).ok
    _fresh(kv)
    kv._plan_state()
    before = kv._plan.planner._rounds.tolist()
    assert _run(kv, {"a": 1}).ok
    kv._plan_dirty.clear()
    # a token is not a page: the row is not touched, and the next plan counts it
    assert kv._streams["a"]["main"].stored_len == 41
    kv._reserved_epoch += 1
    after = kv._plan_state().planner._rounds.tolist()
    assert after == [before[0] - 1]
    _fresh(kv)


LABELS = ("main", "cfg_text", "cfg_img")


def _labelled_run(kv: KVManager, seed: int, ops: int) -> tuple[list, dict]:
    """Requests on one to three labels, some that decode on none, on some, or on all of them,
    written at their own pace, asked about in passes and now and then aborted: rows that are of
    different widths, and finished requests that wait to be removed. Returns the log of what decided who is
    admitted, and what the run saw."""
    rng = random.Random(seed)
    queue: list[str] = []
    shapes: dict[str, dict[str, int]] = {}
    admitted: set[str] = set()
    left: dict[str, dict[str, int]] = {}
    log: list = []
    seen = dict(admitted=0, finished=0, aborted=0, deferred=0, widest=0, undecoded=0)

    def write(rid: str, spans: dict[str, int], walk: str) -> bool:
        step = KVStep(segments=tuple(Segment(rid, label, span) for label, span in spans.items()))
        outcome = kv.admit(step, _ctx(rid, walk=walk))
        if outcome.ok:
            kv.plan(step, _ctx(rid, walk=walk))
            kv.commit(step, _ctx(rid, walk=walk))
        else:
            seen["deferred"] += 1
        log.append((walk, rid, tuple(spans.items()), outcome.ok))
        return outcome.ok

    def remove(rid: str) -> None:
        for group in (admitted, left, shapes):
            group.discard(rid) if isinstance(group, set) else group.pop(rid, None)
        if rid in queue:
            queue.remove(rid)
        kv.remove_request(rid)

    for i in range(ops):
        roll = rng.random()
        if roll < 0.25 and len(queue) < 40:
            rid = f"m{i}"
            labels = ["main", *rng.sample(LABELS[1:], rng.randrange(0, 3))]
            slots = {label: rng.randrange(5, 60) for label in labels}
            decoding = [label for label in labels if rng.random() < 0.6]
            seen["undecoded"] += not decoding
            seen["widest"] = max(seen["widest"], len(decoding))
            kv.ingest_request(rid, KVReqConfig(
                max_tokens=rng.randrange(1, 6 * PAGE_SIZE), needed_labels=labels,
                prompt_slots=slots, decode_labels=decoding,
            ))
            queue.append(rid)
            shapes[rid] = slots
            left[rid] = {label: 0 for label in labels}
            left[rid].update({label: kv._overrides[rid].max_tokens for label in decoding})
        elif roll < 0.45 and queue:
            asked = list(queue)
            rng.shuffle(asked)
            let_in = [rid for rid in asked if _ready(kv, rid).ready]
            for rid in let_in:
                queue.remove(rid)
                admitted.add(rid)
            seen["admitted"] += len(let_in)
            log.append(("pass", tuple(asked), tuple(let_in), tuple(kv._reserved)))
        elif roll < 0.6 and admitted:
            rid = rng.choice(sorted(admitted))
            if write(rid, shapes[rid], PREFILL):
                admitted.discard(rid)
        elif roll < 0.9 and left:
            live = sorted(set(left) - admitted - set(queue))
            if not live:
                continue
            rid = rng.choice(live)
            open_labels = [label for label, n in left[rid].items() if n]
            if not open_labels:
                remove(rid)
                seen["finished"] += 1
                log.append(("finished", rid))
                continue
            spans = {
                label: rng.randrange(1, min(left[rid][label], 2 * PAGE_SIZE) + 1)
                for label in rng.sample(open_labels, rng.randrange(1, len(open_labels) + 1))
            }
            if write(rid, spans, DECODE):
                for label, span in spans.items():
                    left[rid][label] -= span
        elif queue or admitted or left:
            rid = rng.choice(sorted(set(queue) | admitted | set(left)))
            remove(rid)
            seen["aborted"] += 1
            log.append(("abort", rid))
        kv.assert_pages_conserved()
    return log, seen


def _same_labelled_answers(fit: str, order: str, seed: int, audited: bool = False) -> dict:
    runs = {}
    for cached in (True, False):
        saved = manager_mod._PLAN_CACHE
        manager_mod._PLAN_CACHE = cached
        try:
            kv = _manager(32, fit=fit, order=order)
        finally:
            manager_mod._PLAN_CACHE = saved
        if audited and cached:
            _audited(kv)
        decisions = _Decisions(kv)
        log, seen = _labelled_run(kv, SEED + seed, ops=2400)
        runs[cached] = (log, seen, decisions)
    (log_on, seen, on), (log_off, seen_off, off) = runs[True], runs[False]
    print(f"\n{fit}+{order} seed {seed}: {seen}; {len(on.log)} decisions, {on.rebuilds} plans built")
    assert log_on == log_off, "keeping the plan changed who was admitted, or when"
    assert on.log == off.log, "keeping the plan changed an answer on the way"
    assert seen == seen_off
    assert seen["admitted"] > 30 and seen["finished"] > 10 and seen["widest"] == 3
    assert seen["undecoded"] > 0
    # only the peak fit refuses a grant that would leave the admitted requests unable to finish
    assert seen["deferred"] > 0 or fit == "sum"
    return seen


@pytest.mark.parametrize(("fit", "order"), sorted(PLANNED))
@pytest.mark.parametrize("seed", [0, 1])
def test_requests_on_several_labels_are_admitted_as_they_are_without_the_plan_kept(
    monkeypatch, fit, order, seed,
):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    _same_labelled_answers(fit, order, seed, audited=True)


@pytest.mark.parametrize(("fit", "order"), sorted(PLANNED))
def test_requests_on_several_labels_are_checked_at_every_plan_built(monkeypatch, fit, order):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    _same_labelled_answers(fit, order, seed=5)


def test_a_pre_fork_planned_ahead_and_abandoned_moves_the_row():
    """A pre-plan forks onto a label the request had not opened; clearing it takes that stream away."""
    kv = _manager(32, fit="peak", order="backfill")
    _audited(kv)
    labels = ["main", "cfg_text"]
    kv.ingest_request("a", KVReqConfig(
        max_tokens=2 * PAGE_SIZE, needed_labels=labels,
        prompt_slots=dict.fromkeys(labels, 2 * PAGE_SIZE), decode_labels=["main"],
    ))
    assert _ready(kv, "a").ready
    grow = KVStep(segments=(Segment("a", "main", 2 * PAGE_SIZE),))
    assert kv.admit(grow, _ctx("a", walk=PREFILL)).ok
    kv.plan(grow, _ctx("a", walk=PREFILL))
    kv.commit(grow, _ctx("a", walk=PREFILL))

    step = KVStep(segments=(Segment("a", "main", 4),), pre_forks=(("main", "cfg_text"),))
    ahead = _ctx("a", preplan=True)
    assert kv.admit(step, ahead).ok
    kv.plan(step, ahead)
    assert "cfg_text" in kv._streams["a"]
    kv.clear_preplan()
    assert "cfg_text" not in kv._streams["a"]

    # and promoted, the stream stays
    assert kv.admit(step, ahead).ok
    kv.plan(step, ahead)
    assert kv.admit(step, _ctx("a")).ok
    kv.plan(step, _ctx("a"))
    kv.commit(step, _ctx("a"))
    assert "cfg_text" in kv._streams["a"]


def test_a_pool_that_reserves_at_its_first_step_has_a_row_as_it_does(monkeypatch):
    """Rank 1 learns of a request at its first step, and reserves what rank 0 admitted."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    script = [("c", 4 * PAGE_SIZE, PREFILL), ("a", PAGE_SIZE, PREFILL), ("b", PAGE_SIZE, PREFILL)]
    script += [(rid, PAGE_SIZE, DECODE) for _ in range(12) for rid in ("a", "b")]
    verdicts = {}
    for cached in (True, False):
        monkeypatch.setattr(manager_mod, "_PLAN_CACHE", cached)
        follower = _peak_manager(21, fit="peak", order="fifo", rank=1, world_size=2)
        assert follower._plan_cache == cached
        if cached:
            _audited(follower)
        for rid, prompt, max_tokens in (
            ("c", 4 * PAGE_SIZE, 2 * PAGE_SIZE), ("a", PAGE_SIZE, 9 * PAGE_SIZE),
            ("b", PAGE_SIZE, 9 * PAGE_SIZE),
        ):
            base = {"c": 0, "a": 1000, "b": 2000}[rid]
            follower.ingest_request(rid, _request(list(range(base, base + prompt)), max_tokens))
        got = []
        for rid, span, walk in script:
            step = KVStep(segments=(Segment(rid, "main", span),))
            ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
            outcome = follower.admit(step, ctx)
            if outcome.ok:
                follower.plan(step, ctx)
                follower.commit(step, ctx)
            got.append(None if outcome.ok else type(outcome.reason).__name__)
        verdicts[cached] = got
        assert list(follower._reserved) == ["c", "a", "b"]
        if cached:
            assert follower._plan_table.rids == ["c", "a", "b"]
    assert verdicts[True] == verdicts[False]
    assert any(v is not None for v in verdicts[True]), "the guard never fired"


# ── a grant that leaves the admitted requests safe is not looked into ──


def test_a_grant_the_free_pages_cover_is_never_deferred_by_the_full_test(monkeypatch):
    """Where the shortcut answers, so does counting every request: each answer is checked against it."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    kv = _manager(24, fit="peak", order="backfill")
    defer = kv._defer_grant
    seen = dict(shortcut=0, full=0, deferred=0)

    def checking(rid, label, n):
        reserved = kv._reserved.get(rid)
        covered = reserved is not None and max(
            0, reserved.pages - kv._held_fresh(rid), n,
        ) <= kv._supply()
        answer = defer(rid, label, n)
        kv._plan_cache = False
        try:
            full = defer(rid, label, n)
        finally:
            kv._plan_cache = True
        assert (answer is None) == (full is None), f"{rid}: the shortcut and the full test disagree"
        seen["shortcut" if covered else "full"] += 1
        seen["deferred"] += answer is not None
        if covered:
            assert answer is None
        return answer

    kv._defer_grant = checking
    _queue_run(kv, SEED + 3, ops=2400)
    print(f"\n{seen}")
    assert seen["shortcut"] > 20 and seen["full"] > 20 and seen["deferred"] > 0
