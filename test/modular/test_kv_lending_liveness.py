"""Every request finishes on a pool where guaranteed requests lend what they haven't written (`KVManager._borrowable`).

A preemptible request that no drop can free and the host can't take whole may have to turn guaranteed
(`KVManager._cover_move`), which takes room no one lends, so only a request a drop can always take off the device
borrows. The drivers here run requests the way the worker does: a readiness check for every row on every pass, one step
for every ready row (or one per row), a refused step rolled back whole and its rows asked again, the refused row
preempted and its moves made, then `make_room`.
"""

from __future__ import annotations

import sys
from dataclasses import replace

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
from test_kv_admission import _RUNNERS, _remove, _runner
from test_kv_preemption import _drop, _make_room, _offload, _pool, _preempt, _ready, _request, _running
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _manager, _StubTransfer

from mstar.engine.resources.base import MoveKind
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import ASKS, Segment
from mstar.testing_utils.resource import StubResource


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _droppable(kv):
    return lambda rid: kv._overrides[rid].rebuildable and kv.check_drop(rid, kv._laid_out_tokens(rid)) is None


def _admit(kv, rows: dict[str, int]):
    """One step of ``rows``' tokens, committed when it is admitted and rolled back whole when it isn't."""
    step = KVStep(segments=tuple(Segment(rid, "main", tokens) for rid, tokens in rows.items()))
    ctx = replace(_ctx(*rows), graph_walk=WALK)
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    else:
        kv.rollback_admit(step, ctx)
    return outcome


def _move(kv, victims, owner: str, rows: dict[str, "_Row"]) -> bool:
    if victims is None:
        return False
    for rid in victims.to_host:
        _offload(kv, rid)
    for rid in victims.to_drop:
        _drop(kv, rid)
        # its rebuild writes what it had written, alone, before it decodes on
        rows[rid].rebuilding = True
    for rid in victims.to_ask:
        _runner(kv).readmit(rid, NODE)
    if victims.queued:
        _runner(kv).readmit(owner, NODE)
    return bool(victims.to_host or victims.to_drop or victims.to_ask or victims.queued)


class _Row:
    """A request's next step: its prompt (or its rebuild) first, then a page of decode per step."""

    def __init__(self, prompt_pages: int, decode_pages: int, written: int = 0, step: int = PAGE_SIZE):
        self.prompt, self.written, self.rebuilding = prompt_pages * PAGE_SIZE, written, False
        self.left, self.step = decode_pages * PAGE_SIZE // step, step

    def tokens(self) -> int:
        return self.written if self.rebuilding else self.prompt if not self.written else self.step


def _drive(kv, rows: dict[str, _Row], arrivals=(), batched: bool = True, passes: int = 600, decode: int = PAGE_SIZE,
           refuse=()) -> int:
    """Run ``rows`` (already running, oldest first) and ``arrivals``, (pass, rid, prompt pages, decode pages,
    guaranteed, rebuildable), decoding ``decode`` tokens a step, to the end, and return the passes it took. ``refuse``
    holds the (pass, rid) at which `KVManager.refuse_drops` marks a row. Raise once preempt answers a refusal with no
    move twice over the same state, or once no pass makes progress for a while."""
    arrivals, arrival, refuse = sorted(arrivals), len(rows), dict(refuse)
    last_empty, idle = None, 0
    for n in range(passes):
        while arrivals and arrivals[0][0] <= n:
            _, rid, prompt, decodes, guaranteed, rebuildable = arrivals.pop(0)
            _request(kv, rid, prompt, arrival, guaranteed=guaranteed, rebuildable=rebuildable,
                     decodes=decodes * PAGE_SIZE)
            rows[rid], arrival = _Row(prompt, decodes, step=decode), arrival + 1
        if refuse.get(n) in rows:
            kv.refuse_drops(refuse[n])
        if not rows and not arrivals:
            return n
        kv._cpu_pool.land()
        progressed = False
        ready = [rid for rid in rows if _ready(kv, rid)]
        rebuilds = [[rid] for rid in ready if rows[rid].rebuilding]
        steps = rebuilds + ([[rid for rid in ready if not rows[rid].rebuilding]] if batched else
                            [[rid] for rid in ready if not rows[rid].rebuilding])
        for step in [step for step in steps if step]:
            if not batched and not _ready(kv, step[0]):
                # a move an earlier row's refusal made this pass
                continue
            outcome = _admit(kv, {rid: rows[rid].tokens() for rid in step})
            if outcome.ok:
                progressed, last_empty = True, None
                for rid in step:
                    row = rows[rid]
                    if row.rebuilding:
                        kv.land_rebuild(rid)
                        row.rebuilding = False
                        continue
                    row.left -= bool(row.written)
                    row.written += row.tokens()
                    if not row.left:
                        _remove(kv, rid)
                        del rows[rid]
                continue
            rid = outcome.reason.request_id
            victims = _preempt(kv, rid, outcome.reason.pages_short, _droppable(kv))
            if victims is not None and not (victims.to_host or victims.to_drop or victims.to_ask or victims.queued):
                state = (rid, kv._room(), kv._borrowable, tuple((r, row.left) for r, row in rows.items()))
                assert state != last_empty, (
                    f"pass {n}: {rid} ran short again with nothing moved since preempt answered that the room "
                    f"covers its grant: room {kv._room()}, lent {kv._borrowable}, asks {kv._asks}"
                )
                last_empty = state
            progressed |= _move(kv, victims, rid, rows)
            if batched:
                break
        progressed |= _move(kv, _make_room(kv, _droppable(kv)), "make_room", rows)
        idle = 0 if progressed or arrivals else idle + 1
        assert idle < 30, (
            f"stalled at pass {n} with {({rid: row.left for rid, row in rows.items()})} left: asks {kv._asks}, "
            f"room {kv._room()}, lent {kv._borrowable}, kept {kv._kept} for {kv._keeper}, "
            f"request pages {kv.config.request_pages}"
        )
    raise AssertionError(f"not done after {passes} passes")


def _lender_and_borrower(host_pages: int = 1):
    """g, guaranteed, lends the 6 pages it hasn't written. p can't be dropped and outgrows what the host sets aside."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(host_pages)
    _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=6 * PAGE_SIZE)
    _running(kv, "p", 1, arrival=1, rebuildable=False, decodes=4 * PAGE_SIZE)
    return kv


def test_only_a_request_a_drop_can_always_take_off_the_device_borrows():
    kv = _lender_and_borrower(host_pages=8)
    _running(kv, "r", 1, arrival=2, decodes=2 * PAGE_SIZE)
    assert kv._borrowable == 6

    assert kv._room_for("p") == kv._room(), "p, which may have to turn guaranteed, borrows what g lends"
    assert kv._room_for("r") == kv._room() + kv._borrowable, "r, which a drop frees, doesn't borrow what g lends"
    assert kv._cpu_pool.reserve("p", kv._footprint("p"))
    assert kv._room_for("p") == kv._room(), "p borrows what g lends once the host sets its footprint aside"


@pytest.mark.parametrize("batched", [False, True])
def test_a_request_that_turns_guaranteed_asks_for_no_page_its_lender_lent(batched):
    """p1 ran short turning guaranteed and asked for its growth and the 6 pages g lent, 12 pages of a pool of 11. It
    moved itself to the host and never fit again."""
    kv = _manager(max_num_pages=12)
    kv.local_loops = {"decode"}
    kv._cpu_pool = host._HostPool(1)
    _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=6 * PAGE_SIZE)
    _running(kv, "p1", 1, arrival=1, rebuildable=False, decodes=3 * PAGE_SIZE)
    _running(kv, "p2", 1, arrival=2, rebuildable=False, decodes=3 * PAGE_SIZE)
    asks = []
    wait = kv._wait

    def waits(rid, pages):
        wait(rid, pages)
        asks.append((rid, kv._allotments[rid].asks_again, kv._footprint(rid)))
    kv._wait = waits

    _drive(kv, {"g": _Row(1, 6, PAGE_SIZE), "p1": _Row(1, 3, PAGE_SIZE), "p2": _Row(1, 3, PAGE_SIZE)}, batched=batched)

    assert asks and all(again <= footprint for _, again, footprint in asks), f"asked past the footprint: {asks}"


@pytest.mark.parametrize("order", [["g", "q", "p"], ["q", "g", "p"]])
@pytest.mark.parametrize("q_rebuildable", [True, False])
def test_a_borrower_ahead_in_the_step_doesnt_hold_off_a_request_turning_guaranteed(order, q_rebuildable):
    """p back from the host holds every page it grows into, and turns guaranteed once the room is at zero. q, ahead
    of it in the step, borrows a lent page first, and the step ran short on p again and again with nothing moved."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(1 if q_rebuildable else 5)
    _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=3 * PAGE_SIZE)
    _running(kv, "q", 1, arrival=1, rebuildable=q_rebuildable, decodes=2 * PAGE_SIZE)
    _running(kv, "p", 1, arrival=2, rebuildable=False, decodes=(4 if q_rebuildable else 3) * PAGE_SIZE)

    pages = {"g": 3, "q": 2, "p": 4 if q_rebuildable else 3}
    _drive(kv, {rid: _Row(1, pages[rid], PAGE_SIZE) for rid in order})


def test_a_follower_replays_the_guarantee_preempt_makes():
    """Rank 0 alone turns p guaranteed in `preempt`. A follower, from the same counts, replays it from the journal."""
    leader, follower = (_pool() for _ in range(2))
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(1)
        _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=3 * PAGE_SIZE)
        _running(kv, "q", 1, arrival=1, decodes=2 * PAGE_SIZE)
        _running(kv, "p", 1, arrival=2, rebuildable=False, decodes=4 * PAGE_SIZE)
    moves = []
    leader.journal = lambda kind, rid, count, victims=(): moves.append((kind, rid, count, list(victims)))
    for _ in range(20):
        outcomes = [_admit(kv, dict.fromkeys(["g", "q", "p"], PAGE_SIZE)) for kv in (leader, follower)]
        assert outcomes[0].ok == outcomes[1].ok
        if outcomes[0].ok:
            continue
        reason = outcomes[0].reason
        moves.clear()
        victims = _preempt(leader, reason.request_id, reason.pages_short, lambda rid: False)
        if (MoveKind.GUARANTEE, "p") in [(kind, rid) for kind, rid, *_ in moves]:
            break
        _preempt(follower, reason.request_id, reason.pages_short, lambda rid: False)
        for kv in (leader, follower):
            _move(kv, victims, reason.request_id, {})
            assert _ready(kv, "p")
    else:
        raise AssertionError("preempt never turned p guaranteed")

    for kind, rid, count, victims in moves:
        follower.replay(kind, rid, count, victims)

    assert "p" in follower._guaranteed
    assert follower._allotments["p"] == leader._allotments["p"]
    counts = [(kv._unused, kv._borrowable, kv._asks, kv._guaranteed) for kv in (leader, follower)]
    assert counts[0] == counts[1]
    assert all(_admit(kv, dict.fromkeys(["g", "q", "p"], PAGE_SIZE)).ok for kv in (leader, follower)), (
        "p ran short again once guaranteed"
    )


@pytest.mark.parametrize("allotted", [True, False], ids=["holding an allotment", "asking for one"])
def test_a_follower_stops_on_a_guarantee_for_a_request_it_already_guarantees(allotted):
    """Rank 0 journals a guarantee only for a request it doesn't guarantee yet, so a follower that already does has
    diverged from it. It raises, as for any move that doesn't fit its counts, rather than apply the move again."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(1)
    if allotted:
        _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=3 * PAGE_SIZE)
    else:
        _request(kv, "g", 1, 0, guaranteed=True, decodes=3 * PAGE_SIZE)
        assert kv.ask("g", NODE, WALK) is not None and "g" in kv._guaranteed
    allotment = kv._allotments.get("g")
    before = (allotment and replace(allotment), dict(kv._asks), kv._unused, kv._borrowable)

    with pytest.raises(RuntimeError, match="already guarantees"):
        kv.replay(MoveKind.GUARANTEE, "g", kv._footprint("g"), [])
    assert (allotment and replace(allotment), dict(kv._asks), kv._unused, kv._borrowable) == before


class _Refuses(StubResource):
    """A resource after the KV pool that asks for the requests in ``refuse`` and refuses each one's next reservation,
    so admission gives back what the KV pool reserved."""

    def __init__(self):
        self.refuse: set[str] = set()

    def ask(self, rid, node_name, graph_walk):
        return ASKS if rid in self.refuse else None

    def reserve(self, rid, node_name, graph_walk, published):
        if rid in self.refuse:
            self.refuse.discard(rid)
            return 0
        return None


def test_a_reservation_given_back_after_a_wait_for_no_page_keeps_the_allotment():
    """p waits holding its allotment for 0 pages, as `preempt` has a request wait that can't turn guaranteed until
    borrowers give back the room. Its reservation grows the allotment by nothing, and another resource refuses it."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(8)
    refuses = _Refuses()
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "zz": refuses}, node_resources={NODE: ["kv", "zz"]}))
    _running(kv, "g", 1, arrival=0, guaranteed=True, decodes=6 * PAGE_SIZE)
    _running(kv, "p", 1, arrival=1, rebuildable=False, decodes=4 * PAGE_SIZE)
    allotment = kv._allotments["p"]
    before = (allotment.pages, allotment.remaining, kv._unused, kv._borrowable)
    with kv._lock:
        kv._wait("p", 0)
    refuses.refuse.add("p")
    _runner(kv).readmit("p", NODE)

    assert not _ready(kv, "p")
    assert kv._allotments.get("p") is allotment, "giving back a reservation of no page gave back p's whole allotment"
    assert kv._asks["p"] == 0 and "p" in kv._stalled, "p no longer waits for the room borrowers took"
    assert _ready(kv, "p")
    assert (allotment.pages, allotment.remaining, kv._unused, kv._borrowable) == before


# configs that stalled on the over-ask or livelocked on an empty `Victims`, (pool, host, rows) with rows
# (rid, prompt pages, decode pages, guaranteed, rebuildable, arrival pass)
_TABLE = [
    (11, 1, [("g", 1, 6, True, False, 0), ("p1", 1, 3, False, False, 0), ("p2", 1, 3, False, False, 0)]),
    (11, 1, [("p", 1, 5, False, False, 0), ("g", 1, 6, True, False, 1), ("g2", 1, 2, True, False, 3)]),
    (9, 1, [("g", 1, 7, True, False, 0), ("p", 1, 3, False, False, 0)]),
    (10, 3, [("p0", 1, 1, False, False, 4), ("p1", 1, 4, False, False, 4), ("p2", 3, 2, False, False, 1),
             ("g3", 3, 3, True, False, 1)]),
    (8, 2, [("p0", 1, 3, False, False, 3), ("g1", 2, 2, True, False, 1), ("p2", 2, 2, False, False, 2),
            ("g3", 2, 4, True, False, 2), ("g4", 2, 3, True, False, 0), ("g5", 2, 4, True, False, 4)]),
    (11, 1, [("p0", 2, 1, False, False, 3), ("p1", 1, 5, False, False, 1), ("p2", 3, 4, False, False, 3),
             ("p3", 2, 7, False, False, 0), ("p4", 3, 2, False, False, 0), ("p5", 1, 2, False, False, 4)]),
]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pool,host_pages,rows", _TABLE)
def test_requests_arriving_over_passes_all_finish(pool, host_pages, rows, batched):
    kv = _manager(max_num_pages=pool + 1)
    kv.local_loops = {"decode"}
    kv._cpu_pool = host._HostPool(host_pages)

    _drive(kv, {}, [(at, rid, prompt, decodes, g, rb) for rid, prompt, decodes, g, rb, at in rows], batched)


@pytest.mark.parametrize("batched", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "open liveness gap, unreachable while no RebuildPolicy sets pad_ids: a borrower that loses its drops "
        "(`KVManager.refuse_drops`, world size 1) holding lent pages it can't turn guaranteed past waits on room "
        "only its lenders could give back"
    ))),
])
def test_a_borrower_refused_its_drops_holding_lent_pages(batched):
    """b1 holds 4 lent pages when it loses its drops, with no host."""
    kv = _manager(max_num_pages=26)
    kv.local_loops = {"decode"}
    kv._cpu_pool = host._HostPool(0)
    rows = [(0, "b1", 3, 5, False, True), (0, "g3", 3, 6, False, False), (0, "g5", 3, 7, False, False),
            (5, "r0", 1, 2, False, True)]

    _drive(kv, {}, rows, batched, decode=PAGE_SIZE // 2, refuse=[(9, "b1")])
