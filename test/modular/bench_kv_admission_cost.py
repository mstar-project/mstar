"""What one KV admission decision costs, on a pool that is under pressure, on CPU.

Not a test (pytest does not collect it). Run from the repository root:

    .venv/bin/python test/modular/bench_kv_admission_cost.py                 # the table
    .venv/bin/python test/modular/bench_kv_admission_cost.py --profile       # cProfile of one config
    .venv/bin/python test/modular/bench_kv_admission_cost.py --audit         # what moves owner_changes
    .venv/bin/python test/modular/bench_kv_admission_cost.py --per-token     # every request one token a pass
    .venv/bin/python test/modular/bench_kv_admission_cost.py --diff A.json B.json   # same decisions?

The pool is the one a BAGEL run leaves: 2048 pages of 128 tokens, the prefix index
holding about 1500 of them from many short independent prompts (most evictable), the
free list near zero, and the admitted requests' reservations covering what is left.
``n_reserved`` requests decode, a page at a time, and a queue of ``n_waiting`` keeps
asking. Between two passes over the queue the admitted requests are granted pages,
which evict cached pages once the free list is empty, and now and then one finishes
(early, as a stop token does) and gives its reservation back, which is when the
queue gets in.

A decision is one call of ``_admissible`` (the plan and the safe-state test, run when
the pool fits by peak or backfills) or, for the summed test in arrival order that
admission-control defaults to, one call of ``_try_reserve``. The ``reserve`` columns
are the decisions that admitted, which is what ``planner_us`` is logged for: they
come right after something moved, so what a pool keeps between asks is cold for them.

Each result carries a digest of everything the pool decided (who was let in at each pass,
which grants were refused, every call of the decisions above), so two runs of the same
seed, with a setting changed, can be shown to have decided the same: ``--diff``.
"""

from __future__ import annotations

import argparse
import collections
import cProfile
import hashlib
import json
import multiprocessing
import os
import pstats
import random
import sys
import tempfile
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_kv_admission_peak import (
    DECODE,
    NODE,
    PREFILL,
    ROOT,
    _prefill,
    _ready,
    _run,
    _StubTransfer,
)

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager

POOL = 2048
PAGE = 128
HEADROOM = 8
TENSOR = "text_inputs"
MODES = [("peak", "backfill"), ("sum", "backfill"), ("sum", "fifo")]


# ── the pool ────────────────────────────────────────────────────────────


def _request(tokens: list[int], max_tokens: int, decode_keys: bool) -> KVReqConfig:
    whole = len(tokens) // PAGE
    return KVReqConfig(
        max_tokens=max_tokens,
        prefix_keys={"main": chain([tokens[at:at + PAGE] for at in range(0, len(tokens), PAGE)])},
        prefix_tail={"main": tokens[whole * PAGE:]},
        prompt_slots={"main": len(tokens)}, decode_labels=["main"],
        prefix_decode={"main": TENSOR} if decode_keys else None,
    )


class Pool:
    """A manager under pressure, and the requests that drive it."""

    def __init__(self, fit, order, n_reserved, n_waiting, seed, args):
        self.args = args
        self.rng = random.Random(seed)
        self.n_reserved, self.n_waiting = n_reserved, n_waiting
        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=POOL * PAGE,
                max_num_pages=POOL, page_size=PAGE, admission_fit=fit, admission_order=order,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.kv.enable_prefix_cache(ROOT, {"main": (PREFILL, DECODE)})
        self.next_token = 0
        self.next_id = 0
        self.shared = self._fresh(PAGE)
        self.running: dict[str, dict] = {}
        self.queue: list[str] = []
        self.meta: dict[str, tuple[int, int]] = {}
        self.deferred = 0
        # of what was decided, in order: see `--diff`
        self.digest = hashlib.blake2b(digest_size=8)
        self.decided = 0
        self._populate()
        self._admit_reserved()
        for _ in range(n_waiting):
            self._arrive()

    def _fresh(self, n: int) -> list[int]:
        start = self.next_token
        self.next_token += n
        return list(range(start, start + n))

    def _prompt(self, pages: int) -> list[int]:
        """A prompt of about ``pages`` pages; a fraction begins with one shared page."""
        tokens = self._fresh(pages * PAGE - self.rng.randrange(0, PAGE // 2))
        if self.rng.random() < self.args.shared_frac:
            tokens[:PAGE] = self.shared
        return tokens

    def _rid(self, prefix: str) -> str:
        self.next_id += 1
        return f"{prefix}{self.next_id}"

    def _populate(self) -> None:
        """Many short independent prompts, run and finished, so the index holds their pages."""
        kv = self.kv
        while len(kv._index._by_key) < self.args.background_pages:
            tokens = self._prompt(self.rng.choice([1, 1, 2, 2, 3, 4, 6]))
            rid = self._rid("p")
            kv.ingest_request(rid, _request(tokens, 4, False))
            assert _ready(kv, rid).ready
            assert _prefill(kv, rid, len(tokens)).ok
            kv.remove_request(rid)

    def _start(self, rid: str, tokens: list[int]) -> None:
        kv = self.kv
        assert _prefill(kv, rid, len(tokens)).ok
        if self.args.decode_keys:
            self._sample(rid, 1)
        left = self.rng.randint(1, self.args.max_life_pages)
        self.running[rid] = {"left": left, "tokens": left * PAGE}

    def _sample(self, rid: str, n: int) -> None:
        self.kv.extend_prefix_chain(
            rid, NODE, DECODE, {TENSOR: [torch.tensor(self._fresh(n))]},
        )

    def _admit_reserved(self) -> None:
        kv = self.kv
        claim = (POOL - 1 - HEADROOM) // self.n_reserved
        # prompts as long as leaves no page free once they are in: the rest of the pool is theirs
        self.prompt_pages = max(2, (POOL - 1 - len(kv._index._by_key)) // self.n_reserved)
        self.claim = max(claim, self.prompt_pages + 3)
        admitted = []
        for _ in range(self.n_reserved):
            tokens = self._prompt(self.prompt_pages)
            rid = self._rid("r")
            kv.ingest_request(rid, _request(
                tokens, max(1, claim - len(tokens) // PAGE - 1) * PAGE, self.args.decode_keys,
            ))
            assert _ready(kv, rid).ready, f"{rid} was not admitted into the pool"
            admitted.append((rid, tokens))
        for rid, tokens in admitted:
            self._start(rid, tokens)

    def _arrive(self) -> None:
        """A request about the size of the admitted ones, so what the queue lets in about
        replaces what leaves, and the number admitted stays near ``n_reserved``."""
        lo, hi = self.prompt_pages * 3 // 4, self.prompt_pages * 5 // 4
        prompt_pages = self.rng.randint(max(1, lo), max(1, hi))
        pages = max(
            prompt_pages + 3, self.rng.randint(self.claim * 3 // 4, self.claim * 5 // 4),
        )
        tokens = self._prompt(prompt_pages)
        rid = self._rid("w")
        self.kv.ingest_request(rid, _request(
            tokens, max(1, pages - len(tokens) // PAGE - 1) * PAGE, self.args.decode_keys,
        ))
        self.meta[rid] = (len(tokens), pages)
        self.queue.append(rid)

    def decide(self, *what) -> None:
        """Put what the pool decided into the digest."""
        self.digest.update(repr(what).encode())
        self.decided += 1

    # one scheduler tick -------------------------------------------------

    def grant(self) -> None:
        """A page for one admitted request, as decode asks for one every ``PAGE`` tokens."""
        if not self.running:
            return
        rid = self.rng.choice(list(self.running))
        info = self.running[rid]
        if info["left"] <= 0:
            self.finish(rid)
            return
        ok = _run(self.kv, {rid: PAGE}).ok
        self.decide("grant", rid, ok)
        if not ok:
            self.deferred += 1
            return
        if self.args.decode_keys:
            self._sample(rid, PAGE)
        info["left"] -= 1

    def finish(self, rid: str) -> None:
        self.decide("finish", rid)
        self.kv.remove_request(rid)
        del self.running[rid]

    def pass_over_queue(self) -> int:
        """Readiness for every request waiting, in arrival order; those let in start."""
        let_in = []
        for rid in self.queue:
            if _ready(self.kv, rid).ready:
                let_in.append(rid)
        self.decide("pass", let_in)
        for rid in let_in:
            self.queue.remove(rid)
            tokens_len = self.meta.pop(rid)[0]
            self._start(rid, [0] * tokens_len)
            self._arrive()
        return len(let_in)

    def advance_tokens(self) -> None:
        """What a decode loop does between two passes: one step, every request that runs one
        token on (so the committed length of each moves, and a page is granted when a token
        starts a new one), and a request that ran its course leaves.

        A step is refused for the request the pool names, which is held for the pass; the
        rest run again, as the worker answers a refusal.
        """
        for rid in [r for r, info in self.running.items() if info["tokens"] <= 0]:
            self.finish(rid)
        batch = list(self.running)
        while batch:
            outcome = _run(self.kv, {rid: 1 for rid in batch})
            self.decide("step", tuple(batch), outcome.ok)
            if outcome.ok:
                break
            self.deferred += 1
            held = getattr(outcome.reason, "request_id", None)
            if held not in batch:
                return
            batch.remove(held)
        for rid in batch:
            self.running[rid]["tokens"] -= 1
            if self.args.decode_keys:
                self._sample(rid, 1)

    def advance(self, grants: float) -> None:
        """What decode does between two passes: pages granted, and a request that ran its
        course (what a stop token is) leaves."""
        if self.args.per_token:
            return self.advance_tokens()
        for _ in range(int(grants) + (self.rng.random() < grants - int(grants))):
            self.grant()
        for rid in [r for r, info in self.running.items() if info["left"] <= 0]:
            self.finish(rid)

    def tick(self, grants: float) -> int:
        self.advance(grants)
        return self.pass_over_queue()


# ── what is measured ────────────────────────────────────────────────────


class Probe:
    """Times what a pool does per decision, on the instance, leaving the pool as it is."""

    def __init__(self, kv: KVManager, pool: Pool | None = None):
        self.kv = kv
        self.pool = pool
        self.rows: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
        self.enabled = True
        for name, kind in (
            ("_admissible", "admissible"), ("_try_reserve", "try_reserve"),
            ("_ruled_out", "ruled_out"),
        ):
            setattr(kv, name, self._timed(getattr(kv, name), kind))

    def _timed(self, fn, kind):
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            started = time.perf_counter_ns()
            answer = fn(*args, **kwargs)
            took = (time.perf_counter_ns() - started) / 1e3
            admitted = answer is None if kind == "try_reserve" else bool(answer) and kind == "admissible"
            self.rows[kind].append((took, admitted))
            if self.pool is not None:
                self.pool.decide(kind, args[0], bool(answer) if kind != "try_reserve" else answer is None)
            return answer
        return wrapper


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _summary(rows: list[tuple[float, bool]]) -> dict:
    times = [t for t, _ in rows]
    reserve = [t for t, admitted in rows if admitted]
    return {
        "n": len(times), "p50": _pct(times, 0.5), "p99": _pct(times, 0.99),
        "max": max(times, default=float("nan")),
        "n_reserve": len(reserve), "reserve_p50": _pct(reserve, 0.5),
        "reserve_p99": _pct(reserve, 0.99),
    }


def _flush_log(kv: KVManager) -> int:
    """Where in the admission log the next row goes, with everything so far written out."""
    if kv._alog is None:
        return 0
    kv._alog.flush()
    return os.path.getsize(kv._alog.path) if os.path.exists(kv._alog.path) else 0


def _logged_planner_us(kv: KVManager, offset: int) -> list[float]:
    """``planner_us`` of the ``reserve`` rows written since ``offset``: the number the server logs."""
    if kv._alog is None:
        return []
    kv._alog.flush()
    with open(kv._alog.path) as rows:
        rows.seek(offset)
        parsed = (json.loads(line) for line in rows)
        return [row["planner_us"] for row in parsed if row["event"] == "reserve"]


def run_config(job: tuple) -> dict:
    fit, order, n_reserved, n_waiting, args = job
    pool = Pool(fit, order, n_reserved, n_waiting, args.seed, args)
    kv = pool.kv
    grants = args.grants_per_pass if args.grants_per_pass is not None else n_reserved / PAGE
    for _ in range(args.warmup):
        pool.tick(grants)
    probe = Probe(kv, pool)
    logged_from = _flush_log(kv)
    start = {
        "free": kv._arena.num_free, "indexed": len(kv._index._by_key),
        "evictable": len(kv._index.evictable()), "reserved": len(kv._reserved),
        "waiting": len(kv._waiting),
    }
    pass_us = []
    admitted = 0
    reserved_seen = 0
    for _ in range(args.passes):
        pool.advance(grants)
        reserved_seen += len(kv._reserved)
        started = time.perf_counter_ns()
        admitted += pool.pass_over_queue()
        pass_us.append((time.perf_counter_ns() - started) / 1e3)
    end = {
        "free": kv._arena.num_free, "indexed": len(kv._index._by_key),
        "evictable": len(kv._index.evictable()), "reserved": len(kv._reserved),
        "waiting": len(kv._waiting),
    }
    decision = "try_reserve" if (fit, order) == ("sum", "fifo") else "admissible"
    planner_us = _logged_planner_us(kv, logged_from)
    return {
        "fit": fit, "order": order, "n_reserved": n_reserved, "n_waiting": n_waiting,
        "decision": decision, "start": start, "end": end, "admitted": admitted,
        "mean_reserved": reserved_seen / args.passes,
        "deferred_grants": pool.deferred,
        "decisions": _summary(probe.rows[decision]),
        "try_reserve": _summary(probe.rows["try_reserve"]),
        "pass_p50": _pct(pass_us, 0.5), "pass_p99": _pct(pass_us, 0.99),
        "arena_owner_changes": kv._arena.owner_changes,
        "decided": {"n": pool.decided, "digest": pool.digest.hexdigest()},
        "logged_reserve": {
            "n": len(planner_us), "p50": _pct(planner_us, 0.5), "p99": _pct(planner_us, 0.99),
        },
    }


def print_table(results: list[dict]) -> None:
    print(
        f"{'mode':14s} {'nres':>4s} {'nwait':>5s} | {'decisions':>9s} {'p50':>7s} {'p99':>8s}"
        f" | {'reserve':>7s} {'p50':>7s} {'p99':>8s} | {'try_res p50':>11s} | {'pass p50':>8s}"
        f" | free/indexed/evictable at the end"
        f"{' | logged reserve planner_us n/p50/p99' if results and results[0]['logged_reserve']['n'] else ''}"
    )
    for r in results:
        d, t, e = r["decisions"], r["try_reserve"], r["end"]
        print(
            f"{r['fit'] + '+' + r['order']:14s} {r['n_reserved']:4d} {r['n_waiting']:5d} |"
            f" {d['n']:9d} {d['p50']:7.0f} {d['p99']:8.0f} |"
            f" {d['n_reserve']:7d} {d['reserve_p50']:7.0f} {d['reserve_p99']:8.0f} |"
            f" {t['p50']:11.0f} | {r['pass_p50']:8.0f} |"
            f" {e['free']}/{e['indexed']}/{e['evictable']}, {r['mean_reserved']:.0f} reserved on average"
            + (
                f" | {r['logged_reserve']['n']}/{r['logged_reserve']['p50']:.0f}/{r['logged_reserve']['p99']:.0f}"
                if r["logged_reserve"]["n"] else ""
            )
        )


# ── where the time goes ─────────────────────────────────────────────────

PROFILED = [
    "_admissible", "_try_reserve", "_ruled_out", "_supply", "evictable", "_find_evictable",
    "_plan_state", "_plan_entry", "__init__", "peak_from", "banker_safe", "_head_shadow",
    "shadow", "_outstanding", "_outstanding_afresh", "_room", "exceeds", "easy_allows",
    "footprint", "_held_fresh", "_refusal_key", "_gate", "_reservation", "_lease", "peek",
    "_plan_from_table", "_plan_fields", "_plan_row", "fields", "from_arrays", "with_candidate",
    "_defer_grant",
]


def profile(args, fit, order, n_reserved, n_waiting) -> None:
    """cProfile of the decisions alone (``--scope decision``: only while ``_admissible``, or
    ``_try_reserve`` for the summed test in arrival order, runs), or of whole passes."""
    pool = Pool(fit, order, n_reserved, n_waiting, args.seed, args)
    kv = pool.kv
    grants = args.grants_per_pass if args.grants_per_pass is not None else n_reserved / PAGE
    for _ in range(args.warmup):
        pool.tick(grants)
    decision = "_try_reserve" if (fit, order) == ("sum", "fifo") else "_admissible"
    prof = cProfile.Profile()
    seen = {"decisions": 0, "outermost": 0}
    merged: list[pstats.Stats] = []
    if args.scope == "decision":
        inner = getattr(kv, decision)

        def profiled(*a, **kw):
            if seen["outermost"]:
                return inner(*a, **kw)
            seen["outermost"] = 1
            one = cProfile.Profile()
            one.enable()
            try:
                answer = inner(*a, **kw)
            finally:
                one.disable()
                seen["outermost"] = 0
            admitted = answer is None if decision == "_try_reserve" else bool(answer)
            if args.only == "all" or (args.only == "reserve") == admitted:
                seen["decisions"] += 1
                merged.append(pstats.Stats(one))
            return answer
        setattr(kv, decision, profiled)
    for _ in range(args.passes):
        pool.advance(grants)
        if args.scope == "pass":
            prof.enable()
        pool.pass_over_queue()
        if args.scope == "pass":
            prof.disable()
    n_decisions = seen["decisions"]
    if args.scope == "pass":
        n_decisions = args.passes
    if args.scope == "decision":
        if not merged:
            print(f"no {args.only} decision in {args.passes} passes")
            return
        stats = merged[0]
        for other in merged[1:]:
            stats.add(other)
    else:
        stats = pstats.Stats(prof)
    unit = "pass" if args.scope == "pass" else "decision"
    print(f"\ncProfile, {fit}+{order}, {n_reserved} reserved, {n_waiting} waiting, scope {args.scope}, {args.only}: "
          f"{n_decisions} {unit}s over {args.passes} passes (the profiler inflates every figure "
          f"about 2x; the shares are what counts)")
    rows = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for (path, _, name), (_cc, nc, tt, ct, _) in stats.stats.items():
        if name in PROFILED and ("kv" in path or "peak_plan" in path or "prefix_index" in path):
            row = rows[name]
            row[0] += nc
            row[1] += tt
            row[2] += ct
    top = max((row[2] for row in rows.values()), default=1.0) or 1.0
    print(f"{'function':22s} {'calls/' + unit:>14s} {'own us/' + unit:>14s} "
          f"{'cum us/' + unit:>14s} {'cum % of top':>13s}")
    for name, (calls, tot, cum) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
        print(f"{name:22s} {calls / n_decisions:14.2f} {tot / n_decisions * 1e6:14.1f} "
              f"{cum / n_decisions * 1e6:14.1f} {100 * cum / top:12.0f}%")


# ── what moves owner_changes ────────────────────────────────────────────


def audit(args, fit, order, n_reserved, n_waiting) -> None:
    """Which operations tick the arena's owner counter, on which pages, and how often
    a tick leaves the evictable set as it was."""
    pool = Pool(fit, order, n_reserved, n_waiting, args.seed, args)
    kv, arena, index = pool.kv, pool.kv._arena, pool.kv._index
    grants = args.grants_per_pass if args.grants_per_pass is not None else n_reserved / PAGE
    for _ in range(args.warmup):
        pool.tick(grants)

    ticks = collections.Counter()
    for name in ("retain", "release"):
        original = getattr(arena, name)

        def wrapped(pages, _original=original, _name=name):
            caller = sys._getframe(1).f_code.co_name
            held = any(index._key[p] is not None for p in pages)
            ticks[(_name, caller, "index holds it" if held else "index does not")] += 1
            return _original(pages)
        setattr(arena, name, wrapped)

    refinds = collections.Counter()
    find = index._find_evictable
    last = [index._evictable]

    def counting_find():
        found = find()
        refinds["recomputed"] += 1
        refinds["unchanged"] += found == last[0]
        last[0] = found
        return found
    index._find_evictable = counting_find

    per_grant = []
    decisions = 0
    probe = Probe(kv)
    for _ in range(args.passes):
        for _ in range(int(grants) + (pool.rng.random() < grants - int(grants))):
            before, free_before = arena.owner_changes, arena.num_free
            pool.grant()
            per_grant.append((arena.owner_changes - before, free_before))
        pool.advance(0)
        pool.pass_over_queue()
    decisions = len(probe.rows["admissible" if (fit, order) != ("sum", "fifo") else "try_reserve"])

    print(f"\naudit, {fit}+{order}, {n_reserved} reserved, {n_waiting} waiting, "
          f"{args.passes} passes, {len(per_grant)} page grants, {decisions} decisions")
    print("owner_changes ticks per page grant: "
          f"{collections.Counter(t for t, _ in per_grant).most_common()}  "
          f"(grants with the free list empty: {sum(1 for _, f in per_grant if f == 0)})")
    print("pages named in each retain/release (caller, whether the index holds the pages):")
    for (name, caller, held), count in sorted(ticks.items(), key=lambda kv: -kv[1]):
        print(f"  {name:8s} {caller:28s} {held:15s} {count:8d}")
    print(f"_find_evictable: {refinds['recomputed']} walks, {refinds['unchanged']} of them "
          f"to the set that was there before ({refinds['recomputed'] / max(1, decisions):.2f} per decision)")


# ── main ────────────────────────────────────────────────────────────────


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reserved", type=int, nargs="+", default=[8, 32, 64])
    p.add_argument("--waiting", type=int, nargs="+", default=[10, 100, 1000])
    p.add_argument("--modes", nargs="+", default=[f"{f}+{o}" for f, o in MODES])
    p.add_argument("--passes", type=int, default=3000, help="scheduler passes timed per config")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--background-pages", type=int, default=1400,
                   help="pages the index holds from finished requests before the admitted ones start; "
                        "their prompts are sized to take the rest of the pool")
    p.add_argument("--shared-frac", type=float, default=0.3,
                   help="share of prompts that begin with one shared page")
    p.add_argument("--grants-per-pass", type=float, default=None,
                   help="page grants between two passes; default n_reserved / page_size, "
                        "one decoded token per request per pass")
    p.add_argument("--max-life-pages", type=int, default=12,
                   help="pages a request decodes before it stops, drawn from 1..this")
    p.add_argument("--per-token", action="store_true",
                   help="every admitted request runs one token a pass, in one batched step, as a "
                        "decode loop does, instead of a page granted at a time to one of them: "
                        "the committed lengths move every pass, and the plan's key does not")
    p.add_argument("--no-decode-keys", dest="decode_keys", action="store_false",
                   help="do not index the pages a request generates")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--json", help="write every result here")
    p.add_argument("--log", action="store_true",
                   help="log decisions as a server with telemetry on does (planner_us rows)")
    p.add_argument("--profile", action="store_true", help="cProfile the first reserved, waiting, per mode")
    p.add_argument("--only", choices=["all", "reserve", "wait"], default="all",
                   help="which decisions --profile --scope decision covers")
    p.add_argument("--scope", choices=["decision", "pass"], default="decision",
                   help="what --profile covers: the decisions alone, or whole passes over the queue")
    p.add_argument("--audit", action="store_true", help="what ticks owner_changes, for the first of each")
    p.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"),
                   help="whether two runs decided the same, config by config; runs nothing")
    return p.parse_args()


def diff(first: str, second: str) -> int:
    """Whether the two runs' digests of what was decided agree in every config. The exit code."""
    with open(first) as a, open(second) as b:
        runs = json.load(a), json.load(b)
    key = lambda r: (r["fit"], r["order"], r["n_reserved"], r["n_waiting"])  # noqa: E731
    left, right = ({key(r): r for r in run} for run in runs)
    differ = [k for k in left.keys() | right.keys() if (
        k not in left or k not in right or left[k].get("decided") != right[k].get("decided")
    )]
    for k in sorted(differ):
        print(f"DIFFERENT {k}: {left.get(k, {}).get('decided')} vs {right.get(k, {}).get('decided')}")
    print(f"{len(left.keys() | right.keys()) - len(differ)} of {len(left.keys() | right.keys())} "
          f"configs decided the same; {sum(r['decided']['n'] for r in left.values())} decisions in the first")
    return 1 if differ else 0


def main() -> None:
    args = _parse()
    if args.diff:
        sys.exit(diff(*args.diff))
    torch.set_num_threads(1)
    manager_mod.KVTransferManager = _StubTransfer
    for name in ("MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW"):
        os.environ.pop(name, None)
    os.environ.pop("MSTAR_STEP_TELEMETRY_DIR", None)
    if args.log:
        os.environ["MSTAR_STEP_TELEMETRY_DIR"] = tempfile.mkdtemp(prefix="admission_bench_")
    modes = [tuple(m.split("+")) for m in args.modes]
    if args.profile or args.audit:
        for fit, order in modes:
            for n_reserved in args.reserved[:1]:
                for n_waiting in args.waiting[:1]:
                    (profile if args.profile else audit)(args, fit, order, n_reserved, n_waiting)
        return
    jobs = [
        (fit, order, n_reserved, n_waiting, args)
        for fit, order in modes for n_reserved in args.reserved for n_waiting in args.waiting
    ]
    if args.jobs > 1:
        with multiprocessing.get_context("fork").Pool(args.jobs) as workers:
            results = workers.map(run_config, jobs, chunksize=1)
    else:
        results = [run_config(job) for job in jobs]
    print(f"times in microseconds; {args.passes} passes per config after {args.warmup} warm-up, "
          f"background {args.background_pages} pages, shared-prefix share {args.shared_frac}, "
          f"decode keys {args.decode_keys}, log {args.log}, per token {args.per_token}")
    print_table(results)
    if args.json:
        with open(args.json, "w") as out:
            json.dump([{k: v for k, v in r.items()} for r in results], out, indent=1)


if __name__ == "__main__":
    main()
