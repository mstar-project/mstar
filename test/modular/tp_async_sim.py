"""CPU lockstep simulator for the TP async-scheduling protocol.

Models the leader/follower worker protocol for speculative scheduling on
lockstep-parallel (TP) nodes — the case ``Speculator.can_speculate`` currently refuses
— as an explicit-state model checker: every interleaving of
{follower message processing, per-rank GPU progress, leader spec build, leader
post-step} is explored by DFS over a small scripted workload, and the design's
invariants are checked at every terminal state.

Invariants (checked at every terminal state):
  I1 lockstep   — every rank posts the identical batch sequence in identical
                  order (a divergence is the NCCL-hang class).
  I2 KV symmetry — spec allocation/rewind identical across ranks (modeled as
                  per-batch page liveness).
  I3 atomic enqueue — a gated batch never executes before its Commit
                  (structural: the GPU action refuses gated heads).
  I4 liveness   — a run that cannot progress and is not terminal is a deadlock.

Modes:
  SERIAL      — today's path, no speculation (harness sanity baseline).
  B1          — gated commit/cancel: all ranks hold spec batch S at the launch
                gate until Commit(seq)/Cancel(seq).
  B2_RETRACT  — run-ahead where Cancel removes S from any rank that has not
                executed it yet. Deliberately included as the naive reading of
                "cancel": the checker is expected to REFUTE it (I1).
  B2_VOID     — run-ahead where a broadcast S is *always* executed by every
                rank; Cancel only voids its effects (outputs dropped, pages
                freed). This is the semantics the design must specify.
  B2_LOCAL    — the IMPLEMENTED protocol (MSTAR_TP_ASYNC_SCHED=1): no Cancel
                message at all. A rank may launch spec batch S only after it
                has itself completed S's parent step N, and at that moment it
                applies the same verdict the leader applies at its post(N) —
                derived from state that is identical on every rank
                (structural failure on N → drop S locally, never run it).
                A SPEC that arrives after the rank already completed N is
                reconciled on arrival. Retract is safe here for the reason
                B2_RETRACT is not: verdict(N) happens-before launch(S) on
                every rank, so no rank can have run S when another voids it.

Follower behavior encodes a property of the graph layer: spec batches are
buildable from replicated graph state alone (``speculative_signals``
placeholders, graph/base.py), so a follower builds S without waiting for its
own step-N completion.

No mstar imports, no third-party deps. Run directly:
    python3 test/modular/tp_async_sim.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

STATE_CAP = 400_000

# ---------------------------------------------------------------------------
# World representation (plain dicts/tuples so states hash cheaply)
# ---------------------------------------------------------------------------
# batch: (kind, step, rids, seq)   kind in {"step", "spec"}
# msg:   ("STEP", batch) | ("SPEC", batch) | ("COMMIT", seq) | ("CANCEL", seq)
# rank:  {"fifo": tuple(msg), "q": tuple((batch, gated, void)),
#         "exec": tuple(batch), "live": frozenset(batch), "voided": frozenset(seq)}


def _new_rank():
    return {
        "fifo": (), "q": (), "exec": (),
        "live": frozenset(), "voided": frozenset(),
    }


def make_world(mode, nranks, comp0, script):
    w = {
        "mode": mode,
        "nranks": nranks,
        "script": script,
        "seq": 0,
        "L": {
            "step": 0,
            "comp": tuple(comp0),
            "cur": None,          # batch currently owed a post-step
            "spec": None,         # outstanding spec batch for step+1
            "spec_closed": False, # spec window consumed for this step
            "waste": 0,
            "done": False,
        },
        "ranks": [_new_rank() for _ in range(nranks)],
    }
    b = ("step", 0, tuple(comp0), 0)
    _broadcast(w, ("STEP", b))
    _build(w, 0, b, gated=False)
    w["L"]["cur"] = b
    return w


def _broadcast(w, msg):
    for r in range(1, w["nranks"]):
        w["ranks"][r]["fifo"] = w["ranks"][r]["fifo"] + (msg,)


def _build(w, rank, batch, gated):
    rk = w["ranks"][rank]
    rk["live"] = rk["live"] | {batch}
    rk["q"] = rk["q"] + ((batch, gated, False),)


def _unbuild(w, rank, seq, executed_ok):
    """Cancel handling on one rank. Returns True if state changed."""
    rk = w["ranks"][rank]
    mode = w["mode"]
    inq = [(b, g, v) for (b, g, v) in rk["q"] if b[0] == "spec" and b[3] == seq]
    ran = [b for b in rk["exec"] if b[0] == "spec" and b[3] == seq]
    if mode == "B1":
        # gate guarantees not executed; remove from queue, free pages
        assert not ran, "B1: cancelled spec executed before commit (I3 breach)"
        if inq:
            batch = inq[0][0]
            rk["q"] = tuple(e for e in rk["q"] if e[0] != batch)
            rk["live"] = rk["live"] - {batch}
        return True
    if mode == "B2_RETRACT":
        if ran:
            rk["live"] = rk["live"] - {ran[0]}      # void after the fact
        elif inq:
            batch = inq[0][0]
            rk["q"] = tuple(e for e in rk["q"] if e[0] != batch)
            rk["live"] = rk["live"] - {batch}        # retract pre-execution
        return True
    if mode == "B2_VOID":
        # never retract: mark void; pages freed at/after execution
        if ran:
            rk["live"] = rk["live"] - {ran[0]}
        else:
            rk["voided"] = rk["voided"] | {seq}
        return True
    raise AssertionError(mode)


def _local_verdict_on_step(w, rank, step):
    """B2_LOCAL: rank ``rank`` just completed ``step``. Apply the verdict for
    any spec speculated FROM ``step`` that this rank holds: structural failure
    on ``step`` → retract it locally (it never launches here — and, by the
    same rule on every rank, nowhere else either); otherwise release its
    launch gate. Same information the leader uses in its post(step)."""
    rk = w["ranks"][rank]
    structural = bool(w["script"].get(step, {}).get("structural"))
    newq = []
    for (b, g, v) in rk["q"]:
        if b[0] == "spec" and b[1] == step + 1:
            if structural:
                rk["live"] = rk["live"] - {b}
                continue          # retract: verdict(N) precedes launch(S) here
            newq.append((b, False, v))   # release the local launch gate
        else:
            newq.append((b, g, v))
    rk["q"] = tuple(newq)


def _local_verdict_on_arrival(w, rank, spec):
    """B2_LOCAL: a SPEC for parent step ``spec[1]-1`` arrives at ``rank``.
    If the rank already completed the parent, reconcile now (this is
    ``Worker._register_tp_follow``): drop on structural, else build ungated.
    Otherwise build gated; ``_local_verdict_on_step`` will release it."""
    rk = w["ranks"][rank]
    parent = spec[1] - 1
    done = any(b[1] == parent for b in rk["exec"])
    if not done:
        _build(w, rank, spec, gated=True)
        return
    if w["script"].get(parent, {}).get("structural"):
        return  # dropped on arrival, never enters the queue
    _build(w, rank, spec, gated=False)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def enabled_actions(w):
    acts = []
    L = w["L"]
    # follower message processing (strict FIFO order = ZMQ stream)
    for r in range(1, w["nranks"]):
        if w["ranks"][r]["fifo"]:
            acts.append(("proc", r))
    # per-rank GPU progress; gated head blocks the queue (I3 by construction)
    for r in range(w["nranks"]):
        q = w["ranks"][r]["q"]
        if q and not q[0][1]:
            acts.append(("gpu", r))
    if not L["done"]:
        # leader spec build for step+1, inside the open window
        if (w["mode"] != "SERIAL" and L["spec"] is None and not L["spec_closed"]
                and L["comp"]):
            acts.append(("spec",))
        # leader post-step once its own GPU finished the current batch
        if L["cur"] is not None and L["cur"] in w["ranks"][0]["exec"]:
            acts.append(("post",))
    return acts


def apply_action(w, act):
    w = copy.deepcopy(w)
    kind = act[0]
    if kind == "proc":
        _do_proc(w, act[1])
    elif kind == "gpu":
        _do_gpu(w, act[1])
    elif kind == "spec":
        _do_spec(w)
    elif kind == "post":
        _do_post(w)
    return w


def _do_proc(w, r):
    rk = w["ranks"][r]
    msg, rk["fifo"] = rk["fifo"][0], rk["fifo"][1:]
    tag = msg[0]
    if tag == "STEP":
        _build(w, r, msg[1], gated=False)
    elif tag == "SPEC":
        if w["mode"] == "B2_LOCAL":
            _local_verdict_on_arrival(w, r, msg[1])
        else:
            # Buildable now — placeholder inputs, no wait on local step-N
            _build(w, r, msg[1], gated=(w["mode"] == "B1"))
    elif tag == "COMMIT":
        seq = msg[1]
        rk["q"] = tuple(
            (b, False if (b[0] == "spec" and b[3] == seq) else g, v)
            for (b, g, v) in rk["q"]
        )
    elif tag == "CANCEL":
        _unbuild(w, r, msg[1], executed_ok=True)


def _do_gpu(w, r):
    rk = w["ranks"][r]
    (batch, gated, void), rk["q"] = rk["q"][0], rk["q"][1:]
    assert not gated, "I3 breach: gated batch executed"
    rk["exec"] = rk["exec"] + (batch,)
    if batch[0] == "spec" and batch[3] in rk["voided"]:
        rk["live"] = rk["live"] - {batch}
        rk["voided"] = rk["voided"] - {batch[3]}
    if w["mode"] == "B2_LOCAL" and r != 0:
        # A follower's "await(N)" is completing N: the moment it decides
        # about specs speculated from N. The leader decides at post(N) — it
        # may still be BUILDING S when its GPU finishes N.
        _local_verdict_on_step(w, r, batch[1])


def _do_spec(w):
    L = w["L"]
    ev = w["script"].get(L["step"], {})
    L["spec_closed"] = True
    if ev.get("alloc_fail"):
        return  # spec build failed → serial fallback this step
    w["seq"] += 1
    b = ("spec", L["step"] + 1, L["comp"], w["seq"])
    _broadcast(w, ("SPEC", b))
    # B1: gated on Commit. B2_LOCAL: gated on the leader's OWN completion of
    # step N (released/dropped in _local_verdict_on_step). B2_*: run-ahead.
    _build(w, 0, b, gated=(w["mode"] in ("B1", "B2_LOCAL")))
    L["spec"] = b


def _do_post(w):
    L = w["L"]
    mode = w["mode"]
    ev = w["script"].get(L["step"], {})
    stops = frozenset(ev.get("stop", ())) & set(L["comp"])
    structural = bool(ev.get("structural"))
    new_comp = tuple(x for x in L["comp"] if x not in stops)
    new_comp = new_comp + tuple(ev.get("arrival", ()))
    S = L["spec"]

    def _serial_next():
        if new_comp:
            b = ("step", L["step"] + 1, new_comp, 0)
            _broadcast(w, ("STEP", b))
            _build(w, 0, b, gated=False)
            L["cur"] = b
        else:
            L["done"] = True
            L["cur"] = None

    if S is None:
        _serial_next()
    elif mode == "B2_LOCAL":
        # No message. The leader applies its own verdict for step N to the
        # S it built (gated on itself until now): structural → retract
        # locally and reschedule; else release and continue the chain.
        # Followers reach the identical verdict at their own completion of N
        # (_local_verdict_on_step / _local_verdict_on_arrival).
        _local_verdict_on_step(w, 0, L["step"])
        if structural:
            _serial_next()
        else:
            if stops:
                L["waste"] += len(stops & set(S[2]))
            L["cur"] = S
    elif mode == "B1":
        if not structural and not stops:
            _broadcast(w, ("COMMIT", S[3]))
            rk = w["ranks"][0]
            rk["q"] = tuple(
                (b, False if b == S else g, v) for (b, g, v) in rk["q"]
            )
            L["cur"] = S
        else:
            _broadcast(w, ("CANCEL", S[3]))
            _unbuild(w, 0, S[3], executed_ok=False)
            _serial_next()
    elif structural:
        # structural invalidation → cancel + authoritative reschedule.
        # S still executes on every rank under VOID; RETRACT is the bug.
        _broadcast(w, ("CANCEL", S[3]))
        _unbuild(w, 0, S[3], executed_ok=True)
        L["waste"] += 1
        _serial_next()
    else:
        # implicit commit; stopped rids ride along as waste
        if stops:
            L["waste"] += len(stops & set(S[2]))
        L["cur"] = S
    L["comp"] = new_comp
    if not L["done"]:
        L["step"] += 1
        L["spec"] = None
        L["spec_closed"] = False
        if not L["comp"] and L["cur"] is None:
            L["done"] = True


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

def world_key(w):
    L = w["L"]
    return (
        w["seq"], L["step"], L["comp"], L["cur"], L["spec"], L["spec_closed"],
        L["waste"], L["done"],
        tuple(
            (rk["fifo"], rk["q"], rk["exec"],
             frozenset(rk["live"]), frozenset(rk["voided"]))
            for rk in w["ranks"]
        ),
    )


def is_terminal(w):
    if not w["L"]["done"]:
        return False
    return all(not rk["fifo"] and not rk["q"] for rk in w["ranks"])


@dataclass
class Result:
    scenario: str
    mode: str
    states: int = 0
    terminals: int = 0
    max_waste: int = 0
    violations: list = field(default_factory=list)  # (kind, trace)

    @property
    def ok(self):
        return not self.violations


def check_terminal(w, must_execute):
    execs = [rk["exec"] for rk in w["ranks"]]
    if any(e != execs[0] for e in execs[1:]):
        return "I1 divergence: " + " | ".join(
            ",".join(f"{b[0]}{b[1]}" for b in e) for e in execs
        )
    lives = [rk["live"] for rk in w["ranks"]]
    if any(lv != lives[0] for lv in lives[1:]):
        return "I2 page-liveness asymmetry"
    for rid in must_execute:
        if not any(rid in b[2] for b in execs[0]):
            return f"liveness: rid {rid} never executed"
    return None


def explore(mode, nranks, comp0, script, scenario, must_execute=()):
    res = Result(scenario, mode)
    root = make_world(mode, nranks, comp0, script)
    seen = set()
    stack = [(root, ())]
    while stack:
        w, trace = stack.pop()
        k = world_key(w)
        if k in seen:
            continue
        seen.add(k)
        res.states += 1
        if res.states > STATE_CAP:
            raise RuntimeError(f"state cap hit: {scenario}/{mode}")
        acts = enabled_actions(w)
        if not acts:
            if is_terminal(w):
                res.terminals += 1
                res.max_waste = max(res.max_waste, w["L"]["waste"])
                err = check_terminal(w, must_execute)
                if err and len(res.violations) < 3:
                    res.violations.append((err, trace))
            elif len(res.violations) < 3:
                res.violations.append(("I4 deadlock", trace))
            continue
        for a in acts:
            try:
                stack.append((apply_action(w, a), trace + (a,)))
            except AssertionError as e:  # I3 breach surfaces here
                res.violations.append((f"I3/assert: {e}", trace + (a,)))
    return res


# ---------------------------------------------------------------------------
# Scenarios — the failure matrix
# ---------------------------------------------------------------------------

SCENARIOS = {
    # name: (comp0, script, must_execute)
    "clean-3step": (("r0",), {2: {"stop": {"r0"}}}, ()),
    "midstream-stop": (("r0", "r1"),
                       {1: {"stop": {"r0"}}, 3: {"stop": {"r1"}}}, ()),
    "alloc-fail": (("r0",),
                   {1: {"alloc_fail": True}, 2: {"stop": {"r0"}}}, ()),
    "structural-cancel": (("r0", "r1"),
                          {1: {"structural": True}, 2: {"stop": {"r0", "r1"}}}, ()),
    "late-arrival": (("r0",),
                     {1: {"arrival": ("r1",)}, 3: {"stop": {"r0", "r1"}}},
                     ("r1",)),
}

MODES = ("SERIAL", "B1", "B2_VOID", "B2_LOCAL", "B2_RETRACT")


def run_all(nranks=2):
    results = []
    for name, (comp0, script, must) in SCENARIOS.items():
        for mode in MODES:
            results.append(explore(mode, nranks, comp0, script, name, must))
    return results


def main():
    for nranks in (2, 3):
        print(f"\n=== {nranks} ranks ===")
        print(f"{'scenario':<18} {'mode':<11} {'states':>7} {'terms':>6} "
              f"{'waste':>5}  result")
        for r in run_all(nranks):
            verdict = "PASS" if r.ok else f"FAIL ({r.violations[0][0]})"
            print(f"{r.scenario:<18} {r.mode:<11} {r.states:>7} "
                  f"{r.terminals:>6} {r.max_waste:>5}  {verdict}")
            if r.violations:
                kind, trace = r.violations[0]
                print(f"{'':<18} shortest-found trace: "
                      f"{' '.join('/'.join(map(str, a)) for a in trace)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Verify results (speculative decoding): record at postprocess, apply at admit
# ---------------------------------------------------------------------------
#
# A verifying node commits step S on every rank; postprocess(S) records its
# result; the take-back at the head of admit(M) applies results by a due rule
# (`VerifyLedger.take_due`). Each rank interleaves its own commit, postprocess
# and admit, constrained only by its worker's stage order:
#   - admit(S + 1) follows commit(S) (the plan thread waits on S's commit
#     event; the GPU thread commits in order),
#   - postprocess(S) follows commit(S),
#   - asynchronous: postprocess(S) precedes admit(S + 2) (the worker arms S + 2
#     only after post-processing S); postprocess(S) may land before or after
#     admit(S + 1),
#   - synchronous: postprocess(S) precedes admit(S + 1).
# Invariants: at every admit, every rank applies the same set of results
# (lockstep: the ranks then admit from the same lengths), and no due result
# is unrecorded (the ledger raises). Rule "ledger" is `take_due`; rule
# "arrived" applies whatever is recorded, which the checker must refute.


@dataclass
class VerifyResult:
    violations: list = field(default_factory=list)
    states: int = 0
    # offloads shipped with a speculative head (explore_offload's coverage)
    speculative_offloads: int = 0

    @property
    def ok(self):
        return not self.violations


def _due(rule, synchronous, step, recorded, applied):
    """What admit(``step``) applies on one rank, or None if a due one is unrecorded."""
    if rule == "arrived":
        return frozenset(s for s in recorded if s not in applied and s < step)
    due = {s for s in range(1, step - 1) if s not in applied}
    if synchronous and step - 1 >= 1 and step - 1 not in applied:
        # the step's own rows ran step - 1 (one decoding row, every step)
        due.add(step - 1)
    if not due <= recorded:
        return None
    return frozenset(due)


def explore_verify(rule="ledger", synchronous=False, nranks=2, steps=4, stage_order=True) -> VerifyResult:
    """DFS over every interleaving of the ranks' admit, commit and postprocess.
    ``stage_order`` False drops the worker's guarantee that postprocess
    precedes the admit its result comes due at."""
    res = VerifyResult()
    # per rank: (admitted, committed, recorded, applied, log)
    start = tuple((0, 0, frozenset(), frozenset(), ()) for _ in range(nranks))
    seen = set()
    stack = [start]
    while stack:
        world = stack.pop()
        if world in seen:
            continue
        seen.add(world)
        res.states += 1
        moved = False
        for r, (admitted, committed, recorded, applied, log) in enumerate(world):
            nxt = []
            step = admitted + 1
            gate = (step - 1 in recorded) if synchronous else (step - 2 < 1 or step - 2 in recorded)
            gate = gate or not stage_order
            if step <= steps and committed == admitted and (step == 1 or gate):
                due = _due(rule, synchronous, step, recorded, applied)
                if due is None:
                    res.violations.append(("unrecorded-due", r, step))
                    continue
                nxt.append((step, committed, recorded, applied | due, log + ((step, due),)))
            if committed < admitted:
                nxt.append((admitted, committed + 1, recorded, applied, log))
            for s in range(1, committed + 1):
                if s not in recorded:
                    nxt.append((admitted, committed, recorded | {s}, applied, log))
                    break  # postprocess runs in step order
            for rank_state in nxt:
                moved = True
                stack.append(world[:r] + (rank_state,) + world[r + 1:])
        if not moved:
            logs = [rank[4] for rank in world]
            if any(log != logs[0] for log in logs):
                res.violations.append(("diverged", logs))
        if len(res.violations) > 5:
            break
    return res


# ---------------------------------------------------------------------------
# Offload replay of a verifying row (speculative decoding)
# ---------------------------------------------------------------------------
#
# Rank 0 offloads a row V while building step M and ships the move with M's
# broadcast; it settles V first (applies V's recorded results), and never picks
# a row with an unrecorded step (`Engine.verify_unrecorded`). A follower replays
# the move when it arms M, and settles V there, which raises if one of V's
# steps is unrecorded on the follower. When it arms M:
#   - rank 0 builds M speculatively when M - 1 is still un-post-processed on
#     rank 0, else serially;
#   - a follower follows a speculative head M after post-processing M - 2 (the
#     worker arms S + 2 only after S's postprocess), and takes a serial head
#     only with no step of the node in flight: after post-processing M - 1;
#   - synchronous verify nodes never speculate.
# V is in every step until it is offloaded.


def explore_offload(synchronous=False, nranks=2, steps=5, serial_gate=True, victim_stays=False) -> VerifyResult:
    """DFS over every interleaving, and over which steps V runs in (a row may
    skip a step: not ready, held back); a violation is a follower settling V
    with one of V's steps unrecorded. ``serial_gate`` False lets a follower take
    a serial head as early as a speculative one (the negative control).
    ``victim_stays`` is an offload that moved nothing: its settle still ships, and
    V may run in that step and later ones."""
    res = VerifyResult()
    # per rank: (armed, committed, recorded); then, per step rank 0 built, whether it was
    # speculative and whether V ran in it; and the step whose broadcast carries V's offload
    start = (tuple((0, 0, frozenset()) for _ in range(nranks)), (), (), 0)
    seen, stack = set(), [start]
    while stack:
        world = stack.pop()
        if world in seen:
            continue
        seen.add(world)
        res.states += 1
        ranks, spec, member, offload_at = world
        for r, (armed, committed, recorded) in enumerate(ranks):
            nxt = []
            m = armed + 1
            if m <= steps and committed == armed:
                if r == 0:
                    gate = (m - 1 < 1 or m - 1 in recorded) if synchronous else (m - 2 < 1 or m - 2 in recorded)
                    if gate:
                        speculative = not synchronous and m - 1 >= 1 and m - 1 not in recorded
                        v_steps = {s for s in range(1, committed + 1) if member[s - 1]}
                        options = []
                        if offload_at:
                            options.append((offload_at, False))
                            if victim_stays:
                                options.append((offload_at, True))
                        else:
                            options += [(0, True), (0, False)]
                            if v_steps <= recorded:
                                # V is a victim rank 0 may pick now, gone from this step on unless it stays
                                options.append((m, victim_stays))
                        for at, runs in options:
                            if at == m and speculative:
                                res.speculative_offloads += 1
                            nxt.append(((m, committed, recorded), spec + (speculative,), member + (runs,), at))
                elif ranks[0][0] >= m:
                    lag = 2 if spec[m - 1] or not serial_gate else 1
                    if m - lag < 1 or m - lag in recorded:
                        if offload_at == m:
                            v_steps = {s for s in range(1, min(committed, m - 1) + 1) if member[s - 1]}
                            if not v_steps <= recorded:
                                res.violations.append(("unrecorded-at-replay", r, m, sorted(v_steps - recorded)))
                                continue
                        nxt.append(((m, committed, recorded), spec, member, offload_at))
            if committed < armed:
                nxt.append(((armed, committed + 1, recorded), spec, member, offload_at))
            for s in range(1, committed + 1):
                if s not in recorded:
                    nxt.append(((armed, committed, recorded | {s}), spec, member, offload_at))
                    break  # postprocess runs in step order
            for rank_state, spec2, member2, at in nxt:
                stack.append((ranks[:r] + (rank_state,) + ranks[r + 1:], spec2, member2, at))
        if len(res.violations) > 5:
            break
    return res
