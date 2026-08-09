"""CPU lockstep simulator for the TP async-scheduling protocol.

Models the leader/follower worker protocol for speculative scheduling on
lockstep-parallel (TP) nodes — the case ``_can_speculate`` currently refuses
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


def _do_spec(w):
    L = w["L"]
    ev = w["script"].get(L["step"], {})
    L["spec_closed"] = True
    if ev.get("alloc_fail"):
        return  # spec build failed → serial fallback this step
    w["seq"] += 1
    b = ("spec", L["step"] + 1, L["comp"], w["seq"])
    _broadcast(w, ("SPEC", b))
    _build(w, 0, b, gated=(w["mode"] == "B1"))
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

MODES = ("SERIAL", "B1", "B2_VOID", "B2_RETRACT")


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
