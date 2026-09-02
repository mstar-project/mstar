"""Chaos + parity harness for an mstar deployment.

Points at either surface, native server or Dynamo frontend; they expose the
same OpenAI routes, so the same run exercises both:

    python stress-test/chaos.py --url http://localhost:8000 --model bagel --duration 7200

What it adds over benchmark/runner.py is cancellation: a fraction of requests
are dropped mid-flight at a sampled point. Three cancel points, because they
hit different code:

  early   before the first chunk arrives  -- image/video produce nothing until
                                             done, so this is the case a
                                             chunk-sampled cancel check misses
  mid     after N chunks
  late    just before the stream ends

Cancellation is a socket drop (task cancel while reading), not a graceful
abort, so it exercises the is_killed path rather than is_stopped.

Every request logs (seq, endpoint, seed, cancel_at) so a failure is replayable.
Invariants are checked at the end -- that's the part that finds bugs; the
metrics only tell you it was slow.

Rounds
------
`--rounds N` runs the load N times with a drain to idle in between. Idle means
every in-flight request has finished, no request starts for `--idle-seconds`,
and /health answers. Per round we record achieved concurrency (mean in-flight,
sampled), TTFT p50/p90, request count and failure count, so a capacity or
latency drift across rounds is visible even when no single round fails.

Volume is rate x duration: the producer paces submissions at `--rate`/s and
`--max-concurrency` caps how many are in flight at once. A round therefore
submits about rate x duration requests, and achieved concurrency settles at
rate x mean-latency (capped). Set `--rate 0` to run unpaced (closed loop at
the concurrency cap).

Deadline
--------
A request with no terminal event -- response, error, or cancel acknowledgement
-- within `--deadline` seconds is recorded as a hang with its id and elapsed
time, and fails the run. This covers the terminal admit path
(AdmitRuntimeError -> _fail_requests): a terminally failed request must produce
an error, not silence.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------- endpoints

def chat(model, seed):
    return "/v1/chat/completions", {
        "model": model, "stream": True, "seed": seed, "max_tokens": 128,
        "messages": [{"role": "user", "content": "Describe a mountain at dawn."}],
    }

def images(model, seed):
    return "/v1/images/generations", {
        "model": model, "prompt": "a red bicycle against a white wall",
        "seed": seed, "n": 1, "size": "512x512",
    }

def speech(model, seed):
    return "/v1/audio/speech", {
        "model": model, "input": "The quick brown fox jumps over the lazy dog.",
        "voice": "default", "seed": seed, "stream": True,
    }

def videos(model, seed):
    return "/v1/videos/generations", {
        "model": model, "prompt": "waves breaking on a rocky shore",
        "seed": seed, "num_frames": 16,
    }

ENDPOINTS = {"chat": chat, "images": images, "speech": speech, "videos": videos}

# ---------------------------------------------------------------- accounting

@dataclass
class Result:
    seq: int
    endpoint: str
    seed: int
    cancel_at: str | None
    rnd: int = 0
    state: str = "pending"        # ok | cancelled | error | timeout | hang | empty
    detail: str = ""
    chunks: int = 0
    content_chunks: int = 0
    ttft: float | None = None
    elapsed: float = 0.0
    t_start: float = 0.0          # wall clock at send
    t_end: float = 0.0            # wall clock at terminal event
    server_id: str | None = None  # x-request-id if the server sets one

@dataclass
class RoundStats:
    rnd: int
    duration: float = 0.0
    submitted: int = 0
    failures: int = 0             # error | timeout | hang | empty, any request
    hangs: int = 0
    ttft_p50: float | None = None
    ttft_p90: float | None = None
    # per endpoint: mixing endpoints makes the pooled median bimodal and its
    # round-over-round drift meaningless, so the threshold is applied per shape
    ttft_by_ep: dict = field(default_factory=dict)
    achieved_concurrency: float = 0.0
    peak_concurrency: int = 0
    shm_after: int = 0
    health_ok: bool = False
    load_seconds: float = 0.0
    drain_seconds: float = 0.0
    conc_samples: int = 0

@dataclass
class Run:
    results: list[Result] = field(default_factory=list)
    rounds: list[RoundStats] = field(default_factory=list)
    shm_baseline: int = 0
    shm_samples: list[tuple[float, int]] = field(default_factory=list)
    seq_counter: int = 0

def shm_count(uid: int | None = None) -> int:
    """Segments in /dev/shm owned by this uid.

    Other users' processes on this node create segments too -- on a shared
    login node they outnumber ours by orders of magnitude -- so an unfiltered
    count measures their jobs, not ours.
    """
    uid = os.getuid() if uid is None else uid
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return -1
    n = 0
    for name in names:
        try:
            if os.stat(os.path.join("/dev/shm", name)).st_uid == uid:
                n += 1
        except OSError:
            pass          # raced away between listdir and stat
    return n

def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]

# ---------------------------------------------------------------- payloads

def has_content(ep_name: str, raw: bytes) -> bool:
    """Does this streamed line carry generated output, as opposed to framing?

    A worker that fails mid-stream can still close the stream cleanly: the
    chat surface emits a role delta and a finish_reason with no content and
    returns HTTP 200. Counting those as successes hides exactly the terminal
    failures this harness exists to catch, so success requires real output.
    """
    b = raw.strip()
    if not b or b == b"data: [DONE]":
        return False
    if b.startswith(b"data:"):
        b = b[5:].strip()
    try:
        obj = json.loads(b)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    for ch in obj.get("choices", []) or []:
        d = ch.get("delta") or ch.get("message") or {}
        if d.get("content"):
            return True
    if obj.get("data") or obj.get("b64_json") or obj.get("url"):
        return True
    return False

# ---------------------------------------------------------------- one request

async def _stream(session, base, path, body, r: Result, cancel_at, timeout):
    """Drive one request to a terminal state, writing it onto `r`."""
    cut_after = random.randint(2, 8) if cancel_at == "mid" else None
    t0 = time.perf_counter()
    try:
        async with session.post(base + path, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            r.server_id = resp.headers.get("x-request-id")
            if resp.status >= 400:
                r.state, r.detail = "error", f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return
            if cancel_at == "early":
                # drop before reading anything: nothing has streamed yet, which
                # for image/video means the whole generation is still ahead
                resp.close()
                r.state = "cancelled"
                return
            # Read raw chunks and split on newlines ourselves. Iterating
            # `resp.content` directly uses readline(), which caps at 512 KB and
            # raises LineTooLong on a non-streamed image body (one ~1.6 MB JSON
            # line) -- and LineTooLong is not a ClientError, so it would escape
            # the handlers below and the request would vanish without a Result.
            def take(line: bytes) -> bool:
                r.chunks += 1
                if has_content(r.endpoint, line):
                    r.content_chunks += 1
                    if r.ttft is None:
                        # first real output, not the role/framing chunk that
                        # the server emits before generation starts
                        r.ttft = time.perf_counter() - t0
                return cut_after is not None and r.chunks >= cut_after

            buf = b""
            cut = False
            async for blob in resp.content.iter_any():
                buf += blob
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip() and take(line):
                        cut = True
                        break
                if cut:
                    break
            if not cut and buf.strip():
                take(buf)
            if cut:
                resp.close()
                r.state = "cancelled"
                return
            if cancel_at == "late":
                r.state = "cancelled"   # stream ended first; count it honestly
                r.detail = "stream ended before the late cut"
            elif r.content_chunks == 0:
                r.state = "empty"
                r.detail = f"stream closed with no output ({r.chunks} framing chunks)"
            else:
                r.state = "ok"
    except asyncio.TimeoutError:
        # aiohttp's own ClientTimeout. Kept distinct from a deadline hang so a
        # run with --timeout < --deadline still reports something sensible.
        r.state, r.detail = "timeout", f"no completion in {timeout}s"
    except aiohttp.ClientError as e:
        r.state, r.detail = ("cancelled" if cancel_at else "error"), f"{type(e).__name__}: {e}"
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # any other failure is still a result -- a request that raises here and
        # is never recorded silently shrinks the denominator
        r.state, r.detail = "error", f"{type(e).__name__}: {str(e)[:200]}"

async def fire(session, base, ep_name, model, seq, seed, cancel_at,
               timeout, deadline, rnd) -> Result:
    path, body = ENDPOINTS[ep_name](model, seed)
    r = Result(seq=seq, endpoint=ep_name, seed=seed, cancel_at=cancel_at, rnd=rnd)
    r.t_start = time.time()
    t0 = time.perf_counter()
    try:
        if isinstance(cancel_at, float):
            # Time-based cancel (--cancel-at-seconds). The chunk-position
            # cancels above key off `r.chunks`, which never advances on the
            # non-streamed image endpoint: /v1/images/generations emits one
            # ~1.6 MB body at the very end, so "mid" can only ever fire after
            # the whole generation is done. A wall-clock cut is the only way to
            # drop an image request while it is still generating.
            inner = asyncio.create_task(
                _stream(session, base, path, body, r, None, timeout))
            _, pending = await asyncio.wait({inner},
                                            timeout=min(cancel_at, deadline))
            if pending:
                inner.cancel()
                try:
                    await inner
                except asyncio.CancelledError:
                    pass
                r.state = "cancelled"
                r.detail = f"socket dropped {cancel_at:.2f}s after submit"
            else:
                inner.result()   # re-raise anything _stream let escape
        else:
            await asyncio.wait_for(
                _stream(session, base, path, body, r, cancel_at, timeout),
                timeout=deadline,
            )
    except asyncio.TimeoutError:
        # no terminal event inside the deadline -- the invariant this exists for
        r.state = "hang"
        r.detail = f"no terminal event in {deadline}s"
    finally:
        r.elapsed = time.perf_counter() - t0
        r.t_end = time.time()
    return r

# ---------------------------------------------------------------- health

async def health_ok(session, base, timeout=10.0) -> bool:
    try:
        async with session.get(base + "/health",
                               timeout=aiohttp.ClientTimeout(total=timeout)) as h:
            return h.status == 200
    except Exception:
        return False

# ---------------------------------------------------------------- one round

async def run_round(args, run: Run, session, rnd: int) -> RoundStats:
    st = RoundStats(rnd=rnd)
    stop_at = time.time() + args.duration
    sem = asyncio.Semaphore(args.max_concurrency)
    inflight: set[asyncio.Task] = set()
    samples: list[int] = []
    t_round0 = time.time()

    stop_sampling = asyncio.Event()
    async def sampler():
        while not stop_sampling.is_set():
            samples.append(len(inflight))
            await asyncio.sleep(args.sample_interval)
    samp = asyncio.create_task(sampler())

    async def one(seq, ep, seed, cancel_at):
        try:
            res = await fire(session, args.url, ep, args.model, seq, seed,
                             cancel_at, args.timeout, args.deadline, rnd)
            run.results.append(res)
            if res.state in ("error", "timeout", "hang", "empty"):
                print(f"  [{res.state}] r{rnd} seq={res.seq} {res.endpoint} "
                      f"seed={res.seed} cancel={res.cancel_at} "
                      f"elapsed={res.elapsed:.1f}s :: {res.detail}", flush=True)
        finally:
            sem.release()

    last_sample = 0.0
    while time.time() < stop_at:
        await sem.acquire()
        if time.time() >= stop_at:
            sem.release()
            break
        ep = random.choice(args.endpoints)
        seed = random.randint(0, 2**31 - 1)
        if random.random() < args.cancel_rate:
            cancel_at = (random.uniform(*args.cancel_at_seconds)
                         if args.cancel_at_seconds
                         else random.choice(["early", "mid", "late"]))
        else:
            cancel_at = None
        t = asyncio.create_task(one(run.seq_counter, ep, seed, cancel_at))
        inflight.add(t)
        t.add_done_callback(inflight.discard)
        run.seq_counter += 1
        st.submitted += 1
        now = time.time()
        if now - last_sample > 30:
            run.shm_samples.append((now, shm_count()))
            last_sample = now
        if args.rate > 0:
            await asyncio.sleep(1.0 / args.rate)

    # ---- end of load phase. Stop sampling here: the drain tail is a
    # ramp-down whose length varies run to run, and mixing it into the mean
    # skews achieved concurrency between identical rounds.
    stop_sampling.set()
    await asyncio.gather(samp, return_exceptions=True)
    st.load_seconds = time.time() - t_round0

    # ---- drain to idle: everything finishes, then quiet, then /health
    t_drain0 = time.time()
    if inflight:
        print(f"  round {rnd}: draining {len(inflight)} in-flight...", flush=True)
        await asyncio.gather(*list(inflight), return_exceptions=True)
    quiet_from = time.time()
    while time.time() - quiet_from < args.idle_seconds:
        if inflight:
            quiet_from = time.time()
        await asyncio.sleep(0.25)
    st.drain_seconds = time.time() - t_drain0
    st.duration = time.time() - t_round0

    st.health_ok = await health_ok(session, args.url)
    st.shm_after = shm_count()

    mine = [r for r in run.results if r.rnd == rnd]
    ttfts = [r.ttft for r in mine if r.state == "ok" and r.ttft is not None]
    st.ttft_p50 = pct(ttfts, 0.50)
    st.ttft_p90 = pct(ttfts, 0.90)
    for ep in sorted({r.endpoint for r in mine}):
        e = [r.ttft for r in mine
             if r.endpoint == ep and r.state == "ok" and r.ttft is not None]
        st.ttft_by_ep[ep] = {"p50": pct(e, 0.50), "p90": pct(e, 0.90), "n": len(e)}
    st.failures = sum(1 for r in mine if r.state in ("error", "timeout", "hang", "empty"))
    st.hangs = sum(1 for r in mine if r.state == "hang")
    # Time-average in-flight over the steady part of the load phase. Zeros are
    # kept: a server that stalls goes idle, and filtering idle samples would
    # report the stalled run at full concurrency.
    steady = samples[int(len(samples) * args.ramp_frac):] if samples else []
    st.achieved_concurrency = statistics.fmean(steady) if steady else 0.0
    st.peak_concurrency = max(samples) if samples else 0
    st.conc_samples = len(steady)

    print(f"  round {rnd}: submitted={st.submitted} failures={st.failures} "
          f"hangs={st.hangs} conc={st.achieved_concurrency:.2f} "
          f"ttft_p50={st.ttft_p50 if st.ttft_p50 is None else round(st.ttft_p50,3)} "
          f"ttft_p90={st.ttft_p90 if st.ttft_p90 is None else round(st.ttft_p90,3)} "
          f"shm={st.shm_after} health={st.health_ok} "
          f"dur={st.duration:.0f}s drain={st.drain_seconds:.0f}s", flush=True)
    return st

# ---------------------------------------------------------------- driver

async def drive(args, run: Run):
    conn = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))
    async with aiohttp.ClientSession(connector=conn) as session:
        run.shm_baseline = shm_count()
        print(f"baseline /dev/shm segments (uid {os.getuid()}): {run.shm_baseline}",
              flush=True)
        for rnd in range(1, args.rounds + 1):
            st = await run_round(args, run, session, rnd)
            run.rounds.append(st)

# ---------------------------------------------------------------- invariants

def report(args, run: Run) -> int:
    states = Counter(r.state for r in run.results)
    total = len(run.results)
    print("\n" + "=" * 62)
    print(f"submitted {total}   " + "  ".join(f"{k}={v}" for k, v in sorted(states.items())))

    by_ep = Counter((r.endpoint, r.state) for r in run.results)
    for ep in sorted(args.endpoints):
        row = {s: by_ep.get((ep, s), 0) for s in ("ok", "cancelled", "error", "timeout", "hang", "empty")}
        print(f"  {ep:8} ok={row['ok']:5} cancelled={row['cancelled']:5} "
              f"error={row['error']:4} timeout={row['timeout']:4} hang={row['hang']:4} empty={row['empty']:4}")

    oks = [r.ttft for r in run.results if r.state == "ok" and r.ttft is not None]
    if oks:
        print(f"  ttft p50={pct(oks,.5):.3f}s p90={pct(oks,.9):.3f}s "
              f"p95={pct(oks,.95):.3f}s  (n={len(oks)})")

    print("-" * 62)
    print("  round  submitted  fails  hangs   conc   ttft_p50  ttft_p90  shm  health")
    for s in run.rounds:
        p50 = "  n/a  " if s.ttft_p50 is None else f"{s.ttft_p50:7.3f}"
        p90 = "  n/a  " if s.ttft_p90 is None else f"{s.ttft_p90:7.3f}"
        print(f"  {s.rnd:5d}  {s.submitted:9d}  {s.failures:5d}  {s.hangs:5d}  "
              f"{s.achieved_concurrency:6.2f}  {p50}   {p90}  {s.shm_after:4d}  {s.health_ok}")

    print("-" * 62)
    failures = []

    # 1. uncancelled requests must not fail
    bad = [r for r in run.results
           if r.cancel_at is None and r.state in ("error", "timeout", "hang", "empty")]
    if bad:
        failures.append(f"{len(bad)} uncancelled request(s) failed, first: seq={bad[0].seq} "
                        f"r{bad[0].rnd} {bad[0].endpoint} seed={bad[0].seed} :: {bad[0].detail}")

    # 2. a cancel must not take down anything else: any failure whose terminal
    #    event lands within --collateral-seconds of a cancel's, either side
    cancel_times = sorted(r.t_end for r in run.results if r.state == "cancelled")
    collateral = []
    for f in (r for r in run.results if r.state in ("error", "timeout", "hang", "empty")):
        i = bisect.bisect_left(cancel_times, f.t_end)
        near = cancel_times[max(0, i - 1): i + 1]
        if any(abs(f.t_end - ct) <= args.collateral_seconds for ct in near):
            collateral.append(f)
    if collateral:
        failures.append(
            f"{len(collateral)} failure(s) within {args.collateral_seconds}s of a cancel "
            f"(seqs {[r.seq for r in collateral][:5]})")

    # 3. shm segments owned by this uid must return to baseline
    final = shm_count()
    drift = final - run.shm_baseline
    print(f"  /dev/shm(uid {os.getuid()})  baseline={run.shm_baseline} final={final} drift={drift:+d}")
    if run.shm_samples:
        print(f"            peak during run={max(c for _, c in run.shm_samples)}")
    if drift > args.shm_tolerance:
        failures.append(f"/dev/shm leaked {drift} segments (tolerance {args.shm_tolerance})")

    # 4. the server must answer /health after every round
    dead = [s.rnd for s in run.rounds if not s.health_ok]
    print(f"  /health answered after rounds: "
          f"{[s.rnd for s in run.rounds if s.health_ok]} of {len(run.rounds)}")
    if dead:
        failures.append(f"server did not answer /health after round(s) {dead}")

    # 5. deadline: no request may go --deadline seconds without a terminal event
    hangs = [r for r in run.results if r.state == "hang"]
    if hangs:
        failures.append(
            f"{len(hangs)} request(s) exceeded the {args.deadline}s deadline: "
            + "; ".join(f"seq={h.seq} r{h.rnd} {h.endpoint} elapsed={h.elapsed:.1f}s"
                        for h in hangs[:5]))

    # 6. round over round: TTFT p50 and achieved concurrency must hold within
    #    --drift-pct of round 1, and must not drift monotonically past it
    def drift_check(name, vals):
        if len(vals) < 2 or vals[0] in (None, 0):
            return
        base = vals[0]
        for i, v in enumerate(vals[1:], start=2):
            if v is None:
                continue
            d = abs(v - base) / base
            if d > args.drift_pct:
                failures.append(f"{name} round {i} is {d*100:.1f}% from round 1 "
                                f"({v:.4g} vs {base:.4g}, tolerance {args.drift_pct*100:.0f}%)")
        clean = [v for v in vals if v is not None]
        if len(clean) >= 3:
            inc = all(b >= a for a, b in zip(clean, clean[1:], strict=False))
            dec = all(b <= a for a, b in zip(clean, clean[1:], strict=False))
            tot = abs(clean[-1] - clean[0]) / clean[0] if clean[0] else 0
            if (inc or dec) and tot > args.drift_pct:
                failures.append(f"{name} drifts monotonically {tot*100:.1f}% across "
                                f"{len(clean)} rounds ({clean[0]:.4g} -> {clean[-1]:.4g})")

    for ep in sorted(args.endpoints):
        vals = [s.ttft_by_ep.get(ep, {}).get("p50") for s in run.rounds]
        if any(v is not None for v in vals):
            drift_check(f"TTFT p50 [{ep}]", vals)
    drift_check("achieved concurrency", [s.achieved_concurrency for s in run.rounds])

    print("-" * 62)
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS - no invariant violated")
    print("=" * 62)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "shm_baseline": run.shm_baseline,
                       "shm_final": final, "shm_samples": run.shm_samples,
                       "uid": os.getuid(),
                       "failures": failures,
                       "rounds": [vars(s) for s in run.rounds],
                       "results": [vars(r) for r in run.results]}, fh, indent=1)
        print(f"wrote {args.out}")
    return 1 if failures else 0

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="http://host:port of either surface")
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoints", nargs="+", default=["chat"], choices=list(ENDPOINTS))
    ap.add_argument("--duration", type=float, default=600, help="seconds per round")
    ap.add_argument("--rounds", type=int, default=1, help="load rounds, drained to idle between")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="submissions/sec; 0 = unpaced (closed loop at the cap)")
    ap.add_argument("--max-concurrency", "--concurrency", dest="max_concurrency",
                    type=int, default=8, help="cap on in-flight requests")
    ap.add_argument("--cancel-rate", type=float, default=0.2, help="fraction cancelled")
    ap.add_argument("--cancel-at-seconds", nargs=2, type=float, metavar=("MIN", "MAX"),
                    default=None,
                    help="cancel by wall clock instead of chunk position: drop the "
                         "socket uniformly between MIN and MAX seconds after submit. "
                         "Required for the non-streamed image endpoint, where the "
                         "early/mid/late cancel points cannot fire mid-generation.")
    ap.add_argument("--timeout", type=float, default=600, help="aiohttp client timeout")
    ap.add_argument("--deadline", type=float, default=120,
                    help="no terminal event within this many seconds = hang")
    ap.add_argument("--idle-seconds", type=float, default=5.0,
                    help="quiet period required between rounds")
    ap.add_argument("--sample-interval", type=float, default=0.1,
                    help="in-flight sampling period for achieved concurrency")
    ap.add_argument("--ramp-frac", type=float, default=0.20,
                    help="leading fraction of a round's samples dropped as ramp-up")
    ap.add_argument("--shm-tolerance", type=int, default=10)
    ap.add_argument("--drift-pct", type=float, default=0.05,
                    help="round-over-round tolerance for TTFT p50 and concurrency")
    ap.add_argument("--collateral-seconds", type=float, default=2.0,
                    help="a failure this close to a cancel counts as collateral")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write a JSON report here")
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"chaos: {args.url} model={args.model} endpoints={args.endpoints} "
          f"rounds={args.rounds} x {args.duration}s @ {args.rate}/s "
          f"maxconc={args.max_concurrency} cancel={args.cancel_rate} "
          f"deadline={args.deadline}s seed={args.seed}")
    run = Run()
    try:
        asyncio.run(drive(args, run))
    except KeyboardInterrupt:
        print("\ninterrupted, reporting on what completed")
    sys.exit(report(args, run))

if __name__ == "__main__":
    main()
