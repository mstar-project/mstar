"""``compare --stress``: load-dependent races (cancels, drops, id reuse), base against head by failed-run counts.

Each arm's server takes ``runs`` runs of a workload in which every client holds one pinned ``request_id`` and
cycles through five patterns (one per client per round, mixed across clients):

  barge_in   stream, close the connection at a random chunk (``/generate`` has no cancel message: a close is it)
  drop       close the connection as soon as the response starts, before reading anything
  retry      stream, close after 3 chunks, resend at once under the same id and read to the end
  reuse      read a reply to the end, then send again under the same id and read that to the end
  overlap    send a second request under the id while the first is streaming: one of the pair should get a 409

A run fails on a hang (no progress for ``hang_s``), an HTTP 5xx, unexpected 4xx, empty reply or in-band error, a
cross-talk (a reply holding another prompt's answer: each prompt's answer has a signature, such as three
consecutive integers or two weekday names), a leak (the server's own M* tensor files in ``/dev/shm`` still above
the post-warm-up count ``settle_s`` after the run) or a crash (a process of the server's group exits). Every request
under one id asks a different prompt, so cross-talk within an id (a retry receiving its predecessor's tail) shows up
as well as across ids. Runs a crash or a server that did not start left unrun count as failed, so N stays the
scheduled one.

Verdict per model: a crash or cross-talk on head that base never shows fails outright; otherwise head fails when
its failed-run count beats base's at one-sided Fisher exact p < ``ALPHA``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PATTERNS = ("barge_in", "drop", "retry", "reuse", "overlap")
CLASSES = ("hang", "error", "crosstalk", "leak", "crash")
OUTRIGHT = ("crash", "crosstalk")
FAILED_REQUEST = ("hang", "error", "crosstalk")
ALPHA = 0.05
PER_SERVER = 5          # runs per server session: a fresh server every 5 runs
CAP_HANGS = 4           # a request still streaming after this many hang_s is a hang (max_tokens bounds a reply)
DEFAULTS = {"runs": 20, "seconds": 60.0, "clients": 32, "hang_s": 30.0, "settle_s": 30.0, "max_tokens": 64,
            "shm_dir": "/dev/shm"}
PROMPTS = (
    "Count from 1 to 40, separated by spaces.",
    "Write the English alphabet in capital letters, separated by spaces.",
    "Write the word banana twenty times, separated by spaces.",
    "List the days of the week, starting with Monday, separated by commas.",
    "List the planets of the solar system in order from the Sun, separated by commas.",
    "List the months of the year, starting with January, separated by commas.",
)


# ---------------------------------------------------------------- one request

@dataclass
class Reply:
    prompt: int
    status: int | None = None
    text: str = ""
    chunks: int = 0
    error: str = ""            # in-band error, HTTP error detail or connection error
    error_status: int | None = None
    hang: bool = False
    closed: bool = False       # the client closed it on purpose (barge-in, drop, retry's first half)
    result: str = ""


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


def _ascending(tok: list[str], n: int, ok, step) -> bool:
    """``n`` adjacent tokens, each ``ok`` and one ``step`` after the previous (1 2 3, or b c d e)."""
    for i in range(len(tok) - n + 1):
        w = tok[i:i + n]
        if all(ok(t) for t in w) and all(step(w[j], w[j + 1]) for j in range(n - 1)):
            return True
    return False


def _names(names: set[str]):
    return lambda tok: len(names & set(tok)) >= 2


def _letters(tok: list[str]) -> bool:
    """A B C D, or ABCD written as one word (four consecutive letters inside an English word are rare)."""
    def step(a: str, b: str) -> bool:
        return ord(b) == ord(a) + 1

    return (_ascending(tok, 4, lambda t: len(t) == 1 and t.isalpha(), step)
            or any(_ascending(list(t), 4, str.isalpha, step) for t in tok if len(t) >= 4))


# What only each prompt's answer holds, whatever prose surrounds it: a predecessor's tail ("13 14 15") still shows.
# "may" and "march" are left out of the months: they are common words.
SIGNATURES = (
    lambda tok: _ascending(tok, 3, str.isdigit, lambda a, b: int(b) == int(a) + 1),
    _letters,
    lambda tok: "banana" in tok or "bananas" in tok,
    _names({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}),
    _names({"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}),
    _names({"january", "february", "april", "june", "july", "august", "september", "october", "november",
            "december"}),
)


class Refs:
    """Each prompt's quiet reply. A signature another prompt's quiet reply already holds is not cross-talk there."""

    def __init__(self, texts: dict[int, str]):
        self.texts = texts
        self.ignore = {p: {q for q in range(len(PROMPTS)) if q != p and SIGNATURES[q](tokens(t))}
                       for p, t in texts.items()}

    def crosstalk(self, text: str, own: int) -> bool:
        """A reply holding another prompt's answer (all of it, or a tail)."""
        tok = tokens(text)
        return any(SIGNATURES[q](tok) for q in range(len(PROMPTS)) if q != own and q not in self.ignore.get(own, ()))

    def matches(self, text: str, own: int) -> bool:
        return SIGNATURES[own](tokens(text))


def judge(r: Reply, refs: Refs, expect_conflict: bool = False) -> str:
    """One request's outcome: ok, closed, conflict (an expected 409), unmatched (own reply not recognised, judged
    as nondeterminism), or a failure: hang, error, crosstalk."""
    if r.hang:
        return "hang"
    if refs.crosstalk(r.text, r.prompt):      # before a 409: a superseded stream may carry foreign text first
        return "crosstalk"
    if r.status == 409 or r.error_status == 409:
        return "conflict" if expect_conflict else "error"
    if r.error or r.status is None or r.status >= 400:
        return "error"
    if r.closed:
        return "closed"
    if not r.text:
        return "error"         # an empty 200
    return "ok" if refs.matches(r.text, r.prompt) else "unmatched"


async def send(session, url: str, rid: str, prompt: int, cfg: dict, *, stop_after: int | None = None,
               drop: bool = False, started: asyncio.Event | None = None) -> Reply:
    """POST /generate (streamed NDJSON) under ``rid``. No text or end for ``hang_s``, or no end after
    ``CAP_HANGS * hang_s``, is a hang: blank or unknown lines are not progress."""
    import aiohttp

    r, hang_s = Reply(prompt), cfg["hang_s"]
    t0 = last = time.monotonic()

    def left() -> float:
        now = time.monotonic()
        return max(0.0, min(last + hang_s, t0 + CAP_HANGS * hang_s) - now)
    form = aiohttp.FormData()
    form.add_field("text", PROMPTS[prompt])
    form.add_field("request_id", rid)
    form.add_field("streaming", "true")
    form.add_field("model_kwargs", json.dumps({**cfg.get("model_kwargs", {}), "temperature": 0.0,
                                               "max_output_tokens": cfg["max_tokens"]}))
    resp = None
    try:
        resp = await asyncio.wait_for(session.post(f"{url}/generate", data=form), left())
        r.status = resp.status
        if resp.status != 200:
            body = await asyncio.wait_for(resp.read(), left())
            r.error = body[:300].decode("utf-8", "replace") or f"HTTP {resp.status}"
            return r
        if drop:
            r.closed = True
            return r
        while True:
            line = await asyncio.wait_for(resp.content.readline(), left())
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("modality") == "error":
                r.error = _b64(msg.get("data"))[:300] or "in-band error"
                r.error_status = (msg.get("metadata") or {}).get("status")
                break
            if msg.get("modality") == "text":
                r.text += _b64(msg.get("data"))
                r.chunks += 1
                last = time.monotonic()
                if started is not None:
                    started.set()
                if stop_after is not None and r.chunks >= stop_after:
                    r.closed = True
                    break
    except asyncio.TimeoutError:
        r.hang = True
    except (aiohttp.ClientError, OSError) as e:
        r.error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if started is not None:
            started.set()
        if resp is not None:
            resp.close()        # also the close that cancels a barge-in or a drop
    return r


def _b64(data) -> str:
    import base64

    try:
        return base64.b64decode(data or "").decode("utf-8", "replace")
    except ValueError:
        return ""


# ---------------------------------------------------------------- the workload

class Client:
    def __init__(self, idx: int, tag: str, cfg: dict, refs: Refs, rng: random.Random):
        self.rid, self.cfg, self.refs, self.rng = f"stress-{tag}-{idx}", cfg, refs, rng
        self.idx, self.n = idx, 0
        self.outcomes: list[tuple[str, str]] = []      # (pattern, result)
        self.examples: dict[str, list[str]] = {}

    def prompt(self) -> int:
        """Consecutive requests under one id never ask the same prompt."""
        self.n += 1
        return (self.idx + self.n) % len(PROMPTS)

    def record(self, pattern: str, r: Reply) -> None:
        self.outcomes.append((pattern, r.result))
        if r.result in FAILED_REQUEST:
            ex = self.examples.setdefault(r.result, [])
            if len(ex) < 3:
                ex.append(f"{pattern}: {r.error or r.text[:80]!r}" if r.result != "hang" else f"{pattern}: no progress")

    async def round(self, session, url: str, pattern: str) -> None:
        cfg, refs = self.cfg, self.refs
        if pattern == "barge_in":
            r = await send(session, url, self.rid, self.prompt(), cfg, stop_after=self.rng.randint(1, 8))
            r.result = judge(r, refs)
            self.record(pattern, r)
        elif pattern == "drop":
            r = await send(session, url, self.rid, self.prompt(), cfg, drop=True)
            r.result = judge(r, refs)
            self.record(pattern, r)
        elif pattern in ("retry", "reuse"):
            first = await send(session, url, self.rid, self.prompt(), cfg,
                               stop_after=3 if pattern == "retry" else None)
            first.result = judge(first, refs)
            self.record(pattern, first)
            second = await send(session, url, self.rid, self.prompt(), cfg)
            second.result = judge(second, refs)
            self.record(pattern, second)
        elif pattern == "overlap":
            started = asyncio.Event()
            a_task = asyncio.ensure_future(send(session, url, self.rid, self.prompt(), cfg, started=started))
            try:
                await started.wait()      # set on the first chunk, or when the first request ends (send is bounded)
                b = await send(session, url, self.rid, self.prompt(), cfg)
                a = await a_task
            finally:
                if not a_task.done():
                    a_task.cancel()
                    await asyncio.gather(a_task, return_exceptions=True)
            a.result, b.result = judge(a, refs, True), judge(b, refs, True)
            if a.result == b.result == "conflict":
                b.result = "error"        # one of the pair must be served
            elif "conflict" not in (a.result, b.result) and b.result in ("ok", "unmatched"):
                b.result = "accepted_twice"
            self.record(pattern, a)
            self.record(pattern, b)

    async def loop(self, session, url: str, deadline: float, patterns: tuple[str, ...]) -> None:
        k = self.idx
        while time.monotonic() < deadline:
            pattern = patterns[k % len(patterns)]
            k += 1
            before = len(self.outcomes)
            await self.round(session, url, pattern)
            if any(res == "error" for _, res in self.outcomes[before:]):
                await asyncio.sleep(0.2)  # a dead server refuses at once; do not spin


async def _session():
    import aiohttp

    # One connection per request: two requests of one client (overlap) never share a socket.
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
                                 timeout=aiohttp.ClientTimeout(total=None))


async def warm(url: str, cfg: dict) -> Refs:
    """Each prompt once on the quiet server, under fresh ids: the references for this server session."""
    async with await _session() as s:
        texts = {}
        for p in range(len(PROMPTS)):
            r = await send(s, url, f"stress-warm-{uuid.uuid4().hex[:8]}", p, cfg)
            if r.hang or r.error or r.status != 200 or not r.text:
                raise RuntimeError(f"warm-up request {p} failed: {'no progress' if r.hang else r.error or r.status}")
            texts[p] = r.text
    return Refs(texts)


async def probe(url: str, cfg: dict, refs: Refs) -> str:
    """A fresh request after the run: a server that no longer answers fails the run even if no client saw it."""
    async with await _session() as s:
        r = await send(s, url, f"stress-probe-{uuid.uuid4().hex[:8]}", 0, cfg)
    return judge(r, refs)


async def workload(url: str, cfg: dict, refs: Refs, tag: str, seed: int) -> dict:
    patterns = tuple(cfg.get("patterns") or PATTERNS)
    clients = [Client(i, tag, cfg, refs, random.Random(seed * 1000 + i)) for i in range(cfg["clients"])]
    t0 = time.monotonic()
    async with await _session() as s:
        await asyncio.gather(*(c.loop(s, url, t0 + cfg["seconds"], patterns) for c in clients))
    results: Counter = Counter()
    by_pattern: dict[str, Counter] = {}
    examples: dict[str, list[str]] = {}
    for c in clients:
        for pattern, res in c.outcomes:
            results[res] += 1
            by_pattern.setdefault(pattern, Counter())[res] += 1
        for k, v in c.examples.items():
            ex = examples.setdefault(k, [])
            ex.extend(v[:max(0, 3 - len(ex))])
    return {"requests": sum(results.values()), "results": dict(results),
            "by_pattern": {p: dict(v) for p, v in by_pattern.items()},
            "classes": sorted(k for k in FAILED_REQUEST if results.get(k)), "examples": examples,
            "elapsed_s": round(time.monotonic() - t0, 1)}


# ---------------------------------------------------------------- the server side (on the GPU host)

def server_namespaces(server) -> set[str]:
    """This server's deployment namespace in its tensor file names (``mstar_<ns>_<entity>_<uuid>``): the SHA-1 of
    its launch token (``launch_tokens``) and its conductor's IPC endpoint (``tensors._deployment_namespace``), found
    under its ``--socket-path-prefix``. With a token, the endpoint-only SHA-1 is kept beside it: a build without the
    token in its names inherits one from a harness environment that carries it. Empty when there is no such socket
    (another layout): its tensor files then go uncounted, never misattributed."""
    try:
        cmd = server.command()
        prefix = Path(cmd[cmd.index("--socket-path-prefix") + 1])
    except (AttributeError, AssertionError, ValueError, IndexError):
        return set()
    tokens = launch_tokens(server.pid) if getattr(server, "pid", None) else set()
    keys = [f"ipc://{os.path.realpath(p)}" for p in prefix.rglob("conductor.ipc")]
    keys += [f"{token}\0{k}" for token in sorted(tokens) for k in list(keys)]
    return {hashlib.sha1(k.encode()).hexdigest()[:12] for k in keys}


def _proc_env_token(pid: int) -> str | None:
    """``MSTAR_DEPLOYMENT_TOKEN`` in ``/proc/<pid>/environ``, the environment at exec. A later ``os.environ`` write
    does not show up here, and an empty value counts as unset."""
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for kv in env:
        if kv.startswith(b"MSTAR_DEPLOYMENT_TOKEN="):
            token = kv.split(b"=", 1)[1].decode()
            return token or None
    return None


def launch_tokens(pgid: int) -> set[str]:
    """Tokens that name this launch's files.

    The API server mints ``MSTAR_DEPLOYMENT_TOKEN`` after it starts, so ``/proc`` of its own pid can still hold
    the empty or previous token it was exec'd with. The processes it spawned carry the token that names the
    files. With none of those, the server's own non-empty token is used: a launcher-set token is kept, and a
    single process is its own server."""
    pids = group_pids(pgid)
    spawned = {t for pid in pids if pid != pgid and (t := _proc_env_token(pid))}
    if spawned:
        return spawned
    own = _proc_env_token(pgid)
    return {own} if own else set()


def namespace_file(name: str, namespaces: set[str]) -> bool:
    """A tensor file under one of ``namespaces``. Arena segments named under it (``mstar_<ns>_arena_*``) are not:
    an arena keeps the segments it grew into until shutdown."""
    return any(name.startswith(f"mstar_{ns}_") and not name.startswith(f"mstar_{ns}_arena_") for ns in namespaces)


def unmapped_arena_segments(root: str, namespaces: set[str], pids: set[int]) -> list[str]:
    """Arena segments under one of ``namespaces`` in ``root`` that none of ``pids`` maps. A live arena maps every
    segment it grew, so these were left by a process that exited without its exit-time unlink (a crash, a
    SIGKILL) and that no live peer still reads. Asked only once a process of the server has exited: a segment the
    dead process grew and a live peer still maps is not counted, so this can miss a leak but never names a live
    arena's memory as one."""
    try:
        names = {n for n in os.listdir(root) if any(n.startswith(f"mstar_{ns}_arena_") for ns in namespaces)}
    except OSError:
        return []
    prefix = os.path.join(root, "")
    for pid in pids:
        if not names:
            break
        try:
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    path = line.split(None, 5)[5:]
                    if path and path[0].startswith(prefix):
                        names.discard(path[0].rstrip("\n").removesuffix(" (deleted)")[len(prefix):])
        except OSError:
            continue   # exited meanwhile, or not ours to read: it maps nothing we can see
    return sorted(names)


def shm_owner(name: str) -> int | None:
    """The pid in mstar's per-process names (``mstar_arena_<entity>_<pid>_<token>``, ``mstar_kv_<uid>_<pid>_<hash>``);
    None for a tensor file (``mstar_<deployment>_<entity>_<uuid>``), which goes once its request is done."""
    if not name.startswith(("mstar_arena_", "mstar_kv_")):
        return None
    parts = name.split(".", maxsplit=1)[0].rsplit("_", 2)
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else -1


def shm_entries(root: str, namespaces: set[str], pids: set[int]) -> int:
    """This server's M* files in ``root``: tensor files under one of its deployment tokens, and per-process ones
    of its processes. Other servers on the host (other GPU slots, other users' jobs) name theirs with another
    token or pid. Arena segments (``mstar_arena_*``, ``mstar_<ns>_arena_*``) are left out: an arena keeps the
    segments it grew into until shutdown, so their count rises under load without a leak."""
    try:
        names = [n for n in os.listdir(root) if n.startswith("mstar_") and not n.startswith("mstar_arena_")]
    except OSError:
        return -1
    n = 0
    for name in names:
        pid = shm_owner(name)
        n += namespace_file(name, namespaces) if pid is None else pid in pids
    return n


def helper_process(pid: int) -> bool:
    """A process the server's group holds that is not part of the server: Torch Inductor's compile-worker pool,
    which a cold start spawns (one per core) and which exits once compiles go quiet, so its exit is no crash."""
    try:
        return b"/torch/_inductor/compile_worker/" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def group_pids(pgid: int) -> set[int]:
    """The server's processes, helpers (``helper_process``) left out."""
    r = subprocess.run(["ps", "-o", "pid=", "-g", str(pgid)], capture_output=True, text=True, check=False)
    return {pid for pid in map(int, r.stdout.split()) if not helper_process(pid)}


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def settle(baseline: int, settle_s: float, root: str, namespaces: set[str], pids: set[int]) -> int:
    """The server's ``mstar_*`` count once it is back at the baseline, or after ``settle_s``."""
    deadline = time.monotonic() + settle_s
    n = shm_entries(root, namespaces, pids)
    while n > baseline and time.monotonic() < deadline:
        time.sleep(1.0)
        n = shm_entries(root, namespaces, pids)
    return n


def session_runs(cfg: dict, session: int) -> list[int]:
    first = session * PER_SERVER
    return list(range(first, min(cfg["runs"], first + PER_SERVER)))


def lost_runs(runs: list[int], arm: str, session: int, d: Path, classes: list[str], reason: str, publish,
              **extra) -> None:
    """Records for runs this session could not run: they count as failed, so N stays the scheduled one."""
    sd = d / "stress"
    sd.mkdir(parents=True, exist_ok=True)
    for r in runs:
        publish(f"run{r}.json", {"arm": arm, "session": session, "run": r, "classes": classes, "reason": reason,
                                 "lost": True, **extra}, sd)


def run_session(url: str, pgid: int, arm: str, session: int, d: Path, cfg: dict, status, log,
                publish, namespaces: set[str] = frozenset()) -> None:
    """This server session's share of the arm's runs, one record each under ``d/stress/``. ``namespaces``: the
    server's deployment tokens (``server_namespaces``), so the leak count is its own files only."""
    runs = session_runs(cfg, session)
    sd = d / "stress"
    sd.mkdir(parents=True, exist_ok=True)
    base_rec = {"arm": arm, "session": session}
    procs = group_pids(pgid)
    try:
        refs = asyncio.run(warm(url, cfg))
    except RuntimeError as e:
        crashed = any(not alive(p) for p in procs)
        lost_runs(runs, arm, session, d, ["crash", "error"] if crashed else ["error"], str(e)[:300], publish)
        log(f"{arm} s{session}: {e}")
        return
    (sd / "refs.json").write_text(json.dumps(refs.texts))
    shm = cfg.get("shm_dir", DEFAULTS["shm_dir"])
    procs = group_pids(pgid)
    shm0 = shm_entries(shm, namespaces, procs)
    for i, r in enumerate(runs):
        status(phase="stress", arm=arm, session=session, run=r)
        res = asyncio.run(workload(url, cfg, refs, f"{arm[0]}{r}", seed=cfg.get("seed", 0) + r))
        classes = set(res["classes"])
        after = probe_result = None
        gone = sorted(p for p in procs if not alive(p))
        if not gone:
            probe_result = asyncio.run(probe(url, cfg, refs))
            if probe_result in FAILED_REQUEST:
                classes.add(probe_result)
            after = settle(shm0, cfg["settle_s"], shm, namespaces, procs | group_pids(pgid))
            if shm0 >= 0 and after > shm0:
                classes.add("leak")
            gone = sorted(p for p in procs if not alive(p))
        if gone:
            classes.add("crash")
        rec = {**base_rec, "run": r, **res, "classes": sorted(classes), "probe": probe_result,
               "shm_idle": shm0, "shm_after": after, "exited": len(gone)}
        publish(f"run{r}.json", rec, sd)
        log(f"{arm} s{session} run {r}: {res['requests']} requests, {dict(res['results'])}, "
            f"classes {sorted(classes) or 'none'}")
        if gone:
            lost_runs(runs[i + 1:], arm, session, d, ["crash"], f"not run: the server exited during run {r}",
                      publish)
            left = unmapped_arena_segments(shm, namespaces, group_pids(pgid))
            log(f"{arm} s{session}: {len(gone)} server process(es) exited"
                + (f", leaving {len(left)} arena segment(s) no live process maps" if left else "")
                + "; the next session starts a fresh server")
            return


# ---------------------------------------------------------------- statistics and verdict

def fisher_one_sided(head_fail: int, head_n: int, base_fail: int, base_n: int) -> float:
    """P(head has at least ``head_fail`` failed runs) when failures fall on the two arms at random (one-sided
    Fisher exact test, hypergeometric upper tail)."""
    k, n = head_fail + base_fail, head_n + base_n
    total = math.comb(n, head_n)
    return min(1.0, sum(math.comb(k, x) * math.comb(n - k, head_n - x)
                        for x in range(head_fail, min(k, head_n) + 1)) / total)


def detectable_gap(n: int, alpha: float = ALPHA) -> int | None:
    """The fewest head failures out of ``n`` that base's 0 out of ``n`` flags; None when ``n`` can flag none."""
    return next((k for k in range(1, n + 1) if fisher_one_sided(k, n, 0, n) < alpha), None)


def gap_text(n: int) -> str:
    k = detectable_gap(n)
    if k is None:
        return f"N = {n} runs per arm cannot reach p < {ALPHA} even at 0 vs {n} failed runs"
    return f"N = {n} runs per arm; the smallest gap it can detect at p < {ALPHA} is 0 vs {k} failed runs"


@dataclass
class Arm:
    runs: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.runs if r.get("classes"))

    def with_class(self, c: str) -> int:
        return sum(1 for r in self.runs if c in r.get("classes", []))

    @property
    def requests(self) -> int:
        return sum(r.get("requests", 0) for r in self.runs)

    @property
    def failed_requests(self) -> int:
        return sum(r.get("results", {}).get(k, 0) for r in self.runs for k in FAILED_REQUEST)

    def results(self) -> Counter:
        c: Counter = Counter()
        for r in self.runs:
            c.update(r.get("results", {}))
        return c


@dataclass
class Verdict:
    model: str
    base: Arm
    head: Arm
    status: str = "done"       # the job's done.json status
    reason: str = ""
    p: float | None = None
    outright: list[str] = field(default_factory=list)
    expected: int = 0          # scheduled runs per arm

    @property
    def unfinished(self) -> bool:
        """No Fisher verdict: the job did not finish, an arm is short of its scheduled runs, or base's server did
        not start (head's that did not start count as failed runs)."""
        return (self.status != "done" or min(self.base.n, self.head.n) < max(self.expected, 1)
                or any(r.get("no_server") for r in self.base.runs))

    @property
    def flagged(self) -> bool:
        return bool(self.outright) or (self.p is not None and self.p < ALPHA)

    def text(self) -> str:
        if self.outright:
            return f"**FAIL** ({' and '.join(self.outright)} only on head)"
        if self.p is not None and self.p < ALPHA:
            return f"**FAIL** (more failed runs, p = {self.p:.3f})"
        if self.unfinished:
            if self.status != "done":
                return f"incomplete ({self.reason or self.status})"
            if any(r.get("no_server") for r in self.base.runs):
                return "incomplete (base did not start)"
            return f"incomplete ({min(self.base.n, self.head.n)} of {self.expected} runs)"
        return "same"


def judge_model(model: str, base: Arm, head: Arm, status: str = "done", reason: str = "",
                expected: int = 0) -> Verdict:
    v = Verdict(model, base, head, status, reason, expected=expected)
    if base.n and head.n and not any(r.get("no_server") for r in base.runs):   # an unseen base: no outright
        v.outright = [c for c in OUTRIGHT if head.with_class(c) and not base.with_class(c)]
    if not v.unfinished:
        v.p = fisher_one_sided(head.failed, head.n, base.failed, base.n)
    return v


def load_arms(model_dir: Path) -> dict[str, Arm]:
    arms = {"base": Arm(), "head": Arm()}
    for a, arm in arms.items():
        for f in sorted(model_dir.glob(f"{a}/s*/stress/run*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem))):
            try:
                arm.runs.append(json.loads(f.read_text()))
            except ValueError:
                continue
    return arms


# ---------------------------------------------------------------- report

def _arm_cell(a: Arm) -> str:
    cls = ", ".join(f"{c} {a.with_class(c)}" for c in CLASSES if a.with_class(c))
    return f"{a.failed}/{a.n}" + (f" ({cls})" if cls else "")


def _rate(a: Arm) -> str:
    return f"{100 * a.failed_requests / a.requests:.1f}% of {a.requests}" if a.requests else "n/a"


def render_md(verdicts: list[Verdict], run: dict, chart_url: str | None = None) -> str:
    from benchmark.eval.e2e.report import DASHES

    base, head, cfg = run["base"], run["head"], run["stress"]
    gpu = run.get("gpu", "?").upper()
    flagged = [v for v in verdicts if v.flagged]
    lines = [f"## Stress: {head['label']} against {base['label']}", ""]
    if flagged:
        lines.append(f"**{len(flagged)} of {len(verdicts)} models flagged**: head shows a crash or cross-talk base "
                     f"never shows, or more failed runs at one-sided Fisher exact p < {ALPHA}.")
    elif not any(v.unfinished for v in verdicts):
        lines.append(f"Nothing flagged on {len(verdicts)} model(s).")
    else:
        lines.append("Nothing flagged in the complete rows; some rows are incomplete (see the table).")
    ns = sorted({min(v.base.n, v.head.n) for v in verdicts if v.base.n and v.head.n})
    lines += ["", f"{gap_text(ns[0]) if ns else 'No complete runs'}. Each run: {cfg['clients']} clients with one "
              f"pinned request_id each, {cfg['seconds']:g} s, patterns {', '.join(cfg.get('patterns') or PATTERNS)}; "
              f"a fresh server every {PER_SERVER} runs, arms alternated, one {gpu} per model.", "",
              "| Model | Failed runs, base | Failed runs, head | Fisher p (one-sided) | Failed requests, base "
              "| Failed requests, head | Verdict |", "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        p = f"{v.p:.3f}" if v.p is not None else "n/a"
        lines.append(f"| {v.model} | {_arm_cell(v.base)} | {_arm_cell(v.head)} | {p} | {_rate(v.base)} | "
                     f"{_rate(v.head)} | {v.text()} |")
    if chart_url:
        lines += ["", f"![Failed runs per failure class, base against head, on one {gpu} per model]({chart_url})"]
    lines += ["", "<details>", "<summary>Request outcomes per arm and pattern</summary>", "",
              "| Model | Arm | Pattern | Outcomes |", "|---|---|---|---|"]
    for v in verdicts:
        for name, a in (("base", v.base), ("head", v.head)):
            pats: dict[str, Counter] = {}
            for r in a.runs:
                for pat, res in r.get("by_pattern", {}).items():
                    pats.setdefault(pat, Counter()).update(res)
            for pat, c in pats.items():
                lines.append(f"| {v.model} | {name} | {pat} | "
                             f"{', '.join(f'{k} {n}' for k, n in sorted(c.items()))} |")
    lines += ["", "Outcomes: ok, closed (closed by the client on purpose), conflict (the 409 an overlap expects), "
              "accepted_twice (both overlapping requests served, reported only), unmatched (own reply not "
              "recognised, reported only); hang, error and crosstalk fail the run.", "", "</details>", "",
              "<details>", "<summary>Runs</summary>", "", "| Model | Arm | Run | Server session | Requests | Classes "
              "| Examples |", "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        for name, a in (("base", v.base), ("head", v.head)):
            for r in a.runs:
                ex = "; ".join(e for k in FAILED_REQUEST for e in r.get("examples", {}).get(k, [])[:1])
                ex = (ex or r.get("reason", "")).replace("|", "/")[:160]
                lines.append(f"| {v.model} | {name} | {r.get('run')} | {r.get('session')} | {r.get('requests', 0)} "
                             f"| {', '.join(r.get('classes', [])) or 'none'} | {ex} |")
    lines += ["", "</details>", "", "<details>", "<summary>Setup</summary>", "",
              f"- base: {base['label']} (`{base['sha'][:9]}`), head: {head['label']} (`{head['sha'][:9]}`)",
              f"- Command: `{run.get('command', 'python -m benchmark.eval compare --stress')}`",
              f"- A run fails on a hang (no progress for {cfg['hang_s']:g} s), an HTTP 5xx, unexpected 4xx, empty "
              "reply or in-band error, cross-talk (a reply holding another prompt's answer), a leak (the server's "
              f"own M\\* tensor files in /dev/shm above the post-warm-up count {cfg['settle_s']:g} s after the run; "
              "other servers' files on the host are not counted) or a crash (a "
              "server process exits); runs a crash or a server that did not start left unrun count as failed",
              "- Verdict: a crash or cross-talk on head that base never shows fails outright; else head fails when "
              f"its failed-run count beats base's at one-sided Fisher exact p < {ALPHA}",
              f"- GPU time: {run.get('gpu_hours', '?')} GPU-hours", "", "</details>"]
    text = "\n".join(lines) + "\n"
    for d in DASHES:
        text = text.replace(d, "-")
    return text


def chart(verdicts: list[Verdict], path: Path) -> Path | None:
    done = [v for v in verdicts if v.base.n or v.head.n]
    if not done:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [*CLASSES, "any"]
    fig, axes = plt.subplots(1, len(done), figsize=(5.5 * len(done), 3.2), squeeze=False)
    for ax, v in zip(axes[0], done, strict=True):
        xs = range(len(labels))
        for off, name, a, color in ((-0.2, "base", v.base, "#868e96"), (0.2, "head", v.head, "#4263eb")):
            ys = [a.with_class(c) for c in CLASSES] + [a.failed]
            ax.bar([x + off for x in xs], ys, width=0.4, color=color, label=f"{name} (N = {a.n})")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(v.base.n, v.head.n, 1))
        ax.set_ylabel("failed runs")
        p = f"p = {v.p:.2f}" if v.p is not None else "incomplete"
        ax.set_title(f"{v.model}: {p}, {'flagged' if v.flagged else 'not flagged'}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write(run_dir: Path, models: list[str], chart_url: str | None = None) -> tuple[Path, list[Verdict]]:
    """report/report.md, report/summary.json and (with matplotlib) report/chart.png for a ``--stress`` run."""
    run = json.loads((run_dir / "run.json").read_text())
    status = run.get("model_status", {})
    verdicts = []
    for m in models:
        md = run_dir / "results" / m
        try:
            done = json.loads((md / "done.json").read_text())
        except (OSError, ValueError):
            done = {"status": status.get(m, {}).get("status", "not run"), "reason": status.get(m, {}).get("reason", "")}
        arms = load_arms(md)
        verdicts.append(judge_model(m, arms["base"], arms["head"], done.get("status", "?"), done.get("reason", ""),
                                    expected=run["stress"]["runs"]))
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    png = None
    try:
        png = chart(verdicts, out / "chart.png")
    except ImportError:
        png = None
    md_path = out / "report.md"
    md_path.write_text(render_md(verdicts, run, chart_url or (png.name if png else None)))
    (out / "summary.json").write_text(json.dumps([
        {"model": v.model, "flagged": v.flagged, "unfinished": v.unfinished, "p": v.p, "outright": v.outright,
         "base": {"runs": v.base.n, "failed": v.base.failed, "requests": v.base.requests,
                  "failed_requests": v.base.failed_requests},
         "head": {"runs": v.head.n, "failed": v.head.failed, "requests": v.head.requests,
                  "failed_requests": v.head.failed_requests}} for v in verdicts], indent=1))
    return md_path, verdicts


def plan_text(cfg: dict) -> str:
    sessions = math.ceil(cfg["runs"] / PER_SERVER)
    return (f"stress: {cfg['runs']} runs per arm of {cfg['seconds']:g} s, {cfg['clients']} clients, "
            f"{sessions} server session(s) per arm (a fresh server every {PER_SERVER} runs), arms alternated")
