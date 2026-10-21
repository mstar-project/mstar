"""benchmark.eval compare --stress on CPU: the Fisher exact rule and the gap N can detect, verdicts, request outcomes,
and the workload generator and run classifier end to end through compare's job path against fake /generate servers
(a correct one, and ones that hang, mix replies, leak segments or crash after a client goes away)."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval.e2e import compare, hosts, stress, trees  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402
from benchmark.eval.e2e.stress import Arm, Refs, Reply  # noqa: E402

RUN = {"id": "r", "base": {"label": "main", "sha": "a" * 40}, "head": {"label": "PR #1", "sha": "b" * 40},
       "gpu": "h200", "models": ["fake"]}

# ---------------------------------------------------------------- statistics and verdicts


def test_fisher_one_sided_matches_known_values():
    assert stress.fisher_one_sided(3, 4, 1, 4) == pytest.approx(17 / 70)            # the tea-tasting table
    assert stress.fisher_one_sided(5, 14, 2, 14) == pytest.approx(0.1923, abs=1e-4)  # #461's stalls: p = 0.19
    assert stress.fisher_one_sided(5, 20, 0, 20) == pytest.approx(0.02356, abs=1e-5)
    assert stress.fisher_one_sided(4, 20, 0, 20) == pytest.approx(0.05301, abs=1e-5)
    assert stress.fisher_one_sided(0, 20, 3, 20) == 1.0
    assert stress.fisher_one_sided(2, 2, 0, 2) == pytest.approx(1 / 6)


def test_detectable_gap_and_its_report_line():
    assert stress.detectable_gap(20) == 5 and stress.detectable_gap(4) == 4 and stress.detectable_gap(3) is None
    assert stress.gap_text(20) == ("N = 20 runs per arm; the smallest gap it can detect at p < 0.05 is 0 vs 5 failed "
                                   "runs")
    assert "cannot reach p < 0.05 even at 0 vs 2" in stress.gap_text(2)


def arm(*classes: list[str], requests: int = 10) -> Arm:
    return Arm([{"run": i, "classes": c, "requests": requests, "results": {"ok": requests}}
                for i, c in enumerate(classes)])


def test_verdict_outright_classes_fisher_and_incomplete():
    v = stress.judge_model("m", arm([], []), arm(["crash"], []))
    assert v.flagged and v.outright == ["crash"] and v.p == pytest.approx(0.5)      # outright, whatever N
    v = stress.judge_model("m", arm(["crosstalk"], []), arm(["crosstalk"], ["crosstalk"]))
    assert not v.outright and not v.flagged                                          # base shows it too
    v = stress.judge_model("m", arm(*[[]] * 20), arm(*[["hang"]] * 5, *[[]] * 15))
    assert v.flagged and not v.outright and v.p < 0.05 and "p = 0.024" in v.text()
    v = stress.judge_model("m", arm(*[[]] * 20), arm(*[["hang"]] * 4, *[[]] * 16))
    assert not v.flagged and v.text() == "same"
    v = stress.judge_model("m", arm(), arm([]), status="error", reason="job died")
    assert v.unfinished and not v.flagged and v.p is None and v.text() == "incomplete (job died)"
    v = stress.judge_model("m", arm(*[[]] * 10), arm(*[[]] * 5), expected=10)       # head short of its runs
    assert v.unfinished and v.p is None and v.text() == "incomplete (5 of 10 runs)"
    lost = Arm([{"run": i, "classes": ["error"], "no_server": True} for i in range(5)])
    v = stress.judge_model("m", arm(*[[]] * 5), lost, expected=5)                     # head did not start: failed
    assert not v.unfinished and v.flagged and v.p < 0.01
    v = stress.judge_model("m", lost, arm(*[[]] * 5), expected=5)
    assert v.unfinished and v.p is None and v.text() == "incomplete (base did not start)"
    v = stress.judge_model("m", lost, arm(["crash"], *[[]] * 4), expected=5)
    assert not v.flagged and not v.outright                                          # no base to compare with


def test_request_outcomes():
    refs = Refs({0: "1 2 3 4 5 6 7 8 9 10 11 12 13 14", 1: "Here is the list: A B C D E F G", 2: "banana " * 8,
                 3: "Monday, Tuesday, and a banana"})
    ok = Reply(0, status=200, text="1 2 3 4 5 6 7 8 9 10")
    assert stress.judge(ok, refs) == "ok"
    assert stress.judge(Reply(0, status=200, text="Here is the list: 1 2 3 4"), refs) == "ok"   # prose is not a sign
    assert stress.judge(Reply(1, status=200, text="12 13 14"), refs) == "crosstalk"   # a predecessor's tail
    assert stress.judge(Reply(0, status=200, text="1 2 3 4 5 6 7 8 9 banana banana banana"), refs) == "crosstalk"
    assert stress.judge(Reply(3, status=200, text="Monday, Tuesday, and a banana"), refs) == "ok"  # its quiet reply
    assert stress.judge(Reply(4, status=200, text="Mercury, Venus, Monday, Friday"), refs) == "crosstalk"
    assert stress.judge(Reply(1, status=200, text="ABCDEFGHIJKLMNOPQRSTUVWXYZ"), refs) == "ok"   # as Qwen3.5 writes it
    assert stress.judge(Reply(0, status=200, text="1 2 3 FGHIJK"), refs) == "crosstalk"
    assert stress.judge(Reply(0, status=200, text="Here are the first numbers: 1 2 3"), refs) == "ok"
    assert stress.judge(Reply(0, status=200, text="1 2", closed=True), refs) == "closed"
    assert stress.judge(Reply(0, status=200, text=""), refs) == "error"                       # an empty 200
    assert stress.judge(Reply(0, status=200, text="something else entirely"), refs) == "unmatched"
    assert stress.judge(Reply(0, status=500, error="result delivery timed out"), refs) == "error"
    assert stress.judge(Reply(0, status=200, error="boom", error_status=500), refs) == "error"  # in-band
    assert stress.judge(Reply(0, status=409, error="in flight"), refs, expect_conflict=True) == "conflict"
    assert stress.judge(Reply(0, status=200, error="superseded", error_status=409), refs, True) == "conflict"
    assert stress.judge(Reply(0, status=409, error="superseded"), refs) == "error"           # 409 on a plain reuse
    superseded = Reply(1, status=200, text="13 14 15", error="superseded", error_status=409)
    assert stress.judge(superseded, refs, expect_conflict=True) == "crosstalk"              # foreign text, then 409
    assert stress.judge(Reply(0, status=200, hang=True), refs) == "hang"


def test_a_stream_of_blank_lines_or_no_headers_is_a_hang_not_progress():
    import asyncio
    import time

    from aiohttp import web

    async def blank(request):
        resp = web.StreamResponse()
        await resp.prepare(request)
        while True:
            await resp.write(b"\n")
            await asyncio.sleep(0.01)

    async def silent(request):
        await asyncio.sleep(30)

    async def go():
        app = web.Application()
        app.router.add_post("/generate", blank)
        app.router.add_post("/silent/generate", silent)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        cfg = {**stress.DEFAULTS, "hang_s": 0.3}
        try:
            async with await stress._session() as s:
                t0 = time.monotonic()
                r1 = await stress.send(s, url, "x", 0, cfg)
                r2 = await stress.send(s, url + "/silent", "y", 0, cfg)
                return r1, r2, time.monotonic() - t0
        finally:
            await runner.cleanup()

    r1, r2, took = asyncio.run(go())
    assert r1.hang and r2.hang and took < 3


# ---------------------------------------------------------------- end to end against fake servers

FAKE_GENERATE = r'''
import argparse, asyncio, base64, hashlib, json, os
from pathlib import Path
from aiohttp import web

MODE = {mode!r}
ANSWERS = {answers!r}
SHM = Path(os.environ["FAKE_SHM"])
p = argparse.ArgumentParser()
p.add_argument("cmd"); p.add_argument("model"); p.add_argument("--port", type=int)
p.add_argument("--socket-path-prefix")
a, _ = p.parse_known_args()
# tensor files are named by the deployment token of the conductor's socket, as mstar's are
anchor = Path(a.socket_path_prefix) / "conductor.ipc"
anchor.parent.mkdir(parents=True, exist_ok=True)
anchor.touch()
NS = hashlib.sha1(f"ipc://{{os.path.realpath(anchor)}}".encode()).hexdigest()[:12]
live, aborted, last = {{}}, set(), {{}}


def line(text, modality="text", **meta):
    return (json.dumps({{"modality": modality, "data": base64.b64encode(text.encode()).decode(), "metadata": meta}})
            + "\n").encode()


async def health(request):
    return web.Response(text="ok")


async def generate(request):
    form = await request.post()
    rid, text = form["request_id"], form["text"]
    if rid in live:            # a newer submission supersedes the live one, which ends with an in-band 409
        live[rid]["superseded"] = True
    me = live[rid] = {{"superseded": False}}
    retried = rid in aborted
    if retried and MODE == "crash":
        os._exit(3)
    reply = ANSWERS[last.get(rid, text) if retried and MODE == "mix" else text]   # mix: the predecessor's reply
    last[rid] = text
    seg = SHM / f"mstar_{{NS}}_worker_0_{{len(last)}}{{id(request)}}"
    seg.touch()
    if MODE == "neighbour":   # another server on the host: its token and pid, never given back here
        (SHM / f"mstar_{{'0' * 12}}_worker_0_{{len(last)}}{{id(request)}}").touch()
        (SHM / f"mstar_kv_{{os.getuid()}}_1_{{len(last)}}{{id(request)}}").touch()
    resp = web.StreamResponse()
    resp.content_type = "application/x-ndjson"
    finished = False
    try:
        await resp.prepare(request)
        while retried and MODE == "hang":
            if request.transport is None or request.transport.is_closing():
                raise ConnectionResetError
            await asyncio.sleep(0.02)
        for w in reply.split():
            await asyncio.sleep(0.003)
            if me["superseded"]:
                await resp.write(line("request superseded", "error", status=409))
                break
            await resp.write(line(w + " "))
        await resp.write_eof()
        finished = True
    finally:
        if live.get(rid) is me:
            del live[rid]
        (aborted.discard if finished else aborted.add)(rid)
        if finished or MODE != "leak":
            seg.unlink(missing_ok=True)
    return resp


app = web.Application()
app.router.add_get("/health", health)
app.router.add_post("/generate", generate)
web.run_app(app, host="127.0.0.1", port=a.port, print=None, handle_signals=True)
'''


def fake_tree(d: Path, mode: str) -> Path:
    """A checkout whose `mstar serve` is a /generate server holding pinned ids; ``mode`` sets how it misbehaves
    once a client has gone away under an id: good, hang, mix, leak, crash; neighbour behaves but leaves files
    of another deployment in the shared shm dir."""
    (d / "mstar" / "cli").mkdir(parents=True)
    (d / "mstar" / "__init__.py").write_text("")
    (d / "mstar" / "cli" / "__init__.py").write_text("")
    answers = dict(zip(stress.PROMPTS, (
        " ".join(str(i) for i in range(1, 21)), " ".join("ABCDEFGHIJKLMNOP"), "banana " * 12,
        "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
        "1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. Saturn", "January, February, April, June, July"),
        strict=True))
    (d / "mstar" / "cli" / "main.py").write_text(FAKE_GENERATE.format(mode=mode, answers=answers))
    (d / ".complete").touch()
    return d


def stress_args(tmp_path: Path, runs: int) -> Namespace:
    return Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval",
                     work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"), sessions=1, repeats=None,
                     no_rerun=False,
                     calibration=str(tmp_path / "cal.json"), stress=True, stress_runs=runs, stress_seconds=0.5,
                     stress_clients=4, stress_patterns=",".join(stress.PATTERNS), stress_hang_s=1.0, hosts=None,
                     host_provider=None, base="main", head="x", models="fake", rp=False)


def run_pair(tmp_path: Path, monkeypatch, head_mode: str, runs: int):
    monkeypatch.setattr(compare, "POLL_S", 0.2)
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setitem(stress.DEFAULTS, "settle_s", 0.5)
    monkeypatch.setitem(stress.DEFAULTS, "shm_dir", str(shm))
    man = mf.parse({"defaults": {"startup_timeout_s": 60}, "models": {"fake": {
        "serve": "fake", "family": "LLM", "env": {"FAKE_SHM": str(shm)}, "stress": {"max_tokens": 16},
        "cases": [{"id": "text", "client": "chat_text", "check": {"kind": "text_match", "min_match": 0.5},
                   "perf": {"latency": "lat_p50_s"}}]}}})
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "data").mkdir()
    args = stress_args(tmp_path, runs)
    (run_dir / "run.json").write_text(json.dumps({**RUN, "stress": compare.stress_config(args),
                                                  "command": compare.public_command(args)}))
    runner = compare.Runner(args, man, man.select("all"), base, head, run_dir, [hosts.Slot(hosts.LocalTransport(),
                                                                                           None)])
    runner.cache = tmp_path / "cache"
    fake_tree(runner.cache / "src" / base.sha, "good")
    fake_tree(runner.cache / "src" / head.sha, head_mode)
    assert runner.run_one(runner.slots[0], "fake") is None                       # no ABBA rerun for stress
    job = json.loads((run_dir / "jobs" / "fake.json").read_text())
    assert job["stress"]["max_tokens"] == 16 and job["sessions"] == 1
    md, code = compare.render(run_dir, man, args.calibration)
    return stress.write(run_dir, ["fake"])[1][0], md.read_text(), code


def test_another_servers_files_in_the_shm_dir_are_not_a_leak(tmp_path, monkeypatch):
    """Files under another deployment token or another pid (another server on the node) stay in the shared dir;
    head's own tensor files all go, so nothing is flagged."""
    v, md, code = run_pair(tmp_path, monkeypatch, "neighbour", runs=2)
    assert code == 0 and not v.flagged and v.head.with_class("leak") == 0, v.head.runs
    shm = tmp_path / "shm"
    assert any(n.name.startswith("mstar_000000000000_") for n in shm.iterdir())
    assert any(n.name.startswith("mstar_kv_") for n in shm.iterdir())


def test_identical_servers_are_not_flagged_and_every_pattern_ran(tmp_path, monkeypatch):
    v, md, code = run_pair(tmp_path, monkeypatch, "good", runs=2)
    assert code == 0 and not v.flagged and v.base.n == v.head.n == 2
    assert v.base.failed == v.head.failed == 0, (v.base.runs, v.head.runs)
    res = v.head.results()
    assert res["conflict"] and res["closed"] and res["ok"] and not res["accepted_twice"]
    patterns = {p for r in v.head.runs for p in r["by_pattern"]}
    assert patterns == set(stress.PATTERNS)
    assert "Nothing flagged on 1 model" in md and "cannot reach p < 0.05 even at 0 vs 2" in md
    assert "--stress --stress-runs 2 --stress-seconds 0.5 --stress-clients 4 --stress-hang-s 1" in md


@pytest.mark.parametrize("mode, runs, outright, cls", [
    ("mix", 2, ["crosstalk"], "crosstalk"),
    ("crash", 2, ["crash"], "crash"),
    ("hang", 4, [], "hang"),
    ("leak", 4, [], "leak"),
])
def test_a_head_that_misbehaves_after_a_client_goes_away_is_flagged(tmp_path, monkeypatch, mode, runs, outright, cls):
    v, md, code = run_pair(tmp_path, monkeypatch, mode, runs)
    assert code == 1 and v.flagged and v.outright == outright, (v.head.runs, v.base.runs)
    assert v.base.failed == 0 and v.head.with_class(cls) >= (1 if outright else runs)
    assert "**1 of 1 models flagged**" in md


# ---------------------------------------------------------------- the CLI

def test_cli_validates_selects_and_dry_runs(tmp_path, monkeypatch, capsys):
    from benchmark.eval.cli import build_parser

    parse = build_parser().parse_args
    with pytest.raises(SystemExit, match="unknown nope"):
        compare.stress_config(parse(["compare", "--head", "main", "--stress", "--stress-patterns", "retry,nope"]))
    with pytest.raises(SystemExit, match="no stress workload for kokoro"):
        compare.cmd_compare(parse(["compare", "--head", "main", "--stress", "--models", "kokoro"]))
    assert [n for n, m in mf.load().models.items() if m.stress] == ["qwen3_5_2b"]
    with pytest.raises(mf.ManifestError, match="stress takes"):
        mf.parse({"models": {"m": {"serve": "m", "stress": {"clients": 3}, "cases": [
            {"id": "t", "client": "chat_text", "check": {"kind": "text_match", "min_match": 1},
             "perf": {"latency": "lat_p50_s"}}]}}})
    monkeypatch.setattr(trees, "resolve", lambda repo, spec, remote=None: trees.Ref(spec, "a" * 40, spec))
    monkeypatch.setattr(compare, "make_slots", lambda args, provider: [hosts.Slot(hosts.LocalTransport(), 0)])
    monkeypatch.setattr(compare, "probe", lambda args, slot: "NVIDIA H200")
    args = parse(["compare", "--head", "main", "--stress", "--stress-runs", "8", "--dry-run"])
    assert compare.cmd_compare(args) == 0
    err = capsys.readouterr().err
    assert "1 models" in err and "qwen3_5_2b" in err
    assert ("stress: 8 runs per arm of 60 s, 32 clients, 2 server session(s) per arm (a fresh server every 5 runs), "
            "arms alternated") in err
