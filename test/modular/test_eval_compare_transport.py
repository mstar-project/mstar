"""benchmark.eval compare on CPU, worker-to-worker transport (#575): the split and disaggregated configs over shared
memory, Mooncake TCP and Mooncake RDMA, the cases that cancel mid-transfer (a dropped stream, a server --timeout),
and the worker health checks after each case (server processes alive, mstar /dev/shm entries given back, no crash
or transfer-failure lines in the log), judged head against base."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from test_eval_compare import RUN, chat_model, fake_tree, finish, write_arm  # noqa: E402

from benchmark.eval.e2e import arm, checks, compare, hosts, report, request_path, stress, trees  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402

CONFIGS = ("qwen3tts_1p7b_split", "orpheus_pd_disaggregated", "qwen3omni_pd_disaggregated")


def test_each_multi_worker_config_runs_over_shm_tcp_and_rdma_with_the_multi_gpu_cases():
    m = mf.load()
    assert {x.name for x in m.select("tag:w2w")} == {f"{c}_{t}" for c in CONFIGS for t in ("shm", "tcp")}
    assert {x.name for x in m.select("tag:rdma")} == {f"{c}_rdma" for c in CONFIGS}
    assert {f"{c}_{t}" for c in CONFIGS for t in ("shm", "tcp")} <= {x.name for x in m.select("tag:transport")}
    assert not {x.name for x in m.select("all")} & {f"{c}_{t}" for c in CONFIGS for t in ("shm", "tcp", "rdma")}
    # tag:mooncake stays the two one-worker Mooncake entries: the w2w TCP entries need 2 and 3 GPUs
    assert {x.name for x in m.select("tag:mooncake")} == {"qwen3_omni_mooncake", "orpheus_mooncake"}
    for c in CONFIGS:
        base, shm, tcp, rdma = (m.models[n] for n in (c, f"{c}_shm", f"{c}_tcp", f"{c}_rdma"))
        for x in (shm, tcp, rdma):
            assert (x.config, x.gpus, x.serve, x.family) == (base.config, base.gpus, base.serve, base.family)
            assert x.health and x.optional and x.case_timeout_s
            assert [k.to_dict() for k in x.cases] == [k.to_dict() for k in shm.cases]   # the same cases on each
        assert shm.serve_args == [] and "shm" in shm.tags
        assert tcp.serve_args == ["--tensor-comm-protocol", "TCP"] and "mooncake.engine" in tcp.requires
        assert rdma.serve_args == ["--tensor-comm-protocol", "RDMA"] and "rdma_nic" in rdma.requires
        assert [k.to_dict() for k in shm.cases[:len(base.cases)]] == [k.to_dict() for k in base.cases]
        extra = {k.id: k for k in shm.cases[len(base.cases):]}
        assert set(extra) == {"cancel", "timeout"} and not any(k.optional for k in extra.values())
        assert extra["cancel"].kind.startswith("cancel_") and extra["cancel"].server_timeout_s is None
        assert extra["timeout"].kind.startswith("timeout_") and extra["timeout"].server_timeout_s == 5
    assert all(not x.health for x in m.models.values() if not x.name.endswith(("_shm", "_tcp", "_rdma")))


def test_server_timeout_cases_are_validated():
    one = {"models": {"m": {"serve": "m", "config": "c.yaml", "cases": [
        {"id": "t", "rp": "timeout_speech", "server_timeout_s": 5, "check": {"kind": "status_match"}}]}}}
    assert mf.parse(copy.deepcopy(one)).models["m"].cases[0].server_timeout_s == 5.0
    bad = copy.deepcopy(one)
    bad["models"]["m"]["cases"][0]["server_timeout_s"] = 0
    with pytest.raises(mf.ManifestError, match="server_timeout_s must be a positive"):
        mf.parse(bad)
    bad = copy.deepcopy(one)
    del bad["models"]["m"]["config"]
    with pytest.raises(mf.ManifestError, match="needs 'config'"):
        mf.parse(bad)


# ---------------------------------------------------------------- cancel mid-transfer

class Speech(BaseHTTPRequestHandler):
    """Streams 16 chunks over 2.4 s; with ``dies_after_cut`` it stops serving once a client has dropped a stream.
    Non-streamed requests take 0.002 s per character, and one past ``timeout_s`` is a 500 (the server's --timeout)."""

    protocol_version = "HTTP/1.1"
    dies_after_cut = False
    timeout_s = 600.0
    cut = threading.Event()

    def log_message(self, *args):
        pass

    def _reply(self, code: int, data: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.dies_after_cut and self.cut.is_set():
            return self._reply(500)
        if not body.get("stream"):
            took = len(body["input"]) * 0.002
            if took > self.timeout_s:
                time.sleep(self.timeout_s)
                type(self).cut.set()
                return self._reply(500, b"Request timed out")
            time.sleep(took)
            return self._reply(200, b"RIFF" + body["input"].encode() + body.get("voice", "").encode())
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for i in range(16):
                chunk = bytes([i]) * 512
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                time.sleep(0.15)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            type(self).cut.set()


def serve(handler) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def judge(tmp_path: Path, results: dict) -> tuple[str, str]:
    for arm_name, res in results.items():
        d = tmp_path / arm_name
        d.mkdir(parents=True)
        (d / "rp.json").write_text(json.dumps(res))
    return report._judge_session({"kind": "status_match"}, checks.rp_facts(tmp_path / "base", tmp_path / "head"))


def test_cancel_cases_cut_streams_and_a_server_that_breaks_after_a_cut_is_a_status_change(tmp_path, monkeypatch):
    monkeypatch.setattr(request_path, "SETTLE_S", 0.2)
    good_h = type("Good", (Speech,), {"cut": threading.Event()})
    bad_h = type("Bad", (Speech,), {"cut": threading.Event(), "dies_after_cut": True})
    results = {}
    for name, h in (("base", good_h), ("head", bad_h)):
        srv, url = serve(h)
        try:
            results[name] = request_path.cancel_speech(url, tmp_path, tmp_path)
        finally:
            srv.shutdown()
    base, head = results["base"], results["head"]
    assert base["disconnect_after_3_chunks"] == {"status": 200, "cut": True}
    assert base["disconnect_after_1_5_s"] == {"status": 200, "cut": True}
    assert base["serves_after_cancels"] == {"status": 200, "identical": True}
    assert head["serves_after_cancels"]["status"] == 500
    verdict, detail = judge(tmp_path, results)
    assert verdict == "FAIL" and "serves_after_cancels 200 to 500" in detail


def test_timeout_case_times_out_the_long_request_and_a_server_that_dies_after_it_is_a_status_change(
        tmp_path, monkeypatch):
    monkeypatch.setattr(request_path, "TIMEOUT_SETTLE_S", 0.2)
    monkeypatch.setattr(request_path, "VOICE", "tara")
    good_h = type("Good", (Speech,), {"cut": threading.Event(), "timeout_s": 1.0})
    results = {}
    srv, url = serve(good_h)
    try:
        results["base"] = request_path.timeout_speech(url, tmp_path, tmp_path)
    finally:
        srv.shutdown()
    srv, url = serve(type("Bad", (Speech,), {"cut": threading.Event(), "timeout_s": 1.0}))
    real = request_path._speech_req

    def dies_after_the_timeout(u, text, **kw):   # the worker crashed: the next request finds no server
        fn = real(u, text, **kw)
        return (lambda: (srv.shutdown(), srv.server_close(), fn())[-1]) if len(text) > 200 else fn

    monkeypatch.setattr(request_path, "_speech_req", dies_after_the_timeout)
    try:
        results["head"] = request_path.timeout_speech(url, tmp_path, tmp_path)
    finally:
        srv.server_close()
    assert results["base"] == {"reference": {"status": 200}, "times_out": {"status": 500, "ok": True},
                               "serves_after_timeout": {"status": 200, "identical": True}}
    assert results["head"]["serves_after_timeout"] == {"status": "ConnectionError", "identical": False}
    verdict, detail = judge(tmp_path, results)
    assert verdict == "FAIL" and "serves_after_timeout 200 to ConnectionError" in detail


@pytest.mark.parametrize("handler", [
    {"timeout_s": 600.0},                      # the long request beat the timeout: nothing was cancelled
    {"timeout_s": 1.0, "dies_after_cut": True, "cut": None},   # a 500 that is not the server's timeout
])
def test_a_timeout_case_whose_request_was_not_timed_out_on_base_is_incomplete_not_pass(tmp_path, monkeypatch,
                                                                                       handler):
    monkeypatch.setattr(request_path, "TIMEOUT_SETTLE_S", 0.1)
    ev = threading.Event()
    ev.set()                                   # dies_after_cut with the event set: every request is a bare 500
    attrs = {**handler, "cut": ev if "cut" in handler else threading.Event()}
    results = {}
    for name in ("base", "head"):
        srv, url = serve(type(name, (Speech,), dict(attrs)))
        try:
            results[name] = request_path.timeout_speech(url, tmp_path, tmp_path)
        finally:
            srv.shutdown()
    assert results["base"]["times_out"]["ok"] is False and "_unexercised" in results["base"]
    verdict, detail = judge(tmp_path, results)
    assert verdict == "incomplete" and detail.startswith("not exercised on base: the long request was not timed out")


def test_a_head_that_finishes_before_the_timeout_is_incomplete_and_one_that_errors_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(request_path, "TIMEOUT_SETTLE_S", 0.1)
    runs = {}
    for name, attrs in (("base", {"timeout_s": 1.0}), ("fast", {"timeout_s": 600.0}),
                        ("error", {"timeout_s": 1.0, "dies_after_cut": True})):
        ev = threading.Event()
        if name == "error":
            ev.set()
        srv, url = serve(type(name, (Speech,), {**attrs, "cut": ev}))
        try:
            runs[name] = request_path.timeout_speech(url, tmp_path, tmp_path)
        finally:
            srv.shutdown()
    assert runs["fast"]["times_out"] == {"status": 200, "ok": False} and runs["fast"]["_completed"]
    verdict, detail = judge(tmp_path / "fast", {"base": runs["base"], "head": runs["fast"]})
    assert verdict == "incomplete" and detail.startswith("not exercised on head: the long request")
    verdict, detail = judge(tmp_path / "error", {"base": runs["base"], "head": runs["error"]})
    assert verdict == "FAIL" and "times_out 500 to 500" in detail          # the server's timeout, then a bare 500


def test_streams_that_end_before_the_cut_leave_the_cancel_case_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(request_path, "SETTLE_S", 0.05)

    class Short(Speech):   # one chunk and done: neither cut is reached
        cut = threading.Event()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = b"RIFF" + body["input"].encode()
            self._reply(200, data)

    results = {}
    for name in ("base", "head"):
        srv, url = serve(Short)
        try:
            results[name] = request_path.cancel_speech(url, tmp_path, tmp_path)
        finally:
            srv.shutdown()
    assert results["base"]["disconnect_after_3_chunks"] == {"status": 200, "cut": False}
    verdict, detail = judge(tmp_path, results)
    assert verdict == "incomplete" and detail == "not exercised on base: both streams ended before their cut"


# ---------------------------------------------------------------- worker health

def test_health_findings_name_exited_processes_leftover_shm_and_crash_lines(tmp_path, monkeypatch):
    # an APIServer built earlier in this process leaves its launch token in os.environ
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(arm, "SHM", shm)
    monkeypatch.setattr(arm, "SHM_SETTLE_S", 1.5)
    monkeypatch.setattr(arm, "MIN_AGE_S", 0)
    # a "server" session: a leader and one worker child
    p = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; "
                          "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                          "print(c.pid, flush=True); time.sleep(60)"], stdout=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        child = int(p.stdout.readline())
        sock = tmp_path / "scratch" / "sock"
        sock.mkdir(parents=True)
        (sock / "conductor.ipc").touch()
        ns = hashlib.sha1(f"ipc://{os.path.realpath(sock / 'conductor.ipc')}".encode()).hexdigest()[:12]
        server = Namespace(pid=p.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        log = tmp_path / "server.log"
        log.write_text("INFO started\nERROR Segmentation fault (old, before the case)\n")
        (shm / f"mstar_arena_worker_0_{child}_aa").touch()                   # before the case: not counted
        before = arm.health_before(server, log)
        assert set(before["pids"]) >= {p.pid, child} and before["ns"] == [ns]
        assert arm.health_after(before, server, log) == []                       # nothing changed
        (shm / f"mstar_{ns}_worker_0_7").touch()                             # its tensor file left behind
        (shm / "mstar_0123456789ab_worker_0_9").touch()                      # another deployment's tensor file
        (shm / f"mstar_arena_worker_1_{os.getpid()}_bb").touch()             # a live process of another server
        (shm / f"mstar_{ns}_arena_worker_0_ab12cd34.seg1").touch()          # a segment its arena grew into
        (shm / "nccl-123").touch()                                           # not mstar's
        (shm / f"mstar_arena_worker_0_{child}_cc").touch()                   # orphaned by the worker below
        gone = shm / f"mstar_{ns}_worker_0_8"                                 # given back during the wait
        gone.touch()
        threading.Timer(0.3, gone.unlink).start()
        os.kill(child, signal.SIGKILL)
        time.sleep(0.3)   # a zombie until its parent reaps it: it no longer counts
        with open(log, "a") as f:
            f.write("ERROR worker_1: Mooncake read failed for request r7\nFatal Python error: Segmentation fault\n")
        found = arm.health_after(before, server, log)
        # the arena segment is not a tensor leak, but with a process gone and nothing mapping it, it is reported
        assert found == ["1 server process(es) exited", "/dev/shm +2 entries",
                         "/dev/shm 1 arena segment(s) no live server process maps", "log: segfault",
                         "log: mooncake read failed"]
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


@pytest.mark.parametrize("token", [None, "0123456789abcdef"], ids=["no-token", "launch-token"])
def test_the_deployment_token_matches_the_one_mstar_names_tensor_files_with(tmp_path, monkeypatch, token):
    """The harness rebuilds ``tensors._deployment_namespace`` by hand; this catches either side drifting."""
    tensors = pytest.importorskip("mstar.communication.tensors")
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    from mstar.communication.communicator import BaseCommunicator, CommProtocol

    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "conductor.ipc").touch()
    comm = Namespace(protocol=CommProtocol.IPC, ipc_socket_path_prefix=f"{sock}/")
    comm._endpoint = lambda entity: BaseCommunicator._endpoint(comm, entity)
    if token is None:
        server = Namespace(command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        assert arm._namespaces(server) == {tensors._deployment_namespace(comm)}
        return
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True,
                              env={**os.environ, "MSTAR_DEPLOYMENT_TOKEN": token})
    try:
        server = Namespace(pid=worker.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        monkeypatch.setenv("MSTAR_DEPLOYMENT_TOKEN", token)
        assert tensors._deployment_namespace(comm) in arm._namespaces(server)
    finally:
        worker.kill()
        worker.wait()


@pytest.mark.parametrize("stale", ["oldtokenoldtoken", ""])
def test_a_stale_token_on_the_api_server_does_not_hide_the_one_its_workers_carry(tmp_path, monkeypatch, stale):
    """The API server mints after exec, so ``/proc`` of its pid still shows the token it was started with (a
    previous launch's, or empty). Reading that one, which is usually the lowest pid, misses the token the
    workers were spawned with and the leak check never sees their files."""
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN_MINTED", raising=False)
    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "conductor.ipc").touch()
    endpoint = f"ipc://{os.path.realpath(sock / 'conductor.ipc')}"
    fresh = "newtokennewtoken"
    script = (
        "import os, subprocess, sys, time\n"
        "env = os.environ.copy()\n"
        f"env['MSTAR_DEPLOYMENT_TOKEN'] = {fresh!r}\n"
        "env['MSTAR_DEPLOYMENT_TOKEN_MINTED'] = env['MSTAR_DEPLOYMENT_TOKEN']\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], env=env)\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    parent_env = {**os.environ, "MSTAR_DEPLOYMENT_TOKEN": stale}
    if stale:
        parent_env["MSTAR_DEPLOYMENT_TOKEN_MINTED"] = stale
    parent = subprocess.Popen(
        [sys.executable, "-c", script], start_new_session=True, env=parent_env,
        stdout=subprocess.PIPE, text=True,
    )
    try:
        child = int(parent.stdout.readline())
        assert child in stress.group_pids(parent.pid)
        server = Namespace(pid=parent.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        got = arm._namespaces(server)
        fresh_ns = hashlib.sha1(f"{fresh}\0{endpoint}".encode()).hexdigest()[:12]
        endpoint_only = hashlib.sha1(endpoint.encode()).hexdigest()[:12]
        assert fresh_ns in got and endpoint_only in got
        if stale:
            stale_ns = hashlib.sha1(f"{stale}\0{endpoint}".encode()).hexdigest()[:12]
            assert stale_ns not in got
    finally:
        os.killpg(parent.pid, signal.SIGKILL)
        parent.wait()


def test_the_namespace_follows_the_launch_token_the_server_spawned_its_workers_with(tmp_path, monkeypatch):
    """With a launch token in the SHM namespace (#552) the API server mints MSTAR_DEPLOYMENT_TOKEN after it starts,
    so only the processes it spawns carry it, and the namespace is the SHA-1 of the token and the endpoint. A build
    that does not name files with it may still inherit one, so the endpoint-only namespace stays too."""
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "conductor.ipc").touch()
    endpoint = f"ipc://{os.path.realpath(sock / 'conductor.ipc')}"
    token = "0123456789abcdef"
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True,
                              env={**os.environ, "MSTAR_DEPLOYMENT_TOKEN": token})
    try:
        server = Namespace(pid=worker.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        assert arm._namespaces(server) == {hashlib.sha1(k.encode()).hexdigest()[:12]
                                           for k in (endpoint, f"{token}\0{endpoint}")}
    finally:
        worker.kill()
        worker.wait()


def test_rdma_nic_requirement_skips_a_host_without_a_device(tmp_path, monkeypatch):
    monkeypatch.setattr(arm, "job", {"model": {"requires": ["rdma_nic"]}})
    monkeypatch.setattr(arm, "RDMA_SYSFS", tmp_path / "none")
    assert arm.requirements_problem({}) == "rdma_nic: no RDMA device"
    (tmp_path / "ib" / "mlx5_0").mkdir(parents=True)
    monkeypatch.setattr(arm, "RDMA_SYSFS", tmp_path / "ib")
    assert arm.requirements_problem({}) == ""
    (tmp_path / "loader.py").write_text("def bad():\n    raise ImportError('x')\n")
    monkeypatch.setattr(arm, "job", {"python": sys.executable, "model": {"requires": ["loader:bad", "rdma_nic"]}})
    monkeypatch.setattr(arm, "RDMA_SYSFS", tmp_path / "none")
    assert arm.requirements_problem({"src": str(tmp_path)}) == "loader:bad: ImportError; rdma_nic: no RDMA device"


def write_health(md: Path, arm_name: str, findings: list[str], alive: bool = True) -> None:
    d = md / arm_name / "s0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "startup.json").write_text(json.dumps({"healthy": True, "cases": {"cancel": {
        "rc": 0, "server_alive_after": alive, "health": findings}}}))
    (d / "cancel").mkdir(exist_ok=True)
    (d / "cancel" / "rp.json").write_text(json.dumps({"serves_after_cancels": {"status": 200, "identical": True}}))


def rp_model() -> mf.Model:
    return mf.parse({"models": {"m": {"serve": "m", "health": True, "cases": [
        {"id": "cancel", "rp": "cancel_speech", "check": {"kind": "status_match"}}]}}}).models["m"]


@pytest.mark.parametrize("base,head,base_alive,want", [
    ([], ["1 server process(es) exited"], True, "FAIL"),
    ([], ["log: mooncake read failed"], True, "FAIL"),
    ([], ["--timeout 5 server did not start"], True, "FAIL"),
    (["/dev/shm +2 entries"], ["/dev/shm +5 entries"], True, "pass"),       # the same kind on both arms
    (["log: transfer timeout"], [], True, "pass"),                           # base-only findings never flag head
    (["log: segfault"], [], True, "base broken"),                            # base crashed: nothing to judge
    (["2 server process(es) exited"], ["log: segfault"], True, "base broken"),
    ([], [], False, "base broken"),                                          # base stopped answering
])
def test_a_finding_only_head_has_fails_the_row_and_a_base_crash_reports_base_broken(tmp_path, base, head,
                                                                                    base_alive, want):
    md = tmp_path / "results" / "m"
    write_health(md, "base", base, base_alive)
    write_health(md, "head", head)
    checks.compare_model(md, [c.to_dict() for c in rp_model().cases], 1)
    (md / "done.json").write_text(json.dumps({"status": "done"}))
    (tmp_path / "run.json").write_text(json.dumps({**RUN, "models": ["m"]}))
    row = report.build_rows(tmp_path, [rp_model()], calibration={})[0]
    assert row.correct == want, row.correct_detail
    if want == "FAIL":
        assert row.flagged and row.correct_detail.startswith(f"head worker health: {head[0]}")
    if want == "base broken":
        assert not row.flagged and row.unfinished and row.correct_detail.startswith("base worker crashed")


def test_a_base_crash_outranks_a_head_hang(tmp_path):
    md = tmp_path / "results" / "m"
    write_health(md, "base", ["log: segfault"])
    write_health(md, "head", [])
    startup = md / "head" / "s0" / "startup.json"
    js = json.loads(startup.read_text())
    js["cases"]["cancel"].update(rc="timeout", hung=True, cap_s=900, server_alive_after=False)
    startup.write_text(json.dumps(js))
    checks.compare_model(md, [c.to_dict() for c in rp_model().cases], 1)
    (md / "done.json").write_text(json.dumps({"status": "done"}))
    (tmp_path / "run.json").write_text(json.dumps({**RUN, "models": ["m"]}))
    row = report.build_rows(tmp_path, [rp_model()], calibration={})[0]
    assert row.correct == "base broken" and not row.flagged and row.unfinished, row.correct_detail


def test_a_base_crash_on_a_timed_row_drops_its_perf_verdicts_too(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)       # head 20% slower at every point
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].flagged
    startup = md / "base" / "s0" / "startup.json"
    startup.write_text(json.dumps({"healthy": True, "startup_s": 10, "cases": {"c": {
        "rc": 0, "server_alive_after": True, "health": ["log: segfault"]}}}))
    rows = report.build_rows(rd, [chat_model()], calibration={})
    r = rows[0]
    assert r.correct == "base broken" and not r.flagged and r.unfinished
    assert r.latency.verdict == r.throughput.verdict == "n/a"
    assert report.flagged_perf_cases(rows) == []


# ---------------------------------------------------------------- end to end through `arm`

FAKE_ENTRYPOINT = r'''
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("--port", type=int); p.add_argument("--timeout", type=float, default=600.0)
p.add_argument("--tensor-comm-protocol")
a, _ = p.parse_known_args()
print("protocol", a.tensor_comm_protocol, "timeout", a.timeout, flush=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, code, data=b""):
        self.send_response(code); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, b"ok")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        took = len(req["input"]) * 0.005
        if took > a.timeout:
            time.sleep(a.timeout)
            if {fault}:
                print("Worker worker_1 (pid 4242) exited with signal SIGSEGV", file=sys.stderr, flush=True)
            return self.reply(500, b"Request timed out")
        time.sleep(took)
        self.reply(200, b"RIFF" + req["input"].encode())


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''


def test_seeded_transport_faults_on_the_head_arm_are_flagged_end_to_end(tmp_path):
    """Both arms through ``arm`` with health checks. Head's main server logs a failed Mooncake read on every request
    while answering normally, and head's ``--timeout`` server logs a worker SIGSEGV when it times a request out (the
    seeded faults): only the health check can catch them."""
    raw = {"defaults": {"conc": [1], "repeats": 1, "warmup": 0, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "TTS", "health": True, "config": "fake.yaml",
                 "serve_args": ["--tensor-comm-protocol", "TCP"], "cases": [
                     {"id": "text", "client": "chat_text", "n": 2, "check": {"kind": "text_match", "min_match": 0.5},
                      "perf": {"latency": "lat_p50_s"}},
                     {"id": "timeout", "rp": "timeout_speech", "server_timeout_s": 1,
                      "check": {"kind": "status_match"}}]}}}
    man = mf.parse(copy.deepcopy(raw))
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    args = Namespace(python=sys.executable, work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"),
                     sessions=1, repeats=None)
    s = hosts.Slot(hosts.LocalTransport(), None, host_dir=str(tmp_path / "host"))
    runner = compare.Runner(args, man, man.select("all"), base, head, tmp_path / "run", [s])
    runner.cache = tmp_path / "cache"
    for sha, fault in ((base.sha, False), (head.sha, True)):
        tree = fake_tree(runner.cache / "src" / sha)
        (tree / "mstar" / "api_server").mkdir()
        (tree / "mstar" / "api_server" / "__init__.py").write_text("")
        (tree / "mstar" / "api_server" / "entrypoint.py").write_text(FAKE_ENTRYPOINT.format(fault=fault))
        if fault:
            main_py = tree / "mstar" / "cli" / "main.py"
            main_py.write_text(main_py.read_text().replace(
                "        self.close_connection = True",
                "        self.close_connection = True\n        print('ERROR worker_1: Mooncake read failed', "
                "file=sys.stderr, flush=True)"))
    out = tmp_path / "run" / "results" / "fake"
    out.mkdir(parents=True)
    (out / "job.json").write_text(json.dumps({**runner.job(s, "fake", str(out)), "order": ["base", "head"]}))
    (out / "lease").touch()
    r = subprocess.run([sys.executable, "-m", "benchmark.eval.e2e.arm", str(out / "job.json")], capture_output=True,
                       text=True, timeout=180, check=False, env={**os.environ, "PYTHONPATH": str(compare.TOOL_ROOT)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for a in ("base", "head"):   # the --timeout server: the model's transport, the case's timeout
        assert (out / a / "s0" / "server-timeout1.log").read_text().startswith("protocol TCP timeout 1.0")
    cases = {a: json.loads((out / a / "s0" / "startup.json").read_text())["cases"] for a in ("base", "head")}
    assert cases["base"]["text"]["health"] == [] and cases["base"]["timeout"]["health"] == []
    assert "log: mooncake read failed" in cases["head"]["text"]["health"]
    assert cases["head"]["timeout"]["health"] == ["log: segfault", "log: worker died"]
    rp = json.loads((out / "head" / "s0" / "timeout" / "rp.json").read_text())
    assert rp["times_out"] == {"status": 500, "ok": True} and rp["serves_after_timeout"]["identical"]
    (tmp_path / "run" / "run.json").write_text(json.dumps({**RUN, "models": ["fake"]}))
    rows = {x.case: x for x in report.build_rows(tmp_path / "run", man.select("all"), calibration={})}
    assert rows["text"].flagged and rows["text"].correct_detail.startswith("head worker health: log: mooncake")
    assert rows["timeout"].flagged and rows["timeout"].correct_detail.startswith(
        "head worker health: log: segfault, log: worker died")


def test_a_worker_that_exits_leaves_the_arena_segments_no_live_process_maps(tmp_path, monkeypatch):
    """Arena segments are named by namespace, not pid, so a live arena's segments look like a dead one's. Once a
    server process exits, the ones no live process maps are reported; one a live peer still maps is not."""
    import mmap

    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(arm, "SHM", shm)
    monkeypatch.setattr(arm, "SHM_SETTLE_S", 0.5)
    monkeypatch.setattr(arm, "MIN_AGE_S", 0)
    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "conductor.ipc").touch()
    ns = hashlib.sha1(f"ipc://{os.path.realpath(sock / 'conductor.ipc')}".encode()).hexdigest()[:12]
    segs = []
    for w in (0, 1):
        seg = shm / f"mstar_{ns}_arena_worker_{w}_ab12cd34.seg0"
        seg.write_bytes(b"\0" * mmap.PAGESIZE)
        segs.append(seg)
    # a "server" session whose two workers each map their arena's segment
    mapper = ("import mmap, sys, time; f = open(sys.argv[1], 'r+b'); m = mmap.mmap(f.fileno(), 4096); "
              "print('mapped', flush=True); time.sleep(60)")
    script = ("import subprocess, sys, time\n"
              f"ws = [subprocess.Popen([sys.executable, '-c', {mapper!r}, s], stdout=subprocess.PIPE, text=True)"
              f" for s in {[str(x) for x in segs]!r}]\n"
              "assert all(w.stdout.readline().strip() == 'mapped' for w in ws)\n"
              "print(*[w.pid for w in ws], flush=True); time.sleep(60)\n")
    leader = subprocess.Popen([sys.executable, "-c", script], start_new_session=True, stdout=subprocess.PIPE,
                              text=True)
    try:
        first, _ = map(int, leader.stdout.readline().split())
        server = Namespace(pid=leader.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        before = arm.health_before(server, tmp_path / "server.log")
        assert before["ns"] == [ns]
        assert arm.health_after(before, server, tmp_path / "server.log") == []   # live arenas: no finding
        os.kill(first, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while stress.alive(first) and time.monotonic() < deadline:   # a zombie maps nothing
            time.sleep(0.05)
        assert stress.unmapped_arena_segments(str(shm), {ns}, stress.group_pids(leader.pid)) == [segs[0].name]
        assert arm.health_after(before, server, tmp_path / "server.log") == [
            "1 server process(es) exited", "/dev/shm 1 arena segment(s) no live server process maps"]
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait()


def test_an_unreadable_environ_leaves_the_endpoint_namespace(tmp_path, monkeypatch):
    """A worker whose ``/proc/<pid>/environ`` cannot be read (gone meanwhile, or another user's under hidepid)
    gives no token, so the server's own one or none is used, and its files are never misattributed."""
    monkeypatch.delenv("MSTAR_DEPLOYMENT_TOKEN", raising=False)
    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "conductor.ipc").touch()
    endpoint = f"ipc://{os.path.realpath(sock / 'conductor.ipc')}"
    script = ("import os, subprocess, sys, time\n"
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
              " env={**os.environ, 'MSTAR_DEPLOYMENT_TOKEN': 'workertokenworkr'})\n"
              "print(c.pid, flush=True); time.sleep(60)\n")
    leader = subprocess.Popen([sys.executable, "-c", script], start_new_session=True, stdout=subprocess.PIPE,
                              text=True, env={**os.environ, "MSTAR_DEPLOYMENT_TOKEN": "leadertokenleadr"})
    try:
        child = int(leader.stdout.readline())
        read = Path.read_bytes
        refused = []

        def _read_bytes(self):
            if str(self) == f"/proc/{child}/environ":
                refused.append(child)
                raise PermissionError(13, "Permission denied", str(self))
            return read(self)

        monkeypatch.setattr(Path, "read_bytes", _read_bytes)
        assert stress._proc_env_token(child) is None and refused == [child]
        assert stress._proc_env_token(2 ** 22 + 1) is None           # no such pid
        assert stress.launch_tokens(leader.pid) == {"leadertokenleadr"}
        server = Namespace(pid=leader.pid, command=lambda: ["mstar", "--socket-path-prefix", f"{sock}/"])
        assert arm._namespaces(server) == {hashlib.sha1(k.encode()).hexdigest()[:12]
                                           for k in (endpoint, f"leadertokenleadr\0{endpoint}")}
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait()


def test_an_idle_compile_worker_pool_exiting_is_not_a_server_process_exiting(tmp_path):
    """A cold start spawns Torch Inductor's compile workers (one per core) in the server's group; they exit once
    compiles go quiet, which the stress run and the health check counted as a crash."""
    pool = tmp_path / "site" / "torch" / "_inductor" / "compile_worker"
    pool.mkdir(parents=True)
    (pool / "__main__.py").write_text("import time; time.sleep(60)\n")
    server = tmp_path / "server.py"   # a file, so the server's own command line does not name the pool
    server.write_text("import subprocess, sys, time\n"
                      f"w = subprocess.Popen([sys.executable, {str(pool / '__main__.py')!r}])\n"
                      "print(w.pid, flush=True); time.sleep(60)\n")
    leader = subprocess.Popen([sys.executable, str(server)], start_new_session=True, stdout=subprocess.PIPE,
                              text=True)
    try:
        helper = int(leader.stdout.readline())
        deadline = time.monotonic() + 10
        while b"compile_worker" not in Path(f"/proc/{helper}/cmdline").read_bytes():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert stress.helper_process(helper) and not stress.helper_process(leader.pid)
        assert stress.group_pids(leader.pid) == {leader.pid}
        assert arm._group_pids(leader.pid) == {leader.pid}
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait()
