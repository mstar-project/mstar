"""benchmark.eval compare on CPU, multi-GPU configs (#574): the ``gpus`` and ``case_timeout_s`` manifest fields and
the shipped TP / SP / split / disaggregated entries, GPU slots that lend siblings to an N-GPU model on one host,
the job's CUDA_VISIBLE_DEVICES, and a case that hangs: cut at its wall cap with stacks saved, head failing and base
reported broken."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from test_eval_compare import RUN, chat_model, fake_tree, finish, procs_under, write_arm  # noqa: E402

from benchmark.eval.e2e import arm, compare, hosts, report, trees  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402

SIX = {"whisper_large_tp2": 2, "qwen3omni_thinker_tp2": 3, "cosmos3_nano_sp2": 2, "qwen3tts_1p7b_split": 1,
       "orpheus_pd_disaggregated": 2, "qwen3omni_pd_disaggregated": 3}

ONE = {"defaults": {"conc": [1], "repeats": 1, "warmup": 0, "startup_timeout_s": 60}, "models": {
    "fake": {"serve": "fake", "family": "LLM", "cases": [
        {"id": "text", "client": "chat_text", "n": 2, "check": {"kind": "text_match", "min_match": 0.5},
         "perf": {"latency": "lat_p50_s"}}]}}}


# ---------------------------------------------------------------- manifest

def test_gpus_and_case_timeout_fields_are_validated():
    raw = copy.deepcopy(ONE)
    m = mf.parse(raw).models["fake"]
    assert m.gpus == 1 and m.case_timeout_s is None
    raw["models"]["fake"] |= {"gpus": 3, "case_timeout_s": 600}
    m = mf.parse(raw).models["fake"]
    assert m.gpus == 3 and m.case_timeout_s == 600.0
    assert mf.Model.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    for bad, msg in (({"gpus": 0}, "gpus must be a positive int"), ({"gpus": "2"}, "gpus must be a positive int"),
                     ({"gpus": True}, "gpus must be a positive int"),
                     ({"case_timeout_s": -1}, "case_timeout_s must be a positive")):
        raw = copy.deepcopy(ONE)
        raw["models"]["fake"] |= bad
        with pytest.raises(mf.ManifestError, match=msg):
            mf.parse(raw)


def test_shipped_multi_gpu_configs_match_their_rank_layouts():
    m = mf.load()
    picked = m.select("tag:multi-gpu")
    assert {x.name: x.gpus for x in picked} == SIX
    assert all(x.optional and x.case_timeout_s for x in picked)
    assert not {x.name for x in m.select("all")} & set(SIX)          # out of `all`
    six_configs = {m.models[n].config for n in SIX}                   # the transport variants reuse them
    assert all(x.gpus == 1 for x in m.models.values() if x.name not in SIX and x.config not in six_configs)
    for x in picked:                                   # the config's ranks land on exactly devices 0..gpus-1
        cfg = yaml.safe_load(Path(x.config).read_text())
        devices = {(cfg.get("rank_devices") or {}).get(r, r) for g in cfg["node_groups"] for r in g["ranks"]}
        assert devices == set(range(x.gpus)), x.name
        assert cfg["model"] == x.serve, x.name
    for name in ("whisper_large_tp2", "qwen3omni_thinker_tp2"):      # the #332 load on the TP configs
        assert all(max(c.conc) == 16 for c in m.models[name].cases)


# ---------------------------------------------------------------- slots

def test_parse_hosts_takes_gpu_ranges():
    assert hosts.parse_hosts("u@a:0-2,u@b") == [("u@a", 0), ("u@a", 1), ("u@a", 2), ("u@b", 0)]
    with pytest.raises(ValueError, match="same GPU twice"):
        hosts.parse_hosts("u@a:0-1,u@a:1")


def slots_of(host: str, n: int) -> list[hosts.Slot]:
    t = Namespace(name=host)
    return [hosts.Slot(t, g) for g in range(n)]


def test_an_n_gpu_model_runs_on_n_free_gpus_of_one_host_and_small_ones_fill_the_rest():
    a, b = slots_of("a", 4), slots_of("b", 1)
    need = {"big3": 3, "two": 2, "s1": 1, "s2": 1, "s3": 1, "s4": 1}
    q = hosts.WorkQueue(list(need), {"big3": 30, "two": 20, "s1": 3, "s2": 2, "s3": 1, "s4": 1})
    lock, busy, ran, overlap = threading.Lock(), {"a": set(), "b": set()}, {}, []

    def run_one(s, name):
        gpus = list(s.devices)
        with lock:
            if busy[s.transport.name] & set(gpus):
                overlap.append((name, gpus))
            busy[s.transport.name] |= set(gpus)
            ran[name] = (s.transport.name, gpus)
        time.sleep(0.2)
        with lock:
            busy[s.transport.name] -= set(gpus)

    hosts.run_slots(a + b, q, run_one, need=need.__getitem__)
    assert not overlap and set(ran) == set(need)
    host, gpus = ran["big3"]
    assert host == "a" and len(set(gpus)) == 3                       # 3 distinct GPUs of the 4-GPU host
    assert ran["two"][0] == "a" and len(set(ran["two"][1])) == 2       # never on the 1-GPU host
    assert all(len(ran[f"s{i}"][1]) == 1 for i in range(1, 5))
    assert all(not s.devices for s in a + b)                          # every GPU given back


def test_a_multi_gpu_model_at_the_front_is_not_starved_and_one_that_fits_nowhere_is_given_up():
    two = slots_of("h", 2)
    q = hosts.WorkQueue(["big", "s1", "s2", "s3"], {"big": 10, "s1": 1, "s2": 1, "s3": 1})
    order, lock = [], threading.Lock()

    def run_one(s, name):
        with lock:
            order.append((name, list(s.devices)))
        time.sleep(0.1)

    hosts.run_slots(two, q, run_one, need=lambda n: 2 if n == "big" else 1)
    assert order[0][0] == "big" and sorted(order[0][1]) == [0, 1]       # it waited for both GPUs, nothing jumped it
    q2, gave_up = hosts.WorkQueue(["huge", "s1"], {"huge": 9, "s1": 1}), []
    hosts.run_slots(slots_of("h", 2), q2, lambda s, n: None, on_given_up=gave_up.append,
                    need=lambda n: 5 if n == "huge" else 1)
    assert gave_up == ["huge"]


def test_the_job_carries_its_gpus_into_cuda_visible_devices(tmp_path):
    man = mf.parse(copy.deepcopy(ONE))
    base = trees.Ref("main", "a" * 40, "main")
    args = Namespace(python=sys.executable, work_dir=str(tmp_path / "work"), data_dir=str(tmp_path),
                     sessions=1, repeats=None)
    s = hosts.Slot(hosts.LocalTransport(), 2, host_dir=str(tmp_path))
    runner = compare.Runner(args, man, man.select("all"), base, base, tmp_path / "run", [s])
    assert runner.job(s, "fake", "out")["gpus"] == [2]
    s.devices = [2, 5, 6]
    job = runner.job(s, "fake", "out")
    assert job["gpus"] == [2, 5, 6] and job["gpu"] == 2
    arm.job.clear()
    arm.job.update(job)
    try:
        assert arm.base_env()["CUDA_VISIBLE_DEVICES"] == "2,5,6"
    finally:
        arm.job.clear()


def test_dry_run_lists_which_multi_gpu_models_fit(tmp_path, monkeypatch, capsys):
    from benchmark.eval.cli import build_parser

    monkeypatch.setattr(trees, "resolve", lambda repo, spec, remote=None: trees.Ref(spec, "a" * 40, spec))
    monkeypatch.setattr(compare, "probe", lambda args, slot: "NVIDIA H200")
    args = build_parser().parse_args(["compare", "--head", "main", "--models", "tag:multi-gpu", "--dry-run",
                                      "--hosts", "u@a:0-1"])
    assert compare.cmd_compare(args) == 0
    err = capsys.readouterr().err
    assert "6 models on 2 GPU slot(s)" in err
    assert "whisper_large_tp2: 2 GPUs on one host, fits" in err
    assert "qwen3omni_thinker_tp2: 3 GPUs on one host, does not fit (largest host: 2 GPUs)" in err


# ---------------------------------------------------------------- a hung case

# Well above the fast arm's case on a loaded host (3 s passed 2 of 5 runs there), well below the slow server's delay
CAP_S = 15


def run_job(tmp_path: Path, slow: str) -> Path:
    """Both arms of one fake model through ``arm`` with a CAP_S case cap; the ``slow`` arm's server never answers
    within it. Returns the results directory."""
    raw = copy.deepcopy(ONE)
    raw["models"]["fake"]["case_timeout_s"] = CAP_S
    man = mf.parse(raw)
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    args = Namespace(python=sys.executable, work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"),
                     sessions=1, repeats=None)
    s = hosts.Slot(hosts.LocalTransport(), None, host_dir=str(tmp_path / "host"))
    runner = compare.Runner(args, man, man.select("all"), base, head, tmp_path / "run", [s])
    runner.cache = tmp_path / "cache"
    for ref, name in ((base, "base"), (head, "head")):
        fake_tree(runner.cache / "src" / ref.sha, delay=60 if name == slow else 0.0)
    out = tmp_path / "run" / "results" / "fake"
    out.mkdir(parents=True)
    job = {**runner.job(s, "fake", str(out)), "order": ["base", "head"]}
    (out / "job.json").write_text(json.dumps(job))
    (out / "lease").touch()
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "benchmark.eval.e2e.arm", str(out / "job.json")], capture_output=True,
                       text=True, timeout=180, check=False, env={**os.environ, "PYTHONPATH": str(compare.TOOL_ROOT)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert time.monotonic() - t0 < 90                       # cut at the cap, not at the client's 30 min timeout
    assert not procs_under(runner.cache / "src")            # the hung server was stopped
    (tmp_path / "run" / "run.json").write_text(json.dumps({**RUN, "models": ["fake"]}))
    return out


def test_a_head_that_hangs_is_cut_at_the_cap_with_stacks_and_fails(tmp_path):
    out = run_job(tmp_path, slow="head")
    info = json.loads((out / "head" / "s0" / "startup.json").read_text())["cases"]["text"]
    assert info["hung"] and info["cap_s"] == CAP_S and info["rc"] == "timeout" and info["server_alive_after"] is False
    stacks = (out / "head" / "s0" / "text" / "stacks.txt").read_text()
    assert stacks.startswith("# process group") and "## pid" in stacks
    assert "hung" not in json.loads((out / "base" / "s0" / "startup.json").read_text())["cases"]["text"]
    man = mf.parse({**copy.deepcopy(ONE)})
    row = report.build_rows(tmp_path / "run", man.select("all"), calibration={})[0]
    assert row.correct == "FAIL" and row.flagged and row.correct_detail.startswith("head hung: still running after")


def test_a_base_that_hangs_is_reported_broken_not_flagged(tmp_path):
    out = run_job(tmp_path, slow="base")
    assert (out / "base" / "s0" / "text" / "stacks.txt").exists()
    man = mf.parse(copy.deepcopy(ONE))
    rows = report.build_rows(tmp_path / "run", man.select("all"), calibration={})
    assert rows[0].correct == "base broken" and not rows[0].flagged and rows[0].unfinished
    md, code = compare.render(tmp_path / "run", man, str(tmp_path / "none.json"))
    assert code == 2 and "base hung" in md.read_text()


@pytest.mark.parametrize("stacks_sudo", [False, True])
def test_dump_stacks_retries_with_sudo_only_when_asked(tmp_path, monkeypatch, stacks_sudo):
    """py-spy refused ptrace: without ``--stacks-sudo`` no ``sudo`` runs at all (a refused one is logged and often
    mailed on shared clusters); with it, the dump is retried once per process through ``sudo -n``."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "sudo-calls"
    bin_dir.mkdir()
    spy = bin_dir / "py-spy"
    spy.write_text("#!/bin/sh\necho 'Permission denied (os error 13)' >&2\nexit 1\n")
    (bin_dir / "sudo").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\necho "Thread 1 (idle)"\n')
    for f in (spy, bin_dir / "sudo"):
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(arm, "job", {"stacks_sudo": stacks_sudo})
    arm.dump_stacks(os.getpgid(0), tmp_path / "stacks.txt")
    stacks = (tmp_path / "stacks.txt").read_text()
    assert "## pid" in stacks
    if stacks_sudo:
        assert calls.read_text().split()[:2] == ["-n", str(spy)] and "Thread 1 (idle)" in stacks
    else:
        assert not calls.exists() and "Permission denied" in stacks and "--stacks-sudo" in stacks


def test_stacks_sudo_flag_reaches_the_job(tmp_path):
    man = mf.parse(copy.deepcopy(ONE))
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    s = hosts.Slot(hosts.LocalTransport(), None, host_dir=str(tmp_path / "host"))
    for argv, want in (([], False), (["--stacks-sudo"], True)):
        ap = ArgumentParser()
        compare.add_args(ap.add_subparsers())
        args = ap.parse_args(["compare", "--base", "main", "--head", "x", *argv])
        args.python, args.data_dir = sys.executable, str(tmp_path / "data")
        runner = compare.Runner(args, man, man.select("all"), base, head, tmp_path / "run", [s])
        assert runner.job(s, "fake", str(tmp_path / "out"))["stacks_sudo"] is want


def test_a_base_that_hangs_after_some_points_gates_head_timings_too(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)       # head 20% slower at every point
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].flagged
    startup = md / "base" / "s0" / "startup.json"
    startup.write_text(json.dumps({"healthy": True, "startup_s": 10,
                                   "cases": {"c": {"rc": "timeout", "hung": True, "cap_s": 900}}}))
    rows = report.build_rows(rd, [chat_model()], calibration={})
    r = rows[0]
    assert r.correct == "base broken" and not r.flagged and r.unfinished
    assert r.latency.verdict == r.throughput.verdict == "n/a"
    assert report.flagged_perf_cases(rows) == []                              # no ABBA rerun for it either
