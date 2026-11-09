"""Check live-pr-test's scripts where they spend money or judge a box: gpu.py's create, extend and
offers, bootstrap.sh's arguments, and gpu_health.py's choice of GPUs."""

import argparse
import contextlib
import http.client
import io
import os
import runpy
import subprocess
import tempfile
import time
import unittest
import unittest.mock
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".claude/skills/live-pr-test/scripts"
GPU = runpy.run_path(str(SCRIPTS / "gpu.py"))
HEALTH = runpy.run_path(str(SCRIPTS / "gpu_health.py"))
CFG = {"budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_total_usd": 100}, "ssh_public_key": "/dev/null",
       "runpod": {"image": "img", "hf_home": "/workspace/hf"}}
OFFER = {"id": "runpod:H100:US-1:1", "provider": "runpod", "gpu": "H100", "count": 1, "vram_gb": 80,
         "region": "US-1", "usd_per_hour": 3.0, "availability": "High", "storage": True}


def http_error(code, body=b"{}"):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body))


class Reply:
    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.raw


class TestGpu(unittest.TestCase):
    def setUp(self):
        self.g = GPU["http"].__globals__
        saved = {k: self.g[k] for k in ("STATE", "LEDGER", "runpod", "runpod_list", "runpod_volumes", "find_offer",
                                        "runpod_offers", "lambda_offers")}
        self.addCleanup(self.g.update, saved)
        state = Path(tempfile.mkdtemp())
        self.g.update(STATE=state, LEDGER=state / "instances.json")
        self.g["time"] = unittest.mock.Mock(wraps=time, sleep=lambda s: None)
        self.addCleanup(self.g.update, {"time": time})
        env = unittest.mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LAMBDA_API_KEY", None)  # never reach a real account

    def test_a_create_is_not_sent_again_after_a_5xx_or_a_timeout(self):
        for failure in (http_error(502), TimeoutError("timed out"), urllib.error.URLError("refused"),
                        ConnectionResetError("reset"), http.client.RemoteDisconnected("closed"),
                        Reply(b"<html>proxy error</html>")):
            calls = []

            def urlopen(req, timeout, failure=failure, calls=calls):
                calls.append(req)
                if isinstance(failure, Reply):
                    return failure
                raise failure
            with unittest.mock.patch("urllib.request.urlopen", urlopen):
                with self.assertRaises(GPU["ApiError"]):
                    GPU["http"]("POST", "https://x/pods", "RUNPOD_API_KEY", {}, retry=False)
            self.assertEqual(len(calls), 1, f"a create that may have succeeded was sent again after {failure!r}")
        # a 429 was refused before anything was made, so it is retried, and raised as a 429 in the end
        for replies, want in (([http_error(429), Reply(b'{"id": "p1"}')], {"id": "p1"}), ([http_error(429)] * 6, 429)):
            replies = iter(replies)

            def urlopen(req, timeout, replies=replies):
                reply = next(replies)
                if isinstance(reply, Reply):
                    return reply
                raise reply
            with unittest.mock.patch("urllib.request.urlopen", urlopen):
                if want == 429:
                    with self.assertRaises(GPU["ApiError"]) as refused:
                        GPU["http"]("POST", "https://x/pods", "RUNPOD_API_KEY", {}, retry=False)
                    self.assertEqual(refused.exception.status, 429)
                else:
                    self.assertEqual(GPU["http"]("POST", "https://x/pods", "RUNPOD_API_KEY", {}, retry=False), want)

    def launch(self, answer, shows_up=True):
        """cmd_launch against a RunPod that makes the pod and then answers the create with ``answer``;
        the pod shows in its listing at once, or (``shows_up`` False) only after the launch gave up.
        Returns the pods, the ledger and whether the launch exited."""
        made, listed, deleted = [], [], []

        def runpod(method, path, body=None, params=None, retry=True):
            if method == "POST":
                made.append({"id": "p1", "name": body["name"]})
                (listed if shows_up else []).extend(made)
                raise answer
            if method == "DELETE":
                deleted.append(path)
            return {}
        self.g.update(runpod=runpod, runpod_list=lambda path, field, params=None: listed,
                      find_offer=lambda cfg, oid: OFFER,
                      runpod_volumes=lambda cfg: {"US-1": {"id": "v", "kind": "network"}})
        args = argparse.Namespace(offer=OFFER["id"], pr=7, hours=1, allow_over_budget=False, dry_run=False)
        exited = False
        with unittest.mock.patch("subprocess.Popen") as popen, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                GPU["cmd_launch"](args, CFG)
            except SystemExit:
                exited = True
        self.watchdogs = [c.args[0][-1] for c in popen.call_args_list]
        return made, listed, deleted, exited

    def test_a_launch_whose_answer_was_lost_adopts_the_pod_it_made(self):
        # a 502, and a 503 whose page says "not available" (a proxy's, not RunPod's capacity answer)
        # and a 200 with no id in it
        for answer in (GPU["ApiError"](502, "bad gateway"), GPU["ApiError"](503, "Service not available"),
                       KeyError("id")):
            GPU["save_ledger"]({})
            self.launch(answer)
            self.assertEqual(list(GPU["ledger"]()), ["runpod:p1"],
                             f"the pod the lost create made never reached the ledger ({answer})")

    def test_a_lost_create_whose_pod_shows_up_late_is_still_ended_by_name(self):
        made, _, _, exited = self.launch(GPU["ApiError"](502, "bad gateway"), shows_up=False)
        (key,) = GPU["ledger"]()
        self.assertTrue(exited and key.startswith("pending:"), "the lost create's cost stopped counting")
        self.assertEqual(self.watchdogs, [key], "nothing ends the lost create's pod if nobody acts")
        self.assertTrue(self.launch(GPU["ApiError"](502, "bad gateway"))[3], "a second launch got past the cap")
        entry = GPU["ledger"]()[key]
        for end in ("by hand", "at the deadline"):
            GPU["save_ledger"]({key: {**entry, "deadline": time.time() - 1}})
            deleted, polls = [], iter([[]])  # the first pod appears after its launch gave up, and a poll late
            self.g.update(runpod_list=lambda path, field, params=None, polls=polls: next(polls, made),
                          runpod=lambda method, path, *a, deleted=deleted, **k: deleted.append((method, path)) or {})
            with contextlib.redirect_stdout(io.StringIO()):
                if end == "by hand":
                    GPU["cmd_terminate"](argparse.Namespace(instance=key), CFG)
                else:
                    GPU["cmd_watchdog"](argparse.Namespace(instance=key), CFG)
            self.assertEqual(deleted, [("DELETE", "/pods/p1")], f"ending the pending entry {end} left its pod up")
            self.assertEqual(GPU["ledger"](), {})

    def test_an_extend_or_terminate_while_adopt_looks_is_kept(self):
        """The pending entry is changed from another shell while launch is still looking for its pod."""
        for change, want in (("extend", 0), ("terminate", 60)):
            GPU["save_ledger"]({})
            state = {"polls": 0}

            def runpod(method, path, body=None, params=None, retry=True, state=state):
                if method == "POST":
                    state["name"] = body["name"]
                    raise GPU["ApiError"](502, "bad gateway")
                return {}

            def runpod_list(path, field, params=None, state=state, change=change):
                state["polls"] += 1
                if state["polls"] == 1:  # another shell acts before the pod is listed
                    pending = f"pending:{state['name']}"
                    if change == "extend":
                        with contextlib.redirect_stdout(io.StringIO()):
                            GPU["cmd_extend"](argparse.Namespace(instance=pending, hours=0, allow_over_budget=False),
                                              CFG)
                    else:
                        with GPU["ledger_txn"]() as d:
                            d.pop(pending)
                    return []
                return [{"id": "p1", "name": state["name"]}]
            self.g.update(runpod=runpod, runpod_list=runpod_list, find_offer=lambda cfg, oid: OFFER,
                          runpod_volumes=lambda cfg: {"US-1": {"id": "v", "kind": "network"}})
            args = argparse.Namespace(offer=OFFER["id"], pr=7, hours=1, allow_over_budget=False, dry_run=False)
            with unittest.mock.patch("subprocess.Popen") as popen, contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                GPU["cmd_launch"](args, CFG)
            entry = GPU["ledger"]().get("runpod:p1")
            self.assertIsNotNone(entry, f"the pod found after a {change} meanwhile went unrecorded")
            self.assertIn("runpod:p1", [c.args[0][-1] for c in popen.call_args_list])
            self.assertAlmostEqual((entry["deadline"] - time.time()) / 60, want, delta=1,
                                   msg=f"the {change} made while launch looked was lost")

    def test_a_create_refused_with_a_429_makes_nothing_to_look_for(self):
        made, listed, deleted, exited = self.launch(GPU["ApiError"](429, "slow down"))
        self.assertTrue(exited)
        self.assertEqual(GPU["ledger"](), {}, "a rate-limited create left a pending entry blocking the next launch")

    def test_extend_keeps_to_the_launch_caps(self):
        now = time.time()
        one = {"usd_per_hour": 3.0, "launched_at": now, "deadline": now + 3600}
        dear = {**one, "usd_per_hour": 30.0, "deadline": now + 3 * 3600}  # $90 to its deadline
        for ledger, hours in (({"runpod:p1": one}, 50),  # past budget.max_hours
                              ({"runpod:p1": one, "runpod:p2": dear}, 3.9)):  # past max_total_usd
            GPU["save_ledger"](ledger)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                GPU["cmd_extend"](argparse.Namespace(instance="runpod:p1", hours=hours, allow_over_budget=False), CFG)
            self.assertEqual(GPU["ledger"]()["runpod:p1"]["deadline"], now + 3600)
        # What SKILL.md tells a session without a key to do, even for a run approved past the caps
        GPU["save_ledger"]({"runpod:p1": {**one, "launched_at": now - 5 * 3600, "deadline": now + 3600}})
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["cmd_extend"](argparse.Namespace(instance="runpod:p1", hours=0, allow_over_budget=False), CFG)
        self.assertLess(GPU["ledger"]()["runpod:p1"]["deadline"], time.time() + 1)
        # and for one already past its deadline, whose watchdog hasn't got to it yet
        GPU["save_ledger"]({"runpod:p1": {**one, "launched_at": now - 5 * 3600, "deadline": now - 30}})
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["cmd_extend"](argparse.Namespace(instance="runpod:p1", hours=0, allow_over_budget=False), CFG)
        self.assertLess(GPU["ledger"]()["runpod:p1"]["deadline"], time.time() + 1)

    def test_offers_with_a_provider_not_answering_is_not_no_availability(self):
        def erroring(*a, **k):
            raise GPU["ApiError"](503, "unavailable")
        args = argparse.Namespace(gpus=1, min_vram=80, all_regions=False, allow_arm=False, json=False)
        self.g["runpod_offers"] = erroring
        self.g["lambda_offers"] = lambda *a, **k: []
        # no key anywhere; RunPod erroring with no Lambda key; RunPod erroring and Lambda answering with nothing
        for env in ({}, {"RUNPOD_API_KEY": "k"}, {"RUNPOD_API_KEY": "k", "LAMBDA_API_KEY": "k"}):
            out = io.StringIO()
            with unittest.mock.patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit) as exited, \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                GPU["cmd_offers"](args, {**CFG, "lambda": {}})
            self.assertNotEqual(exited.exception.code, 0)
            self.assertNotIn("no availability", out.getvalue())


class TestBootstrap(unittest.TestCase):
    def test_help_or_a_bad_pr_number_changes_nothing(self):
        for argv, code in ((["--help"], 0), (["-h"], 0), (["abc"], 2), ([], 2)):
            home = tempfile.mkdtemp()
            r = subprocess.run(["bash", str(SCRIPTS / "bootstrap.sh"), *argv], capture_output=True, text=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": home, "HF_HOME": f"{home}/hf"}, check=False)
            self.assertEqual(r.returncode, code, r.stderr)
            self.assertEqual(os.listdir(home), [], f"bootstrap.sh {argv} wrote to HOME")

    def test_untrusted_caches_stay_in_the_work_directory_and_are_never_saved(self):
        home = Path(tempfile.mkdtemp())
        fake = home / "bin"
        fake.mkdir()
        for tool in ("git", "uv", "cargo", "curl", "nvidia-smi"):  # nothing is installed or fetched
            (fake / tool).write_text("#!/bin/sh\nexit 1\n")
            (fake / tool).chmod(0o755)
        script = (f"bash {SCRIPTS / 'bootstrap.sh'} 1 --untrusted --workdir {home}/w --hf-home {home}/vol/hf; "
                  f"source {home}/w/env.sh && echo $TORCHINDUCTOR_CACHE_DIR && save_caches")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False,
                           env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(home)})
        self.assertIn(f"{home}/w/compile/inductor", r.stdout)
        self.assertIn("save_caches:", r.stderr)
        rerun = subprocess.run(["bash", "-c", script.replace(" --untrusted", "")], capture_output=True, text=True,
                               check=False, env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(home)})
        self.assertIn(f"{home}/w/compile/inductor", rerun.stdout, "a rerun without --untrusted lost the flag")
        self.assertIn("save_caches:", rerun.stderr)
        self.assertFalse((home / "vol/live-pr-compile-cache").exists(), "an untrusted run wrote the shared tarball")


class TestHealth(unittest.TestCase):
    SMI = "GPU-aaaa, 35, 345, 0x0\nGPU-bbbb, 60, 1980, 0x0\nGPU-cccc, 61, 1975, 0x0\nGPU-dddd, 34, 345, 0x0\n"

    def test_only_the_visible_gpus_are_judged_by_torch_index(self):
        self.assertEqual(HEALTH["by_torch_index"](self.SMI, ["cccc", "bbbb"]), [(0, 61, 1975, 0), (1, 60, 1980, 0)])

    def test_a_gpu_it_cannot_match_exits_before_anything_is_loaded(self):
        g, burned = HEALTH["main"].__globals__, []
        saved = {k: g[k] for k in ("visible_uuids", "burn")}
        self.addCleanup(g.update, saved)
        g.update(visible_uuids=lambda: ["mig-1234"], burn=lambda i, stop: burned.append(i))
        smi = subprocess.CompletedProcess([], 0, stdout=self.SMI)
        with unittest.mock.patch("subprocess.run", return_value=smi), \
                unittest.mock.patch("sys.argv", ["gpu_health.py"]):
            with self.assertRaises(SystemExit):
                HEALTH["main"]()
        self.assertEqual(burned, [], "the burn started before the GPU mapping was checked, so a mismatch hangs")


if __name__ == "__main__":
    unittest.main()
