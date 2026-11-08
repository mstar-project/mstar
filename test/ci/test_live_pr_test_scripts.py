"""Check live-pr-test's scripts where they spend money or judge a box: gpu.py's create, extend and
offers, bootstrap.sh's arguments, and gpu_health.py's choice of GPUs."""

import argparse
import contextlib
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
CFG = {"budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_total_usd": 100}, "ssh_public_key": "/dev/null",
       "runpod": {"image": "img", "hf_home": "/workspace/hf"}}
OFFER = {"id": "runpod:H100:US-1:1", "provider": "runpod", "gpu": "H100", "count": 1, "vram_gb": 80,
         "region": "US-1", "usd_per_hour": 3.0, "availability": "High", "storage": True}


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"{}"))


class TestGpu(unittest.TestCase):
    def setUp(self):
        self.g = GPU["http"].__globals__
        saved = {k: self.g[k] for k in ("STATE", "LEDGER", "runpod", "runpod_list", "runpod_volumes", "find_offer")}
        self.addCleanup(self.g.update, saved)
        state = Path(tempfile.mkdtemp())
        self.g.update(STATE=state, LEDGER=state / "instances.json")
        self.g["time"] = unittest.mock.Mock(wraps=time, sleep=lambda s: None)
        self.addCleanup(self.g.update, {"time": time})
        os.environ["RUNPOD_API_KEY"] = "k"
        self.addCleanup(os.environ.pop, "RUNPOD_API_KEY")

    def test_a_create_is_not_sent_again_after_a_5xx_or_a_timeout(self):
        for failure in (http_error(502), TimeoutError("timed out"), urllib.error.URLError("reset")):
            calls = []

            def urlopen(req, timeout, failure=failure, calls=calls):
                calls.append(req)
                raise failure
            with unittest.mock.patch("urllib.request.urlopen", urlopen):
                with self.assertRaises(GPU["ApiError"]):
                    GPU["http"]("POST", "https://x/pods", "RUNPOD_API_KEY", {}, retry=False)
            self.assertEqual(len(calls), 1, f"a create that may have succeeded was sent again after {failure!r}")
        # a 429 was refused before anything was made, so it is retried
        replies = iter([http_error(429), None])

        class Ok:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"id": "p1"}'

        def urlopen(req, timeout):
            reply = next(replies)
            if reply is not None:
                raise reply
            return Ok()
        with unittest.mock.patch("urllib.request.urlopen", urlopen):
            self.assertEqual(GPU["http"]("POST", "https://x/pods", "RUNPOD_API_KEY", {}, retry=False), {"id": "p1"})

    def test_a_launch_whose_answer_was_lost_adopts_the_pod_it_made(self):
        made = []

        def runpod(method, path, body=None, params=None, retry=True):
            if method == "POST":
                made.append({"id": "p1", "name": body["name"]})  # RunPod made it, then the answer was lost
                raise GPU["ApiError"](502, "bad gateway")
            return {}
        self.g.update(runpod=runpod, runpod_list=lambda path, field, params=None: made,
                      find_offer=lambda cfg, oid: OFFER,
                      runpod_volumes=lambda cfg: {"US-1": {"id": "v", "kind": "network"}})
        args = argparse.Namespace(offer=OFFER["id"], pr=7, hours=1, allow_over_budget=False, dry_run=False)
        with unittest.mock.patch("subprocess.Popen"), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            GPU["cmd_launch"](args, CFG)
        self.assertEqual(list(GPU["ledger"]()), ["runpod:p1"], "the pod the lost create made never reached the ledger")

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
        with contextlib.redirect_stdout(io.StringIO()):  # what SKILL.md tells a session without a key to do
            GPU["cmd_extend"](argparse.Namespace(instance="runpod:p1", hours=0, allow_over_budget=False), CFG)

    def test_offers_with_no_provider_answering_is_not_no_availability(self):
        os.environ.pop("RUNPOD_API_KEY")
        self.addCleanup(os.environ.__setitem__, "RUNPOD_API_KEY", "k")
        out = io.StringIO()
        args = argparse.Namespace(gpus=1, min_vram=80, all_regions=False, allow_arm=False, json=False)
        with self.assertRaises(SystemExit) as exited, contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
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


if __name__ == "__main__":
    unittest.main()
