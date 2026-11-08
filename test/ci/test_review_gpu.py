"""Check the approval rule in .github/review/gpu_review.py, which decides whether the review bot approves."""

import argparse
import contextlib
import io
import runpy
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_REVIEW = runpy.run_path(str(REPO_ROOT / ".github/review/gpu_review.py"))
blocking, risky_files = GPU_REVIEW["blocking"], GPU_REVIEW["risky_files"]
GPU = runpy.run_path(str(REPO_ROOT / ".claude/skills/live-pr-test/scripts/gpu.py"))
SLURM_CFG = {"slurm": {"host": "root@node", "job_prefix": "review-bot", "workdir": "/workspace/review-bot"}}


def fake_slurm(replies, calls=None):
    """Stands in for gpu.py's slurm_run: each command gets the first reply whose key it contains, and
    an exception is a failed command (empty stdout with check=False)."""
    def slurm_run(cfg, command, check=True):
        if calls is not None:
            calls.append(command)
        reply = next(r for k, r in replies.items() if k in command)
        if isinstance(reply, Exception):
            if not check:
                return ""
            raise reply
        return reply
    return slurm_run


def thread(comment_id, body, outdated=False):
    return {"isOutdated": outdated}, {"databaseId": comment_id, "author": {"login": "mstar-review"}, "body": body}


def review(prior=(), outside=()):
    return {"outside_diff": list(outside), "prior_findings": list(prior)}


class TestApproval(unittest.TestCase):
    def test_open_findings_block_however_their_threads_were_closed(self):
        threads = [
            thread(1, "**[P1]** resolved by the author, never fixed"),
            thread(2, "**[P1]** judged fixed by the re-review"),
            thread(3, "**[P2]** its lines changed", outdated=True),
            thread(4, "**[P3]** a nit"),
            thread(5, "no tag, so its priority is unknown"),
        ]
        prior = [{"ref": "2", "priority": "P1", "status": "fixed", "why": "guard added"},
                 {"ref": "3", "priority": "P2", "status": "open", "why": "moved, not fixed"}]
        outside = [{"priority": "P2", "location": "mstar/engine/engine.py:10", "claim": "c", "detail": "d"}]
        found = blocking(review(prior, outside), threads)
        self.assertEqual(len(found), 4, found)
        for expected in ("inline 1", "inline 5", "engine.py:10", "earlier 3"):
            self.assertTrue(any(expected in f for f in found), f"{expected} would not block approval: {found}")

    def test_only_one_models_code_and_its_tests_can_be_low_risk(self):
        self.assertEqual(risky_files(["mstar/model/kokoro/model.py", "mstar/model/kokoro/tests/test_x.py",
                                      "docs/models.rst"], REPO_ROOT), [])
        for path in (".github/review/review-prompt.md", ".claude/skills/pr-review/SKILL.md", "AGENTS.md",
                     "mstar/worker/worker.py", "mstar/engine/resources/kv.py", "uv.lock", "pyproject.toml"):
            self.assertTrue(risky_files([path], REPO_ROOT), f"{path} could be approved as low risk")
        self.assertTrue(risky_files(["mstar/model/kokoro/model.py", "mstar/model/orpheus/model.py"], REPO_ROOT),
                        "a change to two models could be approved as low risk")



class TestSlurm(unittest.TestCase):
    def setUp(self):
        self.globals = GPU["status"].__globals__
        self.saved = self.globals["slurm_run"]

    def tearDown(self):
        self.globals["slurm_run"] = self.saved

    def test_a_failed_ssh_is_not_a_job_that_ended(self):
        e = {"provider": "slurm", "instance_id": "42", "run": True}
        self.globals["slurm_run"] = fake_slurm({"squeue": GPU["ApiError"](255, "ssh: connect to host node: timed out")})
        with self.assertRaises(GPU["ApiError"], msg="a flaky ssh would end a running live test"):
            GPU["status"](e, SLURM_CFG)
        self.globals["slurm_run"] = fake_slurm(
            {"squeue": GPU["ApiError"](1, "slurm: squeue: slurm_load_jobs error: Invalid job id specified")})
        self.assertEqual(GPU["status"](e, SLURM_CFG)[0], "ended")
        self.globals["slurm_run"] = fake_slurm({"squeue": "RUNNING\n"})
        self.assertEqual(GPU["status"](e, SLURM_CFG)[0], "running")

    def test_terminate_leaves_nothing_of_the_run_on_the_node(self):
        calls, saved = [], {k: self.globals[k] for k in ("load_config", "STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        state = Path(tempfile.mkdtemp())
        self.globals.update(slurm_run=fake_slurm({"scancel": ""}, calls), load_config=lambda: SLURM_CFG,
                            STATE=state, LEDGER=state / "instances.json")
        jobdir = "/workspace/review-bot/review-bot-7-1-1"
        GPU["terminate"]("slurm:42", {"provider": "slurm", "instance_id": "42", "dir": jobdir, "launched_at": 0,
                                      "usd_per_hour": 0.0})
        self.assertTrue(any("scancel 42" in c and f"rm -rf {jobdir}" in c for c in calls),
                        f"the job's directory (the Anthropic key, venvs, builds) would stay on the node: {calls}")

    def test_the_job_uses_the_shared_hf_cache(self):
        key = Path(tempfile.mkdtemp()) / "key.pub"
        key.write_text("ssh-ed25519 AAAA test")
        cfg = {**SLURM_CFG, "ssh_public_key": str(key), "budget": {"max_usd_per_hour": 0, "max_hours": 4},
               "slurm": {**SLURM_CFG["slurm"], "hf_home": "/workspace/shared/hf"}}
        self.globals["slurm_run"] = fake_slurm({"sinfo": "node1 gpu:8 gpu:0 idle\n"})
        args = argparse.Namespace(offer="slurm:node1:1", pr=7, hours=1, tag="1-1", run="gpu_review_box.sh",
                                  input=[], allow_over_budget=False, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            GPU["cmd_launch"](args, cfg)
        self.assertIn("HF_HOME=/workspace/shared/hf", out.getvalue(),
                      "each run would download the weights again into its own directory")


if __name__ == "__main__":
    unittest.main()
