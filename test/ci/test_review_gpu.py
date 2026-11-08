"""Check the approval rule in .github/review/gpu_review.py, which decides whether the review bot approves."""

import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_REVIEW = runpy.run_path(str(REPO_ROOT / ".github/review/gpu_review.py"))
blocking, risky_files = GPU_REVIEW["blocking"], GPU_REVIEW["risky_files"]


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


if __name__ == "__main__":
    unittest.main()
