"""Check how .github/review/render_summary.py renders the review's summary."""

import json
import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = runpy.run_path(str(REPO_ROOT / ".github/review/render_summary.py"))
render, open_priorities, read_state = SUMMARY["render"], SUMMARY["open_priorities"], SUMMARY["read_state"]


def review(skipped=(), prior=()):
    return {"outside_diff": [], "since_last": "", "description_problems": [], "questions": [], "checked": [],
            "skipped": list(skipped), "types": ["bug"], "prior_findings": list(prior), "design": [],
            "risk": {"level": "low", "reason": "docs only"}}


def threads(*nodes):
    return [{"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": list(nodes)}}}}}]


def thread(priority, resolved=False, outdated=False, login="mstar-review", comment_id=7):
    first = {"author": {"login": login}, "databaseId": comment_id,
             "body": f"**[{priority}]** `step` drops the last token"}
    return {"isResolved": resolved, "isOutdated": outdated, "comments": {"nodes": [first]}}


class TestReviewSummary(unittest.TestCase):
    def test_an_open_p1_makes_the_status_red(self):
        inline = open_priorities(threads(thread("P1"), thread("P2")))
        text = render(review(), inline, "0123456789", "light")
        self.assertIn("\n🔴 **1 P1, 1 P2** on `0123456`, light review", text)

    def test_threads_from_before_the_review_app_still_count(self):
        inline = open_priorities(threads(thread("P1", login="github-actions"), thread("P2", login="Gaurav-Shah05")))
        self.assertEqual(inline, [("7", "P1")], "a finding posted as github-actions would drop out of the count")

    def test_resolved_and_outdated_threads_are_not_counted(self):
        inline = open_priorities(threads(thread("P1", resolved=True), thread("P1", outdated=True)))
        self.assertEqual(inline, [], "a fixed or dismissed finding would keep the status red")

    def test_an_earlier_finding_judged_open_counts_once_whatever_its_thread(self):
        # The author resolved thread 8 without fixing it; thread 7 is still open and also judged open.
        prior = [{"ref": "8", "priority": "P1", "status": "open", "why": "unchanged"},
                 {"ref": "7", "priority": "P2", "status": "open", "why": "unchanged"},
                 {"ref": "9", "priority": "P1", "status": "fixed", "why": "guard added"}]
        inline = open_priorities(threads(thread("P2", comment_id=7), thread("P1", resolved=True, comment_id=8)))
        text = render(review(prior=prior), inline, "0123456789", "light")
        self.assertIn("🔴 **1 P1, 1 P2**", text, "resolving a thread would clear a finding the re-review found open")

    def test_state_survives_model_text_that_closes_a_comment(self):
        r = review()
        r["outside_diff"] = [{"priority": "P2", "claim": "x --> <!-- y", "location": "a.py:1", "detail": "-->"}]
        state = read_state(render(r, [], "0123456789", "gpu"))
        self.assertEqual(state["outside_diff"], r["outside_diff"], "a re-review would start from a corrupted state")

    def test_a_review_that_skipped_files_is_not_green(self):
        text = render(review(skipped=["mstar/worker/worker.py"]), [], "0123456789", "light")
        self.assertIn("\n🟡 **No findings**", text, "a partial review would read as a clean one")

    def test_schema_has_no_single_quote(self):
        self.assertNotIn("'", json.dumps(SUMMARY["SCHEMA"]),
                         "claude-review.yml passes the schema to --json-schema in single quotes")


if __name__ == "__main__":
    unittest.main()
