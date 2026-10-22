"""Check how .github/review/render_summary.py renders the review's summary."""

import json
import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = runpy.run_path(str(REPO_ROOT / ".github/review/render_summary.py"))
render, open_priorities = SUMMARY["render"], SUMMARY["open_priorities"]


def review(skipped=()):
    return {"outside_diff": [], "since_last": "", "description_problems": [], "questions": [], "checked": [],
            "skipped": list(skipped), "types": ["bug"]}


def threads(*nodes):
    return [{"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": list(nodes)}}}}}]


def thread(priority, resolved=False, outdated=False, login="mstar-review"):
    first = {"author": {"login": login}, "body": f"**[{priority}]** `step` drops the last token"}
    return {"isResolved": resolved, "isOutdated": outdated, "comments": {"nodes": [first]}}


class TestReviewSummary(unittest.TestCase):
    def test_an_open_p1_makes_the_status_red(self):
        inline = open_priorities(threads(thread("P1"), thread("P2")))
        text = render(review(), inline, "0123456789", "light")
        self.assertIn("\n🔴 **1 P1, 1 P2** on `0123456`, light review\n", text)

    def test_threads_from_before_the_review_app_still_count(self):
        inline = open_priorities(threads(thread("P1", login="github-actions"), thread("P2", login="Gaurav-Shah05")))
        self.assertEqual(inline, ["P1"], "a finding posted as github-actions would drop out of the count")

    def test_resolved_and_outdated_threads_are_not_counted(self):
        inline = open_priorities(threads(thread("P1", resolved=True), thread("P1", outdated=True)))
        self.assertEqual(inline, [], "a fixed or dismissed finding would keep the status red")

    def test_a_review_that_skipped_files_is_not_green(self):
        text = render(review(skipped=["mstar/worker/worker.py"]), [], "0123456789", "light")
        self.assertIn("\n🟡 **No findings**", text, "a partial review would read as a clean one")

    def test_schema_has_no_single_quote(self):
        self.assertNotIn("'", json.dumps(SUMMARY["SCHEMA"]),
                         "claude-review.yml passes the schema to --json-schema in single quotes")


if __name__ == "__main__":
    unittest.main()
