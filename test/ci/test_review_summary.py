"""Check how .github/review/render_summary.py renders the review's summary."""

import json
import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = runpy.run_path(str(REPO_ROOT / ".github/review/render_summary.py"))
render = SUMMARY["render"]


def review(*findings, skipped=()):
    return {"findings": list(findings), "since_last": "", "description_problems": [], "questions": [],
            "checked": [], "skipped": list(skipped), "types": ["bug"]}


def finding(priority, inline=True):
    return {"priority": priority, "claim": "`step` drops the last token.", "location": "mstar/worker/worker.py:10",
            "inline": inline, "detail": "" if inline else "A request with max_tokens=1 returns no token."}


class TestReviewSummary(unittest.TestCase):
    def test_any_p0_or_p1_makes_the_status_red(self):
        text = render(review(finding("P1"), finding("P2")), "0123456789", "light")
        self.assertIn("\n🔴 **1 P1, 1 P2** on `0123456`, light review\n", text)

    def test_a_review_that_skipped_files_is_not_green(self):
        text = render(review(skipped=["mstar/worker/worker.py"]), "0123456789", "light")
        self.assertIn("\n🟡 **No findings**", text, "a partial review would read as a clean one")

    def test_only_findings_outside_the_diff_are_listed(self):
        text = render(review(finding("P2"), finding("P2", inline=False)), "0123456789", "light")
        self.assertEqual(text.count("drops the last token"), 1, "an inline finding would be repeated in the summary")

    def test_schema_has_no_single_quote(self):
        self.assertNotIn("'", json.dumps(SUMMARY["SCHEMA"]),
                         "claude-review.yml passes the schema to --json-schema in single quotes")


if __name__ == "__main__":
    unittest.main()
