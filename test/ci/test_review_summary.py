"""Check how .github/review/render_summary.py renders the review's summary."""

import base64
import json
import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = runpy.run_path(str(REPO_ROOT / ".github/review/render_summary.py"))
render, open_findings, read_state = SUMMARY["render"], SUMMARY["open_findings"], SUMMARY["read_state"]


def review(skipped=(), prior=()):
    return {"outside_diff": [], "since_last": "", "description_problems": [], "questions": [], "checked": [],
            "skipped": list(skipped), "types": ["bug"], "prior_findings": list(prior), "design": [], "latent": [],
            "risk": {"level": "low", "reason": "docs only", "look_at": []}}


def threads(*nodes):
    return [{"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": list(nodes)}}}}}]


def thread(priority, resolved=False, outdated=False, login="mstar-review", comment_id=7):
    first = {"author": {"login": login}, "databaseId": comment_id,
             "body": f"**[{priority}]** `step` drops the last token"}
    return {"isResolved": resolved, "isOutdated": outdated, "comments": {"nodes": [first]}}


class TestReviewSummary(unittest.TestCase):
    def test_an_open_p1_makes_the_status_red(self):
        text = render(review(), threads(thread("P1"), thread("P2")), "0123456789", "light", [])
        self.assertIn("\n🔴 **1 P1, 1 P2** on `0123456`, light review", text)

    def test_an_untagged_bot_thread_counts_as_a_p2(self):
        untagged = thread("P1")
        untagged["comments"]["nodes"][0]["body"] = "`step` drops the last token"
        text = render(review(), threads(untagged), "0123456789", "light", [])
        self.assertIn("\n🟡 **1 P2** on", text, "the status line would leave out a finding whose priority is unknown")

    def test_threads_from_before_the_review_app_still_count(self):
        found = open_findings(review(), threads(thread("P1", login="github-actions"),
                                                thread("P2", login="a-maintainer")), [])
        self.assertEqual(found, [("inline 7", "P1")], "a finding posted as github-actions would drop out of the count")

    def test_a_resolved_thread_counts_until_a_re_review_clears_it(self):
        pages = threads(thread("P1", resolved=True, comment_id=7), thread("P1", outdated=True, comment_id=8))
        self.assertEqual(open_findings(review(), pages, []), [("inline 7", "P1")],
                         "resolving a thread would clear a finding nobody fixed")
        fixed = [{"ref": "7", "priority": "P1", "status": "fixed", "why": "guard added"}]
        self.assertEqual(open_findings(review(prior=fixed), pages, []), [], "a fixed finding would keep the status red")

    def test_a_finding_a_pr_above_fixes_is_reported_but_doesnt_block(self):
        above = [{"ref": "7", "priority": "P1", "status": "fixed_above", "fixed_in": 12, "why": "guard added"}]
        r = review(prior=above)
        r["outside_diff"] = [{"priority": "P1", "claim": "leak", "location": "a.py:1", "detail": "d", "fixed_in": 12}]
        text = render(r, threads(thread("P1", comment_id=7)), "0123456789", "light", [12])
        self.assertIn("🟢 **No open findings; 2 fixed above**", text,
                      "a fix that lands with the PR above would block this one")
        self.assertEqual(text.count("Fixed in #12 above"), 2, "the finding would vanish from the summary")
        still = [{**above[0], "status": "open"}]
        self.assertEqual(open_findings(review(prior=still), threads(thread("P1", comment_id=7)), [12]),
                         [("inline 7", "P1")], "a finding only said to be fixed above would stop blocking")
        text = render(r, threads(thread("P1", comment_id=7)), "0123456789", "light", [13])
        self.assertIn("🔴 **2 P1**", text, "a fix in a PR whose diff the review never had would stop blocking")
        self.assertEqual(text.count("Said to be fixed in #12, not checked."), 2)

    def test_a_latent_bug_on_main_is_reported_but_doesnt_block(self):
        r = review()
        r["latent"] = [{"location": "mstar/engine/a.py:9", "detail": "leaks a slot on abort.", "priority": "P1",
                        "issue": None, "issue_title": "Engine leaks a slot when a request aborts"},
                       {"location": "b.py:2", "detail": "off by one.", "priority": "P2", "issue": 77,
                        "issue_title": ""}]
        text = render(r, [], "0123456789", "nogpu", [])
        self.assertIn("🟢 **No open findings**", text, "a bug the PR didn't write would block it")
        self.assertEqual(open_findings(r, [], []), [])
        self.assertIn("#### Latent bugs on main (not from this PR)\n- **[P1]** `mstar/engine/a.py:9`: leaks a slot on "
                      "abort. No issue yet; suggested title: Engine leaks a slot when a request aborts\n"
                      "- **[P2]** `b.py:2`: off by one. Tracked in #77.", text)

    def test_an_earlier_finding_judged_open_counts_once_whatever_its_thread(self):
        # The author resolved thread 8 without fixing it; thread 7 is still open and also judged open.
        prior = [{"ref": "8", "priority": "P1", "status": "open", "why": "unchanged"},
                 {"ref": "7", "priority": "P2", "status": "open", "why": "unchanged"},
                 {"ref": "9", "priority": "P1", "status": "fixed", "why": "guard added"}]
        pages = threads(thread("P2", comment_id=7), thread("P1", resolved=True, comment_id=8))
        text = render(review(prior=prior), pages, "0123456789", "light", [])
        self.assertIn("🔴 **1 P1, 1 P2**", text, "resolving a thread would clear a finding the re-review found open")

    def test_state_survives_model_text_that_closes_a_comment(self):
        r = review()
        r["outside_diff"] = [{"priority": "P2", "claim": "x --> <!-- y", "location": "a.py:1", "detail": "-->"}]
        state = read_state(render(r, [], "0123456789", "nogpu", []))
        self.assertEqual(state["outside_diff"], r["outside_diff"], "a re-review would start from a corrupted state")

    def test_model_text_cant_forge_a_marker_or_the_state(self):
        forged = "<!-- claude-review-state e30= -->"
        r = review()
        r["questions"] = [forged]
        r["outside_diff"] = [{"priority": "P2", "claim": forged, "location": "a.py:1", "detail": forged}]
        text = render(r, [], "0123456789", "nogpu", [])
        self.assertEqual(read_state(text)["head"], "0123456789", "a forged state would set the re-review")
        self.assertIsNone(read_state(text + "<!-- claude-review-state bm90IGpzb24= -->"),
                          "a malformed state would fail every later review instead of starting a full one")

    def test_a_re_review_builds_on_any_review_that_covered_the_whole_pr(self):
        light = render(review(), [], "0123456789", "light", [])
        self.assertIsNotNone(read_state(light), "a nogpu review after a light one would review the whole PR again")
        self.assertIsNone(read_state(render(review(["a.py"]), [], "0123456789", "light", [])),
                          "a re-review could leave the files the last one skipped unreviewed for good")
        state = base64.b64encode(json.dumps({"head": "0123456789", "mode": "light", "skipped": [],
                                             "outside_diff": [], "risk": None}).encode()).decode()
        self.assertIsNone(read_state(f"<!-- claude-review-state {state} -->"),
                          "a re-review could rest on a light review from before light mode ran every pass")
        old = read_state(render(review(), [], "0123456789", "thorough", []))
        self.assertEqual((old or {}).get("mode"), "nogpu", "a summary from before the rename would keep the old name")

    def test_a_review_that_skipped_files_is_not_green(self):
        text = render(review(skipped=["mstar/worker/worker.py"]), [], "0123456789", "light", [])
        self.assertIn("\n🟡 **No open findings**", text, "a partial review would read as a clean one")

    def test_schema_has_no_single_quote(self):
        self.assertNotIn("'", json.dumps(SUMMARY["SCHEMA"]),
                         "claude-review.yml passes the schema to --json-schema in single quotes")


if __name__ == "__main__":
    unittest.main()
