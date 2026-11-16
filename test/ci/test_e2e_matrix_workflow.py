"""Safety checks for the opt-in GPU matrix workflow and its trusted reporter."""

import json
import runpy
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/e2e-matrix.yml").read_text()
REPORTER = runpy.run_path(str(ROOT / ".github/e2e/report_pr.py"))


def row(*, flagged=False, correct="pass"):
    return {"model": "kokoro", "case": "text", "correct": correct, "flagged": flagged,
            "starts": {"base": True, "head": True}, "quality": [],
            "latency": {"delta_pct": 2.1, "verdict": "regression" if flagged else "same"},
            "throughput": {"delta_pct": -1.3, "verdict": "same"}}


class MatrixWorkflowTests(unittest.TestCase):
    def test_gpu_job_cannot_get_write_token_or_secrets(self):
        compare = WORKFLOW.split("\n  compare:", 1)[1].split("\n  report:", 1)[0]
        self.assertIn("pull_request_target:", WORKFLOW)
        self.assertIn("types: [labeled]", WORKFLOW)
        self.assertIn("needs.gate.outputs.enabled == 'true'", compare)
        self.assertIn("runs-on: [self-hosted, Linux, X64, e2e-ephemeral, gpu]", compare)
        self.assertIn("persist-credentials: false", compare)
        self.assertIn('test "$MSTAR_E2E_EPHEMERAL" = 1', compare)
        self.assertIn('test "$(git rev-parse FETCH_HEAD)" = "$HEAD_SHA"', compare)
        self.assertNotIn("secrets.", compare)
        self.assertNotIn("write", compare.split("\n    permissions:", 1)[1].split("\n    outputs:", 1)[0])

    def test_summary_cannot_turn_missing_or_flagged_result_green(self):
        result = REPORTER["conclusion"]
        self.assertEqual(result("0", []), "neutral")
        self.assertEqual(result("0", [{}]), "neutral")
        self.assertEqual(result("0", [row(correct="FAIL", flagged=False)]), "neutral")
        self.assertEqual(result("0", [row(flagged=True)]), "neutral")
        self.assertEqual(result("0", [row(correct="incomplete")]), "neutral")
        self.assertEqual(result("1", [row(flagged=True)]), "failure")
        self.assertEqual(result("1", [row()]), "neutral")
        self.assertEqual(result("2", [row()]), "neutral")
        self.assertEqual(result("0", [row()]), "success")

    def test_untrusted_fields_cannot_add_markdown_or_mentions(self):
        bad = row()
        bad["model"] = "evil|@owner\n<script>"
        text = REPORTER["render"]([bad], "neutral", "a" * 40, "https://github.com/x/y/actions/runs/1")
        self.assertNotIn("@owner", text)
        self.assertNotIn("<script>", text)
        self.assertNotIn("evil|", text)
        self.assertIn("| evil??owner ?script? |", text)

    def test_reporter_edits_only_its_own_prior_comment(self):
        calls = []

        def fake_api(method, path, token, body=None):
            calls.append((method, path, body))
            if method == "GET":
                return ([{"id": 1, "body": REPORTER["MARKER"], "user": {"login": "a-human"}},
                         {"id": 2, "body": REPORTER["MARKER"], "user": {"login": "github-actions[bot]"}}], {})
            return ({}, {})

        with mock.patch.dict(REPORTER["post"].__globals__,
                             {"api": fake_api, "os": mock.Mock(environ={
                                 "GITHUB_REPOSITORY": "mstar-project/mstar", "GH_TOKEN": "token"})}):
            REPORTER["post"](580, "a" * 40, "https://github.com/x/y/actions/runs/1", "success", [row()])
        self.assertEqual(calls[1][:2], ("PATCH", "/repos/mstar-project/mstar/issues/comments/2"))
        self.assertEqual(calls[2][2]["conclusion"], "success")

    def test_summary_size_and_shape_are_bounded(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            path.write_text(json.dumps([row()] * (REPORTER["MAX_ROWS"] + 1)))
            with self.assertRaises(ValueError):
                REPORTER["load_rows"](path)
            path.write_text(json.dumps([row()]))
            self.assertEqual(REPORTER["load_rows"](path), [row()])
            path.write_text(json.dumps([{}]))
            with self.assertRaises(ValueError):
                REPORTER["load_rows"](path)
            path.write_text(json.dumps([row(correct="FAIL", flagged=False)]))
            with self.assertRaises(ValueError):
                REPORTER["load_rows"](path)


if __name__ == "__main__":
    unittest.main()
