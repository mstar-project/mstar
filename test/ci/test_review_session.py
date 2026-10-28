"""Check .github/review/session.py: the review's usage record and its /review continue handoff."""

import argparse
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION = runpy.run_path(str(REPO_ROOT / ".github/review/session.py"))
load, usage, handoff, diff_paths = SESSION["load"], SESSION["usage"], SESSION["handoff"], SESSION["diff_paths"]
ROOT = "/home/runner/work/mstar/mstar"


def assistant(mid, *content, out=10, parent=None):
    return {"type": "assistant", "parent_tool_use_id": parent, "message": {
        "id": mid, "model": "claude-opus-5-5", "content": list(content),
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
                  "output_tokens": out}}}


def tool(tid, name, **inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def result(tid, error=None, out='{"success": true, "comment_id": 1}'):
    block = {"type": "tool_result", "tool_use_id": tid, "content": error or [{"type": "text", "text": out}]}
    if error:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}}


def text(t):
    return {"type": "text", "text": t}


COMMENT = "mcp__github_inline_comment__create_inline_comment"
# As Claude Code writes it: this system message's "message" is a string, not an object.
DENIED = {"type": "system", "subtype": "permission_denied", "tool_name": "Glob",
          "message": "Claude requested permissions to read from /tmp/x, but you haven't granted it yet."}


def run_main(*argv):
    """``session.py`` as the workflow runs it, with its output kept out of the test log."""
    quiet = io.StringIO()
    with mock.patch.object(sys, "argv", ["session.py", *argv]), \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        return SESSION["main"]()


def usage_args(d, *sources):
    return argparse.Namespace(pr="1", head="abc", mode="light", cont="false", outcome="failure",
                              out=str(Path(d, "u.json")), summary=str(Path(d, "summary.md")), sources=list(sources))


class TestUsage(unittest.TestCase):
    def test_one_response_over_several_lines_counts_once(self):
        msgs = [assistant("msg_a", text("x"), out=5), assistant("msg_a", tool("t1", "Read"), out=40),
                assistant("msg_b", text("y")),
                {"type": "result", "num_turns": 2, "total_cost_usd": 0.5, "subtype": "success"}]
        u = usage(msgs)
        self.assertEqual(u["api_calls"], 2, "each content block's line would count as an API call")
        self.assertEqual((u["cache_read_input_tokens"], u["output_tokens"]), (2000, 50))
        self.assertEqual((u["num_turns"], u["total_cost_usd"]), (2, 0.5))

    def test_the_result_s_counts_win_over_the_messages(self):
        per_model = {"claude-opus-5-5": {"inputTokens": 3, "outputTokens": 70, "cacheReadInputTokens": 5000,
                                         "cacheCreationInputTokens": 200, "costUSD": 0.4},
                     "claude-haiku-4-5": {"inputTokens": 1, "outputTokens": 9, "cacheReadInputTokens": 0,
                                          "cacheCreationInputTokens": 0, "costUSD": 0.01}}
        u = usage([assistant("msg_a", text("x")), {"type": "result", "modelUsage": per_model}])
        self.assertEqual((u["tokens_from"], u["api_calls"]), ("result", 1))
        self.assertEqual((u["input_tokens"], u["output_tokens"], u["cache_read_input_tokens"]), (4, 79, 5000),
                         "calls missing from the transcript, e.g. another model's, would be left out")
        self.assertEqual(u["models"]["claude-haiku-4-5"]["cost_usd"], 0.01)

    def test_synthetic_messages_are_not_api_calls(self):
        msg = assistant("msg_s", text("Credit balance is too low"))
        msg["message"]["model"] = "<synthetic>"
        self.assertEqual(usage([msg])["api_calls"], 0)

    def test_a_message_that_is_a_string_is_skipped(self):
        msgs = [assistant("msg_a", text("x")), DENIED]
        self.assertEqual(usage(msgs)["api_calls"], 1)
        self.assertEqual(handoff(msgs, ROOT, [])["recent_text"], "x")

    def test_execution_file_and_transcript_together_count_once(self):
        msgs = [assistant("msg_a", text("x")), assistant("msg_b", text("y"))]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "exec.json").write_text(json.dumps(msgs))
            proj = Path(d, "projects", "-home-runner")
            proj.mkdir(parents=True)
            Path(proj, "sid.jsonl").write_text("\n".join(map(json.dumps, msgs)) + "\nnot json\n")
            u = usage(load([str(Path(d, "exec.json")), str(Path(d, "projects")), str(Path(d, "missing.json"))]))
        self.assertEqual(u["api_calls"], 2)
        self.assertIsNone(u["num_turns"], "a run without a result message has no turn count")
        self.assertEqual((u["tokens_from"], u["output_tokens"]), ("messages", 20))

    def test_a_corrupt_or_missing_execution_file_still_records(self):
        with tempfile.TemporaryDirectory() as d:
            truncated = Path(d, "truncated.json")
            truncated.write_text('[{"type": "assistant", "message": ')
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                for source in (truncated, Path(d, "missing.json")):
                    self.assertEqual(SESSION["record_usage"](usage_args(d, str(source))), 0, source.name)
                    self.assertEqual(json.loads(Path(d, "u.json").read_text())["api_calls"], 0, source.name)
            self.assertTrue(Path(d, "summary.md").read_text().strip(), "the job summary would have no table")


class TestHandoff(unittest.TestCase):
    def run_messages(self):
        return [
            assistant("m1", text("Reading the diff."), tool("r1", "Read", file_path=f"{ROOT}/pr-head/mstar/a.py")),
            assistant("m2", text("Progress:\n- P1 `mstar/a.py:10` drops a token, verified"),
                      tool("r2", "Read", file_path=f"{ROOT}/pr-head/mstar/b.py", offset=100, limit=50)),
            assistant("m3", tool("c1", COMMENT, path="mstar/a.py", line=10, body="**[P1]** drops a token\n\nmore")),
            result("c1"),
            assistant("m4", tool("c2", COMMENT, path="mstar/b.py", line=5, body="**[P2]** stale comment")),
            result("c2", error="line 5 is not in the diff"),
            assistant("m5", text("subagent text"), parent="task1"),
            assistant("m6", tool("c3", COMMENT, path="mstar/c.py", line=1, body="**[P2]** buffered")),
            result("c3", out='{"success": true, "buffered": true}'),
        ]

    def test_records_comments_reads_and_the_latest_progress_note(self):
        h = handoff(self.run_messages(), ROOT, ["mstar/a.py", "mstar/b.py", "mstar/c.py"])
        status = {c["path"]: c["status"] for c in h["comments"]}
        self.assertEqual(status, {"mstar/a.py": "posted", "mstar/b.py": "failed", "mstar/c.py": "unknown"},
                         "a buffered comment, which may never post, would be listed as on the PR")
        self.assertEqual(h["reads"], ["pr-head/mstar/a.py", "pr-head/mstar/b.py"])
        self.assertEqual(h["not_opened"], ["mstar/c.py"])
        self.assertTrue(h["note"].startswith("Progress:"))
        self.assertEqual(h["recent_text"], "", "a subagent's text would read as the reviewer's own")
        md = SESSION["handoff_md"](h, "568", "abc", "light")
        self.assertIn("## Inline comments already posted\n\n- `mstar/a.py:10` **[P1]** drops a token", md)

    def test_a_continued_run_merges_onto_its_handoff(self):
        after_note = assistant("m7", text("Candidate P1 `mstar/b.py:40`, verifying.\n\nNext: c.py"))
        first = json.loads(json.dumps(handoff(self.run_messages() + [after_note], ROOT, ["mstar/a.py", "mstar/c.py"])))
        second = handoff([assistant("n1", text("No note here"),
                                    tool("r3", "Read", file_path=f"{ROOT}/pr-head/mstar/c.py"))],
                         ROOT, ["mstar/a.py", "mstar/c.py"], first)
        self.assertEqual(second["runs"], 2)
        self.assertEqual(len(second["comments"]), 3, "the first run's inline comments would be forgotten")
        self.assertEqual(second["not_opened"], [])
        self.assertTrue(second["note"].startswith("Progress:"), "the last progress note would be lost")
        self.assertEqual(second["recent_text"], "Candidate P1 `mstar/b.py:40`, verifying.\n\nNext: c.py\nNo note here",
                         "what the first run wrote after its note would be dropped")
        self.assertIn("No note here", SESSION["handoff_md"](second, "568", "abc", "light"),
                      "the first run's note would hide what the second run said last")

    def test_an_api_error_is_not_the_reviewer_s_output(self):
        msg = assistant("m1", text("API Error: 529 Overloaded"))
        msg["message"]["model"] = "<synthetic>"
        h = handoff([msg], ROOT, [])
        self.assertFalse(h["seen_assistant"], "a run that only hit an API error would leave a handoff")
        self.assertEqual(h["recent_text"], "")

    def test_without_a_note_the_recent_texts_are_handed_on_newest_last(self):
        h = handoff([assistant(f"m{i}", text(f"Checking file {i}.")) for i in range(5)], ROOT, [])
        self.assertEqual(h["recent_text"], "\n".join(f"Checking file {i}." for i in range(5)),
                         "only the run's last status line would be handed on")
        long = handoff([assistant(f"m{i}", text(f"{i} " + "x" * 900)) for i in range(5)], ROOT, [])
        self.assertEqual([t.split()[0] for t in long["recent_text"].split("\n")], ["2", "3", "4"],
                         "the bound would cut the newest texts, which say where the run stopped")

    def test_only_a_note_that_starts_with_progress_colon_counts(self):
        h = handoff([assistant("m1", text("**Progress:** P1 `a.py:3`, verified")),
                     assistant("m2", text("Progress was slow on this file."))], ROOT, [])
        self.assertEqual(h["note"], "**Progress:** P1 `a.py:3`, verified", "prose would replace the real note")

    def test_a_long_note_keeps_its_end_and_cuts_between_words(self):
        token = "ghs_" + "A" * 36
        note = "Progress: " + " ".join([token] * 400) + " left: b.py"
        h = handoff([assistant("m1", text(note))], ROOT, [])
        self.assertTrue(h["note"].endswith("left: b.py"), "what is left would be cut off")
        self.assertLessEqual(set(h["note"].split()), {"Progress:", token, "[...]", "left:", "b.py"},
                             "a cut inside a credential leaves a fragment the redaction misses")

    def test_diff_paths_skip_deleted_files_and_hunk_lines(self):
        diff = ("diff --git a/x b/x\n--- a/x\n+++ /dev/null\ndiff --git a/y b/y\n--- a/y\n+++ b/y\n"
                "@@ -1 +1,2 @@\n+++ b/not-a-file\n"
                'diff --git "a/caf\\303\\251" "b/caf\\303\\251"\n--- "a/caf\\303\\251"\n+++ "b/caf\\303\\251"\n')
        self.assertEqual(diff_paths(diff), ["y", "caf\u00e9"], "a path git quotes would be left out")


class TestCli(unittest.TestCase):
    def test_handoff_with_options_before_files_and_a_previous_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            ex, prev, out, partial = Path(d, "exec.json"), Path(d, "prev.json"), Path(d, "out"), Path(d, "partial.md")
            ex.write_text(json.dumps([
                assistant("m1", tool("c1", COMMENT, path="a.py", line=3, body="**[P2]** failed one")),
                result("c1", error="line 3 is not in the diff"),
                assistant("m2", text("**Progress:** P1 `b.py:7` unverified"),
                          tool("c2", COMMENT, path="b.py", line=7, body="**[P1]** unknown one")),
            ]))
            prev.write_text(json.dumps({"runs": 1, "comments": [], "reads": ["pr-head/z.py"], "note": "",
                                        "recent_text": ""}))
            code = run_main("handoff", "--pr", "7", "--head", "abc", "--mode", "light", "--out", str(out),
                            "--comment", str(partial), "--previous", str(prev), str(ex))
            h = json.loads(Path(out, "handoff.json").read_text())
            md = Path(out, "handoff.md").read_text()
            comment = partial.read_text()
        self.assertEqual(code, 0)
        self.assertEqual((h["pr"], h["runs"], h["reads"]), ("7", 2, ["pr-head/z.py"]))
        self.assertIn("## Inline comments that failed to post\n\n"
                      "- `a.py:3` **[P2]** failed one (line 3 is not in the diff)", md)
        self.assertIn("## Inline comments whose result is unknown (check pr-comments.json)\n\n"
                      "- `b.py:7` **[P1]** unknown one", md)
        self.assertNotIn("Inline comments it made", comment, "a comment that may not have posted would show as made")
        self.assertIn("**Progress:** P1 `b.py:7` unverified", md)
        self.assertNotIn("unverified", comment, "the public comment would list unverified leads")

    def test_a_continued_run_that_said_nothing_keeps_its_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            prev, out = Path(d, "prev.json"), Path(d, "out")
            prev.write_text(json.dumps({"runs": 1, "comments": [], "reads": [], "note": "Progress: a.py done",
                                        "recent_text": ""}))
            code = run_main("handoff", "--pr", "7", "--head", "abc", "--mode", "light", "--out", str(out),
                            "--previous", str(prev), str(Path(d, "missing.json")))
            self.assertEqual(code, 0, "the earlier handoff would no longer be offered to /review continue")
            self.assertEqual(json.loads(Path(out, "handoff.json").read_text())["note"], "Progress: a.py done")

    def test_a_corrupt_previous_handoff_starts_afresh(self):
        with tempfile.TemporaryDirectory() as d:
            ex, prev, out = Path(d, "exec.json"), Path(d, "prev.json"), Path(d, "out")
            ex.write_text(json.dumps([assistant("m1", text("Progress: nothing yet"))]))
            prev.write_text("{not json")
            code = run_main("handoff", "--pr", "7", "--head", "abc", "--mode", "light", "--out", str(out),
                            "--previous", str(prev), str(ex))
            h = json.loads(Path(out, "handoff.json").read_text())
        self.assertEqual((code, h["runs"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
