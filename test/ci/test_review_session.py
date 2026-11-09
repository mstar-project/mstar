"""Check .github/review/session.py: the review's usage record and its /review continue handoff."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION = runpy.run_path(str(REPO_ROOT / ".github/review/session.py"))
load, usage, handoff, diff_paths = SESSION["load"], SESSION["usage"], SESSION["handoff"], SESSION["diff_paths"]
numbered = SESSION["numbered"]
ROOT = "/home/runner/work/mstar/mstar"


def assistant(mid, *content, out=10, parent=None):
    return {"type": "assistant", "parent_tool_use_id": parent, "message": {
        "id": mid, "model": "claude-opus-5-5", "content": list(content),
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
                  "output_tokens": out}}}


def tool(tid, name, **inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


# As Claude Code writes it: this system message's "message" is a string, not an object.
DENIED = {"type": "system", "subtype": "permission_denied", "tool_name": "Glob",
          "message": "Claude requested permissions to read from /tmp/x, but you haven't granted it yet."}


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

    def test_a_run_cut_off_before_its_result_is_priced_from_its_tokens(self):
        # 2 input, 100 cache writes, 1000 cache reads and 10 output tokens at Opus 5.5's prices.
        self.assertAlmostEqual(usage([assistant("msg_a", text("x"))])["total_cost_usd"], 908 / 1e6)

    def test_synthetic_messages_are_not_api_calls(self):
        msg = assistant("msg_s", text("Credit balance is too low"))
        msg["message"]["model"] = "<synthetic>"
        self.assertEqual(usage([msg])["api_calls"], 0)

    def test_a_message_that_is_a_string_is_skipped(self):
        msgs = [assistant("msg_a", text("x")), DENIED]
        self.assertEqual(usage(msgs)["api_calls"], 1)
        self.assertEqual(handoff(msgs, ROOT, [])["recent_text"], "x")

    def test_execution_file_and_transcript_together_count_once(self):
        msgs = [{**assistant("msg_a", text("x")), "uuid": "u1"}, {**assistant("msg_b", text("y")), "uuid": "u2"}]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "exec.json").write_text(json.dumps(msgs))
            proj = Path(d, "projects", "-home-runner")
            proj.mkdir(parents=True)
            Path(proj, "sid.jsonl").write_text("\n".join(map(json.dumps, msgs)) + "\nnot json\n")
            loaded = load([str(Path(d, "exec.json")), str(Path(d, "projects")), str(Path(d, "missing.json"))])
        u = usage(loaded)
        self.assertEqual(handoff(loaded, ROOT, [])["recent_text"], "x\ny", "each text would be handed on twice")
        self.assertEqual(u["api_calls"], 2)
        self.assertIsNone(u["num_turns"], "a run without a result message has no turn count")
        self.assertEqual((u["tokens_from"], u["output_tokens"]), ("messages", 20))


class TestHandoff(unittest.TestCase):
    def run_messages(self):
        return [
            assistant("m1", text("Reading the diff."), tool("r1", "Read", file_path=f"{ROOT}/pr-head/mstar/a.py")),
            assistant("m2", text("Progress:\n- P1 `mstar/a.py:10` drops a token, verified"),
                      tool("r2", "Read", file_path=f"{ROOT}/pr-head/mstar/b.py", offset=100, limit=50)),
            assistant("m5", text("subagent text"), parent="task1"),
        ]

    def test_records_reads_and_the_latest_progress_note(self):
        h = handoff(self.run_messages(), ROOT, ["mstar/a.py", "mstar/b.py", "mstar/c.py"])
        self.assertEqual(h["reads"], ["pr-head/mstar/a.py", "pr-head/mstar/b.py"])
        self.assertEqual(h["not_opened"], ["mstar/c.py"])
        self.assertTrue(h["note"].startswith("Progress:"))
        self.assertEqual(h["recent_text"], "", "a subagent's text would read as the reviewer's own")

    def test_a_continued_run_merges_onto_its_handoff(self):
        after_note = assistant("m7", text("Candidate P1 `mstar/b.py:40`, verifying.\n\nNext: c.py"))
        first = json.loads(json.dumps(handoff(self.run_messages() + [after_note], ROOT, ["mstar/a.py", "mstar/c.py"])))
        second = handoff([assistant("n1", text("No note here"),
                                    tool("r3", "Read", file_path=f"{ROOT}/pr-head/mstar/c.py"))],
                         ROOT, ["mstar/a.py", "mstar/c.py"], first)
        self.assertEqual(second["runs"], 2)
        self.assertEqual(second["not_opened"], [])
        self.assertTrue(second["note"].startswith("Progress:"), "the last progress note would be lost")
        self.assertEqual(second["recent_text"], "Candidate P1 `mstar/b.py:40`, verifying.\n\nNext: c.py\nNo note here",
                         "what the first run wrote after its note would be dropped")
        self.assertIn("No note here", SESSION["handoff_md"](second, "568", "abc", "light"),
                      "the first run's note would hide what the second run said last")
        paragraphs = handoff([assistant("p1", text("A" * 1400 + "\n\n" + "B" * 1400))], ROOT, [])
        third = handoff([assistant("p2", text("C" * 300))], ROOT, [], paragraphs)["recent_text"]
        self.assertFalse(third.startswith("\n"), "a kept part of an earlier message would open on a blank line")

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
        kept = "Progress: P1 a.py:3 verified. " + "x " * 10000
        self.assertEqual(handoff([assistant("m1", text(kept))], ROOT, [])["note"], kept.rstrip(),
                         "a stopped run's findings would be cut, and a finding not in the note is lost")
        note = "Progress: " + " ".join([token] * 1000) + " left: b.py"
        h = handoff([assistant("m1", text(note))], ROOT, [])
        self.assertTrue(h["note"].endswith("left: b.py"), "what is left would be cut off")
        self.assertLessEqual(set(h["note"].split()), {"Progress:", token, "[...]", "left:", "b.py"},
                             "a cut inside a credential leaves a fragment the redaction misses")

    def test_diff_paths_skip_deleted_files_and_hunk_lines(self):
        diff = ("diff --git a/x b/x\n--- a/x\n+++ /dev/null\ndiff --git a/y b/y\n--- a/y\n+++ b/y\n"
                "@@ -1 +1,2 @@\n+++ b/not-a-file\n"
                'diff --git "a/caf\\303\\251" "b/caf\\303\\251"\n--- "a/caf\\303\\251"\n+++ "b/caf\\303\\251\\t1"\n'
                # A hunk the PR's author wrote to look like a quoted path.
                '@@ -1 +1 @@\n--- a/x\n+++ "b/x" ' + "+" * 30000 + "\n")
        self.assertEqual(diff_paths(diff), ["y", "caf\u00e9\t1"], "a path git quotes would be left out")

    def test_numbered_gives_each_hunk_line_its_file_line_across_files_renames_and_deletions(self):
        def diff(a, r, b, c, d, e):
            return ("diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n"
                    f"@@ -0,0 +1,2 @@\n{a}+one\n{a}+two\n\\ No newline at end of file\n"
                    "diff --git a/old.py b/moved.py\nsimilarity index 90%\nrename from old.py\nrename to moved.py\n"
                    "--- a/old.py\n+++ b/moved.py\n"
                    f"@@ -10,3 +20,3 @@ def f():\n{r} ctx\n{b}-gone\n{c}+back\n{d} ctx2\n"
                    # A deletion and an addition that look like file headers, then a second hunk.
                    f"@@ -40,2 +50,2 @@\n{b.replace('11', '40')}--- a/x\n{c.replace('21', '50')}+++ b/x\n{e} end\n"
                    "diff --git a/dead.py b/dead.py\ndeleted file mode 100644\n--- a/dead.py\n+++ /dev/null\n"
                    f"@@ -1,2 +0,0 @@\n{b.replace('11', '1')}-a\n{b.replace('11', '2')}-b\n")

        raw = diff(*[""] * 6)
        plain = "R{} |", "L11 |", "R21 |", "R22 |", "R51 |"
        marked = diff("R1 |", "R20 |", "L11 |", "R21 |", "R22 |", "R51 |").replace("R1 |+two", "R2 |+two")
        self.assertEqual(numbered(raw), marked, "a comment would land on the wrong line")
        self.assertEqual(diff_paths(numbered(raw)), ["new.py", "moved.py"],
                         "a handoff would lose the changed files it never opened, or list a hunk line as one")
        self.assertNotIn("L", "".join(x.split("|")[0] for x in numbered(raw, left=False).split("\n") if "|" in x),
                         "a since.diff's old side isn't the PR's base")
        broken = "@@ -1,2 +1,2 @@\n a\nnot a hunk line\n b\n"
        self.assertEqual(numbered(broken), "@@ -1,2 +1,2 @@\nR1 | a\nnot a hunk line\n b\n",
                         "text after a hunk that doesn't match its counts would be numbered")


if __name__ == "__main__":
    unittest.main()
