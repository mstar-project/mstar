"""Check .github/review/bot.py, the review workflow's steps, against a stand-in for `gh`."""

import base64
import contextlib
import fnmatch
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / ".github/review"))
import bot  # noqa: E402
import render_summary  # noqa: E402


class Fail:
    """A `gh` call that fails, as GitHub's errors do."""

    def __init__(self, stderr="HTTP 502"):
        self.stderr = stderr


class FakeGh:
    """Stands in for `bot.run`: answers each call from `routes`, records it with its stdin in `calls`, and
    fails the test on a call no route answers, so a new GitHub write can't go untested.

    A route's key is "<METHOD> <endpoint>" for `gh api` (fnmatch patterns allowed) and the subcommand
    otherwise ("pr comment", "git clone"). Its value is the response: a str is printed as is, a Fail
    fails, a callable gets (argv, stdin) and returns either, and anything else is printed as JSON
    (wrapped as one page under --slurp)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    @staticmethod
    def key(argv):
        argv = list(argv)
        if Path(argv[0]).name.startswith("python"):
            argv = argv[1:]
        name = Path(argv[0]).name
        if name != "gh" or argv[1] != "api":
            return " ".join([name, *argv[1:2]] if name != "gh" else argv[1:3])
        args, method, endpoint, fields = argv[2:], None, None, False
        while args:
            a = args.pop(0)
            if a in ("-X", "--method"):
                method = args.pop(0)
            elif a in ("-f", "-F", "--raw-field", "--field", "--input"):
                fields = True
                args.pop(0)
            elif a in ("-H", "--header", "--jq", "-q"):
                args.pop(0)
            elif not a.startswith("-"):
                endpoint = a
        method = method or ("POST" if fields else "GET")
        return f"{method} {endpoint}"

    def writes(self):
        """(key, stdin) of every call that changes something on GitHub."""
        return [(self.key(argv), stdin) for argv, stdin in self.calls
                if not self.key(argv).startswith(("GET ", "pr diff", "pr list", "issue list", "run download"))
                and not (self.key(argv) == "POST graphql" and not any(a.startswith("query=mutation") for a in argv))]

    def __call__(self, argv, input=None, check=True, timeout=None):
        self.calls.append((list(argv), input))
        key = self.key(argv)
        answer = next((v for k, v in self.routes.items() if fnmatch.fnmatchcase(key, k)), None)
        if answer is None:
            raise AssertionError(f"unexpected call: {key} ({argv})")
        if callable(answer) and not isinstance(answer, Fail):
            answer = answer(argv, input)
        if isinstance(answer, Fail):
            if check:
                raise RuntimeError(f"{key}: {answer.stderr}")
            return subprocess.CompletedProcess(argv, 1, "", answer.stderr)
        if not isinstance(answer, str):
            answer = json.dumps([answer] if "--slurp" in argv else answer)
        return subprocess.CompletedProcess(argv, 0, answer, "")


@contextlib.contextmanager
def workspace():
    """A fresh directory to run a step in, as the workflow's workspace."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield Path(d)
        finally:
            os.chdir(cwd)


def step(fn, routes, **env):
    """Run the step `fn` with `env` as its parameters and `routes` answering `gh`; its outputs and the fake."""
    fake = FakeGh(routes)
    out = {}
    with mock.patch.object(bot, "run", fake):
        try:
            fn(out, **env)
        except SystemExit as e:
            out["exit"] = e.code
    return out, fake


class TestPreflight(unittest.TestCase):
    def preflight(self, command, perms, actor="an-author", author="an-author", assoc="CONTRIBUTOR"):
        routes = {f"GET repos/o/r/collaborators/{login}/permission": {"permission": perm}
                  for login, perm in perms.items()}
        routes["pr comment"] = "https://github.com/o/r/pull/1#issuecomment-9"
        out, fake = step(bot.preflight, routes, repo="o/r", pr="1", key_set="true", command=command, actor=actor,
                         author=author, assoc=assoc, skip_label="false")
        return out, [stdin for key, stdin in fake.writes()]

    def test_who_can_ask_for_which_review(self):
        write, read = {"an-author": "write"}, {"an-author": "read"}
        self.assertEqual(self.preflight("nogpu", write), ({"run": "true", "mode": "nogpu", "turns": "2000"}, []))
        self.assertEqual(self.preflight("light", read)[0],
                         {"run": "true", "mode": "light", "non_write_user": "an-author", "turns": "40"})
        out, replies = self.preflight("nogpu", read)
        self.assertEqual(out, {"run": "false"}, "a contributor without write access could start a nogpu review")
        self.assertIn("/review light", replies[0], "the contributor wouldn't be told what they can run")
        self.assertEqual(self.preflight("light", {"a-maintainer": "read"}, actor="a-maintainer"),
                         ({"run": "false"}, []), "someone without write access could review another author's PR")
        self.assertEqual(self.preflight("continue", read)[0]["mode"], "",
                         "a /review continue would not keep its own mode")


HEAD, LAST = "1" * 40, "2" * 40


class TestRestore(unittest.TestCase):
    def restore(self, meta, non_write="", path=".github/workflows/claude-review.yml"):
        """/review continue finding one stopped review whose saved handoff has `meta`."""
        def download(argv, stdin):
            saved = Path(argv[argv.index("-D") + 1])
            saved.mkdir(parents=True)
            (saved / "handoff.json").write_text(json.dumps({"pr": "7", "head_sha": HEAD, **meta}))
            (saved / "handoff.md").write_text("notes")
            return ""

        with tempfile.TemporaryDirectory() as temp:
            return step(bot.restore, {
                "GET repos/o/r/issues/7/comments": [{"id": 5, "user": {"login": "mstar-review[bot]"},
                                                     "body": bot.INCOMPLETE + "\nThe review didn't complete"}],
                "GET repos/o/r/actions/artifacts?*": {"artifacts": [
                    {"expired": False, "created_at": "2026-11-01", "workflow_run": {"id": 9}}]},
                "GET repos/o/r/actions/runs/9": {"path": path, "event": "issue_comment"},
                "run download": download,
            }, repo="o/r", pr="7", head_sha=HEAD, runner_temp=temp, non_write=non_write)[0]

    def test_a_failure_nobody_expected_is_still_a_refusal(self):
        fake = FakeGh({
            "GET repos/o/r/pulls/7": {"head": {"sha": HEAD, "repo": {"full_name": "o/r"}}, "title": "t"},
            "GET repos/o/r/issues/7/comments": [{"id": 5, "user": {"login": "mstar-review[bot]"},
                                                 "body": bot.INCOMPLETE + "\nThe review didn't complete"}],
            "GET repos/o/r/actions/artifacts?*": {"artifacts": [
                {"expired": False, "created_at": "2026-11-01", "workflow_run": None}]}})
        out = {}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(bot, "run", fake), self.assertRaises(TypeError):
            bot.resolve(out, "o/r", "7", "", temp)
        self.assertEqual(out["refused"], "true", "it would overwrite the stopped review's comment")

    def test_a_continue_takes_only_this_workflows_handoff_of_this_head(self):
        self.assertEqual(self.restore({"mode": "thorough"})["mode"], "nogpu",
                         "a review stopped before the rename couldn't continue")
        self.assertIn("new commits", self.restore({"mode": "light", "head_sha": LAST})["why"],
                      "a handoff from an older head would steer a review of new code")
        self.assertIn("no stopped review", self.restore({"mode": "light"}, path=".github/workflows/fork.yml")["why"],
                      "a fork's own workflow could plant the handoff")
        self.assertIn("write access", self.restore({"mode": "nogpu"}, non_write="an-author")["why"],
                      "a contributor could continue a nogpu review")


def summary(head, mode="light"):
    state = base64.b64encode(json.dumps({"head": head, "mode": mode, "skipped": [], "outside_diff": [],
                                         "risk": None, "latent": []}).encode()).decode()
    return {"id": 3, "user": {"login": "mstar-review[bot]"}, "created_at": "t",
            "body": f"<!-- claude-review-summary -->\nx\n<!-- claude-review-state {state} -->"}


BASE = "b" * 40


def collect(mode="light", diff="diff --git a/a.py b/a.py\n", **routes):
    """bot.collect on PR 7 at HEAD, its earlier comments and compares from `routes`; its outputs and files."""
    with workspace() as ws:
        out, _ = step(bot.collect, {
            "GET repos/o/r/pulls/7": {"title": "T", "body": None, "head": {"sha": HEAD, "ref": "x", "repo": None},
                                      "base": {"ref": "main", "sha": BASE, "repo": {"full_name": "o/r"}}},
            "pr diff": diff,
            "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}},
            "GET repos/o/r/pulls/7/comments": [],
            "GET repos/o/r/issues/7/comments": [],
            "issue list": Fail(),
            **routes}, repo="o/r", pr="7", head_sha=HEAD, mode=mode)
        return out, {p.name: p.read_text() for p in ws.iterdir()}


def compare(status, diff="diff --git a/b.py b/b.py\n", commits=()):
    """A compare of two commits, as JSON or, asked for the diff media type, as `diff`."""
    return lambda argv, stdin: diff if "Accept: application/vnd.github.diff" in argv \
        else json.dumps([{"status": status, "commits": list(commits)}] if "--slurp" in argv else {"status": status})


def pr_diff(context, added, index="1111111..2222222"):
    return (f"diff --git a/a.py b/a.py\nindex {index} 100644\n--- a/a.py\n+++ b/a.py\n"
            f"@@ -10,3 +10,3 @@ def f():\n     {context}\n-    y = 2\n+    y = 3\n     z = 4\n"
            f"@@ -40,2 +40,3 @@ def g():\n     return 1\n+    {added}\n")


class TestCollect(unittest.TestCase):
    CONVERSATION = {"GET repos/o/r/issues/7/comments": [summary(LAST)]}

    def test_a_re_review_reads_the_commits_since_on_an_ancestor(self):
        out, files = collect(**self.CONVERSATION, **{f"GET repos/o/r/compare/{LAST}...{HEAD}": compare("ahead")})
        self.assertEqual((out.get("rereview"), files["since.diff"]), ("true", "diff --git a/b.py b/b.py\n"))
        self.assertEqual(json.loads(files["last-review.json"])["head"], LAST)

    def test_with_the_review_app_only_its_login_is_the_bot(self):
        forged = {**summary(LAST), "user": {"login": "github-actions[bot]"}}
        routes = {"GET repos/o/r/issues/7/comments": [forged],
                  f"GET repos/o/r/compare/{LAST}...{HEAD}": compare("ahead")}
        self.assertEqual(collect(**routes)[0].get("rereview"), "true", "a fork, with no app, would lose re-reviews")
        self.addCleanup(importlib.reload, bot)
        self.addCleanup(importlib.reload, render_summary)
        with mock.patch.dict(os.environ, {"REVIEW_APP_CLIENT_ID": "app-id"}):
            importlib.reload(render_summary)
            importlib.reload(bot)
        self.assertIsNone(collect(**routes)[0].get("rereview"),
                          "any workflow on a branch could post the state a re-review starts from")
        inline = {"id": 7, "user": {"login": "github-actions[bot]"}, "body": "**[P1]** drops a token"}
        files = collect(**{"GET repos/o/r/pulls/7/comments": [inline]})[1]
        self.assertTrue(json.loads(files["pr-comments.json"])[0]["mine"],
                        "a team run's inline finding would stop being re-checked")
        team_run = {"isResolved": False, "isOutdated": False, "comments": {"nodes": [
            {"author": {"login": "github-actions"}, "databaseId": 7, "body": "**[P1]** drops a token"}]}}
        pages = [{"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [team_run]}}}}}]
        review = {"prior_findings": [], "outside_diff": [], "skipped": []}
        self.assertEqual(len(render_summary.open_findings(review, pages, [])), 1,
                         "a team run posts its inline findings as github-actions, and they would stop counting")

    def test_after_a_rebase_a_re_review_reads_only_the_files_that_changed(self):
        reviewed = pr_diff("x = 1", "# unreachable")
        routes = {**self.CONVERSATION, f"GET repos/o/r/compare/{LAST}...{HEAD}": compare("diverged"),
                  f"GET repos/o/r/compare/{BASE}...{LAST}": compare("ahead", reviewed)}
        out, files = collect(diff=pr_diff("x = 1", "# unreachable", "3333333..4444444").replace("@@ -10,3 +10,3",
                                                                                                "@@ -12,3 +12,3"),
                             **routes)
        self.assertEqual((out.get("rereview"), files["interdiff.diff"]), ("true", ""),
                         "a rebase that moved the PR's changes would read as new changes")
        out, files = collect(diff=pr_diff("x = 1", "# dead code"), **routes)
        self.assertEqual(files["interdiff.diff"], f"{pr_diff('x = 1', '# dead code')}{bot.DROPPED}{reviewed}")
        self.assertNotIn("since.diff", files)

    def test_a_merge_in_the_range_takes_the_interdiff(self):
        merge = {"sha": HEAD, "parents": [{"sha": LAST}, {"sha": BASE}], "commit": {"message": "Merge main"}}
        reviewed = pr_diff("x = 1", "# unreachable")
        out, files = collect(**self.CONVERSATION, diff=reviewed, **{
            f"GET repos/o/r/compare/{LAST}...{HEAD}": compare("ahead", commits=[merge]),
            f"GET repos/o/r/compare/{BASE}...{LAST}": compare("ahead", reviewed)})
        self.assertEqual((out.get("rereview"), files["interdiff.diff"]), ("true", ""),
                         "what the merge brought in from main would be reviewed as the author's change")
        self.assertNotIn("since.diff", files)


class TestInterdiff(unittest.TestCase):
    REVIEWED = pr_diff("x = 1", "guard()")

    def test_it_shows_what_moved_or_went_and_not_what_a_rebase_shifted(self):
        moved = self.REVIEWED.replace("def g():", "def h():")
        moved_inside = self.REVIEWED.replace("     return 1\n", "     pass\n")
        without_hunk = self.REVIEWED[:self.REVIEWED.index("@@ -40")]
        other = "diff --git a/b.py b/b.py\nindex 1..2 100644\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-a\n+b\n"
        mode = self.REVIEWED.replace("index 1111111", "old mode 100644\nnew mode 100755\nindex 1111111")
        binary = "diff --git a/w.bin b/w.bin\nindex {}\nBinary files a/w.bin and b/w.bin differ\n"
        between_blanks = ("diff --git a/a.py b/a.py\nindex 1..2 100644\n--- a/a.py\n+++ b/a.py\n"
                          "@@ -10,4 +10,5 @@ def f():\n     {}\n \n+    guard()\n \n     {}\n")
        faked_header = self.REVIEWED.replace("+    guard()\n",
                                             "+    guard()\n+    evil()\n+    s = '\fdiff --git a/a.py b/a.py\n")
        below = self.REVIEWED.replace("     z = 4\n", "     w = 5\n")
        cat = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"]
        carriage_return = bot.run(cat, input=self.REVIEWED.replace(
            "+    guard()\n", "+    guard()\n+    evil()\rdiff --git a/a.py b/a.py\n")).stdout
        type_change = ("diff --git a/l b/l\ndeleted file mode 100644\n--- a/l\n+++ /dev/null\n@@ -1 +0,0 @@\n-{}\n"
                       "diff --git a/l b/l\nnew file mode 120000\n--- /dev/null\n+++ b/l\n@@ -0,0 +1 @@\n+t\n")
        for name, old, new, shown in (
                ("a guard moved to another function", self.REVIEWED, moved, "def h()"),
                ("a guard moved inside its function", self.REVIEWED, moved_inside, "pass"),
                ("a guard moved between blank lines in its function", between_blanks.format("a()", "b()"),
                 between_blanks.format("c()", "d()"), "c()"),
                ("the line below a change", self.REVIEWED, below, "w = 5"),
                ("a form feed faking a file header", self.REVIEWED, faked_header, "evil()"),
                ("a carriage return faking a file header", self.REVIEWED, carriage_return, "evil()"),
                ("the first half of a type change", type_change.format("a"), type_change.format("b"), "-b"),
                ("a dropped hunk", self.REVIEWED, without_hunk, "def g()"),
                ("a dropped file", self.REVIEWED + other, self.REVIEWED, "b.py"),
                ("a mode change beside hunks", self.REVIEWED, mode, "new mode"),
                ("a binary swapped", binary.format("1..2 100644"), binary.format("1..3 100644"), "1..3")):
            with self.subTest(name):
                self.assertIn(shown, bot.interdiff(old, new))
        self.assertIn("evil()\rdiff --git", carriage_return, "a bare \\r would end the line, faking a file header")
        self.assertEqual(bot.interdiff(binary.format("1..2 100644"), binary.format("9..2 100644")), "",
                         "main changing a binary's old side would read as the PR's change")
        rename = ("diff --git a/x b/y\nsimilarity index {}%\nrename from x\nrename to y\n--- a/x\n+++ b/y\n"
                  "@@ -1 +1 @@\n-a\n+b\n")
        self.assertEqual(bot.interdiff(rename.format(90), rename.format(85)), "",
                         "a rebase changing how similar a renamed file is would read as the PR's change")
        self.assertIn(bot.DROPPED, bot.interdiff(self.REVIEWED + other, self.REVIEWED))


class TestDispatcher(unittest.TestCase):
    def test_a_step_that_exits_still_writes_its_outputs_with_env_as_parameters(self):
        seen = {}

        def failing(out, pr, mode="light"):
            seen.update(pr=pr, mode=mode)
            out["reason"] = "a\nb"
            sys.exit(1)

        output = Path(tempfile.mkdtemp()) / "output"
        with mock.patch.dict(os.environ, {"PR": "7", "MODE": "", "GITHUB_OUTPUT": str(output)}), \
                mock.patch.object(sys, "argv", ["bot.py", "failing"]), self.assertRaises(SystemExit):
            bot.main({"failing": failing})
        self.assertEqual(seen, {"pr": "7", "mode": "light"})
        self.assertRegex(output.read_text(), r"^reason<<EOF_\w+\na\nb\nEOF_\w+\n$",
                         "the steps after a failed one read what it had found from $GITHUB_OUTPUT")


COMMENT = "mcp__github_inline_comment__create_inline_comment"


def assistant(*content):
    return {"type": "assistant", "message": {"id": "m", "model": "claude-opus-5-5", "content": list(content)}}


def call(tid, **inp):
    return {"type": "tool_use", "id": tid, "name": COMMENT, "input": inp}


class TestHandoff(unittest.TestCase):
    def handoff(self, messages, previous=None):
        """bot.handoff after a run that left `messages`, continuing a run whose handoff.json is `previous`."""
        with tempfile.TemporaryDirectory() as temp, workspace() as ws:
            if messages is not None:
                Path(temp, bot.EXECUTION).write_text(json.dumps(messages))
            prev = ""
            if previous is not None:
                prev = str(Path(temp, "prev"))
                Path(prev).mkdir()
                Path(prev, "handoff.json").write_text(previous)
            out, _ = step(bot.handoff, {}, pr="7", head_sha="abc", mode="light", runner_temp=temp, home=temp, prev=prev)
            saved = Path(temp, "handoff")
            return (out, json.loads((saved / "handoff.json").read_text()) if saved.exists() else None,
                    (saved / "handoff.md").read_text() if saved.exists() else "",
                    (ws / "partial.md").read_text() if (ws / "partial.md").exists() else "")

    def test_a_handoff_merges_onto_the_one_it_continued_and_keeps_leads_off_the_pr(self):
        earlier = json.dumps({"runs": 1, "comments": [], "reads": ["pr-head/z.py"], "note": "", "recent_text": ""})
        token = "ghp_" + "a" * 36
        out, h, md, partial = self.handoff([
            assistant(call("c1", path="a.py", line=3, body="**[P2]** failed one")),
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": True,
                                                       "content": "line 3 is not in the diff"}]}},
            assistant({"type": "text", "text": f"**Progress:** P1 `b.py:7` unverified {token}"},
                      call("c2", path="b.py", line=7, body="**[P1]** unknown one")),
        ], earlier)
        self.assertEqual((h["pr"], h["runs"], h["reads"]), ("7", 2, ["pr-head/z.py"]))
        self.assertIn("## Inline comments that failed to post\n\n"
                      "- `a.py:3` **[P2]** failed one (line 3 is not in the diff)", md)
        self.assertIn("**Progress:** P1 `b.py:7` unverified [REDACTED]", md, "a token would reach a public artifact")
        self.assertNotIn("unverified", partial, "the public comment would list unverified leads")
        self.assertEqual(self.handoff(None, json.dumps({"runs": 1, "comments": [], "reads": [],
                                                        "note": "Progress: a.py done", "recent_text": ""}))[1]["note"],
                         "Progress: a.py done", "a continued run that said nothing would drop the handoff it had")
        fresh = self.handoff([assistant({"type": "text", "text": "Progress: nothing yet"})], "{not json")[1]
        self.assertEqual(fresh["runs"], 1, "a corrupt earlier handoff would stop this one being saved")
        self.assertEqual(self.handoff(None)[0], {"exit": "No Claude Code messages to hand off."})


class TestAfterTheReview(unittest.TestCase):
    def test_a_corrupt_or_missing_execution_file_still_records_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp, "summary.md")
            for text in ('[{"type": "assistant", "message": ', None):
                if text:
                    Path(temp, bot.EXECUTION).write_text(text)
                step(bot.usage, {}, pr="1", head_sha="abc", mode="light", outcome="failure", runner_temp=temp,
                     home=temp, github_run_id="5", github_run_attempt="1", github_step_summary=str(summary))
                record = json.loads(Path(temp, "usage/review-usage-1-5-1.json").read_text())
                self.assertEqual(record["api_calls"], 0)
                Path(temp, bot.EXECUTION).unlink(missing_ok=True)
            self.assertTrue(summary.read_text().strip(), "the job summary would have no table")

    def test_a_run_past_its_turn_cap_still_posts_and_one_out_of_credit_says_so(self):
        with tempfile.TemporaryDirectory() as temp:
            execution = Path(temp, bot.EXECUTION)
            execution.write_text(json.dumps([{"type": "result", "subtype": "success", "is_error": False,
                                              "num_turns": 41, "structured_output": {"types": ["bug"]}}]))
            out, fake = step(bot.salvage, {}, repo="o/r", pr="7", runner_temp=temp)
            self.assertEqual(out, {"detail": "41 turns", "structured_output": '{"types":["bug"]}'})
            execution.write_text(json.dumps([{"type": "assistant", "error": "billing_error", "message": {}}]))
            out, fake = step(bot.salvage, {"pr comment": ""}, repo="o/r", pr="7", runner_temp=temp)
            self.assertEqual((out, len(fake.writes())), ({"billing": "true"}, 1))

    def test_the_summary_is_redacted_posted_from_stdin_and_resolves_a_stopped_review(self):
        token = "sk-ant-" + "x" * 30
        output = json.dumps({"outside_diff": [], "since_last": f"leaked {token}", "description_problems": [],
                             "questions": [], "checked": [], "skipped": [], "types": ["bug"], "prior_findings": [],
                             "design": [], "latent": [], "risk": {"level": "low", "reason": "r", "look_at": []}})
        with workspace():
            out, fake = step(bot.post, {
                "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}},
                "pr comment": "https://github.com/o/r/pull/7#issuecomment-99",
                "GET repos/o/r/issues/7/comments": [{"id": 5, "user": {"login": "mstar-review[bot]"},
                                                     "body": bot.INCOMPLETE + "\nstopped"}],
                "PATCH repos/o/r/issues/comments/5": {},
            }, repo="o/r", pr="7", output=output, head_sha=HEAD, mode="light", salvaged="true", detail="41 turns")
        (_, posted), (_, resolved) = fake.writes()
        self.assertEqual(out, {"id": "99"})
        self.assertNotIn(token, posted)
        self.assertIn("leaked [REDACTED]", posted)
        self.assertIn("went past the workflow's turn limit (41 turns)", posted)
        self.assertTrue(json.loads(resolved)["body"].startswith(bot.RESOLVED + "\n"))
        self.assertTrue(all(posted not in arg for argv, _ in fake.calls for arg in argv),
                        "posted text would reach gh's command line")

    def test_the_summary_keeps_the_last_reviews_findings_a_re_review_wasnt_shown(self):
        item = {"priority": "P1", "claim": "leak", "location": "a.py:1", "detail": "d"}
        state = {"head": LAST, "mode": "nogpu", "skipped": ["big.py"], "latent": [], "outside_diff": [item]}
        partial = f"{render_summary.MARKER}\n<!-- claude-review-state " \
                  f"{base64.b64encode(json.dumps(state).encode()).decode()} -->"
        review = {"outside_diff": [], "since_last": "", "description_problems": [], "questions": [], "checked": [],
                  "skipped": [], "types": ["bug"], "prior_findings": [], "design": [], "latent": [],
                  "risk": {"level": "low", "reason": "r", "look_at": []}}
        with workspace():
            _, fake = step(bot.post, {
                "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}},
                "pr comment": "https://github.com/o/r/pull/7#issuecomment-99",
                "GET repos/o/r/issues/7/comments": [{"id": 5, "user": {"login": "mstar-review[bot]"},
                                                     "body": partial}],
            }, repo="o/r", pr="7", output=json.dumps(review), head_sha=HEAD, mode="light", salvaged="false")
        (_, posted), = fake.writes()
        self.assertEqual(render_summary.saved_state(posted)["outside_diff"], [item],
                         "a run with no last-review.json, or a /review continue, would drop the finding for good")

    def test_only_the_fixed_type_labels_reach_the_api(self):
        out, fake = step(bot.labels, {"GET repos/o/r/issues/7/labels": [{"name": "type/docs"}, {"name": "area/x"}],
                                      "DELETE *": {}, "GET repos/o/r/labels/*": Fail(), "POST *": {}},
                         repo="o/r", pr="7", output=json.dumps({"types": ["bug", "x; rm -rf /", "ci"]}))
        self.assertEqual([k for k, _ in fake.writes()],
                         ["DELETE repos/o/r/issues/7/labels/type%2Fdocs", "POST repos/o/r/labels",
                          "POST repos/o/r/issues/7/labels", "POST repos/o/r/labels", "POST repos/o/r/issues/7/labels"])
        self.assertFalse(any("rm -rf" in arg for argv, _ in fake.calls for arg in argv))

    def test_a_refused_continue_never_overwrites_a_stopped_reviews_comment(self):
        stopped = {"id": 5, "user": {"login": "mstar-review[bot]"}, "body": bot.INCOMPLETE + "\nThe review stopped"}
        routes = {"GET repos/o/r/issues/7/comments": [stopped], "PATCH *": {}, "pr comment": ""}
        with workspace():
            _, fake = step(bot.incomplete, routes, repo="o/r", pr="7", run_url="u", why="the PR has new commits",
                           refused="true")
            self.assertEqual([k for k, _ in fake.writes()], ["pr comment"])
            self.assertTrue(fake.writes()[0][1].startswith(bot.REFUSED))
            Path("partial.md").write_text("**Inline comments it made (1):**\n")
            _, fake = step(bot.incomplete, routes, repo="o/r", pr="7", run_url="u", detail="40 turns",
                           can_continue="true")
        (key, body), = fake.writes()
        self.assertEqual(key, "PATCH repos/o/r/issues/comments/5", "a retry would stack up a second comment")
        self.assertIn("didn't complete: a step failed (40 turns). [Log](u). Comment `/review continue`",
                      json.loads(body)["body"])
        self.assertIn("What it had so far:\n\n**Inline comments it made (1):**", json.loads(body)["body"])


if __name__ == "__main__":
    unittest.main()
