"""Check .github/review/bot.py, the review workflow's steps, against a stand-in for `gh`."""

import base64
import contextlib
import fnmatch
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


class Fail:
    """A call that fails, as GitHub's errors do, with exit status `code`."""

    def __init__(self, stderr="HTTP 502", code=1):
        self.stderr, self.code = stderr, code


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
            return subprocess.CompletedProcess(argv, answer.code, "", answer.stderr)
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
        self.assertEqual(self.preflight("full", write), ({"run": "true", "mode": "full", "turns": "2000"}, []))
        out, replies = self.preflight("full", {"a-maintainer": "write", "an-author": "read"}, actor="a-maintainer")
        self.assertEqual(out, {"run": "false"}, "a live test could run the code of an author without write access")
        self.assertIn("/review nogpu", replies[0], "the maintainer wouldn't be told what they can run")
        out, replies = self.preflight("full", read)
        self.assertEqual(out, {"run": "false"}, "a contributor without write access could start a live test")
        self.assertIn("/review light", replies[0], "the contributor wouldn't be told what they can run")


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


def compare(status, diff="diff --git a/b.py b/b.py\n"):
    """A compare of two commits, as JSON or, asked for the diff media type, as `diff`."""
    return lambda argv, stdin: diff if "Accept: application/vnd.github.diff" in argv \
        else json.dumps([{"status": status, "commits": []}] if "--slurp" in argv else {"status": status})


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

    def test_after_a_rebase_a_re_review_reads_only_the_prs_hunks_that_changed(self):
        reviewed = pr_diff("x = 1", "# unreachable")
        routes = {**self.CONVERSATION, f"GET repos/o/r/compare/{LAST}...{HEAD}": compare("diverged"),
                  f"GET repos/o/r/compare/{BASE}...{LAST}": compare("ahead", reviewed)}
        out, files = collect(diff=pr_diff("x = 10", "# unreachable", "3333333..4444444").replace("@@ -10,3 +10,3",
                                                                                                 "@@ -12,3 +12,3"),
                             **routes)
        self.assertEqual((out.get("rereview"), files["interdiff.diff"]), ("true", ""),
                         "a rebase that moved the PR's changes would read as new changes")
        out, files = collect(diff=pr_diff("x = 1", "# dead code"), **routes)
        self.assertEqual(files["interdiff.diff"], "diff --git a/a.py b/a.py\nindex 1111111..2222222 100644\n"
                         "--- a/a.py\n+++ b/a.py\n@@ -40,2 +40,3 @@ def g():\n     return 1\n+    # dead code\n")
        self.assertNotIn("since.diff", files)


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
