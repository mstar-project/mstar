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
                                         "risk": None}).encode()).decode()
    return {"id": 3, "user": {"login": "mstar-review[bot]"}, "created_at": "t",
            "body": f"<!-- claude-review-summary -->\nx\n<!-- claude-review-state {state} -->"}


def collect(mode="light", **routes):
    """bot.collect on PR 7 at HEAD, its earlier comments and compares from `routes`; its outputs and files."""
    with workspace() as ws:
        out, _ = step(bot.collect, {
            "GET repos/o/r/pulls/7": {"title": "T", "body": None, "head": {"sha": HEAD, "ref": "x", "repo": None},
                                      "base": {"ref": "main", "repo": {"full_name": "o/r"}}},
            "pr diff": "diff --git a/a.py b/a.py\n",
            "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}},
            "GET repos/o/r/pulls/7/comments": [],
            "GET repos/o/r/issues/7/comments": [],
            "issue list": Fail(),
            **routes}, repo="o/r", pr="7", head_sha=HEAD, mode=mode)
        return out, {p.name: p.read_text() for p in ws.iterdir()}


class TestCollect(unittest.TestCase):
    def test_a_re_review_builds_only_on_an_ancestor(self):
        since = f"GET repos/o/r/compare/{LAST}...{HEAD}"
        conversation = {"GET repos/o/r/issues/7/comments": [summary(LAST)]}

        def compare(status):
            return lambda argv, stdin: "diff --git a/b.py b/b.py\n" if "Accept: application/vnd.github.diff" in argv \
                else json.dumps([{"status": status, "commits": []}] if "--slurp" in argv else {"status": status})

        out, files = collect(**conversation, **{since: compare("ahead")})
        self.assertEqual((out.get("rereview"), files["since.diff"]), ("true", "diff --git a/b.py b/b.py\n"))
        self.assertEqual(json.loads(files["last-review.json"])["head"], LAST)
        out, files = collect(**conversation, **{since: compare("diverged")})
        self.assertNotIn("rereview", out, "a re-review could rest on commits no longer in the PR")
        self.assertNotIn("last-review.json", files)


if __name__ == "__main__":
    unittest.main()
