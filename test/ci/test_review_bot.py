"""Check .github/review/bot.py, the review workflow's steps, against a stand-in for `gh`."""

import base64
import contextlib
import datetime
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
    """A call that fails, as GitHub's errors do, with exit status `code`."""

    def __init__(self, stderr="HTTP 502", code=1, stdout=""):
        self.stderr, self.code, self.stdout = stderr, code, stdout


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
                if not self.key(argv).startswith(("GET ", "pr list", "run download"))
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
            return subprocess.CompletedProcess(argv, answer.code, answer.stdout, answer.stderr)
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
    def preflight(self, command, perms, actor="an-author", author="an-author", assoc="CONTRIBUTOR", head="o/r"):
        routes = {f"GET repos/o/r/collaborators/{login}/permission": {"permission": perm}
                  for login, perm in perms.items()}
        routes["pr comment"] = "https://github.com/o/r/pull/1#issuecomment-9"
        routes["GET repos/o/r/pulls/1"] = {"head": {"repo": head and {
            "full_name": head, "owner": {"login": head.split("/")[0]}}}}
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
        peer = {"a-maintainer": "write", "an-author": "read"}
        out, replies = self.preflight("nogpu", peer, actor="a-maintainer")
        self.assertEqual(out, {"run": "false"}, "a nogpu review could run on the PR of an author without write access")
        self.assertIn("/review light", replies[0], "the maintainer wouldn't be told what they can run")
        self.assertEqual(self.preflight("light", peer, actor="a-maintainer"),
                         ({"run": "true", "mode": "light", "turns": "2000"}, []),
                         "a maintainer's light review of a first-timer's PR would be refused")
        self.assertEqual(self.preflight("light", {"a-maintainer": "read"}, actor="a-maintainer"),
                         ({"run": "false"}, []), "someone without write access could review another author's PR")
        self.assertEqual(self.preflight("continue", read)[0]["mode"], "",
                         "a /review continue would not keep its own mode")
        self.assertEqual(self.preflight("full", write), ({"run": "true", "mode": "full", "turns": "2000"}, []))
        out, replies = self.preflight("full", {"a-maintainer": "write", "an-author": "read"}, actor="a-maintainer")
        self.assertEqual(out, {"run": "false"}, "a live test could run the code of an author without write access")
        self.assertIn("/review light", replies[0], "the maintainer would be pointed at a review they can't run")
        out, replies = self.preflight("full", {"an-author": "write", "a-stranger": "read"}, head="a-stranger/r")
        self.assertEqual(out, {"run": "false"}, "a live test could run the code of a fork its owner holds")
        self.assertIn("/review nogpu", replies[0], "the maintainer wouldn't be told what they can run")
        self.assertEqual(self.preflight("full", write, head=None)[0], {"run": "false"},
                         "a head repository GitHub no longer shows would count as trusted")
        self.assertEqual(self.preflight("full", {"an-author": "write", "a-peer": "write"}, head="a-peer/r"),
                         ({"run": "true", "mode": "full", "turns": "2000"}, []))
        out, replies = self.preflight("full", read)
        self.assertEqual(out, {"run": "false"}, "a contributor without write access could start a live test")
        self.assertIn("/review light", replies[0], "the contributor wouldn't be told what they can run")
        self.assertEqual(self.preflight("continue", peer, actor="a-maintainer"),
                         ({"run": "true", "mode": "", "turns": "2000"}, []),
                         "a maintainer couldn't continue a light review of a first-timer's PR")


HEAD, LAST = "1" * 40, "2" * 40


NOW = datetime.datetime(2026, 11, 8, 12, tzinfo=datetime.timezone.utc)
STATUS = {"number": 2, "user": {"login": "github-actions[bot]"}, "body": bot.STATUS_ISSUE + "\nthe status"}
BOT_AUTHOR = {"__typename": "Bot", "login": "mstar-review"}  # as GraphQL names it


class TestMonitor(unittest.TestCase):
    def test_a_revert_is_found_by_its_message_after_the_merge_only(self):
        commits = [{"sha": "c" * 40, "subject": 'Revert "Faster decode (#7)"', "body": ""},
                   {"sha": "e" * 40, "subject": "Unrelated", "body": f"This reverts commit {'a' * 40}."},
                   {"sha": "d" * 40, "subject": "M", "body": ""},
                   {"sha": "f" * 40, "subject": "Older", "body": f"This reverts commit {'a' * 40}."}]
        hits = bot.reverted(commits, {"d" * 40, "a" * 40}, 7, "d" * 40)
        self.assertEqual([c["sha"] for c in hits], ["c" * 40, "e" * 40],
                         "a revert before the merge, or one of another PR's commit, would be blamed on this PR")

    def test_timeline_cross_references_are_signals_only_when_they_report_or_fix_a_bug(self):
        def ref(number, labels=(), pr=False, body="", merged=None, login="someone"):
            issue = {"number": number, "title": "t", "labels": [{"name": n} for n in labels], "body": body,
                     "user": {"login": login}}
            return {"event": "cross-referenced",
                    "source": {"issue": {**issue, **({"pull_request": {"merged_at": merged}} if pr else {})}}}

        timeline = [ref(10, ["bug"]), ref(11, ["enhancement"]), ref(12, pr=True, body="Fixes #7", merged="t"),
                    ref(13, ["type/bug"], pr=True), ref(14, pr=True, body="Fixes #70"),
                    ref(15, pr=True, body="Fixes #7", login="a-maintainer"),
                    ref(16, pr=True, body="Fixes #7", login="a-stranger"),
                    ref(17, body="`#7`", login="github-actions[bot]"), {"event": "labeled"}]
        with mock.patch.object(bot, "run", FakeGh({
                "GET repos/o/r/collaborators/a-maintainer/permission": {"permission": "write"},
                "GET repos/o/r/collaborators/a-stranger/permission": {"permission": "read"}})):
            signals, mentions = bot.referenced(timeline, 7, "o/r")
        self.assertEqual({int(s["key"].split(":")[-1]) for s in signals}, {10, 12, 13, 15},
                         "anyone could write `Fixes #7` on an open PR and make it a signal")
        self.assertEqual(sorted(mentions), ["#11", "#14", "#16"], "an unrelated mention or the bot's own would count")

    def test_only_a_bug_fix_subject_counts_as_a_fix(self):
        for subject in ("Fix a crash (#9)", "[Worker][Critical fix] Drop late messages", "Revert \"x\"", "bugfix: y",
                        "Fixing a race", "Bugs in decode", "Bugfixes for x", "Reverting x", "Fix format string crash",
                        "fix docs build", "Fix typography crash"):
            self.assertTrue(bot.FIX_SUBJECT.search(subject), subject)
        for subject in ("Fix a typo", "fix typo in README", "fix: docs", "fix lint (#12)", "fix docs.",
                        "fix lint. (#12)", "fix the comments", "prefix cache"):
            self.assertFalse(bot.FIX_SUBJECT.search(subject), subject)

    def test_a_fix_that_changes_a_prs_lines_is_blamed_on_it(self):
        with workspace() as ws:
            def git(*a):
                return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                                      capture_output=True, text=True).stdout.strip()

            git("init", "-q")
            (ws / "a.py").write_text("one\ntwo\nthree\n")
            git("add", "a.py")
            git("commit", "-qm", "base")
            (ws / "a.py").write_text("one\nTWO\nthree\n")
            git("commit", "-qam", "the PR (#7)")
            pr_sha = git("rev-parse", "HEAD")
            (ws / "a.py").write_text("one\nTWO fixed\nthree\n")
            git("commit", "-qam", "Fix a crash (#9)")
            fix = git("rev-parse", "HEAD")
            hits = bot.szz([{"sha": fix, "subject": "Fix a crash (#9)", "body": ""}], {pr_sha}, 7, {"a.py"})
            self.assertEqual([s["text"] for s in hits], ["`#9` changed lines `#7` wrote at `a.py:2`"],
                             "a fix of the PR's lines would go unseen")
            self.assertEqual(bot.szz([{"sha": fix, "subject": "Fix a crash (#9)", "body": ""}], {"e" * 40}, 7,
                                     {"a.py"}), [], "a fix of lines another PR wrote would be blamed on this one")

    def test_a_ci_failure_counts_only_against_a_parent_that_passed(self):
        def checks(argv, stdin):
            sha = next(a for a in argv if "check-runs" in a).split("/")[-2]
            return {"check_runs": [{"name": "cpu-tests", "conclusion": "failure" if sha == "m" else parent}]}

        for parent, regressed in (("success", True), ("failure", False), ("cancelled", False)):
            with mock.patch.object(bot, "run", FakeGh({"GET repos/o/r/commits/*/check-runs*": checks})):
                found = bot.ci_regressions("o/r", 7, "m", "p")
            self.assertEqual(bool(found), regressed, f"a parent that {parent}")

    @staticmethod
    def merged_prs(*reviews):
        """GitHub's answer to the monitor's query, filtering by the query's states as GitHub does: PR 7 with
        `reviews` and PR 8, the bot's approved too but merged before the window."""
        def answer(argv, stdin):
            query = next(a for a in argv if a.startswith("query="))
            nodes = [{"author": r["author"]} for r in reviews
                     if "states: APPROVED" not in query or r["state"] == "APPROVED"]
            return {"data": {"repository": {"pullRequests": {"nodes": [
                {"number": 7, "mergedAt": "2026-11-01T00:00:00Z", "reviews": {"nodes": nodes}},
                {"number": 8, "mergedAt": "2026-01-01T00:00:00Z",
                 "reviews": {"nodes": [{"author": BOT_AUTHOR}]}}]}}}}
        return answer

    def run_monitor(self, body, bug_title="crash in decode", app_id="app-id", **routes_over):
        calls = {"body": None}

        def issue_status(argv, stdin):
            return {"number": 2, "body": calls["body"] or body, "user": {"login": "github-actions[bot]"}}

        def patch(argv, stdin):
            calls["body"] = json.loads(stdin)["body"]
            return {}

        routes = {
            "GET repos/o/r/issues[?]state=all&creator=*": [STATUS],
            "POST graphql": self.merged_prs({"state": "APPROVED", "author": BOT_AUTHOR}),
            "GET repos/o/r/issues/2": issue_status, "PATCH repos/o/r/issues/2": patch, "issue comment": "",
            "GET repos/o/r/pulls/7": {"number": 7, "title": "Faster decode @someone", "state": "closed",
                                      "merged_at": "t", "merge_commit_sha": "d" * 40},
            "GET repos/o/r/pulls/7/commits*": [{"sha": "a" * 40}],
            "GET repos/o/r/issues/7/timeline*": [],
            "GET repos/o/r/commits/*/check-runs*": {"check_runs": []},
            "GET repos/o/r/issues[?]state=all&labels=bug*": [{"number": 30, "title": bug_title,
                                                               "created_at": "2026-11-08T00:00:00Z"}],
            "GET repos/o/r/issues[?]state=closed&labels=type*": [],
            "GET repos/o/r/pulls[?]*": [],
            "git log": (f'{"c" * 40}\x1f{"d" * 40}\x1fRevert "Faster decode (#7)"\x1f\x1e\n'
                        f'{"d" * 40}\x1f{"b" * 40}\x1fFaster decode (#7)\x1f\x1e\n'),
            "git diff": "a.py\n", "git show": "", "git blame": "", "GET repos/o/r/labels/*": {}, **routes_over,
        }
        with mock.patch.object(bot, "utcnow", lambda: NOW), \
                mock.patch.dict(os.environ, {"REVIEW_APP_CLIENT_ID": app_id}):
            out, fake = step(bot.monitor, routes, repo="o/r", approvals="on")
        return calls["body"], fake

    def test_new_signals_reach_subscribers_once_and_the_body_is_safe(self):
        body, fake = self.run_monitor(STATUS["body"], bug_title="crash @someone <!-- x in #99")
        self.assertTrue(body.startswith(bot.STATUS_ISSUE), "the next run couldn't find the status issue")
        self.assertIn("Approvals: **on**", body)
        comments = [s for k, s in fake.writes() if k == "issue comment"]
        self.assertEqual(len(comments), 1, "a new signal would not notify subscribers")
        self.assertIn("reverted", comments[0])
        self.assertNotIn("@someone", body + comments[0], "a PR's title could ping someone from the status issue")
        self.assertNotIn("<!-- x", body, "a title could open an HTML comment")
        self.assertNotIn("#99", body, "a title could mark another issue's timeline")
        _, fake = self.run_monitor(body)
        self.assertEqual([k for k, _ in fake.writes() if k == "issue comment"], [],
                         "a signal already reported would notify subscribers every day")

    def test_approvals_read_off_when_no_app_can_approve(self):
        body, _ = self.run_monitor(STATUS["body"], app_id="")
        self.assertIn("Approvals: **off**", body, "the status issue would claim approvals are on with no app")

    def test_the_first_run_creates_the_status_issue_and_the_label(self):
        created = []
        body, fake = self.run_monitor(STATUS["body"], **{
            "GET repos/o/r/issues[?]state=all&creator=*": [],
            "POST repos/o/r/issues": lambda a, i: created.append(json.loads(i)) or {"number": 2},
            "GET repos/o/r/labels/*": Fail(), "POST repos/o/r/labels": {}})
        self.assertTrue(created[0]["body"].startswith(bot.STATUS_ISSUE), "the next run couldn't find it")
        self.assertIn("Faster decode", body)
        self.assertIn("POST repos/o/r/labels", [k for k, _ in fake.writes()], "skip-ai-review would be a setup step")

    def test_a_failed_post_leaves_the_state_unwritten_so_the_signal_goes_out_tomorrow(self):
        patched = []
        with self.assertRaises(RuntimeError):
            self.run_monitor(STATUS["body"], **{"issue comment": Fail("HTTP 502"),
                                                "PATCH repos/o/r/issues/2": lambda a, i: patched.append(i) or {}})
        self.assertEqual(patched, [], "the day's signals would count as reported though the post failed")

    def test_a_pr_that_cant_be_read_is_reported_and_the_rest_goes_on(self):
        reported, _ = self.run_monitor(STATUS["body"])
        body, _ = self.run_monitor(reported, **{"GET repos/o/r/pulls/7": Fail("HTTP 502")})
        self.assertIn("couldn't read", body)
        self.assertIn("`#7`", body)
        _, fake = self.run_monitor(body)
        self.assertEqual([k for k, _ in fake.writes() if k == "issue comment"], [],
                         "a PR read again after a failed read would re-post its old signals")

    def test_only_the_bots_own_live_approval_counts(self):
        for name, review in (("dismissed", {"state": "DISMISSED", "author": BOT_AUTHOR}),
                             ("a person with the bot's login", {"state": "APPROVED", "author": {
                                 "__typename": "User", "login": "mstar-review"}}),
                             ("another bot", {"state": "APPROVED", "author": {"__typename": "Bot", "login": "ci"}}),
                             ("a deleted account", {"state": "APPROVED", "author": None})):
            with self.subTest(name):
                body, fake = self.run_monitor(STATUS["body"], **{"POST graphql": self.merged_prs(review)})
                self.assertFalse(any(a.startswith("pr=") for argv, _ in fake.calls for a in argv),
                                 "the merged-PR query has no pr variable")
                self.assertNotIn("Faster decode", body, f"{name}: the PR would count as bot-approved")
                self.assertEqual([k for k, _ in fake.writes() if k == "issue comment"], [])

    def test_the_status_issue_is_the_one_a_bot_made_with_the_marker(self):
        routes = {"GET repos/o/r/issues[?]state=all&creator=*": [
            {"number": 3, "user": {"login": "a-person"}, "body": STATUS["body"]},
            {"number": 4, "user": {"login": "github-actions[bot]"}, "body": "<!-- other -->"}, STATUS]}
        with mock.patch.object(bot, "run", FakeGh(routes)):
            self.assertEqual(bot.status_issue("o/r")["number"], 2, "a forged issue would become the status issue")


class TestRestore(unittest.TestCase):
    def restore(self, meta, non_write="", path=".github/workflows/claude-review.yml", author_permission="write",
                through_resolve=False):
        """/review continue finding one stopped review whose saved handoff has `meta`."""
        def download(argv, stdin):
            saved = Path(argv[argv.index("-D") + 1])
            saved.mkdir(parents=True)
            (saved / "handoff.json").write_text(json.dumps({"pr": "7", "head_sha": HEAD, **meta}))
            (saved / "handoff.md").write_text("notes")
            return ""

        routes = {
            "GET repos/o/r/issues/7/comments": [{"id": 5, "user": {"login": "mstar-review[bot]"},
                                                 "body": bot.INCOMPLETE + "\nThe review didn't complete"}],
            "GET repos/o/r/actions/artifacts?*": {"artifacts": [
                {"expired": False, "created_at": "2026-11-01", "workflow_run": {"id": 9}}]},
            "GET repos/o/r/actions/runs/9": {"path": path, "event": "issue_comment"},
            "run download": download,
            "GET repos/o/r/collaborators/an-author/permission": {"permission": author_permission},
        }
        with tempfile.TemporaryDirectory() as temp:
            if through_resolve:
                routes["GET repos/o/r/pulls/7"] = {"head": {"sha": HEAD, "repo": {"full_name": "o/r"}}, "title": "t",
                                                   "user": {"login": "an-author"}}
                return step(bot.resolve, routes, repo="o/r", pr="7", mode="", runner_temp=temp, non_write=non_write)[0]
            return step(bot.restore, routes, repo="o/r", pr="7", head_sha=HEAD, runner_temp=temp, non_write=non_write,
                        author="an-author")[0]

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
        self.assertIn("write access", self.restore({"mode": "nogpu"}, author_permission="read")["why"],
                      "a nogpu review could continue on the PR of an author without write access")
        self.assertEqual(self.restore({"mode": "nogpu"})["mode"], "nogpu", "a writer's nogpu review couldn't continue")
        self.assertIn("write access",
                      self.restore({"mode": "nogpu"}, author_permission="read", through_resolve=True)["why"],
                      "resolve wouldn't hold a nogpu continue to the PR's author")
        self.assertEqual(self.restore({"mode": "light"}, author_permission="read")["mode"], "light",
                         "continuing a light review would need more than a light review does")


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
            f"GET repos/o/r/compare/{BASE}...{HEAD}": diff,
            "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}},
            "GET repos/o/r/pulls/7/comments": [],
            "GET repos/o/r/issues/7/comments": [],
            "GET repos/o/r/issues?state=open*": Fail(),
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


def configure_review_app(test):
    """Reimport the bot as it runs with the review app configured, restoring it after `test`."""
    test.addCleanup(importlib.reload, bot)
    test.addCleanup(importlib.reload, render_summary)
    with mock.patch.dict(os.environ, {"REVIEW_APP_CLIENT_ID": "app-id"}):
        importlib.reload(render_summary)
        importlib.reload(bot)


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
        configure_review_app(self)
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


class TestFullModeInputs(unittest.TestCase):
    def test_only_the_teams_and_bots_text_reaches_the_review_that_steers_the_box(self):
        def comment(user, assoc):
            return {"id": 1, "user": {"login": user}, "author_association": assoc, "body": user}

        routes = {"GET repos/o/r/pulls/7/comments": [comment("a-stranger", "NONE"), comment("a-member", "MEMBER")],
                  "GET repos/o/r/issues/7/comments": [comment("github-actions[bot]", "NONE"),
                                                      comment("a-contributor", "CONTRIBUTOR")],
                  "GET repos/o/r/issues?state=open*": [
                      {"number": 1, "title": "stranger's", "author_association": "NONE"},
                      {"number": 2, "title": "team's", "author_association": "OWNER"},
                      {"number": 3, "title": "a PR", "author_association": "OWNER", "pull_request": {}}]}
        full, light = collect("full", **routes)[1], collect("light", **routes)[1]
        self.assertEqual([c["user"] for c in json.loads(full["pr-comments.json"])], ["a-member"])
        self.assertEqual([c["user"] for c in json.loads(full["pr-conversation.json"])], ["github-actions[bot]"],
                         "the bot's own comments, which a re-review judges, must stay")
        self.assertEqual(json.loads(full["open-issues.json"]), [{"number": 2, "title": "team's"}])
        self.assertEqual([c["user"] for c in json.loads(light["pr-conversation.json"])],
                         ["github-actions[bot]", "a-contributor"], "other modes read everything")
        self.assertEqual([i["number"] for i in json.loads(light["open-issues.json"])], [1, 2])


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
        typed = ("diff --git a/w b/w\ndeleted file mode 120000\nindex 1..0\n--- a/w\n+++ /dev/null\n@@ -1 +0,0 @@\n-t\n"
                 "diff --git a/w b/w\nnew file mode 100644\nindex 0..{}\nBinary files /dev/null and b/w differ\n")
        self.assertIn("0..3", bot.interdiff(typed.format("2"), typed.format("3")),
                      "a binary swapped inside a type change would read as unchanged")
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

    def test_what_a_stopped_review_posts_is_inert(self):
        partial = self.handoff([
            assistant(call("c1", path="a.py", line=3, body="**[P2]** @org/team ![x](https://x.example/a.png)")),
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1",
                                                       "content": '{"comment_id": 1}'}]}},
        ])[3]
        self.assertIn("`a.py:3` **[P2]**", partial)
        for markup in ("@org", "![", "]("):
            self.assertNotIn(markup, partial, "the model's text would ping a team or load an image from the comment")


class TestAfterTheReview(unittest.TestCase):
    def test_a_corrupt_or_missing_execution_file_still_records_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp, "summary.md")
            for text in ('[{"type": "assistant", "message": ', None):
                if text:
                    Path(temp, bot.EXECUTION).write_text(text)
                out, _ = step(bot.usage, {}, pr="1", head_sha="abc", mode="light", outcome="failure", runner_temp=temp,
                              home=temp, github_step_summary=str(summary))
                self.assertNotIn("exit", out)
                Path(temp, bot.EXECUTION).unlink(missing_ok=True)
            self.assertIn("| API calls | 0 |", summary.read_text(), "the job summary would have no table")

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

    def test_the_threads_query_asks_when_a_finding_was_edited(self):
        self.assertIn("lastEditedAt", bot.THREADS, "an edited finding would read as the bot's own text")

    def test_a_gpu_ask_above_the_cap_reaches_the_saved_review(self):
        output = json.dumps({"outside_diff": [], "since_last": "", "description_problems": [], "questions": [],
                             "checked": [], "skipped": [], "types": ["bug"], "prior_findings": [], "design": [],
                             "latent": [], "risk": {"level": "low", "reason": "r", "look_at": []},
                             "live_test": {"run": "full", "reason": "r", "gpus": 8, "untestable": ""}})
        with workspace():
            out, fake = step(bot.post, {
                "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [
                    {"comments": {"nodes": [{"databaseId": 4, "author": {"login": "mstar-review"},
                                             "body": "**[P1]** x"}]}},
                ]}}}}},
                "pr comment": "https://github.com/o/r/pull/7#issuecomment-99",
                "GET repos/o/r/issues/7/comments": [],
            }, repo="o/r", pr="7", output=output, head_sha=HEAD, mode="full", salvaged="false", max_gpus="4")
        self.assertEqual(json.loads(out["result"])["live_test"]["untestable"], "needs 8 GPUs, runs get 4")
        self.assertEqual(json.loads(out["result"])["inline"], {"4": "P1"},
                         "a later step couldn't tell an edited finding")
        self.assertTrue(any("pr=7" in argv for argv, _ in fake.calls), "the threads query needs the PR's number")

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
        self.assertEqual(out["id"], "99")
        self.assertNotIn(token, out["result"], "the raw summary would reach the live-test job unredacted")
        self.assertNotIn(token, posted)
        self.assertIn("leaked [REDACTED]", posted)
        self.assertIn("went past the workflow's turn limit (41 turns)", posted)
        self.assertTrue(json.loads(resolved)["body"].startswith(bot.RESOLVED + "\n"))
        self.assertTrue(all(posted not in arg for argv, _ in fake.calls for arg in argv),
                        "posted text would reach gh's command line")

    def test_the_summary_keeps_the_last_reviews_findings_a_re_review_wasnt_shown(self):
        item = {"priority": "P1", "claim": "leak", "location": "a.py:1", "detail": "d"}
        question = {"location": "a.py:1", "text": "Should this be a resource?"}
        state = {"head": LAST, "mode": "nogpu", "skipped": ["big.py"], "latent": [], "outside_diff": [item],
                 "questions": [question], "risk": {"level": "medium", "reason": "r", "look_at": []}}
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
                                                     "created_at": "t", "body": partial}],
                "GET repos/o/r/pulls/7": {"user": {"login": "an-author"}},
                "GET repos/o/r/pulls/7/comments": [],
                "GET repos/o/r/pulls/7/reviews*": [],
            }, repo="o/r", pr="7", output=json.dumps(review), head_sha=HEAD, mode="light", salvaged="false")
        (_, posted), = fake.writes()
        saved = render_summary.saved_state(posted)
        self.assertEqual(saved["outside_diff"], [item],
                         "a run with no last-review.json, or a /review continue, would drop the finding for good")
        self.assertEqual((saved["questions"], saved["risk"]["level"]), ([question], "medium"),
                         "post would approve past a question and a risk nobody answered")

    def test_only_write_access_other_than_the_author_answers_an_earlier_review(self):
        for name, login, perm, body, answered in (("write access", "m", "write", "keep it", True),
                                                  ("read access, a COLLABORATOR too", "m", "read", "keep it", False),
                                                  ("the author", "an-author", "admin", "it's fine", False),
                                                  ("a /review command", "m", "write", "/review nogpu", False)):
            comment = {"user": {"login": login, "type": "User"}, "body": body, "created_at": "u"}
            with self.subTest(name), workspace(), mock.patch.object(bot, "run", FakeGh({
                    "GET repos/o/r/issues/7/comments": [summary(LAST), comment],
                    "GET repos/o/r/pulls/7": {"user": {"login": "an-author"}},
                    "GET repos/o/r/pulls/7/comments": [],
                    "GET repos/o/r/pulls/7/reviews*": [],
                    "GET repos/o/r/collaborators/*/permission": {"permission": perm}})):
                self.assertEqual(bot.earlier_review("o/r", "7")[1], answered)

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


class TestImprove(unittest.TestCase):
    def inputs(self, routes):
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            return step(bot.improve_inputs, routes, repo="o/r", key_set="true")

    def test_evidence_keeps_what_a_reader_needs_without_names(self):
        state = base64.b64encode(json.dumps({"moot": [{"ref": "11", "why": "gone"}]}).encode()).decode()
        at, old = "2026-11-07T00:00:00Z", "2026-10-01T00:00:00Z"
        bot_user = {"login": "mstar-review[bot]", "type": "Bot"}
        person = {"login": "a-person", "type": "User"}
        routes = {
            "GET repos/o/r/issues[?]state=all&creator=*": [STATUS],
            "GET repos/o/r/issues/comments*": [
                {"issue_url": "https://api.github.com/repos/o/r/issues/5", "user": bot_user,
                 "body": "<!-- claude-review-summary -->\nx"},
                {"issue_url": "https://api.github.com/repos/o/r/issues/6", "user": person,
                 "body": "<!-- claude-review-summary -->\nforged"},
                {"issue_url": "https://api.github.com/repos/o/r/issues/8", "user": bot_user,
                 "body": "an ordinary comment of the bot's"}],
            "GET repos/o/r/issues/2/comments*": [],
            "GET repos/o/r/pulls/5": {"user": {"login": "a-person"}},
            "GET repos/o/r/pulls/5/reviews*": [],
            "POST graphql": {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [
                {"isResolved": True, "isOutdated": False, "comments": {"nodes": [{"databaseId": 11}]}}]}}}}},
            "GET repos/o/r/pulls/5/comments*": [
                {"id": 11, "user": bot_user, "created_at": old, "html_url": "f", "path": "a.py", "line": 1,
                 "body": "[P2] x"},
                {"id": 12, "user": person, "created_at": at, "in_reply_to_id": 11, "html_url": "r",
                 "body": "not a bug, ask @a-person, see https://evil.example/x and #7"},
                {"id": 13, "user": {"login": "ci[bot]", "type": "Bot"}, "created_at": at, "body": "lint"}],
            "GET repos/o/r/issues/5/comments*": [
                {"id": 3, "user": bot_user, "created_at": at, "html_url": "s",
                 "body": f"<!-- claude-review-summary -->\nx\n<!-- claude-review-state {state} -->"}],
        }
        with workspace() as d:
            out, _ = self.inputs(routes)
            text = (d / "improve-evidence/5.json").read_text()
        self.assertEqual(out["run"], "true")
        self.assertEqual(json.loads(out["schema"]), bot.IMPROVE_SCHEMA)
        e = json.loads(text)
        self.assertNotIn("a-person", text, "a login reached the model")
        self.assertNotIn("evil.example", text)
        self.assertEqual([(c["id"], c["role"]) for c in e["inline"]], [(11, "bot"), (12, "author")],
                         "the old resolved finding, the reply, and not another bot's comment")
        self.assertTrue(e["inline"][0]["resolved"] and not e["inline"][0]["outdated"])
        self.assertEqual([m["why"] for m in e["moot"]], ["gone"])

    def post(self, output):
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            return step(bot.improve_post, {"POST repos/o/r/issues": "{}"}, repo="o/r", output=output)

    def test_an_empty_list_opens_no_issue(self):
        out, fake = self.post('{"improvements": []}')
        self.assertNotIn("exit", out)
        self.assertEqual(fake.writes(), [], "an empty week opened an issue")

    def test_a_post_is_redacted_on_stdin_with_links_kept_in_the_repository(self):
        secret = "sk-ant-" + "a1b2c3d4" * 3
        change = ("ask @a-person about #7, see https://evil.example/x.png HTTPS://evil.example/x www.evil.example/a "
                  '![x](HTTP://evil.example/x.png) <img src="//cdn.example/x.png"> [y](//cdn.example) GH-123')
        item = {"title": f"leak {secret}", "change": change, "where": "bot.py",
                "evidence": ["https://github.com/o/r/pull/5#discussion_r1", "https://evil.example/y",
                             "https://github.com/o/r/../../evil/phish", "https://github.com/o/r/%2e%2e/%2e%2e/evil/phish"]}
        out, fake = self.post(json.dumps({"improvements": [item]}))
        self.assertNotIn("exit", out)
        [(key, stdin)] = fake.writes()
        self.assertEqual(key, "POST repos/o/r/issues")
        issue = json.loads(stdin)
        self.assertEqual(issue["title"], "Review bot: what to improve, week of 2026-11-02")
        self.assertNotIn(secret, stdin)
        self.assertIn("[REDACTED]", issue["body"])
        self.assertFalse(any(secret in " ".join(argv) for argv, _ in fake.calls), "text reached the command line")
        self.assertIn("https://redirect.github.com/o/r/pull/5#discussion_r1", issue["body"])
        for text in ("evil.example", "a-person", "<img", "](//", "GH-123", "phish"):
            self.assertNotIn(text, issue["body"])

    def test_safe_and_scrub_leave_no_link_image_mention_or_reference_in_any_spelling(self):
        for text in ("![x](https&colon;&sol;&sol;evil.example/x.png)", "![w](https:\\/\\/evil.example/x.png)",
                     "[w](/\\/evil.example)", "[z](http:evil.example/x)", "&commat;someone", "&num;99", "@someone",
                     "#99", "<!-- x", "<img src=x>", "GH-5", "gh-5", "_GH-123_", "https://github.com/o/r/issues/5",
                     "![x](https://evil.example/x.png)", "www.evil.example", "_www.evil.example_",
                     "[a]: //evil.example/x"):
            for fn in (bot.safe, bot.scrub):
                with self.subTest(text, fn=fn.__name__):
                    out = fn(text)
                    self.assertNotRegex(out, r"(?i)\][(:]|!\[|://|gh-|www\.", "it would render or mark a timeline")
                    self.assertNotRegex(out, "[@#<&\\\\](?!\u200b)", "it would ping, open markup or spell another")
                    self.assertEqual(bot.safe(bot.safe(text)), bot.safe(text), "a second pass would change it")

    def test_safe_and_scrub_keep_ordinary_text_readable(self):
        for text in ("a//b", "use n//2 here", "path/to//file.py", "//TODO", "`a//b`", "`List<int>`", "`a && b`",
                     "`&self`", "`\\n`"):
            for fn in (bot.safe, bot.scrub):
                with self.subTest(text, fn=fn.__name__):
                    self.assertEqual(fn(text).replace("\u200b", ""), text, "it would read as a link or escape")
                    self.assertEqual(bot.safe(bot.safe(text)), bot.safe(text), "a second pass would change it")


if __name__ == "__main__":
    unittest.main()
