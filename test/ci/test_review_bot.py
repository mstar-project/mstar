"""Check .github/review/bot.py, the review workflow's steps, against a stand-in for `gh`."""

import base64
import contextlib
import datetime
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
        routes["GET repos/o/r/issues[?]*"] = []  # no ledger yet: nothing spent
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


NOW = datetime.datetime(2026, 11, 8, 12, tzinfo=datetime.timezone.utc)
LEDGER = {"number": 1, "user": {"login": "github-actions[bot]"}, "body": bot.issue_marker("ledger") + "\nthe ledger"}
STATUS = {"number": 2, "user": {"login": "github-actions[bot]"}, "body": bot.issue_marker("status") + "\nthe status"}


def spent(usd, created="2026-11-08T01:00:00Z", run="1", attempt="1", login="github-actions[bot]", **extra):
    """A ledger comment recording a run that cost `usd`."""
    record = base64.b64encode(json.dumps({"usd": usd, "run": run, "attempt": attempt, **extra}).encode()).decode()
    return {"user": {"login": login}, "created_at": created,
            "body": f"PR #1: ${usd}.\n<!-- review-spend {record} -->\n"}


class TestSpend(unittest.TestCase):
    def spend(self, comments, issues=(LEDGER,)):
        routes = {"GET repos/o/r/issues[?]*": list(issues),
                  "GET repos/o/r/issues/1/comments*": comments, "GET repos/o/r/issues/2/comments*": comments}
        with mock.patch.object(bot, "run", FakeGh(routes)):
            return bot.spend("o/r", NOW)

    def test_spend_sums_this_day_and_month_once_per_run_from_the_bots_comments_alone(self):
        self.assertEqual(self.spend([spent(1.5), spent(2, "2026-11-01T23:59:59Z", run="2"),
                                     spent(99, "2026-10-31T23:59:59Z", run="3")]), {"day": 1.5, "month": 3.5})
        self.assertEqual(self.spend([spent(1), spent(1)]), {"day": 1, "month": 1},
                         "a re-recorded run would count twice")
        self.assertEqual(self.spend([spent(1, attempt="2"), spent(1, attempt="1")])["day"], 2,
                         "a retried run's spend would be dropped")
        self.assertEqual(self.spend([spent(1), spent(100, run="2", login="mallory")])["day"], 1,
                         "anyone could add a record to a public issue and reach the cap")

    def test_spend_reads_every_ledger_issue_and_starts_at_the_lowest_for_writes(self):
        self.assertEqual(self.spend([spent(1)], issues=(LEDGER, {**STATUS, "body": LEDGER["body"]}))["day"], 1,
                         "the same run recorded in two issues would double")
        comments = {1: [spent(1, run="1")], 2: [spent(2, run="2")]}
        with mock.patch.object(bot, "run", FakeGh({
                "GET repos/o/r/issues[?]*": [{**STATUS, "body": LEDGER["body"]}, LEDGER],
                "GET repos/o/r/issues/1/comments*": comments[1], "GET repos/o/r/issues/2/comments*": comments[2]})):
            self.assertEqual(bot.spend("o/r", NOW)["day"], 3, "a run in the second ledger issue would be missed")
            self.assertEqual(bot.issue("o/r", "ledger"), 1, "writers would split across ledger issues")

    def test_the_record_step_appends_what_spend_reads_back(self):
        recorded = {}

        def comment(argv, stdin):
            recorded["body"] = stdin
            return ""

        usage = json.dumps({"total_cost_usd": 2.5, "tokens_from": "result", "pr": "7", "head_sha": HEAD, "mode": "full",
                            "run_id": "9", "run_attempt": "1"})
        out, fake = step(bot.record, {"GET repos/o/r/issues[?]*": [LEDGER], "issue comment": comment}, repo="o/r",
                         pr="7", github_run_id="9", github_run_attempt="1", usage=usage, gpu_usd="3", box_usd="1.5",
                         approval="approve")
        self.assertEqual(self.spend([{"user": {"login": "github-actions[bot]"}, "created_at": "2026-11-08T01:00:00Z",
                                      "body": recorded["body"]}]), {"day": 7.0, "month": 7.0},
                         "a run's GPU and box cost would not count toward the cap")
        self.assertEqual(self.spend_of(recorded["body"]).get("approval"), "approve")
        self.assertTrue(all(recorded["body"] not in arg for argv, _ in fake.calls for arg in argv),
                        "the record would reach gh's command line")

    @staticmethod
    def spend_of(body):
        return json.loads(base64.b64decode(bot.SPEND.search(body)[1]))


class TestCaps(unittest.TestCase):
    def preflight(self, used, alerts=()):
        routes = {"GET repos/o/r/collaborators/an-author/permission": {"permission": "write"},
                  "GET repos/o/r/issues[?]*": [LEDGER, STATUS], "GET repos/o/r/issues/1/comments*": used,
                  "GET repos/o/r/issues/2/comments*": list(alerts), "issue comment": "", "pr comment": ""}
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            out, fake = step(bot.preflight, routes, repo="o/r", pr="1", key_set="true", command="full",
                             actor="an-author", author="an-author", assoc="MEMBER", skip_label="false")
        return out, fake.writes()

    def test_past_a_cap_nothing_starts_it_says_so_on_the_pr_and_alerts_once(self):
        out, writes = self.preflight([spent(10_000.5)])
        self.assertEqual(out, {"run": "false"}, "a review would start past the day's cap")
        said = dict(writes)
        self.assertEqual(sorted(said), ["issue comment", "pr comment"])
        self.assertIn("daily spend cap ($10,000 of $10,000)", said["pr comment"])
        self.assertIn("resets at 00:00 UTC", said["pr comment"])
        marker = said["issue comment"].split("\n")[0]
        again = [{"user": {"login": "github-actions[bot]"}, "body": f"{marker}\nalready"}]
        self.assertEqual([k for k, _ in self.preflight([spent(10_000.5)], again)[1]], ["pr comment"],
                         "every refused request would alert the status issue's subscribers")
        out, writes = self.preflight([spent(50_000, "2026-11-02T01:00:00Z")])
        said = dict(writes)
        self.assertIn("monthly spend cap", said["pr comment"])
        self.assertIn("resets on the 1st", said["pr comment"])
        self.assertEqual(self.preflight([spent(9_999)])[0]["run"], "true", "a run under the caps would be refused")

    def test_an_unreadable_ledger_refuses_rather_than_spends(self):
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            out, fake = step(bot.preflight, {"GET repos/o/r/collaborators/an-author/permission":
                                             {"permission": "write"}, "GET repos/o/r/issues[?]*": Fail(),
                                             "pr comment": ""},
                             repo="o/r", pr="1", key_set="true", command="light", actor="an-author", author="an-author",
                             assoc="MEMBER", skip_label="false")
        self.assertEqual(out, {"run": "false"})
        self.assertIn("couldn't read its spend ledger", fake.writes()[0][1], "a blind bot could spend past its cap")


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
        def ref(number, labels=(), pr=False, body=""):
            issue = {"number": number, "title": "t", "labels": [{"name": n} for n in labels], "body": body}
            return {"event": "cross-referenced", "source": {"issue": {**issue, **({"pull_request": {}} if pr else {})}}}

        timeline = [ref(10, ["bug"]), ref(11, ["enhancement"]), ref(12, pr=True, body="Fixes #7"),
                    ref(13, ["type/bug"], pr=True), ref(14, pr=True, body="Fixes #70"),
                    {"event": "labeled"}]
        signals, mentions = bot.referenced(timeline, 7)
        self.assertEqual({int(s["key"].split(":")[-1]) for s in signals}, {10, 12, 13})
        self.assertEqual(len(mentions), 2, "an unrelated mention would be reported as a bug")

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
        self.assertEqual([s["text"] for s in hits], ["#9 changed lines #7 wrote at `a.py:2`"],
                         "a fix of the PR's lines would go unseen")

    def test_an_open_approved_pr_the_merge_queue_threw_out_is_a_signal(self):
        nodes = [{"createdAt": "2026-11-07T00:00:00Z", "reason": "FAILED_CHECKS"}]
        with mock.patch.object(bot, "run", FakeGh({"GET repos/o/r/issues/8/timeline*": [], "POST graphql": {
                "data": {"repository": {"pullRequest": {"timelineItems": {"nodes": nodes}}}}}})):
            signals, _ = bot.pr_signals("o/r", {"number": 8, "state": "open"}, [], set())
        self.assertEqual(signals, [{"key": "8:ejected:2026-11-07T00:00:00Z",
                                    "text": "#8 removed from the merge queue (FAILED_CHECKS)"}])

    def run_monitor(self, body, bug_title="crash in decode"):
        calls = {"body": None}

        def issue_status(argv, stdin):
            return {"number": 2, "body": calls["body"] or body, "user": {"login": "github-actions[bot]"}}

        def patch(argv, stdin):
            calls["body"] = json.loads(stdin)["body"]
            return {}

        routes = {
            "GET repos/o/r/issues[?]state=all&creator=*": [LEDGER, STATUS],
            "GET repos/o/r/issues/1/comments*": [spent(3.0, run="1", approval="approve", pr="7")],
            "GET repos/o/r/issues/2": issue_status, "PATCH repos/o/r/issues/2": patch, "issue comment": "",
            "GET repos/o/r/pulls/7": {"number": 7, "title": "Faster decode @someone", "state": "closed",
                                      "merged_at": "t", "merge_commit_sha": "d" * 40},
            "GET repos/o/r/pulls/7/reviews*": [{"state": "APPROVED", "user": {"login": "mstar-review[bot]"}}],
            "GET repos/o/r/pulls/7/commits*": [{"sha": "a" * 40}],
            "GET repos/o/r/issues/7/timeline*": [],
            "GET repos/o/r/commits/*/check-runs*": {"check_runs": []},
            "GET repos/o/r/issues[?]state=all&labels=bug*": [{"number": 30, "title": bug_title,
                                                               "created_at": "2026-11-08T00:00:00Z"}],
            "GET repos/o/r/issues[?]state=closed&labels=type*": [],
            "GET repos/o/r/pulls[?]*": [],
            "git log": (f'{"c" * 40}\x1f{"d" * 40}\x1fRevert "Faster decode (#7)"\x1f\x1e\n'
                        f'{"d" * 40}\x1f{"b" * 40}\x1fFaster decode (#7)\x1f\x1e\n'),
            "git diff": "a.py\n", "git show": "", "git blame": "",
        }
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            out, fake = step(bot.monitor, routes, repo="o/r", approvals="on")
        return calls["body"], fake

    def test_new_signals_reach_subscribers_once_and_the_body_is_safe(self):
        body, fake = self.run_monitor(STATUS["body"], bug_title="crash @someone <!-- x")
        self.assertTrue(body.startswith(bot.issue_marker("status")), "the next run couldn't find the status issue")
        self.assertIn("$3 of $10,000", body)
        self.assertIn("Approvals: **on**", body)
        comments = [s for k, s in fake.writes() if k == "issue comment"]
        self.assertEqual(len(comments), 1, "a new signal would not notify subscribers")
        self.assertIn("reverted", comments[0])
        self.assertNotIn("@someone", body + comments[0], "a PR's title could ping someone from the status issue")
        self.assertNotIn("<!-- x", body, "a title could open an HTML comment")
        _, fake = self.run_monitor(body)
        self.assertEqual([k for k, _ in fake.writes() if k == "issue comment"], [],
                         "a signal already reported would notify subscribers every day")


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
                out, _ = step(bot.usage, {}, pr="1", head_sha="abc", mode="light", outcome="failure", runner_temp=temp,
                              home=temp, github_run_id="5", github_run_attempt="1", github_step_summary=str(summary))
                self.assertEqual(json.loads(out["record"])["api_calls"], 0)
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


class TestImprove(unittest.TestCase):
    def inputs(self, routes):
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            return step(bot.improve_inputs, routes, repo="o/r", key_set="true")

    def test_past_a_cap_nothing_runs(self):
        # No PR route: reading any evidence would fail the test.
        routes = {"GET repos/o/r/issues[?]*": [LEDGER, STATUS], "GET repos/o/r/issues/1/comments*": [spent(10_000.5)],
                  "GET repos/o/r/issues/2/comments*": [], "issue comment": ""}
        with workspace() as d:
            out, fake = self.inputs(routes)
            self.assertFalse((d / "improve-evidence").exists(), "evidence was collected past the cap")
        self.assertEqual(out, {"run": "false"}, "the model would run past the cap")
        self.assertEqual([k for k, _ in fake.writes()], ["issue comment"])

    def test_a_finished_run_this_week_starts_nothing_and_a_failed_one_can_be_retried(self):
        routes = {"GET repos/o/r/issues[?]*": [LEDGER, STATUS], "GET repos/o/r/issues/2/comments*": []}
        done = spent(1, mode="improve", usage={"outcome": "success"})
        with workspace():
            routes["GET repos/o/r/issues/1/comments*"] = [done]
            self.assertEqual(self.inputs(routes)[0], {"run": "false"}, "a re-run would pay twice for one week")
        failed = spent(1, mode="improve", usage={"outcome": "failure"})
        with workspace():
            routes["GET repos/o/r/issues/1/comments*"] = [failed]
            routes["GET repos/o/r/issues/2/comments*"] = [{"user": {"login": "github-actions[bot]"},
                                                           "html_url": "u", "body": "a signal"}]
            self.assertEqual(self.inputs(routes)[0]["run"], "true", "a failed run could never be retried")

    def test_evidence_keeps_what_a_reader_needs_without_names(self):
        state = base64.b64encode(json.dumps({"moot": [{"ref": "11", "why": "gone"}]}).encode()).decode()
        at, old = "2026-11-07T00:00:00Z", "2026-10-01T00:00:00Z"
        bot_user = {"login": "mstar-review[bot]", "type": "Bot"}
        person = {"login": "a-person", "type": "User"}
        routes = {
            "GET repos/o/r/issues[?]*": [LEDGER, STATUS],
            "GET repos/o/r/issues/1/comments*": [spent(3, pr="5", mode="light", run="9", usage={"outcome": "failure"})],
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
        self.assertEqual(e["runs"][0]["outcome"], "failure")

    def post(self, output, usage=""):
        routes = {"GET repos/o/r/issues[?]*": [LEDGER], "GET repos/o/r/issues/1/comments*": [], "issue comment": "",
                  "POST repos/o/r/issues": "{}"}
        with mock.patch.object(bot, "utcnow", lambda: NOW):
            return step(bot.improve_post, routes, repo="o/r", github_run_id="7", github_run_attempt="1", output=output,
                        usage=usage)

    def test_an_empty_list_opens_no_issue_and_still_records_the_cost(self):
        out, fake = self.post('{"improvements": []}', json.dumps({"mode": "improve", "total_cost_usd": 1.5}))
        self.assertNotIn("exit", out)
        writes = fake.writes()
        self.assertEqual([k for k, _ in writes], ["issue comment"], "an empty week opened an issue")
        record = json.loads(base64.b64decode(bot.SPEND.search(writes[0][1])[1]))
        self.assertEqual((record["usd"], record["pr"]), (1.5, ""))

    def test_a_post_is_redacted_on_stdin_with_links_kept_in_the_repository(self):
        secret = "sk-ant-" + "a1b2c3d4" * 3
        item = {"title": f"leak {secret}", "change": "ask @a-person about #7, see https://evil.example/x.png",
                "where": "bot.py", "evidence": ["https://github.com/o/r/pull/5#discussion_r1", "https://evil.example/y"]}
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
        self.assertNotIn("evil.example", issue["body"])
        self.assertNotIn("a-person", issue["body"])


if __name__ == "__main__":
    unittest.main()
