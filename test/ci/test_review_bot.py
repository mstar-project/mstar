"""Check .github/review/bot.py, the review workflow's steps, against a stand-in for `gh`."""

import fnmatch
import json
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
