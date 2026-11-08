#!/usr/bin/env python3
"""The review bot's steps: every step of claude-review.yml that isn't an action is a function here,
run with the default branch checked out as

    python3 .github/review/bot.py <step>

A step's parameters are its environment variables, lowercased: the step's `env:` in the workflow is
its interface. It records its outputs in `out`, which `main` writes to $GITHUB_OUTPUT even when the
step fails. Every call to another program (`gh`, `git`, gpu.py, ssh-keygen) goes through `run`, the
one place a test replaces. Text the bot posts reaches `gh` on stdin, never its command line.
"""
import base64
import collections
import datetime
import fnmatch
import inspect
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

import render_summary
import session

# Credential-shaped strings, removed from everything the bot posts or saves.
REDACT = [re.compile(p) for p in (
    r"sk-ant-[A-Za-z0-9_-]+",
    r"(gh[opsur]_|github_pat_)[A-Za-z0-9_]{20,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+",
)]
WRITE = ("admin", "maintain", "write")
# The bot's REST logins: the review app's, and github-actions', which it posted as before the app.
BOT_USERS = [f"{b}[bot]" for b in render_summary.BOTS]
# The marker of a stopped review's "didn't complete" comment, of one a later run resolved, and of a refused
# /review continue's.
INCOMPLETE = "<!-- claude-review-incomplete -->"
RESOLVED = "<!-- claude-review-incomplete-resolved -->"
REFUSED = "<!-- claude-review-continue-refused -->"
# What Claude Code wrote of the run: the action's execution file, and the session transcripts.
EXECUTION = "claude-execution-output.json"
TYPES = render_summary.TYPES
# The first line of the bot's live-test comment (gpu_review.py), the only line of it the bot reads.
LIVE_TEST = re.compile(r"<!-- claude-live-test sha=([0-9a-f]{40}) verdict=([a-z]+) -->")
# A branch name safe for the API path and the command line.
BRANCH = re.compile(r"[A-Za-z0-9._/][A-Za-z0-9._/-]*")
THREADS = """query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $pr) {
    reviewThreads(first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes { isResolved isOutdated comments(first: 1) { nodes { databaseId author { login } body path line } } }
    } } } }"""
# A run's record, base64 JSON, in a ledger comment: one comment per run, so concurrent runs can't lose updates.
SPEND = re.compile(r"<!-- review-spend ([A-Za-z0-9+/=]+) -->")
# USD the bot may spend per UTC day and month on review tokens and Eras GPU time, a change to main like the
# approval rule. Slurm GPUs are free.
CAPS = {"day": 10_000, "month": 50_000}
ISSUES = {"ledger": ("Review bot spend ledger", "The review bot's spend, one comment per run. Read by code."),
          "status": ("Review bot status", "The review bot's status: a daily report in this body, alerts as comments.")}
EJECTIONS = """query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $pr) {
    timelineItems(first: 100, after: $endCursor, itemTypes: [REMOVED_FROM_MERGE_QUEUE_EVENT]) {
      pageInfo { hasNextPage endCursor }
      nodes { ... on RemovedFromMergeQueueEvent { createdAt reason } }
    } } } }"""
# The monitor's state in the status issue's body (signals already reported), base64 JSON.
MONITOR = re.compile(r"<!-- review-monitor-state ([A-Za-z0-9+/=]+) -->")
MONITOR_DAYS = 60
ISSUE_COMMENTS = 2_000  # a bot issue past this rolls over to a new one; GitHub stops an issue at 2,500
# How a PR's body says it fixes or reverts another, and the subjects `szz` treats as fixes.
FIXES = re.compile(r"\b(?:fix(?:e[sd])?|revert(?:s|ed)?)\s+#(\d+)\b", re.IGNORECASE)
# A fix-word, unless what follows is a typo, or all it fixes is a comment, docs or lint: such a word ending the
# subject, before a `.` or a PR number.
FIX_SUBJECT = re.compile(r"\b(?:fix(?:e[sd]|ing)?|bugs?|bugfix(?:es)?|revert(?:s|ed|ing)?)\b"
                         r"(?!:?\s+(?:a\s+|the\s+)?(?:typos?\b|(?:comments?|docs?|lint|format\w*)\s*\.?\s*"
                         r"(?:\(#\d+\)\s*)?$))",
                         re.IGNORECASE)
# A commenter's author_association that makes them a contributor (see the workflow's header).
CONTRIBUTORS = ("OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR")
# The weekly improvement run: how much a reader needs (PRs read, and comments kept per PR; characters of a
# comment), and what the model returns.
IMPROVE_ITEMS = 30
IMPROVE_TEXT = 800
IMPROVE_SCHEMA = {
    "type": "object",
    "properties": {"improvements": {"type": "array", "maxItems": 10, "description": "most useful first; may be empty",
                                    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "change": {"type": "string", "description": "what to change, concretely"},
            "where": {"type": "string", "description": "the skill or code file, and its section or function"},
            "evidence": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"},
                         "description": "`url` values from the evidence files"},
        },
        "required": ["title", "change", "where", "evidence"],
    }}},
    "required": ["improvements"],
}


def run(argv, input=None, check=True, timeout=None):
    """Run `argv` (no shell). A failure raises with the end of what it printed, never its argv, which
    can hold a token. Bytes that aren't UTF-8 pass through unchanged (surrogateescape)."""
    r = subprocess.run(argv, input=input, capture_output=True, encoding="utf-8", errors="surrogateescape",
                       timeout=timeout, check=False)
    if check and r.returncode:
        raise RuntimeError(f"{os.path.basename(argv[0])} {argv[1] if len(argv) > 1 else ''} failed: "
                           f"{(r.stderr or r.stdout).strip()[-500:]}")
    return r


def gh(*args, input=None, check=True):
    return run(["gh", *args], input=input, check=check)


def api(path, *args):
    """`gh api` on `path`, parsed. With --paginate --slurp, the list of its pages."""
    return json.loads(gh("api", *args, path).stdout or "null")


def api_list(path):
    """Every item of a paginated REST list."""
    return [item for page in api(path, "--paginate", "--slurp") for item in page]


def graphql(query, repo, pr):
    """The pages of a paginated GraphQL query on the PR."""
    owner, name = repo.split("/")
    return api("graphql", "--paginate", "--slurp", "-F", f"owner={owner}", "-F", f"name={name}", "-F", f"pr={pr}",
               "-f", f"query={query}")


def write(path, text):
    """Write `text` as `gh` printed it, bytes that aren't UTF-8 included."""
    Path(path).write_text(text, encoding="utf-8", errors="surrogateescape")


def write_json(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def marked(comments, marker):
    """The ids of the bot's own comments that start with `marker`."""
    return [c["id"] for c in comments
            if (c.get("user") or {}).get("login") in BOT_USERS and (c.get("body") or "").startswith(marker)]


def oneline(commits):
    """since-commits.txt and since-live-test-commits.txt: one line per commit, merges marked."""
    lines = []
    for c in commits:
        merge = "[merge] " if len(c["parents"]) > 1 else ""
        subject = c["commit"]["message"].split("\n")[0]
        lines.append(f"{c['sha'][:10]} {merge}{subject}\n")
    return "".join(lines)


def comment(repo, pr, body):
    """Post `body` as a comment on the PR, from stdin; its URL."""
    return gh("pr", "comment", str(pr), "--repo", repo, "--body-file", "-", input=body).stdout.strip()


def permission(repo, login):
    """`login`'s permission on the repository, or "none" when GitHub won't say."""
    try:
        return api(f"repos/{repo}/collaborators/{login}/permission")["permission"]
    except (RuntimeError, ValueError, KeyError, TypeError):
        return "none"


def redact(text, exact=()):
    """`text` without the exact values in `exact`, then without credential-shaped strings."""
    for value in exact:
        if value:
            text = text.replace(value, "[REDACTED]")
    for pattern in REDACT:
        text = pattern.sub("[REDACTED]", text)
    return text


def issue_marker(kind):
    return f"<!-- review-bot-{kind} -->"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def issues(repo, kind):
    """The bot's own issues of `kind`, lowest number first: those a bot login created whose body starts
    with the kind's marker. Created in a race, or rolled over when one is full, several can exist; readers
    take them all, writers the lowest with room (`issue`)."""
    found = {}
    for login in BOT_USERS:
        creator = login.replace("[", "%5B").replace("]", "%5D")
        for i in api_list(f"repos/{repo}/issues?state=all&creator={creator}&per_page=100"):
            if (i.get("user") or {}).get("login") in BOT_USERS \
                    and (i.get("body") or "").startswith(issue_marker(kind)):
                found[i["number"]] = i
    return [found[n] for n in sorted(found)]


def issue(repo, kind):
    """The number of the bot's issue of `kind` to write to: the lowest with room for more comments, created if
    there is none."""
    for i in issues(repo, kind):
        if i.get("comments", 0) < ISSUE_COMMENTS:
            return i["number"]
    title, text = ISSUES[kind]
    body = json.dumps({"title": title, "body": f"{issue_marker(kind)}\n{text}\n"})
    return json.loads(gh("api", "-X", "POST", f"repos/{repo}/issues", "--input", "-", input=body).stdout)["number"]


def comment_on_issue(repo, number, body):
    """Post `body` as a comment on an issue, from stdin."""
    gh("issue", "comment", str(number), "--repo", repo, "--body-file", "-", input=body)


def ledger(repo, since):
    """[(created_at, record)] of the ledger since `since` (ISO 8601): the bot's own comments from every ledger
    issue, once per run and attempt of what they cost. Re-running only a run's spend job records the same costs
    again, with the attempt its usage came from; re-running all its jobs spends again, and re-running only its
    live test spends GPUs and the box again, which counts on top of the review already counted. A failed read
    raises."""
    records, seen = [], set()
    for i in issues(repo, "ledger"):
        for c in api_list(f"repos/{repo}/issues/{i['number']}/comments?since={since}&per_page=100"):
            m = SPEND.search(c.get("body") or "")
            if not m or (c.get("user") or {}).get("login") not in BOT_USERS:
                continue
            try:
                record = json.loads(base64.b64decode(m[1]))
            except ValueError:
                continue
            run_key = record.get("run"), (record.get("usage") or {}).get("run_attempt") or record.get("attempt")
            live_key = (*run_key, record.get("gpu_usd"), record.get("box_usd"))
            if live_key in seen:
                continue
            if run_key in seen:  # the live test ran again: new GPU and box cost, the review's tokens already counted
                record = {**record, "usd": round(finite_usd(record, "gpu_usd") + finite_usd(record, "box_usd"), 4)}
            seen |= {run_key, live_key}
            records.append((c["created_at"], record))
    return records


def spend(repo, now):
    """{"day": USD, "month": USD} spent so far in `now`'s UTC day and month."""
    day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
    used = {"day": 0, "month": 0}
    for created, record in ledger(repo, f"{month}-01T00:00:00Z"):
        usd = finite_usd(record)
        used["month"] += usd if created.startswith(month) else 0
        used["day"] += usd if created.startswith(day) else 0
    return used


def finite_usd(record, key="usd"):
    """A record's USD under `key`, or 0 when it isn't a finite, non-negative number."""
    usd = record.get(key)
    return usd if isinstance(usd, (int, float)) and not isinstance(usd, bool) and 0 <= usd < float("inf") else 0


def over_cap(repo):
    """Why nothing new may start, or None: the first cap reached, with one alert on the status issue for it
    per day or month, or a failed read of the ledger. Checked at every step that starts spending."""
    now = utcnow()
    try:
        used = spend(repo, now)
    except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"::warning::Couldn't read the spend ledger: {e}")
        return "The review bot couldn't read its spend ledger, so no new review or GPU run starts; try again later."
    for window, cap in CAPS.items():
        if used[window] < cap:
            continue
        word, resets, stamp = ("daily", "at 00:00 UTC", now.strftime("%Y-%m-%d")) if window == "day" \
            else ("monthly", "on the 1st", now.strftime("%Y-%m"))
        why = (f"The review bot's {word} spend cap (${used[window]:,.0f} of ${cap:,}) is reached; it resets {resets}. "
               "No new review or GPU run starts until then.")
        alert(repo, f"{window}-{stamp}", why)
        return why
    return None


def alert(repo, key, text):
    """Comment `text` on the status issue, once per `key`, best effort: a failed alert changes no decision."""
    marker = f"<!-- review-alert {key} -->"
    try:
        number = issue(repo, "status")
        if not marked(api_list(f"repos/{repo}/issues/{number}/comments?per_page=100"), marker):
            comment_on_issue(repo, number, f"{marker}\n{text}\n")
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        print(f"::warning::Couldn't post the alert: {e}")


# ---------------------------------------------------------------- preflight job

def preflight(out, repo, pr, key_set, command, actor, author, assoc, skip_label, team_turns="2000",
              self_turns="40"):
    """Whether a comment starts a review, in which mode (`command`, one of the workflow's fixed words)
    and with which turn cap; a request it turns down for a reason the commenter can act on gets a
    one-line reply. Write access is checked against the actor's permission; a contributor may run
    `/review light` or `/review continue` on their own PR, and only then is `non_write_user` set. A
    full review runs the PR's code, so it also takes an author with write access (author_association
    can't tell write from read)."""
    if key_set != "true":
        print("::warning::ANTHROPIC_API_KEY is not set; skipping the review.")
        out["run"] = "false"
        return
    if command not in ("light", "nogpu", "full", "continue"):
        out["run"] = "false"
        return
    mode = "" if command == "continue" else command  # the stopped review's mode, from its handoff

    def start(**outputs):
        """The request is allowed: run it, unless the bot is past a spend cap or can't tell."""
        stop = over_cap(repo)
        if stop:
            print(f"::notice::{stop}")
            out["run"] = "false"
            reply(repo, pr, stop)
            return
        out.update(run="true", **outputs)

    perm = permission(repo, actor)
    if perm in WRITE:
        author_perm = permission(repo, author) if command == "full" else ""
        if command == "full" and author_perm not in WRITE:
            print(f"::notice::{author} has '{author_perm}' access, not write; a full review runs the PR's code.")
            out["run"] = "false"
            reply(repo, pr, "`/review` runs this PR's code on GPUs, so it takes an author with write access; "
                            "comment `/review nogpu` for the review without the live test.")
            return
        start(mode=mode, turns=team_turns)
        return
    own = assoc in CONTRIBUTORS and actor != "" and actor == author and skip_label != "true"
    if own and command in ("light", "continue"):
        print(f"::notice::{actor} has '{perm}' access; reviewing their own PR as its author ({assoc}).")
        start(mode=mode, non_write_user=actor, turns=self_turns)
        return
    out["run"] = "false"
    if not own:
        print(f"::notice::{actor} has '{perm}' access, not write; only people with write access, or a contributor "
              "on their own PR with `/review light`, can request a review.")
        return
    print(f"::notice::{actor} has '{perm}' access, not write; this review needs write access.")
    reply(repo, pr, "This review needs write access; on your own PR, comment `/review light`.")


def reply(repo, pr, text):
    """The one-line reply to a refused request, best effort."""
    try:
        comment(repo, pr, text)
    except RuntimeError as e:
        print(f"::warning::Couldn't reply: {e}")


# ---------------------------------------------------------------- review job: before the review

def resolve(out, repo, pr, mode, comment_id, non_write=""):
    """React 👀 to the /review comment (best effort), then the PR's head, its title and the review's
    settings. A /review continue (`mode` empty) takes the mode, effort and timeout of the review it
    continues (restore). The title is untrusted; it only ever reaches the prompt."""
    try:
        gh("api", "-X", "POST", f"repos/{repo}/issues/comments/{comment_id}/reactions", "-f", "content=eyes")
    except RuntimeError as e:
        print(f"::warning::Couldn't react to the comment: {e}")
    pull = api(f"repos/{repo}/pulls/{pr}")
    head_repo = (pull["head"].get("repo") or {}).get("full_name")
    if not head_repo:
        print("::warning::The PR's head repository no longer exists; can't check it out.")
        out["why"] = "its head repository (the fork) no longer exists"
        sys.exit(1)
    out["continue"] = "true" if mode == "" else "false"
    out.update(head_sha=pull["head"]["sha"], head_repo=head_repo)
    # Mode sets the prompt's reporting bar and the reasoning effort. Without --effort, Claude Code
    # uses its default for Opus (medium).
    if mode in ("nogpu", "full"):
        out.update(mode=mode, effort="xhigh")
        timeout = 45
    else:
        out.update(mode="light", effort="high")
        timeout = 25
    # Someone with write access: a timeout that only stops a runaway, like the turn cap.
    out["timeout"] = 300 if not non_write else timeout
    out["title"] = pull["title"]


def restore(out, repo, pr, head_sha, runner_temp, non_write=""):
    """/review continue: fetch the handoff a stopped review saved (handoff) into $RUNNER_TEMP/prev, and
    that review's mode, effort and timeout. The newest unexpired artifact for this PR wins, and only
    from this workflow on issue_comment, a trigger that runs the base repository's copy of it: a fork's
    own workflow can upload an artifact with any name. Only while the stopped review's "didn't
    complete" comment is still open, since any later run that finishes rewrites it as resolved. The
    handoff's values are checked before use; a nogpu one needs write access to continue."""
    name = f"claude-review-handoff-{pr}"

    def fail(why):
        out["why"] = why
        print(f"::notice::{why}")
        sys.exit(1)

    try:
        comments = api_list(f"repos/{repo}/issues/{pr}/comments")
    except (RuntimeError, ValueError):
        fail("the PR's comments couldn't be read; try `/review continue` again")
    if not marked(comments, INCOMPLETE):
        fail("there is no stopped review of this PR to continue: none stopped part-way, or a later run already "
             "finished (handoffs are kept for 7 days); ask for a new review with `/review light` or `/review`")
    try:
        artifacts = [a for a in api(f"repos/{repo}/actions/artifacts?name={name}&per_page=30")["artifacts"]
                     if not a["expired"]]
    except (RuntimeError, ValueError, KeyError, TypeError):
        artifacts = []
    found = None
    for a in reversed(sorted(artifacts, key=lambda a: a["created_at"])):
        try:
            r = api(f"repos/{repo}/actions/runs/{a['workflow_run']['id']}")
        except (RuntimeError, ValueError):
            continue
        if fnmatch.fnmatchcase(f"{r['path']}:{r['event']}", ".github/workflows/claude-review.yml*:issue_comment"):
            found = a["workflow_run"]["id"]
            break
    if found is None:
        fail("there is no stopped review of this PR to continue (handoffs are kept for 7 days); ask for a new "
             "review with `/review light` or `/review`")
    saved = Path(runner_temp, "prev")
    try:
        gh("run", "download", str(found), "--repo", repo, "-n", name, "-D", str(saved))
    except RuntimeError:
        fail("the saved handoff couldn't be downloaded; try `/review continue` again")
    try:
        meta = json.loads((saved / "handoff.json").read_text())
        mode = meta.get("mode")
    except (OSError, ValueError, AttributeError):
        fail("the saved handoff is malformed")
    # A handoff saved before the modes were renamed carries the old name.
    mode = render_summary.current_mode(mode if isinstance(mode, str) else "")
    if not (saved / "handoff.md").is_file() or not (saved / "handoff.md").stat().st_size:
        fail("the saved handoff is malformed")
    if jq_text(meta.get("pr")) != pr:
        fail("the saved handoff is for another PR")
    if jq_text(meta.get("head_sha")) != head_sha:
        fail("the PR has new commits since that review stopped; ask for a new review of the current head with "
             "`/review light` or `/review`")
    if mode == "light":
        effort, timeout = "high", 25
    elif mode == "nogpu":
        if non_write:
            fail("continuing a nogpu review takes write access, like `/review nogpu`")
        effort, timeout = "xhigh", 45
    elif mode == "full":
        fail("a `/review` that stopped can't be continued (its live test never started); comment `/review` again")
    else:
        fail("the saved handoff is malformed")
    out.update(dir=str(saved), mode=mode, effort=effort, timeout=300 if not non_write else timeout)


def jq_text(value):
    """`value` as text, as `jq -r` prints it."""
    return value if isinstance(value, str) else json.dumps(value)


def file_patch(f):
    """One file of the files API as a diff. A removed file's new side is /dev/null, as in git's own diff."""
    old = f["previous_filename"] if f.get("previous_filename") is not None else f["filename"]
    new = "/dev/null" if f["status"] == "removed" else f"b/{f['filename']}"
    patch = f["patch"] if f.get("patch") is not None else "(no patch from GitHub: binary, or too large to show)"
    return f"diff --git a/{old} b/{f['filename']}\n--- a/{old}\n+++ {new}\n{patch}\n"


def stacked_above(repo, pull, head_sha):
    """The PRs stacked above this one, for a finding one of them fixes (fixed_above): every open PR based
    on this PR's branch, then on theirs, up to 5, as [{number, head_sha}], with above-<n>.diff the change
    from this head to PR n's head. Only a branch of this repository can have PRs based on it, so only
    PRs from this repository count and have their branches walked (a fork's branch name can match a
    branch here), and a PR counts only when its head descends from this head."""
    found = []
    ref = pull["head"]["ref"]
    same = pull["head"].get("repo") and pull["head"]["repo"]["full_name"] == pull["base"]["repo"]["full_name"]
    branches = [ref] if same and BRANCH.fullmatch(ref) else []
    while branches and len(found) < 5:
        listed = json.loads(gh("pr", "list", "--repo", repo, "--base", branches.pop(0), "--state", "open",
                               "--json", "number,headRefName,headRefOid,isCrossRepository").stdout)
        for p in listed:
            if p["isCrossRepository"] or not isinstance(p["number"], int) or isinstance(p["number"], bool) \
                    or not re.fullmatch(r"[0-9a-f]{40}", str(p["headRefOid"])) \
                    or not BRANCH.fullmatch(str(p["headRefName"])) or len(found) >= 5:
                continue
            if api(f"repos/{repo}/compare/{head_sha}...{p['headRefOid']}")["status"] != "ahead":
                continue
            diff = gh("api", "-H", "Accept: application/vnd.github.diff",
                      f"repos/{repo}/compare/{head_sha}...{p['headRefOid']}").stdout
            write(f"above-{p['number']}.diff", diff)
            found.append({"number": p["number"], "head_sha": p["headRefOid"]})
            branches.append(p["headRefName"])
    return found


def diff_files(diff):
    """{"diff --git" line: [its header lines, [its hunks, each a list of lines]]} of a unified diff. Lines
    end only at \\n: a form feed or U+2028 inside a source line (str.splitlines breaks on those) could
    fake a header and hide the code around it."""
    files, current = {}, None
    for line in re.findall(r".*\n|.+", diff):
        if line.startswith("diff --git "):
            current = files[line] = [[line], []]
        elif current is None:
            continue
        elif line.startswith("@@"):
            current[1].append([line])
        elif current[1]:
            current[1][-1].append(line)
        else:
            current[0].append(line)
    return files


def changes(hunk):
    """What a hunk changes: its runs of added and removed lines, each with the function on its `@@` line
    and the nearest non-blank context line on either side within the hunk, so a change moved elsewhere
    in a file counts. The line numbers stay out: a rebase shifts them."""
    section, runs, run, above, open_runs = hunk[0].split("@@", 2)[-1], [], [], "", []
    for line in hunk[1:]:
        if line.startswith(("+", "-", "\\")):
            run.append(line)
            continue
        if run:
            open_runs.append((section, above, tuple(run)))
            run = []
        if line.strip():
            runs += [(*r, line) for r in open_runs]
            open_runs = []
            above = line
    if run:
        open_runs.append((section, above, tuple(run)))
    return runs + [(*r, "") for r in open_runs]


def header_change(head, hunks):
    """What a file's diff header says the PR changes: its mode lines and, for a file with no hunks (binary,
    a pure rename), its other lines and new blob. Not the old blob, which a rebase moves."""
    if hunks:
        return [x for x in head if x.startswith(("old mode", "new mode", "new file mode", "deleted file mode"))]
    return [x.split("..")[-1] if x.startswith("index ") else x for x in head]


def not_in(files, others):
    """The lines of the files and hunks of `files` (diff_files) whose changes `others` doesn't make."""
    parts = []
    for header, (head, hunks) in files.items():
        other_head, other_hunks = others.get(header, ([], []))
        made = collections.Counter(run for h in other_hunks for run in changes(h))
        kept = []
        for h in hunks:
            runs = collections.Counter(changes(h))
            if runs - made:
                kept.append(h)
            else:
                made -= runs
        mine = header_change(head, hunks)
        if kept or mine and mine != header_change(other_head, other_hunks):
            parts += head + [line for h in kept for line in h]
    return parts


DROPPED = "=== dropped since the last review: these were in the PR then, and aren't now ===\n"


def interdiff(old, new):
    """What differs between `old`, a PR's diff at the last review, and `new`, its diff now: the hunks of
    `new` that make a change `old` doesn't and, after DROPPED, the hunks and files of `old` that `new` no
    longer has. Empty means the PR's own changes are the same, whatever a rebase moved around them."""
    before, after = diff_files(old), diff_files(new)
    dropped = not_in(before, after)
    return "".join(not_in(after, before) + ([DROPPED] + dropped if dropped else []))


def discussion(repo, pr):
    """(inline review comments, conversation-tab comments) of the PR. An inline comment is marked with its
    thread's resolved and outdated state (only GraphQL has those), and `mine` when this bot posted it."""
    threads = {}
    for page in graphql(THREADS, repo, pr):
        for t in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = t["comments"]["nodes"][:1]
            if first:
                threads[first[0]["databaseId"]] = {"resolved": t["isResolved"], "outdated": t["isOutdated"]}
    comments = []
    for c in api_list(f"repos/{repo}/pulls/{pr}/comments"):
        user = (c.get("user") or {}).get("login")
        comments.append({"id": c["id"], "user": user, "user_type": (c.get("user") or {}).get("type"),
                         "path": c.get("path"), "line": c.get("line"), "side": c.get("side"),
                         "in_reply_to_id": c.get("in_reply_to_id"), "body": c.get("body"),
                         "created_at": c.get("created_at"), "url": c.get("html_url"),
                         **threads.get(c.get("in_reply_to_id") or c["id"], {}), "mine": user in BOT_USERS})
    # The conversation tab: replies to the summary, and the author's answers.
    conversation = [{"id": c["id"], "user": (c.get("user") or {}).get("login"),
                     "user_type": (c.get("user") or {}).get("type"), "created_at": c.get("created_at"),
                     "url": c.get("html_url"), "body": c.get("body")}
                    for c in api_list(f"repos/{repo}/issues/{pr}/comments")]
    return comments, conversation


def collect(out, repo, pr, head_sha, mode, prev=""):
    """The review's inputs, in the workspace: pr.diff, pr-body.md, pr-comments.json (with each thread's
    resolved and outdated state, and `mine` on the bot's own), pr-conversation.json, a re-review's
    last-review.json with since.diff and since-commits.txt or interdiff.diff, stacked-above.json and
    above-<n>.diff, and
    open-issues.json; and the diff's length and the summary's schema as outputs. The PR's text and
    comments are untrusted: they only ever reach files the prompt calls untrusted."""
    if prev:
        shutil.copyfile(Path(prev, "handoff.md"), "handoff.md")
    pull = api(f"repos/{repo}/pulls/{pr}")
    # GitHub refuses `gh pr diff` past 20,000 lines or 300 files. Fall back to the files API (up to
    # 3,000 files; it omits the patch of a file that is too large on its own).
    r = gh("pr", "diff", str(pr), "--repo", repo, check=False)
    if r.returncode == 0:
        diff = r.stdout
    else:
        print(f"::warning::gh pr diff failed ({r.stderr[:300]}); building pr.diff from the files API.")
        diff = "".join(map(file_patch, api_list(f"repos/{repo}/pulls/{pr}/files")))
    write("pr.diff", diff)
    write("pr-body.md", (pull["body"] or "") + "\n")

    comments, conversation = discussion(repo, pr)
    write_json("pr-comments.json", comments)
    write_json("pr-conversation.json", conversation)

    # A re-review starts from this bot's last summary and reviews only what changed since
    # (review-prompt.md, "Re-review"): the commits since, when that commit is still an ancestor of the
    # head and none of them is a merge (since.diff, since-commits.txt), and otherwise the interdiff, the
    # PR's own hunks whose changes differ from its diff then and what it dropped (interdiff.diff), so a
    # merge of main or a force-push isn't read as the author's change. read_state says when it reviews the
    # whole PR again instead. The state is read only from the bot's own summaries: comments that start
    # with the summary's marker, in which model text can't open an HTML comment. A diff GitHub won't give
    # (too large, say) means a full review, never an empty since.diff.
    if not prev:
        summaries = [c for c in conversation
                     if c["user"] in BOT_USERS and (c["body"] or "").startswith(render_summary.MARKER)]
        state = render_summary.read_state(summaries[-1]["body"]) if summaries else None
        last = (state or {}).get("head")
        since = f"repos/{repo}/compare/{last}...{head_sha}"
        try:
            status = api(since)["status"] if last else ""
        except (RuntimeError, ValueError, KeyError, TypeError):
            status = ""
        listing = None  # since-commits.txt, when the range can be read as the author's commits alone
        if last and (status == "ahead" or last == head_sha):
            try:
                commits = [c for page in api(since, "--paginate", "--slurp") for c in page["commits"]]
                listing = None if any(len(c["parents"]) > 1 for c in commits) else oneline(commits)
            except (RuntimeError, ValueError, KeyError, TypeError):
                listing = None
        r = None
        if listing is not None:
            r = gh("api", "-H", "Accept: application/vnd.github.diff", since, check=False)
            if r.returncode == 0:
                write("since.diff", r.stdout)
                write("since-commits.txt", listing)
        elif last:
            # The PR's diff at that commit: from where it forked from the base branch.
            r = gh("api", "-H", "Accept: application/vnd.github.diff",
                   f"repos/{repo}/compare/{pull['base']['sha']}...{last}", check=False)
            if r.returncode == 0:
                write("interdiff.diff", interdiff(r.stdout, diff))
        if r is not None and r.returncode == 0:
            write("last-review.json", json.dumps(state) + "\n")
            print(f"::notice::Re-review: this PR was last reviewed at {last[:10]}.")
            out["rereview"] = "true"

    # full mode: the diff and commits since the PR's last live test, when it passed on an ancestor of the
    # head, for the review's live-test call (review-prompt.md, Re-review step 4). Read as gpu_review.py
    # carry reads it: the bot's own live-test comments, by their first line, the marker, alone. Without
    # both files (GitHub refused either) the review judges the whole PR.
    if mode == "full":
        live = None
        for c in conversation:
            m = LIVE_TEST.fullmatch((c["body"] or "").split("\n")[0])
            if m and c["user"] in BOT_USERS:
                live = m
        tested = f"repos/{repo}/compare/{live[1] if live else ''}...{head_sha}"
        try:
            ancestor = bool(live) and live[2] == "pass" and api(tested)["status"] in ("ahead", "identical")
        except (RuntimeError, ValueError, KeyError, TypeError):
            ancestor = False
        if ancestor:
            try:
                live_diff = gh("api", "-H", "Accept: application/vnd.github.diff", tested).stdout
                commits = oneline(c for page in api(tested, "--paginate", "--slurp") for c in page["commits"])
                write("since-live-test.diff", live_diff)
                write("since-live-test-commits.txt", commits)
            except (RuntimeError, ValueError, KeyError):
                pass

    # A failed fetch leaves none of these files, so a finding said to be fixed above stays open, and
    # the review still runs.
    try:
        above = stacked_above(repo, pull, head_sha)
        write_json("stacked-above.json", above)
        out["above"] = json.dumps(above, separators=(",", ":"))
    except (RuntimeError, ValueError, KeyError, TypeError):
        for f in Path(".").glob("above-*.diff"):
            f.unlink()

    # The open issues, [{number, title}], so a latent bug on main can name the one that tracks it. Titles
    # are user text: the model reads them as untrusted data. A failed fetch leaves no file, and the review
    # says it couldn't check.
    r = gh("issue", "list", "--repo", repo, "--state", "open", "--limit", "1000", "--json", "number,title",
           check=False)
    if r.returncode == 0:
        write("open-issues.json", r.stdout)

    out["lines"] = diff.count("\n")
    # The summary's fields, from main, for --json-schema.
    out["schema"] = json.dumps(render_summary.SCHEMA, separators=(",", ":"))


# ---------------------------------------------------------------- review job: after the review

def salvage(out, repo, pr, runner_temp):
    """The action fails a run that went past --max-turns even when Claude finished and returned its
    summary, and on a real failure it still writes everything Claude said to its execution file. Read
    the result from there: the summary to post anyway (`structured_output`), or why it stopped. Out of
    API credit: a one-line answer instead of a "didn't complete" comment, since there's nothing to
    retry or continue until it's topped up."""
    f = Path(runner_temp, EXECUTION)
    if not f.is_file() or not f.stat().st_size:
        out["why"] = "it stopped before Claude Code returned (a timeout or a setup error)"
        return
    messages = json.loads(f.read_text())

    def text(m):
        return "".join(str(b.get("text")) for b in session.blocks(m) if b.get("type") == "text")

    if any(m.get("type") == "assistant" and (m.get("error") == "billing_error"
                                               or text(m).startswith("Credit balance is too low")) for m in messages):
        print("::warning::The Anthropic API account is out of credit; skipping the review.")
        out["billing"] = "true"
        comment(repo, pr, "The review didn't run: the API account for the review bot is out of credit.")
        return
    result = ([m for m in messages if m.get("type") == "result"] or [{}])[-1]
    turns = result.get("num_turns")
    out["detail"] = f"{'?' if turns is None or turns is False else jq_text(turns)} turns"
    output = result.get("structured_output")
    if result.get("subtype") == "success" and result.get("is_error") in (None, False) and isinstance(output, dict):
        out["structured_output"] = json.dumps(output, separators=(",", ":"), ensure_ascii=False)
    else:
        out["why"] = {"error_max_turns": "it ran out of turns",
                      "error_during_execution": "Claude Code stopped on an error",
                      "success": "it finished without returning a summary"}.get(
            result.get("subtype"), "Claude Code ended without a result")


def usage(out, pr, head_sha, mode, outcome, runner_temp, home, github_run_id, github_run_attempt,
          github_step_summary, prev=""):
    """What the run used, from the execution file or, after a timeout, the session transcript: a table in
    the job summary, and the record the `spend` job writes to the ledger (`record`). Never posted on the
    PR, and never fails: a record that couldn't be read says why in its `error` field."""
    record = {"pr": pr, "head_sha": head_sha, "mode": mode, "continue": prev != "", "run_id": github_run_id,
              "run_attempt": github_run_attempt, "outcome": outcome}
    try:
        record.update(session.usage(session.load([Path(runner_temp, EXECUTION), Path(home, ".claude/projects")])))
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"::warning::Couldn't read the review's usage: {record['error']}")
    out["record"] = json.dumps(record, separators=(",", ":"))
    try:
        with open(github_step_summary, "a") as f:
            f.write(session.usage_table(record))
    except Exception as e:
        print(f"::warning::Couldn't write the job summary: {e}")


def post(out, repo, pr, output, head_sha, mode, salvaged, detail=""):
    """Post the summary, rendered from the review's fields by render_summary and redacted: model output from
    untrusted input. Read the PR's threads after the review posted its inline comments, so the status line
    counts them along with earlier ones still open. Each run posts its own summary comment, so a re-review
    doesn't overwrite the last one. A "didn't complete" comment an earlier run left is then rewritten to
    point here, and loses its marker so a later failure starts a fresh one."""
    pages = graphql(THREADS, repo, pr)
    try:
        text = render_summary.render(json.loads(output), pages, head_sha, mode,
                                     render_summary.stacked_above("stacked-above.json"), salvaged == "true")
    except Exception as e:
        text = ""
        print(f"::warning::The review's summary didn't render: {type(e).__name__}: {e}")
    text = redact(text)
    if not text.strip():
        print("::warning::The review returned no summary to render.")
        sys.exit(1)
    if salvaged == "true":
        text += (f"\n<sub>This review finished but went past the workflow's turn limit ({detail}), so the action "
                 "reported it as failed.</sub>\n")
    url = comment(repo, pr, text)
    out["id"] = url.rsplit("-", 1)[-1]
    try:
        stopped = marked(api_list(f"repos/{repo}/issues/{pr}/comments"), INCOMPLETE)
    except (RuntimeError, ValueError):
        stopped = []
    for cid in stopped:
        body = f"{RESOLVED}\nAn earlier review run didn't complete; a later run finished: {url}\n"
        try:
            gh("api", "-X", "PATCH", f"repos/{repo}/issues/comments/{cid}", "--input", "-",
               input=json.dumps({"body": body}))
        except RuntimeError as e:
            print(f"::warning::Couldn't mark comment {cid} resolved: {e}")


def hide(out, repo, pr, new_id):
    """Hide the earlier summaries as outdated, so the PR shows only the latest one and each earlier one
    stays a click away. Summaries from before the marker start with the heading instead. Only GraphQL
    says which comments are already hidden."""
    query = """query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
      repository(owner: $owner, name: $name) { pullRequest(number: $pr) {
        comments(first: 100, after: $endCursor) {
          pageInfo { hasNextPage endCursor }
          nodes { id databaseId isMinimized author { login } body }
        } } } }"""
    for page in graphql(query, repo, pr):
        for c in page["data"]["repository"]["pullRequest"]["comments"]["nodes"]:
            if (c.get("author") or {}).get("login") in render_summary.BOTS and not c["isMinimized"] \
                    and str(c["databaseId"]) != new_id \
                    and (c["body"] or "").startswith((render_summary.MARKER, "### Automated review")):
                gh("api", "graphql", "-f", "query=mutation($id: ID!) { minimizeComment(input: {subjectId: $id, "
                   "classifier: OUTDATED}) { clientMutationId } }", "-f", f"id={c['id']}")


def labels(out, repo, pr, output):
    """The PR's type/* labels, from the review's structured output. Model output, so each is checked
    against the fixed list before it reaches the API; type/* labels the review no longer returns are
    removed. Area and size labels come from pr-labels.yml."""
    try:
        types = json.loads(output).get("types") or []
    except (ValueError, AttributeError):
        types = []
    wanted = [f"type/{t}" for t in (types if isinstance(types, list) else []) if t in TYPES]
    if not wanted:
        print("::warning::The review returned no valid type; leaving the type labels alone.")
        return
    try:
        current = [x["name"] for x in api(f"repos/{repo}/issues/{pr}/labels")]
    except (RuntimeError, ValueError, KeyError, TypeError):
        current = []
    for old in current:
        if old.startswith("type/") and old not in wanted:
            gh("api", "-X", "DELETE", f"repos/{repo}/issues/{pr}/labels/{old.replace('/', '%2F')}")
    for label in wanted:
        if gh("api", f"repos/{repo}/labels/{label.replace('/', '%2F')}", check=False).returncode:
            gh("api", "-X", "POST", f"repos/{repo}/labels", "-f", f"name={label}", "-f", "color=fbca04")
        gh("api", "-X", "POST", f"repos/{repo}/issues/{pr}/labels", "-f", f"labels[]={label}")


def handoff(out, pr, head_sha, mode, runner_temp, home, prev=""):
    """A review that stopped before it finished: the handoff a `/review continue` starts from
    ($RUNNER_TEMP/handoff), from the execution file or, after a timeout, the transcript Claude Code writes
    as it goes, and partial.md, what the "didn't complete" comment shows. A continued run that stops too
    merges its handoff onto the one it started from (`prev`). Model notes on the PR's own content, so
    redacted as the summary is; the artifact is public on a public repo."""
    previous = None
    if prev:
        try:
            previous = json.loads(Path(prev, "handoff.json").read_text())
        except (OSError, ValueError) as e:
            print(f"::warning::Couldn't read the earlier handoff, so this one starts afresh: {e}")
    if not isinstance(previous, dict):
        previous = None
    diff = Path("pr.diff").read_text(errors="replace") if Path("pr.diff").is_file() else ""
    # Tool paths are absolute, under the workspace this runs in.
    h = session.handoff(session.load([Path(runner_temp, EXECUTION), Path(home, ".claude/projects")]), os.getcwd(),
                        session.diff_paths(diff), previous)
    # A continued run that stopped before Claude said anything still passes on the handoff it started
    # from, which is still valid for this head.
    if not h.pop("seen_assistant") and previous is None:
        sys.exit("No Claude Code messages to hand off.")
    saved = Path(runner_temp, "handoff")
    saved.mkdir(parents=True, exist_ok=True)
    (saved / "handoff.json").write_text(redact(json.dumps({"pr": pr, "head_sha": head_sha, "mode": mode, **h})))
    (saved / "handoff.md").write_text(redact(session.handoff_md(h, pr, head_sha, mode)))
    Path("partial.md").write_text(redact(session.partial_md(h)))
    out["dir"] = str(saved)


def incomplete(out, repo, pr, run_url, why="a step failed", detail="", can_continue="false", refused="false"):
    """Anything failed: say so on the PR rather than leaving silence, with what the run had so far
    (partial.md). Repeated failures update one comment, found by its marker, so retries don't stack up
    comments. A refused /review continue has a marker of its own, so it never writes over a stopped
    review's comment (its partial results, and that it can still be continued by someone else)."""
    if refused == "true":
        # A /review continue that had nothing usable to continue from; `why` says what to do.
        marker = REFUSED
        text = f"{marker}\n`/review continue` didn't run: {why}. [Log]({run_url}).\n"
    else:
        marker = INCOMPLETE
        then = ("Comment `/review continue` to continue from its notes in a fresh session (kept for 7 days), or the "
                "same `/review` command to start over." if can_continue == "true"
                else "Comment the same `/review` command to retry.")
        text = (f"{marker}\nThe automated review didn't complete: {why}{f' ({detail})' if detail else ''}. "
                f"[Log]({run_url}). {then}\n")
    partial = Path("partial.md")
    if partial.is_file() and partial.stat().st_size:
        text += "\nWhat it had so far:\n\n" + partial.read_text()
    try:
        earlier = marked(api_list(f"repos/{repo}/issues/{pr}/comments"), marker)
    except (RuntimeError, ValueError):
        earlier = []
    if earlier:
        gh("api", "-X", "PATCH", f"repos/{repo}/issues/comments/{earlier[-1]}", "--input", "-",
           input=json.dumps({"body": text}))
    else:
        comment(repo, pr, text)


# ---------------------------------------------------------------- spend job

def record(out, repo, pr, github_run_id, github_run_attempt, usage="", gpu_usd="", box_usd="", approval=""):
    """Append one record for this run to the ledger issue (`spend` job, after the review and the live test),
    so the caps and the monitor read what each run cost: the review's tokens and the live test's GPUs and
    box session. Nothing spent, nothing written."""
    try:
        review = json.loads(usage) if usage else {}
    except ValueError:
        review = {}
    gpu, box = float(gpu_usd or 0), float(box_usd or 0)
    if not review and not gpu and not box:
        return
    cost = review.get("total_cost_usd")
    usd = round((cost if isinstance(cost, (int, float)) else 0) + gpu + box, 4)
    entry = {"pr": pr, "head": review.get("head_sha"), "mode": review.get("mode"), "run": github_run_id,
             "attempt": github_run_attempt, "usage": review, "gpu_usd": gpu, "box_usd": box, "approval": approval,
             "usd": usd}
    b64 = base64.b64encode(json.dumps(entry).encode()).decode()
    what = f"PR `#{pr}`" if pr else "Weekly improvement run"
    try:
        comment_on_issue(repo, issue(repo, "ledger"), f"{what}: ${usd:,.2f}.\n<!-- review-spend {b64} -->\n")
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        alert(repo, f"ledger-{github_run_id}-{github_run_attempt}",
              f"Run {github_run_id}'s spend (${usd:,.2f}) didn't reach the ledger, so the caps don't count it: "
              f"{safe(e)}")
        raise


# ---------------------------------------------------------------- monitor (a daily job; no model)

def main_commits(since):
    """Main's commits since `since`, newest first: {sha, parents, subject, body}."""
    text = run(["git", "log", f"--since={since}", "--format=%H%x1f%P%x1f%s%x1f%b%x1e"]).stdout
    return [dict(zip(("sha", "parents", "subject", "body"), rec.strip("\n").split("\x1f"), strict=False))
            for rec in text.split("\x1e") if rec.strip()]


def after_merge(commits, merge):
    """The commits newer than `merge` (all of them when it isn't in the list)."""
    index = next((i for i, c in enumerate(commits) if c["sha"] == merge), len(commits))
    return commits[:index]


def reverted(commits, shas, number, merge):
    """The commits on main after the merge that revert PR `number`: by the commit named in the message, or
    the default subject of a revert of a squash-merged PR."""
    subject = re.compile(rf'Revert ".*\(#{number}\)"')

    def names_a_commit(c):
        named = re.findall(r"This reverts commit ([0-9a-f]{7,40})", c["body"])
        return any(s.startswith(n) for n in named for s in shas)

    return [c for c in after_merge(commits, merge) if names_a_commit(c) or subject.match(c["subject"])]


def referenced(timeline, number, repo):
    """(signals, mentions) from the PR's cross-references: an issue labeled `bug`, a `type/bug` PR or one that
    says it fixes or reverts this PR, merged or by someone with write access, is a signal; any other reference a
    mention, the bot's own issues aside."""
    signals, mentions = [], []
    for event in timeline:
        if event.get("event") != "cross-referenced":
            continue
        source = (event.get("source") or {}).get("issue") or {}
        if (source.get("user") or {}).get("login") in BOT_USERS:
            continue
        ref, labels = source.get("number"), {x.get("name") for x in source.get("labels") or []}
        is_pr = "pull_request" in source
        claims = is_pr and number in map(int, FIXES.findall(source.get("body") or "")) and bool(
            (source.get("pull_request") or {}).get("merged_at")
            or permission(repo, (source.get("user") or {}).get("login")) in WRITE)
        if (is_pr and (claims or "type/bug" in labels)) or (not is_pr and "bug" in labels):
            how = "fixes or reverts" if claims else "reports a bug in"
            signals.append({"key": f"{number}:ref:{ref}",
                            "text": f"`#{ref}` ({safe(source.get('title'))}) {how} `#{number}`"})
        else:
            mentions.append(f"#{ref}")
    return signals, mentions


def safe(text):
    """Untrusted text for the status issue: redacted, on one line, unable to ping anyone, open a comment or
    mark another issue's timeline, in plain characters or in entity or backslash spellings of them."""
    return (redact(" ".join(str(text or "").split())).replace("\\", "\\\\").replace("&", "&amp;")
            .replace("@", "@\u200b").replace("<!--", "&lt;!--").replace("#", "#\u200b"))


def szz(commits, shas, number, files, bug_prs=(), since=""):
    """Signals for the fixes on main that changed lines PR `number` wrote (`shas` are its commits and merge,
    `files` the files it touched): `git blame` of each fix's parent, back to `since`, on the lines it removed
    or changed. A hit means the fix touched those lines, not that the PR broke them."""
    hits = []
    for c in commits:
        match = re.search(r"\(#(\d+)\)\s*$|Merge pull request #(\d+)", c["subject"])
        fix_pr = int(match[1] or match[2]) if match else None
        if fix_pr == number or not (FIX_SUBJECT.search(c["subject"]) or fix_pr in bug_prs):
            continue
        path = None
        for line in run(["git", "show", "--format=", "-U0", c["sha"]], check=False).stdout.split("\n"):
            if line.startswith("--- "):
                path = line[6:] if line.startswith("--- a/") else None
            hunk = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line) if path in files else None
            if not hunk or not int(hunk[2] or 1):
                continue
            blame = run(["git", "blame", "-l", "-s", *([f"--since={since}"] if since else []), "-L",
                         f"{hunk[1]},+{hunk[2] or 1}", f"{c['sha']}^", "--", path], check=False).stdout
            for offset, row in enumerate(blame.split("\n")):
                if row.lstrip("^").split(" ")[0] in shas:
                    name = f"`#{fix_pr}`" if fix_pr else f"`{c['sha'][:10]}`"
                    at = int(hunk[1]) + offset
                    hits.append({"key": f"{number}:szz:{c['sha'][:10]}:{path}:{at}",
                                 "text": f"{name} changed lines `#{number}` wrote at `{path}:{at}`"})
                    break
    return hits


def ci_regressions(repo, number, merge, parent):
    """Checks that passed on the merge's parent and failed on the merge."""
    def conclusions(sha):
        runs = api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100")["check_runs"]
        return {r["name"]: r.get("conclusion") for r in runs}

    try:
        before, after = conclusions(parent), conclusions(merge)
    except (RuntimeError, ValueError, KeyError, TypeError):
        return []
    return [{"key": f"{number}:ci:{name}",
             "text": f"`#{number}`: check `{safe(name)}` failed on `{merge[:10]}`, its parent passed"}
            for name, c in after.items() if c in ("failure", "timed_out") and before.get(name) == "success"]


def pr_signals(repo, pull, commits, bug_prs, window=""):
    """Everything known about what happened to a PR the bot approved: (signals, mentions), each signal
    {key, text} (key stable across days)."""
    number, merge = pull["number"], pull.get("merge_commit_sha")
    timeline = api_list(f"repos/{repo}/issues/{number}/timeline?per_page=100")
    signals, mentions = referenced(timeline, number, repo)
    if pull.get("merged_at") and merge:
        shas = {merge} | {c["sha"] for c in api_list(f"repos/{repo}/pulls/{number}/commits?per_page=100")}
        signals += [{"key": f"{number}:reverted:{c['sha'][:10]}",
                     "text": f"`#{number}` reverted by `{c['sha'][:10]}` ({safe(c['subject'])})"}
                    for c in reverted(commits, shas, number, merge)]
        files = set(run(["git", "diff", "--name-only", f"{merge}^", merge], check=False).stdout.split())
        signals += szz(after_merge(commits, merge), shas, number, files, bug_prs, window)
        parents = next((c["parents"].split() for c in commits if c["sha"] == merge), [])
        if parents:
            signals += ci_regressions(repo, number, merge, parents[0])
    else:  # open and approved: did the merge queue throw it out? (only GraphQL says)
        for page in graphql(EJECTIONS, repo, number):
            for e in page["data"]["repository"]["pullRequest"]["timelineItems"]["nodes"]:
                signals.append({"key": f"{number}:ejected:{e['createdAt']}",
                                "text": f"`#{number}` removed from the merge queue ({safe(e.get('reason'))})"})
    return signals, mentions


def monitor(out, repo, approvals=""):
    """The daily report (review-monitor.yml), with no model: what happened to the PRs the bot approved
    in the last 60 days, today's spend, the kill switch and new bug reports, written to the status
    issue's body. Signals not reported before go out as one comment, so subscribers get at most one
    notification a day. Untrusted titles go through `safe`, text reaches `gh` on stdin."""
    now = utcnow()
    window = (now - datetime.timedelta(days=MONITOR_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    number = issue(repo, "status")
    body = api(f"repos/{repo}/issues/{number}").get("body") or ""
    try:
        state = json.loads(base64.b64decode(MONITOR.search(body)[1]))
    except (ValueError, TypeError):
        state = {}
    last = state.get("last") or window
    approved = sorted({int(r["pr"]) for _, r in ledger(repo, window)
                       if r.get("approval") == "approve" and str(r.get("pr")).isdigit()}, reverse=True)
    commits = main_commits(window)
    bug_prs = {i["number"] for i in api_list(f"repos/{repo}/issues?state=closed&labels=type%2Fbug&since={window}"
                                             "&per_page=100")
               if (i.get("pull_request") or {}).get("merged_at")}
    rows, signals, linked, unread = [], [], set(), set()
    for pr in approved:
        try:
            pull = api(f"repos/{repo}/pulls/{pr}")
            reviews = api_list(f"repos/{repo}/pulls/{pr}/reviews?per_page=100")
            stood = any(r.get("state") == "APPROVED" and (r.get("user") or {}).get("login") in BOT_USERS
                        for r in reviews)
            if not stood or not (pull.get("merged_at") or pull.get("state") == "open"):
                continue
            found, mentions = pr_signals(repo, {**pull, "number": pr}, commits, bug_prs, window)
        except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
            rows.append(f"| `#{pr}` | ? | couldn't read: {safe(e).replace('|', '/')} |")
            unread.add(str(pr))
            continue
        signals += found
        linked |= {s["key"].split(":")[-1] for s in found if ":ref:" in s["key"]} | {m.lstrip("#") for m in mentions}
        cell = "; ".join(s["text"] for s in found) or "none"
        if mentions:
            cell += " (also mentioned in " + ", ".join(f"`{m}`" for m in mentions) + ")"
        merged = "merged" if pull.get("merged_at") else "open"
        rows.append(f"| `#{pr}` {safe(pull.get('title')).replace('|', '/')} | {merged} | {cell.replace('|', '/')} |")
    bugs = [i for i in api_list(f"repos/{repo}/issues?state=all&labels=bug&since={last}&per_page=100")
            if "pull_request" not in i and (i.get("created_at") or "") >= last and str(i["number"]) not in linked]
    new_prs = [p for p in api(f"repos/{repo}/pulls?state=all&sort=created&direction=desc&per_page=100") or []
               if (p.get("created_at") or "") >= last]
    used = spend(repo, now)
    reported = set(state.get("signals") or [])
    current = {s["key"]: s for s in signals}
    fresh = [current[k]["text"] for k in sorted(current) if k not in reported]
    head = [issue_marker("status"), "### Review bot status", f"Updated {now.strftime('%Y-%m-%d %H:%M')} UTC.", "",
            f"Spend today: ${used['day']:,.0f} of ${CAPS['day']:,}; "
            f"this month: ${used['month']:,.0f} of ${CAPS['month']:,}. "
            f"Approvals: **{'on' if approvals == 'on' else 'off'}** (`REVIEW_APPROVALS`).", "",
            f"#### PRs the bot approved, last {MONITOR_DAYS} days", "| PR | State | Signals |", "| --- | --- | --- |"]
    tail = ["", f"New PRs since the last report: {len(new_prs)}.",
            "#### New bug reports not linked to an approved PR"]
    tail += [f"- `#{i['number']}` {safe(i.get('title'))}" for i in bugs] or ["None."]
    kept = {k for k in reported if k.split(":")[0] in unread}  # a failed read isn't a signal gone
    saved = {"last": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "signals": sorted(set(current) | kept)}
    tail.append(f"\n<!-- review-monitor-state {base64.b64encode(json.dumps(saved).encode()).decode()} -->")
    while len("\n".join(head + rows + tail)) > 60_000 and rows:
        rows.pop()  # the oldest PRs go first: `approved` is newest first
    if fresh:  # first: the body below records these as reported, so a failed post goes out tomorrow
        comment_on_issue(repo, number, "**New signals on PRs the bot approved:**\n\n"
                         + "\n".join(f"- {text}" for text in fresh) + "\n")
    gh("api", "-X", "PATCH", f"repos/{repo}/issues/{number}", "--input", "-",
       input=json.dumps({"body": "\n".join(head + (rows or ["| | | none yet |"]) + tail) + "\n"}))


# ---------------------------------------------------------------- improve (a weekly job)

def scrub(text):
    """Comment or model text for the weekly improvement run: `safe`, clipped, and without names, links,
    markup or references, so it can't name anyone, load an image or mark a PR's timeline."""
    text = re.sub(r"(?i)[a-z][a-z0-9+.-]*://[^\s`]+|(?<![\w./-])//[^\s`]+|\bwww\.[^\s`]+", "[link]",
                  re.sub(r"@[\w-]+", "@someone", str(text or "")))
    return (safe(re.sub(r"(?i)\b(GH-)(?=\d)", "\\1\u200b", text)).replace("<", "<\u200b")
            .replace("](", "]\u200b("))[:IMPROVE_TEXT]


def monday(now):
    """The date this week started, which names its issue."""
    return (now - datetime.timedelta(days=now.weekday())).strftime("%Y-%m-%d")


def pr_evidence(repo, pr, since, records):
    """What happened on PR `pr` since `since`, for the improvement run: its runs from the ledger (cost, outcome),
    the findings a later review judged moot, its reviews, inline comments and conversation. Logins become a role.
    A finding someone resolved stays however old it is: GitHub doesn't say when a thread was resolved."""
    author = (api(f"repos/{repo}/pulls/{pr}").get("user") or {}).get("login")
    inline, conversation = discussion(repo, pr)
    reviews = [{"user": (r.get("user") or {}).get("login"), "user_type": (r.get("user") or {}).get("type"),
                "created_at": r.get("submitted_at"), "url": r.get("html_url"), "body": r.get("body")}
               for r in api_list(f"repos/{repo}/pulls/{pr}/reviews?per_page=100")
               if (r.get("user") or {}).get("login") not in BOT_USERS]

    def kept(comments):
        rows = []
        for c in comments:
            mine = c["user"] in BOT_USERS
            if not c.get("body") or (c.get("user_type") == "Bot" and not mine):
                continue
            if (c.get("created_at") or "") >= since or (mine and c.get("resolved")):
                rows.append({**{k: c[k] for k in ("id", "path", "line", "in_reply_to_id", "resolved", "outdated",
                                                  "url") if k in c},
                             "role": "bot" if mine else "author" if c["user"] == author else "other",
                             "body": scrub(c["body"])})
        return rows[-IMPROVE_ITEMS:]

    runs = [{"mode": r.get("mode"), "outcome": (r.get("usage") or {}).get("outcome"),
             "subtype": (r.get("usage") or {}).get("subtype"), "turns": (r.get("usage") or {}).get("num_turns"),
             "usd": r.get("usd"), "gpu_usd": r.get("gpu_usd"), "box_usd": r.get("box_usd"),
             "approval": r.get("approval"), "url": f"https://github.com/{repo}/actions/runs/{r.get('run')}"}
            for created, r in records if created >= since and str(r.get("pr")) == str(pr)]
    moot = [{"ref": scrub(f.get("ref")), "why": scrub(f.get("why")), "url": c["url"]}
            for c in conversation
            if c["user"] in BOT_USERS and (c["body"] or "").startswith(render_summary.MARKER)
            and (c.get("created_at") or "") >= since
            for f in (render_summary.saved_state(c["body"]) or {}).get("moot", [])]
    return {"runs": runs, "moot": moot, "reviews": kept(reviews), "inline": kept(inline),
            "conversation": kept(conversation)}


def improve_inputs(out, repo, key_set):
    """The evidence for the weekly improvement run (review-improve.yml), in improve-evidence/: one file per PR the
    bot reviewed in the last three weeks with anything from the last 7 days, and status.json, the status
    issue's signals and alerts. `run` stays false without an API key, past a spend cap, once this week's run
    has finished (a failed one can be retried by hand), or with nothing to read."""
    out["run"] = "false"
    if key_set != "true":
        print("::warning::ANTHROPIC_API_KEY is not set; skipping the improvement run.")
        return
    stop = over_cap(repo)  # its alert is best effort: this job's token can only read issues
    if stop:
        print(f"::notice::{stop}")
        return
    now = utcnow()
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    records = ledger(repo, (now - datetime.timedelta(days=21)).strftime(stamp))
    if any(r.get("mode") == "improve" and (r.get("usage") or {}).get("outcome") == "success"
           and created >= f"{monday(now)}T00:00:00Z" for created, r in records):
        print("::notice::This week's improvement run has already finished.")
        return
    since = (now - datetime.timedelta(days=7)).strftime(stamp)
    newest_first = dict.fromkeys(int(r["pr"]) for _, r in reversed(records) if str(r.get("pr")).isdigit())
    files = {}
    for pr in list(newest_first)[:IMPROVE_ITEMS]:
        evidence = pr_evidence(repo, pr, since, records)
        if any(evidence.values()):
            files[f"{pr}.json"] = {"pr": pr, **evidence}
    status = [{"url": c.get("html_url"), "text": scrub(line)}
              for i in issues(repo, "status")
              for c in api_list(f"repos/{repo}/issues/{i['number']}/comments?since={since}&per_page=100")
              if (c.get("user") or {}).get("login") in BOT_USERS
              for line in (c.get("body") or "").splitlines() if line.strip()]
    if status:
        files["status.json"] = status[-IMPROVE_ITEMS:]
    if not files:
        print("::notice::The bot did nothing this week; no improvement run.")
        return
    Path("improve-evidence").mkdir()
    for name, value in files.items():
        write_json(f"improve-evidence/{name}", value)
    out.update(run="true", schema=json.dumps(IMPROVE_SCHEMA, separators=(",", ":")))


def improve_post(out, repo, github_run_id, github_run_attempt, output="", usage=""):
    """Record what the improvement run cost, first, so a failed post can't lose it; then open the week's issue
    from its structured output, redacted, on stdin. Evidence links stay inside this repository (no `..`, which
    would climb out of it) and go through redirect.github.com, which keeps them off the PRs' timelines. An empty
    list opens nothing; a run that returned no list fails. Never touches code, PRs or the bot."""
    record(out, repo, "", github_run_id, github_run_attempt, usage=usage)
    try:
        items = json.loads(output)["improvements"]
    except (ValueError, KeyError, TypeError):
        sys.exit("The improvement run returned no list, so no issue was opened.")
    if not items:
        print("::notice::Nothing worth raising this week; no issue opened.")
        return
    link = re.compile(rf"https://github\.com/({re.escape(repo)}/[\w./?=#&%-]+)")
    parts = ["What the review bot could improve about itself, most useful first. It changes nothing on its own."]
    for n, item in enumerate(items, 1):
        links = [f"- https://redirect.github.com/{m[1]}" for m in map(link.fullmatch, item["evidence"])
                 if m and ".." not in urllib.parse.unquote(m[1])]
        parts.append("\n".join([f"### {n}. {scrub(item['title'])}", "", f"**Change:** {scrub(item['change'])}", "",
                                f"**Where:** {scrub(item['where'])}", "", "**Evidence:**", *links]))
    body = json.dumps({"title": f"Review bot: what to improve, week of {monday(utcnow())}",
                       "body": redact("\n\n".join(parts)) + "\n"})
    gh("api", "-X", "POST", f"repos/{repo}/issues", "--input", "-", input=body)


# ---------------------------------------------------------------- dispatcher

def main(steps):
    """Run the step `sys.argv[1]` names, with its parameters from the environment, and write what it put
    in `out` to $GITHUB_OUTPUT, even when it fails. An empty variable takes the parameter's default."""
    step = steps[sys.argv[1]]
    kwargs = {}
    for name, p in list(inspect.signature(step).parameters.items())[1:]:
        value = os.environ.get(name.upper(), "")
        if value == "" and p.default is not p.empty:
            continue
        if name.upper() not in os.environ:
            raise ValueError(f"{sys.argv[1]}: ${name.upper()} is not set")
        kwargs[name] = value if p.annotation is p.empty or value == "" else p.annotation(value)
    out = {}
    try:
        step(out, **kwargs)
    finally:
        with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a", encoding="utf-8",
                  errors="surrogateescape") as f:
            for key, value in out.items():
                value = str(value)
                if "\n" in value:
                    delim = f"EOF_{secrets.token_hex(8)}"
                    f.write(f"{key}<<{delim}\n{value}\n{delim}\n")
                else:
                    f.write(f"{key}={value}\n")


STEPS = {"preflight": preflight, "resolve": resolve, "restore": restore, "collect": collect, "salvage": salvage,
         "usage": usage, "post": post, "hide": hide, "labels": labels, "handoff": handoff, "incomplete": incomplete,
         "record": record, "monitor": monitor, "improve_inputs": improve_inputs, "improve_post": improve_post}

if __name__ == "__main__":
    main(STEPS)
