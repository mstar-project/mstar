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
import time
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
# The bot's REST logins (render_summary.BOTS and POSTING_BOTS say whose count for what).
BOT_USERS = [f"{b}[bot]" for b in render_summary.BOTS]
POSTING_BOT_USERS = [f"{b}[bot]" for b in render_summary.POSTING_BOTS]
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
      nodes { isResolved isOutdated comments(first: 1) {
        nodes { databaseId author { login } body path line lastEditedAt } } }
    } } } }"""
# Every merged PR with its live approvals. Each author's __typename separates the bot from a person who took its
# name, which GraphQL's bare logins (no `[bot]`) can't.
MERGED_APPROVALS = """query($owner: String!, $name: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes { number mergedAt reviews(states: APPROVED, first: 100) { nodes { author { __typename login } } } }
    } } }"""
# What starts the status issue's body, how the bot finds its own issue.
STATUS_ISSUE = "<!-- review-bot-status -->"
# The monitor's state in the status issue's body (signals already reported), base64 JSON.
MONITOR = re.compile(r"<!-- review-monitor-state ([A-Za-z0-9+/=]+) -->")
MONITOR_DAYS = 60
# How a PR's body says it fixes or reverts another, and the subjects `szz` treats as fixes.
FIXES = re.compile(r"\b(?:fix(?:e[sd])?|revert(?:s|ed)?)\s+#(\d+)\b", re.IGNORECASE)
# A fix-word, unless what follows is a typo, or all it fixes is a comment, docs or lint: such a word ending the
# subject, before a `.` or a PR number.
FIX_SUBJECT = re.compile(r"\b(?:fix(?:e[sd]|ing)?|bugs?|bugfix(?:es)?|revert(?:s|ed|ing)?)\b"
                         r"(?!:?\s+(?:a\s+|the\s+)?(?:typos?\b|(?:comments?|docs?|lint|format\w*)\s*\.?\s*"
                         r"(?:\(#\d+\)\s*)?$))",
                         re.IGNORECASE)
# author_association values: the team, and the team plus anyone with a merged PR (see the workflow's header).
TEAM = ("OWNER", "MEMBER", "COLLABORATOR")
CONTRIBUTORS = (*TEAM, "CONTRIBUTOR")
# The improvement run, every other day: the days of evidence it reads (its cadence, so issues don't repeat it: only
# comments and reviews dated inside it, and the bot's resolved findings on a PR active inside it), how much a reader
# needs (PRs read, and comments kept per PR; characters of a comment), and what the model returns.
IMPROVE_DAYS = 2
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
    can hold a token. What it prints is exactly its bytes: those that aren't UTF-8 pass through
    (surrogateescape), and with no newline translation a bare \\r inside a source line stays inside it."""
    r = subprocess.run(argv, input=input if input is None else input.encode("utf-8", "surrogateescape"),
                       capture_output=True, timeout=timeout, check=False)
    r.stdout, r.stderr = r.stdout.decode("utf-8", "surrogateescape"), r.stderr.decode("utf-8", "surrogateescape")
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


UNREADABLE = (RuntimeError, ValueError)  # gh failing, or printing what isn't JSON


def read(fetch, *args):
    """`fetch(*args)` (api, api_list, ...), or None when it can't be read, for a caller that has a
    no-data path. Only UNREADABLE: anything else, a KeyError say, is a bug in how the result is used."""
    try:
        return fetch(*args)
    except UNREADABLE:
        return None


def graphql(query, repo, pr=None):
    """The pages of a paginated GraphQL query on the repository, or on PR `pr`."""
    owner, name = repo.split("/")
    number = () if pr is None else ("-F", f"pr={pr}")
    return api("graphql", "--paginate", "--slurp", "-F", f"owner={owner}", "-F", f"name={name}", *number,
               "-f", f"query={query}")


def log_value(value):
    """Untrusted `value` for a log line, where it can't run a workflow command: on one line, as JSON (a newline would
    start a new command), and without `##[`, which the runner's older command parser finds anywhere in a line."""
    return json.dumps(value).replace("##[", "# #[")


def write(path, text):
    """Write `text` as `gh` printed it, bytes that aren't UTF-8 included."""
    Path(path).write_text(text, encoding="utf-8", errors="surrogateescape")


def write_json(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def marked(comments, marker):
    """The ids of the bot's own comments that start with `marker`."""
    return [c["id"] for c in comments
            if (c.get("user") or {}).get("login") in BOT_USERS and (c.get("body") or "").startswith(marker)]


def last_summary(conversation):
    """The bot's last summary among the PR's conversation comments, or None."""
    summaries = [c for c in conversation
                 if c["user"] in BOT_USERS and (c["body"] or "").startswith(render_summary.MARKER)]
    return summaries[-1]["body"] if summaries else None


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
    return (read(api, f"repos/{repo}/collaborators/{login}/permission") or {}).get("permission", "none")


def head_trusted(repo, pull):
    """Whether the PR's code is held by someone `repo` trusts: a branch of `repo` itself, or a fork whose owner has
    write access. An author with write access can open a PR from anyone's fork, so the author alone doesn't say. A
    head repository GitHub no longer shows means no."""
    head = (pull.get("head") or {}).get("repo") or {}
    owner = (head.get("owner") or {}).get("login")
    if head.get("full_name") == repo:
        return True
    return bool(owner) and permission(repo, owner) in WRITE


def redact(text, exact=()):
    """`text` without the exact values in `exact`, then without credential-shaped strings."""
    for value in exact:
        if value:
            text = text.replace(value, "[REDACTED]")
    for pattern in REDACT:
        text = pattern.sub("[REDACTED]", text)
    return text


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def status_issue(repo):
    """The bot's status issue, or None before the monitor, its one writer, has made it: the issue a bot login
    created whose body starts with STATUS_ISSUE."""
    for login in POSTING_BOT_USERS:
        creator = login.replace("[", "%5B").replace("]", "%5D")
        for i in api_list(f"repos/{repo}/issues?state=all&creator={creator}&per_page=100"):
            if (i.get("user") or {}).get("login") in POSTING_BOT_USERS \
                    and (i.get("body") or "").startswith(STATUS_ISSUE):
                return i
    return None


def ensure_label(repo, label):
    """Create the label if the repository doesn't have it yet."""
    if gh("api", f"repos/{repo}/labels/{label.replace('/', '%2F')}", check=False).returncode:
        gh("api", "-X", "POST", f"repos/{repo}/labels", "-f", f"name={label}", "-f", "color=fbca04")


def comment_on_issue(repo, number, body):
    """Post `body` as a comment on an issue, from stdin."""
    gh("issue", "comment", str(number), "--repo", repo, "--body-file", "-", input=body)


# ---------------------------------------------------------------- preflight job

AUTHOR_NEEDS_WRITE = ("`/review` and `/review nogpu` take a PR author with write access; "
                      "comment `/review light` instead.")


def preflight(out, repo, pr, key_set, command, actor, author, assoc, skip_label, team_turns="2000",
              self_turns="40"):
    """Whether a comment starts a review, in which mode (`command`, one of the workflow's fixed words)
    and with which turn cap; a request it turns down for a reason the commenter can act on gets a
    one-line reply. Write access is checked against the actor's permission; a contributor may run
    `/review light` or `/review continue` on their own PR, and only then is `non_write_user` set. A nogpu
    or full review also takes an author with write access (author_association can't tell write from read);
    a full one runs the PR's code, so also a head repository head_trusted accepts, and a PR that can't be
    read means no. A /review continue is held to its mode's rule in restore."""
    if key_set != "true":
        print("::warning::ANTHROPIC_API_KEY is not set; skipping the review.")
        out["run"] = "false"
        return
    if command not in ("light", "nogpu", "full", "continue"):
        out["run"] = "false"
        return
    mode = "" if command == "continue" else command  # the stopped review's mode, from its handoff

    perm = permission(repo, actor)
    if perm in WRITE:
        if command in ("nogpu", "full") and permission(repo, author) not in WRITE:
            print(f"::notice::{author} lacks write access; this review needs it.")
            out["run"] = "false"
            reply(repo, pr, AUTHOR_NEEDS_WRITE)
            return
        if command == "full" and not head_trusted(repo, read(api, f"repos/{repo}/pulls/{pr}") or {}):
            print("::notice::The PR's head repository lacks write access; a full review runs its code.")
            out["run"] = "false"
            reply(repo, pr, "`/review` runs this PR's code on GPUs, so it takes a head branch on this repository or "
                            "a fork owned by someone with write access; comment `/review nogpu` for the review "
                            "without the live test.")
            return
        out.update(run="true", mode=mode, turns=team_turns)
        return
    own = assoc in CONTRIBUTORS and actor != "" and actor == author and skip_label != "true"
    if own and command in ("light", "continue"):
        print(f"::notice::{actor} has '{perm}' access; reviewing their own PR as its author ({assoc}).")
        out.update(run="true", mode=mode, non_write_user=actor, turns=self_turns)
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
        print(f"::warning::Couldn't reply: {log_value(str(e))}")


# ---------------------------------------------------------------- review job: before the review

def react(out, repo, comment_id):
    """React 👀 to the /review comment (best effort), so a comment-triggered run isn't invisible from the
    PR. Its own step, because it takes the app token, which can't read the handoff `resolve` fetches."""
    try:
        gh("api", "-X", "POST", f"repos/{repo}/issues/comments/{comment_id}/reactions", "-f", "content=eyes")
    except RuntimeError as e:
        print(f"::warning::Couldn't react to the comment: {log_value(str(e))}")


def resolve(out, repo, pr, mode, runner_temp, non_write=""):
    """The PR's head, its title and the review's settings. A /review continue (`mode` empty) takes the
    mode of the review it continues (restore). The title is untrusted; it only ever reaches the prompt."""
    pull = api(f"repos/{repo}/pulls/{pr}")
    head_repo = (pull["head"].get("repo") or {}).get("full_name")
    if not head_repo:
        print("::warning::The PR's head repository no longer exists; can't check it out.")
        out["why"] = "its head repository (the fork) no longer exists"
        sys.exit(1)
    out.update(head_sha=pull["head"]["sha"], head_repo=head_repo, title=pull["title"])
    # Someone with write access: a timeout that only stops a runaway, like the turn cap.
    out["timeout"] = 25 if non_write else 300
    if mode == "":
        try:
            restore(out, repo, pr, pull["head"]["sha"], runner_temp, non_write,
                    (pull.get("user") or {}).get("login", ""))
        except Exception:  # a refusal too, so it can't overwrite the stopped review's comment
            out.update(why="the saved handoff couldn't be read; try `/review continue` again", refused="true")
            raise
    else:
        out["mode"] = mode if mode in ("nogpu", "full") else "light"  # sets the prompt's reporting bar


def restore(out, repo, pr, head_sha, runner_temp, non_write="", author=""):
    """/review continue: fetch the handoff a stopped review saved (handoff) into $RUNNER_TEMP/prev, and
    that review's mode. The newest unexpired artifact for this PR wins, and only
    from this workflow on issue_comment, a trigger that runs the base repository's copy of it: a fork's
    own workflow can upload an artifact with any name. Only while the stopped review's "didn't
    complete" comment is still open, since any later run that finishes rewrites it as resolved. The
    handoff's values are checked before use; a nogpu one takes what `/review nogpu` takes: a commenter
    and an author with write access."""
    name = f"claude-review-handoff-{pr}"

    def fail(why):
        out.update(why=why, refused="true")
        print(f"::notice::{why}")
        sys.exit(1)

    comments = read(api_list, f"repos/{repo}/issues/{pr}/comments")
    if comments is None:
        fail("the PR's comments couldn't be read; try `/review continue` again")
    if not marked(comments, INCOMPLETE):
        fail("there is no stopped review of this PR to continue: none stopped part-way, or a later run already "
             "finished (handoffs are kept for 7 days); ask for a new review with `/review light` or `/review`")
    listing = read(api, f"repos/{repo}/actions/artifacts?name={name}&per_page=30") or {}
    artifacts = [a for a in listing.get("artifacts") or [] if not a["expired"]]
    found = None
    for a in reversed(sorted(artifacts, key=lambda a: a["created_at"])):
        r = read(api, f"repos/{repo}/actions/runs/{a['workflow_run']['id']}")
        if r is None:
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
    if str(meta.get("pr")) != pr:
        fail("the saved handoff is for another PR")
    if str(meta.get("head_sha")) != head_sha:
        fail("the PR has new commits since that review stopped; ask for a new review of the current head with "
             "`/review light` or `/review`")
    if mode == "full":
        fail("a `/review` that stopped can't be continued (its live test never started); comment `/review` again")
    if mode not in ("light", "nogpu"):
        fail("the saved handoff is malformed")
    if mode == "nogpu" and (non_write or permission(repo, author) not in WRITE):
        fail("continuing a nogpu review takes write access for the commenter and the PR's author, like "
             "`/review nogpu`; ask for a new review with `/review light`")
    out.update(dir=str(saved), mode=mode)


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


def file_patches(diff):
    """{"diff --git" line: the file's lines} of a unified diff. Lines end only at \\n: a form feed or
    U+2028 inside a source line (str.splitlines breaks on those) could fake a header and hide the code
    around it. A header that repeats adds to its file, never replaces it: git prints one twice for a
    type change (the file deleted, then created)."""
    files, current = {}, None
    for line in re.findall(r".*\n|.+", diff):
        if line.startswith("diff --git "):
            current = files.setdefault(line, [])
            current.append(line)
        elif current is not None:
            current.append(line)
    return files


def patch_key(lines):
    """A file's patch as a rebase leaves it: no `@@` line numbers (the function name after them stays, so a
    change moved to another function counts), no `similarity index` or `dissimilarity index` line, and
    from the `index` line only the new blob when the file has no hunks (binary, a pure rename), and nothing
    when it has: a rebase moves those blobs, and what a rename is similar to. Each `diff --git` section is
    judged alone, since a type change has two."""
    starts = [i for i, line in enumerate(lines) if line.startswith("diff --git ")] + [len(lines)]
    key = []
    for start, end in zip(starts, starts[1:], strict=False):
        section = lines[start:end]
        hunks = any(line.startswith("@@") for line in section)
        for line in section:
            if line.startswith("@@"):
                key.append(re.sub(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", "@@", line))
            elif line.startswith("index "):
                if not hunks:
                    key.append(line.split("..")[-1])
            elif not line.startswith(("similarity index ", "dissimilarity index ")):
                key.append(line)
    return key


DROPPED = "=== dropped since the last review: these files as they were at the last review ===\n"


def interdiff(old, new):
    """What differs between `old`, a PR's diff at the last review, and `new`, its diff now: each file that
    differs, compared with line numbers stripped, as `new` has it, numbered as pr.diff is, and, after DROPPED,
    as `old` had it, unnumbered as nothing there can be commented on; files only on one side appear on that
    side. Empty means the PR's own changes are the same, whatever a rebase moved around them. Coarse by design:
    a re-review after a rebase can read a whole file's patch when only part of it changed."""
    before, after = file_patches(old), file_patches(new)
    changed = [f for f in after if patch_key(after[f]) != patch_key(before.get(f, []))]
    dropped = [f for f in before if patch_key(before[f]) != patch_key(after.get(f, []))]
    text = session.numbered("".join(line for f in changed for line in after[f]))
    return text + (DROPPED + "".join(line for f in dropped for line in before[f]) if dropped else "")


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
                         "author_association": c.get("author_association"),
                         "created_at": c.get("created_at"), "url": c.get("html_url"),
                         **threads.get(c.get("in_reply_to_id") or c["id"], {}), "mine": user in POSTING_BOT_USERS})
    # The conversation tab: replies to the summary, and the author's answers.
    conversation = [{"id": c["id"], "user": (c.get("user") or {}).get("login"),
                     "user_type": (c.get("user") or {}).get("type"), "created_at": c.get("created_at"),
                     "author_association": c.get("author_association"), "url": c.get("html_url"),
                     "body": c.get("body")}
                    for c in api_list(f"repos/{repo}/issues/{pr}/comments")]
    return comments, conversation


def collect(out, repo, pr, head_sha, mode, prev=""):
    """The review's inputs, in the workspace: pr.diff (its hunks numbered, session.numbered), pr-body.md,
    pr-comments.json (with each thread's
    resolved and outdated state, and `mine` on the bot's own), pr-conversation.json, a re-review's
    last-review.json with since.diff and since-commits.txt or interdiff.diff (rereview_inputs),
    stacked-above.json and above-<n>.diff, and open-issues.json; and the diff's length and the summary's
    schema as outputs. The PR's text and comments are untrusted: they only ever reach files the prompt
    calls untrusted. In full mode, whose review steers a session with a shell, only the team's comments and
    issue titles (from_the_team) reach those files, and always the bot's own."""
    if prev:
        shutil.copyfile(Path(prev, "handoff.md"), "handoff.md")
    pull = api(f"repos/{repo}/pulls/{pr}")
    # The commit `resolve` fixed, not the PR's live head: three dots, as the PR's own diff. A failure
    # (GitHub can give up on generating a very large one) fails this step: the review reports it didn't run.
    diff = gh("api", "-H", "Accept: application/vnd.github.diff",
              f"repos/{repo}/compare/{pull['base']['sha']}...{head_sha}").stdout
    write("pr.diff", session.numbered(diff))
    write("pr-body.md", (pull["body"] or "") + "\n")

    comments, conversation = discussion(repo, pr)
    write_json("pr-comments.json", from_the_team(comments) if mode == "full" else comments)
    write_json("pr-conversation.json", from_the_team(conversation) if mode == "full" else conversation)

    if not prev:
        rereview_inputs(out, repo, pull, head_sha, diff, conversation)
    if mode == "full":
        since_live_test(repo, head_sha, conversation)
    write_stacked_above(out, repo, pull, head_sha)
    write_open_issues(repo, mode)
    out["lines"] = diff.count("\n")
    # The summary's fields, from main, for --json-schema.
    out["schema"] = json.dumps(render_summary.SCHEMA, separators=(",", ":"))


def from_the_team(comments):
    """Of `comments` (`discussion`), those by the team or the bot (its findings, which a re-review judges)."""
    return [c for c in comments if c["user"] in POSTING_BOT_USERS or c.get("author_association") in TEAM]


def rereview_inputs(out, repo, pull, head_sha, diff, conversation):
    """A re-review starts from this bot's last summary and reviews only what changed since (review-prompt.md,
    "Re-review"): the commits since, when that commit is still an ancestor of the head and none of them is
    a merge (since.diff, since-commits.txt), and otherwise the interdiff, each file whose patch differs from
    its patch then, new and old (interdiff.diff), so a merge of main or a force-push isn't read as the
    author's change. read_state says when it reviews the whole PR again instead. The state is read only
    from the bot's own summaries: comments that start with the summary's marker, in which model text can't
    open an HTML comment. A diff GitHub won't give (too large, say) means a full review, never an empty
    since.diff."""
    summary = last_summary(conversation)
    state = summary and render_summary.read_state(summary)
    last = (state or {}).get("head")
    since = f"repos/{repo}/compare/{last}...{head_sha}"
    status = ((read(api, since) or {}).get("status") or "") if last else ""
    listing = None  # since-commits.txt, when the range can be read as the author's commits alone
    if last and (status == "ahead" or last == head_sha):
        pages = read(api, since, "--paginate", "--slurp")
        commits = None if pages is None else [c for page in pages for c in page["commits"]]
        listing = None if commits is None or any(len(c["parents"]) > 1 for c in commits) else oneline(commits)
    r = None
    if listing is not None:
        r = gh("api", "-H", "Accept: application/vnd.github.diff", since, check=False)
        if r.returncode == 0:
            write("since.diff", session.numbered(r.stdout, left=False))
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


def untouched(earlier):
    """The earlier review's findings outside the diff (`earlier`, what render_summary.carried carries over)
    whose file no commit since that review touched, by the files since.diff or interdiff.diff touches
    (rereview_inputs; not interdiff's dropped part, which holds the old patches as written, unnumbered). With
    neither (no last-review.json, or a /review continue) every one counts as untouched, since that is unknown. Such a
    finding has no thread to show its lines changed, so the approval reads a re-review's `fixed` or `moot` on it as
    `disputed` (gpu_review.for_approval), and the saved state keeps it (render_summary.render)."""
    since = next((Path(p) for p in ("since.diff", "interdiff.diff") if Path(p).exists()), None)
    # As bytes: the diff holds what the commits hold, which needn't be UTF-8, and a CR in a line isn't its end.
    text = since.read_bytes().decode("utf-8", "replace") if since else ""
    touched = touched_files(text.partition(DROPPED)[0])
    return [f for f in (earlier or {}).get("outside_diff", []) if f["location"].partition(":")[0] not in touched]


GIT_HEADER = re.compile(r'diff --git ("(?:[^"\\]|\\.)*"|\S+) ("(?:[^"\\]|\\.)*"|\S+)')


def touched_files(diff_text):
    """Every file a diff touches, on either side: changed, added, deleted (`+++ /dev/null`), renamed from or to,
    and binary or mode-only changes, by their `diff --git` line. Unlike session.diff_paths, the files a review
    can open, a file that is gone counts. Lines end only at \\n, as in session.diff_paths. A binary or mode-only
    change to a path with a space isn't caught (git doesn't quote spaces in its `diff --git` line, so the line can't
    be split), which fails closed: its finding reads as untouched."""
    lines = diff_text.split("\n")
    paths = set()
    for before, line in zip([""] + lines, lines, strict=False):
        sides = ()
        if before.startswith("--- ") and line.startswith("+++ "):
            sides = ((before[len("--- "):], "a/"), (line[len("+++ "):], "b/"))
        elif line.startswith(("rename from ", "rename to ")):
            sides = ((line.split(" ", 2)[2], ""),)
        elif m := GIT_HEADER.fullmatch(line):
            sides = ((m[1], "a/"), (m[2], "b/"))
        for raw, prefix in sides:
            path = session.unquoted(raw)
            if path and path.startswith(prefix):
                paths.add(path[len(prefix):])
    return paths


def last_live_test(comments):
    """The bot's last conclusive live-test comment among `comments` ({"user": login, "body": text}), {"sha",
    "verdict", "body"}, or None: any but an `unavailable` one (no GPUs, or a box session that ended without a
    verdict), which says nothing, so a fail before it still stands. A verdict outside pass, partial and fail
    (`none`: the session wrote no valid one) is conclusive too, and counts as a failure (gpu_review.carry). Only
    its first line, the marker, is read: the report below it is the box's text."""
    last = None
    for c in comments:
        m = LIVE_TEST.fullmatch((c["body"] or "").split("\n")[0])
        if m and c["user"] in BOT_USERS and m[2] != "unavailable":
            last = {"sha": m[1], "verdict": m[2], "body": c["body"]}
    return last


def passed_on_ancestor(repo, live, head_sha):
    """Whether the live test `live` (last_live_test, or None) passed on an ancestor of `head_sha`. A compare
    that can't be read counts as no."""
    if not live or live["verdict"] != "pass":
        return False
    status = (read(api, f"repos/{repo}/compare/{live['sha']}...{head_sha}") or {}).get("status")
    return status in ("ahead", "identical")


def since_live_test(repo, head_sha, conversation):
    """Full mode: since-live-test.diff and since-live-test-commits.txt, the diff and commits since the PR's
    last live test, when it passed on an ancestor of the head (last_live_test, passed_on_ancestor), for the
    review's live-test call (review-prompt.md, Re-review step 4). Without both files (GitHub refused
    either) the review judges the whole PR."""
    live = last_live_test(conversation)
    if not passed_on_ancestor(repo, live, head_sha):
        return
    tested = f"repos/{repo}/compare/{live['sha']}...{head_sha}"
    try:
        live_diff = gh("api", "-H", "Accept: application/vnd.github.diff", tested).stdout
        commits = oneline(c for page in api(tested, "--paginate", "--slurp") for c in page["commits"])
    except UNREADABLE:
        return
    write("since-live-test.diff", live_diff)
    write("since-live-test-commits.txt", commits)


def write_stacked_above(out, repo, pull, head_sha):
    """stacked-above.json and above-<n>.diff. A failed fetch leaves none of these files, so a finding said to
    be fixed above stays open, and the review still runs."""
    above = read(stacked_above, repo, pull, head_sha)
    if above is None:
        for f in Path(".").glob("above-*.diff"):
            f.unlink()
    else:
        write_json("stacked-above.json", above)
        out["above"] = json.dumps(above, separators=(",", ":"))


def write_open_issues(repo, mode):
    """The open issues, [{number, title}], so a latent bug on main can name the one that tracks it, in full mode
    only the team's. Titles are user text: the model reads them as untrusted data. A failed fetch leaves no
    file, and the review says it couldn't check."""
    issues = read(api_list, f"repos/{repo}/issues?state=open&per_page=100")
    if issues is not None:
        write_json("open-issues.json", [{"number": i["number"], "title": i["title"]} for i in issues
                                        if "pull_request" not in i
                                        and (mode != "full" or i.get("author_association") in TEAM)])


# ---------------------------------------------------------------- review job: after the review

def last_result(messages):
    """The final `result` message of an execution file's messages, {} when there is none."""
    return ([m for m in messages if m.get("type") == "result"] or [{}])[-1]


def returned(result):
    """What a `result` message returned, when the run finished and returned a review; otherwise None."""
    output = result.get("structured_output")
    ok = result.get("subtype") == "success" and result.get("is_error") in (None, False) and isinstance(output, dict)
    return output if ok else None


def finished_review(runner_temp):
    """The review a finished run returned, from the action's execution file; None if it returned none. Read here,
    not passed through an env var: the kernel refuses to start a step with one over about 128 KB, which a review with
    many inline findings can pass, and the finished review would be lost."""
    return returned(last_result(json.loads(Path(runner_temp, EXECUTION).read_text())))


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
    result = last_result(messages)
    out["detail"] = f"{result.get('num_turns', '?')} turns"
    output = returned(result)
    if output is not None:
        out["structured_output"] = json.dumps(output, separators=(",", ":"), ensure_ascii=False)
    else:
        out["why"] = {"error_max_turns": "it ran out of turns",
                      "error_during_execution": "Claude Code stopped on an error",
                      "success": "it finished without returning a summary"}.get(
            result.get("subtype"), "Claude Code ended without a result")


def usage(out, pr, head_sha, mode, outcome, runner_temp, home, github_step_summary, prev=""):
    """What the run used, from the execution file or, after a timeout, the session transcript: a table in
    the job summary. Never posted on the PR, and never fails: a run that couldn't be read says why in its
    `Error` row."""
    record = {"pr": pr, "head_sha": head_sha, "mode": mode, "continue": prev != "", "outcome": outcome}
    try:
        record.update(session.usage(session.load([Path(runner_temp, EXECUTION), Path(home, ".claude/projects")])))
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"::warning::Couldn't read the review's usage: {log_value(record['error'])}")
    try:
        with open(github_step_summary, "a") as f:
            f.write(session.usage_table(record))
    except Exception as e:
        print(f"::warning::Couldn't write the job summary: {e}")


def earlier_review(repo, pr):
    """(state, answered): the state of the last review, what render_summary.carried carries over: what
    last-review.json showed this run, or the last summary's when it had no last-review.json (no earlier
    diff, a partial review, or a /review continue); and whether a person other than the author has
    commented since that review. Not knowing reads as not answered; not being able to read the state fails.
    Answering takes someone with write access (the bar `/review` uses) who isn't the author, and a comment that
    isn't a `/review` command."""
    path, state = Path("last-review.json"), None
    if path.exists():
        state = json.loads(path.read_text())
    try:
        conversation = api_list(f"repos/{repo}/issues/{pr}/comments")
        summaries = [c for c in conversation if (c.get("user") or {}).get("login") in BOT_USERS
                     and (c["body"] or "").startswith(render_summary.MARKER)]
        if state is None and summaries:
            state = render_summary.saved_state(summaries[-1]["body"])
        if not state or not summaries:
            return state, False
        since = summaries[-1]["created_at"]
        author = api(f"repos/{repo}/pulls/{pr}")["user"]["login"]
        spoken = [*conversation, *api_list(f"repos/{repo}/pulls/{pr}/comments"),
                  *api_list(f"repos/{repo}/pulls/{pr}/reviews?per_page=100")]
    except UNREADABLE:
        if state is None:
            raise
        return state, False
    answering = {u["login"] for c in spoken if (c.get("submitted_at") or c.get("created_at") or "") > since
                 and (u := c.get("user") or {}).get("type") != "Bot" and u.get("login") not in (author, None)
                 and not (c.get("body") or "").startswith("/review")}
    return state, any(permission(repo, login) in WRITE for login in answering)


def post_inline(repo, pr, head_sha, review, exact=()):
    """Post the review's inline comments (`inline_comments`, taken out of `review`), anchored to the reviewed commit:
    model output from untrusted input, so redacted (with the `exact` values too) and made inert. A fenced block is
    left as written, so a suggestion applies cleanly. One call each, a second apart, since one line GitHub refuses
    fails a whole batched review, and a batch doesn't say which comment got which id. GitHub also decides whether a
    line is in the diff. One it won't take goes on the review's `outside_diff`, as the summary's own, so no finding
    is lost. {id: priority tag} of what posted."""
    posted = {}
    for i, c in enumerate(review.pop("inline_comments", [])):
        if i:
            time.sleep(1)
        body = render_summary.inert_report(redact(c["body"], exact))
        side, tag = c.get("side", "RIGHT"), render_summary.TAG.match(body)
        fields = {"commit_id": head_sha, "path": c["path"], "line": c["line"], "side": side, "body": body}
        if c.get("start_line", c["line"]) < c["line"]:
            fields.update(start_line=c["start_line"], start_side=side)
        try:
            new = json.loads(gh("api", "-X", "POST", f"repos/{repo}/pulls/{pr}/comments", "--input", "-",
                                input=json.dumps(fields)).stdout)
            posted[str(new["id"])] = tag[1] if tag else "P2"
        except UNREADABLE as e:  # the model's path and gh's echo are untrusted
            where = f"{c['path']}:{c['line']}"
            print(f"::warning::Couldn't post the inline comment on {log_value(where)}: {log_value(str(e))}")
            claim, _, detail = body[tag.end() if tag else 0:].strip().partition("\n")
            review["outside_diff"].append({"priority": tag[1] if tag else "P2", "claim": claim, "location": where,
                                           "detail": session.shorten(detail.strip(), 2000)})
    return posted


def redact_fields(value, exact=()):
    """`value`, JSON-shaped, with `redact` run on each string in it. Redacting the JSON's text instead can break
    it: an escape's last letter can join a token pattern."""
    if isinstance(value, str):
        return redact(value, exact)
    if isinstance(value, list):
        return [redact_fields(v, exact) for v in value]
    if isinstance(value, dict):
        return {redact(str(k), exact): redact_fields(v, exact) for k, v in value.items()}
    return value


def job_token_forms(token):
    """The job's token as a read of it could carry it: as is, and base64 with no padding, so a copy that drops it is
    caught too."""
    return (token, base64.b64encode(token.encode()).decode().rstrip("=")) if token else ()


def post(out, repo, pr, runner_temp, head_sha, mode, salvaged, detail="", max_gpus="4", job_token=""):
    """Post the review's inline comments and the summary, rendered from the review's fields by render_summary and
    redacted, and with the job's token removed however it is spelled (`job_token`): model output from untrusted
    input. Read the PR's threads after posting the inline comments, so the
    status line counts them along with earlier ones still open. Each run posts its own summary comment, so a
    re-review doesn't overwrite the last one. A "didn't complete" comment an earlier run left is then rewritten to
    point here, and loses its marker so a later failure starts a fresh one."""
    exact = job_token_forms(job_token)
    try:
        review = render_summary.over_gpu_cap(finished_review(runner_temp), max_gpus)
        posted = post_inline(repo, pr, head_sha, review, exact)
        pages = graphql(THREADS, repo, pr)
        # Before rendering: the summary's hidden state is base64, where `redact` over the finished text can't see it.
        earlier, answered = earlier_review(repo, pr)
        # And the earlier findings outside the diff whose file nothing since touched, for the approval and the state.
        review = redact_fields({**render_summary.carried(review, earlier, answered, pages),
                                "untouched": untouched(earlier)}, exact)
        text = render_summary.render(review, pages, head_sha, mode, render_summary.stacked_above("stacked-above.json"),
                                     salvaged == "true")
    except Exception as e:
        text = ""
        print(f"::warning::The review's summary didn't render: {type(e).__name__}: {log_value(str(e))}")
    text = redact(text, exact)
    if not text.strip():
        print("::warning::The review returned no summary to render.")
        sys.exit(1)
    # What the approval rule reads, with what it carried over and the inline findings as they stand now (and this
    # run's as post_inline made them, so one deleted the moment it posted is still seen), for the rule to check
    # they are still there, unchanged.
    result = {**review, "inline": {**render_summary.inline_findings(pages), **posted}}
    out["result"] = json.dumps(redact_fields(result, exact))
    if salvaged == "true":
        text += (f"\n<sub>This review finished but went past the workflow's turn limit ({detail}), so the action "
                 "reported it as failed.</sub>\n")
    url = comment(repo, pr, text)
    out["id"] = url.rsplit("-", 1)[-1]
    for cid in marked(read(api_list, f"repos/{repo}/issues/{pr}/comments") or [], INCOMPLETE):
        body = f"{RESOLVED}\nAn earlier review run didn't complete; a later run finished: {url}\n"
        try:
            gh("api", "-X", "PATCH", f"repos/{repo}/issues/comments/{cid}", "--input", "-",
               input=json.dumps({"body": body}))
        except RuntimeError as e:
            print(f"::warning::Couldn't mark comment {cid} resolved: {log_value(str(e))}")


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


def labels(out, repo, pr, runner_temp):
    """The PR's type/* labels, from the review's structured output. Model output, so each is checked
    against the fixed list before it reaches the API; type/* labels the review no longer returns are
    removed. Area and size labels come from pr-labels.yml."""
    try:
        types = finished_review(runner_temp).get("types") or []
    except (OSError, ValueError, AttributeError):
        types = []
    wanted = [f"type/{t}" for t in (types if isinstance(types, list) else []) if t in TYPES]
    if not wanted:
        print("::warning::The review returned no valid type; leaving the type labels alone.")
        return
    for old in [x["name"] for x in read(api, f"repos/{repo}/issues/{pr}/labels") or []]:
        if old.startswith("type/") and old not in wanted:
            gh("api", "-X", "DELETE", f"repos/{repo}/issues/{pr}/labels/{old.replace('/', '%2F')}")
    for label in wanted:
        ensure_label(repo, label)
        gh("api", "-X", "POST", f"repos/{repo}/issues/{pr}/labels", "-f", f"labels[]={label}")


def handoff(out, pr, head_sha, mode, runner_temp, home, prev="", job_token=""):
    """A review that stopped before it finished: the handoff a `/review continue` starts from
    ($RUNNER_TEMP/handoff), from the execution file or, after a timeout, the transcript Claude Code writes
    as it goes. A continued run that stops too merges its handoff onto the one it started from (`prev`).
    Model notes on the PR's own content, so redacted as the summary is, the job's token too; the artifact is public
    on a public repo."""
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
    exact = job_token_forms(job_token)
    meta = redact_fields({"pr": pr, "head_sha": head_sha, "mode": mode, **h}, exact)
    (saved / "handoff.json").write_text(json.dumps(meta))
    (saved / "handoff.md").write_text(redact(session.handoff_md(h, pr, head_sha, mode), exact))
    out["dir"] = str(saved)


def incomplete(out, repo, pr, run_url, why="a step failed", detail="", can_continue="false", refused="false"):
    """Anything failed: say so on the PR rather than leaving silence. Repeated failures update one comment,
    found by its marker, so retries don't stack up comments. A refused /review continue has a marker of its
    own, so it never writes over a stopped review's comment (that it can still be continued by someone
    else). A stopped review posted nothing: its findings are in its notes, and the comment says so."""
    if refused == "true":
        # A /review continue that had nothing usable to continue from; `why` says what to do.
        marker = REFUSED
        text = f"{marker}\n`/review continue` didn't run: {why}. [Log]({run_url}).\n"
    else:
        marker = INCOMPLETE
        then = ("Its findings post only when a review finishes, so they aren't on the PR yet; they are in the notes "
                "`/review continue` starts from. Comment `/review continue` to carry on in a fresh session (kept for "
                "7 days), or the same `/review` command to start over." if can_continue == "true"
                else "Comment the same `/review` command to retry.")
        text = (f"{marker}\nThe automated review didn't complete: {why}{f' ({detail})' if detail else ''}. "
                f"[Log]({run_url}). {then}\n")
    earlier = marked(read(api_list, f"repos/{repo}/issues/{pr}/comments") or [], marker)
    if earlier:
        gh("api", "-X", "PATCH", f"repos/{repo}/issues/comments/{earlier[-1]}", "--input", "-",
           input=json.dumps({"body": text}))
    else:
        comment(repo, pr, text)


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
        if (source.get("user") or {}).get("login") in POSTING_BOT_USERS:
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
    """Untrusted text for an issue or comment: redacted, on one line, and made inert (render_summary.inert)."""
    return render_summary.inert(redact(" ".join(str(text or "").split())))


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
    except UNREADABLE:
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
    if merge:
        shas = {merge} | {c["sha"] for c in api_list(f"repos/{repo}/pulls/{number}/commits?per_page=100")}
        signals += [{"key": f"{number}:reverted:{c['sha'][:10]}",
                     "text": f"`#{number}` reverted by `{c['sha'][:10]}` ({safe(c['subject'])})"}
                    for c in reverted(commits, shas, number, merge)]
        files = set(run(["git", "diff", "--name-only", f"{merge}^", merge], check=False).stdout.split())
        signals += szz(after_merge(commits, merge), shas, number, files, bug_prs, window)
        parents = next((c["parents"].split() for c in commits if c["sha"] == merge), [])
        if parents:
            signals += ci_regressions(repo, number, merge, parents[0])
    return signals, mentions


def monitor(out, repo, approvals=""):
    """The daily report (review-monitor.yml), with no model: what happened to the PRs merged in the last 60
    days with the bot's approval (GitHub's own record: an APPROVED review of the bot's), the kill switch and
    new bug reports, written to the status issue's body. Signals not reported before go out as one comment,
    so subscribers get at most one notification a day. Also makes the `skip-ai-review` label if it's missing,
    so neither it nor the issue is a setup step. Untrusted titles go through `safe`, text reaches
    `gh` on stdin."""
    now = utcnow()
    window = (now - datetime.timedelta(days=MONITOR_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = status_issue(repo)
    number = existing["number"] if existing else json.loads(gh(
        "api", "-X", "POST", f"repos/{repo}/issues", "--input", "-",
        input=json.dumps({"title": "Review bot status",
                          "body": f"{STATUS_ISSUE}\nThe review bot's status: a daily report in this body.\n"})
    ).stdout)["number"]
    body = api(f"repos/{repo}/issues/{number}").get("body") or ""
    try:
        state = json.loads(base64.b64decode(MONITOR.search(body)[1]))
    except (ValueError, TypeError):
        state = {}
    last = state.get("last") or window
    approved = sorted((p["number"] for page in graphql(MERGED_APPROVALS, repo)
                       for p in page["data"]["repository"]["pullRequests"]["nodes"] if p["mergedAt"] >= window
                       and any((r.get("author") or {}).get("__typename") == "Bot"
                               and r["author"]["login"] in render_summary.BOTS for r in p["reviews"]["nodes"])),
                      reverse=True)
    commits = main_commits(window)
    bug_prs = {i["number"] for i in api_list(f"repos/{repo}/issues?state=closed&labels=type%2Fbug&since={window}"
                                             "&per_page=100")
               if (i.get("pull_request") or {}).get("merged_at")}
    rows, signals, linked, unread = [], [], set(), set()
    for pr in approved:
        try:
            pull = api(f"repos/{repo}/pulls/{pr}")
            found, mentions = pr_signals(repo, {**pull, "number": pr}, commits, bug_prs, window)
        except UNREADABLE as e:
            rows.append(f"| `#{pr}` | couldn't read: {safe(e).replace('|', '/')} |")
            unread.add(str(pr))
            continue
        signals += found
        linked |= {s["key"].split(":")[-1] for s in found if ":ref:" in s["key"]} | {m.lstrip("#") for m in mentions}
        cell = "; ".join(s["text"] for s in found) or "none"
        if mentions:
            cell += " (also mentioned in " + ", ".join(f"`{m}`" for m in mentions) + ")"
        rows.append(f"| `#{pr}` {safe(pull.get('title')).replace('|', '/')} | {cell.replace('|', '/')} |")
    bugs = [i for i in api_list(f"repos/{repo}/issues?state=all&labels=bug&since={last}&per_page=100")
            if "pull_request" not in i and (i.get("created_at") or "") >= last and str(i["number"]) not in linked]
    new_prs = [p for p in api(f"repos/{repo}/pulls?state=all&sort=created&direction=desc&per_page=100") or []
               if (p.get("created_at") or "") >= last]
    reported = set(state.get("signals") or [])
    current = {s["key"]: s for s in signals}
    fresh = [current[k]["text"] for k in sorted(current) if k not in reported]
    switch = f"**{'on' if approvals == 'on' else 'off'}** (`REVIEW_APPROVALS`)."
    if approvals == "on" and not os.environ.get("REVIEW_APP_CLIENT_ID"):  # the app reads the switch and approves
        switch = "**off**: `REVIEW_APPROVALS` is on, but `REVIEW_APP_CLIENT_ID` isn't set, so no app can approve."
    head = [STATUS_ISSUE, "### Review bot status", f"Updated {now.strftime('%Y-%m-%d %H:%M')} UTC.", "",
            f"Approvals: {switch}", "",
            f"#### PRs merged with the bot's approval, last {MONITOR_DAYS} days", "| PR | Signals |", "| --- | --- |"]
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
       input=json.dumps({"body": "\n".join(head + (rows or ["| | none yet |"]) + tail) + "\n"}))
    ensure_label(repo, "skip-ai-review")  # after the report: a failed write here must not cost the day's report


# ---------------------------------------------------------------- improve (every other day)

def scrub(text):
    """Comment or model text for the improvement run: `safe` and clipped, with names and links taken out,
    which `safe` only makes inert."""
    text = re.sub(r"(?i)[a-z][a-z0-9+.-]*://[^\s`]+|(?<![a-z0-9])www\.[^\s`]+", "[link]",
                  re.sub(r"@[\w-]+", "@someone", str(text or "")))
    return safe(text)[:IMPROVE_TEXT]


def pr_evidence(repo, pr, since):
    """What happened on PR `pr` since `since`, for the improvement run: the findings a later review judged moot,
    its reviews, inline comments and conversation. Logins become a role. GitHub doesn't say when a thread was
    resolved, so a finding of the bot's someone resolved is kept, however old, only when the PR is active since
    `since`: a review or comment from a person or this bot dated inside the window. A quiet PR gives nothing."""
    author = (api(f"repos/{repo}/pulls/{pr}").get("user") or {}).get("login")
    inline, conversation = discussion(repo, pr)
    reviews = [{"user": (r.get("user") or {}).get("login"), "user_type": (r.get("user") or {}).get("type"),
                "created_at": r.get("submitted_at"), "url": r.get("html_url"), "body": r.get("body")}
               for r in api_list(f"repos/{repo}/pulls/{pr}/reviews?per_page=100")
               if (r.get("user") or {}).get("login") not in BOT_USERS]

    def read(c):  # what a reader sees: not empty, and not another bot's
        return c.get("body") and not (c.get("user_type") == "Bot" and c["user"] not in POSTING_BOT_USERS)

    # A person's review counts whatever its text (most approvals have none); a comment, when a reader sees it.
    active = any((c.get("created_at") or "") >= since
                 for c in [*[r for r in reviews if r["user_type"] != "Bot"], *filter(read, [*inline, *conversation])])

    def kept(comments):
        rows = []
        for c in comments:
            mine = c["user"] in POSTING_BOT_USERS
            if not read(c):
                continue
            if (c.get("created_at") or "") >= since or (active and mine and c.get("resolved")):
                rows.append({**{k: c[k] for k in ("id", "path", "line", "in_reply_to_id", "resolved", "outdated",
                                                  "url") if k in c},
                             "role": "bot" if mine else "author" if c["user"] == author else "other",
                             "body": scrub(c["body"])})
        return rows[-IMPROVE_ITEMS:]

    moot = [{"ref": scrub(f.get("ref")), "why": scrub(f.get("why")), "url": c["url"]}
            for c in conversation
            if c["user"] in BOT_USERS and (c["body"] or "").startswith(render_summary.MARKER)
            and (c.get("created_at") or "") >= since
            for f in (render_summary.saved_state(c["body"]) or {}).get("moot", [])]
    return {"moot": moot, "reviews": kept(reviews), "inline": kept(inline),
            "conversation": kept(conversation)}


def improve_inputs(out, repo, key_set):
    """The evidence for the improvement run (review-improve.yml), in improve-evidence/: one file per PR with a
    review summary or a stopped review of the bot's in the last three weeks (where to look: a reply can come long
    after a review) and anything from the last IMPROVE_DAYS days (the evidence), and status.json, the status
    issue's signals. `run` stays false without an API key, or with nothing
    to read."""
    out["run"] = "false"
    if key_set != "true":
        print("::warning::ANTHROPIC_API_KEY is not set; skipping the improvement run.")
        return
    now = utcnow()
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    since = (now - datetime.timedelta(days=IMPROVE_DAYS)).strftime(stamp)
    newest_first = dict.fromkeys(
        int(c["issue_url"].rsplit("/", 1)[1])
        for c in reversed(api_list(f"repos/{repo}/issues/comments"
                                   f"?since={(now - datetime.timedelta(days=21)).strftime(stamp)}&per_page=100"))
        if (c.get("user") or {}).get("login") in BOT_USERS
        and (c.get("body") or "").startswith((render_summary.MARKER, INCOMPLETE)))
    files = {}
    for pr in list(newest_first)[:IMPROVE_ITEMS]:
        evidence = pr_evidence(repo, pr, since)
        if any(evidence.values()):
            files[f"{pr}.json"] = {"pr": pr, **evidence}
    found = status_issue(repo)
    status = [{"url": c.get("html_url"), "text": scrub(line)}
              for c in (api_list(f"repos/{repo}/issues/{found['number']}/comments?since={since}&per_page=100")
                        if found else [])
              if (c.get("user") or {}).get("login") in POSTING_BOT_USERS
              for line in (c.get("body") or "").splitlines() if line.strip()]
    if status:
        files["status.json"] = status[-IMPROVE_ITEMS:]
    if not files:
        print(f"::notice::The bot did nothing in the last {IMPROVE_DAYS} days; no improvement run.")
        return
    Path("improve-evidence").mkdir()
    for name, value in files.items():
        write_json(f"improve-evidence/{name}", value)
    out.update(run="true", schema=json.dumps(IMPROVE_SCHEMA, separators=(",", ":")))


def improve_post(out, repo, output=""):
    """Open the run's issue, named by the days its evidence covers, from the improvement run's structured output,
    redacted, on stdin. Evidence links stay inside this repository (no `..`, which would climb out of it) and go
    through redirect.github.com, which keeps them off the PRs' timelines. An empty list opens nothing; a run that
    returned no list fails. Never touches code, PRs or the bot."""
    try:
        items = json.loads(output)["improvements"]
    except (ValueError, KeyError, TypeError):
        sys.exit("The improvement run returned no list, so no issue was opened.")
    if not items:
        print("::notice::Nothing worth raising this time; no issue opened.")
        return
    link = re.compile(rf"https://github\.com/({re.escape(repo)}/[\w./?=#&%-]+)")
    parts = ["What the review bot could improve about itself, most useful first. It changes nothing on its own."]
    for n, item in enumerate(items, 1):
        links = [f"- https://redirect.github.com/{m[1]}" for m in map(link.fullmatch, item["evidence"])
                 if m and ".." not in urllib.parse.unquote(m[1])]
        parts.append("\n".join([f"### {n}. {scrub(item['title'])}", "", f"**Change:** {scrub(item['change'])}", "",
                                f"**Where:** {scrub(item['where'])}", "", "**Evidence:**", *links]))
    now = utcnow()
    days = f"{now - datetime.timedelta(days=IMPROVE_DAYS):%Y-%m-%d} to {now:%Y-%m-%d}"
    body = json.dumps({"title": f"Review bot: what to improve, {days}",
                       "body": redact("\n\n".join(parts)) + "\n"})
    gh("api", "-X", "POST", f"repos/{repo}/issues", "--input", "-", input=body)


# ---------------------------------------------------------------- dispatcher

def main(steps):
    """Run the step `sys.argv[1]` names, with its parameters from the environment, and write what it put
    in `out` to $GITHUB_OUTPUT, even when it fails. An empty variable takes the parameter's default. Each
    step gets its parameters as strings and validates them itself."""
    step = steps[sys.argv[1]]
    kwargs = {}
    for name, p in list(inspect.signature(step).parameters.items())[1:]:
        value = os.environ.get(name.upper(), "")
        if value == "" and p.default is not p.empty:
            continue
        if name.upper() not in os.environ:
            raise ValueError(f"{sys.argv[1]}: ${name.upper()} is not set")
        kwargs[name] = value
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


# ---------------------------------------------------------------- merge-queue check

# A merge-queue branch GitHub makes for an entry: gh-readonly-queue/<base>/pr-<number>-<base sha>.
QUEUE_REF = re.compile(r"(?:refs/heads/)?gh-readonly-queue/.+/pr-(\d+)-[0-9a-f]{40}")
# The states that change where a reviewer stands. COMMENTED and PENDING don't.
DECIDING = ("APPROVED", "CHANGES_REQUESTED", "DISMISSED")


def merge_approved(reviews, head, has_write, bots):
    """(passes, why): whether the PR, at `head`, may merge on its reviews (GitHub's, oldest first): someone stands at
    APPROVED who is a person with write access, or one of the bot's `bots` logins, on `head`. A person's approval
    is GitHub's business and doesn't go stale on a push (`main`'s protection as it is). A later CHANGES_REQUESTED or
    a dismissal takes back that reviewer's approval. A person without write access doesn't count, or their word
    would rescue a stale bot approval that GitHub's own rule still counts."""
    standing = {}
    for r in reviews:
        if r.get("state") in DECIDING and r.get("user"):
            standing[r["user"]["login"]] = r
    approved = {login: r for login, r in standing.items() if r["state"] == "APPROVED"}
    people = [login for login, r in approved.items() if r["user"].get("type") == "User" and login not in bots]
    person = next((login for login in people if has_write(login)), None)
    if person:
        return True, f"approved by {person}"
    bot = [r for login, r in approved.items() if login in bots]
    if any(r.get("commit_id") == head for r in bot):
        return True, f"the bot approved the PR's head {head[:7]}"
    return False, (f"the bot's approval is on {bot[-1].get('commit_id', '?')[:7]}, not the PR's head {head[:7]}: "
                   "a new `/review` or a person's approval is needed" if bot else "no current approval")


def approval_current(out, repo, head_ref):
    """The merge-queue check that a bot approval counts only for the commit it reviewed (review-approval-current.yml).
    A ref that doesn't parse or any failed read fails it."""
    m = QUEUE_REF.fullmatch(head_ref)
    if not m:
        sys.exit(f"::error::Not a merge-queue branch: {head_ref}")
    pr = int(m[1])
    head = api(f"repos/{repo}/pulls/{pr}")["head"]["sha"]
    passes, why = merge_approved(api_list(f"repos/{repo}/pulls/{pr}/reviews"), head,
                                 lambda login: permission(repo, login) in WRITE, BOT_USERS)
    if not passes:
        sys.exit(f"::error::PR #{pr} has no approval for its current commit: {why}")
    print(f"::notice::PR #{pr}: {why}.")


STEPS = {"preflight": preflight, "react": react, "resolve": resolve, "collect": collect, "salvage": salvage,
         "usage": usage, "post": post, "hide": hide, "labels": labels, "handoff": handoff, "incomplete": incomplete,
         "monitor": monitor, "improve_inputs": improve_inputs, "improve_post": improve_post,
         "approval_current": approval_current}

if __name__ == "__main__":
    main(STEPS)
