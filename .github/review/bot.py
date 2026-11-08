#!/usr/bin/env python3
"""The review bot's steps: every step of claude-review.yml that isn't an action is a function here,
run with the default branch checked out as

    python3 .github/review/bot.py <step>

A step's parameters are its environment variables, lowercased: the step's `env:` in the workflow is
its interface. It records its outputs in `out`, which `main` writes to $GITHUB_OUTPUT even when the
step fails. Every call to another program (`gh`, `git`, gpu.py, ssh-keygen) goes through `run`, the
one place a test replaces. Text the bot posts reaches `gh` on stdin, never its command line.
"""
import collections
import fnmatch
import inspect
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
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
# A branch name safe for the API path and the command line.
BRANCH = re.compile(r"[A-Za-z0-9._/][A-Za-z0-9._/-]*")
THREADS = """query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $pr) {
    reviewThreads(first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes { isResolved isOutdated comments(first: 1) { nodes { databaseId author { login } body path line } } }
    } } } }"""
# A commenter's author_association that makes them a contributor (see the workflow's header).
CONTRIBUTORS = ("OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR")


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
    """since-commits.txt: one line per commit."""
    lines = []
    for c in commits:
        subject = c["commit"]["message"].split("\n")[0]
        lines.append(f"{c['sha'][:10]} {subject}\n")
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


# ---------------------------------------------------------------- preflight job

def preflight(out, repo, pr, key_set, command, actor, author, assoc, skip_label, team_turns="2000",
              self_turns="40"):
    """Whether a comment starts a review, in which mode (`command`, one of the workflow's fixed words)
    and with which turn cap; a request it turns down for a reason the commenter can act on gets a
    one-line reply. Write access is checked against the actor's permission; a contributor may run
    `/review light` or `/review continue` on their own PR, and only then is `non_write_user` set."""
    if key_set != "true":
        print("::warning::ANTHROPIC_API_KEY is not set; skipping the review.")
        out["run"] = "false"
        return
    if command not in ("light", "nogpu", "continue"):
        out["run"] = "false"
        return
    mode = "" if command == "continue" else command  # the stopped review's mode, from its handoff
    perm = permission(repo, actor)
    if perm in WRITE:
        out.update(run="true", mode=mode, turns=team_turns)
        return
    own = assoc in CONTRIBUTORS and actor != "" and actor == author and skip_label != "true"
    if own and command != "nogpu":
        print(f"::notice::{actor} has '{perm}' access; reviewing their own PR as its author ({assoc}).")
        out.update(run="true", mode=mode, non_write_user=actor, turns=self_turns)
        return
    out["run"] = "false"
    if not own:
        print(f"::notice::{actor} has '{perm}' access, not write; only people with write access, or a contributor "
              "on their own PR with `/review light`, can request a review.")
        return
    print(f"::notice::{actor} has '{perm}' access, not write; this review needs write access.")
    try:
        comment(repo, pr, "This review needs write access; on your own PR, comment `/review light`.")
    except RuntimeError as e:
        print(f"::warning::Couldn't reply: {e}")


# ---------------------------------------------------------------- review job: before the review

def resolve(out, repo, pr, mode, comment_id, non_write=""):
    """React 👀 to the /review comment (best effort), then the PR's head, its title and the review's
    settings. A /review continue (`mode` empty) takes the mode and timeout of the review it
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
    # Mode sets the prompt's reporting bar.
    if mode == "nogpu":
        out["mode"] = "nogpu"
        timeout = 45
    else:
        out["mode"] = "light"
        timeout = 25
    # Someone with write access: a timeout that only stops a runaway, like the turn cap.
    out["timeout"] = 300 if not non_write else timeout
    out["title"] = pull["title"]


def restore(out, repo, pr, head_sha, runner_temp, non_write=""):
    """/review continue: fetch the handoff a stopped review saved (handoff) into $RUNNER_TEMP/prev, and
    that review's mode and timeout. The newest unexpired artifact for this PR wins, and only
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
        timeout = 25
    elif mode == "nogpu":
        if non_write:
            fail("continuing a nogpu review takes write access, like `/review nogpu`")
        timeout = 45
    else:
        fail("the saved handoff is malformed")
    out.update(dir=str(saved), mode=mode, timeout=300 if not non_write else timeout)


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

    # Inline review comments, each marked with its thread's resolved and outdated state (only GraphQL
    # has those), and `mine` when this bot posted it.
    threads = {}
    for page in graphql(THREADS, repo, pr):
        for t in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = t["comments"]["nodes"][:1]
            if first:
                threads[first[0]["databaseId"]] = {"resolved": t["isResolved"], "outdated": t["isOutdated"]}
    comments = []
    for c in api_list(f"repos/{repo}/pulls/{pr}/comments"):
        user = (c.get("user") or {}).get("login")
        comments.append({"id": c["id"], "user": user, "path": c.get("path"), "line": c.get("line"),
                         "side": c.get("side"), "in_reply_to_id": c.get("in_reply_to_id"), "body": c.get("body"),
                         **threads.get(c.get("in_reply_to_id") or c["id"], {}), "mine": user in BOT_USERS})
    write_json("pr-comments.json", comments)

    # The conversation tab: replies to the summary, and the author's answers.
    conversation = [{"id": c["id"], "user": (c.get("user") or {}).get("login"), "created_at": c.get("created_at"),
                     "body": c.get("body")} for c in api_list(f"repos/{repo}/issues/{pr}/comments")]
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

    # A failed fetch leaves none of these files, so a finding said to be fixed above stays open, and
    # the review still runs.
    try:
        write_json("stacked-above.json", stacked_above(repo, pull, head_sha))
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
    the job summary, and a small record ($RUNNER_TEMP/usage/) to compare runs by. Never posted on the PR,
    and never fails: a record that couldn't be read says why in its `error` field."""
    record = {"pr": pr, "head_sha": head_sha, "mode": mode, "continue": prev != "", "run_id": github_run_id,
              "run_attempt": github_run_attempt, "outcome": outcome}
    try:
        record.update(session.usage(session.load([Path(runner_temp, EXECUTION), Path(home, ".claude/projects")])))
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"::warning::Couldn't read the review's usage: {record['error']}")
    try:
        path = Path(runner_temp, "usage", f"review-usage-{pr}-{github_run_id}-{github_run_attempt}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        with open(github_step_summary, "a") as f:
            f.write(session.usage_table(record))
    except Exception as e:
        print(f"::warning::Couldn't write the usage record: {e}")


def post(out, repo, pr, output, head_sha, mode, salvaged, detail=""):
    """Post the summary, rendered from the review's fields by render_summary and redacted: model output from
    untrusted input. Read the PR's threads after the review posted its inline comments, so the status line
    counts them along with earlier ones still open. Each run posts its own summary comment, so a re-review
    doesn't overwrite the last one. A "didn't complete" comment an earlier run left is then rewritten to
    point here, and loses its marker so a later failure starts a fresh one."""
    pages = graphql(THREADS, repo, pr)
    try:
        text = render_summary.render(json.loads(output), pages, head_sha, mode,
                                     render_summary.stacked_above("stacked-above.json"))
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
         "usage": usage, "post": post, "hide": hide, "labels": labels, "handoff": handoff, "incomplete": incomplete}

if __name__ == "__main__":
    main(STEPS)
