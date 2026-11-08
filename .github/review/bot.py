#!/usr/bin/env python3
"""The review bot's steps: every step of claude-review.yml that isn't an action is a function here,
run with the default branch checked out as

    python3 .github/review/bot.py <step>

A step's parameters are its environment variables, lowercased: the step's `env:` in the workflow is
its interface. It records its outputs in `out`, which `main` writes to $GITHUB_OUTPUT even when the
step fails. Every call to another program (`gh`, `git`, gpu.py, ssh-keygen) goes through `run`, the
one place a test replaces. Text the bot posts reaches `gh` on stdin, never its command line.
"""
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
# The marker of a stopped review's "didn't complete" comment.
INCOMPLETE = "<!-- claude-review-incomplete -->"
# A branch name safe for the API path and the command line.
BRANCH = re.compile(r"[A-Za-z0-9._/][A-Za-z0-9._/-]*")
THREADS = """query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $pr) {
    reviewThreads(first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes { isResolved isOutdated comments(first: 1) { nodes { databaseId author { login } body } } }
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
    """since-commits.txt: one line per commit, merges marked."""
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
    if mode == "nogpu":
        out.update(mode="nogpu", effort="xhigh")
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


def collect(out, repo, pr, head_sha, mode, prev=""):
    """The review's inputs, in the workspace: pr.diff, pr-body.md, pr-comments.json (with each thread's
    resolved and outdated state, and `mine` on the bot's own), pr-conversation.json, a re-review's
    last-review.json, since.diff and since-commits.txt, stacked-above.json and above-<n>.diff, and
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

    # A re-review starts from this bot's last summary, when its commit is still an ancestor of the
    # head: the review re-checks those findings and reads only what changed since (review-prompt.md,
    # "Re-review"). After a force-push it reviews the whole PR again, and a nogpu review does too
    # unless the last one was nogpu and skipped nothing. The state is read only from the bot's own
    # summaries: comments that start with the summary's marker, in which model text can't open an HTML
    # comment. A diff GitHub won't give (too large, say) means a full review, never an empty since.diff.
    if not prev:
        summaries = [c for c in conversation
                     if c["user"] in BOT_USERS and (c["body"] or "").startswith(render_summary.MARKER)]
        state = render_summary.read_state(summaries[-1]["body"], mode) if summaries else None
        last = (state or {}).get("head")
        since = f"repos/{repo}/compare/{last}...{head_sha}"
        try:
            status = api(since)["status"] if last else ""
        except (RuntimeError, ValueError, KeyError, TypeError):
            status = ""
        if last and (status == "ahead" or last == head_sha):
            r = gh("api", "-H", "Accept: application/vnd.github.diff", since, check=False)
            if r.returncode == 0:
                write("last-review.json", json.dumps(state) + "\n")
                write("since.diff", r.stdout)
                try:
                    write("since-commits.txt", oneline(c for page in api(since, "--paginate", "--slurp")
                                                       for c in page["commits"]))
                except (RuntimeError, ValueError, KeyError):
                    write("since-commits.txt", "")
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


STEPS = {"preflight": preflight, "resolve": resolve, "restore": restore, "collect": collect}

if __name__ == "__main__":
    main(STEPS)
