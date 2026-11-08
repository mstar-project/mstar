#!/usr/bin/env python3
"""The review bot's steps: every step of claude-review.yml that isn't an action is a function here,
run with the default branch checked out as

    python3 .github/review/bot.py <step>

A step's parameters are its environment variables, lowercased: the step's `env:` in the workflow is
its interface. It records its outputs in `out`, which `main` writes to $GITHUB_OUTPUT even when the
step fails. Every call to another program (`gh`, `git`, gpu.py, ssh-keygen) goes through `run`, the
one place a test replaces. Text the bot posts reaches `gh` on stdin, never its command line.
"""
import inspect
import json
import os
import re
import secrets
import subprocess
import sys

# Credential-shaped strings, removed from everything the bot posts or saves.
REDACT = [re.compile(p) for p in (
    r"sk-ant-[A-Za-z0-9_-]+",
    r"(gh[opsur]_|github_pat_)[A-Za-z0-9_]{20,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+",
)]
WRITE = ("admin", "maintain", "write")
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


STEPS = {"preflight": preflight}

if __name__ == "__main__":
    main(STEPS)
