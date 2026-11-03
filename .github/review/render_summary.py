#!/usr/bin/env python3
"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. The status line counts the bot's inline findings from the PR's review
threads, so a finding from an earlier run counts until its thread is resolved or
a commit changes its lines. Run by claude-review.yml with the default branch
checked out:

    python3 .github/review/render_summary.py --schema     # for --json-schema
    python3 .github/review/render_summary.py --head <sha> --mode light --threads threads.json < output.json
"""
import argparse
import json
import re
import sys

# Lets the workflow find earlier summaries to hide as outdated.
MARKER = "<!-- claude-review-summary -->"
# The bot's GraphQL logins: the review app's, and github-actions', which it posted as before the app.
BOTS = ("mstar-review", "github-actions")
TAG = re.compile(r"\*\*\[(P[0-3])\]\*\*")
PRIORITIES = ("P0", "P1", "P2", "P3")
TYPES = ("new-model", "bug", "perf", "feature", "refactor", "test", "docs", "ci")
STRINGS = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object",
    "properties": {
        "outside_diff": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "claim": {"type": "string"},
                "location": {"type": "string", "description": "path:line"},
                "detail": {"type": "string"},
            },
            "required": ["priority", "claim", "location", "detail"],
        }},
        "since_last": {"type": "string"},
        "description_problems": STRINGS,
        "questions": STRINGS,
        "checked": STRINGS,
        "skipped": STRINGS,
        "types": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True,
                  "items": {"type": "string", "enum": list(TYPES)}},
    },
    "required": ["outside_diff", "since_last", "description_problems", "questions", "checked", "skipped", "types"],
}
FOOTER = "<sub>Advisory, not a merge gate. Rules: `AGENTS.md`. Reply to discuss.</sub>"


def open_priorities(pages):
    """The priority of each of the bot's inline findings whose thread is neither resolved nor outdated."""
    found = []
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            if thread["isResolved"] or thread["isOutdated"]:
                continue
            first = thread["comments"]["nodes"][0]
            match = TAG.match(first["body"])
            if match and first["author"] and first["author"]["login"] in BOTS:
                found.append(match[1])
    return found


def status(review, inline, head, mode):
    found = inline + [f["priority"] for f in review["outside_diff"]]
    counts = {p: found.count(p) for p in PRIORITIES}
    if counts["P0"] or counts["P1"]:
        icon = "🔴"
    elif found or any(review[k] for k in ("description_problems", "questions", "skipped")):
        icon = "🟡"
    else:
        icon = "🟢"
    text = ", ".join(f"{n} {p}" for p, n in counts.items() if n) or "No findings"
    return f"{icon} **{text}** on `{head[:7]}`, {mode} review"


def render(review, inline, head, mode):
    """``inline`` is the priority of each open inline finding, from `open_priorities`."""
    parts = [f"{MARKER}\n### Automated review", status(review, inline, head, mode)]
    if review["since_last"]:
        parts.append(review["since_last"])
    sections = [
        ("Outside the diff", [f"**[{f['priority']}]** `{f['location']}`: {f['claim']}\n\n  {f['detail']}"
                              for f in review["outside_diff"]]),
        ("Description", review["description_problems"]),
        ("Questions", review["questions"]),
    ]
    for title, items in sections:
        if items:
            parts.append(f"#### {title}\n" + "\n".join(f"- {item}" for item in items))
    if review["checked"]:
        checked = "\n".join(f"- {item}" for item in review["checked"])
        parts.append(f"<details><summary>What the review checked</summary>\n\n{checked}\n\n</details>")
    if review["skipped"]:
        parts.append("Not reviewed: " + ", ".join(f"`{path}`" for path in review["skipped"]) + ".")
    parts.append(FOOTER)
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true", help="print the fields the review returns, as a JSON schema")
    ap.add_argument("--head")
    ap.add_argument("--mode")
    ap.add_argument("--threads", help="the PR's review threads, from `gh api graphql --paginate --slurp`")
    a = ap.parse_args()
    if a.schema:
        print(json.dumps(SCHEMA, separators=(",", ":")))
    else:
        with open(a.threads) as f:
            inline = open_priorities(json.load(f))
        sys.stdout.write(render(json.load(sys.stdin), inline, a.head, a.mode))


if __name__ == "__main__":
    main()
