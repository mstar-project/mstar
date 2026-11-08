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
    python3 .github/review/render_summary.py --state < summary-comment-body   # the review state a later run reads
"""
import argparse
import base64
import json
import re
import sys

# Lets the workflow find earlier summaries to hide as outdated.
MARKER = "<!-- claude-review-summary -->"
# What a re-review starts from (head, findings outside the diff, risk, live-test call), base64 JSON,
# so model text can't close the HTML comment. Read back only from the bot's own comments.
STATE = re.compile(r"<!-- claude-review-state ([A-Za-z0-9+/=]+) -->")
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
        "prior_findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "inline comment id, or the outside_diff location"},
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "status": {"type": "string", "enum": ["fixed", "open", "moot"]},
                "why": {"type": "string"},
            },
            "required": ["ref", "priority", "status", "why"],
        }},
        "design": STRINGS,
        "risk": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
        }, "required": ["level", "reason"]},
        "live_test": {"type": "object", "properties": {
            "run": {"type": "string", "enum": ["none", "targeted", "full"]},
            "reason": {"type": "string"},
            "confirm": STRINGS,
            "gpus": {"type": "integer", "minimum": 1, "maximum": 8},
        }, "required": ["run", "reason", "confirm", "gpus"]},
    },
    "required": ["outside_diff", "since_last", "description_problems", "questions", "checked", "skipped", "types",
                 "prior_findings", "design", "risk"],
}
FOOTER = "<sub>Advisory, not a merge gate. Rules: `AGENTS.md`. Reply to discuss.</sub>"


def open_priorities(pages):
    """(comment id, priority) of each of the bot's inline findings whose thread is neither resolved nor outdated."""
    found = []
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            if thread["isResolved"] or thread["isOutdated"]:
                continue
            first = thread["comments"]["nodes"][0]
            match = TAG.match(first["body"])
            if match and first["author"] and first["author"]["login"] in BOTS:
                found.append((str(first.get("databaseId", "")), match[1]))
    return found


def open_findings(review, inline):
    """Every open finding's priority: open inline threads, this run's findings outside the diff, and the
    earlier findings a re-review judged still open (whatever their thread's state)."""
    counted = {ref for ref, _ in inline}
    return ([p for _, p in inline] + [f["priority"] for f in review["outside_diff"]]
            + [f["priority"] for f in review.get("prior_findings", [])
               if f["status"] == "open" and f["ref"] not in counted])


def status(review, inline, head, mode):
    found = open_findings(review, inline)
    counts = {p: found.count(p) for p in PRIORITIES}
    if counts["P0"] or counts["P1"]:
        icon = "🔴"
    elif found or any(review[k] for k in ("description_problems", "questions", "skipped")):
        icon = "🟡"
    else:
        icon = "🟢"
    text = ", ".join(f"{n} {p}" for p, n in counts.items() if n) or "No findings"
    line = f"{icon} **{text}** on `{head[:7]}`, {mode} review"
    if review.get("risk"):
        line += f" · risk **{review['risk']['level']}**: {review['risk']['reason']}"
    return line


def render(review, inline, head, mode):
    """``inline`` is the priority of each open inline finding, from `open_priorities`."""
    parts = [f"{MARKER}\n### Automated review", status(review, inline, head, mode)]
    if review["since_last"]:
        parts.append(review["since_last"])
    prior = review.get("prior_findings", [])
    if prior:
        tally = ", ".join(f"{sum(f['status'] == s for f in prior)} {s}" for s in ("fixed", "open", "moot"))
        parts.append(f"Earlier findings: {tally}.")
    sections = [
        ("Still open from earlier", [f"**[{f['priority']}]** {f['ref']}: {f['why']}"
                                     for f in prior if f["status"] == "open"]),
        ("Outside the diff", [f"**[{f['priority']}]** `{f['location']}`: {f['claim']}\n\n  {f['detail']}"
                              for f in review["outside_diff"]]),
        ("Description", review["description_problems"]),
        ("Questions", review["questions"]),
        ("Design", review.get("design", [])),
    ]
    for title, items in sections:
        if items:
            parts.append(f"#### {title}\n" + "\n".join(f"- {item}" for item in items))
    if review["checked"]:
        checked = "\n".join(f"- {item}" for item in review["checked"])
        parts.append(f"<details><summary>What the review checked</summary>\n\n{checked}\n\n</details>")
    if review["skipped"]:
        parts.append("Not reviewed: " + ", ".join(f"`{path}`" for path in review["skipped"]) + ".")
    live = review.get("live_test")
    if live and mode == "gpu":
        parts.append(f"Live test: **{live['run']}**. {live['reason']}")
    parts.append(FOOTER)
    state = {"head": head, "mode": mode, "outside_diff": review["outside_diff"], "risk": review.get("risk"),
             "live_test": live}
    parts.append(f"<!-- claude-review-state {base64.b64encode(json.dumps(state).encode()).decode()} -->")
    return "\n\n".join(parts) + "\n"


def read_state(body):
    """The state a summary comment carries, or None."""
    match = STATE.search(body)
    return json.loads(base64.b64decode(match[1])) if match else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true", help="print the fields the review returns, as a JSON schema")
    ap.add_argument("--head")
    ap.add_argument("--mode")
    ap.add_argument("--threads", help="the PR's review threads, from `gh api graphql --paginate --slurp`")
    ap.add_argument("--state", action="store_true", help="print the state a summary comment (stdin) carries")
    a = ap.parse_args()
    if a.schema:
        print(json.dumps(SCHEMA, separators=(",", ":")))
    elif a.state:
        state = read_state(sys.stdin.read())
        print(json.dumps(state) if state else "")
    else:
        with open(a.threads) as f:
            inline = open_priorities(json.load(f))
        sys.stdout.write(render(json.load(sys.stdin), inline, a.head, a.mode))


if __name__ == "__main__":
    main()
