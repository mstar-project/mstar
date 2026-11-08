#!/usr/bin/env python3
"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. The status line counts the open findings (`open_findings`, which the
approval rule in gpu_review.py uses too), so a finding from an earlier run counts
until a re-review judges it fixed or moot or a commit changes its lines. Run by
claude-review.yml with the default branch checked out:

    python3 .github/review/render_summary.py --schema     # for --json-schema
    python3 .github/review/render_summary.py --head <sha> --mode light --threads threads.json < output.json
    python3 .github/review/render_summary.py --state --mode gpu < summary-comment-body   # what a re-review reads
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
            # Where a human reviewer should look, when the risk isn't low: each place and why.
            "look_at": {"type": "array", "items": {"type": "object", "properties": {
                "location": {"type": "string", "description": "path:line, or a function or class"},
                "why": {"type": "string"},
            }, "required": ["location", "why"]}},
        }, "required": ["level", "reason", "look_at"]},
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


def open_findings(review, pages):
    """(where, priority) of every open finding: the bot's inline threads, this run's findings outside
    the diff, and the earlier findings a re-review judged still open. A thread someone resolved stays
    open: only a re-review that judged it fixed or moot, or a commit that outdated it, clears it. An
    untagged bot thread counts as "untagged", since its priority is unknown. `pages` are the PR's
    review threads, from `gh api graphql --paginate --slurp`."""
    prior = review.get("prior_findings", [])
    cleared = {f["ref"] for f in prior if f["status"] in ("fixed", "moot")}
    found, counted = [], set()
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = thread["comments"]["nodes"][0]
            ref = str(first.get("databaseId", ""))
            if thread["isOutdated"] or ref in cleared or not (first["author"] and first["author"]["login"] in BOTS):
                continue
            match = TAG.match(first["body"])
            found.append((f"inline {ref}", match[1] if match else "untagged"))
            counted.add(ref)
    found += [(f"outside the diff, {f['location']}", f["priority"]) for f in review["outside_diff"]]
    # Judged still open, though its thread went outdated or it was outside the diff.
    found += [(f"earlier {f['ref']}", f["priority"]) for f in prior
              if f["status"] == "open" and f["ref"] not in counted]
    return found


def status(review, pages, head, mode):
    found = [p for _, p in open_findings(review, pages)]
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


def render(review, pages, head, mode):
    parts = ["### Automated review", status(review, pages, head, mode)]
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
        ("Where to look", [f"`{p['location']}`: {p['why']}" for p in (review.get("risk") or {}).get("look_at", [])]),
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
    state = {"head": head, "mode": mode, "skipped": review["skipped"], "outside_diff": review["outside_diff"],
             "risk": review.get("risk"), "live_test": live}
    # Model text can't open an HTML comment, so it can't forge a marker this bot or the workflow reads.
    text = "\n\n".join(parts).replace("<!--", "&lt;!--")
    return f"{MARKER}\n{text}\n\n<!-- claude-review-state {base64.b64encode(json.dumps(state).encode()).decode()} -->\n"


def read_state(body, mode):
    """The state a summary comment carries (its last), for a re-review in `mode` to start from; else
    None, and that run reviews the whole PR. None too when it doesn't decode, and for a thorough or
    gpu review unless the earlier one was thorough or gpu and skipped nothing, since a gpu review's
    result can approve and may build on a thorough one."""
    found = STATE.findall(body)
    try:
        state = json.loads(base64.b64decode(found[-1])) if found else None
    except ValueError:
        return None
    if not isinstance(state, dict):
        return None
    # A state from before `skipped` was kept counts as having skipped files.
    if mode in ("thorough", "gpu") and (state.get("mode") not in ("thorough", "gpu") or state.get("skipped", [None])):
        return None
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true", help="print the fields the review returns, as a JSON schema")
    ap.add_argument("--head")
    ap.add_argument("--mode")
    ap.add_argument("--threads", help="the PR's review threads, from `gh api graphql --paginate --slurp`")
    ap.add_argument("--state", action="store_true",
                    help="print the state a summary comment (stdin) carries, for a re-review in --mode")
    a = ap.parse_args()
    if a.schema:
        print(json.dumps(SCHEMA, separators=(",", ":")))
    elif a.state:
        state = read_state(sys.stdin.read(), a.mode)
        print(json.dumps(state) if state else "")
    else:
        with open(a.threads) as f:
            pages = json.load(f)
        sys.stdout.write(render(json.load(sys.stdin), pages, a.head, a.mode))


if __name__ == "__main__":
    main()
