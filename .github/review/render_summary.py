#!/usr/bin/env python3
"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. The status line counts the open findings (`open_findings`), so a finding
from an earlier run counts until a later review judges it fixed, fixed above or moot, or a commit
changes its lines. Run by claude-review.yml with the default branch checked out:

    python3 .github/review/render_summary.py --schema     # for --json-schema
    python3 .github/review/render_summary.py --head <sha> --mode light --threads threads.json \
        --above stacked-above.json < output.json
    python3 .github/review/render_summary.py --state --mode nogpu < summary-comment-body   # what a re-review reads
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

# Lets the workflow find earlier summaries to hide as outdated.
MARKER = "<!-- claude-review-summary -->"
# What a re-review starts from (head, findings outside the diff, risk), base64 JSON,
# so model text can't close the HTML comment. Read back only from the bot's own comments.
STATE = re.compile(r"<!-- claude-review-state ([A-Za-z0-9+/=]+) -->")
# The bot's GraphQL logins: the review app's, and github-actions', which it posted as before the app.
BOTS = ("mstar-review", "github-actions")
TAG = re.compile(r"\*\*\[(P[0-3])\]\*\*")
PRIORITIES = ("P0", "P1", "P2", "P3")
# fixed_above: a PR stacked above fixes it, so the two merge together; reported, but not open when that
# PR is one the review was given the diff of (stacked-above.json).
STATUSES = ("fixed", "fixed_above", "open", "moot")
# Summaries posted before the modes were renamed carry the old names in their state.
RENAMED_MODES = {"thorough": "nogpu"}
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
                # Set only when a PR stacked above, whose code the review read, fixes it for real.
                "fixed_in": {"type": "integer", "minimum": 1, "description": "that PR number"},
            },
            "required": ["priority", "claim", "location", "detail"],
        }},
        "since_last": {"type": "string"},
        "description_problems": STRINGS,
        "questions": STRINGS,
        "checked": STRINGS,
        "skipped": {**STRINGS, "description": "files not reviewed: only with a small turn cap (a contributor run) "
                                              "and a diff too large for it; else empty"},
        "types": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True,
                  "items": {"type": "string", "enum": list(TYPES)}},
        "prior_findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "inline comment id, or the outside_diff location"},
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "status": {"type": "string", "enum": list(STATUSES)},
                "fixed_in": {"type": "integer", "minimum": 1,
                             "description": "with fixed_above: the PR stacked above that fixes it"},
                "why": {"type": "string"},
            },
            "required": ["ref", "priority", "status", "why"],
        }},
        "design": STRINGS,
        # Bugs in code the PR didn't write or worsen: reported, never counted or blocking.
        "latent": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "path:line"},
                "detail": {"type": "string"},
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "issue": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                          "description": "the open issue that tracks it (open-issues.json), or null"},
                "issue_title": {"type": "string", "description": "with no issue: a title for one"},
            },
            "required": ["location", "detail", "priority", "issue", "issue_title"],
        }},
        "risk": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
            # The design or direction decisions left to a human, when the risk isn't low: where each lives and why.
            "look_at": {"type": "array", "items": {"type": "object", "properties": {
                "location": {"type": "string", "description": "path:line, or a function or class"},
                "why": {"type": "string"},
            }, "required": ["location", "why"]}},
        }, "required": ["level", "reason", "look_at"]},
    },
    "required": ["outside_diff", "since_last", "description_problems", "questions", "checked", "skipped", "types",
                 "prior_findings", "design", "latent", "risk"],
}
FOOTER = "<sub>Advisory, not a merge gate. Rules: `AGENTS.md`. Reply to discuss.</sub>"


def stacked_above(path):
    """The numbers of the PRs stacked above whose diffs the review was given: stacked-above.json, which
    the workflow leaves out when a fetch failed. [] without it."""
    return [int(pr["number"]) for pr in json.loads(Path(path).read_text())] if path and Path(path).exists() else []


def judged(review, stacked):
    """The review's findings as they count. One said to be fixed in a PR stacked above is fixed there only
    when that PR is one of `stacked` (stacked_above); otherwise the claim wasn't checked, and the finding
    is open and says so."""
    def unchecked(f):
        return f"Said to be fixed in #{f.get('fixed_in', '?')}, not checked."
    return {**review,
            "outside_diff": [f if not f.get("fixed_in") or f["fixed_in"] in stacked else
                             {**{k: v for k, v in f.items() if k != "fixed_in"},
                              "detail": f"{f['detail']} {unchecked(f)}"} for f in review["outside_diff"]],
            "prior_findings": [f if f["status"] != "fixed_above" or f.get("fixed_in") in stacked else
                               {**f, "status": "open", "why": f"{f['why']} {unchecked(f)}"}
                               for f in review.get("prior_findings", [])]}


def open_findings(review, pages, stacked):
    """(where, priority) of every open finding: the bot's inline threads, this run's findings outside
    the diff, and the earlier findings a later review judged still open. A thread someone resolved stays
    open: only a later review that judged it fixed, fixed above or moot, or a commit that outdated it,
    clears it. An untagged bot thread counts as a P2, since its priority is unknown. A finding a PR
    stacked above fixes isn't open: it lands with that PR, when that PR is one of `stacked` (judged).
    `pages` are the PR's review threads, from `gh api graphql --paginate --slurp`."""
    review = judged(review, stacked)
    prior = review["prior_findings"]
    cleared = {f["ref"] for f in prior if f["status"] != "open"}
    found, counted = [], set()
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = thread["comments"]["nodes"][0]
            ref = str(first.get("databaseId", ""))
            if thread["isOutdated"] or ref in cleared or not (first["author"] and first["author"]["login"] in BOTS):
                continue
            match = TAG.match(first["body"])
            found.append((f"inline {ref}", match[1] if match else "P2"))
            counted.add(ref)
    found += [(f"outside the diff, {f['location']}", f["priority"]) for f in review["outside_diff"]
              if not f.get("fixed_in")]
    # Judged still open, though its thread went outdated or it was outside the diff.
    found += [(f"earlier {f['ref']}", f["priority"]) for f in prior
              if f["status"] == "open" and f["ref"] not in counted]
    return found


def status(review, pages, head, mode, stacked):
    found = [p for _, p in open_findings(review, pages, stacked)]
    review = judged(review, stacked)
    counts = {p: found.count(p) for p in PRIORITIES}
    if counts["P0"] or counts["P1"]:
        icon = "🔴"
    elif found or any(review[k] for k in ("description_problems", "questions", "skipped")):
        icon = "🟡"
    else:
        icon = "🟢"
    fixed_above = sum(f["status"] == "fixed_above" for f in review["prior_findings"]) \
        + sum(bool(f.get("fixed_in")) for f in review["outside_diff"])
    text = ", ".join(f"{n} {p}" for p, n in counts.items() if n) or "No open findings" + (
        f"; {fixed_above} fixed above" if fixed_above else "")
    line = f"{icon} **{text}** on `{head[:7]}`, {mode} review"
    if review.get("risk"):
        line += f" · design risk **{review['risk']['level']}**: {review['risk']['reason']}"
    return line


def above(finding):
    return f"Fixed in #{finding.get('fixed_in', '?')} above; merge together with it."


def latent(bug):
    issue = f"Tracked in #{bug['issue']}." if bug.get("issue") else \
        f"No issue yet; suggested title: {bug['issue_title']}"
    return f"**[{bug['priority']}]** `{bug['location']}`: {bug['detail']} {issue}"


def render(review, pages, head, mode, stacked):
    """The summary comment. `stacked`: the PRs stacked above whose diffs the review was given (stacked_above)."""
    parts = ["### Automated review", status(review, pages, head, mode, stacked)]
    review = judged(review, stacked)
    if review["since_last"]:
        parts.append(review["since_last"])
    prior = review.get("prior_findings", [])
    if prior:
        tally = ", ".join(f"{sum(f['status'] == s for f in prior)} {s.replace('_', ' ')}" for s in STATUSES)
        parts.append(f"Earlier findings: {tally}.")
    sections = [
        ("Still open from earlier", [f"**[{f['priority']}]** {f['ref']}: {f['why']}"
                                     for f in prior if f["status"] == "open"]),
        ("Outside the diff", [f"**[{f['priority']}]** `{f['location']}`: {f['claim']}\n\n  {f['detail']}"
                              for f in review["outside_diff"] if not f.get("fixed_in")]),
        ("Fixed in a PR above", [f"**[{f['priority']}]** {f['ref']}: {f['why']} {above(f)}"
                                 for f in prior if f["status"] == "fixed_above"]
         + [f"**[{f['priority']}]** `{f['location']}`: {f['claim']} {above(f)}\n\n  {f['detail']}"
            for f in review["outside_diff"] if f.get("fixed_in")]),
        ("Description", review["description_problems"]),
        ("Questions", review["questions"]),
        ("Design", review.get("design", [])),
        ("Where to look", [f"`{p['location']}`: {p['why']}" for p in (review.get("risk") or {}).get("look_at", [])]),
        ("Latent bugs on main (not from this PR)", [latent(b) for b in review.get("latent", [])]),
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
    risk = review.get("risk")  # its rating, not the look_at prose the summary already shows
    state = {"head": head, "mode": mode, "skipped": review["skipped"], "outside_diff": review["outside_diff"],
             "risk": risk and {"level": risk["level"], "reason": risk["reason"]}}
    # Model text can't open an HTML comment, so it can't forge a marker this bot or the workflow reads.
    text = "\n\n".join(parts).replace("<!--", "&lt;!--")
    return f"{MARKER}\n{text}\n\n<!-- claude-review-state {base64.b64encode(json.dumps(state).encode()).decode()} -->\n"


def read_state(body, mode):
    """The state a summary comment carries (its last), for a re-review in `mode` to start from; else
    None, and that run reviews the whole PR. None too when it doesn't decode, and for a nogpu
    review unless the earlier one was nogpu and skipped nothing."""
    found = STATE.findall(body)
    try:
        state = json.loads(base64.b64decode(found[-1])) if found else None
    except ValueError:
        return None
    if not isinstance(state, dict):
        return None
    state["mode"] = RENAMED_MODES.get(state.get("mode"), state.get("mode"))
    # A state from before `skipped` was kept counts as having skipped files.
    if mode == "nogpu" and (state.get("mode") != "nogpu" or state.get("skipped", [None])):
        return None
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true", help="print the fields the review returns, as a JSON schema")
    ap.add_argument("--head")
    ap.add_argument("--mode")
    ap.add_argument("--threads", help="the PR's review threads, from `gh api graphql --paginate --slurp`")
    ap.add_argument("--above", help="stacked-above.json: the PRs stacked above whose diffs the review was given; "
                                    "none when the file is missing")
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
        sys.stdout.write(render(json.load(sys.stdin), pages, a.head, a.mode, stacked_above(a.above)))


if __name__ == "__main__":
    main()
