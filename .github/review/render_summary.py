"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. The status line counts the open findings (`open_findings`), so a finding
from an earlier run counts until a later review judges it fixed, fixed above or moot.
bot.py passes SCHEMA to the review, posts what `render` makes of its fields, and starts
a re-review from what `read_state` reads back.
"""
import base64
import json
import os
import re
from pathlib import Path

# Lets the bot find earlier summaries to hide as outdated.
MARKER = "<!-- claude-review-summary -->"
# What a re-review starts from (head, findings outside the diff, risk), base64 JSON,
# so model text can't close the HTML comment. Read back only from the bot's own comments.
STATE = re.compile(r"<!-- claude-review-state ([A-Za-z0-9+/=]+) -->")
# Every login the bot posts as: the app, and github-actions, the job's own token, for what the app token
# can't reach (a team run's inline comments outlast its hour) and everything before the app. Enough for
# something a forged post can only add to, such as an inline finding.
POSTING_BOTS = ("mstar-review", "github-actions")
# Whose summaries, state and markers count (GraphQL logins). With the app configured, only the app's: any
# workflow on any branch posts as github-actions, so its comments could forge them. Forks have no app.
BOTS = ("mstar-review",) if os.environ.get("REVIEW_APP_CLIENT_ID") else POSTING_BOTS
TAG = re.compile(r"\*\*\[(P[0-3])\]\*\*")
PRIORITIES = ("P0", "P1", "P2", "P3")
# fixed_above: a PR stacked above fixes it, so the two merge together; reported, but not open when that
# PR is one the review was given the diff of (stacked-above.json).
STATUSES = ("fixed", "fixed_above", "open", "disputed", "moot")
# disputed: a P0-P2 the author argues against with no code change. It stays open, for a person to decide.
STILL_OPEN = ("open", "disputed")
# Summaries and handoffs saved before the modes were renamed carry the old names (current_mode).
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
        "questions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "path:line it applies to, or empty for the whole PR"},
                "text": {"type": "string"},
            },
            "required": ["location", "text"],
        }},
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
    bot.collect leaves out when a fetch failed. [] without it."""
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


def carried(review, earlier):
    """`review` with the earlier findings outside the diff it hasn't cleared added to its own: `earlier` is
    the last review's outside_diff. "A finding left out stays open", and stays in the state for the next
    re-review to judge."""
    cleared = {f["ref"] for f in review["prior_findings"] if f["status"] not in STILL_OPEN}
    locations = {f["location"] for f in review["outside_diff"]}
    return {**review, "outside_diff": [*review["outside_diff"], *[
        f for f in earlier if f["location"] not in cleared and f["location"] not in locations]]}


def open_findings(review, pages, stacked):
    """(where, priority, what) of every open finding: the bot's inline threads, outdated ones too, this
    run's findings outside the diff and the earlier ones `carried` added to them, and anything else a
    later review judged still open. A thread someone resolved stays open: only a later review that judged
    it fixed, fixed above or moot clears it. An untagged bot thread counts as a P2, since its priority is
    unknown. A finding a PR stacked above fixes isn't open: it lands with that PR, when that PR is one of
    `stacked` (judged). `pages` are the pages of the PR's review threads (GraphQL)."""
    review = judged(review, stacked)
    prior = review["prior_findings"]
    cleared = {f["ref"] for f in prior if f["status"] not in STILL_OPEN}
    disputed = {f["ref"]: f["why"] for f in prior if f["status"] == "disputed"}

    def what(ref, text):
        return f"disputed, you decide: {disputed[ref]}" if ref in disputed else text

    found, counted = [], {f["location"] for f in review["outside_diff"]}
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = thread["comments"]["nodes"][0]
            ref = str(first.get("databaseId", ""))
            if ref in cleared or not (first["author"] and first["author"]["login"] in POSTING_BOTS):
                continue
            match = TAG.match(first["body"])
            where = ":".join(str(x) for x in (first.get("path"), first.get("line")) if x) or f"inline {ref}"
            found.append((where, match[1] if match else "P2",
                          what(ref, first["body"][match.end() if match else 0:].strip().split("\n")[0])))
            counted.add(ref)
    found += [(f"outside the diff, {f['location']}", f["priority"], what(f["location"], f["claim"]))
              for f in review["outside_diff"] if not f.get("fixed_in")]
    # Judged still open, with no thread or outside_diff item here to count it by.
    found += [(f"earlier {f['ref']}", f["priority"], what(f["ref"], f["why"])) for f in prior
              if f["status"] in STILL_OPEN and f["ref"] not in counted]
    return found


def status(review, pages, head, mode, stacked):
    found = [p for _, p, _ in open_findings(review, pages, stacked)]
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


def checklist(review, pages, stacked, extra=()):
    """The "What's left for you" list: what a human must decide before approving, most important first,
    one line each with its place: open P0/P1, then `extra` (what the live-test job's rule found), the
    design decisions, questions, open P2s and skipped files. P3s and latent bugs aren't decisions. Past
    ten lines the rest fold away, so a question or decision, whose only home this is, is never lost."""
    found = open_findings(review, pages, stacked)
    review = judged(review, stacked)

    def where(item):
        return f"`{item['location']}`" if item.get("location") else "(PR-wide)"

    def findings(*priorities):
        return [f"**[{p}]** `{w}`: {what}" for wanted in priorities for w, p, what in found if p == wanted]

    items = [*findings("P0", "P1"), *extra,
             *[f"{where(d)}: {d['why']}" for d in (review.get("risk") or {}).get("look_at", [])],
             *[f"{where(q)}: {q['text']}" for q in review["questions"]],
             *findings("P2"), *[f"Not reviewed: `{path}`" for path in review["skipped"]]]
    title = "#### What's left for you"
    if not items:
        return f"{title}\nNothing left from the review."
    # Model text and file names: kept to one line, and unable to open an HTML comment.
    lines = [f"- [ ] {' '.join(str(item).split()).replace('<!--', '&lt;!--')}" for item in items]
    text = "\n".join([title, *lines[:10]])
    if lines[10:]:
        rest = "\n".join(lines[10:])
        text += f"\n\n<details><summary>and {len(lines) - 10} more</summary>\n\n{rest}\n\n</details>"
    return f"{text}\n\nApprove if you're OK with these."


def render(review, pages, head, mode, stacked):
    """The summary comment. `stacked`: the PRs stacked above whose diffs the review was given (stacked_above)."""
    parts = ["### Automated review", status(review, pages, head, mode, stacked)]
    review = judged(review, stacked)
    if review["since_last"]:
        parts.append(review["since_last"])
    prior = review.get("prior_findings", [])
    outside = {f["location"] for f in review["outside_diff"]}
    if prior:
        tally = ", ".join(f"{sum(f['status'] == s for f in prior)} {s.replace('_', ' ')}" for s in STATUSES)
        parts.append(f"Earlier findings: {tally}.")
    sections = [
        ("Still open from earlier", [f"**[{f['priority']}]** {f['ref']}: {f['why']}"
                                     for f in prior if f["status"] == "open" and f["ref"] not in outside]),
        ("Disputed: a person decides", [f"**[{f['priority']}]** {f['ref']}: {f['why']}"
                                        for f in prior if f["status"] == "disputed"]),
        ("Outside the diff", [f"**[{f['priority']}]** `{f['location']}`: {f['claim']}\n\n  {f['detail']}"
                              for f in review["outside_diff"] if not f.get("fixed_in")]),
        ("Fixed in a PR above", [f"**[{f['priority']}]** {f['ref']}: {f['why']} {above(f)}"
                                 for f in prior if f["status"] == "fixed_above"]
         + [f"**[{f['priority']}]** `{f['location']}`: {f['claim']} {above(f)}\n\n  {f['detail']}"
            for f in review["outside_diff"] if f.get("fixed_in")]),
        ("Description", review["description_problems"]),
        ("Design", review.get("design", [])),
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
    parts.append(checklist(review, pages, stacked))
    parts.append(FOOTER)
    risk = review.get("risk")  # its rating, not the look_at prose the summary already shows
    state = {"head": head, "mode": mode, "skipped": review["skipped"], "outside_diff": review["outside_diff"],
             "risk": risk and {"level": risk["level"], "reason": risk["reason"]}, "latent": review.get("latent", [])}
    # Model text can't open an HTML comment, so it can't forge a marker this bot or the workflow reads.
    text = "\n\n".join(parts).replace("<!--", "&lt;!--")
    return f"{MARKER}\n{text}\n\n<!-- claude-review-state {base64.b64encode(json.dumps(state).encode()).decode()} -->\n"


def current_mode(mode):
    """`mode` by its current name: summaries and handoffs saved before the modes were renamed carry the old one."""
    return RENAMED_MODES.get(mode, mode)


def read_state(body):
    """The state a summary comment carries (its last), for a re-review to start from; else None, and
    that run reviews the whole PR. None too when it doesn't decode and when the earlier review skipped
    files. Any mode can build on any other, since every mode runs every pass."""
    found = STATE.findall(body)
    try:
        state = json.loads(base64.b64decode(found[-1])) if found else None
    except ValueError:
        return None
    if not isinstance(state, dict):
        return None
    state["mode"] = current_mode(state.get("mode"))
    # A state without `latent` is from before light mode ran every pass, so it counts as a partial review;
    # one without `skipped` is older still.
    if state.get("skipped", [None]) or "latent" not in state:
        return None
    return state
