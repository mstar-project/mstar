"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. The status line counts the open findings (`open_findings`, which the
approval rule in gpu_review.py uses too), so a finding from an earlier run counts
until a later review judges it fixed, fixed above or moot. bot.py passes SCHEMA to
the review, posts what `render` makes of its fields, and starts a re-review from
what `read_state` reads back.
"""
import base64
import itertools
import json
import os
import re
from pathlib import Path

# Lets the bot find earlier summaries to hide as outdated.
MARKER = "<!-- claude-review-summary -->"
# What a re-review starts from (head, findings outside the diff, risk, live-test call), base64 JSON,
# so model text can't close the HTML comment. Read back only from the bot's own comments.
STATE = re.compile(r"<!-- claude-review-state ([A-Za-z0-9+/=]+) -->")
# Every login the bot posts as: the app, and github-actions, the job's own token, when the app has no token to
# give (bot.post falls back to it, and everything before the app). Enough for something a forged post can only
# add to, such as an inline finding.
POSTING_BOTS = ("mstar-review", "github-actions")
# Whose summaries, state and markers count (GraphQL logins). With the app configured, only the app's: any
# workflow on any branch posts as github-actions, so its comments could forge them. Forks have no app.
BOTS = ("mstar-review",) if os.environ.get("REVIEW_APP_CLIENT_ID") else POSTING_BOTS
TAG = re.compile(r"\*\*\[(P[0-3])\]\*\*")
PRIORITIES = ("P0", "P1", "P2", "P3")
# fixed_above: a PR stacked above fixes it, so the two merge together; reported, but not open when that
# PR is one the review was given the diff of (stacked-above.json).
STATUSES = ("fixed", "fixed_above", "open", "disputed", "moot")
# The bot's approval still counts these open when fixed above: a PR must fix its own (gpu_review.fixed_only_above).
MUST_FIX_HERE = ("P0", "P1")
# disputed: a P0-P2 the author argues against with no code change. It stays open, for a person to decide.
STILL_OPEN = ("open", "disputed")
# What takes an earlier finding outside the diff out of the state (`carried`).
CLEARS = ("fixed", "moot")
# Handoffs saved before the modes were renamed carry the old names (current_mode).
RENAMED_MODES = {"thorough": "nogpu"}
TYPES = ("new-model", "bug", "perf", "feature", "refactor", "test", "docs", "ci")
# The first character of anything that can start markup, a link, or a reference to a person, issue or commit.
# `#` not before a space or another `#`: those are a Markdown heading, which links nothing.
MARKUP = re.compile(r"(?i)(?:[@<&\\]|#(?![\s#]|$)|\](?=[(:])|!(?=\[)|:(?=//)|(?<![a-z0-9])(?:g(?=h-)|w(?=ww\.)))"
                    r"(?!\u200b)")
# A live-test report's fenced code, read as GitHub's CommonMark does (`report_blocks`): lines end at \n, \r or \r\n,
# which isn't str.splitlines.
LINE = re.compile(r"[^\r\n]*(?:\r\n?|\n|\Z)")
FENCE_OPEN = re.compile(r"( {0,3})(?:(`{3,})([^`]*)|(~{3,})(.*))")
FENCE_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")
# The start of a run that could open a fence, and only the start: splitting it there leaves no run a fence can open,
# and a run already split isn't matched again.
FENCE_RUN = re.compile(r"(?<![\u200b`~])([`~])(?=\1\1)")
LIST_ITEM = re.compile(r"[ \t]*(?:[-+*]|\d+[.)])(?:[ \t]|$)")
# The HTML a report may keep, exactly so, for a folded section. At the start of a line it opens an HTML block
# that lasts until a blank line.
BARE_TAG = re.compile(r"</?(?:details|summary)>")
# A fence language GitHub leaves as code. These it draws instead, and anything else (`&#109;ermaid`) may spell them.
PLAIN_LANGUAGE = re.compile(r"[A-Za-z0-9_+.-]*")
DRAWN_LANGUAGES = ("mermaid", "geojson", "topojson", "stl", "math")
STRINGS = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object",
    "properties": {
        # Findings on lines of pr.diff, posted by the workflow (bot.post_inline), never by the model.
        "inline_comments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer", "minimum": 1, "description": "the line, or the last line of a range"},
                "start_line": {"type": "integer", "minimum": 1, "description": "the first line, for a range"},
                "side": {"type": "string", "enum": ["RIGHT", "LEFT"], "description": "LEFT for a deleted line"},
                "body": {"type": "string"},
            },
            "required": ["path", "line", "body"],
        }},
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
        "answered_questions": {**STRINGS, "description": "on a re-review, each question in last-review.json that a "
                                                          "comment from a maintainer other than the author answered, "
                                                          "verbatim; empty when none"},
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
            "security": {"type": "boolean", "description": "the change touches security-relevant code (credentials, "
                         "auth, API keys and secrets handling, redaction, the review bot workflows and sandbox): "
                         "a person approves it, whatever the design"},
            # The design or direction decisions left to a human, when the risk isn't low: where each lives and why.
            "look_at": {"type": "array", "items": {"type": "object", "properties": {
                "location": {"type": "string", "description": "path:line, or a function or class"},
                "why": {"type": "string"},
            }, "required": ["location", "why"]}},
        }, "required": ["level", "reason", "security", "look_at"]},
        "live_test": {"type": "object", "properties": {
            "run": {"type": "string", "enum": ["none", "targeted", "full"]},
            "reason": {"type": "string"},
            # The kinds of check the box runs; ignored for `none`.
            "checks": {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": True,
                       "items": {"type": "string", "enum": ["correctness", "performance"]}},
            "gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            # What the review would have the live test try: suggestions for its plan, read as data.
            "focus": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
                "what": {"type": "string", "description": "an input, a claim to measure, or a config or path "
                                                          "the change only shows on"},
                "why": {"type": "string", "description": "one line"},
            }, "required": ["what", "why"]}},
            # What the live test can't exercise in this PR; it stops the bot's approval, not the GPU run.
            "untestable": {"type": "string", "description": "what in this PR the live test cannot exercise: hardware "
                                                            "other than the GPUs the bot runs on, more GPUs or nodes "
                                                            "than a run gets, weights or services it cannot get, "
                                                            "anything else. Short; empty when there is nothing"},
        }, "required": ["run", "reason", "checks", "gpus", "focus", "untestable"]},
    },
    "required": ["inline_comments", "outside_diff", "since_last", "description_problems", "questions",
                 "answered_questions", "checked",
                 "skipped", "types", "prior_findings", "design", "latent", "risk"],
}


def inert(value):
    """Untrusted text, or the strings in a list or dict of it, with a zero-width space after the first character of
    every `@`, `#`, `<`, `&`, `\\`, `](`, `]:`, `![`, `://`, `GH-` and `www.`, in any case. (`#` before whitespace,
    another `#` or the end of the text is left alone: that's a heading or plain, and can't reference anything.) So it
    can't ping anyone, mark another issue's timeline, link (inline or by a reference definition), load an image, open
    HTML or a comment, or spell any of those as an entity or a backslash escape. Nothing is escaped, so text in a code
    span still reads right and a second pass changes nothing."""
    if isinstance(value, str):
        return MARKUP.sub("\\g<0>\u200b", value)
    if isinstance(value, list):
        return [inert(v) for v in value]
    if isinstance(value, dict):
        return {k: inert(v) for k, v in value.items()}
    return value


def report_blocks(text):
    """`text`, at the top level of a comment, as ([piece, in a fence] for each run of lines, the fence left open or
    ""). A fenced code block by CommonMark's rule is left out of `inert` only where GitHub is sure to read it as one.
    An opener that isn't sure ends that for the rest of the text, which is then just `inert`, fence or not, with
    every run that could open a fence split, so GitHub opens none either and none swallows the bot's text after the
    report: one indented under what may be a list item (the item's end also ends the fence, so the next line is
    outside it), one in the HTML block a bare tag opens (no fence at all), one GitHub draws instead of showing.
    Skipping only that line wouldn't do: if GitHub does read it as a fence, GitHub's closing line would look like an
    opener here, and the text after it, outside any fence, would go through unneutralised."""
    lines, fence, sure, listed, html = [], "", True, False, False
    for line in LINE.findall(text):
        body, code = line.rstrip("\r\n"), bool(fence)
        if fence:
            close = FENCE_CLOSE.fullmatch(body)
            if close and close[1][0] == fence[0] and len(close[1]) >= len(fence):
                fence = ""
        else:
            opener = FENCE_OPEN.fullmatch(body) if sure else None
            if opener:
                language = re.match(r"[^ \t]*", (opener[3] or opener[5] or "").strip(" \t"))[0]
                sure = not (opener[1] and listed or html or not PLAIN_LANGUAGE.fullmatch(language)
                            or language.lower() in DRAWN_LANGUAGES)
                if sure:
                    fence, code = opener[2] or opener[4], True
            listed = listed or bool(LIST_ITEM.match(body))
            html = bool(BARE_TAG.search(body)) or (html and body.strip(" \t") != "")
        lines.append((line if sure else FENCE_RUN.sub("\\1\u200b", line), code))
    return [("".join(line for line, _ in run), code) for code, run in itertools.groupby(lines, lambda x: x[1])], fence


def inert_report(text):
    """The box's report as `inert` makes it, except what GitHub shows as written anyway: its fenced code (so a command
    in one still copies) and the exact tags `<details>`, `</details>`, `<summary>` and `</summary>`. A fence can't
    ping, link or load anything, but the bot reads its markers from a comment's raw text, so `<!--` is split there
    too. Leading byte order marks go first: GitHub skips one at the start of a comment, and a fence after it would
    open there and not here. A second pass changes nothing."""
    def neutral(m):
        return m[0] if m[0] == "<" and BARE_TAG.match(m.string, m.start()) else m[0] + "\u200b"

    return "".join(piece.replace("<!--", "<\u200b!--") if code else MARKUP.sub(neutral, piece)
                   for piece, code in report_blocks(text.lstrip("\ufeff"))[0])


def report_closers(text):
    """What ends the fence and the `<details>` the report `text` left open, so the bot's text after it (and a cut) stays
    outside them."""
    pieces, fence = report_blocks(text)
    depth = 0
    for piece, code in pieces:
        for tag in BARE_TAG.findall(piece) if not code else ():
            depth += (tag == "<details>") - (tag == "</details>" and depth > 0)
    closers = [*([fence] if fence else []), *["</details>"] * min(depth, 200)]  # 200: stays within a comment's length
    return "".join(f"\n{c}" for c in closers) + "\n" if closers else ""


def stacked_above(path):
    """The numbers of the PRs stacked above whose diffs the review was given: stacked-above.json, which
    bot.collect leaves out when a fetch failed. [] without it."""
    return [int(pr["number"]) for pr in json.loads(Path(path).read_text())] if path and Path(path).exists() else []


def judged(review, stacked):
    """The review's findings as they count. One said to be fixed in a PR stacked above is fixed there only
    when that PR is one of `stacked` (stacked_above); otherwise the claim wasn't checked, and the finding
    is open and says so. The note is saved with an outside-the-diff item, so a later run doesn't repeat it."""
    def unchecked(f, text):
        note = f"Said to be fixed in #{f.get('fixed_in', '?')}, not checked."
        return text if note in text else f"{text} {note}"
    return {**review,
            "outside_diff": [f if not f.get("fixed_in") or f["fixed_in"] in stacked else
                             {**{k: v for k, v in f.items() if k != "fixed_in"},
                              "detail": unchecked(f, f["detail"])} for f in review["outside_diff"]],
            "prior_findings": [f if f["status"] != "fixed_above" or f.get("fixed_in") in stacked else
                               {**f, "status": "open", "why": unchecked(f, f["why"])}
                               for f in review.get("prior_findings", [])]}


RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def over_gpu_cap(review, max_gpus):
    """`review` with `live_test.untestable` naming a GPU count above what a run gets (`max_gpus`): a pass on the
    smaller box doesn't cover it. Before `carried`, so the saved state keeps it for later reviews."""
    live = review.get("live_test") or {}
    gpus, cap = live.get("gpus"), int(max_gpus)
    note = f"needs {gpus} GPUs, runs get {cap}"
    if gpus and gpus > cap and note not in (live.get("untestable") or ""):
        live["untestable"] = "; ".join(x for x in (live.get("untestable"), note) if x)
    return review


def carried(review, earlier, answered, pages):
    """`review` with what the earlier review's state (`earlier`, last-review.json) still holds: its findings
    outside the diff this one hasn't cleared ("a finding left out stays open"; only `fixed` and `moot` clear
    one, one this run judged fixed above carries that `fixed_in`, which judged checks, and one it judged
    still open loses any `fixed_in` from before); its risk level as a floor, with the `look_at` that names
    the decision, unless a maintainer other than the author has commented since (`answered`), when this review's
    judgement stands (its `security` flag carries whatever anyone has said: a comment isn't an approval);
    its questions, each until a maintainer's comment has answered it and this review says so
    (`answered_questions`), so one comment can't drop the rest and an answered one is shown no more (a question
    no longer holds the bot's approval, but it stays on the checklist until answered); and what the live test
    can't exercise (`live_test.untestable`) when this review names nothing, whatever anyone has said, since no
    comment makes the bot's GPUs able to run it. Everything here is carried into the next state too.
    An earlier finding's priority is the most severe of the bot's record of it (the tag on its thread in `pages`,
    the earlier state's items at its location) and this review's word on it (its verdict, an item restating it):
    a re-review can raise a priority, never lower it. An item at a location this review cleared (`fixed`, `moot`)
    is a new finding, which the earlier one doesn't raise; that verdict keeps the earlier priority."""
    earlier = earlier or {}
    verdicts = {f["ref"]: f for f in review["prior_findings"]}
    locations = {f["location"] for f in review["outside_diff"]}
    cleared = {ref for ref, f in verdicts.items() if f["status"] in CLEARS}
    top = {}
    for ref, p in [*inline_findings(pages).items(), *[(f["location"], f["priority"]) for f in earlier.get(
            "outside_diff", [])], *[(ref, f["priority"]) for ref, f in verdicts.items()]]:
        top[ref] = min(top.get(ref, p), p)

    def raised(f, ref):
        return {**f, "priority": min(f["priority"], top.get(ref, f["priority"]))}

    def with_verdict(f):
        verdict = verdicts.get(f["location"])
        if verdict is None:
            return f
        kept = {k: v for k, v in f.items() if k != "fixed_in"}
        fixed_in = verdict["status"] == "fixed_above" and verdict.get("fixed_in")
        return {**kept, "fixed_in": fixed_in} if fixed_in else kept

    review = {**review, "prior_findings": [raised(f, f["ref"]) for f in review["prior_findings"]], "outside_diff": [
        *[f if f["location"] in cleared else raised(f, f["location"]) for f in review["outside_diff"]],
        *[raised(with_verdict(f), f["location"]) for f in earlier.get("outside_diff", [])
          if f["location"] not in locations and f["location"] not in cleared]]}
    old, new = earlier.get("risk"), review.get("risk")
    if old and new:  # a comment isn't an approval: security-relevant code stays a person's to approve
        review["risk"] = new = {**new, "security": bool(new.get("security") or old.get("security"))}
    if not answered:
        if old and new and RISK_ORDER[old["level"]] > RISK_ORDER[new["level"]]:
            review["risk"] = {**new, "level": old["level"], "reason": f"Kept from the earlier review: {old['reason']}",
                              "look_at": [*new.get("look_at", []), *old.get("look_at", [])]}
    asked = {q["text"] for q in review["questions"]}
    settled = set(review.get("answered_questions") or ()) if answered else set()
    review["questions"] = [*review["questions"], *[q for q in earlier.get("questions", [])
                                                   if q["text"] not in asked and q["text"] not in settled]]
    untestable = (earlier.get("live_test") or {}).get("untestable")
    if untestable and not (review.get("live_test") or {}).get("untestable"):
        review["live_test"] = {**(review.get("live_test") or {}), "untestable": untestable}
    return review


def bot_threads(pages):
    """(first comment, priority, text after the tag, whether GitHub shows it outdated) of every inline thread the bot
    started, a page of the PR's review threads (GraphQL) at a time. An untagged one counts as a P2, since its priority
    is unknown and it should still block approval."""
    for page in pages:
        for thread in page["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]:
            first = thread["comments"]["nodes"][0]
            if first["author"] and first["author"]["login"] in POSTING_BOTS:
                match = TAG.match(first["body"])
                yield first, match[1] if match else "P2", first["body"][match.end() if match else 0:], \
                    thread.get("isOutdated")


def inline_findings(pages):
    """The bot's inline findings as they stand, comment id to priority tag, what a later step checks them against
    (gpu_review.findings_changed)."""
    return {str(first.get("databaseId", "")): priority for first, priority, _, _ in bot_threads(pages)}


EDITED = "edited after the bot posted it; a person decides"
UNFINISHED = "the review didn't finish: it left files unreviewed"


def open_findings(review, pages, stacked):
    """(where, priority, what) of every open finding: the bot's inline threads, outdated ones too, this
    run's findings outside the diff and the earlier ones `carried` added to them, and anything else a
    later review judged still open. A thread someone resolved stays open: only a later review that judged
    it fixed, fixed above or moot clears it. A bot thread someone edited counts as an open P0, whatever a
    later review judged, since the text the review read isn't the bot's. A finding a PR stacked above fixes
    isn't open: it lands with that PR, when that PR is one of `stacked` (judged). Each finding (a thread, an
    earlier finding's ref, a place outside the diff) counts once, as the most severe of the open rows that name it
    (its thread's tag, a verdict on it, the items at that place), with that row's place and text. `pages` are the
    pages of the PR's review threads (GraphQL)."""
    review = judged(review, stacked)
    prior = review["prior_findings"]
    cleared = {f["ref"] for f in prior if f["status"] not in STILL_OPEN}
    disputed = {f["ref"]: f["why"] for f in prior if f["status"] == "disputed"}

    def what(ref, text):
        return f"disputed, you decide: {disputed[ref]}" if ref in disputed else text

    rows = []  # (the finding it names, where, priority, what)
    for first, priority, text, _ in bot_threads(pages):
        ref = str(first.get("databaseId", ""))
        edited = first.get("lastEditedAt")
        if ref in cleared and not edited:
            continue
        where = ":".join(str(x) for x in (first.get("path"), first.get("line")) if x) or f"inline {ref}"
        rows.append((ref, where, "P0", EDITED) if edited
                    else (ref, where, priority, what(ref, text.strip().split("\n")[0])))
    rows += [(f["location"], f"outside the diff, {f['location']}", f["priority"], what(f["location"], f["claim"]))
             for f in review["outside_diff"] if not f.get("fixed_in")]
    rows += [(f["ref"], f"earlier {f['ref']}", f["priority"], what(f["ref"], f["why"])) for f in prior
             if f["status"] in STILL_OPEN]
    found = {}  # the most severe row's place, priority and text; on a tie, the first row's
    for key, where, priority, text in rows:
        if key not in found or priority < found[key][1]:
            found[key] = (where, priority, text)
    return list(found.values())


def status(review, pages, head, mode, stacked):
    found = [p for _, p, _ in open_findings(review, pages, stacked)]
    review = inert(judged(review, stacked))
    counts = {p: found.count(p) for p in PRIORITIES}
    if counts["P0"] or counts["P1"]:
        icon = "🔴"
    elif found or any(review[k] for k in ("description_problems", "questions", "skipped")):
        icon = "🟡"
    else:
        icon = "🟢"
    fixed_above = len(prior_above(review)) + sum(bool(f.get("fixed_in")) for f in review["outside_diff"])
    text = ", ".join(f"{n} {p}" for p, n in counts.items() if n) or "No open findings" + (
        f"; {fixed_above} fixed above" if fixed_above else "")
    line = f"{icon} **{text}** on `{head[:7]}`, {mode} review"
    if review.get("risk"):
        line += f" · design risk **{review['risk']['level']}**: {review['risk']['reason']}"
        line += " · touches security-relevant code" if review["risk"].get("security") else ""
    return line


def prior_above(review):
    """The earlier findings judged fixed above, but not one `carried` also kept in outside_diff (its row there
    counts)."""
    outside = {f["location"] for f in review["outside_diff"]}
    return [f for f in review["prior_findings"] if f["status"] == "fixed_above" and f["ref"] not in outside]


def above(finding):
    must = " A P0/P1 has to be fixed in this PR itself, not only above." if finding["priority"] in MUST_FIX_HERE \
        else ""
    return f"Fixed in #{finding.get('fixed_in', '?')} above; merge together with it.{must}"


def latent(bug):
    issue = f"Tracked in #{bug['issue']}." if bug.get("issue") else \
        f"No issue yet; suggested title: {bug['issue_title']}"
    return f"**[{bug['priority']}]** `{bug['location']}`: {bug['detail']} {issue}"


def checklist(review, pages, stacked, extra=(), approving=False):
    """The "What's left for you" list: what the review leaves to a person (the bot approves over its questions and
    a medium design risk), most important first, one line each with its place: open P0/P1, findings a PR above
    fixes (merge together), then `extra` (what the live-test job's rule found), the design decisions, questions,
    open P2s and skipped files. P3s and latent bugs aren't decisions. Past ten lines the rest fold away, so a
    question or decision, whose only home this is, is never lost. `approving`: the bot may approve over them, which
    the closing line says."""
    found = open_findings(review, pages, stacked)
    review = inert(judged(review, stacked))

    def where(item):
        return f"`{item['location']}`" if item.get("location") else "(PR-wide)"

    def findings(*priorities):
        return [f"**[{p}]** `{inert(w)}`: {inert(what)}"
                for wanted in priorities for w, p, what in found if p == wanted]

    items = [*findings("P0", "P1"),
             *[f"**[{f['priority']}]** {f['ref']}: {above(f)}" for f in prior_above(review)],
             *[f"**[{f['priority']}]** `{f['location']}`: {above(f)}" for f in review["outside_diff"]
               if f.get("fixed_in")],
             *inert(list(extra)),
             *[f"{where(d)}: {d['why']}" for d in (review.get("risk") or {}).get("look_at", [])],
             *[f"{where(q)}: {q['text']}" for q in review["questions"]],
             *findings("P2"), *[f"Not reviewed: `{path}`" for path in review["skipped"]]]
    title = "#### What's left for you"
    if not items:
        return f"{title}\nNothing left from the review."
    # Model text and file names: kept to one line (`inert` above, so unable to open an HTML comment).
    lines = [f"- [ ] {' '.join(str(item).split())}" for item in items]
    text = "\n".join([title, *lines[:10]])
    if lines[10:]:
        rest = "\n".join(lines[10:])
        text += f"\n\n<details><summary>and {len(lines) - 10} more</summary>\n\n{rest}\n\n</details>"
    return text + ("\n\nApprove if you're OK with these. If the bot approves, it does so over open questions "
                   "and a medium design risk, so object on the PR if you aren't." if approving else "")


def render(review, pages, head, mode, stacked, salvaged=False, team=False):
    """The summary comment. `stacked`: the PRs stacked above whose diffs the review was given (stacked_above).
    `salvaged`: the review went past its turn limit, so the action failed it and no live test follows. `team`: the
    run had the team's turn cap, which reviews every file, so files it skipped mean it didn't finish."""
    parts = ["### Automated review", status(review, pages, head, mode, stacked)]
    if team and review["skipped"]:
        parts.append(f"⚠️ **Incomplete review**: {UNFINISHED}, which is the bot's to finish, not the author's. "
                     "Comment `/review` again.")
    raw = judged(review, stacked)  # the saved state and the checklist; what's shown goes through `inert`
    review = inert(raw)
    if review["since_last"].strip(" \t\r\n\"'"):  # a model sometimes returns a pair of quotes for "nothing"
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
                                 for f in prior_above(review)]
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
    live = review.get("live_test")
    if live and "run" in live and mode == "full":  # without `run`: only what `carried` kept
        checks = f" ({', '.join(live.get('checks') or [])})" if live["run"] != "none" and live.get("checks") else ""
        after = (" No live test runs: the review went past its turn limit." if salvaged
                 else " The live-test job posts its outcome in a separate comment.")
        parts.append(f"Live test: **{live['run']}**{checks}. {live['reason']}{after}")
    left = checklist(raw, pages, stacked, approving=mode == "full")
    parts.append(f"{left} The live-test comment adds its own items." if mode == "full" and not salvaged else left)
    # Only full mode can lead to an approval, by gpu_review.py's rule.
    role = ("Any approval is posted by the live-test job, by the rule in `gpu_review.py`." if mode == "full"
            else "Advisory: this review doesn't approve or block.")
    parts.append(f"<sub>{role} Rules: `AGENTS.md`. Reply to discuss.</sub>")
    risk = raw.get("risk")
    # An earlier finding outside the diff that a re-review cleared though nothing touched its file (`untouched`,
    # bot.untouched) stays in the state, at its priority, so the next review still has it: the approval doesn't
    # accept that clear (gpu_review.for_approval).
    # A new item this run put at that place is another finding (carried), so it doesn't stand in for this one.
    cleared = {f["ref"]: f["priority"] for f in raw.get("prior_findings", []) if f["status"] in CLEARS}
    kept = [{**f, "priority": min(f["priority"], cleared[f["location"]])}
            for f in raw.get("untouched") or () if f["location"] in cleared]
    state = {"head": head, "mode": mode, "skipped": raw["skipped"], "outside_diff": [*raw["outside_diff"], *kept],
             "risk": risk and {k: risk[k] for k in ("level", "reason", "security", "look_at") if k in risk},
             "latent": raw.get("latent", []),
             "questions": raw["questions"],
             "live_test": raw.get("live_test"),
             "moot": [f for f in raw.get("prior_findings", []) if f["status"] == "moot"]}
    # `inert` keeps model text from opening an HTML comment, so it can't forge a marker this bot or the workflow reads.
    text = "\n\n".join(parts)
    return f"{MARKER}\n{text}\n\n<!-- claude-review-state {base64.b64encode(json.dumps(state).encode()).decode()} -->\n"


def current_mode(mode):
    """`mode` by its current name: handoffs saved before the modes were renamed carry the old one."""
    return RENAMED_MODES.get(mode, mode)


def saved_state(body):
    """The state a summary comment carries (its last), or None when it has none that decodes."""
    found = STATE.findall(body)
    try:
        state = json.loads(base64.b64decode(found[-1])) if found else None
    except ValueError:
        return None
    if not isinstance(state, dict):
        return None
    return state


def read_state(body):
    """The saved state, when a re-review can build on it; else None, and that run reviews the whole PR. Not
    when it doesn't decode or the earlier review skipped files. Any mode can build on any other, since every
    mode runs every pass."""
    state = saved_state(body)
    if state is None or state.get("skipped"):
        return None
    return state
