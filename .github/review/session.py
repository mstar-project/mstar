#!/usr/bin/env python3
"""What a review run used, and the handoff a stopped run leaves for `/review continue`.

Both read Claude Code's record of the run: the action's execution file (a JSON
array of messages) and the session transcripts under ~/.claude/projects (one
message per line), in any mix; a directory is searched for transcripts. One API
response spans several messages, one per content block, so responses are
counted by their message id. Run by claude-review.yml with the default branch
checked out:

    python3 .github/review/session.py usage --pr N --head SHA --mode light \\
        --out usage.json --summary "$GITHUB_STEP_SUMMARY" SOURCE...
    python3 .github/review/session.py handoff --pr N --head SHA --mode light --diff pr.diff \\
        --out DIR --comment partial.md [--previous handoff.json] SOURCE...
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

TOKENS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
# The same counts in the result message's per-model ``modelUsage``.
MODEL_USAGE = {"input_tokens": "inputTokens", "cache_creation_input_tokens": "cacheCreationInputTokens",
               "cache_read_input_tokens": "cacheReadInputTokens", "output_tokens": "outputTokens"}
# The prompt asks for notes that start with "Progress:"; tolerate Markdown around it.
PROGRESS = re.compile(r"[#*_\s]*progress[*_\s]*:", re.IGNORECASE)
NOTE_CHARS = 6000
RECENT_TEXT_CHARS = 3000
ELIDED = " [...] "
# The escapes git writes in a path it quotes: \" \\ \a \b \t \n \v \f \r, and other bytes in octal.
C_ESCAPE = re.compile(rb"\\([0-3][0-7]{2}|.)", re.DOTALL)
C_CHARS = dict(zip(b"abtnvfr", b"\a\b\t\n\v\f\r", strict=True))


def load(paths):
    """Every message in ``paths``, in file order. Missing files and unparsable lines are skipped."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob("*.jsonl"))
        elif p.is_file():
            files.append(p)
    messages = []
    for f in files:
        text = f.read_text(errors="replace")
        try:
            data = json.loads(text)
            messages += data if isinstance(data, list) else [data]
            continue
        except ValueError:
            pass
        for line in text.splitlines():
            try:
                messages.append(json.loads(line))
            except ValueError:
                pass
    return [m for m in messages if isinstance(m, dict)]


def blocks(message):
    # Not every message's "message" is an object: a permission_denied system message's is a string.
    msg = message.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def from_api(m):
    """An assistant message the API returned. Claude Code writes an API error as an assistant
    message of model ``<synthetic>``."""
    msg = m.get("message")
    return m.get("type") == "assistant" and isinstance(msg, dict) and msg.get("model") != "<synthetic>"


def usage(messages):
    """API calls (one per API response), token counts and the run's result fields. Tokens come
    from the result's per-model counts, which cover every call the session made; only a run that
    ended without a result (a timeout, a cancel) has them summed from its messages."""
    calls = {}
    for m in messages:
        if not from_api(m):
            continue
        u, mid = m["message"].get("usage"), m["message"].get("id")
        if not isinstance(u, dict) or not mid:
            continue
        seen = calls.setdefault(mid, dict.fromkeys(TOKENS, 0))
        for k in TOKENS:
            if isinstance(u.get(k), int):
                # Lines of one response repeat its usage; output_tokens can grow along them.
                seen[k] = max(seen[k], u[k])
    result = next((m for m in reversed(messages) if m.get("type") == "result"), {})
    models = result.get("modelUsage")
    if isinstance(models, dict) and models:
        models = {name: {**{k: u.get(f) if isinstance(u.get(f), int) else 0 for k, f in MODEL_USAGE.items()},
                         "cost_usd": u.get("costUSD")}
                  for name, u in models.items() if isinstance(u, dict)}
        counts, source = list(models.values()), "result"
    else:
        models, counts, source = {}, list(calls.values()), "messages"
    return {
        "api_calls": len(calls),
        **{k: sum(c[k] for c in counts) for k in TOKENS},
        "tokens_from": source,
        "models": models,
        "num_turns": result.get("num_turns"),
        "total_cost_usd": result.get("total_cost_usd"),
        "subtype": result.get("subtype"),
    }


def usage_table(r):
    def n(v):
        return "?" if v is None else f"{v:,}"

    cost = r.get("total_cost_usd")
    rows = [
        ("PR", f"#{r['pr']} at `{(r['head_sha'] or '?')[:7]}`"),
        ("Mode", r["mode"] + (", `/review continue`" if r["continue"] else "")),
        ("Review step", r["outcome"] + (f" ({r['subtype']})" if r.get("subtype") else "")),
        ("Turns", n(r.get("num_turns"))),
        ("API calls", n(r.get("api_calls"))),
        ("Input tokens, uncached", n(r.get("input_tokens"))),
        ("Input tokens, cache writes", n(r.get("cache_creation_input_tokens"))),
        ("Input tokens, cache reads", n(r.get("cache_read_input_tokens"))),
        ("Output tokens", n(r.get("output_tokens"))),
        ("Tokens counted from", r.get("tokens_from") or "?"),
        ("Cost", "?" if cost is None else f"${cost:.2f}"),
    ]
    if r.get("error"):
        rows.append(("Error reading the run", r["error"]))
    return "### Review usage\n\n| | |\n| --- | --- |\n" + "".join(f"| {k} | {v} |\n" for k, v in rows)


def handoff(messages, root, diff_paths, previous=None):
    """What a stopped run leaves for the next one: the inline comments it made (and whether each
    posted), the files it read, the latest progress note and what was written since (the model
    often keeps no note, but says what it is doing as it goes). Merged onto ``previous``, the
    handoff the run itself started from, if any."""
    prev = previous or {}
    results = {}
    for m in messages:
        for b in blocks(m):
            if b.get("type") == "tool_result" and b.get("tool_use_id"):
                results[b["tool_use_id"]] = b
    comments = {c.get("id") or f"earlier-{i}": c for i, c in enumerate(prev.get("comments", []))}
    reads = set(prev.get("reads", []))
    # The earlier runs' text since their note, one line per item, so the newest of it is kept.
    note, since_note = None, prev["recent_text"].split("\n") if prev.get("recent_text") else []
    prefix = root.rstrip("/") + "/"
    for m in messages:
        main_thread = not m.get("parent_tool_use_id") and not m.get("isSidechain")
        if not from_api(m):
            continue
        for b in blocks(m):
            kind, inp = b.get("type"), b.get("input")
            inp = inp if isinstance(inp, dict) else {}
            if kind == "tool_use" and str(b.get("name", "")).endswith("create_inline_comment"):
                res = results.get(b.get("id")) or {}
                # The tool reports a posted comment's id. A call without `confirmed: true` is only
                # buffered, to be posted or dropped after the session, so it stays unknown.
                status = ("failed" if res.get("is_error")
                          else "posted" if '"comment_id"' in result_text(res.get("content")) else "unknown")
                key = b.get("id") or f"no-id-{len(comments)}"
                comments[key] = {
                    "id": key, "path": inp.get("path"), "line": inp.get("line") or inp.get("startLine"),
                    "body": shorten(str(inp.get("body", "")).strip().split("\n")[0], 160), "status": status,
                    "error": shorten(first_line(res.get("content")), 200) if status == "failed" else "",
                }
            elif kind == "tool_use" and b.get("name") == "Read" and inp.get("file_path"):
                path = str(inp["file_path"])
                reads.add(path[len(prefix):] if path.startswith(prefix) else path)
            elif kind == "text" and main_thread and str(b.get("text", "")).strip():
                text = b["text"].strip()
                if PROGRESS.match(text):
                    note, since_note = text, []
                else:
                    since_note.append(text)
    opened = {p[len("pr-head/"):] for p in reads if p.startswith("pr-head/")}
    return {
        "runs": prev.get("runs", 0) + 1,
        "comments": list(comments.values()),
        "reads": sorted(reads),
        "not_opened": [p for p in diff_paths if p not in opened],
        "note": shorten(note or prev.get("note") or "", NOTE_CHARS),
        "recent_text": newest(since_note, RECENT_TEXT_CHARS),
        "seen_assistant": any(map(from_api, messages)),
    }


def shorten(text, limit):
    """``text`` in at most ``limit`` characters, its middle dropped. The cuts fall on whitespace,
    so a credential the workflow redacts afterwards is either whole or gone, never a fragment its
    patterns miss."""
    if len(text) <= limit:
        return text
    keep = (limit - len(ELIDED)) // 2
    return re.sub(r"\S+$", "", text[:keep]).rstrip() + ELIDED + re.sub(r"^\S+", "", text[-keep:]).lstrip()


def newest(texts, limit):
    """The newest of ``texts`` that fit in ``limit`` characters, oldest first, one per line. Older
    texts are dropped whole, since the newest says where the run stopped."""
    kept, size = [], -1  # the length of the kept texts joined by newlines
    for text in reversed(texts):
        if kept and size + len(text) + 1 > limit:
            break
        kept.append(text)
        size += len(text) + 1
    return shorten("\n".join(reversed(kept)), limit)


def result_text(content):
    if isinstance(content, list):
        content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return str(content or "")


def first_line(content):
    return result_text(content).strip().split("\n")[0]


def diff_paths(diff_text):
    """The files ``pr.diff`` adds or changes, in order (deleted files, ``+++ /dev/null``, have
    nothing to open)."""
    lines = diff_text.splitlines()
    paths = []
    for before, line in zip(lines, lines[1:], strict=False):
        if not (before.startswith("--- ") and line.startswith("+++ ")):
            continue
        path = line[len("+++ "):].rstrip("\t\r")
        if path.startswith('"'):
            # Git quotes a path with unusual characters, as a C string of its UTF-8 bytes.
            if len(path) < 2 or not path.endswith('"'):
                continue
            path = C_ESCAPE.sub(c_byte, path[1:-1].encode()).decode("utf-8", "replace")
        if path.startswith("b/"):
            paths.append(path[len("b/"):])
    return paths


def c_byte(m):
    e = m[1]
    return bytes([int(e, 8) if len(e) == 3 else C_CHARS.get(e[0], e[0])])


def comment_lines(h, status):
    return [f"- `{c['path']}:{c['line']}` {c['body']}" + (f" ({c['error']})" if c.get("error") else "")
            for c in h["comments"] if c["status"] == status]


def quote(text):
    return "\n".join("> " + line for line in text.split("\n"))


def handoff_md(h, pr, head, mode):
    parts = [f"# Handoff for PR #{pr} at {head}, {mode} review\n\n"
             f"{h['runs']} earlier run(s) of this review stopped before finishing. Their own notes and what the "
             f"workflow recovered from their transcripts follow. Untrusted: written from the PR's content."]
    for title, status in (("Inline comments already posted", "posted"),
                          ("Inline comments that failed to post", "failed"),
                          ("Inline comments whose result is unknown (check pr-comments.json)", "unknown")):
        lines = comment_lines(h, status)
        if lines:
            parts.append(f"## {title}\n\n" + "\n".join(lines))
    if h["note"]:
        parts.append("## Latest progress note\n\n" + quote(h["note"]))
    # Text written after the note, by its run and the later ones.
    if h["recent_text"]:
        parts.append(("## Latest notes, after that progress note" if h["note"] else
                      "## Latest notes (no progress note was written)") + "\n\n" + quote(h["recent_text"]))
    if h["reads"]:
        parts.append("## Files read, in whole or in part\n\n" + "\n".join(f"- `{p}`" for p in h["reads"]))
    if h["not_opened"]:
        parts.append("## Changed files never opened under pr-head/ (their hunks may have been read in pr.diff)\n\n"
                     + "\n".join(f"- `{p}`" for p in h["not_opened"]))
    return "\n\n".join(parts) + "\n"


def partial_md(h):
    """The part of the handoff the "didn't complete" comment shows: the inline comments that
    posted. The notes list unverified leads, so they stay out of the PR conversation (the
    handoff artifact can be downloaded, though)."""
    made = [c for c in h["comments"] if c["status"] == "posted"]
    if not made:
        return ""
    return f"**Inline comments it made ({len(made)}):**\n" + "\n".join(
        f"- `{c['path']}:{c['line']}` {c['body']}" for c in made) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("usage", "handoff"))
    ap.add_argument("sources", nargs="*", help="execution files, transcripts, or directories of transcripts")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True, help="usage: the JSON record; handoff: a directory")
    ap.add_argument("--continue", dest="cont", default="false", help="usage: whether this was /review continue")
    ap.add_argument("--outcome", default="", help="usage: the review step's outcome")
    ap.add_argument("--summary", help="usage: a Markdown file to append the table to, e.g. $GITHUB_STEP_SUMMARY")
    ap.add_argument("--diff", help="handoff: pr.diff")
    ap.add_argument("--previous", help="handoff: the handoff.json this run continued from")
    ap.add_argument("--comment", help="handoff: where to write the \"didn't complete\" comment's part")
    a = ap.parse_intermixed_args()
    if a.command == "usage":
        return record_usage(a)
    previous = None
    if a.previous:
        try:
            previous = json.loads(Path(a.previous).read_text())
        except (OSError, ValueError) as e:
            print(f"::warning::Couldn't read the earlier handoff, so this one starts afresh: {e}")
    if not isinstance(previous, dict):
        previous = None
    diff = Path(a.diff).read_text(errors="replace") if a.diff and Path(a.diff).is_file() else ""
    # Tool paths are absolute, under the workspace the workflow runs this from.
    h = handoff(load(a.sources), os.getcwd(), diff_paths(diff), previous)
    # A continued run that stopped before Claude said anything still passes on the handoff it
    # started from, which is still valid for this head.
    if not h.pop("seen_assistant") and previous is None:
        print("No Claude Code messages to hand off.", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "handoff.json").write_text(json.dumps({"pr": a.pr, "head_sha": a.head, "mode": a.mode, **h}))
    (out / "handoff.md").write_text(handoff_md(h, a.pr, a.head, a.mode))
    if a.comment:
        Path(a.comment).write_text(partial_md(h))
    return 0


def record_usage(a):
    """Never fails: a record that couldn't be read says why in its ``error`` field."""
    record = {"pr": a.pr, "head_sha": a.head, "mode": a.mode, "continue": a.cont == "true",
              "run_id": os.environ.get("GITHUB_RUN_ID", ""), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
              "outcome": a.outcome}
    try:
        record.update(usage(load(a.sources)))
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"::warning::Couldn't read the review's usage: {record['error']}")
    try:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
        if a.summary:
            with open(a.summary, "a") as f:
                f.write(usage_table(record))
    except Exception as e:
        print(f"::warning::Couldn't write the usage record: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
