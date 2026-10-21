#!/usr/bin/env python3
"""Render the review's summary comment from its structured output.

The review returns the fields in SCHEMA rather than Markdown, so every summary
has the same shape: a status line, then only the sections that have something
in them. Run by claude-review.yml with the default branch checked out:

    python3 .github/review/render_summary.py --schema     # for --json-schema
    python3 .github/review/render_summary.py --head <sha> --mode light < output.json
"""
import argparse
import json
import sys

PRIORITIES = ("P0", "P1", "P2", "P3")
TYPES = ("new-model", "bug", "perf", "feature", "refactor", "test", "docs", "ci")
STRINGS = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "claim": {"type": "string"},
                "location": {"type": "string", "description": "path:line"},
                "inline": {"type": "boolean"},
                "detail": {"type": "string", "description": "Empty when inline is true."},
            },
            "required": ["priority", "claim", "location", "inline", "detail"],
        }},
        "since_last": {"type": "string"},
        "description_problems": STRINGS,
        "questions": STRINGS,
        "checked": STRINGS,
        "skipped": STRINGS,
        "types": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True,
                  "items": {"type": "string", "enum": list(TYPES)}},
    },
    "required": ["findings", "since_last", "description_problems", "questions", "checked", "skipped", "types"],
}
FOOTER = "<sub>Advisory, not a merge gate. Rules: `AGENTS.md`. Reply to discuss.</sub>"


def status(review, head, mode):
    counts = {p: sum(f["priority"] == p for f in review["findings"]) for p in PRIORITIES}
    if counts["P0"] or counts["P1"]:
        icon = "🔴"
    elif any(review[k] for k in ("findings", "description_problems", "questions", "skipped")):
        icon = "🟡"
    else:
        icon = "✅"
    found = ", ".join(f"{n} {p}" for p, n in counts.items() if n) or "No findings"
    return f"{icon} **{found}** on `{head[:7]}`, {mode} review"


def render(review, head, mode):
    parts = ["### Automated review", status(review, head, mode)]
    if review["since_last"]:
        parts.append(review["since_last"])
    # Inline findings are counted in the status line, not repeated here.
    sections = [
        ("Outside the diff", [f"**[{f['priority']}]** `{f['location']}`: {f['claim']}\n\n  {f['detail']}"
                              for f in review["findings"] if not f["inline"]]),
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
    a = ap.parse_args()
    if a.schema:
        print(json.dumps(SCHEMA, separators=(",", ":")))
    else:
        sys.stdout.write(render(json.load(sys.stdin), a.head, a.mode))


if __name__ == "__main__":
    main()
