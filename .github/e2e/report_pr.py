"""Post a bounded e2e matrix summary from an untrusted GPU job.

This runs from main in a separate job. It reads the artifact as data and never
executes or checks out code from the PR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

MARKER = "<!-- mstar-e2e-matrix -->"
MAX_SUMMARY_BYTES = 2_000_000
MAX_ROWS = 200


def _cell(value: object) -> str:
    """Limit artifact text to characters that cannot alter Markdown structure."""
    return re.sub(r"[^a-zA-Z0-9 _./+%-]", "?", str(value).replace("\n", " "))[:80]


def _delta(value: object) -> str:
    if not isinstance(value, dict):
        return "n/a"
    delta = value.get("delta_pct")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta):
        return "n/a"
    return f"{delta:+.1f}%"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.stat().st_size > MAX_SUMMARY_BYTES:
        raise ValueError("matrix summary is too large")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) > MAX_ROWS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("matrix summary must be a bounded list of rows")
    return rows


def conclusion(exit_code: str, rows: list[dict[str, Any]]) -> str:
    """No report or a contradictory exit code cannot become a green check."""
    if not rows:
        return "neutral"
    flagged = any(row.get("flagged") is True for row in rows)
    unfinished = any(row.get("correct") in ("incomplete", "base broken") for row in rows)
    if exit_code == "1":
        return "failure" if flagged else "neutral"
    if exit_code == "2":
        return "neutral"
    if exit_code == "0" and not flagged and not unfinished:
        return "success"
    return "neutral"


def render(rows: list[dict[str, Any]], verdict: str, head: str, run_url: str) -> str:
    lines = [
        MARKER,
        f"E2E matrix for `{head[:12]}`: **{verdict}**. [Run, chart and raw report]({run_url}).",
        "",
    ]
    if rows:
        lines += ["| Model | Case | Correctness | Latency | Throughput | Result |",
                  "| --- | --- | --- | ---: | ---: | --- |"]
        for row in rows:
            status = "flagged" if row.get("flagged") is True else (
                "no verdict" if row.get("correct") in ("incomplete", "base broken") else "ok"
            )
            lines.append("| " + " | ".join((
                _cell(row.get("model", "?")), _cell(row.get("case", "?")),
                _cell(row.get("correct", "?")), _delta(row.get("latency")),
                _delta(row.get("throughput")), status,
            )) + " |")
    else:
        lines.append("No matrix report was produced. Check the run log before retrying.")
    lines += ["", "The GPU job runs the PR on an ephemeral runner; the report job reads its artifact as data."]
    return "\n".join(lines)


def api(method: str, path: str, token: str, body: dict | None = None) -> tuple[Any, dict]:
    base = os.getenv("GITHUB_API_URL", "https://api.github.com")
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        return (json.loads(payload) if payload else None), dict(response.headers)


def post(pr: int, head: str, run_url: str, verdict: str, rows: list[dict[str, Any]]) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]
    message = render(rows, verdict, head, run_url)
    previous = None
    for page in range(1, 11):
        comments, _ = api("GET", f"/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}", token)
        previous = next((c for c in comments
                         if MARKER in c.get("body", "") and
                         c.get("user", {}).get("login") == "github-actions[bot]"), None)
        if previous or len(comments) < 100:
            break
    if previous:
        api("PATCH", f"/repos/{repo}/issues/comments/{previous['id']}", token, {"body": message})
    else:
        api("POST", f"/repos/{repo}/issues/{pr}/comments", token, {"body": message})
    api("POST", f"/repos/{repo}/check-runs", token, {
        "name": "e2e-matrix", "head_sha": head, "status": "completed",
        "conclusion": verdict, "details_url": run_url,
        "output": {"title": f"E2E matrix: {verdict}", "summary": f"{len(rows)} rows. {run_url}"},
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--exit-code", default="")
    args = parser.parse_args()
    if args.pr < 1 or not re.fullmatch(r"[0-9a-f]{40}", args.head):
        parser.error("expected a positive PR number and a 40-character commit SHA")
    try:
        rows = load_rows(args.summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"unreadable matrix artifact: {exc}", file=sys.stderr)
        rows = []
    verdict = conclusion(args.exit_code, rows)
    post(args.pr, args.head, args.run_url, verdict, rows)
    return 1 if verdict == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
