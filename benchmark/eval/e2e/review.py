"""Opt-in, local side-by-side media page for changed e2e compare rows."""

from __future__ import annotations

import html
import os
import wave
from pathlib import Path
from urllib.parse import quote

from benchmark.eval.e2e import checks


def _safe_file(path: Path, root: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())


def _url(path: Path, page: Path) -> str:
    return html.escape(quote(os.path.relpath(path, page.parent).replace(os.sep, "/"), safe="/"), quote=True)


def _media(path: Path, root: Path, page: Path) -> str:
    if not _safe_file(path, root):
        return ""
    label = html.escape(path.name)
    target = path
    if path.suffix == ".pcm":
        target = page.parent / "review-media" / path.relative_to(root).with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(checks.PCM_RATE)
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    out.writeframesraw(block)
    url = _url(target, page)
    if path.suffix in (".wav", ".pcm"):
        return f'<audio controls src="{url}"></audio>'
    if path.suffix == ".png":
        return f'<img src="{url}" alt="{label}">'
    if path.suffix == ".mp4":
        return f'<video controls src="{url}"></video>'
    if path.suffix == ".txt":
        return f"<pre>{html.escape(path.read_text(encoding='utf-8')[:4000])}</pre>"
    return f'<a href="{url}">download {label}</a>'


def write(run_dir: Path, rows: list) -> Path:
    """Write report/review.html using local saved outputs only; no media is fetched here."""
    page = run_dir / "report" / "review.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">",
             "<title>Compare media review</title>",
             "<style>body{font:16px system-ui;margin:2rem;max-width:80rem}"
             ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
             "img,video{max-width:100%}audio{max-width:100%}pre{white-space:pre-wrap}</style>",
             "<h1>Changed output review</h1><p>Listen and look before deciding whether a changed output is worse. "
             "This page does not score quality. Remote outputs require an explicit --review-fetch run.</p>"]
    shown = 0
    for row in rows:
        if row.correct not in ("changed", "FAIL") and not row.quality_worse:
            continue
        model = run_dir / row.model
        case = row.case
        pairs: list[tuple[str, Path | None, Path | None]] = []
        for session in sorted((model / "base").glob("s*")):
            if not session.is_dir() or not session.name.startswith("s"):
                continue
            base = checks.saved_dir(session / case, 1)
            head = checks.saved_dir(model / "head" / session.name / case, 1)
            names = set()
            for directory in (base, head):
                if directory is not None and directory.is_dir():
                    names.update(p.name for p in directory.iterdir() if _safe_file(p, run_dir))
            pairs.extend((f"{session.name}/{name}", base / name if base else None,
                          head / name if head else None) for name in sorted(names))
        if not pairs:
            continue
        shown += 1
        parts.append(f"<section><h2>{html.escape(row.model)} / {html.escape(case)}</h2>"
                     f"<p>{html.escape(row.correct_detail)}</p>")
        for label, base, head in pairs:
            parts.append(f"<h3>{html.escape(label)}</h3><div class=\"pair\">")
            for arm, path in (("base", base), ("head", head)):
                content = _media(path, run_dir, page) if path is not None else ""
                parts.append(f"<div><b>{arm}</b><br>{content or 'No saved output'}</div>")
            parts.append("</div>")
        parts.append("</section>")
    if not shown:
        parts.append("<p>No changed rows with local saved media were found.</p>")
    page.write_text("\n".join(parts + ["</html>"]), encoding="utf-8")
    return page
