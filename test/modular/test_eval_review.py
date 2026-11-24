"""The opt-in compare review page uses saved local media without trusting labels or links."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from benchmark.eval.e2e import compare, hosts, review  # noqa: E402


def row(correct: str = "changed") -> SimpleNamespace:
    return SimpleNamespace(model="model <one>", case="case & one", correct=correct,
                           correct_detail="base < head & review", quality_worse=[], rerun="",
                           no_baseline=False, flagged=False, unfinished=False)


def models(conc: int = 1) -> list[SimpleNamespace]:
    return [SimpleNamespace(name="model <one>", cases=[SimpleNamespace(id="case & one", conc=[conc])])]


def test_review_page_escapes_labels_and_paths_and_ignores_external_symlink(tmp_path: Path) -> None:
    for arm in ("base", "head"):
        saved = tmp_path / "results" / "model <one>" / arm / "s0" / "case & one" / "audio" / "x-c1" / "r2"
        saved.mkdir(parents=True)
        (saved / "voice #1.pcm").write_bytes(b"\0\0" * 16)
        (saved / "text & <x>.txt").write_text("<script>alert(1)</script>")
    external = tmp_path.parent / "secret-review.txt"
    external.write_text("private marker")
    link = (tmp_path / "results" / "model <one>" / "head" / "s0" / "case & one" / "audio" /
            "x-c1" / "r2" / "leak.txt")
    link.symlink_to(external)

    page = review.write(tmp_path, [row()], models())
    text = page.read_text()
    assert "model &lt;one&gt; / case &amp; one" in text
    assert "base &lt; head &amp; review" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<script>" not in text
    assert "voice%20%231.wav" in text
    assert "private marker" not in text
    assert "leak.txt" not in text
    assert (tmp_path / "report" / "review-media" / "results" / "model <one>" / "base" / "s0" /
            "case & one" / "audio" / "x-c1" / "r2" / "voice #1.wav").is_file()


def test_review_is_only_for_changed_rows_and_remote_fetch_needs_review(tmp_path: Path) -> None:
    page = review.write(tmp_path, [row("pass")], models())
    assert "No changed rows with local saved media" in page.read_text()
    with pytest.raises(SystemExit, match="--review-fetch requires --review"):
        compare.cmd_compare(SimpleNamespace(review_fetch=True, review=False))


def test_saved_media_is_copied_only_for_local_or_opted_in_remote_review() -> None:
    local = hosts.Slot(hosts.LocalTransport(), 0)
    remote = hosts.Slot(hosts.SSHTransport("example.invalid"), 0)
    args = SimpleNamespace(review=True, review_fetch=False)
    assert compare.fetch_exclude(args, local) == ()
    assert compare.fetch_exclude(args, remote) == compare.FETCH_EXCLUDE
    args.review_fetch = True
    assert compare.fetch_exclude(args, remote) == ()
    args.review = False
    assert compare.fetch_exclude(args, remote) == compare.FETCH_EXCLUDE


def test_review_text_excerpt_does_not_read_the_whole_file(tmp_path: Path, monkeypatch) -> None:
    saved = tmp_path / "results" / "model <one>" / "base" / "s0" / "case & one" / "audio" / "x-c1" / "r2"
    saved.mkdir(parents=True)
    large = saved / "large.txt"
    large.write_text("x" * 10000)
    original = Path.read_text

    def forbid_whole_file(self, *args, **kwargs):
        if self == large:
            raise AssertionError("review read the whole saved text")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_whole_file)
    text = review.write(tmp_path, [row()], models()).read_text()
    assert "<pre>" + "x" * 4000 + "</pre>" in text
    assert "x" * 4001 not in text


def test_review_refuses_preexisting_media_output_symlink(tmp_path: Path) -> None:
    saved = tmp_path / "results" / "model <one>" / "base" / "s0" / "case & one" / "audio" / "x-c1" / "r2"
    saved.mkdir(parents=True)
    (saved / "voice.pcm").write_bytes(b"\0\0" * 16)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    report = tmp_path / "report"
    report.mkdir()
    (report / "review-media").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="review output path contains a symlink"):
        review.write(tmp_path, [row()], models())
    assert not list(outside.rglob("*.wav"))


def test_render_uses_results_layout_min_concurrency_and_abba_media(tmp_path: Path, monkeypatch) -> None:
    """The page follows compare's results/<model> layout, including the judged ABBA rerun."""
    for name, sessions in (("model <one>", ("s0",)), ("model <one>__abba", ("s0", "s1"))):
        for session in sessions:
            for arm in ("base", "head"):
                saved = (tmp_path / "results" / name / arm / session / "case & one" /
                         "audio" / "x-c2" / "r2")
                saved.mkdir(parents=True)
                (saved / "voice.pcm").write_bytes(b"\0\0" * 16)
    (tmp_path / "run.json").write_text(json.dumps({"models": ["model <one>"], "gpu": "h200"}))
    md = tmp_path / "report" / "report.md"
    md.parent.mkdir()
    md.write_text("report\n")
    changed = row()
    changed.rerun = "judged on a 2-session ABBA rerun"
    monkeypatch.setattr(compare, "run_models", lambda *_: models(2))
    monkeypatch.setattr(compare.calibrate, "load", lambda *_: {})
    monkeypatch.setattr(compare.report, "write", lambda *_: (md, [changed]))

    result, code = compare.render(tmp_path, object(), "unused", with_review=True)
    text = (tmp_path / "report" / "review.html").read_text()
    assert (result, code) == (md, 0)
    assert "first run/s0/voice.pcm" in text
    assert "ABBA rerun/s0/voice.pcm" in text
    assert "ABBA rerun/s1/voice.pcm" in text
    assert text.count("<audio controls") == 6
    assert "[Changed output review](review.html)" in md.read_text()
