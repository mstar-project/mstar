"""CPU diagnostics for joins in saved streamed PCM16 output."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")
pytest.importorskip("aiohttp")

from benchmark.eval.datasets import Item  # noqa: E402
from benchmark.eval.e2e import checks, compare, hosts, review  # noqa: E402
from benchmark.eval.suites.e2e import E2ESuite, save_outputs  # noqa: E402


def _row(item_id: str, chunks: list[np.ndarray]) -> dict:
    pcm = b"".join(chunk.astype("<i2").tobytes() for chunk in chunks)
    offsets = np.cumsum([len(c) for c in chunks[:-1]]).tolist()
    return {"id": item_id, "text": "", "_blob": None, "_audio": pcm,
            "_audio_boundaries": offsets, "_audio_sample_rate": 24000}


def test_clean_and_discontinuous_joins_are_reported_without_changing_verdict(tmp_path: Path) -> None:
    base, head = tmp_path / "base", tmp_path / "head"
    left = np.arange(2400, dtype=np.int16)
    right = np.arange(2400, 4800, dtype=np.int16)
    save_outputs([_row("000", [left, right])], base, "gen_clone")
    save_outputs([_row("000", [left, right + 1000])], head, "gen_clone")

    facts = checks.saved(base, head)
    assert facts["n_base"] == facts["n_head"] == 1
    assert facts["only_base"] == facts["head_undecodable"] == []
    assert facts["audio_joins_base"] == [
        {"file": "000.pcm", "sample": 2400, "jump": 1, "neighbor_step": 1.0, "normalized_jump": 1.0}]
    assert facts["audio_joins_head"] == [
        {"file": "000.pcm", "sample": 2400, "jump": 1001, "neighbor_step": 1.0, "normalized_jump": 1001.0}]
    # These are diagnostic facts only. The existing length check still sees equal-length audio.
    from benchmark.eval.e2e.manifest import Case
    from benchmark.eval.e2e.report import judge_correct

    case = Case(id="speech", modality="audio", driver="client", kind="gen_clone",
                check={"kind": "audio_len", "pct": 25})
    assert judge_correct(case, [facts], {})[0] == "pass"


def test_old_saved_runs_and_invalid_metadata_do_not_become_missing_outputs(tmp_path: Path) -> None:
    base, head = tmp_path / "base", tmp_path / "head"
    save_outputs([_row("000", [np.arange(8), np.arange(8, 16)])], base, "gen_clone")
    save_outputs([_row("000", [np.arange(16)])], head, "gen_clone")
    facts = checks.saved(base, head)
    assert facts["only_base"] == [] and "audio_joins_head" not in facts
    assert facts["audio_joins_base"][0]["sample"] == 8
    (base / "000.pcm.chunks.json").write_text(json.dumps({"schema": "mstar-eval/audio-chunks/1",
                                                         "sample_rate": 24000, "boundaries": [16]}))
    facts = checks.saved(base, head)
    assert facts["audio_joins_invalid_base"] == ["000.pcm"] and facts["only_base"] == []
    assert "audio_joins_base" not in facts
    (base / "000.pcm.chunks.json").write_text("[" * 1100 + "0" + "]" * 1100)
    assert checks.saved(base, head)["audio_joins_invalid_base"] == ["000.pcm"]
    (base / "000.pcm.chunks.json").unlink()
    assert "audio_joins_base" not in checks.saved(base, head)


def test_chunk_metadata_never_reads_a_symlink_or_stale_sidecar(tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "saved"
    row = _row("000", [np.arange(8), np.arange(8, 16)])
    save_outputs([row], d, "gen_clone")
    assert (d / "000.pcm.chunks.json").is_file()
    save_outputs([_row("000", [np.arange(16)])], d, "gen_clone")
    assert not (d / "000.pcm.chunks.json").exists()
    external = tmp_path / "external.json"
    external.write_text("secret")
    (d / "000.pcm.chunks.json").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        checks.audio_joins(d / "000.pcm")
    with pytest.raises(ValueError, match="symlink"):
        save_outputs([row], d, "gen_clone")
    assert external.read_text() == "secret"
    other = tmp_path / "other"
    other.mkdir()
    (other / "000.pcm").symlink_to(d / "000.pcm")
    assert checks.saved(d, other)["only_base"] == ["000.pcm"]

    (d / "000.pcm.chunks.json").unlink()
    save_outputs([row], d, "gen_clone")

    def vanished(_path):
        raise OSError("file vanished")

    monkeypatch.setattr(checks, "audio_joins", vanished)
    assert checks.saved(d, d)["audio_joins_invalid_base"] == ["000.pcm"]
    assert checks.saved(d, d)["audio_joins_invalid_head"] == ["000.pcm"]


def test_chat_and_generate_streams_save_joins_for_local_and_remote_review(tmp_path: Path) -> None:
    first = np.arange(16, dtype=np.int16).tobytes()
    second = np.arange(16, 32, dtype=np.int16).tobytes()

    class Response:
        status = 200

        def __init__(self, wire: bytes):
            self.wire = wire
            self.content = self

        async def iter_any(self):
            yield self.wire

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class Session:
        def __init__(self, wire: bytes):
            self.wire = wire

        def post(self, *args, **kwargs):
            return Response(self.wire)

    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode()
    chat = b"".join(b"data: " + json.dumps(msg).encode() + b"\n\n" for msg in [
        {"choices": [{"delta": {"audio": {"data": b64(first)}}}]},
        {"choices": [{"delta": {"audio": {"data": b64(second)}}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    generate = b"".join(json.dumps({"modality": "audio", "data": b64(data)}).encode() + b"\n"
                        for data in (first, second))
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    for kind, wire in (("chat_text", chat), ("gen_clone", generate)):
        args = Namespace(kind=kind, media=None, extra=[], model_name="m", timeout=2.0,
                         sample_rate=24000, max_tokens=8, audio_out=True)
        suite = E2ESuite(args)
        item = Item("000", text="hello", ref_audio=str(ref))
        result = asyncio.run(suite.request(Session(wire), "http://example", 0, item, time.perf_counter()))
        assert result["ok"] and result["_audio_boundaries"] == [16]
        saved = tmp_path / kind / "audio" / "x-c1" / "r2"
        save_outputs([result], saved, kind)
        assert (saved / "000.pcm").read_bytes() == first + second
        assert json.loads((saved / "000.pcm.chunks.json").read_text())["boundaries"] == [16]
        assert checks.audio_joins(saved / "000.pcm")[0]["sample"] == 16

    # Local review sees the sidecar; remote results keep facts without transferring media by default.
    assert compare.fetch_exclude(Namespace(review=True, review_fetch=False),
                                 hosts.Slot(hosts.LocalTransport(), 0)) == ()
    assert compare.fetch_exclude(Namespace(review=True, review_fetch=False),
                                 hosts.Slot(hosts.SSHTransport("example.invalid"), 0)) == compare.FETCH_EXCLUDE
    assert compare.fetch_exclude(Namespace(review=True, review_fetch=True),
                                 hosts.Slot(hosts.SSHTransport("example.invalid"), 0)) == ()
    for arm in ("base", "head"):
        target = tmp_path / "results" / "model" / arm / "s0" / "speech" / "audio" / "x-c1" / "r2"
        target.mkdir(parents=True)
        save_outputs([_row("000", [np.arange(16), np.arange(16, 32)])], target, "gen_clone")
    row = Namespace(model="model", case="speech", correct="changed", correct_detail="changed",
                    quality_worse=[], rerun="")
    model = Namespace(name="model", cases=[Namespace(id="speech", conc=[1])])
    page = review.write(tmp_path, [row], [model]).read_text()
    assert "Chunk joins (jump / neighboring step): sample 16: 1x" in page
    assert "chunks.json" not in page
