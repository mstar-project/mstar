"""Tests for the correctness runner plumbing, with synthetic runs and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmark.correctness import metrics as M
from benchmark.correctness import report, server, sets
from benchmark.correctness.families import UNSUPPORTED, family_for_config

REPO = Path(__file__).resolve().parents[2]


def test_apply_overrides_dotted_yaml_values():
    cfg = {"model": "orpheus", "resources": {"kv_cache": {"page_size": 16}}}
    server.apply_overrides(cfg, ["resources.attn.flashinfer_backend=fa2", "resources.kv_cache.page_size=32",
                                 "max_seq_len=4096", "flag=true"])
    assert cfg["resources"]["attn"] == {"flashinfer_backend": "fa2"}
    assert cfg["resources"]["kv_cache"]["page_size"] == 32 and cfg["max_seq_len"] == 4096 and cfg["flag"] is True
    with pytest.raises(ValueError):
        server.apply_overrides(cfg, ["no_equals_sign"])
    with pytest.raises(ValueError):
        server.apply_overrides(cfg, ["model.sub=1"])  # model is a string, not a mapping


def test_effective_config_and_gpu_count(tmp_path):
    src = REPO / "configs" / "orpheus.yaml"
    cfg = server.write_effective_config(src, ["resources.attn.flashinfer_backend=fa2"], tmp_path / "c.yaml")
    assert yaml.safe_load((tmp_path / "c.yaml").read_text()) == cfg
    assert server.config_gpu_count(cfg) == 2
    assert server.config_gpu_count(yaml.safe_load((REPO / "configs/orpheus_colocated.yaml").read_text())) == 1
    assert server.resolve_config(REPO, "kokoro") == REPO / "configs" / "kokoro.yaml"
    with pytest.raises(FileNotFoundError):
        server.resolve_config(REPO, "no_such_config")


def test_every_config_has_a_family_or_a_stated_reason():
    for path in sorted((REPO / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        key = cfg["model"]
        if key in UNSUPPORTED:
            assert UNSUPPORTED[key], key
            continue
        fam = family_for_config(path.stem, key, cfg, {})
        assert fam.set_name in sets.BUILDERS, (path.stem, fam.set_name)
        assert fam.reference_id(), path.stem


def test_set_manifest_round_trip_and_tamper_check(tmp_path, monkeypatch):
    root = tmp_path / "sets" / "toy_v1"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("hello")
    items = [sets.Item(id="0", prompt="p", files={"doc": "a.txt"}, params={"seed": 1})]
    sets._write_manifest(root, "toy_v1", {"kind": "test"}, items)
    s = sets.load_set("toy_v1", tmp_path)
    assert s.items[0].params == {"seed": 1} and s.path("a.txt").read_text() == "hello"
    (root / "a.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        sets.load_set("toy_v1", tmp_path)
    with pytest.raises(FileNotFoundError):
        sets.load_set("missing_v1", tmp_path)


def test_text_set_builds_deterministically(tmp_path):
    sets.build_text_v1(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert len(m["items"]) == 32 and len({it["id"] for it in m["items"]}) == 32
    longs = [it for it in m["items"] if "needle" in it["expected"]]
    assert len(longs) == 4 and all(sets.NEEDLE in it["prompt"] for it in longs)
    (tmp_path / "again").mkdir()
    sets.build_text_v1(tmp_path / "again")
    m2 = json.loads((tmp_path / "again" / "manifest.json").read_text())
    assert [it["prompt"] for it in m2["items"]] == [it["prompt"] for it in m["items"]]


def test_report_verdict_is_the_worst_check(tmp_path):
    sc = report.FamilyScore("toy")
    sc.check("a", 1, "PASS", "rule a")
    sc.check("b", 2.5, "WARN", "rule b", "detail")
    sc.rows = [{"id": "x", "v": 1.0}]
    assert report.write_reports(tmp_path, {"config": "toy", "run_id": "r"}, sc) == "WARN"
    sc.check("c", None, "FAIL", "rule c")
    assert report.write_reports(tmp_path, {"config": "toy", "run_id": "r"}, sc) == "FAIL"
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["verdict"] == "FAIL" and len(data["checks"]) == 3
    assert "| b | 2.5 | WARN |" in (tmp_path / "report.md").read_text()
    assert report.write_reports(tmp_path, {"config": "toy"}, None, error="boom") == "ERROR"
    assert report.band(0.5, lambda v: v < 1) == "PASS" and report.band(2, lambda v: v < 1, lambda v: v < 3) == "WARN"
    assert report.band(float("nan"), lambda v: True) == "ERROR"


def _toy_tts_set(tmp_path):
    root = tmp_path / "sets" / "tts_toy_v1"
    root.mkdir(parents=True)
    items = [sets.Item(id=f"{i:03d}", prompt=f"sentence number {i}") for i in range(3)]
    sets._write_manifest(root, "tts_toy_v1", {}, items)
    return sets.load_set("tts_toy_v1", tmp_path)


def test_tts_scoring_on_synthetic_runs(tmp_path, monkeypatch):
    from benchmark.correctness.families import tts

    sset = _toy_tts_set(tmp_path)
    sr = 24000
    tone = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    run, ref = tmp_path / "run" / "r1", tmp_path / "ref"
    (run / "outputs").mkdir(parents=True)
    (ref / "outputs").mkdir(parents=True)
    (ref / "meta.json").write_text("{}")
    for it in sset.items:
        M.write_wav(ref / "outputs" / f"{it.id}.wav", tone, sr)
    M.write_wav(run / "outputs" / "000.wav", tone, sr)
    M.write_wav(run / "outputs" / "001.wav", np.zeros(sr, np.float32), sr)  # silent, fails sanity
    (run / "responses.jsonl").write_text(json.dumps({"id": "002", "ok": False, "error": "HTTP 500"}) + "\n")
    heard = {"000": "sentence number 0", "001": "", "002": "sentence number 2"}

    def fake_judge(paths, device, model_id=None):
        # The reference gets every sentence right, the run is as in ``heard``.
        return [(f"sentence number {int(Path(p).stem)}" if Path(p).parent.parent == ref else heard[Path(p).stem])
                if Path(p).is_file() else "" for p in paths]

    monkeypatch.setattr(tts.asr_judge, "transcribe", fake_judge)
    fam = tts.TtsFamily("toy", {}, {})
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, [run], ref, work, "cpu")
    by = {c.name: c for c in sc.checks}
    assert by["requests ok"].verdict == "FAIL"
    assert by["clips failing sanity"].value == 1
    assert by["WER - reference WER (pt)"].verdict == "FAIL"  # 3 of 9 words missing vs a perfect reference
    assert sc.verdict == "FAIL"


def test_unsupported_reason_is_reported():
    with pytest.raises(NotImplementedError, match="test_text_session"):
        family_for_config("test_text_session", "test_text_session", {}, {})


def test_each_server_gets_private_socket_and_upload_dirs(tmp_path):
    sock, up = server.private_dirs("ws-x", "0123abcd-r1")
    assert sock != up and sock.parent == up.parent and len(str(sock)) < 60  # socket paths max out around 107 chars
    spec = server.ServerSpec(name="t", worktree=REPO, model="kokoro", config_path=tmp_path / "c.yaml", gpus="0",
                             port=1, log_path=tmp_path / "log", socket_dir=tmp_path / "sock",
                             upload_dir=tmp_path / "up")
    (tmp_path / "sock").mkdir()
    (tmp_path / "sock" / "stale").write_text("left by an earlier server")
    spec.fresh_dirs()
    assert list((tmp_path / "sock").iterdir()) == [] and (tmp_path / "up").is_dir()
    cmd = spec.command()
    assert cmd[cmd.index("--socket-path-prefix") + 1] == f"{tmp_path / 'sock'}/"
    assert cmd[cmd.index("--upload-dir") + 1] == str(tmp_path / "up")
    spec.remove_dirs()
    assert not (tmp_path / "sock").exists() and not (tmp_path / "up").exists()


def test_config_name_keeps_dots_in_model_sizes():
    from benchmark.correctness.run import _config_name

    assert _config_name("qwen3_5_0.8b") == "qwen3_5_0.8b"
    assert _config_name("configs/qwen3_5_0.8b.yaml") == "qwen3_5_0.8b"
    assert _config_name("/x/y/kokoro.yml") == "kokoro"


def test_repeated_jit_build_failures_abort_startup(tmp_path):
    log = tmp_path / "server.log"
    assert server._fatal_in_log(log) is None
    log.write_text("RuntimeError: Ninja build failed. Ninja output:\n" * 2)
    assert server._fatal_in_log(log) is None
    log.write_text("cached_ops/batch_prefill_x_hd64/a.o\nfoo.inc:3:17: error: unknown type name 'T'\n"
                   + "RuntimeError: Ninja build failed. Ninja output:\n" * 3)
    msg = server._fatal_in_log(log)
    assert "3x 'Ninja build failed'" in msg and "batch_prefill_x_hd64" in msg and "unknown type name" in msg


def test_image_question_set_for_vision_text_models(tmp_path):
    from benchmark.correctness.families.text import TextImageFamily

    cfg = yaml.safe_load((REPO / "configs/qwen3_5_0.8b.yaml").read_text())
    fam = family_for_config("qwen3_5_0.8b", "qwen3_5_0.8b", cfg, {"set": "image"})
    text = family_for_config("qwen3_5_0.8b", "qwen3_5_0.8b", cfg, {})
    assert isinstance(fam, TextImageFamily) and fam.hf_id == text.hf_id == "Qwen/Qwen3.5-0.8B"
    assert fam.set_name == "omni_v1" and fam.reference_id() != text.reference_id()
    assert fam.set_name in sets.BUILDERS
    items = [sets.Item(id="img_00", prompt="What is this?", files={"image": "image/00.jpg"},
                       params={"modality": "image"}),
             sets.Item(id="aud_00", prompt="Transcribe this.", files={"audio": "audio/00.wav"},
                       params={"modality": "audio"})]
    sset = sets.SampleSet("omni_v1", tmp_path, {}, fam.select_items(items))
    (req,) = fam.requests(sset)
    assert req.item_id == "img_00" and req.files == [tmp_path / "image/00.jpg"]
    assert req.input_modalities == "image,text" and req.output_modalities == "text"
    assert req.model_kwargs["temperature"] == 0.0 and req.model_kwargs["max_output_tokens"] == 128
    duplex = family_for_config("nemotron_duplex", "nemotron_duplex", {}, {})
    assert [it.id for it in duplex.select_items(items)] == ["aud_00"]


def test_duplex_questions_get_trailing_silence(tmp_path):
    root = tmp_path / "sets" / "omni_v1"
    (root / "audio").mkdir(parents=True)
    M.write_wav(root / "audio" / "00.wav", np.full(16000, 0.1, np.float32), 16000)
    sset = sets.SampleSet("omni_v1", root, {}, [
        sets.Item(id="aud_00", prompt="q", files={"audio": "audio/00.wav"}, params={"modality": "audio"})])
    fam = family_for_config("nemotron_duplex", "nemotron_duplex", {}, {"trailing_s": "2"})
    (req,) = fam.requests(sset)
    x, sr = M.read_wav(req.files[0])
    assert sr == 16000 and len(x) == 3 * 16000 and np.abs(x[16000:]).max() == 0
    assert req.files[0].parent == tmp_path / "derived" / "omni_v1-trailing2s"
