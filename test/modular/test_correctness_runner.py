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



def test_duplex_json_keeps_whole_replies(tmp_path, monkeypatch):
    from benchmark.correctness.families import duplex

    root = tmp_path / "sets" / "omni_v1"
    (root / "audio").mkdir(parents=True)
    M.write_wav(root / "audio" / "00.wav", np.full(16000, 0.1, np.float32), 16000)
    sset = sets.SampleSet("omni_v1", root, {}, [
        sets.Item(id="aud_00", prompt="q", files={"audio": "audio/00.wav"}, params={"modality": "audio"})])
    out = tmp_path / "run" / "r1" / "outputs"
    out.mkdir(parents=True)
    text = " ".join(f"word{i}" for i in range(60))
    (out / "aud_00.txt").write_text(text)
    M.write_wav(out / "aud_00.wav", (0.3 * np.sin(np.arange(16000) / 4)).astype(np.float32), 16000)
    heard = " ".join(f"word{i}" for i in range(40))
    monkeypatch.setattr(duplex.asr_judge, "transcribe", lambda paths, device, model_id=None: [heard for _ in paths])
    fam = family_for_config("nemotron_duplex", "nemotron_duplex", {}, {})
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, [out.parent], None, work, "cpu")
    (row,) = json.loads((work / "duplex.json").read_text())
    assert row["text"] == text and row["heard"] == heard and len(text) > 160
    assert row["wer_vs_own_text"] == pytest.approx(1 / 3, abs=1e-3)  # 20 of 60 words never spoken
    assert sc.rows[0]["text"] == text[:160]  # the report table stays short



def test_vjepa2_gate_scales_with_hf_bf16_distance(tmp_path):
    from benchmark.correctness.report import FamilyScore

    rng = np.random.default_rng(0)
    refs = tmp_path / "references"
    fp32 = refs / "hf-vjepa2-vitl-fpc64-256-encoder-fp32" / "droid_v1"
    bf16 = refs / "hf-vjepa2-vitl-fpc64-256-encoder-bf16" / "droid_v1"
    (fp32 / "outputs").mkdir(parents=True)
    (bf16 / "outputs").mkdir(parents=True)
    ids = [f"ep{i:02d}" for i in range(4)]
    for i in ids:
        a = rng.standard_normal(256).astype(np.float32)
        np.save(fp32 / "outputs" / f"{i}.npy", a)
        np.save(bf16 / "outputs" / f"{i}.npy", a + 0.2 * rng.standard_normal(256).astype(np.float32))
    hf_bf16 = [M.cosine(np.load(bf16 / "outputs" / f"{i}.npy"), np.load(fp32 / "outputs" / f"{i}.npy")) for i in ids]
    fam = family_for_config("vjepa2", "vjepa2", {}, {})

    def verdict(served_cos, ref_dir=fp32):
        sc = FamilyScore("vjepa2")
        fam.reference_check(sc, [{"id": i, "ref_cosine": c} for i, c in zip(ids, served_cos, strict=True)], ref_dir)
        return sc.checks[-1]

    (bf16 / "meta.json").write_text("{}")
    assert verdict(hf_bf16).verdict == "PASS"  # as far from fp32 as HF's own bf16
    assert verdict([1 - 3 * (1 - c) for c in hf_bf16]).verdict == "WARN"
    assert verdict([1 - 6 * (1 - c) for c in hf_bf16]).verdict == "FAIL"
    assert "raw min cosine" in verdict(hf_bf16).rule
    (bf16 / "meta.json").unlink()
    fixed = verdict(hf_bf16)  # without the bf16 copy the fixed bar applies
    assert fixed.name == "min embedding cosine vs reference" and fixed.verdict == "FAIL"

def test_make_reference_builds_in_a_child_process(monkeypatch):
    import subprocess
    import sys

    from benchmark.correctness import run as runner

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, len(calls) - 1)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    argv = ["--config", "flux2_klein", "--make-reference", "--device", "cuda:3"]
    assert runner._build_reference_in_child(argv) is None
    assert "exited with 1" in runner._build_reference_in_child(argv)
    cmd = calls[0]
    assert cmd[0] == sys.executable and cmd[1].endswith("benchmark/correctness/run.py")
    assert cmd[2:] == [*argv, "--reference-only"]

def test_cosmos3_modes_and_image_size_override(tmp_path):
    from benchmark.correctness.families import Cosmos3I2V, Cosmos3T2I, Cosmos3Video

    cfg = yaml.safe_load((REPO / "configs/cosmos3_nano.yaml").read_text())
    t2v = family_for_config("cosmos3_nano", "cosmos3", cfg, {})
    t2i = family_for_config("cosmos3_nano", "cosmos3", cfg, {"mode": "t2i", "size": "480x832"})
    i2v = family_for_config("cosmos3_nano", "cosmos3", cfg, {"mode": "i2v"})
    assert type(t2v) is Cosmos3Video and isinstance(t2i, Cosmos3T2I) and isinstance(i2v, Cosmos3I2V)
    assert len({t2v.reference_id(), i2v.reference_id(), t2i.reference_id()}) == 3
    assert t2i.ref_call_kwargs["use_system_prompt"] is False and t2i.generator_device == "cuda"
    items = [sets.Item(id=f"{i:02d}_{h}x{w}", prompt=f"p{i}", params={"seed": i, "height": h, "width": w})
             for h, w in ((1024, 1024), (768, 1344)) for i in range(2)]
    picked = t2i.select_items(items)
    assert [it.id for it in picked] == ["00_1024x1024", "01_1024x1024"]
    assert all((it.params["height"], it.params["width"]) == (480, 832) for it in picked)
    (req, _) = t2i.requests(sets.SampleSet("image_v1", tmp_path, {}, picked))
    kw = req.model_kwargs
    assert (kw["height"], kw["width"], kw["num_inference_steps"]) == (480, 832, 35)
    assert kw["flow_shift"] == 1.0 and kw["guidance_interval"] == [0.0, 1000.0]
    assert t2i.pipeline_kwargs(picked[0])["num_frames"] == 1
    vid = sets.Item(id="00", prompt="a cat", files={"image": "image/00.png"}, params={"seed": 0})
    (r,) = i2v.requests(sets.SampleSet("video_v1", tmp_path, {}, [vid]))
    assert r.files == [tmp_path / "image/00.png"] and r.input_modalities == "image,text"


def test_math_sdpa_reference_spread_is_reported(tmp_path):
    from PIL import Image

    from benchmark.correctness.families import ZImageTurboImage

    fam = ZImageTurboImage("z_image_turbo", {}, {})
    math = ZImageTurboImage("z_image_turbo", {}, {"ref_sdpa": "math"})
    assert math.reference_id() == fam.reference_id() + "-sdpa-math"
    items = [sets.Item(id=f"{i:02d}", prompt="p", params={"seed": i, "height": 16, "width": 16}) for i in range(2)]
    sset = sets.SampleSet("image_v1", tmp_path, {}, items)
    rng = np.random.default_rng(0)
    ref = tmp_path / "references" / fam.reference_id() / "image_v1"
    alt = tmp_path / "references" / math.reference_id() / "image_v1"
    run = tmp_path / "run" / "r1"
    for d in (ref / "outputs", alt / "outputs", run / "outputs"):
        d.mkdir(parents=True)
    for d in (ref, alt):
        (d / "meta.json").write_text("{}")
    for it in items:
        img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        for d in (ref / "outputs", run / "outputs"):
            Image.fromarray(img).save(d / f"{it.id}.png")
        Image.fromarray(np.clip(img.astype(int) + 4, 0, 255).astype(np.uint8)).save(alt / "outputs" / f"{it.id}.png")
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, [run], ref, work, "cpu")
    spread = sc.metrics["reference_vs_math_sdpa_reference_psnr_db"]
    assert spread["n"] == 2 and 30 < spread["median"] < 40
    assert math.math_reference_dir(ref) is None and fam.math_reference_dir(ref) == alt
    cal = sc.metrics["rocm_calibrated"]  # served == reference here, so it clears the floor
    assert cal["verdict"] == "PASS" and cal["floor_db"] == round(spread["median"] - 3.0, 2)
    assert sc.verdict == "PASS"  # the raw gate is unchanged


def test_fp32_anchor_arm_encodes_the_conditioning_frame_in_fp32():
    import torch

    from benchmark.correctness.families import Cosmos3I2V

    fam = Cosmos3I2V("cosmos3_nano", {}, {"ref_anchor": "fp32"})
    assert fam.reference_id() == Cosmos3I2V("cosmos3_nano", {}, {}).reference_id() + "-anchor-fp32"

    class Vae(torch.nn.Linear):
        @property
        def dtype(self):
            return self.weight.dtype

    class Pipe:  # like Cosmos3OmniPipeline._encode_video: encode in the VAE's dtype, return the input's
        def __init__(self):
            self.vae = Vae(64, 64).to(torch.bfloat16)

        def _encode_video(self, x):
            return self.vae(x.to(self.vae.dtype)).to(x.dtype)

    torch.manual_seed(0)
    pipe, moves = Pipe(), []
    x = torch.randn(1, 1, 3, 64)  # [B, C, T, ...]; the anchor is T=0
    with torch.no_grad():
        bf16 = pipe._encode_video(x)
        exact = x @ pipe.vae.weight.float().T + pipe.vae.bias.float()
        fam.fp32_anchor(pipe, moves)
        out = pipe._encode_video(x)
    assert pipe.vae.weight.dtype == torch.bfloat16  # back to bf16 for the decode
    assert (out - exact).abs().max() < (bf16 - exact).abs().max()
    assert len(moves) == 1 and 0 < moves[0] < 0.05


def test_i2v_floor_takes_the_larger_move_per_video(tmp_path):
    from benchmark.correctness.families import Cosmos3I2V
    from benchmark.correctness.families.video import write_video

    fam = Cosmos3I2V("cosmos3_nano", {}, {})
    items = [sets.Item(id=f"{i:02d}", prompt="p", files={"image": "x.png"}, params={"seed": i}) for i in range(3)]
    sset = sets.SampleSet("video_v1", tmp_path, {}, items)
    refs = tmp_path / "references"
    ref = refs / fam.reference_id() / "video_v1"
    math = refs / (fam.reference_id() + "-sdpa-math") / "video_v1"
    anchor = refs / (fam.reference_id() + "-anchor-fp32") / "video_v1"
    run = tmp_path / "run" / "r1"
    for d in (ref / "outputs", math / "outputs", anchor / "outputs", run / "outputs"):
        d.mkdir(parents=True)
    for d in (ref, math):
        (d / "meta.json").write_text("{}")
    (anchor / "meta.json").write_text(json.dumps({"anchor_rel_change_vs_bf16": [0.008, 0.009, 0.01]}))
    rng = np.random.default_rng(0)

    def shifted(frames, by):
        return np.clip(frames.astype(int) + by, 0, 255).astype(np.uint8)

    for k, it in enumerate(items):
        frames = rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
        np.save(ref / "outputs" / f"{it.id}.npy", frames)
        np.save(math / "outputs" / f"{it.id}.npy", shifted(frames, 2))  # a small move on every video
        np.save(anchor / "outputs" / f"{it.id}.npy", shifted(frames, 40 if k == 0 else 1))  # a big one on 00
        write_video(frames, run / "outputs" / f"{it.id}.mp4", 8)
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, [run], ref, work, "cpu")
    by = {r["id"]: r for r in sc.rows}
    assert by["00"]["ref_vs_anchor_ref_psnr_median"] < by["00"]["ref_vs_math_ref_psnr_median"]
    assert by["01"]["ref_vs_anchor_ref_psnr_median"] > by["01"]["ref_vs_math_ref_psnr_median"]
    larger = sorted(min(r["ref_vs_math_ref_psnr_median"], r["ref_vs_anchor_ref_psnr_median"]) for r in sc.rows)
    cal = sc.metrics["rocm_calibrated"]
    assert cal["basis"].startswith("math-SDPA and fp32-anchor") and cal["spread_median_db"] == larger[1]
    assert sc.metrics["reference_vs_math_sdpa_reference_psnr_db"]["n"] == 3
    assert sc.metrics["fp32_anchor_rel_change_vs_bf16"]["max"] == 0.01
    assert fam.anchor_reference_dir(ref) == anchor
    assert Cosmos3I2V("cosmos3_nano", {}, {"ref_anchor": "fp32"}).anchor_reference_dir(ref) is None


@pytest.mark.parametrize("config", ["cosmos3_droid", "cosmos3_edge_droid"])
def test_cosmos3_droid_sends_the_robolab_request(tmp_path, config):
    """The 8 action dims, the composite, the joint state with 1 - gripper as a state row (33 rows and frames,
    15 fps), and the prompt the RoboLab server builds: plain for Nano, JSON plus the cookbook interval for Edge."""
    fam = family_for_config(config, config, {}, {})
    states = np.zeros((10, 8), dtype=np.float32)
    states[4] = [0.1, -0.2, 0.3, -2.0, 0.05, 1.9, 0.8, 0.25]
    np.save(tmp_path / "s.npy", states)
    sset = sets.SampleSet("droid_v2", tmp_path, {}, [sets.Item(
        id="ep00", prompt="Put the marker in the pot", params={"seed": 0, "step": 4},
        files={"composite": "c.png", "joint_states": "s.npy"})])
    (req,) = fam.requests(sset)
    kw = req.model_kwargs
    assert kw["raw_action_dim"] == 8 and kw["action_chunk_size"] == kw["num_frames"] == 33
    assert kw["fps"] == 15.0 and kw["size"] == "736x544" and req.files == [tmp_path / "c.png"]
    assert np.allclose(kw["action"], [[0.1, -0.2, 0.3, -2.0, 0.05, 1.9, 0.8, 0.75]])
    edge = config == "cosmos3_edge_droid"
    assert ("guidance_interval" in kw) == edge
    assert req.text.startswith("{") == edge
    assert "This video is of 544x736 resolution." in req.text or edge
    raw = np.arange(32 * 16, dtype=np.float32)  # the server drops the state row and pads the action width
    assert fam.to_array(raw).shape == (32, 8)


def test_robotics_can_skip_episodes_without_a_task():
    items = [sets.Item(id="ep00", prompt="Put the marker in the pot"), sets.Item(id="ep05", prompt="")]
    fam = family_for_config("cosmos3_droid", "cosmos3_droid", {}, {"skip_empty_prompt": "1"})
    assert [it.id for it in fam.select_items(items)] == ["ep00"]
    assert len(family_for_config("pi05", "pi05", {}, {}).select_items(items)) == 2


def test_vjepa2_ac_fails_a_rollout_that_repeats_a_step(tmp_path):
    """Two instances agreeing says nothing about a response that repeats one step H times."""
    fam = family_for_config("vjepa2_ac", "vjepa2_ac", {}, {})
    sset = sets.SampleSet("droid_v1", tmp_path, {}, [sets.Item(id=i) for i in ("ep00", "ep01")])
    rng = np.random.default_rng(0)
    steps = rng.standard_normal((4, 2, 1408)).astype(np.float32)
    outputs = {"ep00": steps, "ep01": np.repeat(steps[3:], 4, axis=0)}
    runs = []
    for name in ("r1", "r2"):
        (tmp_path / name / "outputs").mkdir(parents=True)
        for item_id, out in outputs.items():
            np.save(tmp_path / name / "outputs" / f"{item_id}.npy", out)
        runs.append(tmp_path / name)
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, runs, None, work, "cpu")
    check = next(c for c in sc.checks if c.name.startswith("outputs with a rollout step"))
    assert (check.value, check.verdict, check.detail) == (1, "FAIL", "ep01")
    assert sc.rows[0]["step_maxabs_min"] > 0 and sc.rows[1]["step_maxabs_min"] == 0.0


def _lossless_mp4(frames: np.ndarray, path: Path, preset: str) -> None:
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=8)
        s.height, s.width = frames.shape[1:3]
        s.pix_fmt = "yuv420p"
        s.options = {"qp": "0", "preset": preset}
        for f in frames:
            for packet in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)


def test_video_identity_compares_decoded_frames(tmp_path):
    from benchmark.correctness.families.video import NoReferenceVideo

    fam = NoReferenceVideo("cosmos3_super", {}, {})
    sset = sets.SampleSet("video_v1", tmp_path, {}, [sets.Item(id="00", prompt="p", params={"seed": 0})])
    frames = np.random.default_rng(0).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    changed = frames.copy()
    changed[2] = np.clip(changed[2].astype(int) + 40, 0, 255)
    runs = {}
    # Lossless, so the decoded frames don't depend on the encoder settings, while two presets give
    # different files for the same frames.
    for name, f, preset in (("r1", frames, "ultrafast"), ("r2", frames, "medium"), ("r3", changed, "ultrafast")):
        runs[name] = tmp_path / name
        (runs[name] / "outputs").mkdir(parents=True)
        _lossless_mp4(f, runs[name] / "outputs" / "00.mp4", preset)
    assert (runs["r1"] / "outputs/00.mp4").read_bytes() != (runs["r2"] / "outputs/00.mp4").read_bytes()
    scores = {}
    for other in ("r2", "r3"):
        work = tmp_path / f"work_{other}"
        work.mkdir()
        scores[other] = fam.score(sset, [runs["r1"], runs[other]], None, work, "cpu")
    same, diff = scores["r2"], scores["r3"]
    assert same.rows[0]["frames_identical_r1_r2"] is True
    assert same.metrics["frames_identical_r1_r2"] == {"identical": 1, "n": 1}
    assert any("1/1 videos" in n for n in same.notes)
    assert diff.rows[0]["frames_identical_r1_r2"] is False
    assert diff.metrics["frames_identical_r1_r2"] == {"identical": 0, "n": 1}
    assert same.rows[0]["psnr_r1_r2_min"] == float("inf") > diff.rows[0]["psnr_r1_r2_min"]


def _video_run(tmp_path, fam, ref_frames, served: dict[str, bytes]):
    """One reference video (``00``) and a run per entry of ``served`` (mp4 bytes)."""
    ref_dir = tmp_path / "references" / "ref" / "video_v1"
    (ref_dir / "outputs").mkdir(parents=True)
    np.save(ref_dir / "outputs" / "00.npy", ref_frames)
    (ref_dir / "meta.json").write_text("{}")
    runs = []
    for name, data in served.items():
        (tmp_path / name / "outputs").mkdir(parents=True)
        (tmp_path / name / "outputs" / "00.mp4").write_bytes(data)
        runs.append(tmp_path / name)
    sset = sets.SampleSet("video_v1", tmp_path, {}, [sets.Item(id="00", prompt="p", params={"seed": 0})])
    work = tmp_path / "work"
    work.mkdir()
    return fam.score(sset, runs, ref_dir, work, "cpu")


def test_video_reference_goes_through_the_served_encode(tmp_path):
    """A served video that is the reference's own frames scores inf once the reference takes the same
    encode, where the raw frames would cap it at the encoder's own PSNR."""
    pytest.importorskip("av")
    from benchmark.correctness.families import video

    fam = family_for_config("cosmos3_nano", "cosmos3", {}, {})
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (5, 64, 96, 3), dtype=np.uint8)
    served = video.server_mp4(frames, video.SERVER_X264["cosmos3"], fam.fps)
    sc = _video_run(tmp_path, fam, frames, {"r1": served})
    assert sc.rows[0]["psnr_median"] == float("inf")
    raw = video.read_video(tmp_path / "r1" / "outputs" / "00.mp4")
    assert np.isfinite(min(M.frame_psnrs(frames, raw)))
    assert any("encode" in n for n in sc.notes)
    assert video.SERVER_X264["wan22"] == {}  # Wan2.2 encodes with libx264's defaults


def test_cosmos3_nano_ar_runs_the_windowed_walk(tmp_path):
    from benchmark.correctness.families import Cosmos3Video, Cosmos3WindowedVideo, video

    cfg = yaml.safe_load((REPO / "configs/cosmos3_nano_ar.yaml").read_text())
    sset = sets.SampleSet("video_v1", tmp_path, {}, [sets.Item(id="00", prompt="p", params={"seed": 0})])
    ar = family_for_config("cosmos3_nano_ar", "cosmos3", cfg, {})
    assert isinstance(ar, Cosmos3WindowedVideo) and ar.self_consistency_only
    assert {"window_mode": "kv", "window_frames": 13}.items() <= ar.requests(sset)[0].model_kwargs.items()
    chained = family_for_config("cosmos3_nano_ar", "cosmos3", cfg, {"window_mode": "chained"})
    assert chained.requests(sset)[0].model_kwargs["window_mode"] == "chained"
    plain = family_for_config("cosmos3_nano", "cosmos3", cfg, {})
    assert type(plain) is Cosmos3Video and "window_mode" not in plain.requests(sset)[0].model_kwargs

    pytest.importorskip("av")
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (5, 64, 96, 3), dtype=np.uint8)
    other = rng.integers(0, 256, (5, 64, 96, 3), dtype=np.uint8)
    served = video.server_mp4(other, video.SERVER_X264["cosmos3"], ar.fps)
    sc = _video_run(tmp_path, ar, frames, {"r1": served, "r2": served})
    names = [c.name for c in sc.checks]
    assert "median per-frame PSNR vs reference (dB)" not in names
    assert any(n.startswith("same-seed min frame PSNR") for n in names)
    assert any("for information only" in n for n in sc.notes)


def test_ltx2_5_audio_video_family(tmp_path):
    import io
    import struct

    av = pytest.importorskip("av")
    from benchmark.correctness.client import Response
    from benchmark.correctness.families.av import AudioVideoFamily

    cfg = yaml.safe_load((REPO / "configs/ltx2_5.yaml").read_text())
    geo = {"height": "32", "width": "32", "num_frames": "9", "fps": "9"}  # a 1 s clip
    fam = family_for_config("ltx2_5", "ltx2_5", cfg, geo)
    assert isinstance(fam, AudioVideoFamily) and not fam.self_consistency_only
    assert fam.reference_id() == "diffusers-Lightricks--LTX-2.5-Diffusers-32x32x9-9fps-distilled"
    assert family_for_config("ltx2_5", "ltx2_5", cfg, {}).reference_id() == \
        "diffusers-Lightricks--LTX-2.5-Diffusers-544x960x121-distilled"
    item = sets.Item(id="00", prompt="a cat", params={"seed": 0})
    sset = sets.SampleSet("video_v1", tmp_path, {}, [item])
    (req,) = fam.requests(sset)
    assert req.output_modalities == "video,audio"
    assert req.model_kwargs == {"seed": 0, "height": 32, "width": 32, "num_frames": 9, "fps": 9.0}

    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as c:
        s = c.add_stream("mpeg4", rate=9)
        s.height = s.width = 32
        s.pix_fmt = "yuv420p"
        for _ in range(9):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), format="rgb24")
            for packet in s.encode(frame):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000) * 32767).astype("<i2")
    pcm = np.stack([tone, tone], axis=1).tobytes()  # interleaved stereo
    runs = []
    for k, audio in ((1, pcm), (2, pcm[:-4] + struct.pack("<hh", 0, 0))):
        resp = Response("00", ok=True, video=bytearray(buf.getvalue()), audio=bytearray(audio),
                        metadata=[{"modality": "audio", "sample_rate": 8000, "num_channels": 2}])
        out = tmp_path / "run" / f"r{k}"
        (out / "outputs").mkdir(parents=True)
        fam.save(sset, [resp], out / "outputs")
        runs.append(out)
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, runs, None, work, "cpu")
    by = {c.name: c for c in sc.checks}
    assert by["requests ok"].verdict == "PASS"
    assert by["PSNR vs reference"].verdict == "NO_REFERENCE"
    assert by["videos with the requested frames and size, none flat"].verdict == "PASS"
    assert by["audio tracks failing sanity"].value == 0, sc.rows
    assert by["same-seed min frame PSNR, two server instances (dB)"].verdict == "PASS"  # same mp4 bytes
    assert by["same-seed min audio SNR, two server instances (dB)"].verdict == "PASS"  # one sample differs
    assert sc.metrics["frames_identical_r1_r2"] == {"identical": 1, "n": 1}
    assert sc.metrics["byte_identical_r1_r2"] == {"pcm": 0, "n": 1}


def _tone_pcm(seconds: float, sr: int, freq: float, channels: int = 2) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return np.stack([tone] * channels, axis=1).tobytes()


def _ltx_reference(root: Path, rid: str, frames: np.ndarray, pcm: bytes, sr: int) -> Path:
    from benchmark.correctness.families.av import write_pcm_wav

    d = root / "references" / rid / "video_v1"
    (d / "outputs").mkdir(parents=True)
    np.save(d / "outputs" / "00.npy", frames)
    write_pcm_wav(d / "outputs" / "00.wav", pcm, sr, 2)
    (d / "meta.json").write_text("{}")
    return d


def test_ltx2_5_scores_against_its_reference(tmp_path):
    """A served clip that is the reference's own frames and audio passes the raw and calibrated video gates
    and the audio check, where the math-SDPA copy sets the spread."""
    pytest.importorskip("av")
    from benchmark.correctness.families import video
    from benchmark.correctness.families.av import write_pcm_wav

    geo = {"height": "32", "width": "32", "num_frames": "9", "fps": "9"}
    fam = family_for_config("ltx2_5", "ltx2_5", {}, geo)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (9, 32, 32, 3), dtype=np.uint8)
    moved = np.clip(frames.astype(int) + rng.integers(-20, 21, frames.shape), 0, 255).astype(np.uint8)
    sr = 8000
    pcm, other = _tone_pcm(1.0, sr, 220), _tone_pcm(1.0, sr, 330)
    ref = _ltx_reference(tmp_path, fam.reference_id(), frames, pcm, sr)
    _ltx_reference(tmp_path, fam.reference_id() + "-sdpa-math", moved, other, sr)
    run = tmp_path / "run" / "r1"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "00.mp4").write_bytes(video.server_mp4(frames, video.SERVER_X264["ltx2_5"], 9))
    write_pcm_wav(run / "outputs" / "00.wav", pcm, sr, 2)
    sset = sets.SampleSet("video_v1", tmp_path, {}, [sets.Item(id="00", prompt="p", params={"seed": 0})])
    work = tmp_path / "work"
    work.mkdir()
    sc = fam.score(sset, [run], ref, work, "cpu")
    by = {c.name: c for c in sc.checks}
    assert sc.rows[0]["psnr_median"] == float("inf")
    assert by["median per-frame PSNR vs reference (dB)"].verdict == "PASS"
    assert sc.metrics["rocm_calibrated"]["verdict"] == "PASS"
    assert sc.rows[0]["audio_log_mel_db"] == 0.0 < sc.rows[0]["ref_vs_math_ref_audio_log_mel_db"]
    assert by["median audio log-mel distance to reference minus the reference's own (dB)"].verdict == "PASS"
    assert "PSNR vs reference" not in by and sc.verdict == "PASS", sc.checks


def test_gpu_check_uses_hip_indices(monkeypatch, capsys):
    import subprocess

    from benchmark.correctness import gpus

    seen = []

    def fake_run(cmd, env, **kw):
        seen.append((env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"]))
        if env["HIP_VISIBLE_DEVICES"] == "5":  # a device that doesn't come up
            return subprocess.CompletedProcess(cmd, 1, "", "no device")
        out = json.dumps({"name": "AMD Instinct MI325X", "bdf": "0000:15:00.0", "used_gib": 1.5, "total_gib": 256.0})
        return subprocess.CompletedProcess(cmd, 0, out + "\n", "")

    monkeypatch.setattr(gpus.subprocess, "run", fake_run)
    assert gpus.main(["2,5"]) == 1
    assert seen == [("2", "2"), ("5", "5")]
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["GPU 2 (HIP) 0000:15:00.0 AMD Instinct MI325X: 1.5 of 256.0 GiB in use", "GPU 5 (HIP): missing"]
