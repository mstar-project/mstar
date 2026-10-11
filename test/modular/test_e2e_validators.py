"""The e2e runner's checks must reject wrong outputs. CPU only."""
import base64
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))

from cases import MODELS, ModelSpec, moving_square_mp4, red_square_png, scene_png  # noqa: E402
from matrix import Gpu, gap_report, plan  # noqa: E402
from server import Server, ServerError  # noqa: E402
from validators import (  # noqa: E402
    Expect,
    check_audio,
    check_chat_response,
    check_float32,
    check_frame_stream,
    check_generate_response,
    check_image,
    check_text,
    check_transcription_response,
    check_video,
)

REPO = Path(__file__).resolve().parents[2]


def _png(color, size=(256, 256), square=True, fmt="PNG"):
    image = Image.new("RGB", size, "white" if square else color)
    if square:
        image.paste(color, (64, 64, 192, 192))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _wav(seconds=1.0, rate=24000, amplitude=8000, width=2, extra_chunk=False):
    t = np.arange(int(seconds * rate)) / rate
    samples = (amplitude * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes() if width == 2 else (samples // 256 + 128).astype(np.uint8).tobytes())
    raw = buf.getvalue()
    if extra_chunk:
        # A LIST chunk before "data": the PCM no longer starts at byte 44.
        info = b"INFOISFT\x06\x00\x00\x00mstar\x00"
        chunk = b"LIST" + struct.pack("<I", len(info)) + info
        raw = raw[:36] + chunk + raw[36:]
        raw = raw[:4] + struct.pack("<I", len(raw) - 8) + raw[8:]
    return raw


def _generate_body(modality, raw):
    return json.dumps({"outputs": {modality: [{"data": base64.b64encode(raw).decode(), "metadata": {}}]}}).encode()


def test_image_checks_color_size_and_blankness():
    red = Expect("image", color="red", size=(256, 256))
    assert check_image(_png("red", square=False), red).passed
    assert not check_image(_png("blue", square=False), red).passed
    assert not check_image(_png("red", size=(128, 128), square=False), red).passed
    assert not check_image(_png("white", square=False), Expect("image")).passed
    assert check_image(_png("blue", square=False, fmt="JPEG"), Expect("image", color="blue")).passed
    assert not check_image(b"not an image", Expect("image")).passed


def test_video_check_decodes_frames_and_refuses_an_image():
    pytest.importorskip("av")
    clip = moving_square_mp4()
    assert check_video(clip, Expect("video", min_frames=16)).passed
    assert not check_video(clip, Expect("video", min_frames=17)).passed
    assert not check_video(_png("red"), Expect("video")).passed


@pytest.mark.parametrize("rate", [16000, 24000, 44100])
def test_audio_check_reads_the_header_rather_than_assuming_one(rate):
    verdict = check_audio(_wav(rate=rate), Expect("audio", min_seconds=0.9))
    assert verdict.passed and f"{rate} Hz" in verdict.detail


def test_audio_check_handles_extra_chunks_and_rejects_bad_audio():
    assert check_audio(_wav(extra_chunk=True), Expect("audio", min_seconds=0.9)).passed
    assert not check_audio(_wav(amplitude=0), Expect("audio")).passed
    assert not check_audio(_wav(seconds=0.2), Expect("audio", min_seconds=0.5)).passed
    assert not check_audio(_wav(width=1), Expect("audio")).passed
    assert not check_audio(_wav()[:30], Expect("audio")).passed


def test_float32_check_rejects_nan_wrong_size_and_partial_values():
    good = np.ones(64, dtype="<f4").tobytes()
    assert check_float32(good, Expect("float32", multiple_of=32)).passed
    assert not check_float32(good, Expect("float32", multiple_of=48)).passed
    assert not check_float32(np.array([1, np.nan], dtype="<f4").tobytes(), Expect("float32")).passed
    assert not check_float32(good[:-1], Expect("float32")).passed
    assert not check_float32(b"", Expect("float32")).passed


def test_generate_response_requires_the_expected_modality():
    image = _png("red", square=False)
    assert check_generate_response(_generate_body("image", image), Expect("image", modality="image")).passed
    missing = check_generate_response(_generate_body("image", image), Expect("video", modality="video"))
    assert not missing.passed and "no 'video' output" in missing.detail
    assert not check_generate_response(_generate_body("video", image), Expect("video", modality="video")).passed
    assert not check_generate_response(b"<html>error</html>", Expect("image", modality="image")).passed


def test_chat_stream_must_finish_and_special_tokens_warn():
    def sse(*events, done=True):
        lines = [f"data: {json.dumps(e)}" for e in events] + (["data: [DONE]"] if done else [])
        return "\n\n".join(lines).encode()

    chunk = {"choices": [{"delta": {"content": "4<|im_end|>"}}]}
    verdict = check_chat_response(sse(chunk), Expect("text", contains_any=("4",)), stream=True)
    assert verdict.passed and verdict.warnings
    assert not check_chat_response(sse(chunk, done=False), Expect("text"), stream=True).passed
    assert not check_chat_response(sse({"error": "boom"}), Expect("text"), stream=True).passed
    wrong = {"choices": [{"message": {"content": "five"}}]}
    assert not check_chat_response(json.dumps(wrong).encode(), Expect("text", contains_any=("4",)), stream=False).passed


def test_every_model_case_is_well_formed():
    for spec in MODELS.values():
        assert (REPO / spec.config).is_file(), spec.config
        names = [c.name for c in spec.cases]
        assert len(names) == len(set(names)), spec.name
        for a, b in spec.must_differ:
            assert a in names and b in names
        for case in spec.cases:
            assert case.route in ("chat", "speech", "transcribe", "generate", "generate_stream")
            if case.route.startswith("generate"):
                assert case.expect.modality, f"{spec.name}/{case.name} must name its output modality"
    assert any(c.smoke for spec in MODELS.values() for c in spec.cases)


def test_generated_fixtures_are_valid_inputs():
    assert Image.open(io.BytesIO(red_square_png())).size == (256, 256)
    assert Image.open(io.BytesIO(scene_png())).size == (224, 224)
    pytest.importorskip("av")
    assert check_video(moving_square_mp4(), Expect("video", min_frames=16)).passed


def test_server_refuses_a_busy_port(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        server = Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", port=busy.getsockname()[1])
        with pytest.raises(ServerError, match="already in use"):
            server.__enter__()


def test_transcription_needs_most_reference_words():
    expect = Expect("text", transcript="Mister Quilter is the apostle of the middle classes")
    assert check_text("mister quilter is the apostle of the middle classes.", expect).passed
    assert check_text("Mr Quilter is the apostle of the middle classes", expect).passed  # 8 of 9 words
    assert not check_text("the weather is nice today", expect).passed
    body = json.dumps({"text": "Mister Quilter is the apostle of the middle classes"}).encode()
    assert check_transcription_response(body, expect).passed
    assert not check_transcription_response(b"{}", expect).passed


def _frame_stream(chunks):
    lines = []
    for modality, frames, meta in chunks:
        data = base64.b64encode(np.asarray(frames, dtype=np.uint8).tobytes()).decode()
        lines.append(json.dumps({"modality": modality, "data": data, "metadata": meta}))
    return "\n".join(lines).encode()


def test_frame_stream_checks_sizes_counts_and_errors():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(4, 8, 16, 3))
    meta = {"height": 8, "width": 16, "frame_count": 4}
    expect = Expect("frames", modality="video_frame", min_frames=8)
    good = _frame_stream([("video_frame", frames, meta)] * 2)
    assert check_frame_stream(good, expect).passed
    assert not check_frame_stream(_frame_stream([("video_frame", frames, meta)]), expect).passed
    assert not check_frame_stream(_frame_stream([("video_frame", frames, {**meta, "width": 15})] * 2), expect).passed
    assert not check_frame_stream(_frame_stream([("video_frame", np.zeros((4, 8, 16, 3)), meta)] * 2), expect).passed
    error = _frame_stream([("error", np.frombuffer(b"boom", dtype=np.uint8), {})])
    assert "stream error" in check_frame_stream(error, expect).detail


def test_chat_audio_is_read_from_the_message():
    audio = base64.b64encode(_wav()).decode()
    body = json.dumps({"choices": [{"message": {"content": "hi", "audio": {"data": audio}}}]}).encode()
    assert check_chat_response(body, Expect("audio", min_seconds=0.5), stream=False).passed
    text_only = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    assert not check_chat_response(text_only, Expect("audio"), stream=False).passed


H200 = [Gpu(0, "NVIDIA H200", 140.0, 0.5)]


def _spec(**kw):
    return ModelSpec(name="t", config="configs/qwen3_5_0.8b.yaml", cases=(), family="llm", **kw)


def test_plan_skips_with_a_reason_and_runs_what_fits():
    assert plan(_spec(), REPO / "configs/qwen3_5_0.8b.yaml", H200, offline=True) is None
    assert plan(_spec(), REPO / "configs/bagel.yaml", H200, offline=True) == "needs 2 GPUs; 1 available"
    assert "measured peak 200" in plan(_spec(peak_vram_gb=200), REPO / "configs/qwen3_5_0.8b.yaml", H200, True)
    reason = plan(_spec(requires=("not_a_real_module_e2e",)), REPO / "configs/qwen3_5_0.8b.yaml", H200, True)
    assert reason == "missing package(s): not_a_real_module_e2e"
    assert plan(_spec(), REPO / "configs/qwen3_5_0.8b.yaml", [], True) == "no CUDA GPU visible"


def test_gap_report_lists_uncovered_configs_with_their_gpu_count():
    gaps = gap_report(REPO)["configs_without_case"]
    assert gaps.get("configs/bagel.yaml") == 2
    assert all(spec.config not in gaps for spec in MODELS.values())


def test_truncated_audio_fails_and_the_digest_covers_rate_and_channels():
    full = _wav(seconds=1.0)
    truncated = full[:44 + 478 * 2]  # header still declares one second
    verdict = check_audio(truncated, Expect("audio", min_seconds=0.5))
    assert not verdict.passed and "declares" in verdict.detail

    pcm = (8000 * np.sin(np.arange(24000) / 10)).astype("<i2").tobytes()

    def wav(rate, channels):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return buf.getvalue()

    digests = {check_audio(wav(r, c), Expect("audio")).sha256 for r, c in [(24000, 1), (48000, 1), (24000, 2)]}
    assert len(digests) == 3


def test_answers_must_match_a_whole_word():
    four, red = Expect("text", contains_any=("4", "four")), Expect("text", contains_any=("red",))
    assert check_text("4<|im_end|>", four).passed
    assert check_text("Four.", four).passed
    assert not check_text("14", four).passed
    assert check_text("Red.", red).passed
    assert not check_text("tired", red).passed


def test_float32_exact_count_rejects_a_partial_tensor():
    one_step = np.ones(32, dtype="<f4").tobytes()
    assert not check_float32(one_step, Expect("float32", count=50 * 32)).passed
    assert check_float32(np.ones(1600, dtype="<f4").tobytes(), Expect("float32", count=50 * 32)).passed


def test_setup_errors_are_reported_and_fail_the_run(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "test/e2e/run.py"), "--models", "qwen3_5",
         "--config", f"qwen3_5={tmp_path / 'missing.yaml'}", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=300, check=False)
    assert proc.returncode == 1
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert "does not exist" in report["error"]


FAKE_SERVER = """
import http.server, subprocess, sys
subprocess.Popen(["sleep", "300"])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""

RUNNER = """
import signal, sys, time
from pathlib import Path
sys.path.insert(0, "test/e2e")
from run import _interrupt
from server import Server
signal.signal(signal.SIGTERM, _interrupt)
server = Server(Path("."), Path("configs/qwen3_5_0.8b.yaml"), Path(sys.argv[1]),
                command=[sys.executable, "-c", sys.argv[2], "{port}"], startup_timeout=60)
with server:
    print(server._proc.pid, flush=True)
    time.sleep(300)
"""


def test_sigterm_to_the_runner_tears_down_the_server_group(tmp_path):
    runner = subprocess.Popen([sys.executable, "-c", RUNNER, str(tmp_path / "log"), FAKE_SERVER],
                              cwd=REPO, stdout=subprocess.PIPE, text=True)
    try:
        pgid = int(runner.stdout.readline())
        os.killpg(pgid, 0)  # the server and its child are up
        runner.send_signal(signal.SIGTERM)
        runner.wait(timeout=60)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"process group {pgid} survived SIGTERM to the runner")
    finally:
        runner.kill()


def test_wait_for_gpu_release_returns_once_memory_is_back(monkeypatch):
    import matrix

    readings = iter([10.0, 6.0, 2.5])
    monkeypatch.setattr(matrix, "gpu_inventory", lambda: [Gpu(0, "x", 140.0, next(readings))])
    monkeypatch.setattr(matrix.time, "sleep", lambda _: None)
    assert matrix.wait_for_gpu_release([0], baseline_gb=2.0) <= 1.0
    monkeypatch.setattr(matrix, "gpu_inventory", lambda: [Gpu(0, "x", 140.0, 9.0)])
    assert matrix.wait_for_gpu_release([0], baseline_gb=2.0, timeout=0) == 7.0


def test_speech_must_say_the_requested_sentence():
    from cases import SENTENCE
    from run import SPEECH_TRANSCRIPT

    assert check_text(f" {SENTENCE}", SPEECH_TRANSCRIPT).passed
    assert check_text("the train leaves tomorrow", SPEECH_TRANSCRIPT).passed  # 4 of 5 words
    assert not check_text("the rain leaves soon", SPEECH_TRANSCRIPT).passed


def test_cuda_visible_devices_selects_and_orders_gpus():
    from matrix import visible

    gpus = [Gpu(0, "a", 80.0, 1.0, "GPU-aaa"), Gpu(1, "b", 80.0, 70.0, "GPU-bbb")]
    assert visible(gpus, None) == gpus
    assert [g.index for g in visible(gpus, "1")] == [1]
    assert [g.index for g in visible(gpus, "1,0")] == [1, 0]
    assert [g.index for g in visible(gpus, "GPU-bbb")] == [1]
    assert visible(gpus, "") == []
    assert [g.index for g in visible(gpus, "0,7,1")] == [0]  # CUDA stops at the first unknown entry


def test_every_server_gets_the_same_environment_flags():
    from types import SimpleNamespace

    from run import server_env

    assert server_env(SimpleNamespace(offline=True, eager=True)) == {
        "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "TORCH_COMPILE_DISABLE": "1"}
    assert server_env(SimpleNamespace(offline=False, eager=False)) == {"PYTHONHASHSEED": "0"}


def test_exit_code_fails_on_round_trip_errors_and_on_nothing_run():
    from run import exit_code

    passed = {"m": {"status": "passed"}}
    assert exit_code({"models": passed}) == 0
    assert exit_code({"models": passed, "speech_round_trip": {"server_error": "boom"}}) == 1
    assert exit_code({"models": passed, "speech_round_trip": {"skipped": "needs 2 GPUs"}}) == 0
    assert exit_code({"models": {"m": {"status": "skipped"}}}) == 1
    assert exit_code({"models": {}}) == 1


def test_a_tier_with_no_case_for_the_model_is_reported_and_fails(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "test/e2e/run.py"), "--models", "bagel", "--tier", "smoke", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=300, check=False)
    assert proc.returncode == 1
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert report["models"]["bagel"]["skipped"] == "no case in the smoke tier"


def test_video_must_have_the_requested_frame_count_and_size():
    pytest.importorskip("av")
    expect = Expect("video", frames=9, size=(192, 320))
    tiny = moving_square_mp4(frames=2, size=16)
    verdict = check_video(tiny, expect)
    assert not verdict.passed and "expected 9 frames" in verdict.detail
    clip = moving_square_mp4(frames=9, size=256)
    assert not check_video(clip, expect).passed  # right length, wrong size
    assert check_video(clip, Expect("video", frames=9, size=(256, 256))).passed


def test_video_digest_covers_size_and_timing():
    pytest.importorskip("av")
    expect = Expect("video")
    slow, fast = (check_video(moving_square_mp4(fps=fps), expect).sha256 for fps in (8, 16))
    assert slow != fast
    assert check_video(moving_square_mp4(size=128), expect).sha256 != slow


def test_frame_stream_checks_the_requested_size():
    frames = np.random.default_rng(1).integers(0, 255, size=(4, 8, 16, 3))
    body = _frame_stream([("video_frame", frames, {"height": 8, "width": 16, "frame_count": 4})] * 2)
    assert check_frame_stream(body, Expect("frames", modality="video_frame", frames=8, size=(16, 8))).passed
    assert not check_frame_stream(body, Expect("frames", modality="video_frame", frames=8, size=(32, 8))).passed


def test_offline_fixtures_come_from_the_local_cache_only(monkeypatch):
    import cases
    import huggingface_hub

    calls = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after recording the call")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(cases, "OFFLINE", True)
    cases.speech_sample.cache_clear()
    with pytest.raises(RuntimeError, match="stop after"):
        cases.speech_sample()
    cases.speech_sample.cache_clear()
    assert calls[0]["local_files_only"] is True


def test_every_family_but_omni_has_a_smoke_case():
    families = {spec.family for spec in MODELS.values()}
    with_smoke = {spec.family for spec in MODELS.values() if any(c.smoke for c in spec.cases)}
    assert families - with_smoke == {"omni"}  # its only profile is a 30B model
