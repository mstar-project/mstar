"""The e2e runner's checks must reject wrong outputs. CPU only."""
import base64
import io
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))

from cases import MODELS, Case, ModelSpec, moving_square_mp4, red_square_png, scene_png  # noqa: E402
from matrix import Gpu, cases_without_inputs, config_gpus, gap_report, plan  # noqa: E402
from server import Server, ServerError, free_port, listening_sockets  # noqa: E402
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
    pytest.importorskip("av")
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
    assert not check_text("4.5", four).passed
    assert not check_text("1,400", four).passed
    assert check_text("It is 4.", four).passed
    assert check_text("4, not 5", four).passed
    assert check_text("It is not red.", red).passed  # lenient on purpose: negation is not parsed


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
    report_path = next(tmp_path.glob("*/report.json"))
    assert "does not exist" in json.loads(report_path.read_text())["error"]
    assert re.fullmatch(r"\d{8}-\d{6}-\d+", report_path.parent.name)  # stamped with the runner's pid


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
from run import install_interrupt_handlers
from server import Server
install_interrupt_handlers()
server = Server(Path("."), Path("configs/qwen3_5_0.8b.yaml"), Path(sys.argv[1]),
                command=[sys.executable, "-c", sys.argv[2], "{port}", *sys.argv[3:]], startup_timeout=60,
                settle=2)
with server:
    print(server._proc.pid, server._scratch, flush=True)
    time.sleep(300)
"""


def test_sigterm_to_the_runner_tears_down_the_server_group(tmp_path):
    runner = subprocess.Popen([sys.executable, "-c", RUNNER, str(tmp_path / "log"), FAKE_SERVER],
                              cwd=REPO, stdout=subprocess.PIPE, text=True)
    try:
        pgid = int(runner.stdout.readline().split()[0])
        os.killpg(pgid, 0)  # the server and its child are up
        runner.send_signal(signal.SIGTERM)
        runner.wait(timeout=60)
        assert _group_gone(pgid, 10), f"process group {pgid} survived SIGTERM to the runner"
    finally:
        runner.kill()


def _group_gone(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


# Ignores SIGINT and SIGTERM, as does its child: a worker stuck in a C call.
STUCK_SERVER = """
import http.server, signal, subprocess, sys
for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, signal.SIG_IGN)
subprocess.Popen(["sleep", "300"])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""


def test_a_second_interrupt_during_teardown_still_kills_the_whole_group(tmp_path):
    runner = subprocess.Popen([sys.executable, "-c", RUNNER, str(tmp_path / "log"), STUCK_SERVER],
                              cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pgid = None
    try:
        pgid, scratch = runner.stdout.readline().split()
        pgid = int(pgid)
        runner.send_signal(signal.SIGTERM)  # teardown starts; the server ignores its SIGINT
        time.sleep(1)
        started = time.monotonic()
        runner.send_signal(signal.SIGTERM)  # the second interrupt goes straight to SIGKILL
        runner.wait(timeout=20)
        assert time.monotonic() - started < 15  # well inside the 30 s grace
        assert _group_gone(pgid, 5), f"process group {pgid} survived a double interrupt"
        assert not Path(scratch).exists()
        assert "KeyboardInterrupt" in runner.stderr.read()  # the run still stops
    finally:
        runner.kill()
        _kill_group(pgid if isinstance(pgid, int) else None)


def _kill_group(pgid):
    if pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


NOHUP_PROBE = """
import os, signal, sys, time
sys.path.insert(0, "test/e2e")
from run import install_interrupt_handlers
install_interrupt_handlers()
try:
    os.kill(os.getpid(), signal.SIGHUP)
    time.sleep(0.5)
    print("survived")
except KeyboardInterrupt:
    print("interrupted")
"""


def test_nohup_keeps_sighup_ignored_and_a_plain_run_stops_on_it():
    nohup = shutil.which("nohup")
    if nohup is None:
        pytest.skip("no nohup")

    def probe(*prefix):
        return subprocess.run([*prefix, sys.executable, "-c", NOHUP_PROBE], cwd=REPO, capture_output=True,
                              text=True, timeout=60, check=False).stdout.strip()

    assert probe(nohup) == "survived"
    assert probe() == "interrupted"


# On SIGINT, stops its child (the conductor) with SIGINT, as the entrypoint does.
GRACEFUL_SERVER = """
import http.server, signal, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", sys.argv[2]])
def stop(signum, frame):
    child.send_signal(signal.SIGINT)
    child.wait()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""
CONDUCTOR = """
import signal, sys, time
def record(signum, frame):
    print("conductor got", signal.Signals(signum).name, flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, record)
signal.signal(signal.SIGTERM, record)
while True:
    time.sleep(1)
"""


def test_teardown_stops_the_server_gracefully_before_signalling_the_group(tmp_path):
    log = tmp_path / "log"
    server = Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", log,
                    command=[sys.executable, "-c", GRACEFUL_SERVER, "{port}", CONDUCTOR], startup_timeout=60)
    with server:
        pgid = server._proc.pid
    assert _group_gone(pgid, 1)
    assert "conductor got SIGINT" in log.read_text()
    assert "SIGTERM" not in log.read_text()


# Loads for argv[2] seconds, then binds argv[1].
SLOW_SERVER = """
import http.server, sys, time
time.sleep(float(sys.argv[2]))
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""


def _slow_server(tmp_path, name, load_seconds, port=None):
    return Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / name, port=port, startup_timeout=60,
                  command=[sys.executable, "-c", SLOW_SERVER, "{port}", str(load_seconds)])


def _take_port_soon(port: int) -> socket.socket:
    thief = socket.socket()

    def take():
        thief.bind(("127.0.0.1", port))
        thief.listen()
    threading.Timer(0.3, take).start()
    return thief


needs_proc = pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="port ownership is read from /proc")


@needs_proc
def test_a_port_taken_while_the_model_loads_restarts_the_server_once(tmp_path, monkeypatch):
    import server as server_module

    taken, pick = [], server_module.free_port

    def free_port():
        port = pick()
        if not taken:
            taken.append((port, _take_port_soon(port)))
        return port

    monkeypatch.setattr(server_module, "free_port", free_port)
    server = _slow_server(tmp_path, "log", 2)
    try:
        with server as url:
            assert server.restarts == 1 and server.port != taken[0][0]
            assert requests.get(url + "/health", timeout=5).status_code == 200
    finally:
        taken[0][1].close()


@needs_proc
def test_a_given_port_taken_while_the_model_loads_fails_clearly(tmp_path):
    port = free_port()
    thief = _take_port_soon(port)
    try:
        with pytest.raises(ServerError, match="another process.*--port"):
            _slow_server(tmp_path, "log", 2, port=port).__enter__()
    finally:
        thief.close()


@needs_proc
def test_two_runners_on_one_port_cannot_both_pass(tmp_path):
    port = free_port()
    first, second = _slow_server(tmp_path, "first", 2, port), _slow_server(tmp_path, "second", 5, port)
    entered = {}
    thread = threading.Thread(target=lambda: entered.setdefault("url", first.__enter__()))
    thread.start()
    try:
        time.sleep(0.5)  # both launched before the first one binds
        with pytest.raises(ServerError, match="another process|already in use"):
            second.__enter__()
        thread.join(timeout=60)
        assert requests.get(entered["url"] + "/health", timeout=5).status_code == 200
    finally:
        thread.join(timeout=60)
        first.__exit__(None, None, None)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux only")
def test_a_killed_runner_takes_the_server_with_it(tmp_path):
    runner = subprocess.Popen([sys.executable, "-c", RUNNER, str(tmp_path / "log"), SLOW_SERVER, "0"],
                              cwd=REPO, stdout=subprocess.PIPE, text=True)
    pgid = scratch = None
    try:
        pgid, scratch = runner.stdout.readline().split()
        pgid = int(pgid)
        runner.kill()
        runner.wait(timeout=10)
        assert _group_gone(pgid, 10), "the server outlived a SIGKILLed runner"
    finally:
        runner.kill()
        _kill_group(pgid)
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)  # a killed runner cannot remove it


def _record_launches(monkeypatch, after=None):
    """Patch Popen to record every server launched; ``after(proc)`` runs right after the spawn."""
    spawned, real_popen = [], subprocess.Popen

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        if after:
            after(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    return spawned


def _empty_scratch(monkeypatch, tmp_path) -> Path:
    import server as server_module

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(server_module, "_scratch_parent", lambda: str(scratch))
    return scratch


def test_an_interrupt_right_after_the_spawn_leaves_nothing_behind(tmp_path, monkeypatch):
    scratch = _empty_scratch(monkeypatch, tmp_path)
    spawned = _record_launches(monkeypatch, after=lambda proc: os.kill(os.getpid(), signal.SIGINT))
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    server = _slow_server(tmp_path, "log", 30)
    try:
        with pytest.raises(KeyboardInterrupt):
            server.__enter__()
        assert _group_gone(spawned[0].pid, 5), "the server outlived an interrupt right after its spawn"
        assert server._log is None or server._log.closed
        assert not any(scratch.iterdir())
    finally:
        signal.signal(signal.SIGINT, previous)
        for proc in spawned:
            _kill_group(proc.pid)


def test_a_failed_launch_releases_the_log_and_scratch_dir(tmp_path, monkeypatch):
    scratch = _empty_scratch(monkeypatch, tmp_path)
    server = Server(tmp_path / "missing", REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log",
                    command=[sys.executable, "-c", "pass"])
    with pytest.raises(FileNotFoundError):
        server.__enter__()  # the working directory does not exist
    assert server._log is None or server._log.closed
    assert not any(scratch.iterdir())


@needs_proc
def test_a_group_that_survives_sigkill_is_not_restarted(tmp_path, monkeypatch):
    import server as server_module

    taken, pick = [], server_module.free_port

    def free_port():
        port = pick()
        if not taken:
            taken.append(_take_port_soon(port))
        return port

    monkeypatch.setattr(server_module, "free_port", free_port)
    monkeypatch.setattr(server_module, "_group_alive", lambda pgid: True)  # nothing dies
    spawned = _record_launches(monkeypatch)
    server = Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", startup_timeout=60,
                    command=[sys.executable, "-c", SLOW_SERVER, "{port}", "2"],
                    stop_grace=0.5, settle=0.2, kill_grace=0.5)
    try:
        with pytest.raises(ServerError, match="survived SIGKILL") as failure:
            server.__enter__()
        assert len(spawned) == 1 and server.restarts == 0
        assert str(spawned[0].pid) in str(failure.value)
    finally:
        taken[0].close()
        for proc in spawned:
            _kill_group(proc.pid)


# Loads until argv[2] exists, then binds argv[1] and serves /health.
GATED_SERVER = """
import http.server, os, sys, time
while not os.path.exists(sys.argv[2]):
    time.sleep(0.01)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""


def _gated_server(tmp_path, go: Path) -> Server:
    return Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", startup_timeout=30,
                  command=[sys.executable, "-c", GATED_SERVER, "{port}", str(go)])


def _wait_listening(port: int) -> None:
    deadline = time.monotonic() + 10
    while not listening_sockets(port) and time.monotonic() < deadline:
        time.sleep(0.01)


@needs_proc
@pytest.mark.parametrize("step", ["listening_sockets", "group_sockets"])
def test_a_server_that_starts_listening_mid_check_is_not_restarted(tmp_path, monkeypatch, step):
    import server as server_module

    go = tmp_path / "listen"
    server = _gated_server(tmp_path, go)
    real_step = getattr(server_module, step)

    def listen_after(*args):
        result = real_step(*args)
        if not go.exists():
            go.touch()
            _wait_listening(server.port)
        return result

    monkeypatch.setattr(server_module, step, listen_after)
    with server as url:
        assert server.restarts == 0
        assert requests.get(url + "/health", timeout=5).status_code == 200


@needs_proc
def test_another_listener_seen_while_the_model_loads_does_not_restart_it(tmp_path, monkeypatch):
    import server as server_module

    go, foreign = tmp_path / "listen", socket.socket()
    server = _gated_server(tmp_path, go)
    _record_launches(monkeypatch, after=lambda proc: (foreign.bind(("127.0.0.1", server.port)), foreign.listen()))
    real_listening, reads = server_module.listening_sockets, []

    def listening_sockets(port):
        reads.append(port)
        if len(reads) == 2:  # the first check saw the other listener; it leaves, the model finishes loading
            foreign.close()
            go.touch()
            _wait_listening(port)
        return real_listening(port)

    monkeypatch.setattr(server_module, "listening_sockets", listening_sockets)
    try:
        with server as url:
            assert server.restarts == 0 and len(reads) >= 2
            assert requests.get(url + "/health", timeout=5).status_code == 200
    finally:
        foreign.close()


@needs_proc
def test_a_health_answer_counts_only_if_the_server_still_listens_after_it(tmp_path, monkeypatch):
    server = Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", startup_timeout=3,
                    command=[sys.executable, "-c", FAKE_SERVER, "{port}"])
    real_listening, checks = server._listening, []

    def listener_gone_after_the_answer():
        checks.append(None)
        return real_listening() if len(checks) % 2 else False

    monkeypatch.setattr(server, "_listening", listener_gone_after_the_answer)
    with pytest.raises(ServerError, match="not healthy"):
        server.__enter__()
    assert len(checks) >= 2


@needs_proc
def test_a_server_that_dies_mid_check_is_reported_not_restarted(tmp_path, monkeypatch):
    import server as server_module

    go = tmp_path / "listen"
    go.touch()
    server = _gated_server(tmp_path, go)
    real_group_sockets = server_module.group_sockets

    def die_first(pgid):
        if server_module.listening_sockets(server.port):
            os.killpg(pgid, signal.SIGKILL)
            server._proc.wait(timeout=10)
        return real_group_sockets(pgid)

    monkeypatch.setattr(server_module, "group_sockets", die_first)
    with pytest.raises(ServerError, match="exited with code -9"):
        server.__enter__()
    assert server.restarts == 0


@needs_proc
def test_a_descriptor_closing_mid_scan_hides_no_other_socket(monkeypatch):
    import server as server_module

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        inode = str(os.fstat(listener.fileno()).st_ino)
        real_readlink, mine, failed = os.readlink, f"/proc/{os.getpid()}/fd/", []

        def readlink(path):
            if str(path).startswith(mine) and not failed:
                failed.append(path)
                raise FileNotFoundError(path)  # closed since the listing
            return real_readlink(path)

        monkeypatch.setattr(server_module.os, "readlink", readlink)
        assert inode in server_module.group_sockets(os.getpgrp()) and failed


def _ipv6_listener(v6only: bool) -> socket.socket:
    try:
        listener = socket.socket(socket.AF_INET6)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, int(v6only))
        listener.bind(("::", 0))
    except OSError as exc:
        pytest.skip(f"no IPv6: {exc}")
    listener.listen()
    return listener


def test_an_ipv6_only_listener_does_not_block_the_server_but_a_dual_stack_one_does(tmp_path):
    v6only = _ipv6_listener(True)
    try:
        with _slow_server(tmp_path, "v6only", 0.5, port=v6only.getsockname()[1]) as url:
            assert requests.get(url + "/health", timeout=5).status_code == 200
    finally:
        v6only.close()
    dual = _ipv6_listener(False)
    try:
        with pytest.raises(ServerError, match="in use"):
            _slow_server(tmp_path, "dual", 0.5, port=dual.getsockname()[1]).__enter__()
    finally:
        dual.close()


def test_scratch_dir_stays_short_enough_for_socket_paths(monkeypatch):
    import server as server_module

    monkeypatch.setattr(server_module.tempfile, "gettempdir", lambda: "/" + "x" * 60)
    assert server_module._scratch_parent() == "/tmp"
    monkeypatch.setattr(server_module.tempfile, "gettempdir", lambda: "/tmp/short")
    assert server_module._scratch_parent() == "/tmp/short"


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


def test_an_autotune_shared_memory_rejection_is_not_out_of_memory():
    from run import ran_out_of_memory

    autotune = ("E1017 04:57:20.964000 19461 torch/_inductor/select_algorithm.py:4888] [6/5_1] No valid triton "
                "configs. OutOfMemoryError: out of resource: triton_mm Required: 262144 Hardware limit:232448 "
                "Reducing block sizes or `num_stages` may help..")
    assert not ran_out_of_memory(autotune + "\nINFO ready\n")
    assert ran_out_of_memory(autotune + "\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert ran_out_of_memory("RuntimeError: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)")


def test_exit_code_fails_on_round_trip_errors_and_on_nothing_run():
    from run import exit_code

    passed = {"m": {"status": "passed"}}
    assert exit_code({"models": passed}) == 0
    assert exit_code({"models": passed, "speech_round_trip": {"server_error": "boom"}}) == 1
    assert exit_code({"models": passed, "speech_round_trip": {"server_exit_code": -9}}) == 1
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


def test_only_a_gated_or_missing_repo_skips_and_a_failed_check_runs_with_a_note(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError, OfflineModeIsEnabled

    class GatedError(GatedRepoError):
        def __init__(self):
            Exception.__init__(self, "gated")

    failures = {"org/gated": GatedError(), "org/offline": OfflineModeIsEnabled("offline mode is enabled")}

    def auth_check(repo):
        raise failures[repo]

    monkeypatch.setattr(huggingface_hub, "auth_check", auth_check)
    config = REPO / "configs/qwen3_5_0.8b.yaml"
    notes = []
    gated = plan(_spec(gated=("org/gated",)), config, H200, offline=False, notes=notes)
    assert gated == "gated checkpoint: accept the license at https://huggingface.co/org/gated" and not notes
    assert plan(_spec(gated=("org/offline",)), config, H200, offline=False, notes=notes) is None
    assert len(notes) == 1
    assert notes[0].startswith("could not check access: org/offline (OfflineModeIsEnabled")
    assert notes[0].endswith("; running anyway")


def test_ranks_sharing_a_gpu_count_once():
    assert config_gpus(REPO / "configs/qwen3tts_1p7b_split.yaml") == 1  # rank_devices puts both ranks on GPU 0
    assert config_gpus(REPO / "configs/bagel.yaml") == 2


def test_a_case_whose_input_cannot_be_built_is_skipped_with_a_reason():
    needs = Case("x", "chat", dict, Expect("text"), requires=("not_a_real_module_e2e",))
    assert cases_without_inputs((needs, Case("y", "chat", dict, Expect("text")))) == {
        "x": "its input needs missing package(s): not_a_real_module_e2e"}
    speech_cases = [c for name in ("whisper_large", "higgs_audio", "qwen3tts_base") for c in MODELS[name].cases]
    assert all(c.requires == ("pyarrow", "soundfile") for c in speech_cases)


class _FakeServer:
    started = []

    def __init__(self, *args, **kwargs):
        self.startup_seconds, self.exit_code, self.restarts, self.log_path = 1.0, None, 0, Path("log")
        self.survivors = []
        self.started.append(self)

    def choose_port(self):
        return 0

    def __enter__(self):
        return "http://fake"

    def __exit__(self, *exc):
        return False

    def poll(self):
        return self.exit_code


class _FakeSampler:
    peak = 0.0

    def __init__(self, indices):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_model(monkeypatch, tmp_path, run_case, server=_FakeServer):
    from types import SimpleNamespace

    import run

    _FakeServer.started.clear()
    monkeypatch.setattr(run, "Server", server)
    monkeypatch.setattr(run, "VramSampler", _FakeSampler)
    monkeypatch.setattr(run, "_await_release", lambda *args: None)
    monkeypatch.setattr(run, "run_case", run_case)
    spec = MODELS["qwen3_5"]
    args = SimpleNamespace(port=None, startup_timeout=60, tensor_comm_protocol="SHM", offline=False, eager=True)
    result = {}
    return result, lambda: run.run_model(spec, spec.cases, REPO, REPO / spec.config, tmp_path, args, [], result)


def test_an_interrupted_model_keeps_its_completed_cases(monkeypatch, tmp_path):
    from validators import Verdict

    def run_case(url, case):
        if case.name != "text_to_text":
            raise KeyboardInterrupt
        return Verdict(True, "ok")

    result, run_model = _run_model(monkeypatch, tmp_path, run_case)
    with pytest.raises(KeyboardInterrupt):
        run_model()
    assert list(result["cases"]) == ["text_to_text"]


def test_a_server_that_dies_mid_run_records_its_exit_code(monkeypatch, tmp_path):
    from validators import Verdict

    def run_case(url, case):
        _FakeServer.started[0].exit_code = -9
        return Verdict(True, "ok")

    result, run_model = _run_model(monkeypatch, tmp_path, run_case)
    run_model()
    assert list(result["cases"]) == ["text_to_text"]
    assert result["server_exit_code"] == -9
    assert "exited with code -9 during text_to_text" in result["server_error"]


def test_teardown_records_an_exit_it_sees_before_signalling(tmp_path, monkeypatch):
    import server as server_module

    server = Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", startup_timeout=60,
                    command=[sys.executable, "-c", FAKE_SERVER, "{port}"], stop_grace=5, settle=0.2)
    real_kill, sigints = os.kill, []

    def kill(pid, sig):
        if sig == signal.SIGINT:
            sigints.append(pid)
        return real_kill(pid, sig)

    with server:
        pid, real_poll = server._proc.pid, server.poll

        def crash_right_after_a_check():
            code = real_poll()
            if code is None:  # the server dies just after teardown's check
                real_kill(pid, signal.SIGKILL)
                os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
            return code

        monkeypatch.setattr(server, "poll", crash_right_after_a_check)
        monkeypatch.setattr(server_module.os, "kill", kill)
    assert server.exit_code == -9 or sigints == [pid]  # skipping SIGINT means the exit was seen, so recorded


def test_a_server_that_dies_after_the_last_case_fails_the_profile(monkeypatch, tmp_path):
    from run import model_status
    from validators import Verdict

    class DiesBeforeTeardown(_FakeServer):
        def __exit__(self, *exc):
            self.exit_code = -9  # what teardown's first poll records
            return False

    result, run_model = _run_model(monkeypatch, tmp_path, lambda url, case: Verdict(True, "ok"),
                                   server=DiesBeforeTeardown)
    run_model()
    assert result["server_exit_code"] == -9 and "server_error" not in result
    assert model_status({"source": result}) == "failed"


def test_a_group_that_survives_sigkill_fails_the_profile_with_its_pids(monkeypatch, tmp_path):
    from run import model_status
    from validators import Verdict

    class Stuck(_FakeServer):
        def __exit__(self, *exc):
            self.survivors = [4242]
            raise ServerError("pid(s) 4242 of the server's group survived SIGKILL")

    result, run_model = _run_model(monkeypatch, tmp_path, lambda url, case: Verdict(True, "ok"), server=Stuck)
    run_model()
    assert result["server_survivors"] == [4242] and "survived SIGKILL" in result["server_error"]
    assert model_status({"source": result}) == "failed"


def test_the_port_is_picked_before_the_vram_sampler_starts_forking(monkeypatch, tmp_path):
    import run
    from validators import Verdict

    order = []

    class Recorded(_FakeServer):
        def choose_port(self):
            order.append("port")

    class Sampler(_FakeSampler):
        def __enter__(self):
            order.append("sampler")
            return self

    _, run_model = _run_model(monkeypatch, tmp_path, lambda url, case: Verdict(True, "ok"), server=Recorded)
    monkeypatch.setattr(run, "VramSampler", Sampler)
    run_model()
    assert order == ["port", "sampler"]


class _LaunchedError(Exception):
    pass


def _stop_after_the_port_check(self):
    raise _LaunchedError(self.port)


def _spawn_until(stop: threading.Event):
    while not stop.is_set():
        subprocess.run(["true"], check=False)


def _enter_while_forking(server: Server) -> int:
    """Enter ``server`` (launch stubbed out) just after a thread starts forking, as
    the VRAM sampler does; return the port it would have launched on."""
    stop = threading.Event()
    spawner = threading.Thread(target=_spawn_until, args=(stop,), daemon=True)
    spawner.start()
    try:
        with pytest.raises(_LaunchedError) as launched:
            server.__enter__()
        return launched.value.args[0]
    finally:
        stop.set()
        spawner.join()


def test_a_thread_that_forks_never_makes_a_fresh_port_read_as_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(Server, "_launch", _stop_after_the_port_check)
    for _ in range(60):  # about one in six picks raced before the retry
        _enter_while_forking(Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log"))


def test_an_explicit_port_is_never_re_picked(tmp_path, monkeypatch):
    import server as server_module

    port = free_port()
    monkeypatch.setattr(server_module, "free_port", lambda: pytest.fail("an explicit port was re-picked"))
    monkeypatch.setattr(Server, "_launch", _stop_after_the_port_check)
    for _ in range(30):  # the runner never binds an explicit port, so forks cannot make it busy
        assert _enter_while_forking(Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log",
                                           port=port)) == port
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", port))
        holder.listen()
        with pytest.raises(ServerError, match=f"port {port} is already in use"):
            Server(REPO, REPO / "configs/qwen3_5_0.8b.yaml", tmp_path / "log", port=port).__enter__()
