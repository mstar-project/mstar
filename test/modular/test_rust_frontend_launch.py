"""What ``launch_rust_server`` hands ``mstar-server`` in its environment."""
import os
import time

import pytest

from mstar.api_server import rust_frontend


@pytest.fixture
def launched(monkeypatch):
    seen = {}

    class _Popen:
        pid = 1

        def __init__(self, argv, env):
            seen["argv"], seen["env"] = argv, env

    monkeypatch.setattr(rust_frontend.subprocess, "Popen", _Popen)
    monkeypatch.setattr(rust_frontend, "openai_adapter_class", lambda name: "Qwen3_5Adapter")
    return seen


def test_the_timeout_and_the_models_sample_rate_win_over_the_environment(launched, monkeypatch):
    monkeypatch.setenv("MSTAR_REQUEST_TIMEOUT_S", "5")
    monkeypatch.setenv("MSTAR_SAMPLE_RATE", "16000")
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u", timeout_s=1800.5, sample_rate=48000)
    env = launched["env"]
    assert env["MSTAR_REQUEST_TIMEOUT_S"] == "1800.5"
    assert env["MSTAR_SAMPLE_RATE"] == "48000"
    assert env["MSTAR_OPENAI_ADAPTER"] == "Qwen3_5Adapter"


# inf and nan never time out on the Python frontend, and a timeout at or below 0 times
# every request out; the Rust frontend reads a finite number of seconds at least 0
@pytest.mark.parametrize(("timeout", "seconds"), [(float("inf"), "1000000000.0"), (float("nan"), "1000000000.0"),
                                                  (-5.0, "0.0"), (float("-inf"), "0.0")])
def test_a_timeout_the_rust_frontend_cannot_read_is_sent_as_its_effect(launched, timeout, seconds):
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u", timeout_s=timeout)
    assert launched["env"]["MSTAR_REQUEST_TIMEOUT_S"] == seconds


def test_unset_settings_leave_the_frontends_defaults(launched, monkeypatch):
    monkeypatch.delenv("MSTAR_REQUEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MSTAR_SAMPLE_RATE", raising=False)
    rust_frontend.launch_rust_server("bin", "qwen3_5_0.8b", 8000, "/b", "/u")
    assert "MSTAR_REQUEST_TIMEOUT_S" not in launched["env"]
    assert "MSTAR_SAMPLE_RATE" not in launched["env"]


def test_the_transcription_facts_are_the_launchers_and_never_inherited(launched, monkeypatch):
    import json

    monkeypatch.setenv("MSTAR_TRANSCRIPTION", '{"languages": ["stale"]}')
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u")
    assert "MSTAR_TRANSCRIPTION" not in launched["env"]
    facts = {"request_kwargs": None, "max_concurrent_requests": 8, "errors_url": "u"}
    rust_frontend.launch_rust_server("bin", "whisper_large", 8000, "/b", "/u", transcription=facts)
    assert json.loads(launched["env"]["MSTAR_TRANSCRIPTION"]) == facts


def test_the_websocket_facts_are_the_launchers_and_never_inherited(launched, monkeypatch):
    import json

    for name in ("MSTAR_OPENPI", "MSTAR_WS_MAX_BYTES", "MSTAR_REALTIME"):
        monkeypatch.setenv(name, "stale")
    monkeypatch.setenv("MSTAR_CLIENT_STREAMS", '["stale"]')
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u")
    env = launched["env"]
    assert not {"MSTAR_OPENPI", "MSTAR_WS_MAX_BYTES", "MSTAR_REALTIME"} & env.keys()
    assert json.loads(env["MSTAR_CLIENT_STREAMS"]) == []
    openpi = {"image_keys": ["cam"], "metadata": ""}
    realtime = {"onnxruntime": None, "silero": None, "smart_turn_problem": "x", "tokenizer": None}
    rust_frontend.launch_rust_server("bin", "qwen3_asr", 8000, "/b", "/u", openpi=openpi, ws_max_bytes=1 << 20,
                                     client_streams={"text", "audio"}, realtime=realtime)
    env = launched["env"]
    assert json.loads(env["MSTAR_OPENPI"]) == openpi
    assert env["MSTAR_WS_MAX_BYTES"] == str(1 << 20)
    assert json.loads(env["MSTAR_CLIENT_STREAMS"]) == ["audio", "text"]
    assert json.loads(env["MSTAR_REALTIME"]) == realtime


def test_mstar_server_gets_the_libsndfile_soundfile_loaded(launched, monkeypatch):
    monkeypatch.setenv("MSTAR_LIBSNDFILE", "/stale/libsndfile.so")
    monkeypatch.setattr(rust_frontend, "soundfile_library", lambda: None)
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u")
    assert "MSTAR_LIBSNDFILE" not in launched["env"]
    monkeypatch.setattr(rust_frontend, "soundfile_library", lambda: "/sf/libsndfile_x86_64.so")
    rust_frontend.launch_rust_server("bin", "kokoro", 8000, "/b", "/u")
    assert launched["env"]["MSTAR_LIBSNDFILE"] == "/sf/libsndfile_x86_64.so"


def test_mstar_server_gets_the_api_servers_served_models(launched, monkeypatch):
    from mstar.api_server.served_models import ServedModels

    monkeypatch.setenv("MSTAR_SERVED_MODELS", '{"base": ["stale"]}')
    rust_frontend.launch_rust_server("bin", "qwen3_5_0.8b", 8000, "/b", "/u")
    assert "MSTAR_SERVED_MODELS" not in launched["env"]
    served = ServedModels(base=["qwen3_5_0.8b", "Qwen/Qwen3.5-0.8B"], digests={"sql": b"\x01" * 32}, lora=True)
    rust_frontend.launch_rust_server("bin", "qwen3_5_0.8b", 8000, "/b", "/u", served_models=served)
    import json
    assert json.loads(launched["env"]["MSTAR_SERVED_MODELS"]) == {
        "base": ["Qwen/Qwen3.5-0.8B", "qwen3_5_0.8b"], "adapters": ["sql"], "lora": True}


def test_realtime_facts_carry_what_a_conversation_reads_off_the_model(tmp_path):
    """A model with a ClientAudio stream gets ``conversation`` facts; only one
    that can converse (turn tokens, one audio stream, mono output) gets its
    stream and reply settings, which only a conversation reads."""
    from types import SimpleNamespace

    from mstar.model.base import Model
    from mstar.streaming.topology import CLIENT, ClientAudio, Connection

    class _Duplex:
        turn_tokens, turn_idle_steps, turn_silence_rms = (1, 2), 25, 0.01
        get_output_sample_rate = Model.get_output_sample_rate
        get_output_audio_channels = Model.get_output_audio_channels

        def get_default_voice(self):
            return "Aria"

    ear = Connection(CLIENT, "Encoder", "audio_in", chunk_policy_factory=lambda: None, audio=ClientAudio(16000, 1280))
    server = SimpleNamespace(model=_Duplex(), client_audio=[ear])
    assert rust_frontend.realtime_facts(server, None, str(tmp_path)) == {"conversation": {
        "streams": 1, "turn_tokens": True, "channels": 1, "stream": "audio_in", "rate": 16000, "frame_samples": 1280,
        "output_rate": 24000, "voice": "Aria", "idle_steps": 25, "silence_rms": 0.01,
    }}
    server.model.turn_tokens = None
    assert rust_frontend.realtime_facts(server, None, str(tmp_path)) == {
        "conversation": {"streams": 1, "turn_tokens": False, "channels": 1}}
    assert rust_frontend.realtime_facts(SimpleNamespace(model=_Duplex(), client_audio=[]), None, str(tmp_path)) is None


def _wait_until_caught(pid: int, signum: int, timeout: float = 30) -> None:
    """Waits until ``pid`` has a handler installed for ``signum`` (``SigCgt``
    in /proc/<pid>/status), so a test signals it only once it is listening."""

    if not os.path.exists("/proc/self/status"):
        pytest.skip("no /proc to tell when a signal handler is in place")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with open(f"/proc/{pid}/status") as status:
            caught = next(int(line.split()[1], 16) for line in status if line.startswith("SigCgt:"))
        if caught >> (signum - 1) & 1:
            return
        time.sleep(0.01)
    raise AssertionError(f"pid {pid} never installed a handler for signal {signum}")


@pytest.mark.parametrize("signame", ["SIGINT", "SIGTERM"])
def test_the_bridge_stops_on_a_signal_even_with_sigint_ignored(tmp_path, signame):
    """As uvicorn stops on either: a server started as a background job inherits
    SIGINT ignored, where asyncio.run would leave the bridge's loop running."""
    import signal
    import subprocess
    import sys

    pytest.importorskip("mstar_rust")
    script = (
        "import signal, sys, types\n"
        # what a background job inherits; the bridge is built after, as it is in a launched server
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "from mstar.api_server.rust_frontend import RustFrontendBridge\n"
        f"bridge = RustFrontendBridge(types.SimpleNamespace(), {str(tmp_path)!r})\n"
        "bridge.run()\n"
        "print('stopped', flush=True)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        # the bridge installs SIGINT's handler, then SIGTERM's
        _wait_until_caught(proc.pid, signal.SIGTERM)
        proc.send_signal(getattr(signal, signame))
        out, _ = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert "stopped" in out and proc.returncode == 0


@pytest.mark.parametrize("signals", [1, 2], ids=["drains", "second-signal"])
def test_a_signal_lets_mstar_server_drain_before_the_bridge_stops(tmp_path, signals):
    """The first SIGINT goes on to mstar-server, and the bridge keeps relaying
    while it drains, so in-flight streams can finish; mstar-server's exit then
    stops the bridge. A second stops the bridge at once, as a second Ctrl-C
    stops uvicorn."""
    import signal
    import subprocess
    import sys

    pytest.importorskip("mstar_rust")
    events = tmp_path / "events"
    log = f"def log(event): open({str(events)!r}, 'a').write(event + '\\n')\n"
    frontend = (  # mstar-server's stand-in: draining takes 1 s, or longer than the test with a second signal coming
        "import signal, sys, time\n" + log +
        "def drain(signum, frame):\n"
        "    log('drain ' + signal.Signals(signum).name)\n"
        f"    time.sleep({60 if signals == 2 else 1})\n"
        "    log('frontend exited')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGINT, drain)\n"
        "print(flush=True)\n"
        "time.sleep(60)\n"
    )
    script = (
        "import subprocess, sys, threading, types\n" + log +
        "from mstar.api_server.entrypoint import _stop_with\n"
        "from mstar.api_server.rust_frontend import RustFrontendBridge\n"
        f"frontend = subprocess.Popen([sys.executable, '-c', {frontend!r}], stdout=subprocess.PIPE)\n"
        "frontend.stdout.readline()  # its handler is in place\n"
        f"bridge = RustFrontendBridge(types.SimpleNamespace(), {str(tmp_path)!r})\n"
        "threading.Thread(target=_stop_with, args=(frontend, bridge), daemon=True).start()\n"
        "bridge.run(frontend=frontend)\n"
        "log('bridge stopped')\n"
        "frontend.kill()\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        _wait_until_caught(proc.pid, signal.SIGTERM)
        proc.send_signal(signal.SIGINT)
        if signals == 2:
            deadline = time.monotonic() + 30
            while not (events.exists() and "drain" in events.read_text()):
                assert time.monotonic() < deadline, "mstar-server was never told to stop"
                time.sleep(0.01)
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    finally:
        proc.kill()
    drained = ["frontend exited"] if signals == 1 else []  # with a second signal it was still draining
    assert events.read_text().splitlines() == ["drain SIGINT", *drained, "bridge stopped"]
    assert proc.returncode == 0
