"""What ``launch_rust_server`` hands ``mstar-server`` in its environment."""
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
