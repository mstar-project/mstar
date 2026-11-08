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


def test_unset_settings_leave_the_frontends_defaults(launched, monkeypatch):
    monkeypatch.delenv("MSTAR_REQUEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MSTAR_SAMPLE_RATE", raising=False)
    rust_frontend.launch_rust_server("bin", "qwen3_5_0.8b", 8000, "/b", "/u")
    assert "MSTAR_REQUEST_TIMEOUT_S" not in launched["env"]
    assert "MSTAR_SAMPLE_RATE" not in launched["env"]
