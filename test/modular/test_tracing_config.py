"""Tracing startup checks that run without the optional OpenTelemetry SDK."""

import pytest
from api_server_helpers import make_test_api_server

from mstar.api_server import entrypoint


def test_tracing_is_absent_on_a_server_without_opt_in():
    server = make_test_api_server()
    assert server.tracing is None


def test_missing_tracing_extra_fails_before_config_or_model_load(monkeypatch, capsys):
    def missing_extra():
        raise RuntimeError("--otlp-traces-endpoint needs OpenTelemetry: pip install 'mstar-ai[tracing]'")

    monkeypatch.setattr(entrypoint, "require_tracing", missing_extra)
    with pytest.raises(SystemExit):
        entrypoint.main([
            "--config", "/nonexistent/config.yaml", "--otlp-traces-endpoint",
            "http://localhost:4318/v1/traces",
        ])
    assert "mstar-ai[tracing]" in capsys.readouterr().err


def test_unsupported_protocol_fails_before_config_or_model_load(monkeypatch, capsys):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/json")
    with pytest.raises(SystemExit):
        entrypoint.main([
            "--config", "/nonexistent/config.yaml", "--otlp-traces-endpoint",
            "http://localhost:4318/v1/traces",
        ])
    assert "Unsupported OTLP traces protocol" in capsys.readouterr().err
