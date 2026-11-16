"""CPU checks for the opt-in API request trace."""

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from api_server_helpers import make_test_api_server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import PendingRequest
from mstar.observability.tracing import RequestTracing, TraceContextMiddleware
from mstar.profile.format import RequestProfile, RequestTiming


def test_request_trace_reuses_existing_timing_and_records_outcome():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = RequestTracing("dummy", provider=provider)
    server = make_test_api_server()
    server.tracing = tracing
    now = time.perf_counter() - .04
    req = PendingRequest(
        streaming=False, input_modalities=["text"], output_modalities=["text"],
        profile=RequestProfile("client-id", timing=RequestTiming(recv_time=now)),
    )
    req.trace = tracing.start(req, "client-id#1")
    timing = req.profile.timing
    timing.preprocess_finish_time = now + .01
    timing.conductor_ingest_time = now + .02
    timing.conductor_finish_time = now + .03
    server._record_end(req, "error")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"request", "preprocess", "graph", "delivery"}
    root = spans["request"]
    assert root.attributes["mstar.request_id"] == "client-id"
    assert root.attributes["mstar.submission_key"] == "client-id#1"
    assert root.attributes["mstar.outcome"] == "error"
    for name in ("preprocess", "graph", "delivery"):
        assert spans[name].parent.span_id == root.context.span_id
        assert spans[name].start_time <= spans[name].end_time
    provider.shutdown()


def test_tracing_is_absent_on_a_server_without_opt_in():
    server = make_test_api_server()
    assert server.tracing is None


def test_submit_request_starts_span_on_the_api_server_path():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    worker = SimpleNamespace(new_request=lambda inp: None)
    server = make_test_api_server(preprocess_worker=worker)
    server.tracing = RequestTracing("dummy", provider=provider)
    rid = server.submit_request(
        request_id="client", input_modalities=["text"], output_modalities=["text"],
    )
    req = server.pending_requests[server.submission_key(rid)]
    assert req.trace is not None
    server._record_end(req, "success")
    root, = exporter.get_finished_spans()
    assert root.name == "request"
    assert root.attributes["mstar.request_id"] == "client"
    provider.shutdown()


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


def test_python_http_traceparent_parents_the_request_span():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = RequestTracing("dummy", provider=provider)

    async def app(scope, receive, send):
        req = PendingRequest(
            streaming=False, input_modalities=["text"], output_modalities=["text"],
            profile=RequestProfile("client", timing=RequestTiming(recv_time=time.perf_counter())),
        )
        req.trace = tracing.start(req, "client#1")
        tracing.finish(req, "success")

    middleware = TraceContextMiddleware(app, tracing=lambda: tracing)
    parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    asyncio.run(middleware(
        {"type": "http", "headers": [(b"traceparent", parent.encode())]},
        None, None,
    ))
    request_span, = exporter.get_finished_spans()
    assert request_span.context.trace_id == int("0123456789abcdef0123456789abcdef", 16)
    assert request_span.parent.span_id == int("0123456789abcdef", 16)
    assert trace.get_current_span().get_span_context().is_valid is False
    provider.shutdown()
