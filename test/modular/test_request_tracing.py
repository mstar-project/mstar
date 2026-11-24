"""CPU checks for the opt-in API request trace."""

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from api_server_helpers import make_test_api_server
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mstar.api_server.entrypoint import PendingRequest
from mstar.observability.tracing import RequestTracing
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


def test_request_span_is_root_even_with_ambient_context():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = RequestTracing("dummy", provider=provider)

    parent = provider.get_tracer("test").start_span("ambient")
    token = context.attach(trace.set_span_in_context(parent))
    try:
        req = PendingRequest(
            streaming=False, input_modalities=["text"], output_modalities=["text"],
            profile=RequestProfile("client", timing=RequestTiming(recv_time=time.perf_counter())),
        )
        req.trace = tracing.start(req, "client#1")
        tracing.finish(req, "success")
    finally:
        context.detach(token)
        parent.end()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["request"].parent is None
    assert spans["request"].context.trace_id != spans["ambient"].context.trace_id
    provider.shutdown()
