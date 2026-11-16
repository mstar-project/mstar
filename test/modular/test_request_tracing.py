"""CPU checks for the opt-in API request trace."""

import asyncio
import logging
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
from mstar.api_server.rust_frontend import RustFrontendBridge
from mstar.observability.tracing import RequestTracing
from mstar.profile.format import GraphTiming, RequestProfile, RequestTiming
from mstar.utils.logging_config import REQUEST_LOGGER


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


def test_node_activity_windows_are_children_of_graph_span():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = RequestTracing("dummy", provider=provider)
    now = time.perf_counter() - .04
    req = PendingRequest(
        streaming=False, input_modalities=["text"], output_modalities=["text"],
        profile=RequestProfile("client", timing=RequestTiming(recv_time=now)),
    )
    req.trace = tracing.start(req, "client#1")
    req.profile.timing.conductor_ingest_time = now + .01
    req.profile.timing.conductor_finish_time = now + .03
    req.profile.graph_timings = [
        GraphTiming("decoder", "decode", 2, .012, .006, .004, .002,
                    first_start=now + .012, last_end=now + .028),
        GraphTiming("outside", "decode", 1, .001, .001, 0, 0,
                    first_start=now - .1, last_end=now - .09),
        GraphTiming("missing", "decode", 1, .001, .001, 0, 0),
    ]
    tracing.finish(req, "success")

    spans = exporter.get_finished_spans()
    root, graph, node = (next(s for s in spans if s.name == name)
                         for name in ("request", "graph", "node.activity"))
    assert len([s for s in spans if s.name == "node.activity"]) == 1
    assert graph.parent.span_id == root.context.span_id
    assert node.parent.span_id == graph.context.span_id
    assert graph.start_time <= node.start_time <= node.end_time <= graph.end_time
    assert node.attributes == {
        "mstar.node": "decoder", "mstar.graph_walk": "decode", "mstar.exec_count": 2,
        "mstar.forward_sum_s": .006, "mstar.preprocess_sum_s": .004,
        "mstar.postprocess_sum_s": .002,
    }
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


def test_submit_sends_its_own_span_context_to_preprocess_worker(caplog):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inputs = []
    server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=inputs.append))
    server.tracing = RequestTracing("dummy", provider=provider)
    parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    with caplog.at_level(logging.DEBUG, logger=REQUEST_LOGGER):
        rid = server.submit_request(
            request_id="client", input_modalities=["text"], output_modalities=["text"],
            traceparent=parent,
        )
    child = inputs[0].traceparent
    assert child is not None and child != parent
    _, trace_id, span_id, flags = child.split("-")
    req = server.pending_requests[server.submission_key(rid)]
    context = req.trace.span.get_span_context()
    assert int(trace_id, 16) == context.trace_id
    assert int(span_id, 16) == context.span_id
    assert flags == "01"
    assert f"Request client submitted as {server.submission_key(rid)}" in caplog.text
    assert f"trace_id={trace_id}" in caplog.text
    server._record_end(req, "success")
    provider.shutdown()


def test_untraced_submit_log_has_no_trace_id(caplog):
    worker = SimpleNamespace(new_request=lambda inp: None)
    server = make_test_api_server(preprocess_worker=worker)
    with caplog.at_level(logging.DEBUG, logger=REQUEST_LOGGER):
        server.submit_request(
            request_id="client", input_modalities=["text"], output_modalities=["text"],
        )
    assert "Request client submitted as client#1" in caplog.text
    assert "trace_id=" not in caplog.text


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


@pytest.mark.parametrize("traceparent,has_parent", [
    ("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01", True),
    ("not-a-traceparent", False),
])
def test_submit_traceparent_parents_only_a_valid_request(traceparent, has_parent):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=lambda inp: None))
    server.tracing = RequestTracing("dummy", provider=provider)
    rid = server.submit_request(
        request_id="client", input_modalities=["text"], output_modalities=["text"],
        traceparent=traceparent,
    )
    req = server.pending_requests[server.submission_key(rid)]
    server._record_end(req, "success")
    span, = exporter.get_finished_spans()
    if has_parent:
        assert span.parent.trace_id == int("0123456789abcdef0123456789abcdef", 16)
        assert span.parent.span_id == int("0123456789abcdef", 16)
    else:
        assert span.parent is None
    provider.shutdown()


def test_rust_bridge_keeps_the_parent_on_the_submission():
    async def run():
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        inputs = []
        server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=inputs.append))
        server.tracing = RequestTracing("dummy", provider=provider)

        async def empty_result(request_id):
            if False:
                yield request_id

        server.iter_result_chunks = empty_result

        bridge = RustFrontendBridge.__new__(RustFrontendBridge)
        bridge.server = server
        bridge._keys = {}
        bridge._send = lambda msg: None
        header = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        bridge._submit({"t": "submit", "rid": "chatcmpl-1", "sub": 1, "traceparent": header})
        key = server.submission_key("chatcmpl-1")
        assert key is not None
        assert inputs[0].traceparent is not None
        assert inputs[0].traceparent != header
        assert inputs[0].traceparent.split("-")[1] == header.split("-")[1]
        server._record_end(server.pending_requests[key], "success")
        await asyncio.sleep(0)
        span, = exporter.get_finished_spans()
        assert span.parent.trace_id == int("0123456789abcdef0123456789abcdef", 16)
        provider.shutdown()

    asyncio.run(run())


def test_rust_bridge_leaves_other_routes_without_a_traceparent_kwarg():
    async def run():
        server = make_test_api_server(preprocess_worker=SimpleNamespace(new_request=lambda inp: None))
        submit = server.submit_request

        def without_parent(**kwargs):
            assert "traceparent" not in kwargs
            return submit(**kwargs)

        server.submit_request = without_parent

        async def empty_result(request_id):
            if False:
                yield request_id

        server.iter_result_chunks = empty_result
        bridge = RustFrontendBridge.__new__(RustFrontendBridge)
        bridge.server = server
        bridge._keys = {}
        bridge._send = lambda msg: None
        bridge._submit({"t": "submit", "rid": "other-route-1", "sub": 1})
        assert server.submission_key("other-route-1") is not None
        await asyncio.sleep(0)

    asyncio.run(run())
