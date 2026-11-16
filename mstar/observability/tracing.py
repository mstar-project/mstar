"""Opt-in request tracing from the API server's existing per-request timings.

The worker and conductor report numbers as before. Only the API server loads
OpenTelemetry, and this module loads it only when tracing is enabled.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import PendingRequest

INSTALL_HINT = "--otlp-traces-endpoint needs OpenTelemetry: pip install 'mstar-ai[tracing]'"


def otlp_protocol() -> str:
    """Validate the configured OTLP wire format before model loading."""
    protocol = os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
    )
    if protocol not in ("http/protobuf", "grpc"):
        raise ValueError(f"Unsupported OTLP traces protocol: {protocol!r} (use http/protobuf or grpc)")
    return protocol


def require_tracing() -> None:
    """Fail before model loading, rather than after starting worker processes."""
    otlp_protocol()
    try:
        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
        import opentelemetry.sdk.trace  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("opentelemetry"):
            raise RuntimeError(INSTALL_HINT) from exc
        raise


@dataclass
class RequestSpan:
    span: Any
    wall_ns: int
    perf_ns: int

    def epoch_ns(self, perf_seconds: float) -> int:
        return self.wall_ns + int(perf_seconds * 1_000_000_000) - self.perf_ns


class RequestTracing:
    """One live request span and stage spans reconstructed on completion."""

    def __init__(self, model_name: str, endpoint: str | None = None, *, provider: Any = None):
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        if provider is None:
            require_tracing()
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            protocol = otlp_protocol()
            if protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            else:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider = TracerProvider(resource=Resource.create({
                "service.name": os.getenv("OTEL_SERVICE_NAME", "mstar"),
            }))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        self.provider = provider
        self.tracer = provider.get_tracer("mstar.api_server")
        self._trace = trace
        self._root_context = context.Context()
        self._propagator = TraceContextTextMapPropagator()
        self.model_name = model_name

    def start(self, req: PendingRequest, key: str, traceparent: str | None = None) -> RequestSpan:
        # A request's wall/perf pair is local to its API server. Do not use
        # another host's raw clock for a reconstructed span.
        wall_ns, perf_ns = time.time_ns(), time.perf_counter_ns()
        timing = req.profile.timing
        parent = self._root_context
        if traceparent:
            parent = self._propagator.extract(
                {"traceparent": traceparent}, context=self._root_context,
            )
        span = self.tracer.start_span(
            "request",
            context=parent,
            start_time=wall_ns + int(timing.recv_time * 1_000_000_000) - perf_ns,
            attributes={
                "mstar.model_name": self.model_name,
                "mstar.request_id": req.profile.rid,
                "mstar.submission_key": key,
            },
        )
        return RequestSpan(span, wall_ns, perf_ns)

    def finish(self, req: PendingRequest, outcome: str) -> None:
        state = req.trace
        if state is None:
            return
        span = state.span
        end_ns = time.time_ns()
        try:
            span.set_attribute("mstar.outcome", outcome)
            if outcome != "success":
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, outcome))
            if span.is_recording():
                timing = req.profile.timing
                parent = self._trace.set_span_in_context(span)
                self._stage("preprocess", timing.recv_time, timing.preprocess_finish_time,
                            state.epoch_ns, parent, end_ns)
                self._stage("graph", timing.conductor_ingest_time, timing.conductor_finish_time,
                            state.epoch_ns, parent, end_ns)
                self._stage("delivery", timing.conductor_finish_time,
                            time.perf_counter(), state.epoch_ns, parent, end_ns)
        finally:
            span.end(end_time=end_ns)
            req.trace = None

    def shutdown(self) -> None:
        """Flush the batch exporter when serving stops."""
        self.provider.shutdown()

    def _stage(
        self, name: str, start: float | None, end: float | None,
        epoch_ns: Callable[[float], int], parent: Any, request_end_ns: int,
    ) -> None:
        if start is None or end is None or start <= 0 or end < start:
            return
        start_ns, end_ns = epoch_ns(start), min(epoch_ns(end), request_end_ns)
        if end_ns < start_ns:
            return
        span = self.tracer.start_span(name, context=parent, start_time=start_ns)
        span.end(end_time=end_ns)
