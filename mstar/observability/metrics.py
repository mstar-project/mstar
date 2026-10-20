"""Prometheus metrics for the API server: ``--enable-metrics`` and ``GET /metrics``.

Requests are counted where they leave :class:`~mstar.api_server.entrypoint.APIServer`,
which ``/generate``, ``/generate/ws``, the OpenAI routes, the Rust-frontend
bridge and the Dynamo bridge all go through. Each family carries the
deployment's ``model_name``; every other label takes values from a fixed set.

Only the API server process imports ``prometheus_client`` and renders
``/metrics``. Other processes (data worker, conductor, workers) send plain
numbers on messages they already send the API server, which records them
here; no multiprocess directory, which would only aggregate one host.

``prometheus_client`` is optional (``pip install 'mstar[metrics]'``) and only
needed when metrics are on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
    from prometheus_client.core import GaugeMetricFamily
except ModuleNotFoundError as e:
    # Only a missing package means "not installed"; a broken install must
    # still raise.
    if e.name != "prometheus_client":
        raise
    MISSING_DEPENDENCY: ModuleNotFoundError | None = e
else:
    MISSING_DEPENDENCY = None

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import PendingRequest

INSTALL_HINT = "--enable-metrics needs prometheus_client: pip install 'mstar[metrics]'"

#: ``success``: the result went to the HTTP layer. ``error``: the request
#: failed (engine, preprocessing, result delivery, conductor gone).
#: ``abort``: the client went away, or its finished result was dropped
#: unread. ``timeout``: the per-request ``--timeout`` expired.
OUTCOMES = ("success", "error", "abort", "timeout")

# Outputs range from a text token in milliseconds to minutes of video, and
# --timeout defaults to 600 s.
FIRST_OUTPUT_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)
E2E_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0,
    15.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0,
)


def require_prometheus_client() -> None:
    if MISSING_DEPENDENCY is not None:
        raise RuntimeError(INSTALL_HINT) from MISSING_DEPENDENCY


def modalities_label(modalities: list[str]) -> str:
    """One label value per set of output modalities, e.g. ``audio,text``."""
    return ",".join(sorted(set(modalities)))


class _InFlight:
    """``mstar_requests_in_flight``, read when scraped rather than counted up
    and down, so a request that ends on an unusual path cannot leave it off."""

    def __init__(self, model_name: str, count: Callable[[], int]):
        self.model_name = model_name
        self.count = count

    def collect(self):
        family = GaugeMetricFamily(
            "mstar_requests_in_flight",
            "Requests the API server holds: submitted and not yet returned, "
            "failed or aborted.",
            labels=["model_name"],
        )
        family.add_metric([self.model_name], self.count())
        yield family


class ServingMetrics:
    """The API server's registry and its request metrics."""

    def __init__(self, model_name: str, in_flight: Callable[[], int]):
        require_prometheus_client()
        self.model_name = model_name
        self.content_type = CONTENT_TYPE_LATEST
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "mstar_requests",
            "Requests that left the API server, by outcome.",
            ["model_name", "output_modalities", "outcome"],
            registry=self.registry,
        )
        self.e2e_latency = Histogram(
            "mstar_e2e_request_latency_seconds",
            "Successful requests: submission to the result leaving the API server.",
            ["model_name", "output_modalities"],
            buckets=E2E_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_output = Histogram(
            "mstar_time_to_first_output_seconds",
            "Successful requests: submission to the first result chunk reaching "
            "the API server.",
            ["model_name", "output_modalities"],
            buckets=FIRST_OUTPUT_BUCKETS,
            registry=self.registry,
        )
        self.registry.register(_InFlight(model_name, in_flight))
        self._seen_modalities: set[str] = set()

    def record(self, req: PendingRequest, outcome: str) -> None:
        """Count one request that left the API server's table."""
        now = time.perf_counter()
        modalities = modalities_label(req.output_modalities)
        if modalities not in self._seen_modalities:
            # rate() and increase() miss the first increment of a series that
            # appears at 1, so every outcome is exported at 0 from here on.
            for o in OUTCOMES:
                self.requests.labels(self.model_name, modalities, o)
            self._seen_modalities.add(modalities)
        self.requests.labels(self.model_name, modalities, outcome).inc()
        if outcome != "success":
            return
        timing = req.profile.timing
        self.e2e_latency.labels(self.model_name, modalities).observe(
            now - timing.recv_time)
        if timing.first_chunk_time is not None:
            self.time_to_first_output.labels(self.model_name, modalities).observe(
                timing.first_chunk_time - timing.recv_time)

    def render(self) -> bytes:
        return generate_latest(self.registry)
