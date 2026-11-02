"""Prometheus metrics for the API server: ``--enable-metrics`` and ``GET /metrics``.

Requests are counted where they leave :class:`~mstar.api_server.entrypoint.APIServer`,
which ``/generate``, ``/generate/ws``, the OpenAI routes, the Rust-frontend
bridge and the Dynamo bridge all go through. Requests the HTTP frontend
refuses before that (a missing API key, a body over the limit, a bad form)
are counted by route and status in ``mstar_http_refusals_total``. Each family
carries the deployment's ``model_name``; every other label takes values from
a fixed set.

Only the API server process imports ``prometheus_client`` and renders
``/metrics``. Other processes (data worker, conductor, workers) send plain
numbers on messages they already send the API server, plus the conductor's
periodic status message, and the API server records them here; no
multiprocess directory, which would only aggregate one host. Families whose
values come from another process are defined in :mod:`.catalog`.

``prometheus_client`` is optional (``pip install 'mstar[metrics]'``) and only
needed when metrics are on.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
    from prometheus_client.core import (
        CounterMetricFamily,
        GaugeMetricFamily,
        HistogramMetricFamily,
    )
    from prometheus_client.utils import floatToGoString
except ModuleNotFoundError as e:
    # Only a missing package means "not installed"; a broken install must
    # still raise.
    if e.name != "prometheus_client":
        raise
    MISSING_DEPENDENCY: ModuleNotFoundError | None = e
else:
    MISSING_DEPENDENCY = None

from mstar.observability import catalog
from mstar.observability.recorders import MetricSnapshot, merge

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import PendingRequest
    from mstar.api_server.request_types import ConductorStatus, DataWorkerProfile

logger = logging.getLogger(__name__)

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
# Preprocess and delivery: under a millisecond for a short text prompt, up to
# seconds for a long video upload or a large final payload.
STAGE_BUCKETS = (
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)
# Input files: a few KiB for a small image to MSTAR_MAX_BODY_MB (128 MiB by
# default) and the transcription route's larger cap.
_MiB = 1 << 20
UPLOAD_SIZE_BUCKETS = (
    1 << 10, 4 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10,
    _MiB, 2 * _MiB, 4 * _MiB, 8 * _MiB, 16 * _MiB, 32 * _MiB, 64 * _MiB, 128 * _MiB,
    256 * _MiB, 1024 * _MiB,
)
# 16-bit PCM: the data worker emits audio as headerless s16le.
_AUDIO_BYTES_PER_SAMPLE = 2


def require_prometheus_client() -> None:
    if MISSING_DEPENDENCY is not None:
        raise RuntimeError(INSTALL_HINT) from MISSING_DEPENDENCY


def modalities_label(modalities: list[str]) -> str:
    """One label value per set of input or output modalities, e.g. ``audio,text``."""
    return ",".join(sorted(set(modalities)))


class _ScrapedGauge:
    """A gauge read when scraped rather than counted up and down, so a request
    that ends on an unusual path cannot leave it off."""

    def __init__(self, name: str, documentation: str, model_name: str, count: Callable[[], int]):
        self.name = name
        self.documentation = documentation
        self.model_name = model_name
        self.count = count

    def collect(self):
        family = GaugeMetricFamily(self.name, self.documentation, labels=["model_name"])
        family.add_metric([self.model_name], self.count())
        yield family


class _RequestSeries(NamedTuple):
    """One output-modality set's labeled series, looked up once: a
    ``labels()`` call costs about as much as the observation itself."""
    outcomes: dict[str, Any]
    e2e: Any
    first_output: Any
    preprocess: Any
    engine: Any
    delivery: Any
    queue: Any


def _fits(family: catalog.Family, snap: MetricSnapshot) -> bool:
    """Whether ``snap`` has the shape ``family`` exports."""
    rows = len(snap.label_values)
    if any(len(row) != len(family.labels) for row in snap.label_values):
        return False
    if family.kind != catalog.HISTOGRAM:
        return len(snap.values) == rows
    width = len(family.buckets) + 1
    return (len(snap.sums) == rows and len(snap.bucket_counts) == rows
            and all(len(counts) == width for counts in snap.bucket_counts))


class _ConductorStatus:
    """The families in the conductor's latest status message
    (``catalog.STATUS_FAMILIES``), and ``mstar_conductor_status_age_seconds``.

    ``local`` returns the snapshots this process records itself (the data
    worker's transfers), summed into the same families at each scrape.

    The message thread swaps in each message's snapshots whole and a scrape
    reads them whole, so neither needs a lock.
    """

    def __init__(self, model_name: str,
                 local: Callable[[], list[MetricSnapshot]] | None = None):
        self.model_name = model_name
        self._local = local
        self._snapshots: dict[str, MetricSnapshot] = {}
        # Until the first message, the age counts from enabling metrics, so a
        # conductor that never reports shows a growing age, not a missing one.
        self._received = time.monotonic()
        self._warned: set[str] = set()
        self._le = {
            f.name: [floatToGoString(b) for b in f.buckets] + ["+Inf"]
            for f in catalog.STATUS_FAMILIES.values() if f.kind == catalog.HISTOGRAM
        }

    def update(self, status: ConductorStatus) -> None:
        snapshots = {}
        for snap in status.metrics:
            family = catalog.STATUS_FAMILIES.get(snap.name)
            # Only a sender bug gets here; dropping the snapshot keeps it from
            # failing every scrape.
            if family is None or not _fits(family, snap):
                if snap.name not in self._warned:
                    self._warned.add(snap.name)
                    logger.warning("Dropping conductor status for unknown or malformed "
                                   "family %r", snap.name)
                continue
            snapshots[snap.name] = snap
        self._snapshots = snapshots
        self._received = time.monotonic()

    def collect(self):
        age = GaugeMetricFamily(
            "mstar_conductor_status_age_seconds",
            "Seconds since the API server received the conductor's last status "
            "message (since metrics were enabled, before the first one).",
            labels=["model_name"],
        )
        age.add_metric([self.model_name], time.monotonic() - self._received)
        yield age
        snapshots = self._snapshots
        local = {s.name: s for s in self._local()} if self._local is not None else {}
        for family in catalog.STATUS_FAMILIES.values():
            labels = ["model_name", *family.labels]
            snap = snapshots.get(family.name) or MetricSnapshot(family.name, [])
            if family.name in local:
                snap = merge(family, [snap, local[family.name]])
            if family.kind == catalog.HISTOGRAM:
                out = HistogramMetricFamily(family.name, family.help, labels=labels)
                le = self._le[family.name]
                for row, counts, total in zip(
                        snap.label_values, snap.bucket_counts, snap.sums, strict=True):
                    out.add_metric([self.model_name, *row], list(zip(le, counts, strict=True)),
                                   sum_value=total)
            else:
                cls = CounterMetricFamily if family.kind == catalog.COUNTER else GaugeMetricFamily
                out = cls(family.name, family.help, labels=labels)
                for row, value in zip(snap.label_values, snap.values, strict=True):
                    out.add_metric([self.model_name, *row], value)
            yield out


class _Refusals:
    """``mstar_http_refusals_total``: what the Python frontend counts here,
    plus the latest cumulative counts the Rust frontend sends with each
    ``/metrics`` request (``mstar-server`` keeps them; see ``rust/server/src/refusals.rs``).
    Only one of the two frontends serves a deployment, so the sum is that
    frontend's counts."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._python: dict[tuple[str, str], int] = {}
        self._rust: dict[tuple[str, str], int] = {}

    def inc(self, route: str, status: int) -> None:
        key = (route, str(status))
        with self._lock:
            self._python[key] = self._python.get(key, 0) + 1

    def update_rust(self, routes: list[str], statuses: list[int], counts: list[int]) -> None:
        """Take the Rust frontend's cumulative counts (three parallel lists).
        Two scrapes can cross on their way here, so a count only moves up:
        an older snapshot never lowers what a newer one set."""
        with self._lock:
            for route, status, count in zip(routes, statuses, counts, strict=True):
                key = (str(route), str(status))
                self._rust[key] = max(self._rust.get(key, 0), int(count))

    def collect(self):
        family = CounterMetricFamily(
            "mstar_http_refusals",
            "HTTP requests answered with an error before they reached the API server "
            "(a 4xx, or the Rust frontend's admission cap 503), by route template and status.",
            labels=["model_name", "route", "status"],
        )
        with self._lock:
            counts = dict(self._python)
            for key, n in self._rust.items():
                counts[key] = counts.get(key, 0) + n
        for (route, status), n in sorted(counts.items()):
            family.add_metric([self.model_name, route, status], n)
        yield family


class _Request:
    """One HTTP request seen by :class:`RefusalMiddleware`."""

    __slots__ = ("submitted",)

    def __init__(self):
        self.submitted = False


# A mutable holder rather than a bool: run_in_threadpool and new tasks run in
# a copy of the context, which still holds the same object.
_REQUEST: ContextVar[_Request | None] = ContextVar("mstar_http_request", default=None)


def mark_submitted() -> None:
    """The HTTP request being served reached ``APIServer``: it is counted in
    ``mstar_requests_total`` when it ends, so it is never a refusal."""
    request = _REQUEST.get()
    if request is not None:
        request.submitted = True


def route_template(scope: dict, routes: Iterable) -> str:
    """The route path the request matched (``/v1/chat/completions``), never
    the raw URL; ``unmatched`` when no route matches. ``routes`` are the
    app's routes, searched for a request refused before routing (the API
    key, a declared body size) the way the router matches: a full match
    before a method-only one."""
    route = scope.get("route")
    if route is None:
        from starlette.routing import Match

        partial = None
        for candidate in routes:
            # FastAPI 0.150+ lists an included router as one entry with no
            # path, whose match names no route; its routes are listed too.
            if getattr(candidate, "path", None) is None:
                continue
            match, _ = candidate.matches(scope)
            if match is Match.FULL:
                route = candidate
                break
            if match is Match.PARTIAL and partial is None:
                partial = candidate
        route = route or partial
    return getattr(route, "path", None) or "unmatched"


class RefusalMiddleware:
    """Counts ``mstar_http_refusals_total`` for the Python frontend: every
    HTTP response with a 4xx status for a request that never reached
    ``APIServer.submit_request`` (:func:`mark_submitted`). A request that was
    submitted and then failed is counted by ``mstar_requests_total`` instead.

    Sits outside the API key and body limit middlewares (their 401 and 413 are
    sent before routing) and inside CORS (a preflight is not a request).
    WebSocket handshakes are not counted. With metrics off it only checks
    ``metrics()`` for None.
    """

    def __init__(
        self, app, metrics: Callable[[], ServingMetrics | None], routes: Callable[[], Iterable],
    ):
        self.app = app
        self.metrics = metrics
        self.routes = routes

    async def __call__(self, scope, receive, send):
        metrics = self.metrics() if scope["type"] == "http" else None
        if metrics is None:
            await self.app(scope, receive, send)
            return
        request = _Request()
        status = 0

        async def send_and_note_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        token = _REQUEST.set(request)
        try:
            await self.app(scope, receive, send_and_note_status)
        finally:
            _REQUEST.reset(token)
            if 400 <= status < 500 and not request.submitted:
                metrics.refusals.inc(route_template(scope, self.routes()), status)


class _DataWorkerSeries(NamedTuple):
    """One input-modality set's data worker series."""
    seconds: Any
    failures: Any


class ServingMetrics:
    """The API server's registry and its request metrics."""

    def __init__(
        self, model_name: str, in_flight: Callable[[], int],
        data_worker_queue_depth: Callable[[], int],
        local_status: Callable[[], list[MetricSnapshot]] | None = None,
    ):
        """``local_status``: status-family snapshots this process records
        itself, summed with the conductor's (see ``_ConductorStatus``)."""
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
        queue = catalog.REQUEST_QUEUE_SECONDS
        self.queue_time = Histogram(
            queue.name, queue.help, ["model_name", *queue.labels],
            buckets=queue.buckets, registry=self.registry,
        )
        self.registry.register(_ScrapedGauge(
            "mstar_requests_in_flight",
            "Requests the API server holds: submitted and not yet returned, "
            "failed or aborted.",
            model_name, in_flight,
        ))
        self.refusals = _Refusals(model_name)
        self.registry.register(self.refusals)
        self._conductor = _ConductorStatus(model_name, local_status)
        self.registry.register(self._conductor)

        # Where a successful request's time went, from the timestamps it
        # carries. The conductor's perf_counter is compared with this
        # process's: both read the host's monotonic clock.
        stages = ["model_name", "output_modalities"]
        self.preprocess_latency = Histogram(
            "mstar_request_preprocess_seconds",
            "Successful requests: submission to the data worker handing the "
            "request to the conductor, including its wait in the data worker's queue.",
            stages, buckets=STAGE_BUCKETS, registry=self.registry,
        )
        self.engine_latency = Histogram(
            "mstar_request_engine_seconds",
            "Successful requests: conductor ingest to conductor finish.",
            stages, buckets=E2E_BUCKETS, registry=self.registry,
        )
        self.delivery_latency = Histogram(
            "mstar_request_delivery_seconds",
            "Successful requests: conductor finish to the result leaving the API server.",
            stages, buckets=STAGE_BUCKETS, registry=self.registry,
        )

        # Output volume, counted per chunk as it reaches the API server, for
        # every request; modality names come from the model's graph.
        self.output_chunks = Counter(
            "mstar_output_chunks",
            "Result chunks that reached the API server, by modality.",
            ["model_name", "modality"], registry=self.registry,
        )
        self.output_bytes = Counter(
            "mstar_output_bytes",
            "Bytes of result chunks that reached the API server, by modality.",
            ["model_name", "modality"], registry=self.registry,
        )
        self.output_audio_seconds = Counter(
            "mstar_output_audio_seconds",
            "Seconds of audio in the result chunks that reached the API server.",
            ["model_name"], registry=self.registry,
        ).labels(model_name)
        self.time_to_first_chunk = Histogram(
            "mstar_time_to_first_chunk_seconds",
            "Successful requests: submission to the first result chunk of each "
            "output modality reaching the API server.",
            ["model_name", "modality"], buckets=FIRST_OUTPUT_BUCKETS, registry=self.registry,
        )
        # The data worker's own work, from the profile it reports per request
        # it took off its queue; perf_counter is shared, it is a thread here.
        data_worker = ["model_name", "input_modalities"]
        self.data_worker_seconds = Histogram(
            "mstar_data_worker_preprocess_seconds",
            "Requests the data worker preprocessed: from taking the request off "
            "its queue to handing it to the conductor.",
            data_worker, buckets=STAGE_BUCKETS, registry=self.registry,
        )
        self.data_worker_failures = Counter(
            "mstar_data_worker_preprocess_failures",
            "Requests whose preprocessing failed in the data worker (media "
            "limits, undecodable media, the model's prompt processing, any other error).",
            data_worker, registry=self.registry,
        )
        self.upload_size = Histogram(
            "mstar_upload_size_bytes",
            "Size of each image, audio or video file a request carried to the "
            "data worker, uploaded or inline.",
            ["model_name", "modality"], buckets=UPLOAD_SIZE_BUCKETS, registry=self.registry,
        )
        self.registry.register(_ScrapedGauge(
            "mstar_data_worker_queue_depth",
            "Requests submitted and waiting for the data worker to take them up.",
            model_name, data_worker_queue_depth,
        ))

        # Label lookups resolved once per modality set or modality. Two
        # threads may resolve the same key at once; labels() returns the same
        # series to both, so the race is harmless.
        self._request_series: dict[str, _RequestSeries] = {}
        self._first_chunk_series: dict[str, Any] = {}
        self._chunk_series: dict[str, tuple] = {}
        self._data_worker_series: dict[str, _DataWorkerSeries] = {}
        self._upload_series: dict[str, Any] = {}

    def _series(self, modalities: str) -> _RequestSeries:
        series = self._request_series.get(modalities)
        if series is None:
            labels = (self.model_name, modalities)
            series = _RequestSeries(
                # Exported at 0 from here on: rate() and increase() miss the
                # first increment of a series that appears at 1.
                outcomes={o: self.requests.labels(*labels, o) for o in OUTCOMES},
                e2e=self.e2e_latency.labels(*labels),
                first_output=self.time_to_first_output.labels(*labels),
                preprocess=self.preprocess_latency.labels(*labels),
                engine=self.engine_latency.labels(*labels),
                delivery=self.delivery_latency.labels(*labels),
                queue=self.queue_time.labels(*labels),
            )
            self._request_series[modalities] = series
        return series

    def conductor_status(self, status: ConductorStatus) -> None:
        """Export the conductor's latest status message."""
        self._conductor.update(status)

    def record(self, req: PendingRequest, outcome: str) -> None:
        """Count one request that left the API server's table."""
        now = time.perf_counter()
        series = self._series(modalities_label(req.output_modalities))
        series.outcomes[outcome].inc()
        if outcome != "success":
            return
        timing = req.profile.timing
        series.e2e.observe(now - timing.recv_time)
        if timing.first_chunk_time is not None:
            series.first_output.observe(timing.first_chunk_time - timing.recv_time)
        if timing.preprocess_finish_time is not None:
            series.preprocess.observe(timing.preprocess_finish_time - timing.recv_time)
        # Every successful request got its RequestComplete, which carries
        # both; a value from another host's clock would come out negative.
        if timing.conductor_finish_time is not None and now >= timing.conductor_finish_time:
            series.engine.observe(timing.conductor_finish_time - timing.conductor_ingest_time)
            series.delivery.observe(now - timing.conductor_finish_time)
        # Both come on the conductor's RequestComplete; only a conductor spawned
        # with metrics on stamps the arrival.
        if timing.conductor_arrival_time is not None:
            series.queue.observe(timing.conductor_ingest_time - timing.conductor_arrival_time)
        for modality, first in req.first_chunk_times.items():
            first_chunk = self._first_chunk_series.get(modality)
            if first_chunk is None:
                first_chunk = self.time_to_first_chunk.labels(self.model_name, modality)
                self._first_chunk_series[modality] = first_chunk
            first_chunk.observe(first - timing.recv_time)

    def record_chunk(self, modality: str, data: bytes, metadata: dict | None) -> None:
        """Count one result chunk that reached the API server (not an error chunk)."""
        series = self._chunk_series.get(modality)
        if series is None:
            series = (self.output_chunks.labels(self.model_name, modality),
                      self.output_bytes.labels(self.model_name, modality))
            self._chunk_series[modality] = series
        chunks, nbytes = series
        chunks.inc()
        nbytes.inc(len(data))
        if modality == "audio" and metadata:
            # Chunks from the result-token path carry no sample rate.
            rate = metadata.get("sample_rate")
            if rate:
                channels = metadata.get("num_channels") or 1
                self.output_audio_seconds.inc(
                    len(data) / (_AUDIO_BYTES_PER_SAMPLE * channels * rate))

    def record_preprocess(self, profile: DataWorkerProfile) -> None:
        """Record the data worker's numbers for one request it took off its
        queue: its time if preprocessing finished, else a failure, and the size
        of each input file either way."""
        modalities = modalities_label(profile.input_modalities)
        series = self._data_worker_series.get(modalities)
        if series is None:
            labels = (self.model_name, modalities)
            # Both from the first request of a modality set, so the first
            # failure is not lost to rate().
            series = _DataWorkerSeries(
                self.data_worker_seconds.labels(*labels),
                self.data_worker_failures.labels(*labels),
            )
            self._data_worker_series[modalities] = series
        if profile.failed:
            series.failures.inc()
        else:
            series.seconds.observe(profile.preprocess_finish_time - profile.dequeue_time)
        for modality, sizes in profile.upload_sizes.items():
            upload = self._upload_series.get(modality)
            if upload is None:
                upload = self.upload_size.labels(self.model_name, modality)
                self._upload_series[modality] = upload
            for size in sizes:
                upload.observe(size)

    def render(self) -> bytes:
        return generate_latest(self.registry)
