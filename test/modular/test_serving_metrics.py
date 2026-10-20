"""Prometheus metrics at ``/metrics`` (``--enable-metrics``).

Requests go through the FastAPI app and the API server's real message thread;
only the data worker and the conductor are stand-ins, fed the messages the
real ones send. Each test scrapes ``/metrics`` and reads the samples back.
"""

from __future__ import annotations

import asyncio
import importlib
import queue
import sys
import threading
import time
import types

import httpx
import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestComplete,
    RequestFailed,
    ResultChunk,
)
from mstar.observability import metrics as metrics_module

TEXT = dict(input_modalities=["text"], output_modalities=["text"])


class _DataWorker:
    """What the API server calls on its data worker."""

    def __init__(self):
        self.inputs = []
        self.aborted = []
        self.chunks = queue.Queue()

    def new_request(self, inp):
        self.inputs.append(inp)

    def abort_request(self, key):
        self.aborted.append(key)

    def get_result_chunks(self):
        out = []
        while not self.chunks.empty():
            out.append(self.chunks.get())
        return out

    def get_profile_updates(self):
        return []

    def has_pending_tensors(self, key):
        return False

    def received_final_chunks(self, key, output_chunk_counts):
        return True

    def finished_reading(self, key, drained=True):
        pass


class _Conductor:
    """The API server's socket to the conductor."""

    def __init__(self):
        self.inbox = queue.Queue()

    def get_all_new_messages(self):
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get())
        return out

    def send(self, entity, msg):
        pass


class _Served:
    def __init__(self, server, data_worker, conductor):
        self.server, self.data_worker, self.conductor = server, data_worker, conductor

    def key(self, i=0):
        return self.data_worker.inputs[i].request_id

    def answer(self, key, data=b"hi"):
        """The engine's result chunk, then its completion, as the real data
        worker and conductor deliver them."""
        self.data_worker.chunks.put(ResultChunk(request_id=key, modality="text", data=data))
        self.conductor.inbox.put(APIServerMessage("request_complete", RequestComplete(
            request_id=key, output_chunk_counts={"text": 1},
            conductor_ingest_time=0.0, conductor_finish_time=0.0,
        )))

    def fail(self, key, status=500):
        self.conductor.inbox.put(APIServerMessage("request_failed", RequestFailed(
            request_id=key, error_message="boom", status=status,
        )))

    def scrape(self) -> dict:
        from prometheus_client.parser import text_string_to_metric_families

        r = TestClient(entrypoint.app).get("/metrics")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/plain")
        samples = {}
        for family in text_string_to_metric_families(r.text):
            for s in family.samples:
                samples[(s.name, tuple(sorted(s.labels.items())))] = s.value
        return samples


@pytest.fixture
def served(monkeypatch, tmp_path):
    pytest.importorskip("prometheus_client")
    data_worker, conductor = _DataWorker(), _Conductor()
    server = make_test_api_server(
        preprocess_worker=data_worker, communicator=conductor, upload_dir=tmp_path,
    )
    server.enable_metrics()
    monkeypatch.setattr(entrypoint, "api_server", server)
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    yield _Served(server, data_worker, conductor)
    server.running = False
    thread.join(timeout=5)


def _labels(**kv):
    return tuple(sorted({"model_name": "dummy", **kv}.items()))


def _count(samples, outcome, modalities="text"):
    return samples.get(("mstar_requests_total", _labels(output_modalities=modalities, outcome=outcome)), 0)


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_metrics_answer_404_until_enabled(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", make_test_api_server())
    r = TestClient(entrypoint.app).get("/metrics")
    assert r.status_code == 404
    assert "--enable-metrics" in r.json()["detail"]


@pytest.mark.parametrize("streaming", [False, True])
def test_a_generate_request_moves_the_counter_and_both_histograms(served, streaming):
    form = {"text": "hi", "output_modalities": "text,audio", "streaming": str(streaming).lower()}
    labels = _labels(output_modalities="audio,text")

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/generate", data=form))
            while not served.data_worker.inputs:
                await asyncio.sleep(0.005)
            during = served.scrape()
            served.answer(served.key())
            return await pending, during

    r, during = asyncio.run(run())

    assert r.status_code == 200, r.text
    assert during[("mstar_requests_in_flight", _labels())] == 1
    after = served.scrape()
    assert after[("mstar_requests_in_flight", _labels())] == 0
    assert _count(after, "success", "audio,text") == 1
    for name in ("mstar_e2e_request_latency_seconds", "mstar_time_to_first_output_seconds"):
        assert after[(f"{name}_count", labels)] == 1
        observed = after[(f"{name}_sum", labels)]
        buckets = {
            float(dict(k[1])["le"]): v for k, v in after.items()
            if k[0] == f"{name}_bucket" and dict(k[1])["output_modalities"] == "audio,text"
        }
        assert len(buckets) > 2 and float("inf") in buckets
        # The one observation sits in every bucket at or above its value.
        assert buckets == {le: float(le >= observed) for le in buckets}, name
    first = after[("mstar_time_to_first_output_seconds_sum", labels)]
    e2e = after[("mstar_e2e_request_latency_seconds_sum", labels)]
    assert 0 < first <= e2e


def test_metrics_need_no_api_key_while_generate_still_does(served):
    """Scrapers carry no key, like the /health probe; the request routes keep it."""
    from mstar.api_server import auth

    entrypoint.api_key_guard.set("s3cret-key")
    try:
        client = TestClient(entrypoint.app)
        assert client.post("/generate", data={"text": "hi"}).status_code == 401
        assert client.post("/generate", data={"text": "hi"},
                           headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert not served.data_worker.inputs
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "mstar_requests_in_flight" in r.text
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())


def _engine_fails(served):
    async def run():
        served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
        served.fail(served.key())
        with pytest.raises(HTTPException) as e:
            await served.server.collect_results("r")
        assert e.value.status_code == 500
    asyncio.run(run())


def _stream_is_dropped_after_its_first_chunk(served):
    async def run():
        served.server.submit_request(text="x", request_id="r", **TEXT)
        stream = served.server.iter_result_chunks("r")
        served.data_worker.chunks.put(ResultChunk(request_id=served.key(), modality="text", data=b"a"))
        assert (await stream.__anext__()).data == b"a"
        await stream.aclose()
    asyncio.run(run())


def _non_streaming_times_out(served):
    served.server.timeout_seconds = 0.05

    async def run():
        served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
        with pytest.raises(HTTPException, match="timed out"):
            await served.server.collect_results("r")
    asyncio.run(run())


def _stream_times_out(served):
    served.server.timeout_seconds = 0.05

    async def run():
        served.server.submit_request(text="x", request_id="r", **TEXT)
        with pytest.raises(HTTPException, match="timed out"):
            async for _ in served.server.iter_result_chunks("r"):
                pass
    asyncio.run(run())


def _finished_result_is_released_unread(served):
    served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
    served.answer(served.key())
    _wait(served.server.pending_requests[served.key()].event.is_set)
    served.server.release_request("r")


@pytest.mark.parametrize("end,outcome", [
    (_engine_fails, "error"),
    (_stream_is_dropped_after_its_first_chunk, "abort"),
    (_non_streaming_times_out, "timeout"),
    (_stream_times_out, "timeout"),
    (_finished_result_is_released_unread, "abort"),
], ids=lambda v: v if isinstance(v, str) else v.__name__.strip("_"))
def test_each_way_a_request_ends_counts_once_under_its_outcome(served, end, outcome):
    end(served)

    samples = served.scrape()
    counted = {o: _count(samples, o) for o in metrics_module.OUTCOMES if _count(samples, o)}
    assert counted == {outcome: 1}
    # Latency histograms hold successful requests only.
    assert samples.get(("mstar_e2e_request_latency_seconds_count", _labels(output_modalities="text")), 0) == 0
    assert samples[("mstar_requests_in_flight", _labels())] == 0


def test_every_outcome_is_exported_at_zero_once_its_modalities_are_seen(served):
    """A series that first appears at 1 loses that increment to rate() and
    increase(), so the first error must find its series already at 0."""
    _finished_result_is_released_unread(served)
    samples = served.scrape()
    exported = {o: samples[("mstar_requests_total", _labels(output_modalities="text", outcome=o))]
                for o in metrics_module.OUTCOMES
                if ("mstar_requests_total", _labels(output_modalities="text", outcome=o)) in samples}
    assert exported == {"success": 0, "error": 0, "abort": 1, "timeout": 0}
    assert ("mstar_requests_total", _labels(output_modalities="audio", outcome="error")) not in samples


class _StopError(Exception):
    pass


def _config(monkeypatch, tmp_path, text="model: fake\n") -> str:
    """A config for a model that needs nothing to build."""
    from mstar.model import registry

    class _Model:
        PREPROCESS_TORCH_THREADS = None

        def __init__(self, **kwargs):
            pass

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text)
    monkeypatch.setattr(registry, "get_model_class", lambda name: _Model)
    monkeypatch.setattr(entrypoint, "model_init_kwargs", lambda name: {})
    return str(cfg)


def _main_until_api_server(monkeypatch, tmp_path, *argv):
    """Run ``mstar-serve`` up to building the API server; True if it got there."""
    reached = []

    def fake_api_server(**kwargs):
        reached.append(kwargs)
        raise _StopError

    cfg = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(entrypoint, "APIServer", fake_api_server)
    with pytest.raises((_StopError, SystemExit)):
        entrypoint.main(["--config", cfg, *argv])
    return bool(reached)


def test_metrics_start_after_the_warmup_requests_and_before_the_bind(monkeypatch, tmp_path):
    """Warmups compile first shapes; counting them would fill the latency
    histograms with startup time."""
    events = []

    class _Server:
        fatal_error = None

        def __init__(self, **kwargs):
            pass

        def finalize_setup(self):
            events.append("setup")

        def enable_metrics(self):
            events.append("metrics")

        def set_on_fatal(self, callback):
            pass

        def cleanup(self):
            pass

    class _Conductor:
        pid = 1

        def start(self):
            pass

    class _Uvicorn:
        started = True

        def __init__(self, config):
            pass

        def run(self):
            events.append("bind")

    cfg = _config(monkeypatch, tmp_path, "model: fake\nwarmup_requests:\n  - text: hi\n")
    # main() assigns the module's api_server; restore it for the tests after this one.
    monkeypatch.setattr(entrypoint, "api_server", entrypoint.api_server)
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", None)
    monkeypatch.setattr(entrypoint, "APIServer", _Server)
    monkeypatch.setattr(entrypoint.mp, "get_context",
                        lambda method: types.SimpleNamespace(Process=lambda **kw: _Conductor()))
    monkeypatch.setattr(entrypoint, "_run_warmup_requests", lambda server, specs: events.append("warmup"))
    monkeypatch.setattr(entrypoint.uvicorn, "Server", _Uvicorn)
    monkeypatch.setattr(entrypoint, "_shutdown_conductor_process", lambda proc: None)

    entrypoint.main(["--config", cfg, "--enable-metrics"])

    assert events == ["setup", "warmup", "metrics", "bind"]


def test_enable_metrics_without_the_package_stops_before_anything_starts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", ModuleNotFoundError("prometheus_client"))
    assert not _main_until_api_server(monkeypatch, tmp_path, "--enable-metrics")
    assert "mstar[metrics]" in capsys.readouterr().err
    assert _main_until_api_server(monkeypatch, tmp_path)


def test_enable_metrics_with_the_rust_frontend_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", None)
    assert not _main_until_api_server(monkeypatch, tmp_path, "--enable-metrics", "--rust-frontend")
    assert "Rust frontend" in capsys.readouterr().err


def test_mstar_serve_forwards_enable_metrics(monkeypatch):
    from mstar.cli import main as cli

    seen = []
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", types.SimpleNamespace(main=seen.append))
    cli.main(["serve", "kokoro", "--enable-metrics"])
    cli.main(["serve", "kokoro"])
    assert "--enable-metrics" in seen[0]
    assert "--enable-metrics" not in seen[1]


def test_a_missing_package_is_reported_and_a_broken_one_raises(monkeypatch):
    pytest.importorskip("prometheus_client")
    try:
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "prometheus_client", None)
            importlib.reload(metrics_module)
            assert metrics_module.MISSING_DEPENDENCY is not None
            with pytest.raises(RuntimeError, match=r"mstar\[metrics\]"):
                metrics_module.ServingMetrics("dummy", in_flight=int)
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "prometheus_client.core", None)
            with pytest.raises(ModuleNotFoundError) as e:
                importlib.reload(metrics_module)
            assert e.value.name == "prometheus_client.core"
    finally:
        importlib.reload(metrics_module)
    assert metrics_module.MISSING_DEPENDENCY is None
