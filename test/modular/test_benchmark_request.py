"""benchmark/request.py and runner.py: what the client sends and how a run's figures are counted (CPU)."""

import asyncio
import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from benchmark import runner  # noqa: E402
from benchmark.base import Model, Orpheus, RequestType  # noqa: E402
from benchmark.request import (  # noqa: E402
    OursOpenAI,
    RequestInput,
    RequestMetrics,
    SGLangOmni,
    _record_vllm_omni_chunk,
    aggregate_metrics,
)


class _TextModel(Model):
    def get_hf_url(self):
        return "m"

    def get_supported_modalities(self):
        return {RequestType.T2T}


class _Response:
    status = 200

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    @property
    def content(self):
        return self

    async def _iter(self):
        for line in self._lines:
            yield line

    def __aiter__(self):
        return self._iter()

    def iter_any(self):
        return self._iter()


class _Session:
    """aiohttp.ClientSession stand-in: keeps the last payload, answers with ``lines``."""

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.payload = None

    def post(self, url, json=None, **kwargs):
        self.payload = json
        return _Response(self.lines)


def _metrics(request_id, *, ok, ttft, tokens):
    m = RequestMetrics(request_id=request_id, type=RequestType.T2T)
    m.start_time = 0.0
    m.ttft = {"text": ttft}
    m.output_text_tokens = m.input_tokens = tokens
    if ok:
        m.record_completion()
        m.e2e_latency = 1.0
    else:
        m.record_error("HTTP 500")
    return m


def test_a_failed_request_counts_in_no_figure():
    """A fast failure would pull the latency stats down, and a request cut
    mid-stream would add its partial tokens to the throughput (#1043)."""
    agg = aggregate_metrics(
        [_metrics(0, ok=True, ttft=0.5, tokens=10), _metrics(1, ok=False, ttft=0.01, tokens=4)], wall_time=1.0,
    )
    assert (agg.ttft["text"].mean, agg.e2e_latency.mean) == (0.5, 1.0)
    assert (agg.total_text_tokens, agg.total_input_tokens, agg.text_token_throughput) == (10, 10, 10.0)


def test_an_audio_chunk_with_completion_tokens_zero_keeps_the_text_count():
    """vllm-omni and sglang-omni send completion_tokens=0 on audio chunks (#1045)."""
    m = RequestMetrics(request_id=0, type=RequestType.T2S)
    _record_vllm_omni_chunk({"usage": {"completion_tokens": 7}, "choices": []}, m)
    _record_vllm_omni_chunk({"usage": {"completion_tokens": 0}, "modality": "audio", "choices": []}, m)
    assert m.output_text_tokens == 7

    lines = [b"data: " + json.dumps({"usage": {"completion_tokens": n}, "choices": []}).encode() for n in (7, 0)]
    req = RequestInput(req_type=RequestType.T2T, prompt="hi")
    m = asyncio.run(SGLangOmni().send_request(_Session(lines + [b"data: [DONE]"]), req, "http://x", 0, _TextModel()))
    assert m.output_text_tokens == 7


@pytest.mark.parametrize(
    ("system", "model", "req_type"),
    [
        pytest.param(SGLangOmni, _TextModel, RequestType.T2T, id="sglang_omni"),
        pytest.param(OursOpenAI, Orpheus, RequestType.T2S, id="mstar_audio_speech"),
    ],
)
def test_the_stamped_output_length_reaches_the_payload(system, model, req_type):
    """The runner stamps max_tokens / ignore_eos per request; an engine that
    drops them generates its own lengths in a fixed-length comparison (#1044)."""
    session = _Session()
    req = RequestInput(req_type=req_type, prompt="hi", model_kwargs={"max_tokens": 33, "ignore_eos": True})
    asyncio.run(system().send_request(session, req, "http://x", 0, model()))
    assert (session.payload["max_tokens"], session.payload["ignore_eos"]) == (33, True)


def test_an_offline_run_stamps_each_request_its_own_length_and_fills_its_last_wave(monkeypatch):
    """A dataset resized past its length repeats the same RequestInput objects:
    stamping them in place gave repeats one shared length. The offline round-up
    to a whole wave must happen before the dataset is sized."""
    cfg = runner.BenchmarkConfig(
        url="http://x", model=_TextModel(), dataset=runner.DatasetType.TEXT, num_requests=5,
        request_type=RequestType.T2T, local_cache_dir=".", num_warmup=0, batch_size=4,
        profiling_type=runner.ProfilingType.OFFLINE, output_len_min=1, output_len_max=1000,
    )
    bench = runner.Benchmark.__new__(runner.Benchmark)
    bench.config = cfg
    base = [RequestInput(req_type=RequestType.T2T, prompt=str(i)) for i in range(2)]
    bench._get_dataset = lambda: SimpleNamespace(get_requests=lambda: (base * 8)[: cfg.num_requests])
    sent = []

    async def offline(session, requests):
        sent.extend(requests)
        return []

    bench._run_concurrent_offline = offline
    monkeypatch.setattr(runner, "aggregate_metrics", lambda *a, **k: None)
    asyncio.run(bench.run())
    rng = random.Random(cfg.output_len_seed)
    assert [r.model_kwargs["max_tokens"] for r in sent] == [rng.randint(1, 1000) for _ in range(8)]
