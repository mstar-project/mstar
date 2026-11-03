"""benchmark/request.py and runner.py: what the client sends and how a run's figures are counted (CPU)."""

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from benchmark.base import Model, RequestType  # noqa: E402
from benchmark.request import (  # noqa: E402
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
