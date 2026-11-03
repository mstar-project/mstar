"""benchmark/request.py and runner.py: what the client sends and how a run's figures are counted (CPU)."""

import pytest

pytest.importorskip("aiohttp")

from benchmark.base import RequestType  # noqa: E402
from benchmark.request import RequestMetrics, aggregate_metrics  # noqa: E402


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
