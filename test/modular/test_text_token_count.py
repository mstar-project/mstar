"""The harness counts generated tokens, not stream chunks: runner, sweep stats, scorer and the re-count of old runs."""

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from benchmark.amd import recount, sweep, table
from benchmark.base import ModelType, RequestType
from benchmark.correctness.families import text as T
from benchmark.correctness.report import FamilyScore
from benchmark.correctness.sets import Item
from benchmark.request import RequestMetrics, aggregate_metrics
from benchmark.runner import Benchmark, BenchmarkConfig, DatasetType, InferenceSystemType, ProfilingType

WORDS = [b" the", b" then", b" cat", b" sat", b" on", b" mat", b".", b"\n", b"\n\n", b".\n\n", b" Tort", b"illa",
         b" \xf0", b"\x9f\xa5", b"\x94", b"\xef\xbf\xbd"]


class ToyTokenizer:
    """Greedy longest match over bytes, decoding like a byte-level BPE (a cut character reads U+FFFD)."""

    eos_token_id = 9999

    def encode(self, text, add_special_tokens=False):
        data, ids = text.encode("utf-8"), []
        while data:
            word = max((w for w in WORDS if data.startswith(w)), key=len, default=None)
            ids.append(WORDS.index(word) if word else 1000 + data[0])
            data = data[len(word) if word else 1:]
        return ids

    def decode(self, ids):
        return b"".join(WORDS[i] if i < 1000 else bytes([i - 1000]) for i in ids).decode("utf-8", errors="replace")


TOK = ToyTokenizer()
THE, THEN, CAT, SAT, ON, MAT, DOT, NL, NLNL, DOTNLNL, TORT, ILLA, EMOJI1, EMOJI2, EMOJI3 = range(15)


def _ref(ids, near=(), n=None, stopped=False):
    """A reference whose top-k at each step is its own token, plus ``near`` = {step: token} one 0.1 nat behind."""
    topk = [[[t, -0.1]] + ([[dict(near)[i], -0.2]] if i in dict(near) else []) for i, t in enumerate(ids)]
    return {"ids": ids, "topk": topk, "stopped": stopped, "max_new_tokens": n or len(ids), "eos_id": None}


def _text(ids):
    return TOK.decode(ids)


def _stream(chunk_texts, gap=0.01, usage=None):
    m = RequestMetrics(request_id=0, type=RequestType.T2T, start_time=100.0)
    for i, c in enumerate(chunk_texts):
        data = c if isinstance(c, bytes) else c.encode()
        m.record_output_chunk("text", base64.b64encode(data).decode(), arrival_time=100.0 + i * gap)
    m.e2e_latency = len(chunk_texts) * gap
    m.record_usage(usage)
    return m


class _Model:
    def get_tokenizer(self):
        return TOK


# --- the runner ----------------------------------------------------------------------------------

def test_one_token_chunks_count_chunks_and_gaps_are_itl():
    # Re-tokenized this is shorter ("\n", "\n" reads as one token), and a chunk still holds at least one token.
    m = _stream([" the", "\n", "\n", " cat", " sat"])
    agg = aggregate_metrics([m], wall_time=1.0, model=_Model())
    assert (m.output_text_tokens, m.text_token_source) == (5, "chunks")
    assert agg.itl["text"].p50 == pytest.approx(0.01) and agg.text_chunk_gap.p50 == pytest.approx(0.01)
    assert agg.text_token_throughput == 5.0 and agg.text_chunk_throughput == 5.0


def test_multi_token_chunks_are_counted_and_have_no_itl():
    m = _stream([" the cat sat", " on the mat", " the cat", " sat on the"], gap=0.03)
    agg = aggregate_metrics([m], wall_time=2.0, model=_Model())
    assert (m.output_text_tokens, m.text_token_source) == (11, "tokenizer") and m.response_chunks["text"] == 4
    assert "text" not in agg.itl  # never a chunk's gap split over its tokens
    assert agg.text_chunk_gap.p50 == pytest.approx(0.03)
    assert agg.text_token_throughput == 5.5 and agg.text_chunk_throughput == 2.0
    assert agg.text_token_sources == {"tokenizer": 1}


def test_usage_count_wins_and_a_zero_says_nothing():
    m = _stream([" the cat", " sat on", " the mat"], usage={"completion_tokens": 7, "prompt_tokens": 3})
    aggregate_metrics([m], wall_time=1.0, model=_Model())
    assert (m.output_text_tokens, m.text_token_source, m.input_tokens) == (7, "usage", 3)
    m = _stream([" the cat", " sat on", " the mat"], usage={"completion_tokens": 0})
    aggregate_metrics([m], wall_time=1.0, model=_Model())
    assert (m.output_text_tokens, m.text_token_source) == (6, "tokenizer")


def test_a_character_cut_between_chunks_is_tokenized_whole():
    # Decoded one chunk at a time, the two halves would read as U+FFFD and re-tokenize differently.
    m = _stream([b" the cat sat \xf0", b"\x9f\xa5\x94 on the mat"])
    aggregate_metrics([m], wall_time=1.0, model=_Model())
    assert (m.output_text_tokens, m.text_token_source) == (9, "tokenizer")


def test_without_a_tokenizer_the_count_says_it_is_unchecked():
    m = _stream([" the cat", " sat on"])
    agg = aggregate_metrics([m], wall_time=1.0)
    assert (m.output_text_tokens, m.text_token_source) == (2, "unchecked")
    assert agg.text_token_sources == {"unchecked": 1}


def _sse(delta, **extra):
    return f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta}], **extra})}\n\n".encode()


def _run_runner(tmp_path, chunks, usage=None):
    """The real client and runner against a small server speaking mstar's chat stream."""
    async def chat(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for c in chunks:
            await resp.write(_sse({"content": c}))
            await asyncio.sleep(0.002)
        if usage:
            await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def main():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        server = TestServer(app)
        await server.start_server()
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\n")
        model = ModelType.QWEN3_5.inst()
        model.get_tokenizer = lambda: TOK
        cfg = BenchmarkConfig(url=str(server.make_url("")).rstrip("/"), model=model, dataset=DatasetType.TEXT,
                              num_requests=2, request_type=RequestType.T2T, local_cache_dir=str(tmp_path),
                              num_warmup=0, profiling_type=ProfilingType.CLOSED_LOOP, max_concurrency=1,
                              inference_system=InferenceSystemType.OURS_OPENAI, output_dir=str(tmp_path / "out"),
                              request_txt_file=str(prompts))
        await Benchmark(cfg).run()
        await server.close()

    asyncio.run(main())
    return json.loads((tmp_path / "out" / "results.json").read_text())


def test_runner_results_multi_token_stream(tmp_path):
    res = _run_runner(tmp_path, [" the cat sat", " on the mat", " the cat", " sat on the", " mat"])
    st = sweep.repeat_stats(res, "text", output_len=None)
    assert st["out_tokens_per_req"] == 12 and st["tokens_per_chunk"] == pytest.approx(12 / 5)
    wall = res["wall_time_s"]
    assert st["chunks_s"] == pytest.approx(10 / wall) and st["tok_s"] == pytest.approx(24 / wall)
    assert st["itl_p50_s"] is None and st["chunk_gap_p50_s"] > 0 and st["tokens_unchecked"] == 0
    per = res["per_request"][0]
    assert st["tpot_p50_s"] == pytest.approx((per["jct_ms"] / 1e3 - per["ttft_s"]["text"]) / 11, rel=0.5)


def test_runner_results_one_token_stream_and_usage(tmp_path):
    res = _run_runner(tmp_path, [" the", " cat", " sat", " on"])
    st = sweep.repeat_stats(res, "text")
    assert st["out_tokens_per_req"] == 4 and st["tokens_per_chunk"] == 1 and st["itl_p50_s"] == st["chunk_gap_p50_s"]
    res = _run_runner(tmp_path, [" the cat", " sat on"], usage={"completion_tokens": 4, "prompt_tokens": 2})
    assert res["aggregate"]["text_token_sources"] == {"usage": 2}
    assert res["per_request"][0]["output_text_tokens"] == 4 and res["per_request"][0]["response_chunks"]["text"] == 2


def test_table_names_what_each_latency_is():
    heads = [h for _, h, _, _ in table.COLUMNS["text"]]
    assert "chunks/s" in heads and "tok/chunk" in heads and "chunk gap p50 ms" in heads
    assert "TPOT p50 ms (wall/token)" in heads and "ITL p50 ms (one-token chunks only)" in heads


# --- the scorer ----------------------------------------------------------------------------------

def _compare(ref, chunks):
    return T.compare_chunks_to_reference(TOK, ref, chunks, near_tie_nats=1.0)


def test_one_token_chunks_line_up_as_before():
    ids = [THE, CAT, SAT, ON]
    c, out = _compare(_ref(ids), [_text([t]) for t in ids])
    assert c.exact and out.ids == ids
    c, out = _compare(_ref(ids, near=[(1, MAT)]), [" the", " mat", " sat", " on"])
    assert (c.first_divergence, c.near_tie, c.out_token) == (1, True, MAT) and out.ids == [THE, MAT, SAT, ON]
    # not in the top-k and no id of its own: a real divergence
    c, _ = _compare(_ref(ids), [" the", "zz", " sat", " on"])
    assert (c.first_divergence, c.near_tie) == (1, False)


def test_a_chunk_of_three_tokens_scores_as_three():
    ids = [THE, CAT, SAT, ON, THE, MAT, DOT]
    c, out = _compare(_ref(ids), [_text(ids[:3]), _text(ids[3:6]), _text(ids[6:])])
    assert c.exact and out.ids == ids and out.texts == [_text([t]) for t in ids]


def test_divergence_inside_a_chunk_lands_on_its_token():
    ids = [THE, CAT, SAT, ON, THE, MAT]
    c, out = _compare(_ref(ids, near=[(3, MAT)]), [" the cat", " sat mat the", " mat"])
    assert (c.first_divergence, c.near_tie, c.out_token) == (3, True, MAT) and out.ids[:4] == [THE, CAT, SAT, MAT]


def test_a_run_stops_at_a_tokenizer_boundary():
    # " the" starts " then" as text, and is not the token the model chose
    c, out = _compare(_ref([THE, CAT, SAT], near=[(0, THEN)]), [" then cat sat"])
    assert (c.first_divergence, c.near_tie, c.out_token) == (0, True, THEN) and out.ids == [THEN, CAT, SAT]


def test_a_run_may_split_what_the_tokenizer_would_merge():
    ids = [THE, NL, NL, CAT, DOT, NLNL]
    c, out = _compare(_ref(ids), [" the\n\n", " cat.\n\n"])
    assert c.exact and out.ids == ids


def test_a_chunk_that_ends_inside_a_character():
    ids = [THE, EMOJI1, EMOJI2, EMOJI3, TORT, ILLA]
    assert _text([EMOJI1]) == _text([EMOJI1, EMOJI2]) == " �"
    # all in one chunk, the character whole
    c, _ = _compare(_ref(ids), [_text(ids)])
    assert c.exact
    # cut after the first byte token, and after the second: the text alone doesn't say which
    for cut in (2, 3):
        chunks = [_text(ids[:cut]), _text(ids[cut:])]
        c, out = _compare(_ref(ids), chunks)
        assert c.exact and out.ids == ids
    # one token per chunk
    c, out = _compare(_ref(ids), [_text([t]) for t in ids])
    assert c.exact and out.ids == ids


def test_a_short_answer_stops_on_tokens_not_chunks():
    ids = [THE, CAT, SAT, ON]
    c, _ = _compare(_ref(ids, n=8, stopped=True), [" the cat", " sat on"])
    assert c.exact
    # the output ends early by itself; four tokens in two chunks is not a stop at two
    c, _ = _compare(_ref(ids + [MAT, DOT], n=6), [" the cat", " sat on"])
    assert (c.first_divergence, c.out_token) == (4, TOK.eos_token_id)


def test_rows_and_noise_floor_are_in_tokens():
    ids = [THE, CAT, SAT, ON, THE, MAT]
    items = [Item(id="a", prompt="p")]
    sc = FamilyScore("text")
    first, second = [" the cat", " sat on the", " mat"], [" the", " cat sat", " on the mat"]
    rows = T.score_greedy_text(sc, TOK, items, {"a": _ref(ids)}, [{"a": first}, {"a": second}], 1.0)
    assert rows[0]["out_tokens"] == 6 and rows[0]["out_chunks"] == 3 and rows[0]["exact"]
    assert rows[0]["r1_r2_first_div"] is None
    sc = FamilyScore("text")
    ref = _ref(ids, near=[(4, MAT)])
    rows = T.score_greedy_text(sc, TOK, items, {"a": ref}, [{"a": [" the cat", " sat on mat"]}], 1.0)
    assert rows[0]["first_div"] == 4 and rows[0]["out_at_div"] == " the cat sat on mat"
    # no reference: still counted in tokens
    rows = T.score_greedy_text(FamilyScore("text"), TOK, items, {}, [{"a": first}], 1.0)
    assert rows[0]["out_tokens"] == 6 and rows[0]["out_chunks"] == 3


# --- the re-count of a saved run -----------------------------------------------------------------

def _saved(tmp_path, texts, chunks, tokens=None):
    per = []
    for i, (t, c) in enumerate(zip(texts, chunks, strict=True)):
        (tmp_path / f"req_{i}.txt").write_text(t)
        per.append({"request_id": i, "jct_ms": 1000.0, "ttft_s": {"text": 0.1}, "response_chunks": {"text": c},
                    "output_text_tokens": tokens or c})
    total = sum(p["output_text_tokens"] for p in per)
    agg = {"ttft_s": {"text": {"p50": 0.1, "p95": 0.1}}, "itl_s": {"text": {"p50": 0.01, "p95": 0.02}},
           "total_text_tokens": total, "text_token_throughput": total / 2.0}
    return {"completed": len(per), "failed": 0, "wall_time_s": 2.0, "request_throughput": len(per) / 2.0,
            "aggregate": agg, "per_request": per}


def test_recount_of_a_multi_token_stream(tmp_path):
    res = _saved(tmp_path, [" the cat sat on the mat"] * 2, [2, 2])
    st = sweep.repeat_stats(recount.recount_results(res, tmp_path, TOK), "text", output_len=5)
    assert st["out_tokens_per_req"] == 6 and st["tokens_per_chunk"] == 3 and st["chunks_s"] == 2.0
    assert st["tok_s"] == 5.0 and st["itl_p50_s"] is None
    assert st["tpot_p50_s"] == pytest.approx(0.9 / 5)


def test_recount_leaves_a_one_token_stream_alone(tmp_path):
    # the tokenizer would read 3 tokens where 4 were sent; the chunk count stands
    res = _saved(tmp_path, [" the\n\n cat"] * 2, [4, 4])
    new = recount.recount_results(res, tmp_path, TOK)
    assert new["per_request"] == res["per_request"]
    st = sweep.repeat_stats(new, "text")
    assert st == {**sweep.repeat_stats(res, "text"), "chunk_gap_p50_s": 0.01}


def test_recount_keeps_a_usage_count(tmp_path):
    res = _saved(tmp_path, [" the cat sat on the mat"] * 2, [2, 2], tokens=30)
    new = recount.recount_results(res, tmp_path, TOK)
    assert [p["output_text_tokens"] for p in new["per_request"]] == [30, 30]
