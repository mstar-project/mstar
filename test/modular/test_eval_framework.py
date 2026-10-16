"""The eval framework (benchmark/eval) on CPU: statistics, WER, audio helpers,
the results schema, load generation, reports, and every suite end to end
against a stdlib fake server launched through the managed-server lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval import audio, stats, text_metrics  # noqa: E402
from benchmark.eval.datasets import Item, load_manifest, save_manifest  # noqa: E402
from benchmark.eval.load import LoadSpec, poisson_offsets, run_load, warmup  # noqa: E402
from benchmark.eval.report import load_results, render_table  # noqa: E402
from benchmark.eval.schema import SCHEMA, RepeatResult, RunResult, redact_url, validate  # noqa: E402


def test_percentile_summary_and_formatting():
    assert stats.percentile([], 50) is None
    assert stats.percentile([3, 1, 2, None, float("nan")], 50) == 2
    assert stats.percentile([0, 10], 90) == pytest.approx(9.0)
    s = stats.summarize([1.0, 2.0, 3.0])
    assert s == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0, "n": 3}
    assert stats.summarize([5.0])["std"] == 0.0
    assert stats.fmt_mean_std(s, 1) == "2.0 +- 1.0"
    assert stats.fmt_mean_std(stats.summarize([0.05]), 1, 100) == "5.0"
    assert stats.fmt_mean_std(stats.summarize([]), 1) == "n/a"


def test_label_agreement_matches_cohens_kappa():
    a = ["yes"] * 20 + ["no"] * 30
    b = ["yes"] * 15 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 20
    r = stats.label_agreement(a, b)
    # p_o = 35/50, p_e = 0.4*0.5 + 0.6*0.5 = 0.5, kappa = 0.4
    assert r["agreement"] == pytest.approx(0.7) and r["kappa"] == pytest.approx(0.4)
    assert stats.label_agreement(["x"], ["x"])["kappa"] == 1.0


def test_whisper_normalizers_keep_parenthesized_speech():
    pytest.importorskip("transformers")
    name, norm = text_metrics.get_normalizer("whisper_basic")
    assert name == "whisper_basic"
    assert norm("Ions (since atoms [mostly] lose one)").split() == "ions since atoms mostly lose one".split()


def test_rescore_recomputes_stored_transcripts(tmp_path):
    items = [Item("a", text="Hello (big) world", lang="en"), Item("b", text="good day", lang="en")]
    save_manifest(items, tmp_path / "m.jsonl")
    rows = [{"id": "a", "ok": True, "hyp": "hello, big world!"}, {"id": "b", "ok": False, "error": "HTTP 500"}]
    r = RunResult("asr", "x", {"suite": {"normalizer": "whisper_en"}}, {},
                  [RepeatResult(0, 1.0, {"wer": 0.9, "errors": 1}, rows)]).finalize()
    r.save(tmp_path / "in" / "asr-x-c1.json")
    from benchmark.eval.cli import main

    main(["rescore", str(tmp_path / "in"), "--data", str(tmp_path / "m.jsonl"), "--normalizer", "basic",
          "--out", str(tmp_path / "out")])
    back = RunResult.load(tmp_path / "out" / "asr-x-c1.json")
    assert back.summary["wer"]["mean"] == pytest.approx(2 / 5)  # the failed request still counts: 2 deletions
    assert back.config["suite"]["normalizer"] == "basic"
    assert back.config["rescored"] == [{"from": "whisper_en", "to": "basic"}]


def test_edit_counts_and_corpus_wer():
    c = text_metrics.edit_counts("a b c d".split(), "a c d e".split())
    assert (c.substitutions, c.deletions, c.insertions, c.reference_len) == (0, 1, 1, 4)
    c = text_metrics.edit_counts("the cat sat on the mat".split(), "the cat sit on mat today".split())
    assert c.errors == 3 and c.reference_len == 6
    rate, per = text_metrics.error_rate(["a b c d", "e f"], ["a x c d", "e f g"])
    assert rate == pytest.approx(2 / 6)  # corpus rate, not the mean of 0.25 and 0.5
    assert [p.rate for p in per] == [0.25, 0.5]
    assert text_metrics.edit_counts([], []).rate is None
    assert text_metrics.edit_counts([], ["x"]).rate is None  # per utterance; the corpus rate counts it
    rate, per = text_metrics.error_rate(["uh", "hello world"], ["yes", "hello world"],
                                        normalize=lambda t: "" if t == "uh" else t)
    assert rate == pytest.approx(0.5) and [p.rate for p in per] == [None, 0.0]


def test_cer_for_character_languages_and_normalizers():
    rate, _ = text_metrics.error_rate(["今天天气很好。"], ["今天天汽很好"], langs=["zh"])
    assert rate == pytest.approx(1 / 6)
    _, basic = text_metrics.get_normalizer("basic")
    assert basic("Hello, World!  It's  FINE.") == "hello world it s fine"
    _, seed = text_metrics.get_normalizer("seedtts")
    assert seed("Hello, World! It's FINE.") == "hello world it's fine"
    with pytest.raises(ValueError):
        text_metrics.get_normalizer("nope")


def _wav_header(rate: int, size: int = 0xFFFFFFFF, extra_chunk: bytes = b"") -> bytes:
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", size) + b"WAVE" + fmt + extra_chunk + b"data" + struct.pack("<I", size)


def test_parse_streaming_wav_header():
    head = _wav_header(24000)
    fmt = audio.parse_wav_header(head + b"\x00\x01")
    assert (fmt.sample_rate, fmt.channels, fmt.bits_per_sample, fmt.data_offset) == (24000, 1, 16, 44)
    assert audio.parse_wav_header(head[:30]) is None
    assert audio.parse_wav_header(b"\x00" * 64) is None
    odd = b"LIST" + struct.pack("<I", 3) + b"abc\x00"  # odd chunk sizes are padded to even
    assert audio.parse_wav_header(_wav_header(16000, extra_chunk=odd)).data_offset == 56


def test_pcm_roundtrip_and_wav_io(tmp_path):
    x = (0.5 * np.sin(np.linspace(0, 20, 1600))).astype(np.float32)
    assert np.allclose(audio.pcm16_to_float(audio.float_to_pcm16(x)), x, atol=1e-4)
    audio.write_wav(tmp_path / "a.wav", x, 16000)
    y, rate = audio.read_wav(tmp_path / "a.wav")
    assert rate == 16000 and len(y) == 1600
    assert len(audio.resample(y, 16000, 8000)) == 800


def test_playback_stalls():
    assert audio.playback_stalls([]) == audio.StallStats(0, 0.0, 0.0)
    smooth = [(0.0, 0.5), (0.4, 0.5), (0.9, 0.5)]
    assert audio.playback_stalls(smooth).count == 0
    st = audio.playback_stalls([(0.0, 0.5), (0.8, 0.5), (1.2, 0.5)])
    assert st.count == 1 and st.total_s == pytest.approx(0.3)
    assert audio.playback_stalls([(0.0, 0.5), (0.8, 0.5)], prebuffer_s=0.3).count == 0
    assert audio.playback_stalls([(0.0, 0.5), (0.51, 0.5)], min_stall_s=0.02).count == 0


def test_energy_vad_segments():
    rate = 16000
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
    x = np.concatenate([np.zeros(rate // 2), tone, np.zeros(rate // 10), tone, np.zeros(rate)]).astype(np.float32)
    assert audio.speech_segments(x, rate) == [pytest.approx((0.5, 2.6), abs=0.03)]
    split = audio.speech_segments(x, rate, min_silence_s=0.05)
    assert len(split) == 2 and split[1][0] == pytest.approx(1.6, abs=0.03)


def _result(label, conc, values, suite="tts"):
    reps = [RepeatResult(i, 1.0, {"requests": 4, "ttfa_ms_p50": v, "wer": 0.01 * (i + 1)}) for i, v in
            enumerate(values)]
    meta = {"git": {"commit": "abcdef1234567", "dirty": False}, "machine": {"gpus": [{"name": "NVIDIA B200"}]}}
    return RunResult(suite, label, {"load": {"mode": "closed", "concurrency": conc}, "repeats": len(values)},
                     meta, reps).finalize()


def test_results_schema_roundtrip_and_validation(tmp_path):
    r = _result("kokoro", 4, [100.0, 110.0, 120.0])
    assert r.summary["ttfa_ms_p50"]["mean"] == pytest.approx(110.0)
    assert r.summary["ttfa_ms_p50"]["std"] == pytest.approx(10.0)
    path = r.save(tmp_path / "r.json")
    d = json.loads(path.read_text())
    assert d["schema"] == SCHEMA and validate(d) == []
    back = RunResult.load(path)
    assert back.summary == r.summary and back.repeats[1].metrics == r.repeats[1].metrics
    assert validate({"schema": "x"})[0].startswith("schema is")
    with pytest.raises(ValueError):
        RunResult.from_dict({"schema": SCHEMA})


def test_results_always_save_with_non_finite_numbers(tmp_path):
    rows = [{"id": "a", "ok": True, "wer": float("inf"), "utmos": float("nan")}]
    r = RunResult("tts", "x", {}, {}, [RepeatResult(0, 1.0, {"wer": float("nan"), "errors": 0}, rows)]).finalize()
    back = RunResult.load(r.save(tmp_path / "r.json"))
    assert back.repeats[0].requests[0]["wer"] is None and back.repeats[0].metrics["wer"] is None


def test_redact_url_keeps_only_loopback():
    assert redact_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert redact_url("http://10.1.2.3:9000/x?k=1") == "http://remote-host:9000/x"


def test_render_table_mean_std_and_provenance(tmp_path):
    for name, r in {"a": _result("kokoro", 1, [100.0, 102.0]), "b": _result("kokoro", 8, [300.0, 340.0]),
                    "c": _result("qwen3_tts", 1, [90.0, 90.0])}.items():
        r.save(tmp_path / f"{name}.json")
    table = render_table(load_results([tmp_path]))
    lines = table.splitlines()
    assert lines[0] == "**tts**: commit abcdef1234; NVIDIA B200; 2 repeats, mean +- std over repeats"
    assert lines[2] == "| label | load | n | TTFA p50 (ms) | WER % |"
    assert lines[4] == "| kokoro | c=1 | 4 | 101 +- 1 | 1.50 +- 0.71 |"
    assert lines[5].startswith("| kokoro | c=8 | 4 | 320 +- 28 |")
    assert lines[6].startswith("| qwen3_tts | c=1 |")


def test_render_chart(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmark.eval.report import render_chart

    rs = [_result("kokoro", c, [100.0 * c, 110.0 * c]) for c in (1, 2, 4)]
    path = render_chart(rs, ["ttfa_ms_p50", "wer"], tmp_path / "c.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_manifest_roundtrip_relative_paths(tmp_path):
    (tmp_path / "audio").mkdir()
    audio.write_wav(tmp_path / "audio" / "u1.wav", np.zeros(160, np.float32), 16000)
    save_manifest([Item("u1", text="hi", audio_path=str(tmp_path / "audio" / "u1.wav"), lang="en")],
                  tmp_path / "manifest.jsonl")
    assert json.loads((tmp_path / "manifest.jsonl").read_text())["audio_path"] == "audio/u1.wav"
    moved = tmp_path / "moved"
    moved.mkdir()
    (tmp_path / "audio").rename(moved / "audio")
    (tmp_path / "manifest.jsonl").rename(moved / "manifest.jsonl")
    items = load_manifest(moved / "manifest.jsonl")  # paths follow the manifest's directory
    assert items[0].audio_path == str(moved / "audio" / "u1.wav")
    (tmp_path / "s.txt").write_text("one\n\ntwo\n")
    assert [i.text for i in load_manifest(tmp_path / "s.txt")] == ["one", "two"]
    (tmp_path / "bad.jsonl").write_text('{"id": "a", "voice": "x"}\n')
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "bad.jsonl")


def test_closed_loop_caps_in_flight_and_keeps_order():
    live = {"now": 0, "max": 0}

    async def fn(i, item, t0):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.01 * (i % 3))
        live["now"] -= 1
        return i * 10

    results, wall = asyncio.run(run_load(list(range(12)), fn, LoadSpec("closed", 3)))
    assert results == [i * 10 for i in range(12)] and live["max"] == 3 and wall > 0
    calls = []

    async def count(i, item, t0):
        calls.append(item)

    asyncio.run(warmup(["a", "b"], count, LoadSpec("closed", 2), 5))
    assert calls.count("a") == 3 and calls.count("b") == 2


def test_open_loop_is_seeded_poisson():
    assert poisson_offsets(5, 2.0, 7) == poisson_offsets(5, 2.0, 7)
    assert poisson_offsets(5, 2.0, 7) != poisson_offsets(5, 2.0, 8)
    offs = poisson_offsets(2000, 50.0, 1)
    assert offs[0] == 0 and (offs[-1] / 1999) == pytest.approx(1 / 50, rel=0.1)
    starts = []

    async def fn(i, item, t0):
        starts.append((i, t0))
        return i

    results, _ = asyncio.run(run_load(list(range(6)), fn, LoadSpec("open", 0, 200.0, seed=3)))
    assert results == list(range(6)) and [i for i, _ in sorted(starts, key=lambda s: s[1])] == list(range(6))
    with pytest.raises(ValueError):
        LoadSpec("open", 0, None)


FAKE_SERVER = r'''
import io, json, math, sys, time, wave, struct
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

RATE = 24000

def wav_header(rate):
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)

def tone(seconds, amp=0.3, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * 32767 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes()

def form(handler, body):
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + handler.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
    return {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in msg.iter_parts()}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b'{"status": "healthy"}')
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/v1/audio/speech":
            req = json.loads(body)
            if req.get("voice") == "bad":
                self.send_response(400); self.end_headers(); self.wfile.write(b"bad voice"); return
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.end_headers()
            self.wfile.write(wav_header(RATE)); self.wfile.flush()
            for i in range(len(req["input"].split()) * 2):
                if "STALL" in req["input"] and i == 2:
                    time.sleep(0.3)
                if "SLOW" in req["input"] and i == 1:
                    time.sleep(30)
                time.sleep(0.005)
                self.wfile.write(tone(0.05)); self.wfile.flush()
        elif self.path == "/v1/audio/transcriptions":
            f = form(self, body)
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"text": "Hello, world."}).encode())
        elif self.path == "/generate":
            f = form(self, body)
            with wave.open(io.BytesIO(f["files"]), "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
            hop = 1280  # 80 ms at 16 kHz
            loud = [float(np.sqrt((x[i:i + hop] ** 2).mean())) > 0.01 for i in range(0, len(x) - hop + 1, hop)]
            first_end = next(i for i in range(1, len(loud)) if loud[i - 1] and not loud[i])
            barge = next((i for i in range(first_end + 1, len(loud)) if loud[i]), None)
            start, stop = first_end + 6, first_end + 46
            if barge is not None and start <= barge < stop:
                stop = barge + 5
            self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
            for i in range(len(loud)):
                speaking = start <= i < stop
                tok = "hi" if i == start else "<pad>"
                pcm = tone(0.08, 0.3) if speaking else b"\x00\x00" * int(0.08 * RATE)
                for modality, data, meta in (("text", tok.encode(), {}), ("audio", pcm, {"sample_rate": RATE})):
                    import base64
                    line = {"modality": modality, "data": base64.b64encode(data).decode(), "metadata": meta}
                    self.wfile.write((json.dumps(line) + "\n").encode())
            self.wfile.flush()
        else:
            self.send_response(404); self.end_headers()

if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(str(__import__("os").getpid()))
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    return f"{sys.executable} {script} {{port}}"


def _run_cli(argv):
    from benchmark.eval.cli import main

    main(argv)


def test_managed_server_lifecycle(fake_cmd):
    from benchmark.eval.server import ManagedServer, ServerError, is_healthy

    with ManagedServer("fake", command=fake_cmd.split()) as srv:
        assert is_healthy(srv.url) and srv.info()["startup_s"] is not None
        url = srv.url
    time.sleep(0.2)
    assert not is_healthy(url)
    with pytest.raises(ServerError, match="exited with code"):
        with ManagedServer("dead", command=[sys.executable, "-c", "raise SystemExit(3)"]):
            pass


def test_sigterm_to_the_runner_stops_the_managed_server(tmp_path, fake_cmd):
    pidfile = tmp_path / "server.pid"
    (tmp_path / "s.txt").write_text("SLOW one two\n")
    runner = subprocess.Popen(
        [sys.executable, "-m", "benchmark.eval", "run", "tts", "--serve-cmd", f"{fake_cmd} {pidfile}",
         "--data", str(tmp_path / "s.txt"), "--repeats", "1", "--warmup", "0", "--asr-judge", "none",
         "--out", str(tmp_path / "out")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)  # the slow request is in flight
    runner.terminate()
    assert runner.wait(timeout=30) == 128 + 15
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("managed server survived SIGTERM to the runner")


def test_metadata_never_carries_addresses_secrets_or_home(tmp_path, monkeypatch):
    from benchmark.eval.schema import collect_metadata, scrub_argv

    home = os.path.expanduser("~")
    monkeypatch.setattr(sys, "argv", ["eval", "run", "--url", "http://10.1.2.3:8000", "--serve-env",
                                      "HF_TOKEN=hf_secret", "--extra=api_key=sk-1", "--serve-config", f"{home}/c.yaml"])
    monkeypatch.setenv("MSTAR_ZMQ_TCP_HOST", "10.9.8.7")
    meta = collect_metadata(server={"env": {"HF_TOKEN": "hf_secret"}, "url": "http://u:p@127.0.0.1:1/?k=v"})
    r = RunResult("tts", "x", {"suite": {"extra": {"authorization": "Bearer t"}}}, meta, []).finalize()
    text = json.dumps(r.to_dict())
    for leaked in ("10.1.2.3", "10.9.8.7", "hf_secret", "sk-1", "Bearer t", home, "u:p@", "k=v"):
        assert leaked not in text, leaked
    assert scrub_argv(["--serve-env", "MSTAR_X=1"]) == ["--serve-env", "MSTAR_X=1"]


def test_asr_stream_errors_are_error_rows():
    from types import SimpleNamespace

    from benchmark.eval.suites.asr import ASRSuite

    events = {
        "error_after_delta": ['{"type": "transcript.text.delta", "delta": "hello"}',
                              '{"type": "error", "error": {"message": "boom", "code": 500}}'],
        "no_done": ['{"type": "transcript.text.delta", "delta": "hello"}'],
        "ok": ['{"type": "transcript.text.delta", "delta": "hel"}',
               '{"type": "transcript.text.done", "text": "hello"}'],
    }

    class Resp:
        def __init__(self, lines):
            self.status = 200
            self.content = self._iter(lines)

        async def _iter(self, lines):
            for line in [*lines, "[DONE]"]:
                yield f"data: {line}\n".encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    class Session:
        def __init__(self, lines):
            self.lines = lines

        def post(self, *a, **k):
            return Resp(self.lines)

    args = SimpleNamespace(model_name="m", send_language=False, stream=True, normalizer="basic", extra=[],
                           timeout=5.0)
    suite = ASRSuite(args)
    item = Item("u", text="hello", audio_path=__file__)
    rows = {k: asyncio.run(suite.request(Session(v), "http://x", 0, item, time.perf_counter()))
            for k, v in events.items()}
    assert rows["error_after_delta"]["error"] == "stream error" and "boom" in rows["error_after_delta"]["_detail"]
    assert rows["no_done"]["error"] == "incomplete stream" and not rows["no_done"]["ok"]
    assert rows["ok"]["ok"] and rows["ok"]["hyp"] == "hello" and rows["ok"]["first_text_s"] is not None


def test_duplex_timing_censors_what_the_replay_did_not_observe():
    from benchmark.eval.suites.duplex import timing_metrics

    # agent still speaking at the end of a 3.4 s replay, barge-in at 3.0 s: not a yield
    m = timing_metrics("interruption", [(1.0, 3.4)], None, 0.8, 3.0, 5.0, 2.0, duration_s=3.4)
    assert m["agent_speaking_at_barge_in"] and m["yielded"] is None and m["yield_censored"]
    assert "stop_latency_s" not in m and m["took_turn"] is True
    # ... and a full yield window after the barge-in it is a failure, not censored
    m = timing_metrics("interruption", [(1.0, 6.0)], None, 0.8, 3.0, 5.0, 2.0, duration_s=6.0)
    assert m["yielded"] is False and "yield_censored" not in m
    # an observed stop well before the end
    m = timing_metrics("interruption", [(1.0, 3.4)], None, 0.8, 3.0, 5.0, 2.0, duration_s=8.0)
    assert m["stop_latency_s"] == pytest.approx(0.4) and m["yielded"] is True
    # no answer, but the clip ended inside the response window
    m = timing_metrics("turn_taking", [], None, 2.0, None, 5.0, 2.0, duration_s=4.0)
    assert m["took_turn"] is None and m["turn_censored"]
    assert timing_metrics("turn_taking", [], None, 2.0, None, 5.0, 2.0, duration_s=9.0)["took_turn"] is False
    # a reply after the take-turn window is late, but its latency is still measured
    m = timing_metrics("turn_taking", [(10.0, 11.0)], None, 2.0, None, 5.0, 2.0, duration_s=14.0)
    assert m["took_turn"] is False and m["response_latency_s"] == pytest.approx(8.0)


def test_custom_commands_hosts_and_ipv6_never_reach_results():
    from benchmark.eval.schema import describe_command, scrub, scrub_argv

    for cmd in ("env HF_TOKEN=abc vllm serve m --api-key k1 --host gpu.internal --token=t2",
                "bash -lc 'export HF_TOKEN=abc; exec vllm serve m --host gpu.internal --api-key k1'"):
        out = scrub_argv(["--serve-cmd", cmd])[1]
        assert out == describe_command(cmd) and out.startswith("<command ")
        for leaked in ("abc", "k1", "gpu.internal", "t2"):
            assert leaked not in out
    assert describe_command("env A=1 vllm serve x").startswith("<command vllm sha256:")
    env = scrub({"MSTAR_ZMQ_TCP_HOST": "gpu.internal", "MSTAR_SERVER_HOST": "2001:db8::42",
                 "MSTAR_ADDR": "127.0.0.1", "PYTHONPATH": "/home/someone/a:/Users/x/b", "note": "at 07:02:51"})
    assert env == {"MSTAR_ZMQ_TCP_HOST": "<host>", "MSTAR_SERVER_HOST": "<host>", "MSTAR_ADDR": "127.0.0.1",
                   "PYTHONPATH": "~/a:~/b", "note": "at 07:02:51"}
    assert scrub_argv(["--serve-env", "MSTAR_ZMQ_TCP_HOST=gpu.internal"])[1] == "MSTAR_ZMQ_TCP_HOST=<host>"
    nested = scrub_argv(["--extra", 'metadata={"api_key": "sk-x", "endpoint": "gpu.internal", "voice": "af"}'])[1]
    assert "sk-x" not in nested and "gpu.internal" not in nested and '"voice": "af"' in nested
    listed = scrub({"extra": {"routing": {"hosts": ["gpu-private.internal:9000", {"name": "b.internal"}]}}})
    assert "internal" not in json.dumps(listed)
    assert "internal" not in scrub_argv(["--extra", 'routing={"hosts": ["gpu-private.internal:9000"]}'])[1]
    for extra in ('hosts=["gpu-private.internal:9000"]', 'hosts={"gpu-private.internal": 9000, "b.internal": 1}'):
        assert "internal" not in scrub_argv(["--extra", extra])[1]
    assert "internal" not in json.dumps(scrub({"extra": {"hosts": {"gpu-private.internal": 9000, "b.internal": 1}}}))


def test_tts_suite_end_to_end(tmp_path, fake_cmd):
    (tmp_path / "s.txt").write_text("one two three four\nSTALL here now please\nfive six\n")
    out = tmp_path / "out"
    _run_cli(["run", "tts", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "s.txt"), "--concurrency", "1,2",
              "--repeats", "2", "--warmup", "1", "--asr-judge", "none", "--sim-judge", "none",
              "--label", "fake", "--out", str(out)])
    r = RunResult.load(out / "tts-fake-c2.json")
    assert r.config["load"]["concurrency"] == 2 and len(r.repeats) == 2
    assert r.config["suite"]["request"]["response_format"] == "wav" and r.meta["server"]["managed"]
    rows = {row["id"]: row for row in r.repeats[0].requests}
    assert all(row["ok"] for row in rows.values())
    assert rows["0001"]["audio_s"] == pytest.approx(0.4, abs=1e-3)  # 4 words x 2 chunks x 50 ms
    assert 0 < rows["0001"]["ttfa_s"] < rows["0001"]["e2e_s"]
    assert rows["0002"]["stalls"] == 1 and rows["0001"]["stalls"] == 0
    assert r.summary["stall_rate"]["mean"] == pytest.approx(1 / 3)
    assert r.summary["errors"]["mean"] == 0
    assert "_audio" not in rows["0001"]
    assert sorted(p.name for p in (out / "audio" / "fake-c2" / "r1").iterdir()) == ["0001.wav", "0002.wav",
                                                                                    "0003.wav"]
    assert not (out / "audio" / "fake-c2" / "r0").exists()


def test_open_loop_run_records_dispatch_lateness_and_reports(tmp_path, fake_cmd, capsys):
    (tmp_path / "s.txt").write_text("one two\nthree four\nfive six\nseven eight\n")
    _run_cli(["run", "tts", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "s.txt"), "--mode", "open",
              "--rate", "40", "--repeats", "1", "--warmup", "0", "--asr-judge", "none", "--label", "open",
              "--out", str(tmp_path)])
    r = RunResult.load(tmp_path / "tts-open-rate40pers.json")
    assert r.config["load"] == {"mode": "open", "concurrency": 0, "rate": 40.0, "seed": 0}
    rows = r.repeats[0].requests
    assert all(row["arrival_lateness_s"] < 0.1 and row["slot_wait_s"] < 0.05 for row in rows)
    assert r.summary["arrival_lateness_s_p99"]["n"] == 1
    _run_cli(["report", str(tmp_path / "tts-open-rate40pers.json"), "--metrics", "ttfa_ms_p50,arrival_lateness_s_p99"])
    out = capsys.readouterr().out
    assert "| label | load | n | TTFA p50 (ms) | arrival late p99 (s) |" in out and "| open | 40 req/s | 4 |" in out


KEEPALIVE_SERVER = r'''
import struct, sys
from aiohttp import web

def header(rate=24000):
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)

async def health(request):
    return web.json_response({"status": "healthy"})

async def speech(request):
    await request.json()
    return web.Response(body=header() + b"\x10\x00" * 2400, content_type="audio/wav")

app = web.Application()
app.add_routes([web.get("/health", health), web.post("/v1/audio/speech", speech)])
# Idle keep-alive connections are closed after 0.3 s, like uvicorn after 5 s.
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), keepalive_timeout=0.3, print=None)
'''


def test_repeats_survive_server_keepalive_expiry_during_scoring(tmp_path, monkeypatch):
    from benchmark.eval.suites.tts import TTSSuite

    script = tmp_path / "keepalive_server.py"
    script.write_text(KEEPALIVE_SERVER)
    score = TTSSuite.score

    def slow_score(self, *a, **k):
        time.sleep(0.8)  # judges run between repeats, longer than the server keeps idle connections
        return score(self, *a, **k)

    monkeypatch.setattr(TTSSuite, "score", slow_score)
    (tmp_path / "s.txt").write_text("a b\nc d\ne f\nf g\n")
    _run_cli(["run", "tts", "--serve-cmd", f"{sys.executable} {script} {{port}}", "--data", str(tmp_path / "s.txt"),
              "--concurrency", "4", "--repeats", "3", "--warmup", "4", "--asr-judge", "none", "--label", "ka",
              "--out", str(tmp_path)])
    r = RunResult.load(tmp_path / "tts-ka-c4.json")
    assert [rep.metrics["errors"] for rep in r.repeats] == [0, 0, 0]


def test_cli_refuses_abbreviated_flags_and_labels_never_carry_paths(tmp_path, fake_cmd):
    from benchmark.eval.cli import build_parser

    for argv in (["run", "tts", "--serve-cm", "x", "--data", "d"], ["run", "tts", "--serve-cm=x", "--data", "d"],
                 ["run", "tts", "--ur", "http://h", "--data", "d"], ["report", "r", "--metric", "wer"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)
    (tmp_path / "s.txt").write_text("one two\n")
    home = os.path.expanduser("~")
    _run_cli(["run", "tts", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "s.txt"), "--repeats", "1",
              "--warmup", "0", "--asr-judge", "none", "--model-name", f"{home}/models/kokoro", "--out", str(tmp_path)])
    r = RunResult.load(tmp_path / "tts-kokoro-c1.json")
    assert r.label == "kokoro" and home not in json.dumps(r.to_dict())


def test_tts_suite_records_http_errors(tmp_path, fake_cmd):
    (tmp_path / "s.txt").write_text("one two\n")
    _run_cli(["run", "tts", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "s.txt"), "--repeats", "1",
              "--warmup", "0", "--voice", "bad", "--asr-judge", "none", "--label", "bad", "--out", str(tmp_path)])
    r = RunResult.load(tmp_path / "tts-bad-c1.json")
    row = r.repeats[0].requests[0]
    assert not row["ok"] and row["error"] == "HTTP 400" and r.summary["errors"]["mean"] == 1
    assert "_detail" not in row  # the server's body goes to the local log only
    assert "bad voice" in (tmp_path / "errors-tts-bad-c1.log").read_text()


def test_asr_suite_end_to_end(tmp_path, fake_cmd):
    (tmp_path / "audio").mkdir()
    items = []
    for i, ref in enumerate(["hello world", "hello there world"]):
        p = tmp_path / "audio" / f"{i}.wav"
        audio.write_wav(p, np.zeros(8000, np.float32), 16000)
        items.append(Item(f"u{i}", text=ref, audio_path=str(p), lang="en"))
    save_manifest(items, tmp_path / "m.jsonl")
    _run_cli(["run", "asr", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "m.jsonl"), "--repeats", "2",
              "--warmup", "0", "--normalizer", "basic", "--label", "fake", "--out", str(tmp_path / "out")])
    r = RunResult.load(tmp_path / "out" / "asr-fake-c1.json")
    assert r.summary["wer"]["mean"] == pytest.approx(1 / 5) and r.summary["wer"]["std"] == 0
    assert not (tmp_path / "out" / "errors-asr-fake-c1.log").exists()
    assert r.repeats[0].requests[0]["audio_s"] == pytest.approx(0.5)
    assert r.summary["rtfx"]["mean"] > 0 and r.config["suite"]["normalizer"] == "basic"


def test_duplex_suite_end_to_end(tmp_path, fake_cmd):
    from benchmark.eval.suites.duplex import compose_items

    rate = 16000
    src = []
    for i in range(4):
        p = tmp_path / f"utt{i}.wav"
        burst = 0.3 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)
        audio.write_wav(p, np.concatenate([np.zeros(3200), burst, np.zeros(3200)]).astype(np.float32), rate)
        src.append(Item(f"utt{i}", text="x", audio_path=str(p), lang="en"))
    items = compose_items(src, tmp_path / "clips", n=2, gap_s=2.5)
    assert [i.extra["category"] for i in items] == ["turn_taking"] * 2 + ["interruption"] * 2
    assert items[0].extra["speech_end_s"] == pytest.approx(0.3 + 1.2, abs=0.03)
    save_manifest(items, tmp_path / "d.jsonl")
    _run_cli(["run", "duplex", "--serve-cmd", fake_cmd, "--data", str(tmp_path / "d.jsonl"), "--repeats", "1",
              "--warmup", "0", "--concurrency", "2", "--label", "fake", "--out", str(tmp_path / "out")])
    r = RunResult.load(tmp_path / "out" / "duplex-fake-c2.json")
    m = r.repeats[0].metrics
    assert m["errors"] == 0 and m["take_turn_rate"] == 1.0 and m["yield_rate"] == 1.0
    # the fake answers 6 frames (480 ms) after the 80 ms frame in which the user stopped
    assert 0.40 <= m["response_latency_s_p50"] <= 0.60
    assert 0.30 <= m["stop_latency_s_p50"] <= 0.50  # and stops 5 frames after the barge-in frame
    assert m["text_latency_s_p50"] == pytest.approx(m["response_latency_s_p50"], abs=0.1)
    assert m["barge_in_applicable_rate"] == 1.0 and m["on_time_rate"] is not None
