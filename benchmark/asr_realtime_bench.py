"""Streaming ASR benchmark: partial-result latency over ``/v1/realtime``.

Plays each benchmark utterance into a Realtime transcription session at
real time (``--speed 1.0``; higher plays faster), ``--sessions`` files at a
time, and measures for every partial result the delay between the moment
the audio it covers was *sent* and the moment the partial arrived —
the protocol's "streaming partial-result latency". Also reports the final
transcript WER against the references and the end-of-utterance latency
(commit -> completed).

    python -m benchmark.asr_realtime_bench --url ws://localhost:8000 --model qwen3_asr_realtime \\
        --system "M* <sha>" --sessions 1 8 --output-json results/asr_rt_mstar.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp

from benchmark.asr_bench import Utterance, environment_record, load_set, make_normalizer, word_error_rate

FRAME_MS = 100  # audio is sent in 100 ms PCM16 frames


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))]


# WebSocket dialects. ``openai`` is the OpenAI Realtime transcription protocol
# (what M* serves): ``transcription_session.*``, ``conversation.item.input_audio_
# transcription.{delta,completed}`` plus M*'s ``mstar.transcription.partial``
# (stable hypothesis so far with the seconds of audio it covers). ``vllm`` is
# vLLM's own: ``session.created`` / ``session.update`` / a ``commit`` to open,
# ``transcription.delta`` / ``transcription.done``, and ``commit {final: true}``
# to close (docs/serving/online_serving/speech_to_text.md).
DIALECTS = ("openai", "vllm")


def _vllm_transcript(raw: str) -> str:
    """The spoken words in vLLM's realtime output: one ``language X<asr_text>text``
    line per internal chunk, a chunk repeated when it is re-decoded with more
    audio. Keeps each distinct line's text, in order."""
    lines: list[str] = []
    for line in raw.split("\n"):
        text = line.split("<asr_text>", 1)[1] if "<asr_text>" in line else line
        text = text.strip()
        if text and (not lines or lines[-1] != text):
            lines.append(text)
    return " ".join(lines)


async def run_session(
    session: aiohttp.ClientSession, base_url: str, utt: Utterance, speed: float,
    language: str | None, chunk_seconds: float | None, dialect: str = "openai", model: str = "",
) -> dict:
    """Stream one utterance; return partial latencies, final text, timings."""
    pcm = utt.audio[44:]  # WAV header off: benchmark files are 16 kHz mono PCM16
    frame_bytes = 16000 * 2 * FRAME_MS // 1000
    partial_latencies: list[float] = []
    delta_latencies: list[float] = []
    sent_up_to = 0.0          # seconds of audio sent so far
    send_times: list[tuple[float, float]] = []  # (audio_seconds_covered, wall time sent)
    final_text = ""
    commit_time = completed_time = None
    path = "/v1/realtime" if dialect == "vllm" else "/v1/realtime?intent=transcription"

    async with session.ws_connect(f"{base_url}{path}", heartbeat=30) as ws:
        created = json.loads((await ws.receive()).data)
        if dialect == "vllm":
            assert created["type"] == "session.created", created
            await ws.send_str(json.dumps({"type": "session.update", "model": model}))
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        else:
            assert created["type"] == "transcription_session.created", created
            update: dict = {"input_audio_transcription": {}}
            if language:
                update["input_audio_transcription"]["language"] = language
            if chunk_seconds:
                update["mstar"] = {"chunk_seconds": chunk_seconds}
            await ws.send_str(json.dumps({"type": "transcription_session.update", "session": update}))

        async def sender():
            nonlocal sent_up_to, commit_time
            t0 = time.perf_counter()
            for i in range(0, len(pcm), frame_bytes):
                frame = pcm[i:i + frame_bytes]
                # pace to real time (or faster)
                target = t0 + (i / (16000 * 2)) / speed
                now = time.perf_counter()
                if target > now:
                    await asyncio.sleep(target - now)
                await ws.send_str(json.dumps({"type": "input_audio_buffer.append",
                                              "audio": base64.b64encode(frame).decode("ascii")}))
                sent_up_to = (i + len(frame)) / (16000 * 2)
                send_times.append((sent_up_to, time.perf_counter()))
            commit: dict = {"type": "input_audio_buffer.commit"}
            if dialect == "vllm":
                commit["final"] = True
            await ws.send_str(json.dumps(commit))
            commit_time = time.perf_counter()

        send_task = asyncio.create_task(sender())
        deltas: list[str] = []
        while True:
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.TEXT:  # close handshake, error, ping/pong
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                continue
            event = json.loads(msg.data)
            now = time.perf_counter()
            kind = event.get("type")
            if kind == "mstar.transcription.partial":
                covered = event.get("audio_seconds", 0.0)
                # latency = now - the time the last frame of the covered audio was sent
                sent_at = next((t for secs, t in send_times if secs >= covered - 1e-6), None)
                if sent_at is not None:
                    partial_latencies.append(now - sent_at)
            elif kind in ("conversation.item.input_audio_transcription.delta", "transcription.delta"):
                deltas.append(event.get("delta", ""))
                if send_times:
                    delta_latencies.append(now - send_times[-1][1])
            elif kind == "conversation.item.input_audio_transcription.completed":
                final_text = event["transcript"]
                completed_time = now
                break
            elif kind == "transcription.done":
                # vLLM streams the model's raw lines (``language X<asr_text>...`` per
                # internal chunk, re-emitted as chunks grow): keep the transcript text
                final_text = _vllm_transcript("".join(deltas) or event.get("text") or "")
                completed_time = now
                break
            elif kind == "error":
                final_text = ""
                break
        await send_task
    return {
        "uid": utt.uid,
        "duration": utt.duration,
        "text": final_text,
        "partial_latencies": partial_latencies,
        "delta_latencies": delta_latencies,
        "final_latency": (completed_time - commit_time) if commit_time and completed_time else None,
    }


async def run(args) -> dict:
    items = load_set(Path(args.data_dir) / args.short_set)[: args.num_utterances]
    normalize = make_normalizer(args.normalizer)
    record: dict = {
        "system": args.system, "model": args.model, "url": args.url, "speed": args.speed,
        "chunk_seconds": args.chunk_seconds, "num_utterances": len(items), "env": environment_record(),
        "sessions": {},
    }
    connector = aiohttp.TCPConnector(limit=max(64, max(args.sessions) + 8))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=connector) as session:
        for n_sessions in args.sessions:
            sem = asyncio.Semaphore(n_sessions)

            async def one(utt, sem=sem):
                async with sem:
                    return await run_session(session, args.url, utt, args.speed, args.language or None,
                                             args.chunk_seconds, args.dialect, args.model)

            t0 = time.perf_counter()
            results = await asyncio.gather(*(one(u) for u in items))
            wall = time.perf_counter() - t0
            partials = [lat for r in results for lat in r["partial_latencies"]]
            deltas = [lat for r in results for lat in r["delta_latencies"]]
            finals = [r["final_latency"] for r in results if r["final_latency"] is not None]
            ok = [r for r in results if r["text"]]
            wer = word_error_rate(
                [u.reference for u in items if u.uid in {r["uid"] for r in ok}],
                [r["text"] for r in ok], normalize,
            ) if ok else None
            record["sessions"][str(n_sessions)] = {
                "partial_p50_ms": 1000 * (statistics.median(partials) if partials else float("nan")),
                "partial_p95_ms": 1000 * (_percentile(partials, 0.95) or float("nan")),
                # text deltas, measured from the most recent frame sent (the
                # only partial signal every dialect has)
                "delta_p50_ms": 1000 * (statistics.median(deltas) if deltas else float("nan")),
                "delta_p95_ms": 1000 * (_percentile(deltas, 0.95) or float("nan")),
                "num_deltas": len(deltas),
                "final_p50_ms": 1000 * (statistics.median(finals) if finals else float("nan")),
                "final_p95_ms": 1000 * (_percentile(finals, 0.95) or float("nan")),
                "num_partials": len(partials),
                "wer": wer,
                "errors": len(items) - len(ok),
                "wall_s": wall,
                "audio_s": sum(u.duration for u in items),
            }
            # the transcripts of the last session count, for a diff against the references
            record["hypotheses"] = {res["uid"]: res["text"] for res in results}
            r = record["sessions"][str(n_sessions)]
            wer_pct = 100 * wer if wer is not None else float("nan")
            print(f"[sessions={n_sessions:>3}] partial p50 {r['partial_p50_ms']:7.0f} ms  "
                  f"p95 {r['partial_p95_ms']:7.0f} ms  delta p50 {r['delta_p50_ms']:7.0f} ms  "
                  f"final p50 {r['final_p50_ms']:7.0f} ms  WER {wer_pct:5.2f}%  errors {r['errors']}", flush=True)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description="streaming ASR partial-latency benchmark over /v1/realtime")
    p.add_argument("--url", required=True, help="ws://host:port")
    p.add_argument("--model", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("MSTAR_WS_ROOT", "."), "commons/bench/data/asr"))
    p.add_argument("--short-set", default="test_clean_200")
    p.add_argument("--num-utterances", type=int, default=50)
    p.add_argument("--sessions", type=int, nargs="+", default=[1, 8])
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    p.add_argument("--language", default="en")
    p.add_argument("--chunk-seconds", type=float, default=None)
    p.add_argument("--dialect", choices=DIALECTS, default="openai", help="WebSocket protocol the server speaks")
    p.add_argument("--normalizer", choices=["whisper", "basic"], default="whisper")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()
    record = asyncio.run(run(args))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        Path(args.output_json).write_text(json.dumps(record, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
