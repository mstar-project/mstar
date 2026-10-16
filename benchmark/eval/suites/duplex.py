"""Full duplex: turn-taking and barge-in timing of a frame-synchronous
speech-to-speech model (Nemotron-Duplex) over streamed ``/generate``.

The server consumes the user clip frame by frame and emits one output frame
(text token + audio) per input frame, so the output audio is time-aligned
with the input: output sample i plays at input time i / rate. That lets
Full-Duplex-Bench style latencies be measured in interaction time from an
offline replay, independent of how fast the server runs:

* ``turn_taking`` items (user speaks, then silence): response latency from the
  end of the user's speech to the agent's first speech onset (energy VAD on
  the output audio), the same for the first text token, and whether the agent
  took the turn within ``--response-window-s``;
* ``interruption`` items (user speaks, pauses, then talks again): if the agent
  is speaking when the user barges in, stop latency from the barge-in onset
  to the end of that agent segment, and whether it yielded within
  ``--yield-window-s``.

Wall-clock numbers say whether the server keeps up live: per-frame wall time
and the fraction of sessions that finished within frames x frame duration.
Item fields: ``audio_path`` (16 kHz mono user clip), ``extra.category``, and
optionally ``extra.speech_end_s`` / ``extra.interrupt_s`` (else found by VAD
on the input).
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from benchmark.eval import stats
from benchmark.eval.audio import pcm16_to_float, read_wav, speech_segments, write_wav
from benchmark.eval.datasets import Item
from benchmark.eval.suites.base import Suite, error_row, exception_row

CATEGORIES = ("turn_taking", "interruption")


def _special(token: str) -> bool:
    t = token.strip()
    return not t or (t.startswith("<") and t.endswith(">"))


def user_events(item: Item, user_db: float) -> tuple[float | None, float | None]:
    """(end of the user's first turn, barge-in onset or None) for an item."""
    speech_end, interrupt = item.extra.get("speech_end_s"), item.extra.get("interrupt_s")
    if speech_end is None or (item.extra.get("category") == "interruption" and interrupt is None):
        x, rate = read_wav(item.audio_path)
        segs = speech_segments(x, rate, threshold_db=user_db)
        if not segs:
            return None, None
        if item.extra.get("category") == "interruption":
            speech_end = speech_end if speech_end is not None else segs[0][1]
            if interrupt is None:
                interrupt = next((s for s, _ in segs if s > speech_end), None)
        else:
            speech_end = segs[-1][1]
    return speech_end, interrupt


# A segment ending this close to the end of the replay may have been cut by the end, not by the agent.
EOF_GUARD_S = 0.3


def timing_metrics(
    category: str,
    agent_segs: list[tuple[float, float]],
    first_text_s: float | None,
    speech_end: float | None,
    interrupt: float | None,
    response_window_s: float,
    yield_window_s: float,
    duration_s: float,
) -> dict[str, Any]:
    """Per-item interaction-time metrics from the agent's speech segments.

    Outcomes the replay was too short to observe are ``None`` with a ``*_censored``
    flag rather than counted either way: a turn not taken before the clip ended
    inside the response window, or agent speech still running at the end of the
    clip before the yield window had passed.
    """
    out: dict[str, Any] = {}
    if speech_end is None:
        return out
    turn = next(((s, e) for s, e in agent_segs if e > speech_end), None)
    took = turn is not None and turn[0] - speech_end <= response_window_s
    if not took and duration_s - speech_end < response_window_s:
        out["took_turn"], out["turn_censored"] = None, True
    else:
        out["took_turn"] = took
    # The latency of every observed reply counts, however slow; the window only decides took_turn.
    out["response_latency_s"] = (turn[0] - speech_end) if turn is not None else None
    if first_text_s is not None and first_text_s >= speech_end:
        out["text_latency_s"] = first_text_s - speech_end
    if category == "interruption" and interrupt is not None:
        speaking = next(((s, e) for s, e in agent_segs if s <= interrupt < e), None)
        out["agent_speaking_at_barge_in"] = speaking is not None
        if speaking is not None:
            if speaking[1] < duration_s - EOF_GUARD_S:
                stop = speaking[1] - interrupt
                out["stop_latency_s"] = stop
                out["yielded"] = stop <= yield_window_s
            elif duration_s - interrupt >= yield_window_s:
                out["yielded"] = False  # still talking when the clip ended, a full window after the barge-in
            else:
                out["yielded"], out["yield_censored"] = None, True
    return out


class DuplexSuite(Suite):
    name = "duplex"
    help = "full-duplex speech-to-speech timing over streamed /generate (frame-synchronous models)"
    requires = ("audio_path",)

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("duplex")
        g.add_argument("--temperature", type=float, default=0.0)
        g.add_argument("--frame-ms", type=float, default=80.0, help="model frame duration (12.5 Hz = 80 ms)")
        g.add_argument("--agent-vad-db", type=float, default=-38.0,
                       help="agent speech threshold, dBFS (above the codec's idle output)")
        g.add_argument("--user-vad-db", type=float, default=-40.0, help="user speech threshold for the input clip")
        g.add_argument("--response-window-s", type=float, default=5.0)
        g.add_argument("--yield-window-s", type=float, default=2.0)
        g.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout, seconds")

    def config(self) -> dict[str, Any]:
        a = self.args
        keys = ("temperature", "frame_ms", "agent_vad_db", "user_vad_db", "response_window_s", "yield_window_s")
        return {k: getattr(a, k) for k in keys} | {"vad": "energy"}

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        a = self.args
        form = aiohttp.FormData()
        form.add_field("files", Path(item.audio_path).read_bytes(), filename=Path(item.audio_path).name,
                       content_type="application/octet-stream")
        form.add_field("input_modalities", "audio")
        form.add_field("output_modalities", "text,audio")
        form.add_field("streaming", "true")
        form.add_field("model_kwargs", json.dumps({"temperature": a.temperature}))
        frames, first_text_frame, first_frame_wall = 0, None, None
        texts: list[str] = []
        pcm = bytearray()
        rate = 24000
        try:
            timeout = aiohttp.ClientTimeout(total=a.timeout)
            async with session.post(f"{url}/generate", data=form, timeout=timeout) as resp:
                if resp.status != 200:
                    return error_row(item, f"HTTP {resp.status}", await resp.text())
                async for raw in resp.content:
                    line = raw.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    data = base64.b64decode(msg["data"]) if msg.get("data") else b""
                    if "error" in msg or msg.get("modality") == "error":
                        detail = f"{msg.get('error', '')} {data.decode('utf-8', 'replace')}"
                        return error_row(item, "stream error", detail)
                    if msg.get("modality") == "text":
                        token = data.decode("utf-8", "replace")
                        if first_frame_wall is None:
                            first_frame_wall = time.perf_counter() - t0
                        if first_text_frame is None and not _special(token):
                            first_text_frame = frames
                        texts.append(token)
                        frames += 1
                    elif msg.get("modality") == "audio":
                        rate = int((msg.get("metadata") or {}).get("sample_rate", rate))
                        pcm.extend(data)
        except Exception as exc:  # noqa: BLE001 - every failure is an error row
            return exception_row(item, exc)
        e2e = time.perf_counter() - t0
        if frames == 0:
            return error_row(item, "no frames", e2e_s=e2e)
        frame_s = a.frame_ms / 1000
        audio = pcm16_to_float(bytes(pcm))
        category = item.extra.get("category", "turn_taking")
        speech_end, interrupt = user_events(item, a.user_vad_db)
        segs = speech_segments(audio, rate, threshold_db=a.agent_vad_db)
        first_text_s = None if first_text_frame is None else first_text_frame * frame_s
        row = {
            "id": item.id, "ok": True, "error": None, "category": category,
            "e2e_s": e2e, "frames": frames, "wall_ms_per_frame": 1000 * e2e / frames,
            "on_time": e2e <= frames * frame_s, "first_frame_wall_s": first_frame_wall,
            "speech_end_s": speech_end, "interrupt_s": interrupt,
            "agent_segments": [[round(s, 3), round(e, 3)] for s, e in segs],
            "text": "".join(t for t in texts if not _special(t)),
            "sample_rate": rate, "_audio": audio,
        }
        row |= timing_metrics(category, segs, first_text_s, speech_end, interrupt,
                              a.response_window_s, a.yield_window_s, len(audio) / rate)
        return row

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        ok = [r for r in rows if r["ok"]]
        tt = [r for r in ok if r["category"] == "turn_taking"]
        bi = [r for r in ok if r["category"] == "interruption"]
        applicable = [r for r in bi if r.get("agent_speaking_at_barge_in")]
        if audio_dir is not None:
            audio_dir.mkdir(parents=True, exist_ok=True)
            for r in ok:
                write_wav(audio_dir / f"{r['id']}.wav", r["_audio"], r["sample_rate"])
        return {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            "response_latency_s_p50": stats.percentile([r.get("response_latency_s") for r in tt], 50),
            "response_latency_s_mean": stats.mean(r.get("response_latency_s") for r in tt),
            "text_latency_s_p50": stats.percentile([r.get("text_latency_s") for r in tt], 50),
            "take_turn_rate": _rate([r.get("took_turn") for r in tt]),
            "stop_latency_s_p50": stats.percentile([r.get("stop_latency_s") for r in applicable], 50),
            "yield_rate": _rate([r.get("yielded") for r in applicable]),
            "barge_in_applicable_rate": _rate([r.get("agent_speaking_at_barge_in") for r in bi]),
            "censored": sum(bool(r.get("turn_censored") or r.get("yield_censored")) for r in ok),
            "wall_ms_per_frame_p50": stats.percentile([r["wall_ms_per_frame"] for r in ok], 50),
            "wall_ms_per_frame_max": max((r["wall_ms_per_frame"] for r in ok), default=None),
            "on_time_rate": _rate([r["on_time"] for r in ok]),
            "frames_per_s": sum(r["frames"] for r in ok) / wall_s if wall_s > 0 else None,
        }


def _rate(flags: list[Any]) -> float | None:
    vals = [bool(f) for f in flags if f is not None]
    return sum(vals) / len(vals) if vals else None


def compose_items(asr_items: list[Item], out: Path, n: int, gap_s: float = 2.5, tail_s: float = 6.0,
                  lead_s: float = 0.3) -> list[Item]:
    """Synthetic duplex clips from single-speaker utterances (e.g. LibriSpeech):
    ``n`` turn-taking clips (utterance, then ``tail_s`` of silence) and ``n``
    interruption clips (utterance, ``gap_s`` pause for the agent to start, a
    second utterance as the barge-in, then silence). Event times are the
    speech edges found by the energy VAD on each source utterance."""
    out.mkdir(parents=True, exist_ok=True)
    items: list[Item] = []
    clips = []
    for it in asr_items[: 2 * n]:
        x, rate = read_wav(it.audio_path)
        if rate != 16000:
            raise ValueError(f"{it.audio_path}: expected 16 kHz, got {rate}")
        clips.append((it, x))
    rate = 16000

    def silence(s: float) -> np.ndarray:
        return np.zeros(int(round(s * rate)), dtype=np.float32)

    def speech_bounds(x: np.ndarray) -> tuple[float, float]:
        # Recordings carry their own leading/trailing silence; event times are the speech edges.
        segs = speech_segments(x, rate)
        return (segs[0][0], segs[-1][1]) if segs else (0.0, len(x) / rate)

    for i in range(min(n, len(clips))):
        it, x = clips[i]
        path = out / f"tt_{it.id}.wav"
        write_wav(path, np.concatenate([silence(lead_s), x, silence(tail_s)]), rate)
        items.append(Item(id=f"tt_{it.id}", text=it.text, audio_path=str(path), lang=it.lang,
                          extra={"category": "turn_taking", "speech_end_s": lead_s + speech_bounds(x)[1]}))
    for i in range(min(n, len(clips) // 2)):
        (a_it, a), (b_it, b) = clips[2 * i], clips[2 * i + 1]
        path = out / f"bi_{a_it.id}.wav"
        write_wav(path, np.concatenate([silence(lead_s), a, silence(gap_s), b, silence(tail_s)]), rate)
        end_a = lead_s + speech_bounds(a)[1]
        onset_b = lead_s + len(a) / rate + gap_s + speech_bounds(b)[0]
        items.append(Item(id=f"bi_{a_it.id}", text=a_it.text, audio_path=str(path), lang=a_it.lang,
                          extra={"category": "interruption", "speech_end_s": end_a, "interrupt_s": onset_b}))
    return items
