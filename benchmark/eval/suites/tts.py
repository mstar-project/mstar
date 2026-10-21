"""TTS: streaming ``/v1/audio/speech`` latency, throughput, stalls, and quality.

Per request: time to first audio (TTFA, first PCM byte after the WAV header),
end-to-end latency, seconds of audio, RTF (latency / audio seconds), and the
playback stalls a real-time player fed by the stream would hit. Per repeat,
optionally: round-trip WER/CER (an ASR judge transcribes the audio and is
scored against the input text), speaker similarity to the item's reference
clip (voice-clone sets such as Seed-TTS), and predicted MOS (UTMOS).

Engines differ only in how audio is streamed: M* sends one open-ended WAV,
vLLM-Omni and SGLang-Omni send raw PCM16 (``--sample-rate`` then applies).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.eval import judges, stats
from benchmark.eval.audio import parse_wav_header, pcm16_to_float, playback_stalls, read_wav, write_wav
from benchmark.eval.datasets import Item
from benchmark.eval.suites.base import Suite, data_url, error_row, exception_row, parse_extra
from benchmark.eval.text_metrics import NORMALIZERS, check_languages, error_rate, get_normalizer

ENGINES = ("mstar", "vllm-omni", "sglang-omni")


class TTSSuite(Suite):
    name = "tts"
    help = "text to speech over /v1/audio/speech (streaming)"
    requires = ("text",)

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("tts")
        g.add_argument("--engine", choices=ENGINES, default="mstar", help="how the server streams audio")
        g.add_argument("--voice", default=None)
        g.add_argument("--language", default=None, help="sent as 'language' (e.g. English for Qwen3-TTS)")
        g.add_argument("--instructions", default=None)
        g.add_argument("--seed", type=int, default=None,
                       help="per-item seed = SEED + item index (same in every repeat); default: server sampling")
        g.add_argument("--clone", action="store_true",
                       help="send each item's ref_audio (data URL) and ref_text: zero-shot voice cloning")
        g.add_argument("--sample-rate", type=int, default=24000, help="PCM rate for engines that send raw PCM")
        g.add_argument("--extra", action="append", default=[], metavar="KEY=JSON", help="extra request field")
        g.add_argument("--asr-judge", default="openai/whisper-large-v3",
                       help="HF id of the Whisper judge for round-trip WER, or 'none'")
        g.add_argument("--normalizer", default="whisper", choices=NORMALIZERS,
                       help="text normalizer for WER, per item language for 'whisper' (see text_metrics)")
        g.add_argument("--judge-language", default="en", help="language for the ASR judge when items have none")
        g.add_argument("--sim-judge", default="microsoft/wavlm-base-plus-sv",
                       help="speaker-similarity judge (used with --clone on items with ref_audio), or 'none'")
        g.add_argument("--mos", action="store_true", help="also score UTMOS22 (torch.hub)")
        g.add_argument("--judge-device", default=None, help="device for the judges (default cuda if available)")
        g.add_argument("--stall-prebuffer-ms", type=float, default=0.0,
                       help="audio the modeled player buffers before it starts")
        g.add_argument("--stall-min-ms", type=float, default=20.0, help="ignore underruns shorter than this")
        g.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.extra = parse_extra(args.extra)
        self.normalizer_name, self.normalize = get_normalizer(args.normalizer)

    def check_items(self, items: list[Item]) -> None:
        super().check_items(items)
        check_languages(self.normalize, [it.lang or self.args.judge_language for it in items])

    def config(self) -> dict[str, Any]:
        a = self.args
        return {
            "engine": a.engine,
            "request": {k: v for k, v in self._payload(Item(id="", text=""), 0).items() if k != "input"},
            "clone": a.clone,
            "seed": a.seed,
            "asr_judge": None if a.asr_judge == "none" else f"whisper:{a.asr_judge}",
            "normalizer": self.normalizer_name,
            "judge_language": a.judge_language,
            "error_unit": "auto (CER for zh/ja/ko/th, WER otherwise)",
            "sim_judge": None if a.sim_judge == "none" else f"wavlm-sv:{a.sim_judge}",
            "mos_judge": "utmos22_strong" if a.mos else None,
            "stall_prebuffer_ms": a.stall_prebuffer_ms,
            "stall_min_ms": a.stall_min_ms,
        }

    def _payload(self, item: Item, index: int) -> dict[str, Any]:
        a = self.args
        body: dict[str, Any] = {"model": a.model_name, "input": item.text, "stream": True}
        body["response_format"] = "wav" if a.engine == "mstar" else "pcm"
        if a.engine == "vllm-omni":
            body["stream_format"] = "audio"
        for key in ("voice", "language", "instructions"):
            if getattr(a, key):
                body[key] = getattr(a, key)
        if a.seed is not None:
            body["seed"] = a.seed + index
        if a.clone and item.ref_audio:
            body["ref_audio"] = data_url(item.ref_audio)
            if item.ref_text:
                body["ref_text"] = item.ref_text
        return body | self.extra

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        a = self.args
        body = self._payload(item, index)
        rate, channels = a.sample_rate, 1
        header_done = a.engine != "mstar"
        prefix = bytearray()
        pcm = bytearray()
        timeline: list[tuple[float, float]] = []  # (arrival s from t0, audio s in the chunk)
        ttfa = None
        try:
            timeout = aiohttp.ClientTimeout(total=a.timeout, sock_read=a.timeout)
            async with session.post(f"{url}/v1/audio/speech", json=body, timeout=timeout) as resp:
                if resp.status != 200:
                    return error_row(item, f"HTTP {resp.status}", await resp.text())
                async for chunk in resp.content.iter_any():
                    now = time.perf_counter() - t0
                    if not header_done:
                        prefix.extend(chunk)
                        fmt = parse_wav_header(bytes(prefix))
                        if fmt is None:
                            if len(prefix) > 4096:
                                return error_row(item, "not a WAV stream", repr(bytes(prefix[:64])))
                            continue
                        rate, channels, header_done = fmt.sample_rate, fmt.channels, True
                        chunk = bytes(prefix[fmt.data_offset :])
                    if not chunk:
                        continue
                    if ttfa is None:
                        ttfa = now
                    before = len(pcm) // (2 * channels)
                    pcm.extend(chunk)
                    timeline.append((now, (len(pcm) // (2 * channels) - before) / rate))
        except Exception as exc:  # noqa: BLE001 - every failure is an error row
            return exception_row(item, exc)
        e2e = time.perf_counter() - t0
        audio = pcm16_to_float(bytes(pcm), channels)
        audio_s = len(audio) / rate
        if audio_s <= 0:
            return error_row(item, "empty audio", e2e_s=e2e)
        st = playback_stalls(timeline, a.stall_prebuffer_ms / 1000, a.stall_min_ms / 1000)
        return {
            "id": item.id, "ok": True, "error": None,
            "ttfa_s": ttfa, "e2e_s": e2e, "audio_s": audio_s, "rtf": e2e / audio_s,
            "chunks": len(timeline), "stalls": st.count, "stall_s": st.total_s, "sample_rate": rate,
            "_audio": audio,
        }

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        a = self.args
        ok = [r for r in rows if r["ok"]]
        audio_total = sum(r["audio_s"] for r in ok)
        m: dict[str, float | None] = {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            "ttfa_ms_p50": _ms(stats.percentile([r["ttfa_s"] for r in ok], 50)),
            "ttfa_ms_p90": _ms(stats.percentile([r["ttfa_s"] for r in ok], 90)),
            "ttfa_ms_p99": _ms(stats.percentile([r["ttfa_s"] for r in ok], 99)),
            "e2e_s_p50": stats.percentile([r["e2e_s"] for r in ok], 50),
            "e2e_s_p90": stats.percentile([r["e2e_s"] for r in ok], 90),
            "rtf_mean": stats.mean(r["rtf"] for r in ok),
            "rtf_p90": stats.percentile([r["rtf"] for r in ok], 90),
            "audio_s_per_s": audio_total / wall_s if wall_s > 0 else None,
            "req_per_s": len(ok) / wall_s if wall_s > 0 else None,
            "stall_rate": (sum(r["stalls"] > 0 for r in ok) / len(ok)) if ok else None,
            "stall_s_mean": stats.mean(r["stall_s"] for r in ok),
            "audio_s_mean": stats.mean(r["audio_s"] for r in ok),
        }
        by_id = {it.id: it for it in items}
        if audio_dir is not None:
            audio_dir.mkdir(parents=True, exist_ok=True)
            for r in ok:
                write_wav(audio_dir / f"{r['id']}.wav", r["_audio"], r["sample_rate"])
        if ok and a.asr_judge != "none":
            judge = judges.asr_judge(a.asr_judge, a.judge_device)
            langs = [by_id[r["id"]].lang or a.judge_language for r in ok]
            hyps = judge.transcribe([(r["_audio"], r["sample_rate"]) for r in ok], langs)
            refs = [by_id[r["id"]].text for r in ok]
            rate, per = error_rate(refs, hyps, self.normalize, langs)
            for r, hyp, counts in zip(ok, hyps, per, strict=True):
                r["hyp"], r["wer"] = hyp, counts.rate
            m["wer"] = rate
            m["wer_utt_mean"] = stats.mean(c.rate for c in per)
        # Without --clone the reference clip is never sent, so there is no voice to compare.
        clone = [r for r in ok if by_id[r["id"]].ref_audio] if a.clone else []
        if clone and a.sim_judge != "none":
            judge = judges.sim_judge(a.sim_judge, a.judge_device)
            for r in clone:
                r["sim"] = judge.similarity((r["_audio"], r["sample_rate"]), read_wav(by_id[r["id"]].ref_audio))
            m["sim_mean"] = stats.mean(r["sim"] for r in clone)
        if ok and a.mos:
            judge = judges.mos_judge(a.judge_device)
            for r in ok:
                r["utmos"] = judge.score((r["_audio"], r["sample_rate"]))
            m["utmos_mean"] = stats.mean(r["utmos"] for r in ok)
        return m


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else 1000.0 * seconds

