"""ASR: ``/v1/audio/transcriptions`` accuracy (WER/CER) and speed.

Per request: end-to-end latency, audio seconds and RTF; with ``--stream``,
time to the first transcript delta too. Per repeat: corpus WER (CER for
character languages) against the manifest's reference text, latency
percentiles, RTFx (audio seconds transcribed per wall second) and requests/s.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.eval import stats
from benchmark.eval.datasets import Item
from benchmark.eval.suites.base import Suite, error_row, exception_row, parse_extra
from benchmark.eval.text_metrics import NORMALIZERS, check_languages, error_rate, get_normalizer


def audio_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        try:
            import soundfile as sf

            return float(sf.info(path).duration)
        except Exception:  # noqa: BLE001 - duration is informational
            return None


class ASRSuite(Suite):
    name = "asr"
    help = "speech to text over /v1/audio/transcriptions"
    requires = ("audio_path", "text")

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("asr")
        g.add_argument("--send-language", action="store_true", help="send each item's lang as 'language'")
        g.add_argument("--stream", action="store_true", help="stream deltas and record time to first text")
        g.add_argument("--normalizer", default="whisper", choices=NORMALIZERS,
                       help="text normalizer for WER, per item language for 'whisper' (see text_metrics)")
        g.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE", help="extra form field")
        g.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.extra = parse_extra(args.extra)
        self.normalizer_name, self.normalize = get_normalizer(args.normalizer)
        self._durations: dict[str, float | None] = {}

    def check_items(self, items: list[Item]) -> None:
        super().check_items(items)
        check_languages(self.normalize, [it.lang for it in items])

    def config(self) -> dict[str, Any]:
        a = self.args
        return {
            "send_language": a.send_language,
            "stream": a.stream,
            "normalizer": self.normalizer_name,
            "error_unit": "auto (CER for zh/ja/ko/th, WER otherwise)",
            "extra": self.extra,
        }

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        a = self.args
        form = aiohttp.FormData()
        form.add_field("file", Path(item.audio_path).read_bytes(), filename=Path(item.audio_path).name,
                       content_type="application/octet-stream")
        form.add_field("model", a.model_name)
        form.add_field("response_format", "json")
        form.add_field("temperature", "0")
        if a.send_language and item.lang:
            form.add_field("language", item.lang)
        if a.stream:
            form.add_field("stream", "true")
        for k, v in self.extra.items():
            form.add_field(k, v if isinstance(v, str) else json.dumps(v))
        first = None
        try:
            timeout = aiohttp.ClientTimeout(total=a.timeout)
            async with session.post(f"{url}/v1/audio/transcriptions", data=form, timeout=timeout) as resp:
                if resp.status != 200:
                    return error_row(item, f"HTTP {resp.status}", await resp.text())
                if a.stream:
                    text = None
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        ev = json.loads(line[5:])
                        kind = ev.get("type")
                        if kind == "error" or "error" in ev:
                            return error_row(item, "stream error", line)
                        if kind == "transcript.text.delta":
                            first = first or time.perf_counter() - t0
                        elif kind == "transcript.text.done":
                            text = ev.get("text", "")
                    if text is None:
                        # Deltas alone are not a transcript: the server never said it finished.
                        return error_row(item, "incomplete stream", "no transcript.text.done event")
                else:
                    text = (await resp.json()).get("text", "")
        except Exception as exc:  # noqa: BLE001 - every failure is an error row
            return exception_row(item, exc)
        e2e = time.perf_counter() - t0
        if item.id not in self._durations:
            self._durations[item.id] = audio_seconds(item.audio_path)
        dur = self._durations[item.id]
        return {
            "id": item.id, "ok": True, "error": None, "e2e_s": e2e, "first_text_s": first,
            "audio_s": dur, "rtf": (e2e / dur) if dur else None, "hyp": text.strip(),
        }

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        by_id = {it.id: it for it in items}
        ok = [r for r in rows if r["ok"]]
        # A failed request scores as an empty transcript: errors are never silently dropped from WER.
        refs = [by_id[r["id"]].text for r in rows]
        hyps = [r.get("hyp", "") if r["ok"] else "" for r in rows]
        langs = [by_id[r["id"]].lang for r in rows]
        rate, per = error_rate(refs, hyps, self.normalize, langs)
        for r, counts in zip(rows, per, strict=True):
            r["wer"] = counts.rate
        audio_total = sum(r["audio_s"] or 0.0 for r in ok)
        return {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            "wer": rate,
            "e2e_s_p50": stats.percentile([r["e2e_s"] for r in ok], 50),
            "e2e_s_p90": stats.percentile([r["e2e_s"] for r in ok], 90),
            "first_text_s_p50": stats.percentile([r["first_text_s"] for r in ok], 50),
            "rtf_mean": stats.mean(r["rtf"] for r in ok),
            "rtfx": audio_total / wall_s if wall_s > 0 else None,
            "req_per_s": len(ok) / wall_s if wall_s > 0 else None,
        }
