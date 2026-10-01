"""Stress a running Kokoro server through the OpenAI speech route.

    python test/kokoro/stress_server.py --url http://127.0.0.1:8000 --sentences sentences.txt --out results/stress \
        [--scenarios soak,burst,long,edge,disconnect] [--soak-concurrency 64] [--soak-seconds 300] [--burst 128]

Every request records its status, time to first audio byte, total latency and the seconds of audio received.
Scenarios:

- ``soak``: N clients in a closed loop for a fixed time over random sentences, voices (including blends), speeds,
  streaming on or off and both audio formats. Passes when nothing fails.
- ``burst``: N requests fired at the same instant. Passes when all succeed.
- ``long``: paragraphs of 2k, 8k and 20k characters at concurrency 4. Passes when all succeed with audio.
- ``edge``: malformed and unusual inputs. Passes when the server never answers 5xx and rejects what it must
  reject with 4xx.
- ``disconnect``: streaming clients that drop the connection after the first chunk, then a normal request.

GPU memory is sampled with nvidia-smi when it is on PATH so growth across the run shows up. Writes
``<out>/summary.json`` and ``<out>/requests.jsonl`` and exits non-zero when a scenario fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import time
from pathlib import Path

import aiohttp

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
WAV_HEADER = 44


def gpu_memory_mib() -> int | None:
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out[0]) if out and out[0].strip().isdigit() else None


async def speech(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    *,
    stream: bool = True,
    fmt: str = "pcm",
    cancel_after_first_chunk: bool = False,
    timeout: float = 900.0,
    scenario: str = "",
) -> dict:
    body = {"model": "kokoro", "response_format": fmt, "stream": stream, **payload}
    rec = {
        "scenario": scenario,
        "ttfa": None,
        "latency": None,
        "bytes": 0,
        "status": None,
        "error": None,
        "cancelled": False,
        "stream": stream,
        "format": fmt,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/audio/speech", json=body, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = (await resp.text())[:200]
            else:
                async for chunk in resp.content.iter_chunked(1 << 16):
                    if rec["ttfa"] is None and chunk:
                        rec["ttfa"] = time.perf_counter() - t0
                    rec["bytes"] += len(chunk)
                    if cancel_after_first_chunk:
                        rec["cancelled"] = True
                        break
    except Exception as exc:  # noqa: BLE001 - every failure is a data point here
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    rec["latency"] = time.perf_counter() - t0
    header = WAV_HEADER if fmt == "wav" else 0
    rec["audio_s"] = max(0, rec["bytes"] - header) / BYTES_PER_SECOND
    return rec


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["status"] == 200 and r["error"] is None]
    ttfa = [r["ttfa"] for r in ok if r["ttfa"] is not None]
    return {
        "requests": len(recs),
        "ok": len(ok),
        "failed": len(recs) - len(ok),
        "status_counts": {
            str(k): sum(1 for r in recs if r["status"] == k) for k in sorted({r["status"] for r in recs}, key=str)
        },
        "ttfa_p50": percentile(ttfa, 0.5),
        "ttfa_p95": percentile(ttfa, 0.95),
        "ttfa_p99": percentile(ttfa, 0.99),
        "latency_p50": percentile([r["latency"] for r in ok], 0.5),
        "latency_p99": percentile([r["latency"] for r in ok], 0.99),
        "audio_s_total": round(sum(r["audio_s"] for r in ok), 1),
        "errors": sorted({r["error"] for r in recs if r["error"]})[:8],
    }


class Stress:
    def __init__(self, url: str, sentences: list[str], voices: list[str], out: Path, seed: int):
        self.url = url
        self.sentences = sentences
        self.voices = voices
        self.out = out
        self.rng = random.Random(seed)
        self.records: list[dict] = []
        self.results: dict[str, dict] = {}
        self.failures: list[str] = []
        base = [v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_"))] or voices
        self.blends = [f"{a}+{b}" for a, b in zip(base, base[1:] + base[:1], strict=True)][:8]
        self.blends += [f"{base[0]}(2)+{base[-1]}(1)"] if len(base) > 1 else []

    def voice(self) -> str:
        pool = self.voices + self.blends
        return self.rng.choice(pool)

    async def _run_many(self, session, jobs):
        recs = await asyncio.gather(*jobs)
        self.records.extend(recs)
        return list(recs)

    async def soak(self, session, concurrency: int, seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        per_minute: dict[int, float] = {}
        gpu_by_minute: dict[int, int | None] = {}
        t_start = time.perf_counter()
        recs: list[dict] = []

        async def worker():
            while time.perf_counter() < deadline:
                stream = self.rng.random() < 0.7
                fmt = "pcm" if self.rng.random() < 0.6 else "wav"
                payload = {
                    "input": self.rng.choice(self.sentences),
                    "voice": self.voice(),
                    "speed": round(self.rng.uniform(0.7, 1.5), 2),
                }
                rec = await speech(session, self.url, payload, stream=stream, fmt=fmt, scenario="soak")
                minute = int((time.perf_counter() - t_start) // 60)
                per_minute[minute] = per_minute.get(minute, 0.0) + (rec["audio_s"] if rec["status"] == 200 else 0.0)
                if minute not in gpu_by_minute:
                    gpu_by_minute[minute] = gpu_memory_mib()
                recs.append(rec)

        gpu_before = gpu_memory_mib()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_start
        self.records.extend(recs)
        summary = summarize(recs)
        summary.update(
            {
                "concurrency": concurrency,
                "wall_s": round(wall, 1),
                "audio_s_per_s": round(summary["audio_s_total"] / wall, 1) if wall else None,
                "audio_s_per_s_by_minute": {
                    str(m): round(v / 60, 1) for m, v in sorted(per_minute.items()) if m * 60 + 60 <= wall
                },
                "gpu_mib_before": gpu_before,
                "gpu_mib_after": gpu_memory_mib(),
                "gpu_mib_by_minute": {str(m): v for m, v in sorted(gpu_by_minute.items())},
            }
        )
        self.results["soak"] = summary
        if summary["failed"]:
            self.failures.append(f"soak: {summary['failed']} of {summary['requests']} requests failed")

    async def burst(self, session, n: int) -> None:
        jobs = [
            speech(
                session,
                self.url,
                {"input": self.sentences[i % len(self.sentences)], "voice": self.voice(), "speed": 1.0},
                stream=True,
                scenario="burst",
            )
            for i in range(n)
        ]
        t0 = time.perf_counter()
        recs = await self._run_many(session, jobs)
        summary = summarize(recs)
        summary["wall_s"] = round(time.perf_counter() - t0, 2)
        summary["n"] = n
        self.results["burst"] = summary
        if summary["failed"]:
            self.failures.append(f"burst: {summary['failed']} of {n} failed")

    async def long(self, session) -> None:
        paragraphs = []
        for target in (2_000, 8_000, 20_000):
            text, i = "", 0
            while len(text) < target:
                text += self.sentences[i % len(self.sentences)] + " "
                i += 1
            paragraphs.append(text.strip())
        jobs = [
            speech(
                session,
                self.url,
                {"input": p, "voice": self.voices[0], "speed": 1.0},
                stream=stream,
                fmt=fmt,
                scenario="long",
            )
            for p in paragraphs
            for stream, fmt in ((True, "pcm"), (False, "wav"))
        ]
        sem = asyncio.Semaphore(4)

        async def limited(job):
            async with sem:
                return await job

        recs = await self._run_many(session, [limited(j) for j in jobs])
        summary = summarize(recs)
        summary["chars"] = [len(p) for p in paragraphs]
        summary["audio_s_each"] = [round(r["audio_s"], 1) for r in recs]
        self.results["long"] = summary
        bad = [r for r in recs if r["status"] != 200 or r["audio_s"] < 10]
        if bad:
            self.failures.append(f"long: {len(bad)} paragraph requests failed or returned under 10 s of audio")

    async def edge(self, session) -> None:
        v = self.voices[0]
        cases = [
            ("empty text", {"input": "", "voice": v}, {400, 422}),
            ("whitespace only", {"input": "   \n\t ", "voice": v}, {200, 400, 422}),
            ("punctuation only", {"input": "... !!! ??? --- ***", "voice": v}, {200, 400, 422}),
            ("one 500-char word", {"input": "a" * 500, "voice": v}, {200}),
            (
                "numbers, dates, money",
                {
                    "input": "On 12/31/2025 at 9:45 a.m., $1,234.56 was wired to account no. 0042; call 555-0100.",
                    "voice": v,
                },
                {200},
            ),
            (
                "unicode and emoji",
                {"input": "Café, naïve façade — “curly quotes” … 3½ and ½, 25°C, résumé 😀 ✅", "voice": v},
                {200},
            ),
            ("cjk text, english voice", {"input": "今日は良い天気ですね。北京欢迎你。", "voice": v}, {200, 400, 422}),
            ("unknown voice", {"input": "Hello there.", "voice": "zz_nobody"}, {400, 422}),
            ("speed too high", {"input": "Hello there.", "voice": v, "speed": 9.0}, {400, 422}),
            ("speed too low", {"input": "Hello there.", "voice": v, "speed": 0.05}, {400, 422}),
            ("speed 0.25", {"input": "Hello there, this is slow.", "voice": v, "speed": 0.25}, {200}),
            ("speed 4.0", {"input": "Hello there, this is fast.", "voice": v, "speed": 4.0}, {200}),
            (
                "four-voice weighted blend",
                {
                    "input": "Four voices in one.",
                    "voice": "+".join(f"{x}({w})" for x, w in zip(self.voices[:4], (2, 1, 0.5, 1), strict=False)),
                },
                {200},
            ),
            (
                "negative weight blend",
                {"input": "Subtracting a voice.", "voice": f"{self.voices[0]}-{self.voices[-1]}(0.5)"},
                {200},
            ),
            ("blend with a bad member", {"input": "Hello.", "voice": f"{v}+zz_nobody"}, {400, 422}),
            ("lang_code override", {"input": "The colour of the harbour.", "voice": v, "lang_code": "b"}, {200}),
            ("phonemes passthrough", {"input": "ignored", "voice": v, "phonemes": "həlˈoʊ wˈɜːld"}, {200}),
            ("mandarin voice without misaki[zh]", {"input": "你好", "voice": "zf_xiaobei"}, {200, 400, 422}),
            ("very long sentence, no punctuation", {"input": " ".join(["word"] * 1500), "voice": v}, {200}),
            ("html and code", {"input": "<p>Use <b>bold</b> &amp; x = y**2 // 3 </p>", "voice": v}, {200}),
        ]
        recs = []
        for name, payload, expect in cases:
            fmt = "wav" if len(recs) % 2 else "pcm"
            # A rejected streaming request answers 200 with an empty body until the speech route looks at
            # its first result before answering (mstar PR #300), so status checks go non-streaming.
            stream = len(recs) % 3 != 0 and expect == {200}
            rec = await speech(session, self.url, payload, stream=stream, fmt=fmt, scenario=f"edge:{name}")
            rec["case"] = name
            rec["expected"] = sorted(expect)
            rec["ok"] = rec["status"] in expect and (
                rec["status"] != 200 or rec["audio_s"] > 0 or payload.get("input", "x").strip() == ""
            )
            recs.append(rec)
        rec = await speech(
            session, self.url, {"input": "Hello.", "voice": "zz_nobody"}, stream=True,
            scenario="edge:streaming rejection",
        )
        rec["case"] = "streaming request with an unknown voice (informational, see PR #300)"
        rec["expected"] = [400, 422]
        rec["ok"] = True
        recs.append(rec)
        for fmt in ("mp3", "opus", "flac", "aac"):
            rec = await speech(
                session,
                self.url,
                {"input": "Format check.", "voice": v},
                stream=False,
                fmt=fmt,
                scenario=f"edge:format {fmt}",
            )
            rec["case"] = f"format {fmt}"
            rec["expected"] = [200, 400, 415, 422]
            rec["ok"] = rec["status"] in (200, 400, 415, 422)
            recs.append(rec)
        self.records.extend(recs)
        self.results["edge"] = {
            "cases": [
                {
                    "case": r["case"],
                    "status": r["status"],
                    "audio_s": round(r["audio_s"], 2),
                    "ok": r["ok"],
                    "error": (r["error"] or "")[:80],
                }
                for r in recs
            ],
            "unexpected": [r["case"] for r in recs if not r["ok"]],
            "server_errors": [r["case"] for r in recs if r["status"] is not None and r["status"] >= 500],
        }
        if self.results["edge"]["unexpected"]:
            self.failures.append(f"edge: unexpected answers for {self.results['edge']['unexpected']}")

    async def disconnect(self, session, n: int) -> None:
        jobs = [
            speech(
                session,
                self.url,
                {
                    "input": " ".join(self.sentences[i % len(self.sentences)] for i in range(k, k + 6)),
                    "voice": self.voice(),
                },
                stream=True,
                cancel_after_first_chunk=True,
                scenario="disconnect",
            )
            for k in range(n)
        ]
        recs = await self._run_many(session, jobs)
        await asyncio.sleep(1.0)
        after = await speech(
            session,
            self.url,
            {"input": "Still here after the disconnect storm.", "voice": self.voices[0]},
            stream=True,
            scenario="disconnect:after",
        )
        self.records.append(after)
        health = None
        try:
            async with session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                health = resp.status
        except Exception as exc:  # noqa: BLE001
            health = str(exc)[:80]
        cancelled = sum(1 for r in recs if r["cancelled"])
        self.results["disconnect"] = {
            "clients": n,
            "cancelled_after_first_chunk": cancelled,
            "after_status": after["status"],
            "after_audio_s": round(after["audio_s"], 2),
            "after_ttfa": after["ttfa"],
            "health": health,
        }
        if after["status"] != 200 or after["audio_s"] <= 0 or health != 200:
            self.failures.append("disconnect: the server did not serve a normal request afterwards")


async def fetch_voices(session, url: str) -> list[str]:
    async with session.get(f"{url}/v1/audio/voices", timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return [v["id"] for v in data["voices"]]


async def main_async(args) -> int:
    sentences = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        voices = await fetch_voices(session, args.url)
        stress = Stress(args.url, sentences, voices, out, args.seed)
        gpu_start = gpu_memory_mib()
        scenarios = args.scenarios.split(",")
        t0 = time.perf_counter()
        for name in scenarios:
            t = time.perf_counter()
            if name == "soak":
                await stress.soak(session, args.soak_concurrency, args.soak_seconds)
            elif name == "burst":
                await stress.burst(session, args.burst)
            elif name == "long":
                await stress.long(session)
            elif name == "edge":
                await stress.edge(session)
            elif name == "disconnect":
                await stress.disconnect(session, args.disconnect)
            else:
                raise SystemExit(f"unknown scenario {name!r}")
            stress.results[name]["seconds"] = round(time.perf_counter() - t, 1)
            shown = {k: v for k, v in stress.results[name].items() if k != "cases"}
            print(f"[{name}] {json.dumps(shown, default=str)[:600]}", flush=True)
        gpu_end = gpu_memory_mib()
    summary = {
        "url": args.url,
        "voices": len(voices),
        "scenarios": scenarios,
        "wall_s": round(time.perf_counter() - t0, 1),
        "gpu_mib_start": gpu_start,
        "gpu_mib_end": gpu_end,
        "results": stress.results,
        "failures": stress.failures,
        "requests_total": len(stress.records),
        "requests_failed": sum(
            1 for r in stress.records if r["status"] != 200 and not r["scenario"].startswith("edge")
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with (out / "requests.jsonl").open("w") as fh:
        for rec in stress.records:
            fh.write(json.dumps(rec, default=str) + "\n")
    print(
        f"\nrequests: {summary['requests_total']}, non-edge failures: {summary['requests_failed']}, "
        f"gpu MiB {gpu_start} -> {gpu_end}, wall {summary['wall_s']} s"
    )
    print("FAILURES: " + "; ".join(stress.failures) if stress.failures else "ALL SCENARIOS PASSED")
    return 1 if stress.failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenarios", default="edge,burst,long,disconnect,soak")
    parser.add_argument("--soak-concurrency", type=int, default=64)
    parser.add_argument("--soak-seconds", type=float, default=300)
    parser.add_argument("--burst", type=int, default=128)
    parser.add_argument("--disconnect", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
