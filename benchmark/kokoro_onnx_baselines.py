"""Closed-loop benchmark of the ONNX-runtime Kokoro engines, written like ``benchmark/runner.py`` results.

    python benchmark/kokoro_onnx_baselines.py --engine kokoro-onnx --model kokoro-v1.0.onnx --voices voices-v1.0.bin \
        --provider cuda --sentences sentences.txt --out results/<date>/onnx_kokoro --concurrency 1 8 32 --wer-wavs 60
    python benchmark/kokoro_onnx_baselines.py --engine sherpa-onnx --model-dir kokoro-multi-lang-v1_0 --provider cpu \
        --sentences sentences.txt --out results/<date>/onnx_sherpa --concurrency 1

Runs inside the baseline environment that has ``kokoro_onnx`` / ``sherpa_onnx`` installed (not M*'s venv). Neither
engine streams, so time to first audio is the whole synthesis latency. Requests run in a thread pool that shares
one session, the way these libraries are used from a server; audio seconds per wall second is the closed-loop
throughput. Writes ``<out>/c<C>_r1/results.json`` in the shape ``benchmark/tts_table.py`` reads, ``<out>/env.txt``,
and, with ``--wer-wavs N``, the first N sentences as ``<out>/../wer_<label>/wav/NNN.wav`` for
``benchmark/tts_wer.py --engine wavs``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


class KokoroOnnxEngine:
    """thewh1teagle/kokoro-onnx on onnxruntime (CUDA or CPU execution provider)."""

    name = "kokoro-onnx"

    def __init__(self, model: str, voices: str, provider: str, voice: str, threads: int):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        if provider == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model, sess_options=options, providers=providers)
        self.provider = session.get_providers()[0]
        self.kokoro = Kokoro.from_session(session, voices)
        self.voice = voice
        self.version = f"kokoro-onnx / onnxruntime {ort.__version__}"

    def __call__(self, text: str, speed: float):
        audio, rate = self.kokoro.create(text, voice=self.voice, speed=speed, lang="en-us")
        return audio, rate


class SherpaOnnxEngine:
    """k2-fsa/sherpa-onnx OfflineTts with the Kokoro multi-lang v1.0 export."""

    name = "sherpa-onnx"

    def __init__(self, model_dir: str, provider: str, sid: int, threads: int):
        import sherpa_onnx

        d = Path(model_dir)
        lexicon = ",".join(str(p) for p in sorted(d.glob("lexicon*.txt")))
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.onnx"),
                    voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data"),
                    dict_dir=str(d / "dict"),
                    lexicon=lexicon,
                ),
                num_threads=threads,
                provider=provider,
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        self.provider = provider
        self.sid = sid
        self.version = f"sherpa-onnx {sherpa_onnx.__version__}"

    def __call__(self, text: str, speed: float):
        out = self.tts.generate(text, sid=self.sid, speed=speed)
        return out.samples, out.sample_rate


def closed_loop(engine, sentences: list[str], concurrency: int, num_requests: int, warmup: int) -> dict:
    for i in range(warmup):
        engine(sentences[i % len(sentences)], 1.0)
    latencies, audio_seconds, failures = [], [], []

    def one(i: int):
        text = sentences[i % len(sentences)]
        t0 = time.perf_counter()
        try:
            audio, rate = engine(text, 1.0)
        except Exception as exc:  # noqa: BLE001 - a failed request is a data point
            return None, None, f"{type(exc).__name__}: {exc}"[:120]
        return time.perf_counter() - t0, len(audio) / rate, None

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency, seconds, error in pool.map(one, range(num_requests)):
            if error:
                failures.append(error)
            else:
                latencies.append(latency)
                audio_seconds.append(seconds)
    wall = time.perf_counter() - t_start
    rtf = [lat / sec for lat, sec in zip(latencies, audio_seconds, strict=True) if sec > 0]
    aggregate = {
        "ttft_s": {
            "audio": {
                "mean": statistics.fmean(latencies),
                "p50": _percentile(latencies, 0.5),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            }
        },
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "rtf": {"p50": _percentile(rtf, 0.5), "p95": _percentile(rtf, 0.95)},
        "audio_seconds_total": sum(audio_seconds),
        "wall_seconds": wall,
        "audio_seconds_throughput": sum(audio_seconds) / wall,
        "requests_per_second": len(latencies) / wall,
        "note": "no streaming: time to first audio is the whole synthesis",
    }
    return {
        "concurrency": concurrency,
        "completed": len(latencies),
        "failed": len(failures),
        "failures": failures[:5],
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["kokoro-onnx", "sherpa-onnx"], required=True)
    parser.add_argument("--model", help="kokoro-onnx: path to kokoro-v1.0.onnx")
    parser.add_argument("--voices", help="kokoro-onnx: path to voices-v1.0.bin")
    parser.add_argument("--model-dir", help="sherpa-onnx: the kokoro-multi-lang-v1_0 directory")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--voice", default="af_heart", help="kokoro-onnx voice name")
    parser.add_argument("--sid", type=int, default=0, help="sherpa-onnx speaker id")
    parser.add_argument("--threads", type=int, default=4, help="intra-op threads of the ONNX session")
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--wer-wavs", type=int, default=0, help="also render the first N sentences to ../wer_<label>/wav"
    )
    args = parser.parse_args()

    sentences = [s.strip() for s in Path(args.sentences).read_text(encoding="utf-8").splitlines() if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.engine == "kokoro-onnx":
        engine = KokoroOnnxEngine(args.model, args.voices, args.provider, args.voice, args.threads)
    else:
        engine = SherpaOnnxEngine(args.model_dir, args.provider, args.sid, args.threads)
    env = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\nengine={engine.name} version={engine.version} provider={engine.provider}\npython={sys.version.split()[0]}\n"
    (out / "env.txt").write_text(env)
    print(env.strip(), flush=True)

    for concurrency in args.concurrency:
        result = closed_loop(engine, sentences, concurrency, args.num_requests, args.warmup)
        run_dir = out / f"c{concurrency}_r1"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(result, indent=2))
        agg = result["aggregate"]
        print(
            f"c={concurrency}: {result['completed']}/{args.num_requests} ok, latency p50={agg['ttft_s']['audio']['p50']:.3f}s "
            f"p95={agg['ttft_s']['audio']['p95']:.3f}s, RTF p50={agg['rtf']['p50']:.4f}, {agg['audio_seconds_throughput']:.1f} audio-s/s",
            flush=True,
        )

    if args.wer_wavs:
        import soundfile as sf

        wav_dir = out.parent / f"wer_{out.name}" / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(sentences[: args.wer_wavs]):
            audio, rate = engine(text, 1.0)
            sf.write(wav_dir / f"{i + 1:03d}.wav", audio, rate)
        print(f"wrote {min(args.wer_wavs, len(sentences))} wavs to {wav_dir}", flush=True)


if __name__ == "__main__":
    main()
