"""Rough throughput and latency of a running Omnilingual ASR server.

For each concurrency level, keeps that many ``/v1/audio/transcriptions``
requests in flight over the manifest's clips (cycled) and reports requests/s,
audio seconds transcribed per second, the real-time factor (wall time over
audio time, aggregate) and latency percentiles. ``--offline`` also times the
encoder alone in-process at batch 1 (RTF per clip, no HTTP or scheduling)::

    python test/omnilingual_asr/bench_transcriptions.py --url http://127.0.0.1:8000 \\
        --model omnilingual_asr_ctc_300m --manifest clips/manifest.json --levels 1 4 16 64 --out bench.json
"""

import argparse
import json
import statistics
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch


def seconds_of(path: str) -> float:
    try:
        import soundfile as sf

        return sf.info(path).duration
    except ImportError:
        with wave.open(path) as w:
            return w.getnframes() / w.getframerate()


def one(url: str, model: str, path: str) -> float:
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post(
            f"{url}/v1/audio/transcriptions",
            data={"model": model},
            files={"file": (Path(path).name, f, "audio/wav")},
            timeout=600,
        )
    r.raise_for_status()
    return time.perf_counter() - t0


def level(url: str, model: str, clips: list[tuple[str, float]], concurrency: int, requests_n: int) -> dict:
    jobs = [clips[i % len(clips)] for i in range(requests_n)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        lat = list(pool.map(lambda c: one(url, model, c[0]), jobs))
    wall = time.perf_counter() - t0
    audio_s = sum(s for _, s in jobs)
    lat.sort()
    return {
        "concurrency": concurrency,
        "requests": requests_n,
        "wall_s": round(wall, 3),
        "requests_per_s": round(requests_n / wall, 2),
        "audio_s_per_s": round(audio_s / wall, 1),
        "rtf": round(wall / audio_s, 5),
        "latency_p50_ms": round(1000 * statistics.median(lat), 1),
        "latency_p90_ms": round(1000 * lat[int(0.9 * (len(lat) - 1))], 1),
    }


def offline(clips, checkpoint: str, size: str, dtype: str) -> dict:
    from mstar.model.omnilingual_asr import audio, config, weight_loader
    from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC

    dt = config.parse_dtype(dtype)
    with torch.device("meta"):
        model = Wav2Vec2CTC(config.ctc_config(size, "v2"))
    weight_loader.load_into(
        model, weight_loader.convert_state_dict(weight_loader.read_checkpoint(checkpoint), dt, "cuda")
    )
    waves = [audio.preprocess(audio.load_waveform(p)).to("cuda", dt)[None] for p, _ in clips]
    rtfs = []
    with torch.inference_mode():
        for w in waves[:3]:
            model(w)  # warm up
        for w, (_, s) in zip(waves, clips, strict=True):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(w)[0].argmax(-1)
            torch.cuda.synchronize()
            rtfs.append((time.perf_counter() - t0) / s)
    return {
        "encoder_rtf_batch1_median": round(statistics.median(rtfs), 5),
        "encoder_rtf_batch1_max": round(max(rtfs), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--requests-per-level", type=int, default=0, help="default: max(2 x level, clips)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clips = [(c["path"], seconds_of(c["path"])) for c in json.load(open(args.manifest))]
    clips = [c for c in clips if c[1] <= 40.0]
    one(args.url, args.model, clips[0][0])  # warm up
    report = {"model": args.model, "clips": len(clips), "mean_clip_s": round(statistics.mean(s for _, s in clips), 2)}
    report["levels"] = []
    for c in args.levels:
        n = args.requests_per_level or max(2 * c, len(clips))
        report["levels"].append(level(args.url, args.model, clips, c, n))
        print(json.dumps(report["levels"][-1]), flush=True)
    if args.offline:
        report["offline"] = offline(clips, args.checkpoint, args.model.rsplit("_", 1)[1], args.dtype)
    print(json.dumps(report, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
