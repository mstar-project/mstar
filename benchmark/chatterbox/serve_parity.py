#!/usr/bin/env python3
"""Compare a served M* Chatterbox synthesis with a greedy reference run.

Posts the same text to ``/v1/audio/speech`` with sampling off and the seed the
reference was run with (``reference_greedy.py --seed``), then reports length,
alignment and SNR against ``reference.wav``. With the server in offline S3Gen
mode (``model_kwargs: {stream_chunk_tokens: 0, t3_dtype: float32}``) and
identical speech tokens the two waveforms should match to float noise; in the
default streaming bf16 configuration expect the same length within one token
(960 samples) and a lower but positive SNR (the streamed mel is re-estimated
with growing context)::

    python benchmark/chatterbox/serve_parity.py --url http://127.0.0.1:8000 \
        --text "..." --seed 1234 --reference /tmp/ref/reference.wav --out /tmp/ref/mstar.wav
"""

from __future__ import annotations

import argparse
import io
import json
import time

import numpy as np
import requests
import soundfile as sf


def synthesize(url: str, text: str, seed: int, voice: str, extra: dict) -> tuple[np.ndarray, int, float]:
    payload = {
        "model": "chatterbox", "input": text, "voice": voice, "response_format": "wav",
        "stream": False, "seed": seed, "do_sample": False, **extra,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/v1/audio/speech", json=payload, timeout=600)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
    return wav[:, 0], sr, elapsed


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    noise = float(np.sum((a - b) ** 2))
    return float("inf") if noise == 0 else float(10 * np.log10(float(np.sum(a ** 2)) / noise))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--reference", required=True, help="reference.wav from reference_greedy.py")
    parser.add_argument("--extra", default="{}", help='JSON of extra request fields, e.g. {"cfg_weight": 0.5}')
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ours, sr, elapsed = synthesize(args.url, args.text, args.seed, args.voice, json.loads(args.extra))
    sf.write(args.out, ours, sr)
    ref, ref_sr = sf.read(args.reference, dtype="float32", always_2d=True)
    ref = ref[:, 0]
    assert ref_sr == sr, (ref_sr, sr)

    report = {
        "elapsed_s": elapsed, "sample_rate": sr,
        "ours_samples": int(len(ours)), "reference_samples": int(len(ref)),
        "length_diff_tokens": (len(ours) - len(ref)) / 960.0,
        "max_abs_diff": float(np.max(np.abs(ours[: min(len(ours), len(ref))] - ref[: min(len(ours), len(ref))]))),
        "snr_db": snr_db(ours, ref),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
