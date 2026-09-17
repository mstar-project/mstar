#!/usr/bin/env python3
"""Offline benchmark of the chatterbox-vllm port (github.com/randombk/chatterbox-vllm).

chatterbox-vllm has no server: it takes a list of prompts and lets vLLM batch
the T3 decode, then runs S3Gen per prompt. This script replays the shared
sentence set at a "concurrency" of ``--batch`` prompts per ``generate`` call,
which is the most favourable way to drive it, and reports what the protocol
asks for: time to first audio, RTF, audio seconds per second and T3 tokens
per second. Run it inside the baseline env::

    baselines/chatterbox-vllm/.venv/bin/python benchmark/chatterbox/bench_chatterbox_vllm.py \
        --sentences /path/sentences_200.txt --batch 8 --out results/<date>/chatterbox_vllm_c8

Per-batch timing: ``generate_with_conds`` prints ``[T3] ... time`` and
``[S3Gen] ... time``; the same numbers are measured here around the calls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch

S3_TOKEN_RATE = 25  # T3 emits 25 speech tokens per second of audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sentences", required=True, help="one prompt per line (shared benchmark set)")
    parser.add_argument("--batch", type=int, default=1, help="prompts per generate() call (= concurrency)")
    parser.add_argument("--num", type=int, default=200, help="how many sentences to synthesise")
    parser.add_argument("--warmup", type=int, default=3, help="sentences synthesised before timing")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--audio-prompt", default=None, help="reference clip (default: built-in voice)")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=1200)
    parser.add_argument("--out", required=True, help="output directory (wavs + summary.json)")
    args = parser.parse_args()

    from chatterbox_vllm.tts import ChatterboxTTS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sentences = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()][: args.num]

    load_start = time.perf_counter()
    model = ChatterboxTTS.from_pretrained(max_batch_size=max(args.batch, 1), max_model_len=args.max_model_len)
    load_time = time.perf_counter() - load_start

    def run(prompts: list[str]):
        t0 = time.perf_counter()
        wavs = model.generate(
            prompts,
            audio_prompt_path=args.audio_prompt,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            diffusion_steps=args.diffusion_steps,
        )
        return wavs, time.perf_counter() - t0

    run(sentences[: args.warmup] or sentences[:1])
    torch.cuda.synchronize()

    per_batch = []
    for rep in range(args.repeats):
        for start in range(0, len(sentences), args.batch):
            prompts = sentences[start:start + args.batch]
            wavs, elapsed = run(prompts)
            audio_s = sum(w.shape[-1] for w in wavs) / model.sr
            per_batch.append({
                "batch": len(prompts), "elapsed_s": elapsed, "audio_s": audio_s,
                "t3_tokens": int(round(audio_s * S3_TOKEN_RATE)),
            })
            if rep == 0:
                for i, w in enumerate(wavs):
                    sf.write(str(out / f"{start + i:03d}.wav"), w.reshape(-1).cpu().numpy(), model.sr)

    total_elapsed = sum(b["elapsed_s"] for b in per_batch)
    total_audio = sum(b["audio_s"] for b in per_batch)
    # With whole-batch synthesis the first audio of a batch arrives when the
    # batch completes: TTFA = per-batch wall time (upper bound of what a
    # client-side chunker could achieve).
    ttfa = [b["elapsed_s"] for b in per_batch]
    summary = {
        "system": "chatterbox-vllm", "batch": args.batch, "num_sentences": len(sentences),
        "repeats": args.repeats, "model_load_s": load_time,
        "ttfa_p50_s": statistics.median(ttfa), "ttfa_p95_s": sorted(ttfa)[int(0.95 * (len(ttfa) - 1))],
        "rtf": total_elapsed / total_audio, "audio_seconds_per_second": total_audio / total_elapsed,
        "t3_tokens_per_second": sum(b["t3_tokens"] for b in per_batch) / total_elapsed,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CHATTERBOX_CFG_SCALE")},
        "per_batch": per_batch,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_batch"}, indent=2))


if __name__ == "__main__":
    main()
