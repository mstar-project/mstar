"""Build long-form A2T inputs for E6E from LibriSpeech dev-clean.

Concatenates each chapter's utterances (in utterance order, 0.3 s silence
between) until a seeded per-file target in [60, 120] s is reached, one output
wav per chapter, first N chapters in sorted (speaker, chapter) order.
Output: 16 kHz mono int16 wavs long_###.wav + manifest.json.

Usage: python make_libri_long.py <dev-clean_root> <out_dir> [n_files] [lo_s] [hi_s] [seed]
"""

import glob
import json
import os
import random
import sys
import wave

import numpy as np
from torchcodec.decoders import AudioDecoder

SR = 16000
GAP_S = 0.3
TARGET_LO, TARGET_HI = 60.0, 120.0
SEED = 0


def main() -> None:
    global TARGET_LO, TARGET_HI, SEED
    root, out_dir = sys.argv[1], sys.argv[2]
    n_files = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    if len(sys.argv) > 5:
        TARGET_LO, TARGET_HI = float(sys.argv[4]), float(sys.argv[5])
    if len(sys.argv) > 6:
        SEED = int(sys.argv[6])
    os.makedirs(out_dir, exist_ok=True)

    chapters = []
    for spk in sorted(os.listdir(root), key=int):
        for chap in sorted(os.listdir(os.path.join(root, spk)), key=int):
            chapters.append((spk, chap))

    rng = random.Random(SEED)
    targets = [rng.uniform(TARGET_LO, TARGET_HI) for _ in range(len(chapters))]

    manifest = []
    made = 0
    for (spk, chap), target in zip(chapters, targets):
        if made >= n_files:
            break
        flacs = sorted(glob.glob(os.path.join(root, spk, chap, "*.flac")))
        gap = np.zeros(int(GAP_S * SR), dtype=np.int16)
        pieces, sources, dur = [], [], 0.0
        for f in flacs:
            frames = AudioDecoder(f).get_all_samples()
            assert frames.sample_rate == SR, f"{f}: sr={frames.sample_rate}"
            x = frames.data.numpy()
            x = x.mean(axis=0) if x.ndim == 2 else x
            pcm = (x * 32767.0).clip(-32768, 32767).astype(np.int16)
            if pieces:
                pieces.append(gap)
                dur += GAP_S
            pieces.append(pcm)
            sources.append(os.path.basename(f))
            dur += len(pcm) / SR
            if dur >= target:
                break
        if dur < TARGET_LO:
            continue  # chapter too short for the minimum length
        name = f"long_{made:03d}.wav"
        with wave.open(os.path.join(out_dir, name), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(np.concatenate(pieces).tobytes())
        manifest.append({
            "file": name,
            "speaker": spk,
            "chapter": chap,
            "n_utterances": len(sources),
            "duration_s": round(dur, 2),
            "target_s": round(target, 2),
            "sources": sources,
        })
        made += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({
            "corpus": "LibriSpeech dev-clean (openslr 12)",
            "sample_rate": SR,
            "gap_s": GAP_S,
            "target_range_s": [TARGET_LO, TARGET_HI],
            "seed": SEED,
            "files": manifest,
        }, f, indent=2)
    durs = [m["duration_s"] for m in manifest]
    print(f"wrote {made} wavs to {out_dir}: "
          f"min={min(durs):.1f}s p50={sorted(durs)[len(durs)//2]:.1f}s max={max(durs):.1f}s")


if __name__ == "__main__":
    main()
