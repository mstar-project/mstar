"""Whisper transcripts of each parity stage's input, NVIDIA and M* audio, and the word error rate between agents.

An automatic aid beside listening, not a quality score: a low word error rate
says the two agents said the same words; it says nothing about how they sound.
Whisper invents text for silence or near-silence ("Thank you.", runs of dots),
so transcripts of short or quiet clips say little.
"""

import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np

LABELS = ("input", "nvidia", "mstar")


def words(text):
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_error_rate(reference, hypothesis):
    """Word-level Levenshtein distance over the reference length; None for an empty reference."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return None
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / len(ref)


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth()) != (1, 2):
            raise ValueError(f"{path} is not mono PCM16")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768, wav.getframerate()


def summarize(rows, asr_model):
    """PASS only when every requested stage was transcribed; a stage without recordings is BLOCKED."""
    missing = sorted(name for name, row in rows.items() if row.get("missing"))
    summary = {"status": "BLOCKED" if missing or not rows else "PASS", "asr_model": asr_model, "stages": rows,
               "scope": "Transcript agreement between the two agents; not an audio quality measure. "
                        "Whisper invents text for near-silence."}
    if missing or not rows:
        summary["reason"] = f"No recordings to judge for {', '.join(missing) or 'any stage'}"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-model", required=True, help="local Whisper checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("parity", type=Path, nargs="+", help="parity stage output directories")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.asr_model, device=0, chunk_length_s=30)
    rows = {}
    for directory in args.parity:
        if not all((directory / f"{label}.wav").is_file() for label in LABELS):
            rows[directory.name] = {"missing": True}
            continue
        texts = {}
        for label in LABELS:
            audio, rate = read_wav(directory / f"{label}.wav")
            result = asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "en"})
            texts[label] = result["text"].strip()
        rows[directory.name] = {"transcripts": texts, "agent_wer_mstar_vs_nvidia": word_error_rate(
            texts["nvidia"], texts["mstar"])}
    summary = summarize(rows, args.asr_model)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
