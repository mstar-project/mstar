#!/usr/bin/env python3
"""Word error rate of synthesized speech, transcribed with whisper-large-v3-turbo.

Quality guard for the TTS benchmark (BENCHMARK_PROTOCOL.md): every engine's WAVs
from ``benchmark/tts_speech_bench.py --save-audio-dir`` are transcribed with the
same ASR model and scored against the same input sentences, after Whisper's
English text normalizer. Engines must land within one WER point of each other.

    python -m benchmark.tts_wer --audio-dir results/mstar_c8_wav \\
        --sentences $BENCH/tts/sentences_200.txt --out results/mstar_c8_wer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words (substitutions + insertions + deletions)."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1]


def load_sentences(path: str) -> dict[int, str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {i + 1: ln.strip() for i, ln in enumerate(lines) if ln.strip()}


def transcribe(audio_paths: list[Path], model_id: str, device: str, batch_size: int) -> list[str]:
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device=device,
    )
    outputs = asr(
        [str(p) for p in audio_paths],
        batch_size=batch_size,
        generate_kwargs={"language": "en", "task": "transcribe"},
        return_timestamps=False,
    )
    return [o["text"] for o in outputs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="directory of <id>.wav files")
    parser.add_argument("--sentences", required=True, help="text file, one sentence per line (id = line number)")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    sentences = load_sentences(args.sentences)
    audio_paths = sorted(Path(args.audio_dir).glob("*.wav"))
    if not audio_paths:
        sys.exit(f"no WAV files in {args.audio_dir}")
    hypotheses = transcribe(audio_paths, args.asr_model, args.device, args.batch_size)
    normalize = EnglishTextNormalizer({})

    rows = []
    total_errors = total_words = 0
    for path, hypothesis in zip(audio_paths, hypotheses, strict=True):
        sid = int(path.stem)
        reference = sentences[sid]
        ref_words = normalize(reference).split()
        hyp_words = normalize(hypothesis).split()
        errors = word_errors(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)
        rows.append({"id": sid, "reference": reference, "hypothesis": hypothesis.strip(),
                     "errors": errors, "words": len(ref_words)})
    wer = 100.0 * total_errors / max(total_words, 1)
    report = {
        "audio_dir": str(Path(args.audio_dir).resolve()),
        "asr_model": args.asr_model,
        "files": len(rows),
        "wer_percent": wer,
        "total_words": total_words,
        "total_errors": total_errors,
        "rows": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WER {wer:.2f}% over {len(rows)} files ({total_errors}/{total_words} words)")


if __name__ == "__main__":
    main()
