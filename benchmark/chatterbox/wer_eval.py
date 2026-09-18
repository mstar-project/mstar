#!/usr/bin/env python3
"""Intelligibility guard for TTS benchmark outputs.

Transcribes every ``<index>.wav`` (or ``req_<index>.wav``) in a directory with Whisper and scores it
against the sentence set the audio was synthesised from (line ``index`` of
``--sentences``), so throughput numbers can be reported alongside a WER that
proves the speed did not come from garbled speech::

    python benchmark/chatterbox/wer_eval.py --wavs results/<date>/mstar_c8/wavs \
        --sentences /path/sentences_200.txt --out results/<date>/mstar_c8/wer.json

Runs offline against the shared HF cache (``HF_HUB_OFFLINE=1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from torchaudio.functional import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.asr_eval import _compute_wer  # noqa: E402

ASR_MODEL = "openai/whisper-large-v3-turbo"


def transcribe(wavs: list[Path], model_id: str, device: str, batch_size: int) -> list[str]:
    """Whisper via the model classes directly: the ASR pipeline's preprocess
    imports torchcodec (FFmpeg libraries the nodes lack) even for in-memory
    arrays, and soundfile already gives us the samples."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    samples = []
    for path in wavs:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
        if sr != 16000:
            wav = resample(wav, sr, 16000)
        samples.append(wav.numpy())
    texts: list[str] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        feats = processor.feature_extractor(batch, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(
                feats.to(device, dtype), language="en", task="transcribe", max_new_tokens=220,
            )
        texts.extend(t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wavs", required=True, help="directory of <index>.wav files")
    parser.add_argument("--sentences", required=True, help="reference sentences, one per line")
    parser.add_argument("--asr-model", default=ASR_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="only the first N files (smoke tests)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sentences = [s.strip() for s in Path(args.sentences).read_text().splitlines() if s.strip()]
    # "<index>.wav" from the offline drivers or "req_<index>.wav" from benchmark.runner
    indexed = {p: int(p.stem.removeprefix("req_")) for p in Path(args.wavs).glob("*.wav")
               if p.stem.removeprefix("req_").isdigit()}
    wavs = sorted(indexed, key=indexed.get)
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        raise SystemExit(f"no <index>.wav / req_<index>.wav files under {args.wavs}")
    references = [sentences[indexed[p] % len(sentences)] for p in wavs]

    hypotheses = transcribe(wavs, args.asr_model, args.device, args.batch_size)
    report = _compute_wer(references, hypotheses)
    durations = [sf.info(str(p)).duration for p in wavs]
    result = {
        "asr_model": args.asr_model, "num_files": len(wavs), "wer": report["wer"],
        "total_audio_s": sum(durations), "mean_audio_s": sum(durations) / len(durations),
        "worst": sorted(report["per_sample"], key=lambda s: -s["wer"])[:10],
        "files": [p.name for p in wavs],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**result, "per_sample": report["per_sample"]}, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "worst"}, indent=2))
    for s in result["worst"][:5]:
        print(f"  [{s['index']}] wer={s['wer']:.2f}\n    ref: {s['ref']}\n    hyp: {s['hyp']}")


if __name__ == "__main__":
    main()
