"""WER quality guard for TTS engines (BENCHMARK_PROTOCOL.md, TTS row).

Synthesizes every line of a sentence file through an OpenAI-compatible
``/v1/audio/speech`` (M*, Kokoro-FastAPI, ...) or through the raw ``kokoro``
package, transcribes the audio with ``openai/whisper-large-v3-turbo`` and
reports the word error rate against the input text, with Whisper's English
normalizer applied to both sides. Per-sentence synthesis wall time and audio
duration are recorded too; for the package path that is the single-request
real-time factor of the reference implementation.

    python benchmark/tts_wer.py --url http://127.0.0.1:8000 --sentences sentences_200.txt --out out/mstar
    python benchmark/tts_wer.py --engine kokoro-package --sentences sentences_200.txt --out out/package
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


def read_wav(data: bytes) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate}")
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def synth_openai(url: str, model: str, voice: str, speed: float, text: str) -> np.ndarray:
    import requests

    body = {"model": model, "input": text, "voice": voice, "speed": speed, "response_format": "wav"}
    r = requests.post(f"{url.rstrip('/')}/v1/audio/speech", json=body, timeout=600)
    r.raise_for_status()
    return read_wav(r.content)


class PackageSynth:
    """The ``kokoro`` package, one request at a time (the reference implementation)."""

    def __init__(self, voice: str, device: str):
        import torch
        from kokoro import KModel, KPipeline

        self.voice = voice
        self.model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self.pipeline = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", model=self.model)
        self.torch = torch

    def __call__(self, text: str, speed: float) -> np.ndarray:
        chunks = [r.audio.numpy() for r in self.pipeline(text, voice=self.voice, speed=speed) if r.audio is not None]
        if self.model.device.type == "cuda":
            self.torch.cuda.synchronize()
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


ASR_SAMPLE_RATE = 16000
WHISPER_WINDOW_SECONDS = 30


def transcribe(wavs: list[Path], asr_model: str, device: str, batch_size: int) -> list[str]:
    """Whisper transcripts of the files (each at most one 30 s window)."""
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(asr_model)
    model = WhisperForConditionalGeneration.from_pretrained(asr_model, dtype=dtype).to(device).eval()
    clips = []
    for path in wavs:
        audio, rate = sf.read(path, dtype="float32")
        audio = audio if audio.ndim == 1 else audio.mean(axis=1)
        g = np.gcd(rate, ASR_SAMPLE_RATE)
        clip = resample_poly(audio, ASR_SAMPLE_RATE // g, rate // g).astype(np.float32)
        clips.append(clip[: WHISPER_WINDOW_SECONDS * ASR_SAMPLE_RATE])
    texts: list[str] = []
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            features = processor(batch, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt").input_features
            ids = model.generate(features.to(device=device, dtype=dtype), language="en", task="transcribe")
            texts.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["openai", "kokoro-package"], default="openai")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="server for --engine openai")
    parser.add_argument("--model", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sentences", required=True, help="text file, one sentence per line")
    parser.add_argument("--out", required=True, help="output directory (wavs, transcripts.tsv, summary.json)")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--device", default=None, help="for the package path and the ASR model (default: cuda if available)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--asr-batch-size", type=int, default=16)
    args = parser.parse_args()

    import jiwer
    import soundfile as sf
    import torch
    from transformers import WhisperTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sentences = [s.strip() for s in Path(args.sentences).read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.limit:
        sentences = sentences[: args.limit]
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)

    synth = PackageSynth(args.voice, device) if args.engine == "kokoro-package" else None
    wavs, seconds, audio_seconds = [], [], []
    for i, text in enumerate(sentences):
        t0 = time.perf_counter()
        audio = synth(text, args.speed) if synth else synth_openai(args.url, args.model, args.voice, args.speed, text)
        seconds.append(time.perf_counter() - t0)
        audio_seconds.append(len(audio) / SAMPLE_RATE)
        path = out / "wav" / f"{i + 1:03d}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        wavs.append(path)

    hyps = transcribe(wavs, args.asr_model, device, args.asr_batch_size)
    normalize = WhisperTokenizer.from_pretrained(args.asr_model).normalize
    refs_n = [normalize(s) for s in sentences]
    hyps_n = [normalize(h) for h in hyps]
    wer = jiwer.wer(refs_n, hyps_n)
    per_sentence = [jiwer.wer(r, h) if r else float("nan") for r, h in zip(refs_n, hyps_n, strict=True)]

    with (out / "transcripts.tsv").open("w", encoding="utf-8") as f:
        f.write("id\twer\tsynth_s\taudio_s\treference\thypothesis\n")
        rows = zip(sentences, hyps, per_sentence, seconds, audio_seconds, strict=True)
        for i, (text, hyp, w, s, a) in enumerate(rows):
            f.write(f"{i + 1}\t{w:.3f}\t{s:.3f}\t{a:.2f}\t{text}\t{hyp.strip()}\n")
    summary = {
        "engine": args.engine,
        "url": args.url if synth is None else None,
        "model": args.model,
        "voice": args.voice,
        "speed": args.speed,
        "asr_model": args.asr_model,
        "num_sentences": len(sentences),
        "wer": wer,
        "audio_seconds_total": sum(audio_seconds),
        "synth_seconds_total": sum(seconds),
        "rtf_sequential": sum(seconds) / max(sum(audio_seconds), 1e-9),
        "synth_seconds_p50": statistics.median(seconds),
        "synth_seconds_p95": sorted(seconds)[int(0.95 * (len(seconds) - 1))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
