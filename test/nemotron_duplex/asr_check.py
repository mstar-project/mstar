#!/usr/bin/env python3
"""Transcribe served agent audio with Whisper large-v3-turbo to check that it is
intelligible and says the reply text.

The talker's MaskGIT sampling is knife-edge, so a different attention backend
or a CUDA-graph replay changes the waveform without changing the words;
comparing transcripts (not samples) is the right parity check for audio. An
RMS "voiced" heuristic is not: it passed on pure hiss once.

    python test/nemotron_duplex/asr_check.py agent_out.wav [more.wav ...]

Runs offline against the shared Hugging Face cache (``HF_HUB_OFFLINE=1``); the
model must have been prefetched. Uses the processor + model API directly
because the transformers pipeline pulls in torchcodec/ffmpeg.
"""
import sys
import wave

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = "openai/whisper-large-v3-turbo"


def transcribe(paths: list[str], device: str = "cuda:0") -> dict[str, str]:
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    out = {}
    for path in paths:
        with wave.open(path) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        x = AF.resample(torch.from_numpy(pcm), sr, 16000).numpy()
        feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=128)
        out[path] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path, text in transcribe(sys.argv[1:]).items():
        print(f"{path.split('/')[-1]:28s} -> {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
