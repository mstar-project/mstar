"""Whisper transcripts of generated speech, the judge for TTS and spoken replies.

Default judge is whisper-large-v3-turbo with Whisper's English normalizer, same as
benchmark/tts_speech_wer.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.correctness.metrics import read_wav, resample

JUDGE_MODEL = "openai/whisper-large-v3-turbo"
ASR_SR = 16000


def transcribe(paths: list[Path], device: str, model_id: str = JUDGE_MODEL, batch_size: int = 16,
               language: str | None = "en") -> list[str]:
    """One transcript per file. Missing or unreadable files give ""."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    gen_kwargs = {"task": "transcribe", **({"language": language} if language else {})}
    clips: list[np.ndarray | None] = []
    for p in paths:
        try:
            x, sr = read_wav(p)
            clips.append(np.nan_to_num(resample(x, sr, ASR_SR)))
        except Exception:
            clips.append(None)
    texts = [""] * len(paths)
    short = [i for i, c in enumerate(clips) if c is not None and 0 < len(c) <= 30 * ASR_SR]
    long = [i for i, c in enumerate(clips) if c is not None and len(c) > 30 * ASR_SR]
    with torch.inference_mode():
        for s in range(0, len(short), batch_size):
            idx = short[s:s + batch_size]
            feats = processor([clips[i] for i in idx], sampling_rate=ASR_SR, return_tensors="pt").input_features
            ids = model.generate(feats.to(device=device, dtype=dtype), max_new_tokens=440, **gen_kwargs)
            for i, t in zip(idx, processor.batch_decode(ids, skip_special_tokens=True), strict=True):
                texts[i] = t.strip()
        for i in long:  # sequential long-form decoding
            inputs = processor(clips[i], sampling_rate=ASR_SR, return_tensors="pt", truncation=False,
                               padding="longest", return_attention_mask=True)
            ids = model.generate(inputs.input_features.to(device=device, dtype=dtype),
                                 attention_mask=inputs.attention_mask.to(device), return_timestamps=True,
                                 **gen_kwargs)
            texts[i] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return texts
