"""Reference transcripts for WER parity: the HF/SDK model on the benchmark set.

Runs the *reference implementation* (not mstar) greedily over the shared
LibriSpeech set and writes its hypotheses plus WER, so a served engine's
WER can be compared against the same reference on the same files:

    # Whisper: transformers' WhisperForConditionalGeneration, greedy, bf16
    python -m benchmark.asr_reference --model openai/whisper-large-v3-turbo \\
        --output-json results/2026-09-18/hf_reference_turbo.json
    # Qwen3-ASR: the Qwen3-ASR SDK's transformers backend (refs/Qwen3-ASR)
    python -m benchmark.asr_reference --model Qwen/Qwen3-ASR-1.7B --sdk $MSTAR_WS/refs/Qwen3-ASR \\
        --output-json results/2026-09-18/sdk_reference_qwen3_asr.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

from benchmark.asr_bench import Utterance, load_set, make_normalizer, word_error_rate


def _wave(utt: Utterance):
    import soundfile as sf

    audio, sr = sf.read(str(utt.path), dtype="float32", always_2d=True)
    assert sr == 16000, utt.path
    return audio.mean(axis=1)


def whisper_reference(model_id: str, items: list[Utterance], language: str, batch_size: int, dtype) -> dict[str, str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).cuda().eval()
    out: dict[str, str] = {}
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        feats = processor([_wave(u) for u in batch], sampling_rate=16000, return_tensors="pt")["input_features"]
        with torch.no_grad():
            ids = model.generate(
                feats.cuda().to(dtype), language=language or None, task="transcribe",
                do_sample=False, num_beams=1, max_new_tokens=440,
            )
        for u, text in zip(batch, processor.batch_decode(ids, skip_special_tokens=True), strict=True):
            out[u.uid] = text.strip()
        print(f"  {min(i + batch_size, len(items))}/{len(items)}", flush=True)
    return out


def qwen3_asr_reference(model_id: str, sdk: str, items: list[Utterance], language: str, batch_size: int, dtype):
    pkg = types.ModuleType("qwen_asr")
    pkg.__path__ = [str(Path(sdk) / "qwen_asr")]
    sys.modules.setdefault("qwen_asr", pkg)
    core = types.ModuleType("qwen_asr.core")
    core.__path__ = [str(Path(sdk) / "qwen_asr" / "core")]
    sys.modules.setdefault("qwen_asr.core", core)
    backend = importlib.import_module("qwen_asr.core.transformers_backend")
    model = backend.Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    processor = backend.Qwen3ASRProcessor.from_pretrained(model_id, fix_mistral_regex=True)
    names = {"en": "English", "zh": "Chinese", "de": "German", "fr": "French", "es": "Spanish"}
    lang_name = names.get(language, language)
    prompt = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    ) + (f"language {lang_name}<asr_text>" if language else "")
    out: dict[str, str] = {}
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        inputs = processor(text=[prompt] * len(batch), audio=[_wave(u) for u in batch], return_tensors="pt",
                           padding=True).to("cuda")
        inputs["input_features"] = inputs["input_features"].to(dtype)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=512)
        texts = processor.batch_decode(gen.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for u, text in zip(batch, texts, strict=True):
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1]
            out[u.uid] = text.strip()
        print(f"  {min(i + batch_size, len(items))}/{len(items)}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="reference-implementation transcripts + WER on the benchmark set")
    p.add_argument("--model", required=True, help="HF id or local dir")
    p.add_argument("--sdk", default=None, help="Qwen3-ASR SDK checkout (selects the Qwen3-ASR reference)")
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("MSTAR_WS_ROOT", "."), "commons/bench/data/asr"))
    p.add_argument("--short-set", default="test_clean_200")
    p.add_argument("--language", default="en")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--normalizer", choices=["whisper", "basic"], default="whisper")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    items = load_set(Path(args.data_dir) / args.short_set)
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    if args.sdk:
        hyps = qwen3_asr_reference(args.model, args.sdk, items, args.language, args.batch_size, dtype)
    else:
        hyps = whisper_reference(args.model, items, args.language, args.batch_size, dtype)
    elapsed = time.perf_counter() - t0
    normalize = make_normalizer(args.normalizer)
    wer = word_error_rate([u.reference for u in items], [hyps[u.uid] for u in items], normalize)
    record = {
        "model": args.model, "reference": "qwen_asr sdk (transformers)" if args.sdk else "transformers greedy",
        "dtype": args.dtype, "language": args.language, "normalizer": args.normalizer,
        "num_utterances": len(items), "audio_seconds": sum(u.duration for u in items),
        "elapsed_s": elapsed, "wer": wer, "hypotheses": hyps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    Path(args.output_json).write_text(json.dumps(record, indent=2))
    print(f"WER {100 * wer:.2f}% on {len(items)} utterances ({elapsed:.0f} s); wrote {args.output_json}")


if __name__ == "__main__":
    main()
