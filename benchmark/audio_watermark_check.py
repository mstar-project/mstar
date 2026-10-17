#!/usr/bin/env python3
"""Checks behind the opt-in AudioSeal watermark (``mstar/model/audio_watermark.py``).

Needs ``pip install -e '.[audioseal,audio]'``; ``quality`` also needs ``pesq``.

``exactness``: on a synthetic 3 s, 24 kHz speech-like signal, the streamed
watermark (aligned, uneven and batched chunkings) against AudioSeal's own
whole-utterance ``get_watermark``, plus AudioSeal 0.2's ``streaming()`` for
contrast (it does not carry history, upstream issue #105)::

    python -m benchmark.audio_watermark_check exactness --device cuda

``cost``: one ``apply_pcm16`` call per codec chunk length (1 to 25 Qwen3-TTS
frames of 1920 samples), batch 1 and 8, streams already under way::

    python -m benchmark.audio_watermark_check cost --device cuda

``quality``: from the per-request WAVs ``benchmark.tts_speech_bench
--save-audio-dir`` writes, one directory served with the watermark off and
one with it on. The clean files are marked here chunk by chunk along the
codec's schedule (served greedy speech does not repeat across server
restarts, so served on/off pairs are not the same speech) for SNR, PESQ and
streamed-vs-whole; the detector then scores every served file, whole and one
codec chunk at a time::

    python -m benchmark.audio_watermark_check quality --clean off_wav --marked on_wav --out quality.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from mstar.model.audio_watermark import AudioSealDetector, AudioSealWatermarker, AudioWatermarkConfig, WatermarkStream

MESSAGE = 0xBEEF
FRAME = 1920  # Qwen3-TTS samples per codec frame (24 kHz)
SCHEDULE = (1, 3, 8, 16)  # the codec's first windows, in frames
CHUNK = 25  # frames per window after the ramp


def _stream(wmk: AudioSealWatermarker, x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    stream, out, pos = WatermarkStream(), [], 0
    for i, n in enumerate(sizes):
        out.append(wmk.apply([x[pos : pos + n]], [stream], [i == len(sizes) - 1])[0])
        pos += n
    return torch.cat(out)


def exactness(args) -> dict:
    from audioseal.libs.moshi.utils.compile import no_compile

    torch.manual_seed(0)
    cfg = AudioWatermarkConfig(enabled=True, message=MESSAGE)
    wmk = AudioSealWatermarker.build(cfg, args.device)
    sr = 24000
    t = torch.arange(3 * sr) / sr
    phase = 2 * torch.pi * torch.cumsum(140 + 30 * torch.sin(2 * torch.pi * 0.7 * t), 0) / sr
    x = sum(torch.sin(k * phase) / k for k in range(1, 12)) * (0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * t)) * 0.15
    x = (x + 0.005 * torch.randn_like(x)).to(args.device)
    msg = torch.tensor([cfg.message_bits()], device=args.device)

    def library(audio):
        with torch.no_grad(), no_compile():
            return audio + wmk.generator.get_watermark(audio.view(1, 1, -1), message=msg).view(-1)

    ref = library(x)
    n = x.shape[0]
    aligned = [1920, 5760, 15360, 48000]
    aligned.append(n - sum(aligned))
    uneven = [2048] * 10 + [n - 20480]
    report = {
        "device": str(args.device),
        "samples": n,
        "offline_max_abs": (wmk.offline(x) - ref).abs().max().item(),
        "aligned_max_abs": (_stream(wmk, x, aligned) - ref).abs().max().item(),
        "uneven_max_abs": (_stream(wmk, x, uneven) - ref).abs().max().item(),
    }
    xs = [x, torch.roll(x, 777), 0.5 * torch.flip(x, [0])]
    streams = [WatermarkStream() for _ in xs]
    outs: list[list[torch.Tensor]] = [[] for _ in xs]
    pos = 0
    for i, size in enumerate(aligned):
        marked = wmk.apply([xi[pos : pos + size] for xi in xs], streams, [i == len(aligned) - 1] * len(xs))
        for k, m in enumerate(marked):
            outs[k].append(m)
        pos += size
    errors = [(torch.cat(o) - library(xi)).abs().max().item() for o, xi in zip(outs, xs, strict=True)]
    report["batched_max_abs"] = max(errors)
    with torch.no_grad(), no_compile(), wmk.generator.streaming(1):
        parts, pos = [], 0
        for size in aligned:
            parts.append(wmk.generator(x[pos : pos + size].view(1, 1, -1), message=msg).view(-1))
            pos += size
    report["library_streaming_max_abs"] = (torch.cat(parts) - ref).abs().max().item()
    return report


def cost(args) -> dict:
    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    wmk = AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True, message=MESSAGE), device)
    rows = []
    for batch in (1, 8):
        for frames in (1, 3, 8, 16, 25):
            streams = [WatermarkStream() for _ in range(batch)]
            chunks = [(torch.randn(FRAME * frames, device=device) * 3000).to(torch.int16) for _ in range(batch)]
            for _ in range(5):
                wmk.apply_pcm16(chunks, streams)
            times = []
            for _ in range(args.iters):
                sync()
                t0 = time.perf_counter()
                wmk.apply_pcm16(chunks, streams)
                sync()
                times.append(1000 * (time.perf_counter() - t0))
            times.sort()
            p50, p90 = statistics.median(times), times[int(0.9 * len(times))]
            rows.append({"batch": batch, "frames": frames, "audio_ms": FRAME * frames / 24,
                         "ms_p50": round(p50, 2), "ms_p90": round(p90, 2)})
            print(json.dumps(rows[-1]), flush=True)
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "rows": rows}


def _chunk_edges(n: int) -> list[int]:
    edges, pos, sizes = [0], 0, [s * FRAME for s in SCHEDULE]
    while pos < n:
        pos = min(n, pos + (sizes.pop(0) if sizes else CHUNK * FRAME))
        edges.append(pos)
    return edges


def _wavs(directory: str) -> dict[str, np.ndarray]:
    import soundfile as sf

    return {p.stem: sf.read(p, dtype="int16")[0] for p in sorted(Path(directory).glob("*.wav"))}


def _rates(marked: list[tuple[float, int]], clean: list[tuple[float, int]], threshold: float) -> dict:
    return {
        "marked": len(marked),
        "marked_detected": sum(s > threshold for s, _ in marked),
        "marked_message_exact": sum(m == MESSAGE for _, m in marked),
        "clean": len(clean),
        "clean_flagged": sum(s > threshold for s, _ in clean),
        "clean_max_score": round(max((s for s, _ in clean), default=0.0), 3),
    }


def _detect(det: AudioSealDetector, files: dict[str, np.ndarray]) -> tuple[list, dict[int, list]]:
    whole, per_chunk = [], {}
    for x in files.values():
        if len(x) < FRAME:
            continue
        whole.append(det.detect(torch.from_numpy(x.astype(np.float32) / 32767)))
        edges = _chunk_edges(len(x))
        for a, b in zip(edges, edges[1:], strict=False):
            if b - a == FRAME * ((b - a) // FRAME):  # whole codec windows only
                score = det.detect(torch.from_numpy(x[a:b].astype(np.float32) / 32767))
                per_chunk.setdefault((b - a) // FRAME, []).append(score)
    return whole, per_chunk


def quality(args) -> dict:
    import torchaudio
    from pesq import pesq

    def to_16k(audio: np.ndarray) -> np.ndarray:
        return torchaudio.functional.resample(torch.from_numpy(audio), 24000, 16000).numpy()

    wmk = AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True, message=MESSAGE), args.device)
    det = AudioSealDetector.build(args.device)
    clean = _wavs(args.clean)
    snrs, pesqs, lsb = [], [], []
    for x in clean.values():
        pcm = torch.from_numpy(x.copy())
        stream, edges = WatermarkStream(), _chunk_edges(len(x))
        spans = zip(edges, edges[1:], strict=False)
        parts = [wmk.apply_pcm16([pcm[a:b]], [stream], [b == edges[-1]])[0] for a, b in spans]
        streamed = torch.cat(parts).cpu().numpy()
        whole = wmk.apply_pcm16([pcm], [WatermarkStream()], [True])[0].cpu().numpy()
        lsb.append(int(np.abs(streamed.astype(np.int32) - whole.astype(np.int32)).max()))
        c, m = x.astype(np.float64) / 32767, streamed.astype(np.float64) / 32767
        snrs.append(10 * np.log10((c**2).sum() / max(((m - c) ** 2).sum(), 1e-20)))
        pesqs.append(pesq(16000, to_16k(c), to_16k(m), "wb"))
    report = {
        "files": len(snrs),
        "snr_db_mean_min": [round(statistics.mean(snrs), 1), round(min(snrs), 1)],
        "pesq_wb_mean_min": [round(statistics.mean(pesqs), 2), round(min(pesqs), 2)],
        "streamed_vs_whole_max_lsb": max(lsb),
    }
    if args.marked:
        on_whole, on_chunks = _detect(det, _wavs(args.marked))
        off_whole, off_chunks = _detect(det, clean)
        report["served_whole_files"] = _rates(on_whole, off_whole, args.threshold)
        report["served_per_chunk"] = {
            f"{f} frames ({f * FRAME // 24} ms)": _rates(on_chunks.get(f, []), off_chunks.get(f, []), args.threshold)
            for f in sorted(set(on_chunks) | set(off_chunks))
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("exactness", "cost", "quality"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--out", default=None, help="also write the report as JSON")
        if name == "cost":
            p.add_argument("--iters", type=int, default=40)
        if name == "quality":
            p.add_argument("--clean", required=True, help="WAVs served with the watermark off")
            p.add_argument("--marked", default=None, help="WAVs served with the watermark on")
            p.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    report = {"exactness": exactness, "cost": cost, "quality": quality}[args.command](args)
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
