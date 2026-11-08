#!/usr/bin/env python3
"""Checks behind the opt-in AudioSeal watermark (``mstar/model/audio_watermark.py``).

Needs ``pip install -e '.[audioseal,audio]'``; ``quality`` also needs ``pesq``.

``exactness``: on a synthetic 3 s, 24 kHz speech-like signal, the streamed
watermark (aligned, uneven and batched chunkings) against AudioSeal's own
whole-utterance ``get_watermark``, plus AudioSeal 0.2's ``streaming()`` for
contrast (it does not carry history, upstream issue #105)::

    python -m benchmark.audio_watermark_check exactness --device cuda

``cost``: one ``apply_pcm16`` call per codec chunk length (1 to 25 Qwen3-TTS
frames of 1920 samples), batch 1, 8 and 32, streams already under way: the
time until the audio is ready and until the call returns (how long it holds
the worker loop); ``--captured`` adds the replay of the engine's captures::

    python -m benchmark.audio_watermark_check cost --device cuda --captured

``stream``: served clean WAVs marked as the codec would, all files in flight
together with staggered starts, one call per step; ``--compare`` reports the
int16 difference to another run, e.g. the same command on another commit::

    python -m benchmark.audio_watermark_check stream --wavs off_wav --out marked --captured

``quality``: from the per-request WAVs ``benchmark.tts_speech_bench
--save-audio-dir`` writes, one directory served with the watermark off and
one with it on. The clean files are marked here chunk by chunk along the
codec's schedule (served greedy speech does not repeat across server
restarts, so served on/off pairs are not the same speech) for SNR, PESQ and
streamed-vs-whole; the detector then scores every served file, whole and one
codec chunk at a time, and whole copies resampled to 16 and 48 kHz::

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
CAPTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)  # the codec's
RESAMPLED_RATES = (16000, 48000)  # copies the detector must still find the mark in


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
        # AudioSeal's own call in full fp32; on a GPU the watermarker's TF32 cuDNN convs land a few int16 steps off
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad(), no_compile():
                return audio + wmk.generator.get_watermark(audio.view(1, 1, -1), message=msg).view(-1)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn

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


class _NoResources:
    """The engine step runner a node without resources sees at capture."""

    def build_accelerator_graph_buffers(self, *args, **kwargs) -> None:
        pass

    def ingest_capture_row(self, rid) -> None:
        pass


def captured_runners(wmk: AudioSealWatermarker, batch_sizes=CAPTURE_BATCH_SIZES) -> dict:
    """The engine's captures of the codec's watermark lengths, as Qwen3-TTS's
    ``get_piecewise_accelerator_graph_configs`` declares them (CUDA only)."""
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner

    configs = wmk.piecewise_configs([f * FRAME for f in {*SCHEDULE, CHUNK}], batch_sizes)
    pool = get_accelerator_graph_backend(wmk.device).graph_pool_handle()
    runners = {}
    for label, config in configs.items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources={}, step_runner=_NoResources(), device=wmk.device,
            autocast_dtype=None, num_slots=1, memory_pool=pool,
        )
        runner.warmup_and_capture()
        runners[label] = runner
    return runners


def _supports_runners(wmk: AudioSealWatermarker) -> bool:
    return hasattr(wmk, "piecewise_configs")


def cost(args) -> dict:
    """Per ``apply_pcm16`` call: ``ms`` until the audio is on the GPU's
    clock (synchronized) and ``host_ms`` until the call returns, which is how
    long it holds the worker loop. With ``--captured``, also the replay of the
    engine's capture."""
    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    wmk = AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True, message=MESSAGE), device)
    paths = {"eager": None}
    if args.captured and device.type == "cuda" and _supports_runners(wmk):
        paths["captured"] = captured_runners(wmk)
    rows = []
    for path, runners in paths.items():
        for batch in (1, 8, 32):
            for frames in (1, 3, 8, 16, 25):
                streams = [WatermarkStream() for _ in range(batch)]
                chunks = [(torch.randn(FRAME * frames, device=device) * 3000).to(torch.int16) for _ in range(batch)]
                kwargs = {"runners": runners} if runners else {}
                for _ in range(5):
                    wmk.apply_pcm16(chunks, streams, **kwargs)
                times, host = [], []
                for _ in range(args.iters):
                    sync()
                    t0 = time.perf_counter()
                    wmk.apply_pcm16(chunks, streams, **kwargs)
                    host.append(1000 * (time.perf_counter() - t0))
                    sync()
                    times.append(1000 * (time.perf_counter() - t0))
                rows.append({"path": path, "batch": batch, "frames": frames, "audio_ms": FRAME * frames / 24,
                             "ms_p50": round(statistics.median(times), 3),
                             "host_ms_p50": round(statistics.median(host), 3)})
                print(json.dumps(rows[-1]), flush=True)
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "rows": rows}


def stream(args) -> dict:
    """Mark served clean WAVs as the codec would: every file one stream, all
    of them in flight together (file k starting at step k % 5, so a step
    mixes ramp, full and final chunks), one ``apply_pcm16`` call per step,
    with ``--captured`` through the engine's captures. Writes the marked
    WAVs to ``--out``; ``--compare DIR`` reports the int16 difference to
    another such run (e.g. the same command on another commit)."""
    import soundfile as sf

    device = torch.device(args.device)
    wmk = AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True, message=MESSAGE), device)
    runners = None
    if args.captured and device.type == "cuda" and _supports_runners(wmk):
        runners = captured_runners(wmk)
    files = _wavs(args.wavs)
    names = sorted(files)
    edges = {n: _chunk_edges(len(files[n])) for n in names}
    streams = {n: WatermarkStream() for n in names}
    parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    # the codec batches at most this many rows, as serving does; a bigger call
    # would run past every capture, eagerly
    max_rows = max(CAPTURE_BATCH_SIZES)
    for step in range(max(k % 5 + len(edges[n]) - 1 for k, n in enumerate(names))):
        live = [n for k, n in enumerate(names) if 0 <= step - k % 5 < len(edges[n]) - 1]
        for lo in range(0, len(live), max_rows):
            rows = live[lo:lo + max_rows]
            chunks, finals = [], []
            for n in rows:
                i = step - names.index(n) % 5
                a, b = edges[n][i], edges[n][i + 1]
                chunks.append(torch.from_numpy(files[n][a:b].copy()).to(device))
                finals.append(i + 1 == len(edges[n]) - 1)
            kwargs = {"runners": runners} if runners else {}
            marked = wmk.apply_pcm16(chunks, [streams[n] for n in rows], finals, **kwargs)
            for n, m in zip(rows, marked, strict=True):
                parts[n].append(m.cpu())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    marked = {}
    for n in names:
        marked[n] = torch.cat(parts[n]).numpy()
        sf.write(out / f"{n}.wav", marked[n], 24000, subtype="PCM_16")
    report = {"files": len(names), "captured": runners is not None, "out": str(out)}
    if args.compare:
        other = _wavs(args.compare)
        diffs = [int(np.abs(marked[n].astype(np.int32) - other[n].astype(np.int32)).max()) for n in names]
        report["compare"] = {
            "dir": args.compare, "bit_identical_files": sum(d == 0 for d in diffs),
            "max_abs_int16": max(diffs),
            "differing_samples": int(sum((marked[n] != other[n]).sum() for n in names)),
            "samples": int(sum(len(marked[n]) for n in names)),
        }
    return report


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


def _detect(det: AudioSealDetector, files: dict[str, np.ndarray]) -> tuple[list, dict[int, list], dict[int, list]]:
    import torchaudio

    whole, per_chunk, resampled = [], {}, {}
    for x in files.values():
        if len(x) < FRAME:
            continue
        audio = torch.from_numpy(x.astype(np.float32) / 32767)
        whole.append(det.detect(audio))
        for rate in RESAMPLED_RATES:
            copy = torchaudio.functional.resample(audio, 24000, rate)
            resampled.setdefault(rate, []).append(det.detect(copy, rate))
        edges = _chunk_edges(len(x))
        for a, b in zip(edges, edges[1:], strict=False):
            if b - a == FRAME * ((b - a) // FRAME):  # whole codec windows only
                score = det.detect(torch.from_numpy(x[a:b].astype(np.float32) / 32767))
                per_chunk.setdefault((b - a) // FRAME, []).append(score)
    return whole, per_chunk, resampled


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
        on_whole, on_chunks, on_resampled = _detect(det, _wavs(args.marked))
        off_whole, off_chunks, off_resampled = _detect(det, clean)
        report["served_whole_files"] = _rates(on_whole, off_whole, args.threshold)
        for rate in RESAMPLED_RATES:
            report[f"served_whole_files_at_{rate // 1000}k"] = _rates(
                on_resampled[rate], off_resampled[rate], args.threshold
            )
        report["served_per_chunk"] = {
            f"{f} frames ({f * FRAME // 24} ms)": _rates(on_chunks.get(f, []), off_chunks.get(f, []), args.threshold)
            for f in sorted(set(on_chunks) | set(off_chunks))
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("exactness", "cost", "quality", "stream"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--out", default=None, help="also write the report as JSON (stream: the WAV directory)")
        if name in ("cost", "stream"):
            p.add_argument("--captured", action="store_true", help="replay the engine's captures (CUDA)")
        if name == "cost":
            p.add_argument("--iters", type=int, default=40)
        if name == "stream":
            p.add_argument("--wavs", required=True, help="clean WAVs (served with the watermark off)")
            p.add_argument("--compare", default=None, help="marked WAVs of another run to compare against")
        if name == "quality":
            p.add_argument("--clean", required=True, help="WAVs served with the watermark off")
            p.add_argument("--marked", default=None, help="WAVs served with the watermark on")
            p.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.command == "stream" and not args.out:
        ap.error("stream needs --out")
    report = {"exactness": exactness, "cost": cost, "quality": quality, "stream": stream}[args.command](args)
    print(json.dumps(report, indent=1))
    if args.out and args.command != "stream":
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
