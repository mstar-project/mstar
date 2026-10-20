"""Joint audio+video generation (LTX-2.5) on the 5 prompts of video_v1, without a reference.

The diffusers pipeline is the model's own implementation, so there's nothing independent to
compare against. The server returns the video as an mp4 and the audio as separate interleaved
16-bit PCM. Both are saved and checked for sanity (frames, size, flat frames; audio length against
the clip, silence, clipping), and each item runs on two server instances with the same seed:
frame PSNR and audio SNR between them, with identical decoded frames and identical PCM counted.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, frame_identity, self_consistency_check
from benchmark.correctness.families.video import read_video
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Interleaved int16 PCM into a WAV, as is."""
    pcm = pcm[: len(pcm) - len(pcm) % (2 * channels)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    """[samples, channels] float32 in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as w:
        ch, sr, raw = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0, sr


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    """SNR of b against a over the common length (inf when identical)."""
    n = min(len(a), len(b))
    err = float(np.sum((a[:n].astype(np.float64) - b[:n]) ** 2))
    sig = float(np.sum(a[:n].astype(np.float64) ** 2))
    if err == 0:
        return math.inf
    return 10 * math.log10(sig / err) if sig > 0 else -math.inf


class AudioVideoFamily(Family):
    name = "audio-video"
    set_name = "video_v1"
    default_restarts = 2
    default_concurrency = 1
    request_timeout_s = 3600.0
    self_consistency_only = True
    # the model card's defaults: 960x544, 121 frames (8k + 1) at 24 fps, about 5 s
    height, width, num_frames, fps = 544, 960, 121, 24.0

    def geometry(self) -> dict:
        return {"height": self.opt("height", self.height, int), "width": self.opt("width", self.width, int),
                "num_frames": self.opt("num_frames", self.num_frames, int), "fps": self.opt("fps", self.fps, float)}

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=it.prompt, output_modalities="video,audio",
                        model_kwargs={"seed": it.params["seed"], **self.geometry()}) for it in sset.items]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            if not r.ok:
                continue
            if r.video:
                (out_dir / f"{r.item_id}.mp4").write_bytes(bytes(r.video))
            if r.audio:
                meta = next((m for m in r.metadata if m.get("modality") == "audio"), {})
                write_pcm_wav(out_dir / f"{r.item_id}.wav", bytes(r.audio), int(meta.get("sample_rate", 48000)),
                              int(meta.get("num_channels", 2)))

    def reference_id(self) -> str:
        return f"{self.model_key}-no-reference"

    def reference_description(self) -> str:
        return "none: the diffusers pipeline is the model's own implementation, not an independent reference"

    def make_reference(self, sset, ref_dir, device):
        raise NotImplementedError(self.reference_description())

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        geo = self.geometry()
        n = len(sset.items)
        fails = failed_items(runs[0])
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL",
                 f"all {n} requests return video and audio", "; ".join(f"{k}: {v[:80]}" for k, v in
                                                                     list(fails.items())[:3]))
        rows, bad_video, bad_audio, video_repro, audio_repro, sheet, labels = [], [], [], [], [], [], []
        clip_s = geo["num_frames"] / geo["fps"]
        for it in sset.items:
            row = {"id": it.id, "prompt": it.prompt[:60]}
            mp4, wav = runs[0] / "outputs" / f"{it.id}.mp4", runs[0] / "outputs" / f"{it.id}.wav"
            frames = read_video(mp4) if mp4.is_file() else None
            if frames is None:
                bad_video.append(it.id)
            else:
                row.update(frames=len(frames), dims="x".join(map(str, frames.shape[1:3])))
                flat = int((frames.reshape(len(frames), -1).std(axis=1) < 2).sum())
                if len(frames) != geo["num_frames"] or frames.shape[1:3] != (geo["height"], geo["width"]) or flat:
                    bad_video.append(it.id)
                    row["video_problems"] = f"{len(frames)} frames, {frames.shape[1]}x{frames.shape[2]}, {flat} flat"
                sheet.append([frames[0], frames[len(frames) // 2], frames[-1]])
                labels += [f"{it.id} first", "mid", "last"]
            if not wav.is_file():
                bad_audio.append(it.id)
            else:
                audio, sr = read_pcm_wav(wav)
                stats = M.audio_stats(audio.mean(axis=1), sr)
                problems = M.audio_problems(stats, max_silence_frac=0.95)
                if abs(stats["duration_s"] - clip_s) > 0.25:
                    problems.append(f"audio {stats['duration_s']:.2f} s for a {clip_s:.2f} s clip")
                if audio.shape[1] != 2:
                    problems.append(f"{audio.shape[1]} channel(s)")
                row.update(audio_s=round(stats["duration_s"], 2), channels=audio.shape[1],
                           silence=round(stats["silence_frac"], 2), rms_dbfs=round(stats["rms_dbfs"], 1))
                if problems:
                    bad_audio.append(it.id)
                    row["audio_problems"] = "; ".join(problems)
            if len(runs) > 1:
                other = runs[1] / "outputs"
                if frames is not None and (other / f"{it.id}.mp4").is_file():
                    g = read_video(other / f"{it.id}.mp4")
                    row["frames_identical_r1_r2"] = M.frames_equal(frames, g)
                    if g.shape == frames.shape:
                        row["r1_r2_frame_psnr_min"] = min(M.frame_psnrs(frames, g))
                        video_repro.append(row["r1_r2_frame_psnr_min"])
                if wav.is_file() and (other / f"{it.id}.wav").is_file():
                    row["pcm_identical"] = wav.read_bytes() == (other / f"{it.id}.wav").read_bytes()
                    b, _ = read_pcm_wav(other / f"{it.id}.wav")
                    row["r1_r2_audio_snr_db"] = snr_db(read_pcm_wav(wav)[0], b)
                    audio_repro.append(row["r1_r2_audio_snr_db"])
            rows.append(row)
        sc.check("videos with the requested frames and size, none flat", n - len(bad_video),
                 "PASS" if not bad_video else "FAIL", f"{geo['num_frames']} frames of {geo['width']}x{geo['height']}",
                 ", ".join(bad_video[:8]))
        sc.check("audio tracks failing sanity", len(bad_audio),
                 band(len(bad_audio), lambda v: v == 0, lambda v: v <= 1),
                 "0 PASS, 1 WARN: stereo, length within 0.25 s of the clip, <= 95% silent, <= 1% clipped",
                 ", ".join(bad_audio[:8]))
        self_consistency_check(sc, video_repro, "same-seed min frame PSNR, two server instances (dB)", len(runs),
                               lambda v: v >= 40, lambda v: v >= 25,
                               ">=40 PASS (inf = identical decoded frames), >=25 WARN (provisional)")
        self_consistency_check(sc, audio_repro, "same-seed min audio SNR, two server instances (dB)", len(runs),
                               lambda v: v >= 40, lambda v: v >= 20, ">=40 PASS (inf = identical), >=20 WARN "
                                                                     "(provisional)")
        if len(runs) > 1:
            frame_identity(sc, rows, encoded=True)
            sc.metrics["byte_identical_r1_r2"] = {"pcm": sum(bool(r.get("pcm_identical")) for r in rows), "n": n}
        if sheet:
            M.contact_sheet(sheet, labels, work / "contact_sheet.png")
            sc.artifacts.append(str(work / "contact_sheet.png"))
        sc.rows = rows
        (work / "audio_video.json").write_text(json.dumps(rows, indent=1, default=str))
        sc.artifacts += [str(r / "outputs") for r in runs]
        return sc
