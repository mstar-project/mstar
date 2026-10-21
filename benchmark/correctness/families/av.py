"""Joint audio+video generation (LTX-2.5) on the 5 prompts of video_v1.

The server returns the video as an mp4 and the audio as separate interleaved 16-bit PCM. Both are
saved and checked for sanity (frames, size, flat frames; audio length against the clip, silence,
clipping). The reference is diffusers' LTX2Pipeline with the same distilled recipe and CPU-generator
seed: per-frame PSNR/SSIM of the decoded mp4 against its frames after the served encode, judged like
the video families (the raw 30 dB bar plus a calibrated floor 3 dB under the reference's own
math-SDPA spread), and the audio's log-mel distance to its waveform next to the same spread.
Without a reference the verdict is NO_REFERENCE, and two server instances with the same seed are
still compared (frame PSNR and audio SNR, identical decoded frames and identical PCM counted).
"""

from __future__ import annotations

import json
import math
import statistics
import wave
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import (
    Family,
    MathSdpaReference,
    failed_items,
    frame_identity,
    self_consistency_check,
    write_meta,
)
from benchmark.correctness.families.video import SERVER_X264, as_served, read_video, write_video
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


class AudioVideoFamily(MathSdpaReference, Family):
    name = "audio-video"
    set_name = "video_v1"
    default_restarts = 2
    default_concurrency = 1
    request_timeout_s = 3600.0
    hf_id = "Lightricks/LTX-2.5-Diffusers"
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

    # --- reference ------------------------------------------------------------------------------
    def reference_id(self) -> str:
        g = self.geometry()
        fps = "" if g["fps"] == self.fps else f"-{g['fps']:g}fps"
        return (f"diffusers-{self.hf_id.replace('/', '--')}-{g['height']}x{g['width']}x{g['num_frames']}{fps}"
                f"-distilled{self.arm_suffix()}")

    def reference_description(self) -> str:
        g = self.geometry()
        return (f"diffusers LTX2Pipeline for {self.hf_id}, bf16, cpu-generator seed, {g['height']}x{g['width']}, "
                f"{g['num_frames']} frames at {g['fps']:g} fps, distilled sigmas, guidance off"
                + (", math SDPA" if self.ref_sdpa_math else ""))

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        """The model card's distilled recipe (8 sigmas, every guidance off), the call of
        test/ltx2_5/record_oracle.py, with the noise from a CPU generator at the item seed as mstar's
        dit draws it. Frames go to <id>.npy and .mp4, audio to <id>.wav and <id>.audio.npy."""
        import torch
        from diffusers import LTX2Pipeline
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

        g = self.geometry()
        pipe = LTX2Pipeline.from_pretrained(self.hf_id, prompt_enhancer=None, torch_dtype=torch.bfloat16).to(device)
        pipe.set_progress_bar_config(disable=True)
        sr = int(pipe.vocoder.config.output_sampling_rate)
        out = ref_dir / "outputs"
        for it in sset.items:
            with torch.inference_mode(), self.sdpa_context():
                video, audio = pipe(
                    prompt=it.prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    height=g["height"], width=g["width"], num_frames=g["num_frames"], frame_rate=g["fps"],
                    sigmas=DISTILLED_SIGMA_VALUES, guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0,
                    audio_stg_scale=0.0, modality_scale=1.0, audio_modality_scale=1.0,
                    generator=torch.Generator("cpu").manual_seed(int(it.params["seed"])),
                    output_type="np", return_dict=False,
                )
            frames = (np.clip(np.asarray(video[0]), 0, 1) * 255).round().astype(np.uint8)
            np.save(out / f"{it.id}.npy", frames)
            write_video(frames, out / f"{it.id}.mp4", int(g["fps"]))
            samples = audio[0].float().cpu().numpy()  # [channels, samples]
            np.save(out / f"{it.id}.audio.npy", samples)
            pcm = (np.clip(samples.T, -1, 1) * 32767).round().astype("<i2").tobytes()
            write_pcm_wav(out / f"{it.id}.wav", pcm, sr, samples.shape[0])
            print(f"  ref {it.id}: video {frames.shape}, audio {samples.shape} at {sr} Hz", flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   pipeline=type(pipe).__name__, audio_sample_rate=sr)

    # --- scoring --------------------------------------------------------------------------------
    def _reference_row(self, row: dict, frames: np.ndarray | None, wav: Path, ref_out: Path, math_out: Path | None,
                       cache: dict) -> None:
        """Per-frame PSNR/SSIM and audio distances against the reference, and the math-SDPA arm's own."""
        fps = self.geometry()["fps"]

        def ref_frames(d: Path) -> np.ndarray | None:
            p = d / f"{row['id']}.npy"
            if p.is_file() and p not in cache:
                cache[p] = as_served(np.load(p), self.model_key, fps)
            return cache.get(p)

        ref = ref_frames(ref_out)
        if frames is not None and ref is not None and frames.shape[1:] == ref.shape[1:]:
            ps = M.frame_psnrs(ref, frames)
            n = min(len(ref), len(frames))
            row.update(psnr_median=round(statistics.median(ps), 2), psnr_min=round(min(ps), 2),
                       psnr_first=round(ps[0], 2), ssim_median=round(statistics.median(
                           M.ssim(ref[i], frames[i]) for i in range(0, n, max(1, n // 8))), 4),
                       frame_count_match=len(frames) == len(ref))
        mref = ref_frames(math_out) if math_out else None
        if ref is not None and mref is not None and mref.shape == ref.shape:
            row["ref_vs_math_ref_psnr_median"] = round(statistics.median(M.frame_psnrs(ref, mref)), 2)
        ref_wav = ref_out / f"{row['id']}.wav"
        if wav.is_file() and ref_wav.is_file():
            a, sr_a = read_pcm_wav(ref_wav)
            b, sr_b = read_pcm_wav(wav)
            if sr_a == sr_b:
                row["audio_snr_db"] = round(snr_db(a.mean(axis=1), b.mean(axis=1)), 2)
                row["audio_log_mel_db"] = round(M.log_mel_distance(a.mean(axis=1), b.mean(axis=1), sr_a), 3)
            if math_out and (math_out / f"{row['id']}.wav").is_file():
                m, sr_m = read_pcm_wav(math_out / f"{row['id']}.wav")
                if sr_m == sr_a:
                    row["ref_vs_math_ref_audio_log_mel_db"] = round(
                        M.log_mel_distance(a.mean(axis=1), m.mean(axis=1), sr_a), 3)

    def _reference_checks(self, sc: FamilyScore, rows: list[dict], n: int) -> None:
        medians = [r["psnr_median"] for r in rows if "psnr_median" in r]
        if not medians:
            sc.check("PSNR vs reference", None, "ERROR", "no served video comparable with its reference")
            return
        med = statistics.median(medians)
        sc.metrics["per_video_median_frame_psnr_db"] = M.distribution(medians)
        sc.check("median per-frame PSNR vs reference (dB)", round(med, 2),
                 band(med, lambda v: v >= 30, lambda v: v >= 20), ">=30 PASS (repo bar), >=20 WARN (provisional)")
        mism = [r["id"] for r in rows if r.get("frame_count_match") is False]
        sc.check("frame count equal to reference", n - len(mism), "PASS" if not mism else "FAIL", "all",
                 ", ".join(mism))
        sc.notes.append(f"The reference frames were compared after the served mp4's encode "
                        f"(libx264 {SERVER_X264.get(self.model_key) or 'defaults'}) and a decode.")
        spread = [r["ref_vs_math_ref_psnr_median"] for r in rows if "ref_vs_math_ref_psnr_median" in r]
        if spread:
            sc.metrics["reference_vs_math_sdpa_reference_psnr_db"] = M.distribution(spread)
            self.calibrated_verdict(sc, spread, med)
        mel = [r["audio_log_mel_db"] for r in rows if "audio_log_mel_db" in r]
        if mel:
            sc.metrics["audio_log_mel_distance_db"] = M.distribution(mel)
            sc.metrics["audio_snr_db"] = M.distribution([r["audio_snr_db"] for r in rows if "audio_snr_db" in r])
        mel_spread = [(r["audio_log_mel_db"], r["ref_vs_math_ref_audio_log_mel_db"]) for r in rows
                      if "audio_log_mel_db" in r and "ref_vs_math_ref_audio_log_mel_db" in r]
        if mel_spread:
            gap = statistics.median(a - b for a, b in mel_spread)
            sc.check("median audio log-mel distance to reference minus the reference's own (dB)", round(gap, 3),
                     band(gap, lambda v: v <= 1.0, lambda v: v <= 2.0),
                     "<=1 PASS, <=2 WARN (provisional); the reference's own is its math-SDPA copy's distance")
        elif mel:
            sc.notes.append(f"Audio log-mel distance to the reference: median {statistics.median(mel):.2f} dB, not "
                            "judged without the math-SDPA arm.")

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        geo = self.geometry()
        n = len(sset.items)
        fails = failed_items(runs[0])
        ref_out = self.ref_outputs(ref_dir)
        math_dir = self.math_reference_dir(ref_dir)
        math_out = math_dir / "outputs" if math_dir else None
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL",
                 f"all {n} requests return video and audio", "; ".join(f"{k}: {v[:80]}" for k, v in
                                                                     list(fails.items())[:3]))
        rows, bad_video, bad_audio, video_repro, audio_repro, sheet, labels = [], [], [], [], [], [], []
        cache: dict[Path, np.ndarray] = {}
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
            if ref_out:
                self._reference_row(row, frames, wav, ref_out, math_out, cache)
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
            ref = cache.get(ref_out / f"{it.id}.npy") if ref_out else None
            for f, tag in ((ref, "ref"), (frames, "out")):
                if f is not None:
                    sheet.append([f[0], f[len(f) // 2], f[-1]])
                    labels += [f"{tag} {it.id} first", "mid", "last"]
        sc.check("videos with the requested frames and size, none flat", n - len(bad_video),
                 "PASS" if not bad_video else "FAIL", f"{geo['num_frames']} frames of {geo['width']}x{geo['height']}",
                 ", ".join(bad_video[:8]))
        sc.check("audio tracks failing sanity", len(bad_audio),
                 band(len(bad_audio), lambda v: v == 0, lambda v: v <= 1),
                 "0 PASS, 1 WARN: stereo, length within 0.25 s of the clip, <= 95% silent, <= 1% clipped",
                 ", ".join(bad_audio[:8]))
        if ref_out:
            self._reference_checks(sc, rows, n)
            # With a reference the instances' agreement is context, as for the video families.
            if video_repro:
                sc.metrics["r1_r2_frame_psnr_min_db"] = M.distribution(video_repro)
            if audio_repro:
                sc.metrics["r1_r2_audio_snr_db"] = M.distribution(audio_repro)
        else:
            sc.notes.append(f"No reference in {ref_dir}.")
            sc.check("PSNR vs reference", None, "NO_REFERENCE", "needs the diffusers LTX-2.5 reference "
                     "(--make-reference, and --family-opt ref_sdpa=math for its spread)")
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
