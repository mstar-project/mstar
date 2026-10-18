"""Video generation: 5 prompts at fixed seeds (i2v models also get a first frame).

Per-frame PSNR/SSIM vs diffusers, judged on the median frame against the repo's 30 dB bar
(cosmos3 test_video.py), plus a contact sheet of first/middle/last frames.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, self_consistency_check, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


def read_video(path: Path) -> np.ndarray:
    """Decode a video file to [T, H, W, 3] uint8 frames with PyAV."""
    import av

    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    from benchmark.correctness.sets import _write_mp4

    _write_mp4(np.asarray(frames, dtype=np.uint8), path, fps)


class VideoFamily(Family):
    name = "video"
    set_name = "video_v1"
    default_concurrency = 1
    request_timeout_s = 3600.0
    hf_id = ""
    image_to_video = False
    height, width, num_frames, steps, fps = 480, 832, 33, 20, 16
    guidance: float | None = None
    negative_prompt: str | None = None
    generator_device = "cpu"
    request_extra: dict = {}  # extra model_kwargs for the server only
    ref_call_kwargs: dict = {}  # extra pipeline call kwargs for the reference only

    def model_kwargs(self, it) -> dict:
        kw = {"seed": it.params["seed"], "height": self.height, "width": self.width, "num_frames": self.num_frames,
              "num_inference_steps": self.steps}
        if self.guidance is not None:
            kw["guidance_scale"] = self.guidance
        if self.negative_prompt is not None:
            kw["negative_prompt"] = self.negative_prompt
        return {**kw, **self.request_extra}

    def requests(self, sset: SampleSet) -> list[Request]:
        reqs = []
        for it in sset.items:
            files = [sset.path(it.files["image"])] if self.image_to_video else []
            reqs.append(Request(it.id, text=it.prompt, files=files, output_modalities="video",
                                input_modalities="image,text" if files else None, model_kwargs=self.model_kwargs(it)))
        return reqs

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            if r.ok and r.video:
                (out_dir / f"{r.item_id}.mp4").write_bytes(bytes(r.video))

    def reference_id(self) -> str:
        return (f"diffusers-{self.hf_id.replace('/', '--')}-{self.height}x{self.width}x{self.num_frames}"
                f"-s{self.steps}" + ("-i2v" if self.image_to_video else ""))

    def reference_description(self) -> str:
        return (f"diffusers pipeline for {self.hf_id}, bf16, {self.generator_device}-generator seed, "
                f"{self.height}x{self.width}, "
                f"{self.num_frames} frames, {self.steps} steps")

    def load_pipeline(self, device: str):
        import torch
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(self.hf_id, torch_dtype=torch.bfloat16).to(device)

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        import torch
        from PIL import Image

        pipe = self.load_pipeline(device)
        pipe.set_progress_bar_config(disable=True)
        for it in sset.items:
            kw = {"height": self.height, "width": self.width, "num_frames": self.num_frames,
                  "num_inference_steps": self.steps, "output_type": "np"}
            if self.guidance is not None:
                kw["guidance_scale"] = self.guidance
            if self.negative_prompt is not None:
                kw["negative_prompt"] = self.negative_prompt
            if self.image_to_video:
                kw["image"] = Image.open(sset.path(it.files["image"])).convert("RGB")
            kw.update(self.ref_call_kwargs)
            with torch.inference_mode():
                gen = torch.Generator(self.generator_device if self.generator_device == "cpu" else device)
                frames = self.reference_frames(pipe(prompt=it.prompt, generator=gen.manual_seed(int(it.params["seed"])),
                                                    **kw))
            frames = (np.clip(np.asarray(frames), 0, 1) * 255).round().astype(np.uint8)
            np.save(ref_dir / "outputs" / f"{it.id}.npy", frames)
            write_video(frames, ref_dir / "outputs" / f"{it.id}.mp4", self.fps)
            print(f"  ref {it.id}: {frames.shape}", flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   pipeline=type(pipe).__name__)

    @staticmethod
    def reference_frames(out) -> np.ndarray:
        """[T, H, W, 3] floats in [0, 1] from the pipeline output (output_type="np")."""
        return out.frames[0]

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        ref_out = self.ref_outputs(ref_dir)
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return a video",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        medians, sheet, labels = [], [], []
        for it in sset.items:
            p = runs[0] / "outputs" / f"{it.id}.mp4"
            row = {"id": it.id, "prompt": it.prompt[:60]}
            out = read_video(p) if p.is_file() else None
            ref = None
            if ref_out and (ref_out / f"{it.id}.npy").is_file():
                ref = np.load(ref_out / f"{it.id}.npy")
            if out is not None:
                row["frames"] = len(out)
                row["dims"] = "x".join(map(str, out.shape[1:3]))
            if len(runs) > 1 and out is not None and (runs[1] / "outputs" / f"{it.id}.mp4").is_file():
                o2 = read_video(runs[1] / "outputs" / f"{it.id}.mp4")
                if o2.shape == out.shape:
                    row["psnr_r1_r2_min"] = min(M.frame_psnrs(out, o2))
            if out is not None and ref is not None and out.shape[1:] == ref.shape[1:]:
                ps = M.frame_psnrs(ref, out)
                row.update(psnr_median=round(statistics.median(ps), 2), psnr_min=round(min(ps), 2),
                           psnr_first=round(ps[0], 2), ssim_median=round(statistics.median(
                               M.ssim(ref[i], out[i]) for i in range(0, min(len(ref), len(out)),
                                                                      max(1, len(out) // 8))), 4),
                           frame_count_match=len(out) == len(ref))
                medians.append(row["psnr_median"])
            sc.rows.append(row)
            for frames, tag in ((ref, "ref"), (out, "out")):
                if frames is None:
                    sheet.append([None, None, None])
                else:
                    sheet.append([frames[0], frames[len(frames) // 2], frames[-1]])
                labels += [f"{tag} {it.id} first", "mid", "last"]
        if medians:
            med = statistics.median(medians)
            sc.metrics["per_video_median_frame_psnr_db"] = M.distribution(medians)
            sc.check("median per-frame PSNR vs reference (dB)", round(med, 2),
                     band(med, lambda v: v >= 30, lambda v: v >= 20), ">=30 PASS (repo bar), >=20 WARN (provisional)")
            mism = [r["id"] for r in sc.rows if r.get("frame_count_match") is False]
            sc.check("frame count equal to reference", n - len(mism), "PASS" if not mism else "FAIL", "all",
                     ", ".join(mism))
        elif not self.self_consistency_only:
            sc.notes.append(f"No comparable reference videos in {ref_dir}.")
            sc.check("PSNR vs reference", None, "NO_REFERENCE", "needs diffusers reference videos")
        if self.self_consistency_only:
            repro = [r["psnr_r1_r2_min"] for r in sc.rows if r.get("psnr_r1_r2_min") is not None]
            self_consistency_check(sc, repro, "same-seed min frame PSNR, two server instances (dB)", len(runs),
                                   lambda v: v >= 40, lambda v: v >= 25,
                                   ">=40 PASS (inf = identical after the same lossy encode), >=25 WARN (provisional)")
        M.contact_sheet(sheet, labels, work / "contact_sheet.png")
        sc.artifacts += [str(work / "contact_sheet.png"), str(runs[0] / "outputs")]
        (work / "video_scores.json").write_text(json.dumps(sc.rows, indent=1))
        return sc


class NoReferenceVideo(VideoFamily):
    """Video models with no reference (Cosmos3 4-step distilled). Checks frames and same-seed
    repeatability across two server instances.
    """

    default_restarts = 2
    self_consistency_only = True
    no_reference_reason = ""

    def reference_id(self) -> str:
        return f"{self.model_key}-no-reference"

    def reference_description(self) -> str:
        return self.no_reference_reason or "no reference for this checkpoint"

    def make_reference(self, sset, ref_dir, device):
        raise NotImplementedError(self.reference_description())
