"""Video generation: 5 prompts at fixed seeds (i2v models also get a first frame).

Per-frame PSNR/SSIM vs diffusers, judged on the median frame against the repo's 30 dB bar
(cosmos3 test_video.py), plus a contact sheet of first/middle/last frames. A served video is an
H.264 mp4, so the reference frames go through the same encode (SERVER_X264) before they are
compared with it. The encode alone caps PSNR at 34-41 dB and darkens frames by about a luma level.
"""

from __future__ import annotations

import json
import statistics
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


# x264 options of each model's served mp4 (libx264, yuv420p), by model key. Cosmos3Model.postprocess uses
# CRF 18, preset ultrafast (COSMOS3_X264_PRESET can change it) and threads 0. Wan22Model.postprocess and
# LTX2_5Model.postprocess keep libx264's defaults, CRF 23 and preset medium.
_COSMOS3_X264 = {"crf": "18", "preset": "ultrafast", "threads": "0"}
SERVER_X264: dict[str, dict[str, str]] = {
    "cosmos3": _COSMOS3_X264,
    "cosmos3_edge": _COSMOS3_X264,
    "cosmos3_super": _COSMOS3_X264,
    "cosmos3_super_i2v_4step": _COSMOS3_X264,
    "wan22": {},
    "ltx2_5": {},
}


def server_mp4(frames: np.ndarray, options: dict[str, str], fps: int) -> bytes:
    """``frames`` [T, H, W, 3] uint8 as an mp4 the way a server encodes them (libx264, yuv420p,
    ``options``)."""
    import io
    from fractions import Fraction

    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        s.height, s.width = frames.shape[1:3]
        s.pix_fmt = "yuv420p"
        s.options = dict(options)
        for f in np.asarray(frames, dtype=np.uint8):
            for packet in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    return buf.getvalue()


def server_mp4_roundtrip(frames: np.ndarray, options: dict[str, str], fps: int) -> np.ndarray:
    """``frames`` through ``server_mp4`` and decoded again with PyAV, as the runner reads a served video."""
    import io

    import av

    with av.open(io.BytesIO(server_mp4(frames, options, fps))) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])


def as_served(frames: np.ndarray, model_key: str, fps: float) -> np.ndarray:
    """Reference frames as a served video carries them: through the model's mp4 encode (SERVER_X264),
    or raw for a model with no entry there."""
    options = SERVER_X264.get(model_key)
    return frames if options is None else server_mp4_roundtrip(frames, options, fps)


class VideoFamily(MathSdpaReference, Family):
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
    # False when the served request samples differently from the reference (a windowed AR walk):
    # PSNR against it is reported for information, not judged.
    reference_is_gate = True

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
                f"-s{self.steps}" + ("-i2v" if self.image_to_video else "") + self.arm_suffix())

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
        anchor_moves = []
        if self.ref_anchor_fp32:
            if not self.image_to_video:
                raise ValueError("ref_anchor=fp32 only applies to image-to-video references")
            self.fp32_anchor(pipe, anchor_moves)
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
            with torch.inference_mode(), self.sdpa_context():
                gen = torch.Generator(self.generator_device if self.generator_device == "cpu" else device)
                frames = self.reference_frames(pipe(prompt=it.prompt, generator=gen.manual_seed(int(it.params["seed"])),
                                                    **kw))
            frames = (np.clip(np.asarray(frames), 0, 1) * 255).round().astype(np.uint8)
            np.save(ref_dir / "outputs" / f"{it.id}.npy", frames)
            write_video(frames, ref_dir / "outputs" / f"{it.id}.mp4", self.fps)
            print(f"  ref {it.id}: {frames.shape}", flush=True)
        extra = {"anchor_rel_change_vs_bf16": [round(m, 6) for m in anchor_moves]} if anchor_moves else {}
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   pipeline=type(pipe).__name__, **extra)

    def fp32_anchor(self, pipe, moves: list[float]) -> None:
        """Make the pipeline encode the conditioning frame in fp32 and append, per encode, how far that
        moves the anchor latent from the bf16 encode (relative L2)."""
        raise NotImplementedError(f"{type(self).__name__} has no fp32-anchor arm")

    @staticmethod
    def reference_frames(out) -> np.ndarray:
        """[T, H, W, 3] floats in [0, 1] from the pipeline output (output_type="np")."""
        return out.frames[0]

    def comparable(self, path: Path, cache: dict) -> np.ndarray:
        """A reference arm's saved frames as a served video would carry them: through the model's mp4
        encode (SERVER_X264), or raw for a model with no entry there."""
        if path not in cache:
            cache[path] = as_served(np.load(path), self.model_key, self.fps)
        return cache[path]

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        ref_out = self.ref_outputs(ref_dir)
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return a video",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        medians, sheet, labels = [], [], []
        encoded: dict[Path, np.ndarray] = {}
        for it in sset.items:
            p = runs[0] / "outputs" / f"{it.id}.mp4"
            row = {"id": it.id, "prompt": it.prompt[:60]}
            out = read_video(p) if p.is_file() else None
            ref = None
            if ref_out and (ref_out / f"{it.id}.npy").is_file():
                ref = self.comparable(ref_out / f"{it.id}.npy", encoded)
            if out is not None:
                row["frames"] = len(out)
                row["dims"] = "x".join(map(str, out.shape[1:3]))
            o2 = None
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
            if o2 is not None:  # last, so the report's per-item table keeps the reference columns
                row["frames_identical_r1_r2"] = M.frames_equal(out, o2)
            sc.rows.append(row)
            for frames, tag in ((ref, "ref"), (out, "out")):
                if frames is None:
                    sheet.append([None, None, None])
                else:
                    sheet.append([frames[0], frames[len(frames) // 2], frames[-1]])
                labels += [f"{tag} {it.id} first", "mid", "last"]
        # Per video, how far the reference moves from itself: with math SDPA, and for i2v with the conditioning
        # frame encoded in fp32. The floor takes the larger move per video, then the median over videos.
        arms = {"math": (self.math_reference_dir(ref_dir), "reference_vs_math_sdpa_reference_psnr_db",
                         "with math SDPA"),
                "anchor": (self.anchor_reference_dir(ref_dir) if self.image_to_video else None,
                           "reference_vs_fp32_anchor_reference_psnr_db", "with the conditioning frame encoded in fp32")}
        moves: dict[str, list[float]] = {}
        for arm, (arm_dir, metric, how) in arms.items():
            if not (ref_out and arm_dir):
                continue
            for r in sc.rows:
                a, b = ref_out / f"{r['id']}.npy", arm_dir / "outputs" / f"{r['id']}.npy"
                if a.is_file() and b.is_file():
                    psnr = round(statistics.median(M.frame_psnrs(self.comparable(a, encoded),
                                                                 self.comparable(b, encoded))), 2)
                    r[f"ref_vs_{arm}_ref_psnr_median"] = psnr
                    moves.setdefault(r["id"], []).append(psnr)
            sc.metrics[metric] = M.distribution([r[f"ref_vs_{arm}_ref_psnr_median"] for r in sc.rows
                                                 if f"ref_vs_{arm}_ref_psnr_median" in r])
            sc.notes.append(f"The reference against itself {how}, per-video median frame PSNR: median "
                            f"{sc.metrics[metric].get('median')} dB.")
            anchor_moves = json.loads((arm_dir / "meta.json").read_text()).get("anchor_rel_change_vs_bf16")
            if arm == "anchor" and anchor_moves:
                sc.metrics["fp32_anchor_rel_change_vs_bf16"] = M.distribution(anchor_moves)
                sc.notes.append(f"The fp32 encode moved the anchor latent by a relative {max(anchor_moves):.1e} at "
                                "most from the bf16 encode.")
        if moves:
            used = [arm for arm in arms if any(f"ref_vs_{arm}_ref_psnr_median" in r for r in sc.rows)]
            basis = {("math",): "math-SDPA spread", ("anchor",): "fp32-anchor spread"}.get(
                tuple(used), "math-SDPA and fp32-anchor spreads (per video the larger move)")
            served = statistics.median(medians) if medians else None
            if self.reference_is_gate:
                self.calibrated_verdict(sc, [min(v) for v in moves.values()], served, basis)
        if ref_out and SERVER_X264.get(self.model_key) is not None:
            sc.notes.append(f"The reference frames were compared after the served mp4's encode "
                            f"(libx264 {SERVER_X264[self.model_key] or 'defaults'}) and a decode.")
        if medians:
            med = statistics.median(medians)
            sc.metrics["per_video_median_frame_psnr_db"] = M.distribution(medians)
            if self.reference_is_gate:
                sc.check("median per-frame PSNR vs reference (dB)", round(med, 2),
                         band(med, lambda v: v >= 30, lambda v: v >= 20),
                         ">=30 PASS (repo bar), >=20 WARN (provisional)")
            else:
                sc.notes.append(f"Median per-frame PSNR against {self.reference_description()}: {med:.2f} dB, "
                                "for information only.")
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
                                   ">=40 PASS (inf = identical decoded frames), >=25 WARN (provisional)")
        frame_identity(sc, sc.rows, encoded=True)
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
