"""Image generation: 20 prompts x 2 sizes at fixed seeds, vs the diffusers pipeline.

PSNR/SSIM per image plus a contact sheet. The 30 dB bar comes from cosmos3's test_t2i.py. The 40 dB
FLUX/Z-Image equivalence tests assume the same platform and SDPA, so they don't apply here.
"""

from __future__ import annotations

import io
import json
import statistics
from pathlib import Path

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, self_consistency_check, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


class ImageFamily(Family):
    name = "image"
    set_name = "image_v1"
    default_concurrency = 1
    hf_id = ""
    steps: int | None = None  # pinned on both sides; None = model default
    guidance: float | None = None  # sent to the server (models that read it)
    ref_extra: dict = {}  # extra pipeline kwargs for the reference only (e.g. a guidance the server ignores)
    generator_device = "cpu"  # where the serving code draws its initial noise from the request seed

    def model_kwargs(self, it) -> dict:
        kw = {"seed": it.params["seed"], "height": it.params["height"], "width": it.params["width"]}
        if self.steps is not None:
            kw["num_inference_steps"] = self.steps
        if self.guidance is not None:
            kw["guidance_scale"] = self.guidance
        return kw

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=it.prompt, output_modalities="image", model_kwargs=self.model_kwargs(it))
                for it in sset.items]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        from PIL import Image

        for r in responses:
            if r.ok and r.images:
                Image.open(io.BytesIO(r.images[-1])).convert("RGB").save(out_dir / f"{r.item_id}.png")

    # --- reference ------------------------------------------------------------------------------
    def reference_id(self) -> str:
        return f"diffusers-{self.hf_id.replace('/', '--')}" + (f"-s{self.steps}" if self.steps else "")

    def reference_description(self) -> str:
        return (f"diffusers pipeline for {self.hf_id}, bf16, {self.generator_device}-generator seed, "
                f"steps={self.steps or 'default'} {self.ref_extra or ''}")

    def load_pipeline(self, device: str):
        import torch
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(self.hf_id, torch_dtype=torch.bfloat16).to(device)

    def pipeline_kwargs(self, it) -> dict:
        kw = {"height": it.params["height"], "width": it.params["width"], **self.ref_extra}
        if self.steps is not None:
            kw["num_inference_steps"] = self.steps
        if self.guidance is not None:
            kw["guidance_scale"] = self.guidance
        return kw

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        import torch

        pipe = self.load_pipeline(device)
        pipe.set_progress_bar_config(disable=True)
        for it in sset.items:
            gen = torch.Generator(self.generator_device if self.generator_device == "cpu" else device)
            gen.manual_seed(int(it.params["seed"]))
            with torch.inference_mode():
                img = pipe(prompt=it.prompt, generator=gen, **self.pipeline_kwargs(it)).images[0]
            img.save(ref_dir / "outputs" / f"{it.id}.png")
            print(f"  ref {it.id}", flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name,
                   pipeline=type(pipe).__name__)

    # --- scoring --------------------------------------------------------------------------------
    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        ref_out = self.ref_outputs(ref_dir)
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return an image",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        psnrs, ssims, sheet_rows, labels, wrong_size = [], [], [], [], []
        for it in sset.items:
            p = runs[0] / "outputs" / f"{it.id}.png"
            out = M.load_image(p) if p.is_file() else None
            ref = M.load_image(ref_out / f"{it.id}.png") if ref_out and (ref_out / f"{it.id}.png").is_file() else None
            row = {"id": it.id, "prompt": it.prompt[:60]}
            if out is not None and out.shape[:2] != (it.params["height"], it.params["width"]):
                wrong_size.append(it.id)
            if out is not None and ref is not None and out.shape == ref.shape:
                row["psnr"] = round(M.psnr(ref, out), 2)
                row["ssim"] = round(M.ssim(ref, out), 4)
                psnrs.append(row["psnr"])
                ssims.append(row["ssim"])
            if len(runs) > 1:
                p2 = runs[1] / "outputs" / f"{it.id}.png"
                if out is not None and p2.is_file():
                    o2 = M.load_image(p2)
                    row["psnr_r1_r2"] = round(M.psnr(out, o2), 2) if o2.shape == out.shape else None
            sc.rows.append(row)
            sheet_rows.append([ref, out])
            labels += [f"ref {it.id}", f"out {row.get('psnr', '-')} dB"]
        sc.check("images with the requested size", n - len(fails) - len(wrong_size),
                 "PASS" if not wrong_size else "FAIL", "all", ", ".join(wrong_size[:6]))
        flat = [it.id for it, (_, out) in zip(sset.items, sheet_rows, strict=True) if out is not None and out.std() < 2]
        sc.check("flat images (pixel std < 2)", len(flat), "PASS" if not flat else "FAIL", "0", ", ".join(flat[:8]))
        if psnrs:
            sc.metrics["psnr_db"] = M.distribution(psnrs)
            sc.metrics["ssim"] = M.distribution(ssims)
            med = statistics.median(psnrs)
            sc.check("median PSNR vs reference (dB)", round(med, 2), band(med, lambda v: v >= 30, lambda v: v >= 20),
                     ">=30 PASS (repo bar), >=20 WARN (provisional)")
            low = [r["id"] for r in sc.rows if r.get("psnr", 99) < 20]
            sc.check("images below 20 dB", len(low), band(len(low), lambda v: v == 0, lambda v: v <= 2),
                     "0 PASS, <=2 WARN (a different image, not drift)", ", ".join(low[:8]))
        elif not self.self_consistency_only:
            sc.notes.append(f"No reference images in {ref_dir}.")
            sc.check("PSNR vs reference", None, "NO_REFERENCE", "needs diffusers reference images")
        repro = [r["psnr_r1_r2"] for r in sc.rows if r.get("psnr_r1_r2") is not None]
        if repro:
            sc.metrics["psnr_r1_vs_r2_db"] = M.distribution(repro)
        if self.self_consistency_only:
            self_consistency_check(sc, repro, "same-seed min PSNR, two server instances (dB)", len(runs),
                                   lambda v: v >= 40, lambda v: v >= 25,
                                   ">=40 PASS (inf = byte-identical), >=25 WARN (provisional)")
        sheet = work / "contact_sheet.png"
        M.contact_sheet(sheet_rows, labels, sheet)
        sc.artifacts += [str(sheet), str(runs[0] / "outputs")]
        (work / "image_scores.json").write_text(json.dumps(sc.rows, indent=1))
        return sc


class NoReferenceImage(ImageFamily):
    """Image models with no reference we can run here (BAGEL). Checks size, flat images and
    same-seed repeatability across two server instances.
    """

    default_restarts = 2
    self_consistency_only = True

    def reference_id(self) -> str:
        return f"{self.model_key}-no-reference"

    def reference_description(self) -> str:
        return "none runs on ROCm here, official BAGEL needs flash_attn and uses the global RNG"

    def make_reference(self, sset, ref_dir, device):
        raise NotImplementedError(self.reference_description())
