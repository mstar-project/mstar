"""MiniMax-H3-Image reference (diffusers 875373ac modular blocks) on two GPUs, with per-step recording.

The text encoder runs as its own pipeline on --te-gpu, the rest (VAE encode, denoise, decode) on --dit-gpu;
`before_encode` (the deterministic resize) runs in both. Writes, per run: the PNG, and with --record the
prompt embeds, token tags, reference latents, the initial packed rows, every step's input rows and velocity
and the final rows, as a torch file.

    python oracle.py --task t2i --prompt-file p.txt --size 2048x2048 --seed 42 --out DIR [--record] [--runs 2]
    python oracle.py --task edit --prompt-file p.txt --image a.png --image b.png --size 2048x2048 ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = "MiniMaxAI/MiniMax-H3-Image"
RECORD = [False]
JOB_DEFAULTS = {"task": "t2i", "prompt_file": None, "images": [], "size": "2048x2048", "seed": 42,
                "min_area": 2048 * 2048, "attn": "native", "runs": 1, "record": False, "tag": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True, help="JSON list of jobs (keys: see JOB_DEFAULTS)")
    ap.add_argument("--te-gpu", type=int, default=0)
    ap.add_argument("--dit-gpu", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = [{**JOB_DEFAULTS, **job} for job in json.loads(args.jobs.read_text())]

    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3_image import denoise as dn

    args.out.mkdir(parents=True, exist_ok=True)
    te_dev, dit_dev = torch.device(f"cuda:{args.te_gpu}"), torch.device(f"cuda:{args.dit_gpu}")

    full = ModularPipeline.from_pretrained(REPO)
    blocks = full.blocks
    te_blocks = blocks.sub_blocks.pop("text_encoder")
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    te_seq = SequentialPipelineBlocks.from_blocks_dict(
        {"before_encode": blocks.sub_blocks["before_encode"], "text_encoder": te_blocks}
    )
    te_pipe = te_seq.init_pipeline(REPO)
    te_pipe.load_components(names=["text_encoder", "tokenizer", "processor"], dtype=torch.bfloat16)
    te_pipe.text_encoder.to(te_dev)
    main_pipe = blocks.init_pipeline(REPO)
    main_pipe.load_components(names=["transformer", "vae", "scheduler"], dtype=torch.bfloat16)
    main_pipe.transformer.to(dit_dev)
    main_pipe.vae.to(dit_dev)
    print("dtypes", main_pipe.transformer.dtype, main_pipe.vae.dtype, te_pipe.text_encoder.dtype, flush=True)

    rec: dict = {}
    orig_den, orig_upd = dn.MiniMaxH3ImageLoopDenoiser.__call__, dn.MiniMaxH3ImageLoopSchedulerStep.__call__

    def den(self, components, bs, i, t):
        if RECORD[0]:
            rec.setdefault("step_in", []).append(bs.latents.detach().float().cpu().clone())
        out = orig_den(self, components, bs, i, t)
        if RECORD[0]:
            rec.setdefault("velocity", []).append(bs.noise_pred.detach().float().cpu().clone())
            rec.setdefault("plan", []).append(tuple(x.detach().cpu().clone() for x in bs.row_timestep_plan[i]))
        return out

    def upd(self, components, bs, i, t):
        out = orig_upd(self, components, bs, i, t)
        if RECORD[0]:
            rec.setdefault("step_out", []).append(bs.latents.detach().float().cpu().clone())
        return out

    dn.MiniMaxH3ImageLoopDenoiser.__call__ = den
    dn.MiniMaxH3ImageLoopSchedulerStep.__call__ = upd

    for job in jobs:
        run_job(job, te_pipe, main_pipe, rec, te_dev, dit_dev, args.out)
    print("ALL JOBS DONE", flush=True)


def run_job(job, te_pipe, main_pipe, rec, te_dev, dit_dev, out: Path) -> None:
    from PIL import Image

    width, height = (int(v) for v in job["size"].lower().split("x"))
    prompt = Path(job["prompt_file"]).read_text().strip()
    images = [Image.open(p).convert("RGB") for p in job["images"]]
    main_pipe.transformer.set_attention_backend(job["attn"])
    tag = job["tag"] or job["task"]
    RECORD[0] = bool(job["record"])
    for d in (te_dev, dit_dev):
        torch.cuda.reset_peak_memory_stats(d)
    timings = []
    for run in range(job["runs"]):
        rec.clear()
        kw = dict(prompt=prompt, height=height, width=width, min_generation_area=job["min_area"])
        if images:
            kw["image"] = images if len(images) > 1 else images[0]
        torch.cuda.synchronize(te_dev)
        torch.cuda.synchronize(dit_dev)
        t0 = time.perf_counter()
        te_state = te_pipe(**kw, output=["prompt_embeds", "text_token_tags"])
        prompt_embeds = te_state["prompt_embeds"].to(dit_dev)
        torch.cuda.synchronize(te_dev)
        t1 = time.perf_counter()
        generator = torch.Generator().manual_seed(job["seed"])
        state = main_pipe(
            **kw, prompt_embeds=prompt_embeds, text_token_tags=te_state["text_token_tags"], generator=generator,
            output=None,
        )
        torch.cuda.synchronize(dit_dev)
        t2 = time.perf_counter()
        img = state.get("images")[0]
        timings.append({"te_s": t1 - t0, "rest_s": t2 - t1, "total_s": t2 - t0})
        print(f"run {run}: te {t1 - t0:.2f}s rest {t2 - t1:.2f}s total {t2 - t0:.2f}s", flush=True)
        name = f"{tag}_seed{job['seed']}_run{run}"
        img.save(out / f"{name}.png")
        if job["record"]:
            rec["prompt_embeds"] = te_state["prompt_embeds"].detach().float().cpu()
            rec["text_token_tags"] = te_state["text_token_tags"].cpu()
            ref = state.get("reference_latents")
            if ref is not None:
                rec["reference_latents"] = [r.detach().float().cpu() for r in ref]
            rec["num_reference_rows"] = state.get("num_reference_rows")
            rec["latent_hw"] = (state.get("latent_height"), state.get("latent_width"))
            rec["position_ids"] = state.get("position_ids").detach().cpu()
            rec["token_tags"] = state.get("token_tags").detach().cpu()
            rec["timesteps"] = state.get("timesteps").detach().cpu()
            rec["final_latents"] = state.get("latents").detach().float().cpu()
            torch.save(rec, out / f"{name}.pt")
    peak = {str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in (te_dev, dit_dev)}
    meta = {"job": job, "timings": timings, "peak_gib": peak,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dit_dev)}
    (out / f"{tag}_seed{job['seed']}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
