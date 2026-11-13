# Step cache verification harness

Measures every step-cache setting (`mstar/model/components/diffusion/step_cache.py`) against dense on the same
prompts and seeds, and answers one question: is the default on the speed/quality frontier?

    python -m benchmark.step_cache.run --backend inproc --model wan22_ti2v_5b \
        --size 832x480 --frames 33 --steps 30 --prompts 16 --out runs/wan5b

`--list-methods` prints a model's grid; `--methods mag,int3` keeps the methods whose names start with a prefix
(dense always runs). `--methods-file plans.json` adds methods from a JSON list of
`{"name", "step_cache", "label"?, "is_default"?}`, such as the plans a search picked, and `--methods none` leaves
the grid out. A run writes `records.jsonl` as clips finish, then `report.md`, `summary.json` and `pareto.svg`.
`--resume` continues the run in `--out`: clips already there are kept, whichever invocation made them, and the
report covers all of them, so a methods file can join an earlier sweep.

`--model tiny` is Wan's DiT at toy size on the CPU with random weights (in-process backend only): a dry run of
the harness, not a measurement.

## What it reports, per method

- **s/video and speedup** over dense, steps that ran blocks, and branch forwards (block-stack runs in units of
  one guidance branch through every block; dense with guidance is 2 per step).
- **PSNR, SSIM, LPIPS** per frame against the dense clip: mean over frames and prompts, and the worst frame.
  SSIM is the gaussian-window form; `--ssim-window uniform` is scikit-image's default, which the quality tables
  in `docs/models.rst` used.
- **CLIP delta**: the clip's CLIP text-video similarity minus the dense clip's, so a drop means the clip shows
  the prompt less well, which a pixel metric cannot tell from a harmless change of detail.
- **Flicker**: VBench's temporal-flickering score next to the dense clip's; flicker excess (how much more the
  clip changes back and forth between frames than dense, 0 = as steady); and the PSNR between the two clips'
  frame differences (did the cache change the motion).

LPIPS needs the `lpips` package and CLIP the `openai/clip-vit-base-patch32` weights; without them the column is
`-` and the report says why. `--clip-model DIR` reads the weights from a snapshot directory when the hub cache
in use holds only the model's; `--no-optional-metrics` skips both.

## Backends

- `inproc` builds the model once and swaps the policy between runs, so a wide grid costs one load. Use it to find
  the frontier.
- `server` boots an M* server per method (a method is a serving config) on the base yaml with the method's
  `step_cache` under `model_kwargs`, warms each shape up with one untimed request, and times requests from the
  client. Use it to confirm the frontier's points as served: `--methods` picks them. `--url URL --running NAME`
  uses a server already booted with that method. `--server-arg --python-frontend` serves without `mstar-server`.

The harness claims no GPU and sets no environment: export `CUDA_VISIBLE_DEVICES`, `MSTAR_COMPILE_LEVEL` and the
caches yourself. Give every method's server the same `TORCHINDUCTOR_CACHE_DIR`: two boots that compile into
different caches can pick different kernels, and their dense clips then differ.

## Reading the plot

`pareto.svg` puts speedup across and the quality metric up (`--metric psnr|ssim|lpips`). The line joins the
frontier: the methods no other beats at both. The default is the diamond. It passes while no method is at least
2% faster and 0.3 dB better at once (the margins absorb run-to-run noise); when it fails the report names the
methods that beat it. Hover a point for its numbers; the report's table has all of them.

## Regression mode

    python -m benchmark.step_cache.run --ci --backend inproc --model wan22_ti2v_5b --out runs/ci

Three prompts at 832x480, 17 frames, 20 steps: dense, the default, and a policy that skips nothing. It exits 1
when the no-skip policy is not bit-identical to dense (the cache's compute path changed an output), or when the
default falls under `ci_baseline/<model>.json` by more than its tolerances (0.5 dB, 0.01 SSIM, 10% of the
speedup). `--write-baseline` writes that file from a run on a tree known good, on the hardware the check runs on.

## Rules for the numbers

These are lossy, opt-in knobs. A table of them states the hardware, torch and CUDA versions, the size, frames
and steps; compares each engine against its own dense output on the same prompts and seeds (two engines draw
different noise from one seed); uses at least 8 prompts; and comes from warm runs in one window on one GPU.
