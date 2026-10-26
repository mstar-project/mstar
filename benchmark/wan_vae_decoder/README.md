# Wan VAE streaming-decode benchmark

Measures the Wan VAE decoder the way a streaming video rollout uses it: the
latent stream arrives in chunks, and each chunk is decoded as soon as it lands.
This directory reproduces the headline: the change ladder through M*'s own
streaming decoder node (`Cosmos3VAEDecoderARSubmodule`), on the Wan 2.1 and
Wan 2.2 VAEs.

The full tooling is on the results branch,
[`perf/wan-vae-decode-results`](https://github.com/arun-elr/mstar/tree/perf/wan-vae-decode-results/benchmark/wan_vae_decoder),
at the same paths: the prototype harness and ablations (`run_ablation.sh`),
output-equivalence checks (`quality.py`), kernel profiles, summaries and
charts, served-window and end-to-end drivers, and the vLLM-Omni / SGLang
harnesses. It also holds the data: B200 and H200 CSVs, `RESULTS.md`, charts
and samples.

The full data stays archived on that branch after this change merges; nothing
here depends on it, and [Reproduce](#reproduce) below gets the headline from
this directory alone.

## Method

| | |
|---|---|
| VAE | Wan 2.1 (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 16 latent channels, 8× spatial); Wan 2.2 (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, 48 channels, 16× spatial) |
| Chunk | 3 latent frames → 12 RGB frames at 480×832, batch 1, bf16 |
| Timed | decoder + conversion to uint8 frames, CUDA events around each chunk, synchronized per chunk |
| Stream | first latent untimed, 10 warm-up chunks untimed (absorbs `torch.compile` and cuDNN autotuning), 75 timed chunks; the stream is replayed twice from an empty cache and the 150 samples are pooled |
| Reported | median ms per chunk; decoder fps = 12 / median; peak `max_memory_allocated` over the steady chunks |
| Latents | a natural video clip, cropped to 832×480, ping-ponged to 1,021 frames and encoded with the VAE's own encoder: 256 latent frames = 1 + 85 chunks. `prepare_latents.py --video PATH` encodes any clip; `--random` is the fallback. Timing does not depend on latent values; PSNR does. |
| Processes | one variant per process, fresh Inductor/Triton caches |
| Accuracy | PSNR against diffusers fp32, eager, **TF32 off**, over the first 10 steady chunks. M*'s node variants return uint8 frames and take the normalized latents the DiT hands the node. |

`mstar_step0` is the stateless context re-decode (before this change); step 1
adds the warm cache, step 2 channels-last with conv-internal padding.
`mstar_warm` is the node at this change's defaults (one latent per decoder
call, `cudnn.benchmark` off); `mstar_warm_lpc0` is the opt-in one decoder call
per window and `mstar_warm_bench` the opt-in `cudnn.benchmark`. `bench_wan_decoder.py`'s docstring lists every variant.

## Reproduce

Needs a CUDA GPU, the M* environment and the two VAEs in the local Hugging Face
cache (`hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*"`, likewise
for Wan 2.2).

```bash
cd benchmark/wan_vae_decoder
GPU=0 VIDEO=/path/to/clip.mp4 WAN22=1 bash run_all.sh   # prints ms per chunk, peak GiB, PSNR per variant
python bench_wan_decoder.py --variant mstar_warm --latents runs/<run>/wan21/latents.pt \
  --ref runs/<run>/wan21/ref_fp32_outputs.pt --out runs/<run>/wan21   # one variant by hand
```

The tables in the PR are 3-run means (fresh process each, rotated order) from
`run_ablation.sh ladder` on the results branch.
