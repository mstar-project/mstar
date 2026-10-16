---
name: diffusion-scaffold
description: Building an image/video flow or diffusion model on M*'s DiT scaffold (mstar/model/components/diffusion/, DenoiseLoopSubmodule) — bucket keys, schedules and seeding, multi-stage recipes, ragged attention spans, compile and per-bucket CUDA graphs, and the async overshoot veto. Use when porting a DiT-style model, writing a DenoiseLoopSubmodule subclass, or changing the scaffold itself.
---

# The DiT scaffold

`mstar/model/components/diffusion/` holds what every text → flow transformer → VAE model shares: sigma schedules and the Euler step (`flow_match`), multi-axis RoPE, joint attention, compile and decode wrappers, weight streaming, and the `DenoiseLoopSubmodule` that runs as the body of a model's `Loop("denoise_loop", dit)`. `flux2_klein` and `z_image` are built on it and are the references to read. `cosmos3` and `wan22` predate it; don't copy their single-request, self-compiled, uncaptured loop.

Read the module docstring of `denoise_loop.py` first. Its two-column table is the contract: on the left what the base does (step index from the engine's loop counter, per-request schedule and seeded noise, overshoot veto, equal-key batching, stop at the request's own step count, per-key graph buckets, the ragged step declaration), on the right the hooks a model implements. This skill covers the decisions behind those hooks and what has gone wrong.

Read the [add-mstar-model skill](../add-mstar-model/SKILL.md) for the port as a whole; this one is about the `dit` node.

## The bucket key

`bucket_key_for(fwd_info)` returns everything that has to match for two requests to share a forward. It is also the CUDA-graph bucket key and the key of the per-bucket layout cache (`build_layout`: rotary tables and the like, built once and never evicted because a captured graph reads them at fixed addresses).

- **It is not just shape.** The latent grid and the text length belong in it, but so does anything that changes the forward: whether guidance is on, the conditioning layout, which recipe stage is running.
- **Too fine a key means no batching, too coarse means wrong output.** Two requests that differ in something the key leaves out are stacked into one forward and one of them is computed with the other's layout. A row-independence test (each row of a batched forward equals that request run alone) catches this.
- **The key comes from host values.** Request geometry decided in `process_prompt` travels in `step_metadata` / `kwargs`; nothing in `bucket_key_for` should read a device tensor.

Batching rarely buys throughput for a large video DiT: one 544×960×121 row is already ~8k tokens and saturates the GEMMs, so two rows cost two steps' time. It still matters for smaller image models and costs nothing to keep correct.

## Schedules, seeding and per-step scalars

- `schedule_for` returns a `FlowMatchSchedule`; the base stages its tensors on the device once per request, and `STEP_SCALARS` slices `sigma`, `sigma_next` and `timestep` out per step as `[1]` device tensors. Rows at different steps therefore share one forward, and a captured graph reads them from staged buffers. A scheduler that needs another per-step value (a guidance scale) adds a `StepScalar` entry instead of overriding `prepare_inputs`.
- **Match the reference's RNG exactly.** `seed_latents` draws from a CPU generator the base seeds per request; draw the same shapes in the same order and dtype as the reference pipeline, or the comparison against the oracle is meaningless from step 0. A recipe that draws more than once (a second stage's noise) has to continue the same generator, not reseed.
- Seeded noise is staged through a pinned ring (`stage_seed`, `NoiseStager`); `tensor.to(device)` from pageable memory blocks the GPU thread and stalls the pre-plan.
- `SOLVER_STATE` names extra loop-back edges for a multi-step scheduler (wan22's UniPC); `seed_loop_back` supplies iteration 0's values and `denoise` returns them alongside the sample.

## Multi-stage recipes

A recipe such as LTX-2.5's two-stage distilled one (half-resolution denoise → latent upsampler → a short refine schedule at full resolution) is several walks over the same `dit` node, each with its own schedule and bucket key.

- **The schedule is per (request, walk).** A request returns to the node on a new walk with a different schedule; cache keyed by the request alone and the second stage runs the first stage's sigmas.
- **Iteration 0 of a later stage is not fresh noise**: it is the previous stage's output (upsampled) blended with noise at the stage's first sigma. That is a seeding hook, not model code in `denoise`: `initial_loop_back(fwd_info, inputs, bucket_key, generator)`, which defaults to `seed_loop_back` (added in #373).
- **Keep request-level facts away from step-metadata names.** The conductor merges each pass's `step_metadata` into `metadata.kwargs`, so a stage-1 `height` sent as step metadata overwrites the request's `height` before stage 2 reads it. Nest request facts under their own key.

## Attention

Every attention in the DiT should go through a ragged resource, so the model gets the engine's kernels and its backend can change in config.

- `attention_segments(bucket_key)` declares every span a request's layers attend over, as `(label, length)`; the default is one `"main"` span of all its tokens. A layer that attends over a sub-span (a refiner over the image tokens alone, a joint model's per-modality self-attention) gets its own label, and `ragged_for(label)` gives that layer its callable.
- `ragged_for` memoizes one callable per label for the life of a binding. The compiled transformer guards on the identity of the callables it is handed; a fresh one per step recompiles every step.
- A model with more than one head geometry (LTX-2.5's video 32×128 and audio 32×64 heads) declares one resource per geometry.
- Cross-attention between spans of the same request (video queries over text keys) is a separate cross-attention resource kind (`RaggedCrossAttentionSpec`, added in #372) whose step names `(q_label, kv_label)` pairs; see "Cross-attention between spans" in `docs/adding_models.rst`. Don't fall back to SDPA for it because the shapes are fixed inside a bucket.
- `joint_attention` in `attention.py` packs per-row q/k/v into the ragged layout (or runs SDPA when no resource is bound, which is what a parity backend uses).

## Compile and capture

- The engine's blanket compile is off for the node (`disable_torch_compile = True`); the model compiles its transformer region with `compile_utils.compile_transformer_forward`, which keeps the exact-op exclusions that preserve eager rounding. Compiling the DiT mostly buys the memory-bound elementwise work (modulation, fp32 norms, RoPE, gating), which on LTX-2.5 was ~45% of an eager step.
- `capture_buckets` lists `(graph_walk, bucket_key)` pairs to capture; `capture_request_inputs` supplies one placeholder row per bucket. A bucket is captured per batch size in `capture_batch_sizes`, and each one costs startup, so capture the shapes you serve, not every shape you accept. `replay_walks` lets another walk reuse a captured bucket.
- The captured graph removes the gaps the per-layer attention graph breaks leave in a compiled-but-eager step; measure both before deciding a model doesn't need capture.

## The overshoot veto

The async worker speculates the next iteration before the current one's stop is known. Flow models know their step count at ingestion, so `prepare_inputs` returns `None` for an iteration past the request's last step and the engine skips the forward. That is the base's job; a subclass that overrides `prepare_inputs` must keep it.

## State

Per-request state lives in `self.request_state(fwd_info.rid_handle)`, which the engine frees with the request. It is keyed by the integer handle (the key `check_stop` and cleanup receive), not by the `request_id` string; see invariant 14.

## Validating

Share the initial noise with the oracle and compare the velocity at every step teacher-forced (each step fed the oracle's input), then the free-running loop, then the decoded media. A few-step distilled sampler amplifies kernel-level differences, so hold the result to the reference's own spread under a different SDPA kernel rather than to the oracle's bytes (see Phase 4 of add-mstar-model).
