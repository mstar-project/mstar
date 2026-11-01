"""The DiT scaffold: shared pieces of a text -> flow-transformer loop -> VAE model.

Each module here is model-agnostic and tested against the reference library
once, so a new image/video diffusion model brings only its config, its exact
component ports, a weight loader and a thin graph file:

* ``flow_match``   sigma schedules (linear / dynamic / FLUX.2 empirical shift) + the Euler step
* ``rope``         multi-axis rotary tables and the interleaved application FLUX-family DiTs use
* ``attention``    joint attention over packed text+image tokens: SDPA eagerly, the ragged
                   FlashInfer resource under CUDA graphs
* ``qwen3``        the native Qwen3-family prompt encoder the DiTs condition on, and the
                   checkpoint rules that fill it (model-specific, hence its own package)
* ``denoise_loop`` the Loop-body submodule: seeded noise, equal-shape request batching, per-step
                   CUDA graphs, stop at the request's step count
* ``image_io``     pixel/latent packing helpers and PNG encoding
* ``lora``         static LoRA merging at load time
* ``compile_utils``/``decode_utils``  the transformer/VAE ``torch.compile`` wrappers, the exact-op
                   exclusions, and the batch splitting that keeps a compiled decode on warm shapes
* ``weight_loading``  model-agnostic component streaming under the completeness contract
"""
