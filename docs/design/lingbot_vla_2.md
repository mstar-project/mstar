# LingBot-VLA 2.0 port

This draft tracks the model port in [issue #828](https://github.com/mstar-project/mstar/issues/828). It does not add model serving yet.

The first implementation step reads the released checkpoint's `qwen3_vl` family marker and the separate Qwen3-VL base config. It streams the VLA checkpoint's vision and language weights into the existing shared Qwen3 towers. The released RoboTwin index has 315 vision tensors and 398 language tensors under that prefix; all vision names match the tower, and the language names map to its fused Q/K/V and gate/up parameters, with the output head tied to the input embedding.

The model still needs the graph node and interleaved MRoPE binding before the prefix can run. The action expert, embodiment handling, and served request path follow as separate changes. Before choosing an attention or normalization path, compare the released RoboTwin checkpoint against the pinned reference at the prefix, denoising-step, and action outputs. Measure both FP32 and BF16 behavior and record the difference in the implementation PRs.
