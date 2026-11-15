# LingBot-VLA 2.0 port

This draft tracks the model port in [issue #828](https://github.com/mstar-project/mstar/issues/828). It does not add model serving yet.

The first implementation step is the Qwen3-VL prefix and weight loader. The action expert, embodiment handling, and served request path follow as separate changes. Before choosing an attention or normalization path, compare the released RoboTwin checkpoint against the pinned reference at the prefix, denoising-step, and action outputs. Measure both FP32 and BF16 behavior and record the difference in the implementation PRs.

The issue records the checkpoint shapes, proposed split, and open decisions. This file can be replaced by the implementation notes as those decisions are tested.
