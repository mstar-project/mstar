# QwenVL PR 0 design

This directory is the design record for the first QwenVL integration slice for
MStar issue [#127](https://github.com/mstar-project/mstar/issues/127).

PR 0 establishes only the single-GPU, non-tensor-parallel correctness baseline
for `Qwen/Qwen3-VL-30B-A3B-Instruct`. It contains no claim about continuous
batching, tensor parallelism, target-scale memory, or comparative performance.

## Included design

| Document | Scope | Merge claim allowed |
| --- | --- | --- |
| [Single-GPU correctness](PR_0_SINGLE_GPU_CORRECTNESS.md) | Official config/checkpoint mapping, image/text graph, MRoPE, bounded KV residency, and greedy decode | Local non-TP baseline only |

## Evidence labels

- **Unit**: isolated tensor/helper behavior.
- **Component**: real model component with fake engine boundaries.
- **Integration**: real MStar scheduler/engine path.
- **System**: launched server with the published checkpoint.

PR 0 has unit/component evidence only until its real-checkpoint CUDA protocol
passes. “Tests pass” without its environment and evidence label is not an
acceptance claim.
