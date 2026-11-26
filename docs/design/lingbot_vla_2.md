# LingBot-VLA 2.0 port

This note documents the model port for [issue #828](https://github.com/mstar-project/mstar/issues/828).

## Architecture and State Contract

LingBot-VLA 2.0 (`LingBotVLA2Model`, registry key `lingbot_vla2`, default HF ID `robbyant/lingbot-vla-v2-6b-robotwin`) runs across three nodes (`vit_encoder`, `LLM`, `action_expert`) and two graph walks (`prefill` and `action_gen`), with `LLM` and `action_expert` co-located in one node group (`configs/lingbot_vla2.yaml`) so they share the paged `LLM_KV` array:

- **`vit_encoder` (`LingBotVLA2ViTEncoderSubmodule`)**: Resizes, normalizes (`[-1, 1]`), and patchifies multi-camera inputs into `QwenVisionModel` (`VISION_TOKENS` with `Block` attention), emitting visual embeddings and deepstack addends.
- **`LLM` (`LingBotVLA2LLMSubmodule`)**: Splices visual embeddings, deepstack features, and optional learned alignment query tokens (`LingBotVLA2PrefixQueries`) into `Qwen3DenseLM` with interleaved 3D MRoPE, writing `LLM_KV` (`Causal` read, `keep=True`) and persisting `prefix_done` across graph walks.
- **`action_expert` (`LingBotVLA2ActionSubmodule`)**: Runs the 36-layer MoE + AdaRMSNorm flow-matching action expert (`LingBotVLAActionExpert`) across `num_flow_steps` Euler steps, reading cached prefix KV + `[state, noisy_actions]` suffix via `PrefixSuffixGroups(first_group_len=1)` (`keep=False`), then unnormalizes and slices to `action_dim` (14 for RoboTwin).

## Serving & Verification Notes

- Registered in `mstar/model/registry.py`, `mstar/cli/main.py`, `mstar/integrations/dynamo/bridges.py` (`LingBotVLA2TensorSpec`), `docs/models.rst`, and `docs/mintlify/models.mdx`.
- Note: `block_future_depth_to_action` and `block_suffix_to_future_video` are rejected at config validation time (unused in the released RoboTwin checkpoint; requires per-segment sub-prefix suffix masking if enabled in future checkpoints).
- All CPU unit, parity, state-contract, `StepRunner`, and `Engine` tests (`test/modular/test_lingbot_vla2_*.py`) pass against synthetic sharded checkpoints and eager references. End-to-end CUDA FlashInfer `PrefixSuffixPagedWrapper` execution and live `robbyant/lingbot-vla-v2-6b-robotwin` checkpoint parity remain pending on a GPU host.
