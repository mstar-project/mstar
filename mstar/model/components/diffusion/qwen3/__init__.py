"""The Qwen3 prompt encoder the scaffold's DiTs condition on.

FLUX.2 [klein] and Z-Image feed their DiTs intermediate hidden states of a Qwen3
LM rather than its logits, so this is a Qwen3 LM truncated at the deepest tapped
layer: ``encoder`` is the module and its config, ``weight_loading`` the
checkpoint rules that fill it. Distinct from ``components.qwen3_lm``, which
serves Qwen3 as a language model.
"""
