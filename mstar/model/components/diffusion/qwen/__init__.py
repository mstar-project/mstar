"""The Qwen-family prompt encoder the scaffold's DiTs condition on.

FLUX.2 [klein] and Z-Image feed their DiTs intermediate hidden states of a Qwen3 LM, MiniMax-H3
a Qwen3-VL LM's, and Qwen-Image a Qwen2.5-VL LM's normed last hidden state, rather than logits:
``encoder`` is that LM truncated at the deepest tapped layer, and its config; ``vl_prompt`` the
vision tower's output it takes for a prompt with images; ``weight_loading`` the checkpoint rules
that fill it; ``submodule`` the padded-prompt ``text_encoder`` node klein and Z-Image serve it
from. Distinct from ``components.qwen3_lm``, which serves Qwen3 as a language model.

``dit_kernels`` holds the Qwen-Image DiTs' fused elementwise kernels (Edit-2511/2509 and 2.1 share the per-head
QwenRMSNorm + complex rotary chain and the zero_cond_t modulation), bit-exact with the eager ops.
"""
