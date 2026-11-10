"""The Qwen-family prompt encoder the scaffold's DiTs condition on.

FLUX.2 [klein] and Z-Image feed their DiTs intermediate hidden states of a Qwen3 LM, MiniMax-H3
a Qwen3-VL LM's, Qwen-Image a Qwen2.5-VL LM's normed last hidden state and Qwen-Image-2.1 a
Qwen3-VL LM's last one before the norm, rather than logits: ``encoder`` is that LM truncated at
the deepest tapped layer, and its config; ``vl_prompt`` the vision tower's output it takes for a
prompt with images, and the Qwen3-VL tower call; ``weight_loading`` the checkpoint rules that fill
it and the Qwen3-VL tower; ``submodule`` the padded-prompt ``text_encoder`` node klein and Z-Image serve it
from. Distinct from ``components.qwen3_lm``, which serves Qwen3 as a language model.
"""
