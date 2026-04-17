"""Map ``lerobot/pi05_base`` (and openpi-compatible) safetensors keys onto
the state_dict layout of mminf's :class:`Pi05Model` submodules.

Production Pi0.5 weights live at ``lerobot/pi05_base`` on HuggingFace as a
single ~14 GB safetensors blob. Their key naming follows lerobot's
``PaliGemmaWithExpertModel`` wrapper, which is structurally different from
mminf's ``Pi05Model`` decomposition into ``vit_encoder`` (SigLIP +
connector) and ``LLM`` (embed_tokens + paligemma + action_expert + the
flow-matching projections + time MLP). This module bridges the two.

Public entry points
-------------------

``remap_lerobot_state_dict(state_dict)``
    Pure function: takes a flat ``{lerobot_key: tensor}`` dict and returns
    ``{"vit_encoder": {...}, "embed_tokens": {...}, "paligemma": {...},
    "action_expert": {...}, "action_in_proj": {...},
    "action_out_proj": {...}, "time_mlp": {...}}``, where each inner dict
    is a state_dict ready to be passed to ``module.load_state_dict``.

``load_lerobot_pi05_into_model(model, state_dict)``
    Convenience: walks the given mminf ``Pi05Model``, lazily creates the
    submodules via ``get_submodule`` if needed, and loads the matching
    state_dict bucket into each one. Returns a list of any unmapped or
    extra keys.

These are intentionally split so the pure remapping logic can be unit
tested without instantiating any heavy modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

# ----- Lerobot key prefixes (constants for readability) -----
_PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma.model."
_GEMMA_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
_PALIGEMMA_LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_GEMMA_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"


def remap_lerobot_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Bucket a lerobot Pi0.5 state_dict by mminf submodule.

    Returns a dict mapping submodule name -> state_dict ready for that
    submodule's ``load_state_dict``. The submodule names are exactly the
    attribute names on :class:`Pi05Model` that hold the underlying
    ``nn.Module`` instances.

    Unmapped keys (e.g. ``paligemma.lm_head.weight`` once it's been
    re-routed to ``embed_tokens.weight``, or the gemma_expert lm_head
    which Pi0.5 inference doesn't use) are silently dropped.
    """
    buckets: dict[str, dict[str, torch.Tensor]] = {
        "siglip": {},
        "embed_tokens": {},
        "paligemma": {},
        "action_expert": {},
        "action_in_proj": {},
        "action_out_proj": {},
        "time_mlp": {},
    }

    for key, tensor in state_dict.items():
        # ----- Top-level wrapper layers -----
        if key.startswith("action_in_proj."):
            buckets["action_in_proj"][key.removeprefix("action_in_proj.")] = tensor
            continue
        if key.startswith("action_out_proj."):
            buckets["action_out_proj"][key.removeprefix("action_out_proj.")] = tensor
            continue
        if key.startswith("time_mlp_in."):
            sub = key.removeprefix("time_mlp_in.")
            buckets["time_mlp"][f"linear_in.{sub}"] = tensor
            continue
        if key.startswith("time_mlp_out."):
            sub = key.removeprefix("time_mlp_out.")
            buckets["time_mlp"][f"linear_out.{sub}"] = tensor
            continue

        # ----- PaliGemma side -----
        if key == _PALIGEMMA_LM_HEAD:
            # PaliGemma uses tied embeddings; lm_head.weight IS the input
            # embedding matrix. mminf has a separate Pi05LLMSubmodule.embed_tokens
            # nn.Embedding layer, but at the model level it lives on Pi05Model
            # itself, so we route it into its own bucket.
            buckets["embed_tokens"]["weight"] = tensor
            continue
        if key.startswith(_PALIGEMMA_PREFIX):
            inner = key.removeprefix(_PALIGEMMA_PREFIX)
            if inner.startswith("language_model."):
                pali_key = inner.removeprefix("language_model.")
                buckets["paligemma"][pali_key] = tensor
                continue
            if inner.startswith("vision_tower.vision_model."):
                # The lerobot key is
                #   paligemma.model.vision_tower.vision_model.<rest>
                # Pi05SiglipEncoder owns ``self.vision_model = SiglipVisionModel(...)``,
                # and HF's SiglipVisionModel has its own inner ``.vision_model``
                # attribute, so the corresponding Pi05SiglipEncoder state_dict
                # key is ``vision_model.vision_model.<rest>``. We replace
                # ``vision_tower`` with ``vision_model`` to make that explicit.
                siglip_key = "vision_model." + inner.removeprefix("vision_tower.")
                buckets["siglip"][siglip_key] = tensor
                continue
            if inner.startswith("multi_modal_projector.linear."):
                sub = inner.removeprefix("multi_modal_projector.linear.")
                buckets["siglip"][f"connector.{sub}"] = tensor
                continue
            # Anything else under paligemma.model is silently dropped (the
            # only thing here in practice is multi_modal_projector aliases
            # we already handled).
            continue

        # ----- Action expert side -----
        if key == _GEMMA_EXPERT_LM_HEAD:
            # The action expert's lm_head exists in the checkpoint because
            # GemmaForCausalLM owns it, but Pi0.5 inference never decodes
            # tokens through it. mminf doesn't model this layer at all.
            continue
        if key.startswith(_GEMMA_EXPERT_PREFIX):
            inner = key.removeprefix(_GEMMA_EXPERT_PREFIX)
            buckets["action_expert"][inner] = tensor
            continue

        # Anything else (e.g. unrelated buffers) is dropped silently.

    return buckets


def load_lerobot_pi05_into_model(
    model,
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, list[str]]:
    """Load a lerobot Pi0.5 state_dict into an mminf :class:`Pi05Model`.

    Triggers lazy submodule construction (``get_submodule("vit_encoder")``
    and ``get_submodule("LLM")``) on the model so the underlying nn.Modules
    exist before we copy weights into them. Then walks the buckets returned
    by :func:`remap_lerobot_state_dict` and ``load_state_dict``s each one.

    Args:
        model: an instantiated :class:`Pi05Model` (typically with
            ``skip_weight_loading=True`` so the meta-device init succeeds
            without trying to download anything).
        state_dict: a flat ``{lerobot_key: tensor}`` dict, e.g. produced by
            ``safetensors.torch.load_file("model.safetensors")``.
        device: device to materialize the submodules on before loading.
        strict: forwarded to each ``load_state_dict`` call. When True,
            unexpected keys raise. Set False if your config differs from
            the production checkpoint and you don't want a hard failure.

    Returns:
        Dict ``{submodule_name: [list of missing keys]}`` from each
        ``load_state_dict`` call, for diagnosis.
    """
    buckets = remap_lerobot_state_dict(state_dict)

    # Trigger lazy submodule construction.
    vit_submodule = model.get_submodule("vit_encoder", device=device)
    llm_submodule = model.get_submodule("LLM", device=device)

    missing: dict[str, list[str]] = {}

    if vit_submodule is not None and "siglip" in buckets:
        # Pi05ViTEncoderSubmodule wraps a Pi05SiglipEncoder; the encoder is
        # accessed via .encoder, and our remap keys are already in the form
        # vision_model.* / connector.* matching Pi05SiglipEncoder.state_dict.
        result = vit_submodule.encoder.load_state_dict(buckets["siglip"], strict=strict)
        missing["siglip"] = list(result.missing_keys)

    if llm_submodule is not None:
        # Pi05LLMSubmodule has fields embed_tokens, paligemma, action_expert,
        # action_in_proj, action_out_proj, time_mlp — each loaded separately.
        for name in (
            "embed_tokens",
            "paligemma",
            "action_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp",
        ):
            mod = getattr(llm_submodule, name)
            result = mod.load_state_dict(buckets[name], strict=strict)
            missing[name] = list(result.missing_keys)

    return missing
