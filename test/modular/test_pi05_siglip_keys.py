"""pi05's SigLIP checkpoint keys against the encoder's own parameter names (CPU,
no weights). The layout of HF's SiglipVisionModel changed between transformers
4.x and 5.20, and a key that reaches no parameter leaves it uninitialized."""

import torch

from mstar.model.pi05.config import Pi05Config


def test_extract_siglip_state_dict_filters_and_renames_vision_keys():
    """Verify that ``Pi05Model._extract_siglip_state_dict`` maps the lerobot
    SigLIP and connector keys onto ``Pi05SiglipEncoder``'s own parameter names
    (whatever layout the installed transformers gives SiglipVisionModel) and
    drops everything else.
    """
    from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
    from mstar.model.pi05.pi05_model import Pi05Model

    def t(*shape):
        return torch.zeros(*shape)

    with torch.device("meta"):
        encoder = Pi05SiglipEncoder(Pi05Config())
    params = dict(encoder.named_parameters())
    prefix = Pi05Model._siglip_tower_prefix(encoder)

    pali = "paligemma_with_expert.paligemma.model"
    flat = {
        # every tower parameter under its lerobot name
        **{f"{pali}.vision_tower.vision_model.{name.removeprefix(prefix)}": t(1)
           for name in params if name.startswith(prefix)},
        # multi_modal_projector.linear -> connector
        f"{pali}.multi_modal_projector.linear.weight": t(2048, 1152),
        f"{pali}.multi_modal_projector.linear.bias": t(2048),
        # Non-siglip keys should be dropped
        "action_in_proj.weight": t(1024, 32),
        f"{pali}.language_model.norm.weight": t(2048),
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": t(2048, 1024),
    }
    assert f"{pali}.vision_tower.vision_model.embeddings.patch_embedding.weight" in flat
    assert f"{pali}.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight" in flat
    siglip = Pi05Model._extract_siglip_state_dict(flat, prefix)

    assert set(siglip) == set(params)


def test_siglip_tower_prefix_follows_the_module_layout():
    """transformers 4.x nests SigLIP's tower one level deeper than 5.20."""
    from torch import nn

    from mstar.model.pi05.pi05_model import Pi05Model

    def tower():
        m = nn.Module()
        m.embeddings = nn.Module()
        m.embeddings.patch_embedding = nn.Conv2d(3, 4, 2)
        return m

    nested, flat = nn.Module(), nn.Module()
    nested.vision_model = nn.Module()
    nested.vision_model.vision_model = tower()
    flat.vision_model = tower()
    assert Pi05Model._siglip_tower_prefix(nested) == "vision_model.vision_model."
    assert Pi05Model._siglip_tower_prefix(flat) == "vision_model."
