"""SigLIP vision encoder for Pi0.5.

Thin wrapper around the HuggingFace SiglipVisionModel that produces a fixed
number of image tokens (default 256) per camera image at the resolution Pi0.5
expects (224x224). A learned linear projection maps SigLIP's hidden dim to the
LLM hidden dim so the resulting tokens can be concatenated with PaliGemma
language token embeddings.
"""

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from mminf.model.pi05.config import Pi05Config


class Pi05SiglipEncoder(nn.Module):
    """SigLIP image encoder + linear connector to the LLM hidden size."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config

        siglip_cfg = SiglipVisionConfig(
            hidden_size=config.vit_hidden_size,
            intermediate_size=config.vit_intermediate_size,
            num_hidden_layers=config.vit_num_layers,
            num_attention_heads=config.vit_num_heads,
            num_channels=3,
            image_size=config.vit_image_size,
            patch_size=config.vit_patch_size,
            # Pi0.5 / lerobot's PaliGemma SigLIP does NOT use the pooling
            # head — only ``last_hidden_state`` is consumed downstream by the
            # multi_modal_projector. Disabling the head matches the
            # production checkpoint key set (no ``vision_model.head.*`` keys).
            vision_use_head=False,
        )
        self.vision_model = SiglipVisionModel(siglip_cfg)
        self.connector = nn.Linear(
            config.vit_hidden_size, config.hidden_size, bias=True
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into LLM-space tokens.

        Args:
            pixel_values: Tensor of shape ``(N, 3, H, W)`` where ``N`` is the
                total number of images across cameras and requests.

        Returns:
            Tensor of shape ``(N, tokens_per_image, hidden_size)``.
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        # last_hidden_state: [N, num_patches, vit_hidden_size]
        features = outputs.last_hidden_state
        return self.connector(features)
