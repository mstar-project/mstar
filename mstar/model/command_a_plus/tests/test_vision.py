"""The image encoder's weight loading, and its output against Hugging Face's.

MSTAR_TEST_HF=1 enables the comparison with transformers' SigLIP tower and
Cohere2Vision projector.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mstar.model.command_a_plus.components.vision import CommandAPlusVisionEncoder
from mstar.model.command_a_plus.config import CommandAPlusConfig

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_config.json"


def tiny_encoder() -> CommandAPlusVisionEncoder:
    config = CommandAPlusConfig.from_json(FIXTURE)
    return CommandAPlusVisionEncoder(config.vision_config, config.text_config.hidden_size)


def checkpoint_name(name: str) -> str:
    if name.startswith("layers."):
        return "model.vision_tower.vision_model.encoder." + name
    if name.startswith("projector."):
        return "model.multi_modal_projector." + name.removeprefix("projector.")
    return "model.vision_tower.vision_model." + name


class VisionEncoderTests(unittest.TestCase):
    def test_loads_checkpoint_names_strictly(self):
        torch.manual_seed(0)
        encoder = tiny_encoder()
        weights = {checkpoint_name(n): torch.randn_like(p) for n, p in encoder.named_parameters()}
        encoder.load_weights(weights.items())
        for name, parameter in encoder.named_parameters():
            torch.testing.assert_close(parameter, weights[checkpoint_name(name)], rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            tiny_encoder().load_weights(list(weights.items())[1:])


@unittest.skipUnless(os.environ.get("MSTAR_TEST_HF") == "1", "set MSTAR_TEST_HF=1")
class HuggingFaceVisionParityTests(unittest.TestCase):
    def test_matches_siglip_and_the_cohere2_vision_projector(self):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from transformers.models.cohere2_vision.modeling_cohere2_vision import (
            Cohere2VisionMultiModalProjector,
        )

        torch.manual_seed(0)
        tower = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
            image_size=32, patch_size=8, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
            vision_use_head=False, attn_implementation="eager",
        )).eval()
        projector = Cohere2VisionMultiModalProjector(SimpleNamespace(
            downsample_factor=2, alignment_intermediate_size=32,
            vision_config=SimpleNamespace(hidden_size=16), text_config=SimpleNamespace(hidden_size=32),
        )).eval()
        for parameter in [*tower.parameters(), *projector.parameters()]:
            torch.nn.init.normal_(parameter, std=0.2)

        encoder = tiny_encoder().eval()
        # transformers 5 drops the checkpoint's `vision_model.` level from its own names
        encoder.load_weights([
            *((f"model.vision_tower.vision_model.{n.removeprefix('vision_model.')}", p)
              for n, p in tower.named_parameters()),
            *((f"model.multi_modal_projector.{n}", p) for n, p in projector.named_parameters()),
        ])
        pixels = torch.rand(3, 3, 32, 32) * 2 - 1
        with torch.no_grad():
            expected = projector(tower(pixels).last_hidden_state).reshape(-1, 32)
            torch.testing.assert_close(encoder(pixels), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
