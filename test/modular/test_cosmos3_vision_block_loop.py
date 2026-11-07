"""The Cosmos3-Edge vision tower served on `PackedBlockLoopEncoder`: `embed`,
then `encode` over the frames the step planned on the ragged attention
resource, then `project`, must be the reference `forward` (attention within
each frame of the packed patches), for a step packing several requests'
images and video frames; and the node plans one segment per layout run."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mstar.model.cosmos3.components.vision import Cosmos3VisionModel, frame_attention
from mstar.model.cosmos3.config import Cosmos3ReasonerConfig, Cosmos3VisionEncoderConfig
from mstar.model.keys import VISION_ATTN
from mstar.model.multimodal import MEDIA_LAYOUT


class _PlannedFrames:
    """The ragged attention resource as the encoder blocks see it: attention
    within each planned segment."""

    def __init__(self, lengths: list[int]):
        self.lengths = lengths

    def run(self, q, k, v, label=None):
        return frame_attention(q, k, v, self.lengths)


def _tower() -> Cosmos3VisionModel:
    torch.manual_seed(0)
    config = Cosmos3ReasonerConfig(
        vision=Cosmos3VisionEncoderConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            patch_size=2, num_patches=16,
        ),
        projector_input_hidden_size=32, projector_hidden_size=48, projector_out_hidden_size=24,
    )
    model = Cosmos3VisionModel(config).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model


def test_the_served_path_is_the_reference_forward():
    model = _tower()
    # an image, a two-frame video and another image of another size, packed
    frames = [(1, 4, 6), (1, 4, 4), (1, 4, 4), (1, 6, 4)]
    patches = sum(h * w for _, h, w in frames)
    pixels = torch.randn(patches, 3 * 2 * 2)
    with torch.no_grad():
        ref = model(pixels, frames)
        pos = torch.cat([model.position_table(h, w) for _, h, w in frames])
        plan = _PlannedFrames([h * w for _, h, w in frames])
        for layer in model.visual.encoder.layers:
            layer.self_attn.bind_resources({VISION_ATTN: plan})
        hidden, cos, sin = model.embed(pixels, pos)
        served = model.project(model.encode(hidden, cos, sin))
    assert served.shape == ref.shape == (patches // 4, 24)
    torch.testing.assert_close(served, ref, rtol=1e-5, atol=1e-5)


def test_the_node_plans_a_segment_per_layout_run():
    from mstar.model.cosmos3.submodules import Cosmos3VisionEncoderSubmodule

    model = _tower()
    cfg = SimpleNamespace(reasoner=model.config)
    node = Cosmos3VisionEncoderSubmodule(model, cfg)
    node.__dict__["_cached_device"] = torch.device("cpu")
    layout = {"offset": [3, 12], "tokens": [6, 4], "output": ["vision_embeds"] * 2,
              "grid": [[1, 4, 6], [1, 4, 4]]}
    info = SimpleNamespace(step_metadata={MEDIA_LAYOUT: layout})
    pixels = torch.randn(4 * 6 + 4 * 4, 12)
    with torch.no_grad():
        inp = node.prepare_inputs("prefill_vision", info, {"pixel_values": [pixels]})
    assert inp.input_seq_len == 40 and inp.resource_step_info == {"main": [24, 16]}
    torch.testing.assert_close(
        inp.tensor_inputs["pos_embeds"], torch.cat([model.position_table(4, 6), model.position_table(4, 4)]),
    )
    # the second (4, 4) frame of a step reuses its table
    assert set(node._position_tables) == {(4, 6), (4, 4)}
    try:
        node.prepare_inputs("prefill_vision", info, {"pixel_values": [pixels[:-1]]})
    except ValueError as e:
        assert "39 pixel patches for the layout's 40" in str(e)
    else:
        raise AssertionError("a patch count off the layout was accepted")
