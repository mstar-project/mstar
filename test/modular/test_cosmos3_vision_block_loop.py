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


def test_the_block_loop_runs_in_bf16_without_autocast():
    """The node opts out of autocast, so the engine hands it no autocast
    dtype: the block loop's replay buffers must still be bf16, the tower's
    own dtype, not the fp32 a bare ``torch.zeros`` would make."""
    from mstar.engine.accelerator_graph_config import PiecewiseCaptureShape
    from mstar.model.cosmos3.cosmos3_model import VISION_ENCODER_NODE, Cosmos3Model
    from mstar.model.cosmos3.submodules import Cosmos3VisionEncoderSubmodule

    model = _tower().to(torch.bfloat16)
    node = Cosmos3VisionEncoderSubmodule(model, SimpleNamespace(reasoner=model.config))
    # the dtype the engine hands the node: its param dtype, None as it keeps its own
    dtype = Cosmos3Model.get_node_precision(None, VISION_ENCODER_NODE).param_dtype
    assert dtype is None
    (config,) = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), dtype).values()
    buffers = config.make_static_inputs(PiecewiseCaptureShape(bs=1, seq_lens=[16], total_tokens=16))
    assert {t.dtype for t in buffers.values()} == {torch.bfloat16}


def test_the_vision_attention_declares_fa3():
    """head_dim 72 runs unpadded on FA3 varlen; FlashInfer would pad it to 128."""
    from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    model = Cosmos3Model.__new__(Cosmos3Model)
    model.config = SimpleNamespace(reasoner=SimpleNamespace(
        vision=SimpleNamespace(num_attention_heads=16, hidden_size=1152, patch_size=16),
        image_processor=SimpleNamespace(min_pixels=65536), video_processor=SimpleNamespace(min_pixels=4096),
    ))
    spec = model._vision_attention_spec()
    assert spec.config.head_dim == 72 and spec.config.backend is RaggedAttnBackend.DENSE


def test_position_tables_are_kept_within_a_byte_budget():
    """The node keeps resized position tables up to its byte budget, least
    recently used dropped first, and doesn't keep one over an eighth of it."""
    from mstar.model.cosmos3.submodules import Cosmos3VisionEncoderSubmodule

    config = Cosmos3ReasonerConfig(
        vision=Cosmos3VisionEncoderConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            patch_size=2, num_patches=16, spatial_merge_size=1,
        ),
        projector_input_hidden_size=32, projector_hidden_size=48, projector_out_hidden_size=24,
    )
    model = Cosmos3VisionModel(config).eval()
    node = Cosmos3VisionEncoderSubmodule(model, SimpleNamespace(reasoner=config))
    one = model.position_table(1, 1)
    patch = one.numel() * one.element_size()
    node.POSITION_TABLE_BYTES = 64 * patch  # an eighth: 8 patches
    with torch.no_grad():
        for hw in ((2, 2), (2, 4), (4, 2), (2, 2)):  # 4, 8 and 8 patches; (2, 2) used again
            node._position_table(*hw)
        assert list(node._position_tables) == [(2, 4), (4, 2), (2, 2)]
        node._position_table(6, 4)  # 24 patches, over an eighth: built, not kept
        assert (6, 4) not in node._position_tables
        node.POSITION_TABLE_BYTES = 16 * patch  # an eighth: 2 patches
        node._position_table(1, 1)  # 21 patches held: the least recent, (2, 4), goes
        assert list(node._position_tables) == [(4, 2), (2, 2), (1, 1)]
        assert node._position_table_bytes == 13 * patch
