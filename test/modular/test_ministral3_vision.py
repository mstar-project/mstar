"""Native Pixtral parity; CPU dense attention substitutes only the engine kernel.

No pretrained tensor payloads, image preprocessing, GPU kernels or serving are
validated here. HF is the independent synthetic-weight oracle.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.keys import VISION_ATTN
from mstar.model.ministral3.components.vision import (
    Ministral3Vision,
    PixtralConfig,
    axial_rotary,
    pack_image_patches,
)
from mstar.model.ministral3.image import IMAGE_GRIDS
from mstar.model.ministral3.vision import VISION_NODE, Ministral3VisionEncoder, vision_node_resources
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs

FIXTURES = Path(__file__).parent / "fixtures" / "ministral3"
TINY = PixtralConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=2, head_dim=16, patch_size=2, text_hidden_size=48)


class DenseAttention:
    def __init__(self, lengths):
        self.lengths = lengths

    def run(self, q, k, v, *, label):
        assert label == "main"
        outputs = []
        for qi, ki, vi in zip(q.split(self.lengths), k.split(self.lengths), v.split(self.lengths), strict=True):
            # Match HF's batch/head axes so component parity can be bit exact.
            qi, ki, vi = (x.transpose(0, 1).unsqueeze(0) for x in (qi, ki, vi))
            scores = qi @ ki.transpose(-1, -2) * q.shape[-1] ** -0.5
            outputs.append((scores.softmax(-1) @ vi)[0].transpose(0, 1))
        return torch.cat(outputs)


def bind(model, lengths):
    attention = DenseAttention(lengths)
    for module in model.modules():
        if hasattr(module, "bind_resources"):
            module.bind_resources({VISION_ATTN: attention})


def oracle():
    pytest.importorskip("transformers.models.pixtral.modeling_pixtral")
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    values = asdict(TINY)
    for key in ("spatial_merge_size", "text_hidden_size", "projector_norm_eps", "rope_theta"):
        values.pop(key)
    config = PixtralVisionConfig(**values, image_size=32, hidden_act="silu",
                                 rope_parameters={"rope_type": "axial", "rope_theta": TINY.rope_theta})
    root = Mistral3Config(vision_config=config, text_config={"model_type": "ministral3",
        "hidden_size": TINY.text_hidden_size, "rms_norm_eps": TINY.projector_norm_eps},
        spatial_merge_size=TINY.spatial_merge_size, multimodal_projector_bias=False)
    # The composite config resets the nested backend; choose after constructing it.
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(431)
        model = nn.Module()
        model.vision_tower = PixtralVisionModel(config)
        assert model.vision_tower.config._attn_implementation == "eager"
        model.multi_modal_projector = Mistral3MultiModalProjector(root)
        return model.eval()


def inputs(shapes):
    generator = torch.Generator().manual_seed(52)
    return [torch.randn(3, h, w, generator=generator) for h, w in shapes]


def native_forward(model, images):
    pixels, positions, grids = pack_image_patches(images, TINY)
    bind(model, [h * w for _, h, w in grids])
    hidden, cos, sin = model.embed(pixels, positions)
    encoded = model.encode(hidden, cos, sin)
    return encoded, model.project(encoded), positions


def test_released_vision_parameters_match_every_header_entry_on_meta():
    config = PixtralConfig.from_hf(json.loads((FIXTURES / "config.json").read_text()))
    with torch.device("meta"):
        native = Ministral3Vision(config)
    header = json.loads((FIXTURES / "model.safetensors.header.json").read_text())
    expected = {name: entry for name, entry in header.items()
                if name.startswith(("vision_tower.", "multi_modal_projector."))}
    actual = dict(native.named_parameters())
    assert actual.keys() == expected.keys()
    assert len(actual) == 222
    for name, parameter in actual.items():
        assert list(parameter.shape) == expected[name]["shape"], name
        assert expected[name]["dtype"] == "BF16"


@pytest.mark.parametrize("shapes", [[(8, 12)], [(8, 12), (12, 4)], [(4, 4)]])
@torch.no_grad()
def test_tower_and_projector_match_reference(shapes):
    hf = oracle()
    model = Ministral3Vision(TINY).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    images = inputs(shapes)
    encoded, actual, positions = native_forward(model, images)
    # HF accepts padded NCHW images and crops each image's patch grid.
    padded = torch.zeros(len(images), 3, max(h for h, w in shapes), max(w for h, w in shapes))
    for i, image in enumerate(images):
        padded[i, :, :image.shape[1], :image.shape[2]] = image
    sizes = torch.tensor(shapes)
    reference = hf.vision_tower(padded, image_sizes=sizes).last_hidden_state.squeeze(0)
    projected = hf.multi_modal_projector(reference, sizes)
    # Compare tower output after restoring HF's raster order independently.
    offset, expected = 0, []
    for h, w in shapes:
        h, w = h // TINY.patch_size, w // TINY.patch_size
        coords = positions[offset:offset + h * w]
        expected.append(reference[offset + coords[:, 0] * w + coords[:, 1]])
        offset += h * w
    torch.testing.assert_close(encoded, torch.cat(expected), atol=5e-7, rtol=2e-6)
    torch.testing.assert_close(actual, projected, atol=5e-7, rtol=2e-6)
    individual = torch.cat([native_forward(model, [image])[1] for image in images])
    torch.testing.assert_close(actual, individual, atol=5e-7, rtol=2e-6)


@torch.no_grad()
def test_negative_controls_detect_wrong_rotation_and_cross_image_attention():
    hf = oracle()
    model = Ministral3Vision(TINY).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    images = inputs([(8, 12), (12, 4)])
    _, correct, _ = native_forward(model, images)
    pixels, positions, grids = pack_image_patches(images, TINY)
    hidden, cos, sin = model.embed(pixels, positions)
    no_rope = model.project(model.encode(hidden, torch.ones_like(cos), torch.zeros_like(sin)))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(correct, no_rope, atol=5e-7, rtol=2e-6)
    bind(model, [sum(h * w for _, h, w in grids)])
    leaked = model.project(model.encode(hidden, cos, sin))
    assert (correct - leaked).abs().max() > 1e-4


def test_rotary_matches_reference_exactly():
    hf = oracle()
    positions = torch.tensor([[0, 0], [0, 7], [11, 0], [23, 31]])
    for dtype in (torch.float32, torch.bfloat16):
        actual = axial_rotary(positions, TINY, dtype)
        expected = hf.vision_tower.patch_positional_embedding(torch.empty(1, dtype=dtype), positions)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@torch.no_grad()
def test_each_full_block_is_exact_with_matching_input_and_attention_layout():
    hf = oracle()
    model = Ministral3Vision(TINY).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    pixels, positions, _ = pack_image_patches(inputs([(8, 12)]), TINY)
    bind(model, [len(pixels)])
    actual, cos, sin = model.embed(pixels, positions)
    expected = actual.unsqueeze(0).clone()
    for native_layer, reference_layer in zip(model.vision_tower.transformer.layers,
                                             hf.vision_tower.transformer.layers, strict=True):
        actual = native_layer(actual, cos, sin)
        expected = reference_layer(expected, attention_mask=None, position_embeddings=(cos, sin))
        torch.testing.assert_close(actual, expected[0], atol=0, rtol=0)


@torch.no_grad()
def test_negative_control_detects_wrong_patch_merge_order():
    hf = oracle()
    model = Ministral3Vision(TINY).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    encoded, correct, _ = native_forward(model, inputs([(8, 12)]))
    projector = model.multi_modal_projector
    # Flattening patches directly puts positions before channels, unlike HF unfold.
    wrong = projector.norm(encoded).reshape(-1, TINY.hidden_size * TINY.spatial_merge_size ** 2)
    wrong = projector.patch_merger.merging_layer(wrong)
    wrong = projector.linear_2(torch.nn.functional.gelu(projector.linear_1(wrong)))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(correct, wrong, atol=5e-7, rtol=2e-6)


def test_node_declares_independent_noncausal_image_segments():
    node = Ministral3VisionEncoder(TINY)
    specs = vision_node_resources(TINY)
    assert len(specs) == 1 and specs[0].nodes == {VISION_NODE}
    assert specs[0].resource_key == VISION_ATTN
    assert specs[0].config.head_dim == TINY.head_dim
    rows = [NodeInputs(input_seq_len=36, resource_step_info={"main": (24, 12)}),
            NodeInputs(input_seq_len=4, resource_step_info={"main": (4,)})]
    step = node.declare_step("prefill_image", [7, 9], rows)
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [
        (7, "main", 24), (7, "main", 12), (9, "main", 4)]
    assert not step.steps[VISION_ATTN].causal
    assert node.get_piecewise_accelerator_graph_configs() == {}


@torch.no_grad()
def test_node_packed_forward_splits_request_outputs():
    node = Ministral3VisionEncoder(TINY).eval()
    images = inputs([(8, 12), (12, 4), (4, 4)])
    pixels, positions, grids = pack_image_patches(images, TINY)
    bind(node.model, [h * w for _, h, w in grids])
    result = node.forward_batched("prefill_image", ModelInputsFromEngine(request_ids=[7, 9], per_request_info={}),
        segment_lengths={"main": (24, 12, 4)}, row_patches=(36, 4),
        pixel_values=pixels, position_ids=positions)
    assert result[7][node.output][0].shape == (9, TINY.text_hidden_size)
    assert result[9][node.output][0].shape == (1, TINY.text_hidden_size)
    for rid, selected in ((7, images[:2]), (9, images[2:])):
        torch.testing.assert_close(result[rid][node.output][0], native_forward(node.model, selected)[1])


def test_prepare_preprocess_and_resource_binding():
    node = Ministral3VisionEncoder(TINY)
    pixels, positions, grids = pack_image_patches(inputs([(8, 12), (12, 4)]), TINY)
    info = SimpleNamespace(step_metadata={MEDIA_LAYOUT: {
        IMAGE_GRIDS: grids,
    }})
    inp = node.prepare_inputs("prefill_image", info, {"pixel_values": [pixels]})
    assert inp.input_seq_len == 36 and inp.resource_step_info == {"main": (24, 12)}
    torch.testing.assert_close(inp.tensor_inputs["position_ids"], positions)
    engine = ModelInputsFromEngine(request_ids=[7], per_request_info={})
    packed = node.preprocess("prefill_image", engine, [inp])
    assert packed["row_patches"] == (36,)
    attention = DenseAttention([24, 12])
    node.bind_node_resources({VISION_ATTN: attention})
    assert all(layer.attention.ragged_attn is attention for layer in node.model.vision_tower.transformer.layers)
    with pytest.raises(ValueError, match="disagree"):
        node.prepare_inputs("prefill_image", info, {"pixel_values": [pixels[:-1]]})


@pytest.mark.parametrize("shape", [(3, 7, 8), (3, 0, 4), (1, 4, 4)])
def test_invalid_patch_geometry_fails(shape):
    with pytest.raises(ValueError):
        pack_image_patches([torch.empty(shape)], TINY)


def test_invalid_head_geometry_fails():
    with pytest.raises(ValueError):
        replace(TINY, head_dim=15)
