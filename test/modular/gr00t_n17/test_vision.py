import pytest
import torch
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.step import Segment
from mstar.model.gr00t_n17.components.vision import DeepStackMerger, Gr00tN17VisionEncoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.submodules import Gr00tN17VisionEncoderSubmodule
from mstar.model.qwen3_5.components.vision import vision_interpolation, vision_position_ids
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
from mstar.model.submodule_base import ModelInputsFromEngine

from .helpers import SegmentAttention, randomize

_DEPTH = 6


def _config(taps: tuple[int, ...], depth: int = _DEPTH) -> Gr00tN17Config:
    """A tiny tower; the encoder reads only ``vision`` and the taps."""
    vision = Qwen3_5VisionConfig(
        depth=depth, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=64, out_hidden_size=48,
    )
    return Gr00tN17Config(
        text=None, vision=vision, dit=None, bridge=None,
        deepstack_visual_indexes=taps, backbone_embedding_dim=0,
    )


# ----------------------------------------------------------------------------
# Inline reference: transformers' `Qwen3VLVisionPatchMerger`
# (`models/qwen3_vl/modeling_qwen3_vl.py`), both orderings of its norm. GR00T's
# DeepStack mergers are the `use_postshuffle_norm=True` variant; the tower's own
# merger is the other one.
# ----------------------------------------------------------------------------


class RefPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig, use_postshuffle_norm: bool):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.merged_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def test_the_deepstack_merger_normalises_after_the_shuffle():
    vision = _config(()).vision
    merger = DeepStackMerger(vision)
    randomize(merger, 0, 1.0)
    reference = RefPatchMerger(vision, use_postshuffle_norm=True)
    reference.load_state_dict(merger.state_dict())
    x = torch.randn(8, vision.hidden_size, generator=torch.Generator().manual_seed(1))
    assert torch.equal(merger(x), reference(x))


def test_the_preshuffle_ordering_is_a_different_function():
    # Both norms keep the identity affine and both linears are copied across, so
    # where the norm runs is the only difference left between the two.
    vision = _config(()).vision
    merger = DeepStackMerger(vision)
    preshuffle = RefPatchMerger(vision, use_postshuffle_norm=False)
    preshuffle.linear_fc1.load_state_dict(merger.linear_fc1.state_dict())
    preshuffle.linear_fc2.load_state_dict(merger.linear_fc2.state_dict())
    x = torch.randn(8, vision.hidden_size, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(merger(x), preshuffle(x))


# ----------------------------------------------------------------------------
# The block loop and the taps
# ----------------------------------------------------------------------------


class _ConstantBlock(nn.Module):
    """Stands in for a tower block, adding a distinct power of two: the hidden state
    is then a bitmask of exactly which blocks have run."""

    def __init__(self, layer: int):
        super().__init__()
        self.delta = 2.0**layer

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return hidden + self.delta


def _stubbed_encoder(taps: tuple[int, ...], depth: int = _DEPTH) -> Gr00tN17VisionEncoder:
    """The real embed, mergers and tower merger, with stub blocks: a real block needs the
    ragged attention resource and would only prove attention keeps its shape."""
    config = _config(taps, depth)
    encoder = Gr00tN17VisionEncoder(config)
    for k, merger in enumerate(encoder.deepstack_merger_list):
        randomize(merger, 10 + k, 1.0)
    encoder.tower.blocks = nn.ModuleList(_ConstantBlock(i) for i in range(depth))
    return encoder


def _tower_args(vision: Qwen3_5VisionConfig, grid: list[tuple[int, int, int]]) -> tuple[torch.Tensor, ...]:
    """The grid-derived tensors the tower needs, straight from its own helpers."""
    device = torch.device("cpu")
    indices, weights = vision_interpolation(grid, vision.num_grid_per_side, vision.spatial_merge_size, device)
    return indices, weights, vision_position_ids(grid, vision.spatial_merge_size, device)


def _inputs(vision: Qwen3_5VisionConfig, grid: list[tuple[int, int, int]]) -> tuple[torch.Tensor, ...]:
    """What ``VisionEncoderSubmodule.prepare_inputs`` hands the tower, for one packed run."""
    total_patches = sum(t * h * w for t, h, w in grid)
    patch_numel = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    pixel_values = torch.randn(total_patches, patch_numel, generator=torch.Generator().manual_seed(2))
    return pixel_values, *_tower_args(vision, grid)


@pytest.mark.parametrize("taps", [(0, 2, 5), (1, 3, 5)])
@torch.no_grad()
def test_each_feature_comes_off_its_own_block(taps):
    encoder = _stubbed_encoder(taps)
    args = _inputs(encoder.tower.config, [(1, 4, 4)])
    embeds, features = encoder(*args)

    hidden, cos, sin = encoder.tower.embed(*args)
    # states[i] is the hidden state with blocks 0..i-1 run, accumulated in the
    # forward's own order so the comparison stays bit-exact
    states = [hidden]
    for block in encoder.tower.blocks:
        states.append(block(states[-1], cos, sin))

    assert len(features) == len(taps)
    for k, tap in enumerate(taps):
        merger = encoder.deepstack_merger_list[k]
        assert torch.equal(features[k], merger(states[tap + 1]))
        # one block earlier is a different state, which pins the tap exactly
        assert not torch.equal(features[k], merger(states[tap]))
    assert torch.equal(embeds, encoder.tower.merger(states[_DEPTH]))


@pytest.mark.parametrize("grid, merged_tokens", [([(1, 4, 4)], 4), ([(1, 2, 2), (1, 4, 6)], 7)])
@torch.no_grad()
def test_every_output_is_one_row_per_merged_block_of_patches(grid, merged_tokens):
    encoder = _stubbed_encoder((1, 3, 5))
    vision = encoder.tower.config
    embeds, features = encoder(*_inputs(vision, grid))
    expected = torch.Size((merged_tokens, vision.out_hidden_size))
    assert embeds.shape == expected
    assert [f.shape for f in features] == [expected] * 3


def test_a_tap_past_the_towers_depth_is_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        Gr00tN17VisionEncoder(_config((1, 3, _DEPTH)))


@torch.no_grad()
def test_no_taps_gives_no_features():
    encoder = _stubbed_encoder(())
    embeds, features = encoder(*_inputs(encoder.tower.config, [(1, 2, 2)]))
    assert features == []
    assert embeds.shape == (1, encoder.tower.config.out_hidden_size)


# ----------------------------------------------------------------------------
# The node: `Gr00tN17VisionEncoderSubmodule`, driven the way the engine drives it
# ----------------------------------------------------------------------------

_WALK = "prefill_vision"
_FWD = CurrentForwardPassInfo(request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})

# what the walk's `image_grid_thw` list holds -> the frames it means
_GRIDS = {
    "one image": ([torch.tensor([[1, 4, 4]])], [(1, 4, 4)]),
    "two sizes": ([torch.tensor([[1, 2, 2], [1, 4, 6]])], [(1, 2, 2), (1, 4, 6)]),
    "four cameras": ([torch.tensor([[1, 2, 4]] * 4)], [(1, 2, 4)] * 4),
    # one row, unbatched: three numbers that are one grid, not three images
    "a bare row": ([torch.tensor([1, 4, 6])], [(1, 4, 6)]),
    # t > 1 is several frames off one row, and they attend separately
    "frames": ([torch.tensor([[2, 4, 6]]), torch.tensor([3, 2, 4])], [(2, 4, 6), (3, 2, 4)]),
}


def _node(taps: tuple[int, ...] = (1, 3, 5)) -> Gr00tN17VisionEncoderSubmodule:
    config = _config(taps)
    return Gr00tN17VisionEncoderSubmodule(Gr00tN17VisionEncoder(config), config.vision)


def _walk_inputs(vision: Qwen3_5VisionConfig, grid_thw: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """The engine's two input lists: one grid tensor and one pixel block per entry."""
    patch_numel = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    generator = torch.Generator().manual_seed(4)
    pixel_values = [
        torch.randn(int(g.reshape(-1, 3).prod(-1).sum()), patch_numel, generator=generator) for g in grid_thw
    ]
    return {"image_grid_thw": grid_thw, "pixel_values": pixel_values}


@pytest.mark.parametrize("grid_thw, grid", list(_GRIDS.values()), ids=list(_GRIDS))
def test_the_step_declares_one_attention_segment_per_frame(grid_thw, grid):
    node = _node()
    inputs = node.prepare_inputs(_WALK, _FWD, _walk_inputs(node.config, grid_thw))
    step = node.declare_step(_WALK, ["r0"], [inputs])

    spans = [h * w for t, h, w in grid for _ in range(t)]
    assert step.segments == [Segment("r0", "main", span) for span in spans]
    assert sum(spans) == inputs.input_seq_len
    assert list(step.steps) == [VISION_ATTN]
    assert step.steps[VISION_ATTN].causal is False


@pytest.mark.parametrize("grid_thw, grid", list(_GRIDS.values()), ids=list(_GRIDS))
def test_prepare_inputs_derives_the_grid_the_tower_helpers_expect(grid_thw, grid):
    node = _node()
    walk_inputs = _walk_inputs(node.config, grid_thw)
    tensors = node.prepare_inputs(_WALK, _FWD, walk_inputs).tensor_inputs
    indices, weights, position_ids = _tower_args(node.config, grid)

    assert torch.equal(tensors["indices"], indices)
    assert torch.equal(tensors["weights"], weights)
    assert torch.equal(tensors["position_ids"], position_ids)
    assert torch.equal(tensors["pixel_values"], torch.cat(walk_inputs["pixel_values"]))


@torch.no_grad()
def test_driving_the_node_runs_the_encoders_own_forward():
    """prepare_inputs -> preprocess -> forward is the module's own forward on the same
    images, under both of the forward's keys."""
    node = _node()
    grid_thw, grid = _GRIDS["two sizes"]
    walk_inputs = _walk_inputs(node.config, grid_thw)
    inputs = node.prepare_inputs(_WALK, _FWD, walk_inputs)
    node.bind_node_resources({VISION_ATTN: SegmentAttention(inputs.kwargs["seq_lengths"])})

    tensors = node.preprocess(_WALK, _ENGINE, [inputs])
    # `seq_lengths` is spent on the segments, and is not a forward argument
    assert set(tensors) == {"pixel_values", "indices", "weights", "position_ids"}
    out = node.forward(_WALK, _ENGINE, **tensors)

    embeds, features = node.model(torch.cat(walk_inputs["pixel_values"]), *_tower_args(node.config, grid))
    assert torch.equal(out["vision_embeds"][0], embeds)
    assert all(torch.equal(ours, theirs) for ours, theirs in zip(out["deepstack_embeds"], features, strict=True))
    # one embedding and one feature per tap, each a row per 2x2 block: (2 * 2 + 4 * 6) / 4
    shape = (7, node.config.out_hidden_size)
    assert [tuple(t.shape) for t in out["vision_embeds"] + out["deepstack_embeds"]] == [shape] * 4


# ----------------------------------------------------------------------------
# The rope the blocks get
# ----------------------------------------------------------------------------


def _rope_into_blocks(encoder: Gr00tN17VisionEncoder, args: tuple[torch.Tensor, ...]):
    """The encoder's output, and the (cos, sin) each block was called with."""
    seen = []
    hooks = [b.register_forward_pre_hook(lambda _, a: seen.append(a[1:])) for b in encoder.tower.blocks]
    embeds, _ = encoder(*args)
    for hook in hooks:
        hook.remove()
    return embeds, seen


@torch.no_grad()
def test_a_bf16_encoder_feeds_its_blocks_the_fp32_rope(monkeypatch):
    """transformers' Qwen3-VL keeps the vision rope fp32 into the blocks, where ``embed``
    rounds it; in bf16 that rounding alone put vision_embeds at 1.25x the bf16 floor."""
    encoder = Gr00tN17VisionEncoder(_config((0,), depth=2)).to(torch.bfloat16)
    args = _inputs(encoder.tower.config, [(1, 4, 4)])
    args = (args[0].to(torch.bfloat16), *args[1:])
    for block in encoder.tower.blocks:
        block.attn.bind_resources({VISION_ATTN: SegmentAttention((16,))})
    rope = encoder.tower.rotary_pos_emb
    table = rope(args[3])

    def fp32_table(seen):
        pairs = [(r, t) for got in seen for r, t in zip(got, table, strict=True)]
        return all(r.dtype == torch.float32 and torch.equal(r, t) for r, t in pairs)

    embeds, seen = _rope_into_blocks(encoder, args)
    assert len(seen) == 2 and fp32_table(seen)

    # the old wiring: every block gets `embed`'s rounded copy
    fp32 = rope.forward
    monkeypatch.setattr(rope, "forward", lambda p: [t.to(torch.bfloat16) for t in fp32(p)])
    rounded_embeds, seen = _rope_into_blocks(encoder, args)
    assert not fp32_table(seen)
    # and the rounding reaches the output at this size
    assert not torch.equal(embeds, rounded_embeds)
