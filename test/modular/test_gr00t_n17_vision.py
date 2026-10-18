import dataclasses
import glob
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mstar.model.components.qwen3_vit import VisionPatchEmbed, vision_interpolation, vision_position_ids
from mstar.model.gr00t_n17.components.vision import ConvPatchEmbed, DeepStackMerger, Gr00tN17VisionEncoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.weight_loader import _remapper
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

_DEPTH = 6


@pytest.fixture(autouse=True)
def _pinned_threads():
    """These compare fp32 CPU forwards with ``torch.equal``; reduction order follows the thread count."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _config(taps: tuple[int, ...], depth: int = _DEPTH) -> Gr00tN17Config:
    """A tiny tower. The encoder reads only ``vision`` and the taps; the rest of the
    node's config belongs to the LLM and the action head."""
    vision = Qwen3_5VisionConfig(
        depth=depth, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=64, out_hidden_size=48,
    )
    return Gr00tN17Config(
        text=None, vision=vision, dit=None, bridge=None,
        deepstack_visual_indexes=taps, backbone_embedding_dim=0,
    )


def _randomize(module: nn.Module, seed: int) -> None:
    """LayerNorm initialises to the identity affine, which would hide a norm that is wired up wrong."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=generator))


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
    _randomize(merger, 0)
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
    """The real embed, mergers and tower merger; stubs where the blocks were.

    A real block needs the ragged attention resource, which is a GPU kernel, and
    running one would only prove that attention preserves its shape.
    """
    config = _config(taps, depth)
    encoder = Gr00tN17VisionEncoder(config)
    for k, merger in enumerate(encoder.deepstack_merger_list):
        _randomize(merger, 10 + k)
    encoder.tower.blocks = nn.ModuleList(_ConstantBlock(i) for i in range(depth))
    return encoder


def _inputs(vision: Qwen3_5VisionConfig, grid: list[tuple[int, int, int]]) -> tuple[torch.Tensor, ...]:
    """What ``VisionEncoderSubmodule.prepare_inputs`` hands the tower, for one packed run."""
    total_patches = sum(t * h * w for t, h, w in grid)
    patch_numel = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    pixel_values = torch.randn(total_patches, patch_numel, generator=torch.Generator().manual_seed(2))
    indices, weights = vision_interpolation(grid, vision.num_grid_per_side, vision.spatial_merge_size)
    return pixel_values, indices, weights, vision_position_ids(grid, vision.spatial_merge_size)


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


@torch.no_grad()
def test_the_patch_embed_is_nvidias_conv():
    """NVIDIA's tower embeds patches with a Conv3d; the shared patch embed's matmul is
    the same sum, rounded differently in bf16."""
    assert isinstance(Gr00tN17VisionEncoder(_config(())).tower.patch_embed, ConvPatchEmbed)
    # GR00T's patch, 3 x 2 x 16 x 16: at the tiny tower's 2 x 2 the two sums agree
    vision = dataclasses.replace(_config(()).vision, patch_size=16)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        embed = ConvPatchEmbed(vision).to(torch.bfloat16)
    # a request's patches: at a few dozen rows the two sums often agree outright
    x = torch.randn(1792, 3 * 2 * 16 * 16, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    conv = embed.proj(x.view(-1, 3, 2, 16, 16)).view(-1, vision.hidden_size)
    assert torch.equal(embed(x), conv)
    # the shared forward is a different function at this size, so the check above can fail
    assert not torch.equal(VisionPatchEmbed.forward(embed, x), conv)


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_vision_tensors_name_this_modules_parameters():
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):  # the real 24-block tower, allocated nowhere
        encoder = Gr00tN17VisionEncoder(config)
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    remap = _remapper("vision_encoder")
    assert {remap(name) for name in weight_map} - {None} == {name for name, _ in encoder.named_parameters()}
