"""GR00T N1.7's VL bridge (a LayerNorm, then four self-attention blocks) against an inline
reference; its ``vl_bridge`` node and the split of its rows into the denoise loop's two spans;
and, on recorded prefills, that the split is the reference's own ``input_ids == image_token_id``.
"""

import json
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.gr00t_n17.components.bridge import (
    BasicTransformerBlock,
    Gr00tN17VLBridge,
)
from mstar.model.gr00t_n17.components.llm import Gr00tN17LanguageModel
from mstar.model.gr00t_n17.config import (
    BRIDGE_ATTN,
    Gr00tN17Config,
    Gr00tN17StackConfig,
    Gr00tN17TextConfig,
)
from mstar.model.gr00t_n17.submodules import (
    Gr00tN17LLMSubmodule,
    Gr00tN17VLBridgeSubmodule,
)
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeSubmodule

from .helpers import BACKBONE, SegmentAttention, randomize, relative_gap

_BLOCKS = 4  # the checkpoint's `vl_self_attention_cfg.num_layers`
# spelled out rather than imported from the module under test: the point of the eps
# tests below is to catch this number changing there
_EPS = 1e-5

# head geometries, all tiny. The checkpoint's is 32 x 64; `head_dim` is what decides which
# kernel SDPA picks for the stand-in, so one case keeps the checkpoint's 64 and the other
# is narrow enough that its LayerNorms can be pushed to where eps matters.
_GEOMETRIES = {"the checkpoint's head width": (2, 64), "a narrow block": (2, 4)}


def _config(heads: int, head_dim: int, blocks: int = _BLOCKS) -> Gr00tN17Config:
    """A tiny bridge. ``backbone_embedding_dim`` is the block width because the blocks run
    on the backbone's rows unprojected -- the residuals would not add otherwise."""
    bridge = Gr00tN17StackConfig(
        num_layers=blocks, num_attention_heads=heads, attention_head_dim=head_dim,
    )
    return Gr00tN17Config(
        text=None, vision=None, dit=None, bridge=bridge,
        deepstack_visual_indexes=(), backbone_embedding_dim=bridge.inner_dim,
    )


def _rows(tokens: int, dim: int, seed: int, spread: float = 1.0) -> torch.Tensor:
    """The backbone's rows. ``spread`` shrinks their variation about a constant row, which
    is how the eps test gets a LayerNorm into the range where 1e-5 and 1e-6 differ."""
    generator = torch.Generator().manual_seed(seed)
    return 1.0 + spread * torch.randn(tokens, dim, generator=generator)


def _bind(bridge: Gr00tN17VLBridge, resources: dict) -> None:
    """Bind the way the engine does: on the node, fanned out to the blocks underneath it."""
    node = nn.Module()
    node.bridge = bridge
    NodeSubmodule.bind_node_resources(node, resources)


# ----------------------------------------------------------------------------
# Inline reference: the bridge, written from its description
# ----------------------------------------------------------------------------

# Deliberate deviations, each a mistake that would otherwise look like the real bridge.
_WRONG_NO_ATTN_RESIDUAL = "the attention residual dropped"
_WRONG_NO_FF_RESIDUAL = "the feed-forward residual dropped"
_WRONG_NORMS_SWAPPED = "norm1 and norm3 swapped"
DEVIATIONS = (_WRONG_NO_ATTN_RESIDUAL, _WRONG_NO_FF_RESIDUAL, _WRONG_NORMS_SWAPPED)


def _activate(x: torch.Tensor, activation: str) -> torch.Tensor:
    """"gelu-approximate" is the real one; "gelu" is the exact gelu a different
    ``activation_fn`` would put between the feed-forward's two linears."""
    if activation == "gelu-approximate":
        return F.gelu(x, approximate="tanh")
    assert activation == "gelu"
    return F.gelu(x)


class RefBlock(nn.Module):
    """``norm1 -> attn -> +x -> norm3 -> ff -> +x``, with q, k and v as three separate
    projections and a plain softmax where ours calls the ragged resource, and the
    feed-forward as one projection out to ``4 * dim``, an activation, and one back."""

    def __init__(
        self,
        config: Gr00tN17StackConfig,
        eps: float = _EPS,
        deviation: str | None = None,
        activation: str = "gelu-approximate",
    ):
        super().__init__()
        dim = config.inner_dim
        self.config = config
        self.deviation = deviation
        self.activation = activation
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.proj_in = nn.Linear(dim, 4 * dim)
        self.proj_out = nn.Linear(4 * dim, dim)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        shape = (tokens, self.config.num_attention_heads, self.config.attention_head_dim)
        q = self.to_q(x).view(shape).transpose(0, 1)
        k = self.to_k(x).view(shape).transpose(0, 1)
        v = self.to_v(x).view(shape).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * self.config.attention_head_dim**-0.5
        out = (scores.softmax(dim=-1) @ v).transpose(0, 1)
        return self.to_out(out.reshape(tokens, -1))

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(_activate(self.proj_in(x), self.activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swapped = self.deviation == _WRONG_NORMS_SWAPPED
        attended = self._attention((self.norm3 if swapped else self.norm1)(x))
        x = attended if self.deviation == _WRONG_NO_ATTN_RESIDUAL else attended + x
        ff = self.feed_forward((self.norm1 if swapped else self.norm3)(x))
        return ff if self.deviation == _WRONG_NO_FF_RESIDUAL else ff + x


class RefBridge(nn.Module):
    def __init__(self, config: Gr00tN17Config, eps: float = _EPS, **knobs):
        super().__init__()
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim, eps=eps)
        self.blocks = nn.ModuleList(
            RefBlock(config.bridge, eps, **knobs) for _ in range(config.bridge.num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vlln(x)
        for block in self.blocks:
            x = block(x)
        return x


@torch.no_grad()
def _reference(ours: Gr00tN17VLBridge, config: Gr00tN17Config, **knobs) -> RefBridge:
    """The same weights, one tensor per separate projection."""
    reference = RefBridge(config, **knobs)
    reference.vlln.load_state_dict(ours.vlln.state_dict())
    for ref, block in zip(reference.blocks, ours.vl_self_attention.transformer_blocks, strict=True):
        for name in ("norm1", "norm3"):
            getattr(ref, name).load_state_dict(getattr(block, name).state_dict())
        for name in ("to_q", "to_k", "to_v"):
            getattr(ref, name).load_state_dict(getattr(block.attn1, name).state_dict())
        ref.to_out.load_state_dict(block.attn1.to_out[0].state_dict())
        ref.proj_in.load_state_dict(block.ff.net[0].proj.state_dict())
        ref.proj_out.load_state_dict(block.ff.net[2].state_dict())
    return reference


@torch.no_grad()
def _bridge(config: Gr00tN17Config, tokens: int, seed: int = 0) -> Gr00tN17VLBridge:
    """Ours, filled, with the attention resource bound the way the engine binds it: one
    segment covering the whole step."""
    bridge = Gr00tN17VLBridge(config)
    randomize(bridge, seed, scale=0.5)  # halved so four residual blocks don't run away
    _bind(bridge, {BRIDGE_ATTN: SegmentAttention((tokens,))})
    return bridge


# ----------------------------------------------------------------------------
# The bridge against the reference
# ----------------------------------------------------------------------------

# Not exact: the stand-in's SDPA blocks its reductions differently from the reference's
# written-out matmul-softmax-matmul. Worst measured over both geometries x 5 token counts
# (1 to 300) x 5 weight seeds x 1/2/4 threads: 7.81e-07 relative, ~6 fp32 ulps after four
# blocks.
_MEASURED_GAP = 7.815e-07
# 4x for other BLAS/SDPA blockings; the closest wiring mistake misses by 3.6e+05x. A wrong
# LayerNorm eps does *not* clear it on ordinary rows, hence the eps test's narrow rows.
_BAR = 4 * _MEASURED_GAP


@pytest.mark.parametrize("geometry", list(_GEOMETRIES.values()), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("tokens", [1, 17, 300])
def test_our_bridge_is_a_norm_and_four_reference_blocks(geometry, tokens):
    config = _config(*geometry)
    ours = _bridge(config, tokens)
    rows = _rows(tokens, config.backbone_embedding_dim, seed=1)

    out = ours(rows)
    assert out.shape == rows.shape
    assert relative_gap(out, _reference(ours, config)(rows)) <= _BAR


@pytest.mark.parametrize("deviation", DEVIATIONS)
def test_a_block_assembled_wrongly_misses_by_orders_of_magnitude(deviation):
    """A plausible mistake cannot slip under the bar: measured attention residual dropped
    2.8e+00, feed-forward residual dropped 2.5e+00, norm1/norm3 swapped 1.1e+00 (>= 3.6e+05x)."""
    config = _config(*_GEOMETRIES["the checkpoint's head width"])
    ours = _bridge(config, tokens=17)
    rows = _rows(17, config.backbone_embedding_dim, seed=1)
    theirs = _reference(ours, config, deviation=deviation)(rows)

    assert relative_gap(ours(rows), theirs) > 1000 * _BAR


def test_the_layer_norms_use_eps_1e_5_and_not_1e_6():
    """eps is added to a variance, so 1e-5 and 1e-6 only part when the variance is that
    small: on ordinary rows a 1e-6 build sits within 0.6-5x the bar (asserted below), so the
    rows here are 8 wide with variance ~1e-6, which separates them by 2.3e-01 (7e+04x)."""
    config = _config(*_GEOMETRIES["a narrow block"])
    ours = _bridge(config, tokens=8)
    ordinary = _rows(8, config.backbone_embedding_dim, seed=1)
    near_constant = _rows(8, config.backbone_embedding_dim, seed=1, spread=1e-3)
    wrong = _reference(ours, config, eps=1e-6)

    # the measurement the rest of this test rests on: on ordinary rows the wrong eps sits
    # in the noise, within a few multiples of the bar rather than orders of magnitude off
    assert relative_gap(ours(ordinary), wrong(ordinary)) <= 10 * _BAR
    assert relative_gap(ours(near_constant), _reference(ours, config)(near_constant)) <= _BAR
    assert relative_gap(ours(near_constant), wrong(near_constant)) > 1000 * _BAR


# ----------------------------------------------------------------------------
# The feed-forward, which is diffusers' own
# ----------------------------------------------------------------------------


def test_the_feed_forward_projects_out_to_four_times_the_width_ungated():
    """The checkpoint's ``net.0.proj`` is 8192 x 2048 on a 2048-wide block, the ungated shape
    (a ``geglu`` projection is twice as wide). Both ends are pinned: a gated variant at half
    the inner width matches ``net.0`` and differs only at ``net.2``."""
    config = _config(*_GEOMETRIES["the checkpoint's head width"]).bridge
    dim = config.inner_dim
    ff = BasicTransformerBlock(config).ff

    assert ff.net[0].proj.weight.shape == (4 * dim, dim)
    assert ff.net[2].weight.shape == (dim, 4 * dim)


@pytest.mark.parametrize(
    "activation", ["gelu-approximate", "gelu"], ids=["the tanh gelu", "the exact gelu"],
)
@torch.no_grad()
def test_the_feed_forward_is_one_activation_between_two_linears(activation):
    """A swapped activation loads the checkpoint and runs, so the arithmetic is pinned: the
    real one is ``torch.equal`` (the same two linears and ``F.gelu``), the exact gelu is not."""
    config = _config(*_GEOMETRIES["a narrow block"])
    block = BasicTransformerBlock(config.bridge)
    randomize(block, 2, scale=0.5)
    reference = RefBlock(config.bridge, activation=activation)
    reference.proj_in.load_state_dict(block.ff.net[0].proj.state_dict())
    reference.proj_out.load_state_dict(block.ff.net[2].state_dict())
    x = _rows(6, config.bridge.inner_dim, seed=3)

    if activation == "gelu-approximate":
        assert torch.equal(block.ff(x), reference.feed_forward(x))
    else:
        assert not torch.equal(block.ff(x), reference.feed_forward(x))


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


def test_a_block_has_no_norm2_and_carries_the_reference_names():
    """``norm2`` belongs to a cross-attention this block lacks, and would take a weight the
    checkpoint has not got; ``to_out.0`` counts diffusers' dropout after the projection."""
    block = BasicTransformerBlock(_config(*_GEOMETRIES["a narrow block"]).bridge)
    names = {name for name, _ in block.named_parameters()}

    assert names == {
        f"{module}.{tensor}"
        for module in ("norm1", "norm3", "attn1.to_q", "attn1.to_k", "attn1.to_v",
                       "attn1.to_out.0", "ff.net.0.proj", "ff.net.2")
        for tensor in ("weight", "bias")
    }
    assert not any("norm2" in name for name in names)


# ----------------------------------------------------------------------------
# The attention resource
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resources", [{}, {"vision_attn": object()}], ids=["nothing bound", "the wrong key"],
)
def test_an_unbound_attention_names_the_resource_it_wanted(resources):
    """The bridge's 32 x 64 heads cannot share the tower's 16 x 64 resource; a mis-keyed
    config is a typo, and the message names the key the block looked for."""
    config = _config(*_GEOMETRIES["a narrow block"], blocks=1)
    bridge = Gr00tN17VLBridge(config)
    _bind(bridge, resources)

    with pytest.raises(RuntimeError, match=BRIDGE_ATTN):
        bridge(torch.zeros(2, config.backbone_embedding_dim))


def test_binding_the_node_reaches_every_blocks_attention():
    config = _config(*_GEOMETRIES["a narrow block"])
    bridge = Gr00tN17VLBridge(config)
    attention = SegmentAttention((2,))
    _bind(bridge, {BRIDGE_ATTN: attention})

    blocks = bridge.vl_self_attention.transformer_blocks
    assert [block.attn1.ragged_attn for block in blocks] == [attention] * _BLOCKS


# ----------------------------------------------------------------------------
# The node: `Gr00tN17VLBridgeSubmodule`, driven the way the engine drives it
# ----------------------------------------------------------------------------

_WALK = "prefill"
_FWD = CurrentForwardPassInfo(
    request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0,
)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})
# what `prepare_inputs` stages, which is also `forward`'s signature
_STAGED = {"backbone_features", "image_rows"}

# Runs of text and image rows down a prompt. Two non-contiguous image blocks, as two cameras
# give, so no span can be reversed or rolled back onto itself; the two degenerate splits; and
# as many image rows as text rows, where swapping the spans changes no shape at all.
_PROMPTS = {
    "no image rows": (("text", 9),),
    "two image blocks with text between": (
        ("text", 2), ("image", 4), ("text", 3), ("image", 6), ("text", 2),
    ),
    "every row an image": (("image", 7),),
    "as many image rows as text": (("text", 2), ("image", 3), ("text", 3), ("image", 2)),
}


def _prompt(spec: tuple) -> tuple[int, list[int]]:
    """A spec of runs -> the prompt's row count and its image rows, in prompt order."""
    rows: int = 0
    image: list[int] = []
    for kind, length in spec:
        if kind == "image":
            image += range(rows, rows + length)
        rows += length
    return rows, image


def _node(seed: int = 0) -> Gr00tN17VLBridgeSubmodule:
    """Ours, filled. The resource is bound in `_drive`, from the step's own segments."""
    config = _config(*_GEOMETRIES["the checkpoint's head width"])
    bridge = Gr00tN17VLBridge(config)
    randomize(bridge, seed, scale=0.5)
    return Gr00tN17VLBridgeSubmodule(bridge, config)


class _Run(NamedTuple):
    """One pass of the node, with what the walk handed it and what came back."""

    node: Gr00tN17VLBridgeSubmodule
    features: torch.Tensor
    image: list[int]
    text: list[int]
    step: SubmoduleStep
    tensors: dict
    out: dict

    @torch.no_grad()
    def by_hand(self) -> torch.Tensor:
        """The bridge called directly on the same rows: what the node's two spans are cut
        from, and what scattering them back has to reproduce."""
        return self.node.model(self.features)


@torch.no_grad()
def _drive_rows(rows: int, image: list[int], seed: int = 1) -> _Run:
    """prepare_inputs -> declare_step's resource -> preprocess -> forward, as the engine
    runs it, over ``rows`` backbone rows of which ``image`` are the image rows."""
    node = _node()
    features = _rows(rows, node.config.backbone_embedding_dim, seed=seed)
    walk = {
        "backbone_features": [features],
        "image_rows": [torch.tensor(image, dtype=torch.long)],
    }
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    # the engine plans the resource from the declaration, so the stand-in is planned from it
    step = node.declare_step(_WALK, ["r0"], [prepared])
    node.bind_node_resources(
        {BRIDGE_ATTN: SegmentAttention(tuple(seg.span for seg in step.segments))}
    )
    tensors = node.preprocess(_WALK, _ENGINE, [prepared])
    text = [row for row in range(rows) if row not in set(image)]
    return _Run(
        node, features, image, text, step, tensors,
        node.forward(_WALK, _ENGINE, **tensors),
    )


def _drive(spec: tuple, seed: int = 1) -> _Run:
    """``_drive_rows`` over a spec of runs. The image rows are derived from the spec, so the
    split is checked against the prompt rather than against the mask the node built."""
    return _drive_rows(*_prompt(spec), seed=seed)


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_the_step_is_one_non_causal_segment_over_the_whole_prompt(spec):
    """One non-causal segment over all of a prompt's rows (the vision node cuts one per frame,
    the LLM node attends causally); it lives only in the declaration, so this is where it is said."""
    node = _node()
    rows, image = _prompt(spec)
    prepared = node.prepare_inputs(
        _WALK, _FWD,
        {
            "backbone_features": [_rows(rows, node.config.backbone_embedding_dim, seed=1)],
            "image_rows": [torch.tensor(image, dtype=torch.long)],
        },
    )
    step = node.declare_step(_WALK, ["r0"], [prepared])

    assert step.segments == [Segment("r0", "main", rows)]
    assert list(step.steps) == [BRIDGE_ATTN]
    assert step.steps[BRIDGE_ATTN].causal is False
    # what the resource is planned from: `SubmoduleStep` fills it in from `segments`
    assert step.steps[BRIDGE_ATTN].segments == step.segments
    # no batched forward, so a wider batch would only be the engine's per-request fallback
    assert node.max_batch_size(_WALK) == 1


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_the_split_is_the_mask_and_keeps_every_row_in_prompt_order(spec):
    """prepare_inputs -> preprocess -> forward is the bridge over the walk's rows, cut into
    ``vl_image`` (the image rows) and ``vl_text`` (the rest), each in prompt order, nothing
    dropped or duplicated: gathers, so ``torch.equal``. Order is pinned so a later sort or
    grouping, invisible to a set check, still fails."""
    run = _drive(spec)
    rows = run.by_hand()

    # preprocess stages the two tensors and nothing else
    assert set(run.tensors) == _STAGED
    assert torch.equal(run.tensors["backbone_features"], run.features)
    assert run.tensors["image_rows"].tolist() == run.image
    assert list(run.out) == ["vl_text", "vl_image"]
    assert torch.equal(run.out["vl_image"][0], rows[run.image])
    assert torch.equal(run.out["vl_text"][0], rows[run.text])

    # nothing lost: the two spans scattered back onto their own rows are the output again
    reconstructed = torch.empty_like(rows)
    reconstructed[run.image] = run.out["vl_image"][0]
    reconstructed[run.text] = run.out["vl_text"][0]
    assert torch.equal(reconstructed, rows)

    # the rows differ down each span, which is what makes the equalities above bite: a
    # reversed or rolled span would not match
    for span in (run.out["vl_image"][0], run.out["vl_text"][0]):
        assert span.shape[0] < 2 or not torch.equal(span, span.flip(0))
    # with equal spans a swap changes no shape, so only their contents tell them apart
    if len(run.image) == len(run.text):
        assert not torch.equal(run.out["vl_text"][0], run.out["vl_image"][0])


def test_more_image_rows_than_feature_rows_is_refused_where_they_arrive():
    """The image rows come from the LLM node, which saw the prompt; more of them than feature
    rows is a disagreement between the nodes, refused on arrival instead of an out-of-range
    ``index_fill_`` mid-step."""
    node = _node()
    features = _rows(2, node.config.backbone_embedding_dim, seed=1)

    with pytest.raises(ValueError, match=r"3 image row\(s\) for 2 backbone feature row\(s\)"):
        node.prepare_inputs(
            _WALK, _FWD,
            {"backbone_features": [features], "image_rows": [torch.arange(3)]},
        )


# ----------------------------------------------------------------------------
# What the two spans are: the reference's predicate, over recorded prefills
# ----------------------------------------------------------------------------

# Twenty prefills recorded off the reference, outside the checkpoint download.
_DUMP = Path.home() / "gr00t-runs/dump"
_RECORDED = sorted(_DUMP.glob("req_*.pt")) if _DUMP.is_dir() else []
_HAVE_RECORDED = bool(_RECORDED) and BACKBONE is not None


def _reference_image_token_id() -> int:
    """The id the reference's mask compares against, ``input_ids == image_token_id``
    (``gr00t/model/modules/qwen3_backbone.py:364``), read off the backbone's config rather
    than ours."""
    return json.loads((Path(BACKBONE) / "config.json").read_text())["image_token_id"]


class _Request(NamedTuple):
    """One recorded prefill, with the reference's predicate already applied to its prompt."""

    input_ids: torch.Tensor  # [tokens]
    image_grid_thw: torch.Tensor  # [frames, 3]: the camera set the prompt was built from
    bridge_rows: int  # rows in the recorded `vl_embeds`, which is the bridge's own output
    is_image: torch.Tensor  # [tokens], bool


def _request(path: Path) -> _Request:
    """``mmap``ped: only the prompt, the grids and one shape are read out of tens of MB."""
    dump = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    input_ids = dump["collated"]["inputs"]["input_ids"].reshape(-1)
    return _Request(
        input_ids=input_ids,
        image_grid_thw=dump["collated"]["inputs"]["image_grid_thw"].reshape(-1, 3),
        bridge_rows=dump["vl_embeds"].shape[1],
        is_image=input_ids == _reference_image_token_id(),
    )


def _image_rows(request: _Request) -> torch.Tensor:
    """The image rows as the ``LLM`` node derives them (the bridge never sees a prompt). Only
    ``prepare_inputs`` runs, so the text stack is a one-layer stand-in; the spatial merge is
    the real one, which the M-RoPE grids are measured in."""
    backbone = json.loads((Path(BACKBONE) / "config.json").read_text())
    text = Gr00tN17TextConfig(
        hidden_size=8, intermediate_size=16, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_hidden_layers=1, rms_norm_eps=1e-6, rope_theta=10000.0,
        vocab_size=1, mrope_section=(2, 1, 1), mrope_interleaved=True,
        image_token_id=_reference_image_token_id(), vision_start_token_id=0,
    )
    # the one vision field the node reads, and it has to be the real one: it is what the
    # M-RoPE grids are measured in, so a stand-in would make the recorded grids disagree
    # with the prompt's image runs and raise
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=1, in_channels=3,
        patch_size=2, temporal_patch_size=2, out_hidden_size=8, num_position_embeddings=16,
        spatial_merge_size=backbone["vision_config"]["spatial_merge_size"],
    )
    config = Gr00tN17Config(
        text=text, vision=vision, dit=None, bridge=None,
        # one tap, because the node checks the count it is sent against this and nothing
        # here reads the features
        deepstack_visual_indexes=(0,), backbone_embedding_dim=0,
    )
    node = Gr00tN17LLMSubmodule(Gr00tN17LanguageModel(config), config)
    rows = int(request.is_image.sum())
    prepared = node.prepare_inputs(
        _WALK, _FWD,
        {
            "input_ids": [request.input_ids],
            "image_grid_thw": [request.image_grid_thw],
            "vision_embeds": [torch.zeros(rows, text.hidden_size)],
            "deepstack_embeds": [torch.zeros(rows, text.hidden_size)],
        },
    )
    return prepared.tensor_inputs["image_rows"]


@pytest.mark.skipif(not _HAVE_RECORDED, reason="recorded requests or backbone not cached")
@pytest.mark.parametrize("path", _RECORDED, ids=[path.stem for path in _RECORDED])
@torch.no_grad()
def test_the_split_is_the_references_image_token_predicate(path):
    """On prompts the reference ran: ``vl_image`` is exactly the rows ``input_ids ==
    image_token_id`` selects and ``vl_text`` exactly the rest, with the ``LLM`` node deriving the
    rows as in the engine. A swap changes no shape and costs 31% relative error at full depth.
    Only the row pattern (448 image rows in four blocks) comes from the recording; the rows are
    this file's narrow stand-ins, since the split is a gather."""
    request = _request(path)
    image_rows = _image_rows(request)
    run = _drive_rows(request.input_ids.numel(), image_rows.tolist())
    rows = run.by_hand()

    # the handoff between the two nodes: what the LLM node sends is the predicate's rows
    assert torch.equal(image_rows, request.is_image.nonzero(as_tuple=True)[0])
    # and the recording agrees that the bridge emits one row per prompt token, which is
    # what makes a mask over the prompt a mask over the bridge's output
    assert request.bridge_rows == request.input_ids.numel()

    assert torch.equal(run.out["vl_image"][0], rows[request.is_image])
    assert torch.equal(run.out["vl_text"][0], rows[~request.is_image])
