"""GR00T N1.7's VL bridge: a LayerNorm over the backbone's features, then four
self-attention blocks.

The bridge is checked against an inline reference written from the algorithm -- a norm,
an attention with three separate projections and a plain softmax, a residual, a second
norm, a feed-forward, a second residual -- so the test needs no checkpoint and no
diffusers block. Only the feed-forward *is* diffusers', and it is pinned from both sides:
the width of its projection (a gated variant's is twice as wide) and the arithmetic
between the two linears.

The rest is what the reference cannot see: that the parameters carry the checkpoint's
names and no others, and that the ragged attention resource reaches every block.

The next section drives the ``vl_bridge`` node, ``Gr00tN17VLBridgeSubmodule``, the way the
engine does: the step it declares, the split of the bridge's rows into the denoise loop's
two spans, and that it adds nothing to the forward above.

The last section says what those two spans *are*. The reference's own predicate --
``input_ids == image_token_id`` -- is the definition, so it is read off the backbone's
``config.json`` and checked over twenty recorded prefills, with the ``LLM`` node deriving
the image rows as it does in the engine. Those tests skip without the recordings.
"""

import glob
import json
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from safetensors import safe_open
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
from mstar.model.gr00t_n17.weight_loader import _remapper
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeSubmodule

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

_BLOCKS = 4  # the checkpoint's `vl_self_attention_cfg.num_layers`
# spelled out rather than imported from the module under test: the point of the eps
# tests below is to catch this number changing there
_EPS = 1e-5

# head geometries, all tiny. The checkpoint's is 32 x 64; `head_dim` is what decides which
# kernel SDPA picks for the stand-in, so one case keeps the checkpoint's 64 and the other
# is narrow enough that its LayerNorms can be pushed to where eps matters.
_GEOMETRIES = {"the checkpoint's head width": (2, 64), "a narrow block": (2, 4)}


@pytest.fixture(autouse=True)
def _pinned_threads():
    """fp32 CPU forwards compared against a measured bar; reduction order follows the
    thread count, as it does in the other gr00t tests."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


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


def _randomize(module: nn.Module, seed: int) -> None:
    """LayerNorm initialises to the identity affine, which would hide a norm wired up in
    the wrong place. Halved so four residual blocks don't run away."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))


def _rows(tokens: int, dim: int, seed: int, spread: float = 1.0) -> torch.Tensor:
    """The backbone's rows. ``spread`` shrinks their variation about a constant row, which
    is how the eps test gets a LayerNorm into the range where 1e-5 and 1e-6 differ."""
    generator = torch.Generator().manual_seed(seed)
    return 1.0 + spread * torch.randn(tokens, dim, generator=generator)


def _relative_gap(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return ((ours - theirs).abs().max() / theirs.abs().max()).item()


# ----------------------------------------------------------------------------
# The ragged attention resource's stand-in
# ----------------------------------------------------------------------------


class _SegmentAttention:
    """Stands in for the resource the engine plans from the step's segments: one segment
    per request, each attending only within itself. The bridge is unmasked, so no
    causality here -- every row of a request sees every other row of it.
    """

    def __init__(self, seq_lengths: tuple[int, ...]):
        self.seq_lengths = seq_lengths

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, start = [], 0
        for span in self.seq_lengths:
            rows = slice(start, start + span)
            # the kernel's layout is [tokens, heads, dim]; SDPA wants heads first
            out.append(
                F.scaled_dot_product_attention(
                    q[rows].transpose(0, 1), k[rows].transpose(0, 1), v[rows].transpose(0, 1),
                ).transpose(0, 1)
            )
            start += span
        return torch.cat(out)


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
    """"gelu-approximate" is the real one; the rest are what a different ``activation_fn``
    would put between the feed-forward's two linears at the same width."""
    if activation == "gelu-approximate":
        return F.gelu(x, approximate="tanh")
    if activation == "gelu":
        return F.gelu(x)
    if activation == "silu":
        return F.silu(x)
    assert activation == "none"
    return x


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
    _randomize(bridge, seed)
    _bind(bridge, {BRIDGE_ATTN: _SegmentAttention((tokens,))})
    return bridge


# ----------------------------------------------------------------------------
# The bridge against the reference
# ----------------------------------------------------------------------------

# Not exact, and the reason is the stand-in: ours reaches the attention through
# `F.scaled_dot_product_attention`, where the reference writes the matmul-softmax-matmul
# out. The two disagree by a few ulps on the attention output -- not over the softmax's
# form, which is irrelevant here (scaling q, scaling the scores and subtracting the row
# max all give bit-identical answers), but over how SDPA's fused CPU kernel blocks its
# reductions. At head_dim 16 it happens to agree with `@` exactly; at 4 and at the
# checkpoint's 64 it does not.
#
# Measured over both geometries x 5 token counts (1 to 300) x 5 weight seeds x intra-op
# thread counts 1, 2 and 4, the worst max|ours - ref| / max|ref| was 7.81e-07, about 6 fp32
# ulps after four blocks.
_MEASURED_GAP = 7.815e-07
# Four times that, so a BLAS or an SDPA backend blocking differently from this host's
# still passes. It stays far under a wiring mistake: the closest deviation below misses
# by 3.6e+05 times this bar. It is *not* far under a wrong LayerNorm eps, which is what
# the eps test below has to work around.
_BAR = 4 * _MEASURED_GAP


@pytest.mark.parametrize("geometry", list(_GEOMETRIES.values()), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("tokens", [1, 17, 300])
def test_our_bridge_is_a_norm_and_four_reference_blocks(geometry, tokens):
    config = _config(*geometry)
    ours = _bridge(config, tokens)
    rows = _rows(tokens, config.backbone_embedding_dim, seed=1)

    out = ours(rows)
    assert out.shape == rows.shape
    assert _relative_gap(out, _reference(ours, config)(rows)) <= _BAR


@pytest.mark.parametrize("deviation", DEVIATIONS)
def test_a_block_assembled_wrongly_misses_by_orders_of_magnitude(deviation):
    """The bar above only means something if a plausible mistake cannot slip under it.
    Either residual carries the input forward unchanged, and the two norms have their own
    affines, so swapping which one feeds which half is a third function again.

    Measured relative gaps, against a bar of 3.1e-06: the attention residual dropped
    2.8e+00, the feed-forward residual dropped 2.5e+00, norm1 and norm3 swapped 1.1e+00 --
    the smallest of them 3.6e+05 times the bar.
    """
    config = _config(*_GEOMETRIES["the checkpoint's head width"])
    ours = _bridge(config, tokens=17)
    rows = _rows(17, config.backbone_embedding_dim, seed=1)
    theirs = _reference(ours, config, deviation=deviation)(rows)

    assert _relative_gap(ours(rows), theirs) > 1000 * _BAR


def test_the_layer_norms_use_eps_1e_5_and_not_1e_6():
    """eps is added to a variance, so the two values only part company when the variance is
    of their own size. On ordinary rows a 1e-6 build is *not* caught: measured over 8 weight
    seeds and both geometries it misses by 0.6 to 5 times the parity bar, which is the
    kernel's own noise band -- it would slip through the parity test above on some seeds and
    fail it on others, which is no test at all.

    So the rows here are narrow (8 wide) and barely vary (1 + 1e-3 * randn, variance ~1e-06, the size of the
    epsilons themselves). That separates them by 2.3e-01, 7e+04 times the bar.
    """
    config = _config(*_GEOMETRIES["a narrow block"])
    ours = _bridge(config, tokens=8)
    ordinary = _rows(8, config.backbone_embedding_dim, seed=1)
    near_constant = _rows(8, config.backbone_embedding_dim, seed=1, spread=1e-3)
    wrong = _reference(ours, config, eps=1e-6)

    # the measurement the rest of this test rests on: on ordinary rows the wrong eps sits
    # in the noise, within a few multiples of the bar rather than orders of magnitude off
    assert _relative_gap(ours(ordinary), wrong(ordinary)) <= 10 * _BAR
    assert _relative_gap(ours(near_constant), _reference(ours, config)(near_constant)) <= _BAR
    assert _relative_gap(ours(near_constant), wrong(near_constant)) > 1000 * _BAR


# ----------------------------------------------------------------------------
# The feed-forward, which is diffusers' own
# ----------------------------------------------------------------------------


def test_the_feed_forward_projects_out_to_four_times_the_width_ungated():
    """A gated ``FeedForward`` -- the ``geglu`` default -- splits its projection in two and
    multiplies the halves, so it needs twice the rows for the same inner width. The
    checkpoint's ``net.0.proj`` is 8192 x 2048 on a 2048-wide block, which is the ungated
    shape; the gated one would want 16384.

    Both ends are pinned, because a gated variant built at half the inner width has a
    ``net.0.proj`` of exactly our shape and gives itself away only at ``net.2``.
    """
    config = _config(*_GEOMETRIES["the checkpoint's head width"]).bridge
    dim = config.inner_dim
    ff = BasicTransformerBlock(config).ff

    assert ff.net[0].proj.weight.shape == (4 * dim, dim)
    assert ff.net[2].weight.shape == (dim, 4 * dim)
    gated = FeedForward(dim, activation_fn="geglu", final_dropout=True)
    assert gated.net[0].proj.weight.shape == (8 * dim, dim)
    narrow_gated = FeedForward(dim, activation_fn="geglu", inner_dim=2 * dim, final_dropout=True)
    assert narrow_gated.net[0].proj.weight.shape == ff.net[0].proj.weight.shape
    assert narrow_gated.net[2].weight.shape != ff.net[2].weight.shape


@pytest.mark.parametrize(
    "activation", ["gelu-approximate", "gelu", "silu", "none"],
    ids=["the tanh gelu", "the exact gelu", "silu", "no activation at all"],
)
@torch.no_grad()
def test_the_feed_forward_is_one_activation_between_two_linears(activation):
    """Same width, different function: an activation swapped for one of these would load
    the checkpoint and run, so the arithmetic is pinned as well as the shape. The real one
    is ``torch.equal`` against the reference -- it is the same two ``nn.Linear``s and the
    same ``F.gelu``, in the same order, so there is nothing for a bar to cover.
    """
    config = _config(*_GEOMETRIES["a narrow block"])
    block = BasicTransformerBlock(config.bridge)
    _randomize(block, seed=2)
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
    """``BasicTransformerBlock``'s ``norm2`` belongs to its cross-attention, which this
    setting does not have; a norm added there would take a weight the checkpoint has not
    got. ``to_out.0`` is the ``ModuleList``'s doing -- diffusers holds a dropout after the
    projection, and the checkpoint's key counts it.
    """
    block = BasicTransformerBlock(_config(*_GEOMETRIES["a narrow block"]).bridge)
    names = {name for name, _ in block.named_parameters()}

    assert names == {
        f"{module}.{tensor}"
        for module in ("norm1", "norm3", "attn1.to_q", "attn1.to_k", "attn1.to_v",
                       "attn1.to_out.0", "ff.net.0.proj", "ff.net.2")
        for tensor in ("weight", "bias")
    }
    assert not any("norm2" in name for name in names)


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_bridge_tensors_name_this_modules_parameters():
    """The node owns ``action_head.vlln.`` and ``action_head.vl_self_attention.``, and every
    name under them is the reference's, so the checkpoint's keys land on the module with
    only ``action_head.`` stripped. Shapes come off the safetensors header, so no tensor is
    read, and the module is built on meta -- it is 200M parameters at the real config.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):
        bridge = Gr00tN17VLBridge(config)
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    remap = _remapper("vl_bridge")
    checkpoint = {remap(name): (name, shard) for name, shard in weight_map.items() if remap(name)}

    assert set(checkpoint) == {name for name, _ in bridge.named_parameters()}
    for name, parameter in bridge.named_parameters():
        key, shard = checkpoint[name]
        with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
            assert list(parameter.shape) == shard_file.get_slice(key).get_shape()


# ----------------------------------------------------------------------------
# The attention resource
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resources", [{}, {"vision_attn": object()}], ids=["nothing bound", "the wrong key"],
)
def test_an_unbound_attention_names_the_resource_it_wanted(resources):
    """The bridge's heads are 32 x 64 where the tower's are 16 x 64, so it cannot share the
    tower's resource; a config that declares it under another name is a typo, and the
    message has to say which name the block looked for."""
    config = _config(*_GEOMETRIES["a narrow block"], blocks=1)
    bridge = Gr00tN17VLBridge(config)
    _bind(bridge, resources)

    with pytest.raises(RuntimeError, match=BRIDGE_ATTN):
        bridge(torch.zeros(2, config.backbone_embedding_dim))


def test_binding_the_node_reaches_every_blocks_attention():
    config = _config(*_GEOMETRIES["a narrow block"])
    bridge = Gr00tN17VLBridge(config)
    attention = _SegmentAttention((2,))
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

# Runs of text and image rows down a prompt. The mixed one puts the image rows in two
# non-contiguous blocks, as a real prompt's two cameras do, so neither span can be
# reversed or rolled back onto itself; one block would still line up with itself over
# most of its length. The other two are the degenerate splits.
_PROMPTS = {
    "no image rows": (("text", 9),),
    "two image blocks with text between": (
        ("text", 2), ("image", 4), ("text", 3), ("image", 6), ("text", 2),
    ),
    "every row an image": (("image", 7),),
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
    _randomize(bridge, seed)
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
        {BRIDGE_ATTN: _SegmentAttention(tuple(seg.span for seg in step.segments))}
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
    """Neither of the other two nodes' shapes. The vision node cuts a segment per frame;
    the LLM node cuts one per request but attends causally. The bridge is one segment over
    all of a prompt's rows, unmasked -- every row sees every other row of that prompt, and
    none of anyone else's. All of that lives only in the declaration, so this is the only
    place any of it is said.
    """
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
    """``vl_image`` is the bridge's image rows and ``vl_text`` the rest, each in prompt
    order, and the two together are the bridge's output with nothing dropped, duplicated or
    re-ordered. Gathers of the same rows, so this is ``torch.equal`` and not a bar.

    The order is pinned on purpose: a later change that sorted or grouped the rows would be
    invisible to a set-based check and would still break anyone lining our spans up against
    the reference's masked attention.
    """
    run = _drive(spec)
    rows = run.by_hand()

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


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_driving_the_node_is_the_bridge_called_directly_and_split_by_hand(spec):
    """The node is plumbing: ``prepare_inputs -> preprocess -> forward`` is the bridge over
    the walk's rows, cut in two. Same ops on the same tensors, so ``torch.equal``."""
    run = _drive(spec)
    rows = run.by_hand()

    # preprocess stages the two tensors and nothing else; there are no non-tensor kwargs
    assert set(run.tensors) == _STAGED
    assert torch.equal(run.tensors["backbone_features"], run.features)
    assert run.tensors["image_rows"].tolist() == run.image
    assert torch.equal(torch.cat([run.out["vl_text"][0], run.out["vl_image"][0]]),
                       torch.cat([rows[run.text], rows[run.image]]))
    assert run.out["vl_text"][0].shape == (len(run.text), rows.shape[1])
    assert run.out["vl_image"][0].shape == (len(run.image), rows.shape[1])


def test_more_image_rows_than_feature_rows_is_refused_where_they_arrive():
    """The LLM node derives the image rows from a prompt the bridge never sees, so a count
    that cannot fit the features it sent is a disagreement between the two nodes. Left to
    the forward it would be an ``index_fill_`` out of range mid-step; the mask is built from
    these, so it is checked where they arrive.
    """
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

# Twenty prefills recorded off the reference. They are not part of the checkpoint
# download, so these tests gate on the dump directory as well as on the backbone.
_DUMP = Path.home() / "gr00t-runs/dump"
_RECORDED = sorted(_DUMP.glob("req_*.pt")) if _DUMP.is_dir() else []
_HAVE_RECORDED = bool(_RECORDED) and _BACKBONE is not None

# Equal spans: as many image rows as text rows, in two blocks each. A swap of the two
# spans is invisible to every shape and every length here. The recorded prompts are 448
# image rows against 13, where the counts would give a swap away, but nothing says a
# prompt always has more of one than the other.
_BALANCED = (("text", 2), ("image", 3), ("text", 3), ("image", 2))


def _reference_image_token_id() -> int:
    """The id the reference's predicate compares against.

    ``Qwen3Backbone.forward`` builds the mask the DT's cross-attentions are masked with as
    ``vl_input["input_ids"] == self.model.config.image_token_id``
    (``gr00t/model/modules/qwen3_backbone.py:364``), and its ``self.model`` is the backbone
    loaded from this path -- so the expectation comes off that config and not off ours.
    That our config carries the same number is ``test_gr00t_n17_config``'s to say.
    """
    return json.loads((Path(_BACKBONE) / "config.json").read_text())["image_token_id"]


class _Request(NamedTuple):
    """One recorded prefill, with the reference's predicate already applied to its prompt."""

    input_ids: torch.Tensor  # [tokens]
    image_grid_thw: torch.Tensor  # [frames, 3]: the camera set the prompt was built from
    bridge_rows: int  # rows in the recorded `vl_embeds`, which is the bridge's own output
    is_image: torch.Tensor  # [tokens], bool


def _request(path: Path) -> _Request:
    """``mmap``ped: a recording is tens of MB of pixels and video, and only the prompt,
    the grids and one shape are read out of it."""
    dump = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    input_ids = dump["collated"]["inputs"]["input_ids"].reshape(-1)
    return _Request(
        input_ids=input_ids,
        image_grid_thw=dump["collated"]["inputs"]["image_grid_thw"].reshape(-1, 3),
        bridge_rows=dump["vl_embeds"].shape[1],
        is_image=input_ids == _reference_image_token_id(),
    )


def _image_rows(request: _Request) -> torch.Tensor:
    """The image rows as the ``LLM`` node derives them, which is where the bridge's come
    from: the bridge never sees a prompt, so the two nodes together are the producer.

    Only ``prepare_inputs`` is driven, and it reads the prompt, the grids and
    ``image_token_id``. The language model is there for the device it answers with and the
    two vision entries for the tap-count check, so all three are stand-ins -- the text
    stack is one tiny layer and no forward runs on it.
    """
    backbone = json.loads((Path(_BACKBONE) / "config.json").read_text())
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
    """What the two spans *are*, on prompts the reference actually ran.

    The reference masks each DT cross-attention with ``input_ids == image_token_id`` or
    with its complement, alternating by block; we separate the rows once instead, so
    ``vl_image`` has to be exactly the rows that predicate selects and ``vl_text`` exactly
    the rest. If this fails the DT reads image rows where it should read text and text
    where it should read image -- a swap that changes no shape, raises nothing, and costs
    31% relative error at full depth.

    The two gathers are also the partition: ``is_image`` is a mask over every row and the
    second gather is its complement, so between them each row lands in one span exactly
    once, with its contents and its position inside the span intact.

    Both ends of the chain are in it. The bridge never sees a prompt, so the ``LLM`` node
    derives the image rows here as it does in the engine, and the expectation is the
    backbone's own ``image_token_id`` rather than anything our config did with it.

    Only the row *pattern* comes from the recording: 448 image rows in four blocks with a
    handful of text rows around them, which no prompt above resembles. The rows themselves
    are this file's narrow stand-ins -- the split is a gather and does not depend on the
    width, and the real 2048-wide bridge is 200M parameters.

    Order inside a span is not load-bearing: the loop cross-attends a whole span with no
    mask and softmax does not care in what order the keys arrive, which is D03's premise.
    It is pinned anyway, because a gather in prompt order is the cheapest way to say that
    no row moved, and because anyone lining our spans up against the reference's masked
    attention needs it.
    """
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


def test_a_swap_of_the_two_spans_survives_everything_but_the_contents():
    """Why the assertions above are gathers and not shapes.

    Both spans are the bridge's width -- 2048 in the checkpoint -- so exchanging
    ``vl_text`` and ``vl_image`` in the forward changes no width, no dtype and no key
    order, and raises nothing; on this prompt it does not change a row count either. The
    contents are all that separates them.

    If it fails, every ``vl_text``/``vl_image`` assertion in this file could be passing on
    a producer that has the two the wrong way round.
    """
    run = _drive(_BALANCED)
    rows = run.by_hand()
    text, image = run.out["vl_text"][0], run.out["vl_image"][0]

    assert (text.shape, text.dtype) == (image.shape, image.dtype)
    # which only says something about the split because the two are different rows
    assert not torch.equal(text, image)
    assert torch.equal(text, rows[run.text])
    assert torch.equal(image, rows[run.image])


@pytest.mark.skipif(not _HAVE_RECORDED, reason="recorded requests or backbone not cached")
def test_the_image_span_is_fixed_by_the_cameras_and_the_text_span_is_not():
    """Why a batch's bucket key carries the text length and not the image length.

    A camera set is a fixed grid, so the rows the vision tower packs for it never move and
    one camera set gives one image-span length. The text span is the prompt, and that
    varies request to request. A key on the image length would therefore put prompts of
    different text width in one bucket; a key on the text length separates them, and the
    image length is implied by the cameras.

    The lengths themselves are not pinned. A recording of another camera set is a
    different number and no less correct, so what is asserted is the relationship.
    """
    spans: dict[tuple, set[tuple[int, int]]] = {}
    for path in _RECORDED:
        request = _request(path)
        key = tuple(request.image_grid_thw.flatten().tolist())
        image = int(request.is_image.sum())
        spans.setdefault(key, set()).add((image, request.input_ids.numel() - image))

    for key, lengths in spans.items():
        assert len({image for image, _ in lengths}) == 1, f"image span moves within {key}"
    assert len({text for lengths in spans.values() for _, text in lengths}) > 1, (
        "the recording has to cover more than one prompt length: that the text span "
        "varies where the image span does not is what this test is for"
    )
