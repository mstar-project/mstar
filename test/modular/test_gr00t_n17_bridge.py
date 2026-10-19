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

The last section drives the ``vl_bridge`` node, ``Gr00tN17VLBridgeSubmodule``, the way the
engine does: the step it declares, the split of the bridge's rows into the denoise loop's
two spans, and that it adds nothing to the forward above.
"""

import glob
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from safetensors import safe_open
from torch import nn

from mstar.model.gr00t_n17.components.bridge import (
    BasicTransformerBlock,
    Gr00tN17VLBridge,
)
from mstar.model.gr00t_n17.config import BRIDGE_ATTN, Gr00tN17Config, Gr00tN17StackConfig
from mstar.model.gr00t_n17.weight_loader import _remapper
from mstar.model.submodule_base import NodeSubmodule

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


