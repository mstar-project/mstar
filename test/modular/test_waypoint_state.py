"""Waypoint declares its world, and the engine builds the ring of frames and the decoder's state from the declaration.

Waypoint wrote its ring and flex attention by hand, and kept its frame clock and the decoder's nine histories in the
submodule's per-request state. Now it declares a frame-window array and a Value: the engine builds the same ring, counts
each request's frames and holds the histories. The hand-written specs and steps below are the reference, copied from
the model as it stood before the port, so the declaration can't pass by agreeing with itself.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.kv.config import KVSpec, RingKVConfig, RingKVLayerConfig, RingKVStep
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import node_handles, node_resources, state_steps
from mstar.model.base import serving_resources
from mstar.model.waypoint.components.taehv import _walked_history_shapes, decoder_history_shapes
from mstar.model.waypoint.config import WAYPOINT_VARIANT_360P, WAYPOINT_VARIANT_720P, WaypointConfig
from mstar.model.waypoint.submodules import PRIME_WALK, ROLLOUT_WALK, WaypointDitSubmodule
from mstar.model.waypoint.waypoint_model import AE_DTYPE, WaypointModel

VARIANTS = (WAYPOINT_VARIANT_720P, WAYPOINT_VARIANT_360P)


def _model(variant=WAYPOINT_VARIANT_360P) -> WaypointModel:
    return WaypointModel(variant=variant, skip_weight_loading=True)


def _hand_written_specs(config: WaypointConfig) -> list:
    """``WaypointModel.get_node_resources`` before the port, its per-layer geometry at the values ``ring_frames``,
    ``ring_buckets`` and ``pinned_dilation`` gave with ``full_global_ring`` off."""
    def layer(i):
        if not config.is_global_layer(i):
            return RingKVLayerConfig(ring_frames=config.local_window, ring_buckets=config.local_window,
                                     pinned_dilation=1)
        slots = config.global_window // config.global_pinned_dilation
        return RingKVLayerConfig(ring_frames=slots, ring_buckets=slots, pinned_dilation=config.global_pinned_dilation)

    ring = RingKVConfig(
        num_layers=config.n_layers, num_kv_heads=config.n_kv_heads, num_qo_heads=config.n_heads,
        head_dim=config.d_head, tokens_per_frame=config.tokens_per_frame,
        layers=tuple(layer(i) for i in range(config.n_layers)), num_sessions=1,
    )
    return [
        KVSpec(resource_key="kv", nodes={"dit"}, config=ring),
        AttentionSpec(resource_key="attn", nodes={"dit"}, config=AttentionConfig(kv_cache="kv",
                                                                                   backend=AttnBackend.FLEX)),
    ]


class MemBlock(torch.nn.Module):
    """The class name and the conv the history walk reads; it computes nothing."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(2 * channels, channels, 1, bias=False)])


def _taehv(stage_channels=(256, 128, 64)) -> torch.nn.Module:
    """A TAEHV whose decoder walks like the pinned one's: a conv into each stage's channels, three MemBlocks, and a 2x
    upsample between stages."""
    taehv = torch.nn.Module()
    taehv.patch_size, taehv.latent_channels, taehv.t_downscale, taehv.t_upscale, taehv.frames_to_trim = 2, 32, 4, 4, 3
    taehv.encoder = torch.nn.ModuleList([MemBlock(64) for _ in range(9)])
    decoder, channels = [], 32
    for stage, width in enumerate(stage_channels):
        if stage:
            decoder.append(torch.nn.Upsample(scale_factor=2))
        decoder += [torch.nn.Conv2d(channels, width, 1, bias=False), *(MemBlock(width) for _ in range(3))]
        channels = width
    taehv.decoder = torch.nn.ModuleList(decoder)
    return taehv


class _StubRing:
    """What a ring's handle reads at bind: each layer's view, the frame size and the count."""

    def __init__(self, config: WaypointConfig):
        self.layers = [SimpleNamespace(kv=torch.zeros(2, 1, 1, 1, 1)) for _ in range(config.n_layers)]
        self.tokens_per_frame = config.tokens_per_frame

    def position(self, rid, label):
        return 0


def _handles(model: WaypointModel):
    """The dit node's handles over the declaration's own decoder pool, a stub ring and no flex."""
    decoder = next(spec for spec in node_resources(model) if spec.resource_key == "decoder")
    pool = RecurrentStatePool(torch.device("cpu"), decoder.config)
    resources = {"kv": _StubRing(model.config), "local_attn": object(), "decoder": pool}
    return node_handles(model, "dit", resources), pool


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_specs_equal_the_hand_written_ones(variant):
    """Three differences are the declaration's: the ring holds a request to its seed and ``max_frames`` frames, the
    flex resource is under the first read's key, and the decoder's histories are a Value the engine holds, whose
    pool defaults to the ring's one session and the sink slot."""
    model = _model(variant)
    ring, flex, decoder = node_resources(model)
    hand_ring, hand_flex = _hand_written_specs(model.config)
    assert (ring.resource_key, ring.nodes) == (hand_ring.resource_key, hand_ring.nodes)
    assert replace(ring.config, max_frames=None) == hand_ring.config, "the ring should be the hand-written one"
    assert ring.config.max_frames == 1 + model.config.max_frames
    assert (flex.resource_key, flex.nodes, flex.config) == ("local_attn", hand_flex.nodes, hand_flex.config)
    shapes = decoder_history_shapes(model.config.latent_height, model.config.latent_width)
    assert (decoder.resource_key, decoder.config.max_slots, decoder.config.kv_cache) == ("decoder", 2, None)
    assert decoder.config.blocks == {
        f"history_{i}": RecurrentBlockConfig(shape=shape, dtype=AE_DTYPE, shard_dims=())
        for i, shape in enumerate(shapes)
    }


@pytest.mark.parametrize(("resources", "sessions"), [({}, 1), ({"kv": {"num_sessions": 3}}, 3)],
                         ids=["no_yaml", "yaml"])
def test_a_deployment_sizes_the_ring_s_sessions(resources, sessions):
    (ring, _, _) = serving_resources(_model(), {"resources": resources})
    assert ring.config.num_sessions == sessions, "resources.kv.num_sessions should size the ring as it did"


@pytest.mark.parametrize("walk", [PRIME_WALK, ROLLOUT_WALK])
def test_the_steps_are_the_hand_written_ones_with_each_row_s_segment(walk):
    """Waypoint's step named each row's frame under ``kv`` and an empty ``AttentionStep`` under ``attn``. Now each row
    has a segment of one frame, the ring counts its frames itself, and the decoder's Value takes its own step."""
    model = _model()
    handles, _ = _handles(model)
    submodule = WaypointDitSubmodule(torch.nn.Module(), _taehv(), replace(model.config, accelerator_graph=False))
    declared = submodule.declare_step(walk, ["a", "b"], [])
    step = state_steps(declared, handles, graph_walk=walk)
    rows = [Segment(rid, "main", model.config.tokens_per_frame) for rid in ("a", "b")]
    assert dict(step.steps) == {
        "kv": RingKVStep(segments=rows), "local_attn": AttentionStep(segments=rows),
        "decoder": RecurrentStep(segments=rows),
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_history_shapes_are_the_pinned_decoder_s(variant):
    """The declaration computes the shapes, and the load walks the decoder's modules for them, on the pinned build."""
    taehv = pytest.importorskip("taehv")
    config = _model(variant).config
    ae = taehv.TAEHV(None, patch_size=2, latent_channels=32)
    walked = _walked_history_shapes(ae, config.latent_height, config.latent_width)
    assert walked == decoder_history_shapes(config.latent_height, config.latent_width)


def test_the_history_shapes_follow_the_decoder_s_stages():
    """Three MemBlocks at each of the pinned decoder's three stages, each stage at twice the last one's size."""
    assert _walked_history_shapes(_taehv(), 8, 16) == decoder_history_shapes(8, 16)
    assert decoder_history_shapes(8, 16) == ((256, 8, 16),) * 3 + ((128, 16, 32),) * 3 + ((64, 32, 64),) * 3


def test_waypoint_refuses_history_shapes_its_decoder_disagrees_with():
    """A decoder other than the pinned one would decode into a Value laid out for different histories."""
    config = replace(_model().config, accelerator_graph=False)
    with pytest.raises(ValueError, match="The TAEHV decoder's MemBlocks take .* and Waypoint's state holds"):
        WaypointDitSubmodule(torch.nn.Module(), _taehv(stage_channels=(256, 128, 32)), config)


def test_a_second_request_in_a_reused_slot_reads_zeros():
    """The decoder starts every request from zero histories, as the prime seeded them, even in a slot whose last
    request left its own."""
    model = _model()
    handles, pool = _handles(model)
    decoder = handles["decoder"]
    for rid in ("a", "b"):
        pool.ingest_request(rid)
        step = RecurrentStep(segments=(Segment(rid, "main", 1),))
        ctx = StepContext(request_ids=[rid], graph_walk=ROLLOUT_WALK, slot=0, capture=False)
        assert pool.admit(step, ctx).ok
        pool.plan(step, ctx)
        expanded = state_steps(
            SubmoduleStep(segments=[Segment(rid, "main", 1)], state={"decoder": ArrayStep()}), handles,
            graph_walk=ROLLOUT_WALK,
        )
        expanded.set_ctx(ctx)
        decoder.begin_step(expanded)
        read = decoder.read(0).fields
        assert all(not value.any() for value in read.values()), f"request {rid} should start from zero histories"
        decoder.write(0, **{name: torch.ones_like(value) for name, value in read.items()})
        decoder.check_step(expanded)
        pool.commit(step, ctx)
        decoder.end_step(expanded)
        pool.remove_request(rid)
