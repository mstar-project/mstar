"""GR00T N1.7's model class: the four walks and the edges joining them, the five resource
specs, the walk schedule and request routing, ``get_submodule`` and loading, capture
settings, and ``postprocess``. Walks and specs are checked on a config alone; loading on a
synthetic checkpoint; the real widths on ``meta``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from mstar.conductor.request_info import CurrentForwardConductorMetadata, CurrentForwardPassInfo
from mstar.engine.accelerator_graph_runner import autocast_scope
from mstar.engine.resources import (
    AttentionStep,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
    apply_yaml_overrides,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule
from mstar.model.gr00t_n17 import gr00t_n17_model
from mstar.model.gr00t_n17.config import (
    BRIDGE_ATTN,
    DT_ATTN,
    DT_XATTN,
    DT_XATTN_PAIRS,
    LLM_ATTN,
    Gr00tN17Config,
    Gr00tN17StackConfig,
    Gr00tN17TextConfig,
)
from mstar.model.gr00t_n17.gr00t_n17_model import (
    DENOISE_STEP,
    LLM,
    STATE_ENCODER,
    VISION_ENCODER,
    VL_BRIDGE,
    Gr00tN17Model,
)
from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor
from mstar.model.gr00t_n17.submodules import (
    Gr00tN17DenoiseStepSubmodule,
    Gr00tN17LLMSubmodule,
    Gr00tN17Shape,
    Gr00tN17StateEncoderSubmodule,
    Gr00tN17VisionEncoderSubmodule,
    Gr00tN17VLBridgeSubmodule,
)
from mstar.model.gr00t_n17.weight_loader import _remapper
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs

from .helpers import BACKBONE, GR00T, SegmentAttention, needs_checkpoint

# The walk names, spelled out: the YAML configs and the registry key off them.
_PREFILL, _BACKBONE_WALK, _STATE, _ACTION_GEN = "prefill", "backbone", "state", "action_gen"

# What `prefill`'s nodes park for the request, and what `action_gen` reads back.
_PERSISTED = {"vl_text", "vl_image", "state_features"}
# The loop-carried edge, spelled out: importing `DenoiseLoopSubmodule.LATENTS` would make
# the test that it is that name compare the scaffold with itself.
_CARRIED = "latents"

# Each node's contract, spelled out from `submodules.py` (the cite is the
# `prepare_inputs` that reads the name, or the `forward` that returns it).
_NODE_INPUTS = {
    # `Gr00tN17VisionEncoderSubmodule.prepare_inputs`
    VISION_ENCODER: {"pixel_values", "image_grid_thw"},
    # `Gr00tN17LLMSubmodule.prepare_inputs`. NOT `position_ids` or `image_rows`: it
    # derives both there and emits `image_rows`.
    LLM: {"vision_embeds", "deepstack_embeds", "input_ids", "image_grid_thw"},
    VL_BRIDGE: {"backbone_features", "image_rows"},  # `Gr00tN17VLBridgeSubmodule.prepare_inputs`
    STATE_ENCODER: {"state", "embodiment_id"},  # `Gr00tN17StateEncoderSubmodule.prepare_inputs`
    # the carried sample (`DenoiseLoopSubmodule._carried_inputs`), the conditioning
    # (`Gr00tN17DenoiseStepSubmodule.request_inputs`)
    DENOISE_STEP: {_CARRIED, "vl_text", "vl_image", "state_features", "embodiment_id"},
}
_NODE_OUTPUTS = {
    VISION_ENCODER: {"vision_embeds", "deepstack_embeds"},  # `Gr00tN17VisionEncoderSubmodule.forward`
    LLM: {"backbone_features", "image_rows"},  # `Gr00tN17LLMSubmodule.forward`
    VL_BRIDGE: {"vl_text", "vl_image"},  # `Gr00tN17VLBridgeSubmodule.forward`
    STATE_ENCODER: {"state_features"},  # `Gr00tN17StateEncoderSubmodule.forward`
    DENOISE_STEP: {_CARRIED},  # `DenoiseLoopSubmodule._run`
}
# What a request carries into `prefill`, and where each one goes; `image_grid_thw`
# goes to two nodes.
_ROUTES = (
    ("pixel_values", VISION_ENCODER), ("image_grid_thw", VISION_ENCODER),
    ("image_grid_thw", LLM), ("input_ids", LLM),
    ("state", STATE_ENCODER), ("embodiment_id", STATE_ENCODER),
)
# the distinct signal names, in first-seen order
_ROUTED = tuple(dict.fromkeys(name for name, _ in _ROUTES))

# The checkpoint's (qo heads, kv heads, head dim) per resource, checked against the real
# config at the end.
_REAL = {
    VISION_ATTN: (16, 16, 64),
    LLM_ATTN: (16, 8, 128),
    BRIDGE_ATTN: (32, 32, 64),
    DT_ATTN: (32, 32, 48),
    DT_XATTN: (32, 32, 48),
}


# ----------------------------------------------------------------------------
# A config with no two widths alike, and a model built around it
# ----------------------------------------------------------------------------

# Every head count and head dim differs, and none is the checkpoint's (16, 8, 32, 48,
# 64, 128), so a hardcoded geometry cannot match by accident.
_VISION_HEADS, _VISION_WIDTH = 5, 20
_TEXT_QO, _TEXT_KV, _TEXT_HEAD_DIM = 6, 3, 4
_BRIDGE_HEADS, _BRIDGE_HEAD_DIM = 7, 9
_DT_HEADS, _DT_HEAD_DIM = 3, 11

# Per-embodiment video keys with 1, 3 and 2 views: the widest is neither first nor last.
_VIDEO_KEYS = {
    "gr1": ["video.ego_view"],
    "oxe": ["video.image_0", "video.image_1", "video.image_2"],
    "franka": ["video.front", "video.wrist"],
}
_MAX_FRAMES = 3 * 2  # the widest embodiment's views x the stub's two timesteps


class _Processor:
    """Only the fields the frame bounds and ``__init__``'s embodiment check read."""

    def __init__(self, video_keys: dict[str, list[str]], embodiment_ids: dict[str, int] | None = None,
                 video_steps: dict[str, int] | None = None):
        self.video_keys = video_keys
        # two timesteps per view, as every embodiment in the checkpoint has
        self.video_steps = {tag: 2 for tag in video_keys} if video_steps is None else video_steps
        # the tiny config has 4 embodiments; both ends of the range are used
        self.embodiment_ids = {"gr1": 0, "oxe": 1, "franka": 3} if embodiment_ids is None else embodiment_ids


def _config(scale: int = 1) -> Gr00tN17Config:
    """The tiny config at ``scale`` x every head count, so the geometry tests run at two
    sets of numbers and one hardcoded constant cannot satisfy both."""
    text = Gr00tN17TextConfig(
        hidden_size=20, intermediate_size=48,
        num_attention_heads=scale * _TEXT_QO, num_key_value_heads=scale * _TEXT_KV,
        head_dim=_TEXT_HEAD_DIM, num_hidden_layers=2, rms_norm_eps=1e-6,
        rope_theta=10000.0, vocab_size=37, mrope_section=(1, 1),
        mrope_interleaved=True, image_token_id=5, vision_start_token_id=6,
    )
    vision = Qwen3_5VisionConfig(
        depth=2, hidden_size=scale * _VISION_WIDTH, intermediate_size=40,
        num_heads=scale * _VISION_HEADS, in_channels=3, patch_size=2,
        temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64,
        out_hidden_size=20,
    )
    return Gr00tN17Config(
        text=text, vision=vision,
        bridge=Gr00tN17StackConfig(
            num_layers=2, num_attention_heads=scale * _BRIDGE_HEADS,
            attention_head_dim=_BRIDGE_HEAD_DIM,
        ),
        dit=Gr00tN17StackConfig(
            num_layers=2, num_attention_heads=scale * _DT_HEADS,
            attention_head_dim=_DT_HEAD_DIM, output_dim=13,
        ),
        deepstack_visual_indexes=(0,), backbone_embedding_dim=63,
        input_embedding_dim=33, hidden_size=17, max_num_embodiments=4,
        action_horizon=6, max_action_dim=8, max_state_dim=8, max_seq_len=64,
    )


def _model(config: Gr00tN17Config | None = None, video_keys=None) -> Gr00tN17Model:
    """The model without its ``__init__``: the walks, resources and walk schedule are
    functions of the config alone. ``object.__new__`` because the class has no
    ``skip_weight_loading`` flag, and adding one for a test would be a new mechanism."""
    model = object.__new__(Gr00tN17Model)
    model.config = config if config is not None else _config()
    model.processor = _Processor(_VIDEO_KEYS if video_keys is None else video_keys)
    model._submodules = {}
    # `__init__`'s capture defaults: no graphs
    model.accelerator_graph, model.capture_shapes, model.capture_batch_sizes = False, [], [1, 2, 4, 8]
    return model


def _info(dims: list[int], uuid: int = 0) -> TensorPointerInfo:
    """A descriptor and nothing behind it: the forward-pass hooks read only ``dims``."""
    return TensorPointerInfo(
        dims=dims, dtype=torch.bfloat16, nbytes=2, address=0,
        stride=[1] * len(dims), uuid=uuid, source_session_id="test:0",
        source_entity="test",
    )


def _metadata(model: Gr00tN17Model, walk: str | None = None) -> CurrentForwardConductorMetadata:
    """The request's metadata as the model builds it: the transition hook reads the
    walk schedule the entry point parks in ``kwargs``, so a hand-rolled ``kwargs={}``
    would test a state no request can be in."""
    args = model.get_initial_forward_pass_args(
        "default", ["text", "image"], ["action"], {},
        model_kwargs=None if walk is None else {"walk": walk},
    )
    return args.full_metadata


def _nodes(section) -> dict[str, GraphNode]:
    return section.get_nodes()


def _edges(section) -> list[GraphEdge]:
    return [edge for node in _nodes(section).values() for edge in node.outputs]


# ----------------------------------------------------------------------------
# The four walks
# ----------------------------------------------------------------------------


def test_the_walks_are_four_over_five_nodes():
    """The walk and node names are a contract: the YAML partitions and node_groups key
    off them, and a walk nothing names is refused at startup."""
    model = _model()

    assert set(model.get_graph_walk_graphs()) == {_PREFILL, _BACKBONE_WALK, _STATE, _ACTION_GEN}
    assert model.nodes == [LLM, DENOISE_STEP, STATE_ENCODER, VISION_ENCODER, VL_BRIDGE]
    assert (VISION_ENCODER, LLM, STATE_ENCODER, VL_BRIDGE, DENOISE_STEP) == (
        "vision_encoder", "LLM", "state_encoder", "vl_bridge", "denoise_step",
    )


def test_prefill_is_a_parallel_whose_two_branches_never_wait_on_each_other():
    """The state encoder reads only the request's state, so it must not wait on the
    backbone. A ``Sequential`` over the same branches derives the same I/O signature,
    so the type is the only declaration of independence; the independence itself is
    pinned beside it."""
    prefill = _model().get_graph_walk_graphs()[_PREFILL]

    assert isinstance(prefill, Parallel)
    conditioning, state = prefill.sections
    assert isinstance(conditioning, Sequential)
    assert list(_nodes(conditioning)) == [VISION_ENCODER, LLM, VL_BRIDGE]
    assert isinstance(state, GraphNode) and state.name == STATE_ENCODER

    # independence: neither branch consumes anything the other produces
    left, right = set(_nodes(conditioning)), set(_nodes(state))
    assert left.isdisjoint(right)
    for branch, other_nodes in ((conditioning, right), (state, left)):
        produced = {edge.name for edge in _edges(branch)}
        other = conditioning if branch is state else state
        assert not (produced & {name for name, _ in other.get_inputs_outputs().ext_inputs})
        assert not ({edge.next_node for edge in _edges(branch)} & other_nodes)
    assert prefill.get_inputs_outputs().loop_back == set()

    # every input arrives from the request, keyed by (name, destination)
    assert prefill.get_inputs_outputs().ext_inputs == {
        ("pixel_values", VISION_ENCODER), ("image_grid_thw", VISION_ENCODER),
        ("input_ids", LLM), ("image_grid_thw", LLM),
        ("state", STATE_ENCODER), ("embodiment_id", STATE_ENCODER),
    }
    # the ViT's two outputs are consumed inside the branch, so neither shows here
    assert {name for name, _ in prefill.get_inputs_outputs().ext_inputs}.isdisjoint(
        {"vision_embeds", "deepstack_embeds", "backbone_features", "image_rows"}
    )


def test_prefills_persist_edges_park_rather_than_point_at_a_node():
    """Regression: a persist edge once pointed at ``denoise_step``, which is not in
    ``prefill``, and nothing raised: the tensor is parked either way and the destination is
    read only when routed. ``persist`` and a real destination are compatible in general
    (``PythonGraphRuntime._process_node_outputs``); ``EMPTY_DESTINATION`` is the idiom
    flux2_klein uses (``Flux2KleinModel.get_graph_walk_graphs``)."""
    prefill = _model().get_graph_walk_graphs()[_PREFILL]
    persisted = [edge for edge in _edges(prefill) if edge.persist]

    assert {edge.name for edge in persisted} == _PERSISTED
    assert {edge.next_node for edge in persisted} == {EMPTY_DESTINATION}
    assert DENOISE_STEP not in _nodes(prefill)  # there is nothing here to point at

    # the general form: no edge in this walk names a node outside it
    inside = set(_nodes(prefill)) | {EMPTY_DESTINATION, EMIT_TO_CLIENT}
    assert {edge.next_node for edge in _edges(prefill)} <= inside


def test_a_persisted_request_input_keeps_its_destination():
    """``embodiment_id`` must reach ``state_encoder`` and survive ``prefill``, because
    the denoise step reads it every iteration, so its edge has both a destination and
    ``persist``. The flag is the model's own marker: the conductor parks every initial
    signal (``Conductor._do_ingest_request``) and the runtime reads ``persist`` only on
    outputs (``PythonGraphRuntime._process_node_outputs``).

    The release list is per signal, not per edge. Per edge, ``image_grid_thw`` (routed
    twice) would be released twice and ``action_mask`` (routed nowhere,
    ``Gr00tN17Preprocessor.__call__``) never, held for the request's life since
    ``PreprocessWorkerThread._store_persisted`` persists every initial signal. Regression
    for that leak."""
    model = _model()
    # `action_mask` rides along unrouted: it is the one input the full path
    # receives and no node reads.
    signals = {name: [_info([1], uuid=i)] for i, name in enumerate((*_ROUTED, "action_mask"))}
    args = model.get_initial_forward_pass_args("default", ["text", "image"], ["action"], signals)
    by_name = {edge.name: edge for edge in args.inputs}

    assert {name for name, edge in by_name.items() if edge.persist} == {"embodiment_id"}
    assert by_name["embodiment_id"].next_node == STATE_ENCODER
    assert "action_mask" not in by_name  # routed nowhere ...
    # ... and released all the same: one entry per signal the walk did not park
    assert args.unpersist_tensors == [
        info for name, infos in signals.items() if name != "embodiment_id" for info in infos
    ]
    assert signals["embodiment_id"][0] not in args.unpersist_tensors
    # the two halves the per-edge formula got wrong, stated on the tensors
    # themselves: routed twice, released once; routed never, released once
    assert len([edge for edge in args.inputs if edge.name == "image_grid_thw"]) == 2
    assert args.unpersist_tensors.count(signals["image_grid_thw"][0]) == 1
    assert args.unpersist_tensors.count(signals["action_mask"][0]) == 1


def test_the_loops_terminal_output_is_named_for_the_sections_loop_back_edge():
    """``Loop.__post_init__`` keeps only terminal outputs the section produces, matched
    by name (``graph/base.py``), so a renamed terminal output is silently deleted and
    ``action_gen`` emits nothing (pi05 warns of the same, ``Pi05Model.get_graph_walk_graphs``)."""
    loop = _model().get_graph_walk_graphs()[_ACTION_GEN]

    assert isinstance(loop, Loop)
    assert loop.max_iters == _model().config.num_inference_timesteps
    step = loop.section
    assert isinstance(step, GraphNode) and step.name == DENOISE_STEP
    assert {(edge.name, edge.next_node) for edge in step.outputs} == {(_CARRIED, DENOISE_STEP)}
    assert loop._loop_back_inputs == {(_CARRIED, DENOISE_STEP)}

    (terminal,) = loop.outputs
    assert terminal.name == _CARRIED
    assert (terminal.next_node, terminal.output_modality, terminal.persist) == (
        EMIT_TO_CLIENT, "action", True,
    )
    assert {edge.name for edge in loop.outputs} == {name for name, _ in loop._loop_back_inputs}
    # the graph-internal key is the scaffold's `latents`; the client reads an `action`
    assert terminal.name != terminal.output_modality


# ----------------------------------------------------------------------------
# What each node takes and emits
# ----------------------------------------------------------------------------


def test_each_node_declares_the_contract_its_submodule_reads():
    """Each node's declared inputs and outputs against what ``submodules.py`` reads and
    returns; nothing else checks the two, and an output key matching no edge is dropped
    silently (``PythonGraphRuntime.get_output_signals``). Inputs are ``==``; outputs are
    ``<=`` per walk (reuse walks emit less) and ``==`` over the union.

    No walk may gain a loop-back by accident: declaring ``image_rows`` (an LLM output)
    as an LLM input turns the walk's own edge into a loop-back on a node that already
    ran (``Sequential.get_inputs_outputs``, ``graph/base.py``)."""
    walks = _model().get_graph_walk_graphs()
    emitted: dict[str, set[str]] = {name: set() for name in _NODE_OUTPUTS}

    for walk in walks.values():
        for name, node in _nodes(walk).items():
            assert node.input_names == _NODE_INPUTS[name], name
            assert {edge.name for edge in node.outputs} <= _NODE_OUTPUTS[name], name
            emitted[name] |= {edge.name for edge in node.outputs}

    assert emitted == _NODE_OUTPUTS
    # the DT's self-loop is the only output that is also an input of its own node
    assert {
        name for name in _NODE_OUTPUTS if _NODE_OUTPUTS[name] & _NODE_INPUTS[name]
    } == {DENOISE_STEP}
    for name, walk in walks.items():
        assert walk.get_inputs_outputs().loop_back == (
            {(_CARRIED, DENOISE_STEP)} if name == _ACTION_GEN else set()
        ), name


def test_every_internal_edge_lands_on_an_input_its_destination_declares():
    """A one-sided rename inside a walk. ``deepstack_embeds`` is the dangerous one: it
    rides beside ``vision_embeds`` between the same two nodes, so renaming one side
    leaves a walk that still looks connected."""
    for walk_name, walk in _model().get_graph_walk_graphs().items():
        nodes = _nodes(walk)
        landed = set()
        for source in nodes.values():
            for edge in source.outputs:
                if edge.next_node in nodes:
                    assert edge.name in nodes[edge.next_node].input_names, (walk_name, edge.name)
                    landed.add((edge.name, edge.next_node))
        # and the walk's signature agrees: every input is landed on or external
        io = walk.get_inputs_outputs()
        assert landed == {
            (name, dest) for name, dest in
            {(n, node.name) for node in nodes.values() for n in node.input_names}
            if (name, dest) not in io.ext_inputs
        }, walk_name


# ----------------------------------------------------------------------------
# The join across the walk boundary
# ----------------------------------------------------------------------------


def test_action_gens_conditioning_inputs_are_exactly_what_prefill_persists():
    """The cross-walk join: ``prefill`` parks three tensors, the request a fourth, and
    ``action_gen`` asks for them by name. A one-sided rename leaves a ``denoise_step``
    whose inputs never complete, with no node to blame. Both suppliers are derived,
    and spelled as a partition so a new input with no supplier is caught too."""
    model = _model()
    walks = model.get_graph_walk_graphs()
    node_persisted = {edge.name for edge in _edges(walks[_PREFILL]) if edge.persist}
    initial = model.get_initial_forward_pass_args(
        "default", ["text", "image"], ["action"],
        {name: [_info([1], uuid=i)] for i, name in enumerate(_ROUTED)},
    )
    request_persisted = {edge.name for edge in initial.inputs if edge.persist}
    needs = walks[_ACTION_GEN].section.input_names

    assert node_persisted == _PERSISTED
    assert request_persisted == {"embodiment_id"}
    assert node_persisted.isdisjoint(request_persisted)  # one supplier each
    assert node_persisted < needs and request_persisted < needs
    assert needs - node_persisted - request_persisted == {_CARRIED}
    assert needs == node_persisted | request_persisted | {_CARRIED}


def test_the_loops_carried_edge_is_the_scaffolds_own_latents_name():
    """The carried sample is the scaffold's: ``DenoiseLoopSubmodule._carried_inputs``
    reads ``loop_back_names`` and ``DenoiseLoopSubmodule._run`` returns ``LATENTS``,
    while the worker routes by the graph's edge names.
    Regression: with the walk naming it ``actions``, iteration 1 raised "loop-back
    edge(s) ['latents'] missing". ``denoise``'s parameter must match too, since the
    scaffold passes it by keyword (``DenoiseLoopSubmodule.LATENTS``)."""
    loop = _model().get_graph_walk_graphs()[_ACTION_GEN]
    carried = {name for name, _ in loop._loop_back_inputs}

    assert carried == {DenoiseLoopSubmodule.LATENTS}
    # not overridden, so the base's name is the node's name
    assert Gr00tN17DenoiseStepSubmodule.LATENTS == DenoiseLoopSubmodule.LATENTS
    assert set(Gr00tN17DenoiseStepSubmodule.loop_back_names.fget(
        Gr00tN17DenoiseStepSubmodule
    )) == carried
    assert carried <= set(inspect.signature(Gr00tN17DenoiseStepSubmodule.denoise).parameters)
    # and the terminal output rides the same name out of the loop
    assert {edge.name for edge in loop.outputs} == carried


def test_every_input_the_loop_needs_is_delivered_to_it():
    """The conductor parks tensors but does not route them into the next walk; this hook
    must build an edge for every ``denoise_step`` input
    (``Conductor._process_done_forward``, flux2_klein's shape,
    ``Flux2KleinModel.get_partition_forward_pass_args``). Regression: it once emitted only
    the carried edge, and the loop never became ready."""
    model = _model()
    walks = model.get_graph_walk_graphs()
    metadata = _metadata(model)
    persist = {
        "vl_text": [_info([13, 63], uuid=1)],
        "vl_image": [_info([448, 63], uuid=2)],
        "state_features": [_info([1, 33], uuid=3)],
        "embodiment_id": [_info([1], uuid=4)],
    }
    args = model.get_partition_forward_pass_args("default", metadata, persist)

    assert all(edge.next_node == DENOISE_STEP for edge in args.inputs)
    assert {edge.name for edge in args.inputs} == walks[_ACTION_GEN].section.input_names
    assert len(args.inputs) == len(walks[_ACTION_GEN].section.input_names)  # no duplicates
    # the carried edge arrives empty: the scaffold draws iteration 0's sample itself
    for edge in args.inputs:
        assert edge.tensor_info == ([] if edge.name == _CARRIED else persist[edge.name])
    # released here; the Loop re-injects it across its iterations (`_persist_for_loop`)
    assert args.unpersist_tensors == sum(
        (persist[name] for name in ("vl_text", "vl_image", "state_features", "embodiment_id")),
        start=[],
    )


# ----------------------------------------------------------------------------
# The reuse walks
# ----------------------------------------------------------------------------


def test_the_reuse_walks_expose_the_nodes_prefill_already_contains():
    """Each reuse walk is ``prefill``'s own nodes stopping early (same names, inputs and
    internal edge), not a second description that can disagree. ``vl_bridge`` stays out of
    ``backbone``: its weights are GR00T's ``action_head.*``. The emitted modality is
    ``tensor`` because "embedding" is not in ``entrypoint.py``'s
    ``SUPPORTED_MODALITIES``, and the server would reject the response after the work."""
    walks = _model().get_graph_walk_graphs()
    prefill = _nodes(walks[_PREFILL])
    backbone, state = _nodes(walks[_BACKBONE_WALK]), _nodes(walks[_STATE])

    assert set(backbone) == {VISION_ENCODER, LLM}
    assert set(state) == {STATE_ENCODER}
    assert set(backbone) | set(state) < set(prefill)
    assert VL_BRIDGE in prefill and VL_BRIDGE not in backbone

    # the same nodes: every reuse node takes exactly what it takes in `prefill`
    for name, node in {**backbone, **state}.items():
        assert node.input_names == prefill[name].input_names
    # and the one internal edge is the same edge
    assert [(e.name, e.next_node) for e in backbone[VISION_ENCODER].outputs] == [
        (e.name, e.next_node) for e in prefill[VISION_ENCODER].outputs
    ]

    # where `prefill` forwards or parks, a reuse walk emits a `tensor` to the client
    for nodes, tail, name in ((backbone, LLM, "backbone_features"),
                              (state, STATE_ENCODER, "state_features")):
        (emit,) = nodes[tail].outputs
        assert (emit.name, emit.next_node, emit.output_modality) == (
            name, EMIT_TO_CLIENT, "tensor",
        )
    assert isinstance(walks[_STATE], GraphNode)
    assert isinstance(walks[_BACKBONE_WALK], Sequential)


_REUSE_WALKS = {"the backbone walk": _BACKBONE_WALK, "the state walk": _STATE}


@pytest.mark.parametrize("walk", list(_REUSE_WALKS.values()), ids=list(_REUSE_WALKS))
def test_a_reuse_walk_is_selected_by_name_and_schedules_only_itself(walk):
    """The schedule is decided once and parked in ``metadata.kwargs`` (flux2_klein's
    idiom, ``Flux2KleinModel.get_initial_forward_pass_args``). Its first entry picks the
    worker graphs (``Conductor._do_ingest_request``), so a schedule starting at ``prefill``
    serves an action; a second entry sends the request to ``action_gen``, whose inputs
    never complete. ``walk`` must be in ``request_kwargs`` (``Model.request_kwargs``) or
    the client is told it was ignored."""
    model = _model()
    metadata = _metadata(model, walk=walk)

    assert metadata.graph_walk == walk
    assert metadata.kwargs["walk_schedule"] == [walk]
    assert metadata.kwargs["walk_step"] == 0
    assert metadata.is_prefill
    assert "walk" in model.request_kwargs()
    # the other branch of the same decision: no `walk` key is the full path
    assert _metadata(model).kwargs["walk_schedule"] == [_PREFILL, _ACTION_GEN]


@pytest.mark.parametrize("walk", list(_REUSE_WALKS.values()), ids=list(_REUSE_WALKS))
def test_a_reuse_walk_is_done_at_its_first_transition(walk):
    """A hang, not an error: re-emitting the current walk with no inputs reads to the
    conductor as "partition self-triggers via StreamBuffer" (``Conductor._process_done_forward``)
    and it sends nothing, so the request holds its worker graphs until killed. A reuse
    walk parks nothing, so ``persist_signals`` is empty."""
    model = _model()
    metadata = _metadata(model, walk=walk)

    args = model.get_partition_forward_pass_args("default", metadata, {})

    assert args.request_done is True
    assert args.inputs == []
    assert args.unpersist_tensors == []
    assert args.step_metadata == {"is_prefill": False}
    # the walk it finished in, unchanged: there is no next walk to move to
    assert args.full_metadata.graph_walk == walk
    assert args.full_metadata.is_prefill is False


def test_an_unknown_walk_is_refused_and_the_message_names_all_four_walks():
    """A typo'd ``walk`` raises at admission: falling back serves an action to a caller
    who asked for features. ``prefill`` and ``action_gen`` are refused too: a schedule
    of ``[prefill]`` runs, parks its tensors and returns nothing. The message names all
    four, since this is the only place a caller learns them."""
    model = _model()

    with pytest.raises(ValueError) as refused:
        _metadata(model, walk="vlm")
    message = str(refused.value)
    assert "'vlm' is not selectable" in message
    assert _BACKBONE_WALK in message and _STATE in message
    assert _PREFILL in message and _ACTION_GEN in message

    for walk in (_PREFILL, _ACTION_GEN):
        with pytest.raises(ValueError, match="is not selectable"):
            _metadata(model, walk=walk)


_UNREACHABLE = {
    "state, handed frames": (_STATE, "pixel_values", VISION_ENCODER),
    "state, handed a prompt": (_STATE, "input_ids", LLM),
    "backbone, handed a robot state": (_BACKBONE_WALK, "state", STATE_ENCODER),
}


@pytest.mark.parametrize("walk, name, node", list(_UNREACHABLE.values()), ids=list(_UNREACHABLE))
def test_a_reuse_walk_drops_an_input_it_has_no_node_for(walk, name, node):
    """A reuse request always carries the whole observation
    (``Gr00tN17Preprocessor.__call__``, ``PreprocessWorkerThread._process_input``,
    ``Conductor._do_ingest_request``), so refusing would refuse every real reuse request;
    regression for that. The input must not be routed (an edge to a node outside the walk is
    shipped and never consumed) and must still be released
    (``PreprocessWorkerThread._store_persisted``), with no later hop to release it on. The
    full path routing the same input is the positive control."""
    model = _model()
    signals = {signal: [_info([1], uuid=i)] for i, signal in enumerate(_ROUTED)}
    args = model.get_initial_forward_pass_args(
        "default", ["text", "image"], ["action"], signals, model_kwargs={"walk": walk},
    )
    reachable = Gr00tN17Model.REUSE_WALK_NODES[walk]

    # dropped: no edge for it, under any name
    assert name not in {edge.name for edge in args.inputs}
    # and the walk's own routes are all there, every one of them inside the walk
    assert {(edge.name, edge.next_node) for edge in args.inputs} == {
        route for route in _ROUTES if route[1] in reachable
    }
    assert {edge.next_node for edge in args.inputs} == reachable
    # released: everything the request arrived with, the dropped input included
    assert all(not edge.persist for edge in args.inputs)
    assert args.unpersist_tensors == [info for infos in signals.values() for info in infos]
    assert signals[name][0] in args.unpersist_tensors

    # the positive control
    full = model.get_initial_forward_pass_args("default", ["text", "image"], ["action"], signals)
    assert (name, node) in {(edge.name, edge.next_node) for edge in full.inputs}


def test_the_embodiment_id_is_parked_only_where_a_loop_reads_it_back():
    """Parked only on the full path, where the denoise step reads it every iteration;
    on a reuse walk nothing ever unpersists it, so a parked copy leaks for the request's
    life. Both directions, since the flag (``Gr00tN17Model.get_initial_forward_pass_args``)
    can be wrong either way, and the leak is asserted on the release list itself."""
    model = _model()
    signals = {"state": [_info([1, 8], uuid=1)], "embodiment_id": [_info([1], uuid=2)]}

    full = model.get_initial_forward_pass_args("default", ["text", "image"], ["action"], signals)
    reuse = model.get_initial_forward_pass_args(
        "default", ["text", "image"], ["action"], signals, model_kwargs={"walk": _STATE},
    )

    assert [(edge.name, edge.persist) for edge in full.inputs] == [
        ("state", False), ("embodiment_id", True),
    ]
    assert [(edge.name, edge.persist) for edge in reuse.inputs] == [
        ("state", False), ("embodiment_id", False),
    ]
    # so the full path keeps it and the reuse walk hands it straight back
    assert full.unpersist_tensors == signals["state"]
    assert reuse.unpersist_tensors == signals["state"] + signals["embodiment_id"]
    # and there is no later hop to release it on
    done = model.get_partition_forward_pass_args("default", reuse.full_metadata, {})
    assert done.request_done and done.unpersist_tensors == []


def test_the_reuse_walk_node_sets_match_the_graphs():
    """``Gr00tN17Model.REUSE_WALK_NODES`` is a hand copy of graph structure and goes stale
    both ways: a stale node lets an input through to nothing, a missing one refuses a valid
    request. A selectable walk missing from it gets ``reachable=None``
    (``Gr00tN17Model.get_initial_forward_pass_args``), "route everything", so selectable
    and listed must be the same walks."""
    model = _model()
    walks = model.get_graph_walk_graphs()

    assert Gr00tN17Model.REUSE_WALK_NODES == {
        name: frozenset(_nodes(walks[name])) for name in Gr00tN17Model.REUSE_WALK_NODES
    }
    assert set(Gr00tN17Model.REUSE_WALK_NODES) == set(walks) - {_PREFILL, _ACTION_GEN}

    for name in walks:
        if name in Gr00tN17Model.REUSE_WALK_NODES:
            assert _metadata(model, walk=name).kwargs["walk_schedule"] == [name]
        else:
            with pytest.raises(ValueError, match="is not selectable"):
                _metadata(model, walk=name)


# ----------------------------------------------------------------------------
# The five resource specs
# ----------------------------------------------------------------------------


def _specs(model: Gr00tN17Model) -> dict[str, RaggedAttentionConfig]:
    return {spec.resource_key: spec for spec in model.get_node_resources()}


def test_the_five_specs_land_on_the_right_nodes():
    """A resource the engine does not build leaves its node on SDPA in silence, and a
    spec naming the wrong node binds a wrapper the node never calls."""
    specs = _specs(_model())

    assert set(specs) == {VISION_ATTN, LLM_ATTN, BRIDGE_ATTN, DT_ATTN, DT_XATTN}
    assert {key: spec.nodes for key, spec in specs.items()} == {
        VISION_ATTN: {VISION_ENCODER},
        LLM_ATTN: {LLM},
        BRIDGE_ATTN: {VL_BRIDGE},
        DT_ATTN: {DENOISE_STEP},
        DT_XATTN: {DENOISE_STEP},
    }
    # no KV cache anywhere: every attention is one-shot
    assert all(spec.depends_on() == set() for spec in specs.values())


def test_the_dt_holds_two_specs_because_a_resource_is_one_kind_of_attention():
    """The DT's cross and self blocks share a 32 x 48 geometry but not a resource: a
    resource is one kind of attention, so collapsing them leaves one kind unplanned.
    Exact types, because the two spec classes share a base."""
    specs = _specs(_model())
    dt = [spec for spec in specs.values() if spec.nodes == {DENOISE_STEP}]

    assert len(dt) == 2
    assert type(specs[DT_ATTN]) is RaggedAttentionSpec
    assert type(specs[DT_XATTN]) is RaggedCrossAttentionSpec
    assert {key for key, spec in specs.items() if type(spec) is RaggedCrossAttentionSpec} == {DT_XATTN}
    # the same geometry: the kind is the only difference
    for field in ("num_qo_heads", "num_kv_heads", "head_dim"):
        assert getattr(specs[DT_ATTN].config, field) == getattr(specs[DT_XATTN].config, field)


@pytest.mark.parametrize("scale", [1, 2], ids=["the tiny config", "the same config doubled"])
def test_every_head_geometry_is_read_off_the_config(scale):
    """Run at two sets of head counts so a hardcoded geometry cannot pass both. Head dims
    come three ways: the ViT's ``hidden_size // num_heads``, the LLM's ``head_dim`` (not
    ``hidden_size // heads`` in a Qwen3), the action-head stacks' ``attention_head_dim``."""
    config = _config(scale)
    specs = _specs(_model(config))

    def geometry(key):
        cfg = specs[key].config
        return (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim)

    assert geometry(VISION_ATTN) == (
        config.vision.num_heads, config.vision.num_heads,
        config.vision.hidden_size // config.vision.num_heads,
    )
    assert geometry(LLM_ATTN) == (
        config.text.num_attention_heads, config.text.num_key_value_heads, config.text.head_dim,
    )
    assert geometry(BRIDGE_ATTN) == (
        config.bridge.num_attention_heads, config.bridge.num_attention_heads,
        config.bridge.attention_head_dim,
    )
    for key in (DT_ATTN, DT_XATTN):
        assert geometry(key) == (
            config.dit.num_attention_heads, config.dit.num_attention_heads,
            config.dit.attention_head_dim,
        )
    # the LLM's head dim is not hidden_size // heads, so a wrong-field read is visible
    assert config.text.head_dim * config.text.num_attention_heads != config.text.hidden_size
    # and the five are genuinely five different geometries at this config
    assert len({geometry(key) for key in specs}) == 4  # the DT's two are equal by design


def test_the_vision_bound_is_the_only_one_that_is_not_one_per_request():
    """``max_segments_per_request`` is per span label (``_RaggedManagerBase._cg_wrapper``),
    and the cross resource plans one wrapper per pair
    (``_RaggedCrossManagerBase._plan_layouts``). Only the ViT, one segment per frame,
    exceeds 1. The DT's cross bound looks like it should be ``len(DT_XATTN_PAIRS)``, but
    each pair sees one segment per request per label."""
    specs = _specs(_model())

    assert {key: spec.config.max_segments_per_request for key, spec in specs.items()} == {
        VISION_ATTN: _MAX_FRAMES, LLM_ATTN: 1, BRIDGE_ATTN: 1, DT_ATTN: 1, DT_XATTN: 1,
    }
    # the count that makes 1 look wrong
    assert len(DT_XATTN_PAIRS) == 2
    # the token ceiling is a deployment's choice (the shipped YAML sets it), so none here
    assert all(spec.config.max_tokens_per_request is None for spec in specs.values())


def test_every_spec_pins_its_activation_dtype():
    """A cacheless ragged resource with ``dtype=None`` takes the engine's KV dtype
    (``RaggedAttentionConfig.dtype``), so the parity bar's bf16 would track something
    unrelated to this model (z_image pins too, ``ZImageModel.get_node_resources``)."""
    specs = _specs(_model())

    assert {spec.config.dtype for spec in specs.values()} == {torch.bfloat16}
    assert len(specs) == 5


def test_causality_is_declared_on_the_step_and_not_on_the_resource():
    """GR00T's LLM is the first causal ragged self-attention user; causality is declared
    on the step, so the same resource kind serves the DT's non-causal attention."""
    node = object.__new__(Gr00tN17LLMSubmodule)
    step = node.declare_step(_PREFILL, ["r0"], [NodeInputs(input_seq_len=7)])
    assert set(step.steps) == {LLM_ATTN}
    assert isinstance(step.steps[LLM_ATTN], AttentionStep)
    assert step.steps[LLM_ATTN].causal is True
    assert [segment.span for segment in step.segments] == [7]


# ----------------------------------------------------------------------------
# The ViT's segment bound, a property of the checkpoint
# ----------------------------------------------------------------------------


# per-tag views and timesteps -> a request's most frames
_FRAME_CASES = {
    "three embodiments, the widest in the middle, two steps": (_VIDEO_KEYS, None, 3 * 2),
    "one view, one step": ({"gr1": ["video.ego_view"]}, {"gr1": 1}, 1),
    # the most any one tag carries, not the widest tag's views times the longest tag's steps
    "the widest tag is not the longest": ({"rig": list("abcde"), "gr1": ["x"]}, {"rig": 1, "gr1": 4}, 5 * 1),
}


@pytest.mark.parametrize("video_keys, steps, frames", list(_FRAME_CASES.values()), ids=list(_FRAME_CASES))
def test_the_vit_segment_bound_is_a_requests_most_frames(video_keys, steps, frames):
    """``vision_encoder`` declares one segment per frame, views x timesteps; a bound of views
    alone (3 for DROID's 4 frames) would size a captured wrapper's plan too small."""
    model = _model()
    model.processor = _Processor(video_keys, video_steps=steps)

    assert _specs(model)[VISION_ATTN].config.max_segments_per_request == frames
    assert model.max_input_files() == {"image": frames}


# ----------------------------------------------------------------------------
# The request's two hops
# ----------------------------------------------------------------------------


def test_a_request_starts_in_prefill_with_each_input_at_the_node_that_takes_it():
    """Every routed name is one its destination declares, ``is_prefill`` is set, only the
    signals the request carries are routed, and the routes cover every external input
    ``prefill`` declares (else a node never becomes ready)."""
    model = _model()
    prefill_walk = model.get_graph_walk_graphs()[_PREFILL]
    prefill = _nodes(prefill_walk)
    signals = {
        "pixel_values": [_info([2, 3, 224, 224], uuid=1)],
        "image_grid_thw": [_info([2, 3], uuid=2)],
        "input_ids": [_info([37], uuid=3)],
        "state": [_info([1, 8], uuid=4)],
        "embodiment_id": [_info([1], uuid=5)],
    }
    args = model.get_initial_forward_pass_args("default", ["text", "image"], ["action"], signals)

    assert args.full_metadata.graph_walk == _PREFILL and args.full_metadata.is_prefill
    assert args.step_metadata == {"is_prefill": True}
    assert not args.request_done
    assert {(edge.name, edge.next_node) for edge in args.inputs} == set(_ROUTES)
    # `image_grid_thw` is routed twice, so the pairs are not one per signal
    assert len(args.inputs) == len(_ROUTES) == 6
    assert set(signals) == set(_ROUTED) and len(_ROUTED) == 5

    for edge in args.inputs:
        assert edge.name in prefill[edge.next_node].input_names
        assert edge.tensor_info == signals[edge.name]
    # coverage: nothing the walk needs from outside is unrouted
    assert set(_ROUTES) == prefill_walk.get_inputs_outputs().ext_inputs

    # a request carrying only some of them routes only those
    thin = model.get_initial_forward_pass_args(
        "default", ["text"], ["action"], {"input_ids": signals["input_ids"]},
    )
    assert [(edge.name, edge.next_node) for edge in thin.inputs] == [("input_ids", LLM)]


_SPANS = {
    "a short prompt": (13, 448),
    "a long one": (21, 896),
    "no image rows at all": (17, 0),
}


@pytest.mark.parametrize("text_rows, image_rows", list(_SPANS.values()), ids=list(_SPANS))
def test_the_bucket_keys_span_counts_come_off_the_persisted_tensors_own_dims(text_rows, image_rows):
    """``denoise_step`` needs the span lengths before their tensors arrive, so they ride on
    ``step_metadata`` under the names the node reads
    (``Gr00tN17DenoiseStepSubmodule.bucket_key_for``). Derived from the persisted tensors'
    own ``dims``, because the node's check (``Gr00tN17DenoiseStepSubmodule.request_inputs``)
    only catches a disagreement inside a batch."""
    model = _model()
    metadata = _metadata(model)
    persist = {
        "vl_text": [_info([text_rows, 63], uuid=1)],
        "vl_image": [_info([image_rows, 63], uuid=2)],
        "state_features": [_info([1, 33], uuid=3)],
        "embodiment_id": [_info([1], uuid=4)],
    }
    args = model.get_partition_forward_pass_args("default", metadata, persist)

    assert args.step_metadata == {
        "is_prefill": False, "vl_text_rows": text_rows, "vl_image_rows": image_rows,
    }
    # the names the node reads back, which are not the persisted tensors' names
    assert set(args.step_metadata) - {"is_prefill"} == {"vl_text_rows", "vl_image_rows"}


def test_the_walk_flips_prefill_then_action_gen_then_done():
    """The four denoising steps are the Loop's iterations, not conductor hops, so the
    second transition ends the request. ``is_prefill`` must go false leaving
    ``prefill``, or every loop iteration reads as a conditioning pass."""
    model = _model()
    names = sorted(_PERSISTED | {"embodiment_id"})
    persist = {name: [_info([3, 63], uuid=i)] for i, name in enumerate(names)}
    metadata = _metadata(model)

    first = model.get_partition_forward_pass_args("default", metadata, persist)
    assert first.full_metadata.graph_walk == _ACTION_GEN
    assert first.full_metadata.is_prefill is False
    assert not first.request_done
    # the carried edge is routed last and arrives empty
    assert [edge.next_node for edge in first.inputs] == [DENOISE_STEP] * 5
    assert first.inputs[-1].name == _CARRIED and first.inputs[-1].tensor_info == []

    done = model.get_partition_forward_pass_args("default", first.full_metadata, persist)
    assert done.request_done
    assert done.inputs == [] and done.unpersist_tensors == []
    assert done.step_metadata == {"is_prefill": False}
    assert done.full_metadata.graph_walk == _ACTION_GEN


# ----------------------------------------------------------------------------
# `get_submodule`
# ----------------------------------------------------------------------------

_SUBMODULES = {
    VISION_ENCODER: Gr00tN17VisionEncoderSubmodule,
    LLM: Gr00tN17LLMSubmodule,
    STATE_ENCODER: Gr00tN17StateEncoderSubmodule,
    VL_BRIDGE: Gr00tN17VLBridgeSubmodule,
    DENOISE_STEP: Gr00tN17DenoiseStepSubmodule,
}


# Each node's checkpoint root; the wrapping nodes' tensors are named relative to `.model`.
_ROOTS = {
    VISION_ENCODER: "backbone.model.model.visual.",
    LLM: "backbone.model.model.language_model.",
    VL_BRIDGE: "action_head.", STATE_ENCODER: "action_head.", DENOISE_STEP: "action_head.",
}


def _checkpoint_parts(config: Gr00tN17Config, node: str, name: str, shape) -> list[tuple[str, tuple]]:
    """The checkpoint tensors one served parameter is made of, in row order: one tensor,
    except the LLM's fused q/k/v and gate/up, which the checkpoint keeps apart."""
    if node in (VISION_ENCODER, LLM, VL_BRIDGE):
        name = name.removeprefix("model.")
    if node == VISION_ENCODER:
        name = name.removeprefix("tower.")
    t = config.text
    q, kv = t.num_attention_heads * t.head_dim, t.num_key_value_heads * t.head_dim
    fused = {".qkv_proj": ((".q_proj", q), (".k_proj", kv), (".v_proj", kv)),
             ".gate_up_proj": ((".gate_proj", t.intermediate_size), (".up_proj", t.intermediate_size))}
    for target, parts in fused.items():
        if node == LLM and target in name:
            return [(_ROOTS[node] + name.replace(target, part), (rows, *shape[1:])) for part, rows in parts]
    return [(_ROOTS[node] + name, tuple(shape))]


def _unloaded(model: Gr00tN17Model, node: str):
    """The node as ``_create_submodule`` builds it, before any tensor lands in it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gr00t_n17_model, "load_gr00t_n17_weights", lambda *args: None)
        return model._create_submodule(node, "cpu")


def _served(path: Path, drop: str | None = None) -> tuple[Gr00tN17Model, dict[str, torch.Tensor]]:
    """``_model()`` over a synthetic checkpoint at ``path``: every tensor the five nodes
    read, under the checkpoint's names, each a different draw, so a parameter left at
    init or fed its neighbour's tensor cannot match."""
    model = _model()
    model._repo_dir = path
    generator = torch.Generator().manual_seed(0)
    weights = {}
    for node in _SUBMODULES:
        for name, param in _unloaded(model, node).named_parameters():
            for key, shape in _checkpoint_parts(model.config, node, name, param.shape):
                weights[key] = torch.randn(shape, generator=generator)
    if drop is not None:
        del weights[drop]
    save_file(weights, path / "model.safetensors")
    return model, weights


def test_get_submodule_builds_each_node_once_and_hands_back_the_same_object(tmp_path):
    """Cached by identity: the model loads weights into the object it hands out, the
    engine binds resources onto it, and compiled regions guard on it."""
    model, _ = _served(tmp_path)
    built = {name: model.get_submodule(name) for name in _SUBMODULES}

    for name, cls in _SUBMODULES.items():
        assert type(built[name]) is cls
        assert model.get_submodule(name) is built[name]

    assert set(model._submodules) == set(_SUBMODULES) == set(model.nodes)
    assert all(model._submodules[name] is built[name] for name in _SUBMODULES)
    assert len({id(sub) for sub in built.values()}) == 5


def test_every_submodule_comes_back_in_eval_mode(tmp_path):
    """Every parity number is an eval-mode claim; a dropout or norm added later would make
    the served model a different function. ``nn.Module`` starts in training mode, so
    somebody has to set the flag."""
    model, _ = _served(tmp_path)
    for name in _SUBMODULES:
        submodule = model.get_submodule(name)
        assert submodule.training is False
        assert all(not child.training for child in submodule.modules())


def test_tensor_parallelism_is_refused_before_anything_is_built(tmp_path):
    """No node shards its weights, so a TP group would run whole copies; a group of one
    is no parallelism and builds as usual."""
    model, _ = _served(tmp_path)
    group = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="tensor parallelism"):
        model.get_submodule(VISION_ENCODER, tp_group=group)
    assert model._submodules == {}
    group.world_size = 1
    assert model.get_submodule(VISION_ENCODER, tp_group=group) is not None


def test_an_unknown_node_is_none_and_is_not_cached(tmp_path):
    """``None`` is the engine's "this worker does not host that node", so a typo comes back
    as a miss, and the miss is not cached."""
    model, _ = _served(tmp_path)
    assert model.get_submodule("vlm") is None
    assert model.get_submodule(VISION_ENCODER.upper()) is None
    assert model.get_submodule("") is None
    assert model._submodules == {}


def test_every_served_parameter_is_the_checkpoint_tensor_it_maps_from(tmp_path):
    """Regression: the served path once never called the loader, so a live server ran on
    random init and still returned plausible actions. Values, not shapes; fused LLM
    parameters are the checkpoint's parts stacked in row order."""
    model, weights = _served(tmp_path)
    used = set()
    for node in _SUBMODULES:
        for name, param in model.get_submodule(node).named_parameters():
            parts = _checkpoint_parts(model.config, node, name, param.shape)
            assert torch.equal(param, torch.cat([weights[key] for key, _ in parts])), (node, name)
            used |= {key for key, _ in parts}
    assert used == set(weights)


@pytest.mark.parametrize("node, drop", [
    (VISION_ENCODER, "backbone.model.model.visual.deepstack_merger_list.0.norm.bias"),
    (LLM, "backbone.model.model.language_model.layers.1.mlp.down_proj.weight"),
    (STATE_ENCODER, "action_head.state_encoder.layer2.b"),
    (VL_BRIDGE, "action_head.vlln.weight"),
    (DENOISE_STEP, "action_head.position_embedding.weight"),
])
def test_a_tensor_missing_from_the_checkpoint_fails_its_node_by_name(tmp_path, node, drop):
    """Strict per node: a truncated checkpoint fails at startup, naming the node, rather
    than serving a random-init parameter."""
    model, _ = _served(tmp_path, drop=drop)
    with pytest.raises(ValueError, match=f"^{node}: 1 parameter"):
        model.get_submodule(node)
    assert node not in model._submodules


@pytest.mark.parametrize("part", ["layers.1.self_attn.k_proj.weight", "layers.1.mlp.up_proj.weight"])
def test_one_part_missing_from_a_fused_llm_parameter_fails_the_llm_by_name(tmp_path, part):
    """A fused parameter once counted as loaded when any part landed, serving the missing
    rows at random init. The part is named, since the fused name is not the checkpoint's."""
    model, _ = _served(tmp_path, drop=_ROOTS[LLM] + part)
    with pytest.raises(ValueError, match=rf"^LLM: 1 parameter.*{re.escape(part)}"):
        model.get_submodule(LLM)
    assert LLM not in model._submodules


def test_the_served_vision_node_holds_the_vision_config_and_prepares_a_grid(tmp_path):
    """Regression: handed the whole ``Gr00tN17Config``, the vision node built and loaded,
    then raised ``AttributeError`` in ``prepare_inputs`` on the first frame (a 500 on
    every live request)."""
    model, _ = _served(tmp_path)
    node = model.get_submodule(VISION_ENCODER)
    assert node.config is model.config.vision

    vision = model.config.vision
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    fwd = CurrentForwardPassInfo(request_id="r0", graph_walk=_PREFILL, fwd_index=0, random_seed=0, max_tokens=0)
    grid = {"image_grid_thw": [torch.tensor([[1, 4, 6]])], "pixel_values": [torch.zeros(24, patch)]}
    inputs = node.prepare_inputs(_PREFILL, fwd, grid)
    assert inputs.input_seq_len == 24


def test_only_the_bridge_opts_out_of_autocast_and_it_is_built_in_the_engines_dtype(tmp_path):
    """The engine keeps an opted-out node in its built dtype (``EngineManager.build``),
    so the bridge must be built in bf16 here; in fp32 it runs fp32 with autocast off,
    not NVIDIA's bf16 bridge. The other four keep the engine's autocast."""
    model, _ = _served(tmp_path)
    for name in _SUBMODULES:
        node = model.get_submodule(name, autocast_dtype=torch.bfloat16)
        assert node.disable_autocast is (name == VL_BRIDGE), name
        # compiled, the bridge's answer moved once a prompt of another length was served
        assert node.disable_torch_compile is (name in (VL_BRIDGE, DENOISE_STEP)), name
        expected = torch.bfloat16 if name == VL_BRIDGE else torch.float32
        assert {p.dtype for p in node.parameters()} == {expected}, name


def test_the_opted_out_bridge_hands_the_loop_bf16_spans(tmp_path):
    """Run as the engine runs an opted-out node, both spans come out bf16 as NVIDIA's do.
    Control: under CUDA autocast ``vlln`` runs fp32 and its output is the residual, so
    both spans came out fp32. CPU autocast leaves LayerNorm alone (torch 2.12), so the
    control applies CUDA's fp32 LayerNorm policy by hand."""
    model, _ = _served(tmp_path)
    node = model.get_submodule(VL_BRIDGE, autocast_dtype=torch.bfloat16).to("cpu")
    rows, image = 10, [2, 3, 4, 5]
    inputs = {
        "backbone_features": [torch.randn(rows, model.config.backbone_embedding_dim, dtype=torch.bfloat16)],
        "image_rows": [torch.tensor(image)],
    }
    fwd = CurrentForwardPassInfo(request_id="r0", graph_walk=_PREFILL, fwd_index=0, random_seed=0, max_tokens=0)
    engine = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})
    prepared = node.prepare_inputs(_PREFILL, fwd, inputs)
    step = node.declare_step(_PREFILL, ["r0"], [prepared])
    assert [seg.span for seg in step.segments] == [rows]
    node.bind_node_resources({BRIDGE_ATTN: SegmentAttention()})
    tensors = node.preprocess(_PREFILL, engine, [prepared])

    def spans(autocast_dtype):
        with torch.no_grad(), autocast_scope(autocast_dtype, "cpu"):
            out = node.forward(_PREFILL, engine, **tensors)
        return out["vl_text"][0], out["vl_image"][0]

    text, img = spans(None)
    assert (text.shape[0], img.shape[0]) == (rows - len(image), len(image))
    assert (text.dtype, img.dtype) == (torch.bfloat16, torch.bfloat16)

    layer_norm = torch.nn.functional.layer_norm

    def fp32_layer_norm(x, shape, weight=None, bias=None, eps=1e-5):
        # what CUDA autocast does to `layer_norm`: every operand up to fp32
        return layer_norm(x.float(), shape, weight.float(), bias.float(), eps)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.functional, "layer_norm", fp32_layer_norm)
        assert {t.dtype for t in spans(torch.bfloat16)} == {torch.float32}


# ----------------------------------------------------------------------------
# accelerator graphs
# ----------------------------------------------------------------------------

_SERVING = Path(__file__).resolve().parents[3] / "configs" / "gr00t_n17.yaml"


def _constructed(path: Path, embodiment_ids: dict[str, int] | None = None, **model_kwargs) -> Gr00tN17Model:
    """The real ``__init__`` over ``path`` with the two checkpoint reads stubbed: the
    capture settings and the embodiment check run there and nowhere else."""
    processor = _Processor(_VIDEO_KEYS, embodiment_ids)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Gr00tN17Config, "from_hf", lambda *args: _config())
        patch.setattr(Gr00tN17Preprocessor, "from_pretrained", lambda *args, **kwargs: processor)
        return Gr00tN17Model(model_path_hf=str(path), backbone_path_hf=str(path), **model_kwargs)


@pytest.mark.parametrize("bad", [-1, 4], ids=["negative", "one past the last"])
def test_an_embodiment_table_outside_the_weights_is_refused_at_construction(tmp_path, bad):
    """``W[-1]`` is the last robot's slice, a plausible wrong answer; torch's IndexError
    covers only the top end, and only once a request reaches the GPU."""
    with pytest.raises(ValueError, match=r"'oxe': %d\}.* 4 embodiments" % bad):
        _constructed(tmp_path, embodiment_ids={"gr1": 0, "oxe": bad})


def test_each_capture_shape_is_one_dt_graph_config_with_its_rows_and_batch_sizes(tmp_path):
    """One DT graph per ``[text_rows, image_rows]``, keyed by that shape, at the configured
    batch sizes, with a dummy request of those rows; a wrong row count captures a graph
    no request replays."""
    _served(tmp_path)
    model = _constructed(tmp_path, accelerator_graph=True, capture_shapes=[[3, 8], [5, 8]], capture_batch_sizes=[1, 2])

    configs = model.get_submodule(DENOISE_STEP).get_accelerator_graph_configs(torch.device("cpu"))

    assert [c.additional_key_info for c in configs] == [Gr00tN17Shape(3, 8), Gr00tN17Shape(5, 8)]
    for config, (text, image) in zip(configs, [(3, 8), (5, 8)], strict=True):
        assert config.capture_graph_walk == _ACTION_GEN
        assert config.capture_batch_sizes == [1, 2]
        tensors = config.single_request_inputs.tensor_inputs
        assert tensors["vl_text"].shape[0] == text
        assert tensors["vl_image"].shape[0] == image


@pytest.mark.parametrize("model_kwargs", [{}, {"capture_shapes": [[3, 8]]}], ids=["default", "shapes_but_off"])
def test_with_capture_off_the_dt_declares_no_graphs(tmp_path, model_kwargs):
    """Off by default, and the flag alone turns it on: shapes left in a config with
    ``accelerator_graph`` off capture nothing."""
    _served(tmp_path)
    model = _constructed(tmp_path, **model_kwargs)

    assert model.capture_buckets() == []
    assert model.get_submodule(DENOISE_STEP).get_accelerator_graph_configs(torch.device("cpu")) == []


def test_a_misspelt_model_kwarg_stops_the_server_naming_it(tmp_path):
    """Every kwarg this model reads is a named parameter; a typo would otherwise serve the
    default silently (`capture_batch_size` for `capture_batch_sizes`)."""
    with pytest.raises(TypeError, match="capture_batch_size"):
        _constructed(tmp_path, capture_batch_size=[1])


def test_capture_without_shapes_is_refused(tmp_path):
    """No shapes would silently mean no graphs: the spans are inputs, so there is no
    default bucket."""
    with pytest.raises(ValueError, match="capture_shapes"):
        _constructed(tmp_path, accelerator_graph=True)


def _xattn_ceiling_and_widest_span(serving: dict, path: Path) -> tuple[int, int]:
    """The DT cross-attention's per-request token ceiling once ``serving``'s ``resources:``
    apply, and the widest text or image span its capture shapes give."""
    model = _constructed(path, **serving["model_kwargs"])
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, serving)
    ceiling = next(spec for spec in specs if spec.resource_key == DT_XATTN).config.max_tokens_per_request
    return ceiling, max(max(shape) for shape in model.capture_shapes)


def test_the_shipped_yaml_sizes_the_dt_cross_attention_for_its_widest_capture_span(tmp_path):
    """A captured cross wrapper is sized by ``max_tokens_per_request`` but its keys are a
    request's text or image rows; below the widest span capture fails ("3584 key tokens
    exceeds the 328"). The shapes must also parse into buckets."""
    serving = yaml.safe_load(_SERVING.read_text(encoding="utf-8"))
    kwargs = serving["model_kwargs"]
    assert kwargs["accelerator_graph"] is True

    ceiling, widest = _xattn_ceiling_and_widest_span(serving, tmp_path)
    assert ceiling is not None and ceiling >= widest

    model = _constructed(tmp_path, **kwargs)
    assert model.capture_buckets() == [(_ACTION_GEN, Gr00tN17Shape(*shape)) for shape in kwargs["capture_shapes"]]
    assert model.capture_batch_sizes == kwargs["capture_batch_sizes"]


def test_a_capture_span_wider_than_the_cross_attention_ceiling_is_caught(tmp_path):
    """The negative control for the test above: one image row past the override."""
    serving = yaml.safe_load(_SERVING.read_text(encoding="utf-8"))
    override = serving["resources"][DT_XATTN]["max_tokens_per_request"]
    serving["model_kwargs"] = {**serving["model_kwargs"], "capture_shapes": [[13, override + 1]]}

    ceiling, widest = _xattn_ceiling_and_widest_span(serving, tmp_path)
    assert ceiling < widest


# ----------------------------------------------------------------------------
# postprocess
# ----------------------------------------------------------------------------

# One embodiment, two groups: `arm` is relative with [-1, 1] bounds (chunk + state),
# `grip` absolute with q01 == q99 == 5 (always 5), so the group order is visible.
_DECODE_KWARGS = {
    "shortest_image_edge": 256, "crop_fraction": 0.95,
    "modality_configs": {
        "bot": {"video": {"modality_keys": ["cam"], "delta_indices": [0, 1]},
                "state": {"modality_keys": ["arm", "grip"]},
                "action": {"delta_indices": [0, 1], "modality_keys": ["arm", "grip"],
                           "action_configs": [
                               {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "arm"},
                               {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "grip"}]}},
    },
}
_DECODE_STATS = {
    "bot": {
        "state": {"arm": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}, "grip": {"q01": [0.0], "q99": [1.0]}},
        "action": {"arm": {"q01": [0.0, 0.0], "q99": [2.0, 2.0]}, "grip": {"q01": [5.0], "q99": [5.0]}},
        "relative_action": {"arm": {"min": [-1.0, -1.0], "max": [1.0, 1.0]}},
    },
}
# the chunk the decoder would emit: (1, T=2, 3), padded to the config's width
_CHUNK = torch.tensor([[[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]])
_RAW_STATE = {"arm": np.array([[[10.0, 20.0]]], np.float32)}
# arm = state + chunk, grip = 5, in `modality_keys` order
_DECODED = [[11.0, 19.0, 5.0], [9.0, 21.0, 5.0]]


def _decoding_model() -> Gr00tN17Model:
    """``_model()`` with a processor that can decode, on synthetic statistics, so
    ``postprocess`` meets the real ``unapply`` rather than a stub that agrees with any order."""
    model = _model()
    model.processor = Gr00tN17Preprocessor(_DECODE_KWARGS, _DECODE_STATS, {"bot": 0}, 8, 6)
    return model


class _BytesProcessor:
    """A backbone processor whose ``input_ids`` are the instruction's bytes, so two
    requests tokenize alike only if handed the same text."""

    tokenizer = type("Tokenizer", (), {"padding_side": "right"})()

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        return "<user>" + conversation[0]["content"][-1]["text"]

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.tensor([list(t.encode()) for t in text]),
                "pixel_values": torch.stack(images).float()}


def test_no_instruction_is_served_as_the_empty_instruction():
    """FastAPI turns an empty ``text`` field into ``None``, and 8 of the 20 recorded DROID
    requests carry one; refusing it answered real robot data with a 400. A real
    instruction is the control."""
    model = _model()
    model.processor = Gr00tN17Preprocessor(_DECODE_KWARGS, _DECODE_STATS, {"bot": 0}, 8, 6, _BytesProcessor())
    frames = [torch.full((180, 320, 3), 40 * i, dtype=torch.uint8) for i in range(2)]
    state = {"arm": np.zeros((1, 1, 2), np.float32), "grip": np.zeros((1, 1, 1), np.float32)}

    def same_as_empty(prompt):
        out = model.process_prompt(prompt, ["text", "image"], ["action"], tensors={"image_inputs": frames},
                                   embodiment_tag="bot", state=state)
        empty = model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": frames},
                                     embodiment_tag="bot", state=state)
        assert out.keys() == empty.keys()
        return all(torch.equal(a, b) for k in out for a, b in zip(out[k], empty[k], strict=True))

    assert same_as_empty(None)
    assert not same_as_empty("Pick up the cup")


_STATE_LEADS = {
    "one timestep": (1, 1),
    "a history of 2": (1, 2),
    "no timestep": (1, 0),
    "a batch of 2": (2, 1),
    "a bare vector": (),
    "an extra axis": (1, 1, 2),
}


@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("lead", list(_STATE_LEADS.values()), ids=list(_STATE_LEADS))
def test_only_a_state_of_one_request_and_the_configured_history_is_served(lead, history):
    """The reference asserts ``T == state_history_length``. Past ``process_prompt`` a wrong
    ``T`` fails the state encoder's forward and every request in that step with it, and a
    bare vector fails the decode once the whole forward has run."""
    model = _model(dataclasses.replace(_config(), state_history_length=history))
    model.processor = Gr00tN17Preprocessor(_DECODE_KWARGS, _DECODE_STATS, {"bot": 0}, 8, 6, _BytesProcessor())
    frames = [torch.full((180, 320, 3), 40 * i, dtype=torch.uint8) for i in range(2)]
    state = {"arm": np.zeros((*lead, 2), np.float32), "grip": np.zeros((*lead, 1), np.float32)}

    def serve():
        return model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": frames},
                                    embodiment_tag="bot", state=state)

    if lead == (1, history):
        assert serve()["state"][0].shape == (1, history, 8)
    else:
        with pytest.raises(ValueError, match=rf"`\(1, {history}, D\)`.*'arm'"):
            serve()


# Pillow's format name and save options for each frame format the API accepts
_LOSSLESS = {"PNG": {}, "WEBP": {"lossless": True}}


@pytest.mark.parametrize("fmt", list(_LOSSLESS), ids=str.lower)
def test_a_lossless_frame_loads_as_its_exact_pixels_in_uint8_hwc(tmp_path, fmt):
    """The reference's transform takes uint8 HWC; the base loader's float CHW in [0, 1] would
    be rescaled again by the HF processor into a black image, with no error."""
    from PIL import Image

    pixels = np.random.default_rng(0).integers(0, 256, (6, 10, 3), dtype=np.uint8)
    path = tmp_path / f"frame.{fmt.lower()}"
    Image.fromarray(pixels).save(path, format=fmt, **_LOSSLESS[fmt])
    frame = _model().load_image(str(path), "cpu").data

    assert frame.dtype == torch.uint8 and tuple(frame.shape) == (6, 10, 3)
    assert np.array_equal(frame.numpy(), pixels)


@pytest.mark.parametrize("fmt", ["JPEG", "GIF", "BMP", "TIFF"], ids=str.lower)
def test_any_other_frame_format_is_refused_before_it_is_decoded(tmp_path, fmt):
    """A JPEG passes every count, dtype and shape check and still moves the pixels 167x
    further than a lossless frame: the actions look fine and the robot moves wrong. GIF is
    a 256-colour palette; BMP and TIFF the engine's reader does not decode."""
    from PIL import Image

    path = tmp_path / f"frame.{fmt.lower()}"
    Image.fromarray(np.zeros((6, 10, 3), np.uint8)).save(path, format=fmt)
    with pytest.raises(ValueError, match="lossless|not in a format"):
        _model().load_image(str(path), "cpu")


def test_a_truncated_frame_is_a_400_not_a_500(tmp_path):
    """The header parses, so the format check passes; the decoder's RuntimeError would be a 500."""
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(path, format="PNG")
    path.write_bytes(path.read_bytes()[:80])
    with pytest.raises(ValueError, match="corrupt or truncated"):
        _model().load_image(str(path), "cpu")


def test_a_thin_frame_is_refused_before_the_resize_allocates_it(monkeypatch):
    """2 x 40 resizes to 256 x 5120 here: within any decode budget, past this one once the
    short side is scaled up. Refused before ``image`` runs, so nothing is allocated."""
    model, _, state = _observation_model()
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", str(1_000_000))
    monkeypatch.setattr(model.processor, "image", lambda img: pytest.fail("resized before the check"))
    thin = [torch.zeros((2, 40, 3), dtype=torch.uint8)] * 2
    with pytest.raises(ValueError, match="image budget"):
        model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": thin},
                             embodiment_tag="bot", state=state)


def test_cameras_may_differ_in_size_but_not_one_cameras_frames():
    """The reference takes ``video[view]`` per view; only a view's own frames must agree."""
    model, _, _ = _observation_model()
    two_cameras = json.loads(json.dumps(_DECODE_KWARGS))
    two_cameras["modality_configs"]["bot"]["video"]["modality_keys"] = ["cam", "wrist"]
    model.processor = Gr00tN17Preprocessor(two_cameras, _DECODE_STATS, {"bot": 0}, 8, 6)
    views, steps = model.processor.frame_layout("bot")
    assert len(views) == 2  # one view would never stack two resolutions
    sizes = [(30 + 10 * j, 40 + 10 * j) for j in range(len(views))]
    frames = [torch.full((*sizes[j], 3), 10 * t + j, dtype=torch.uint8)
              for t in range(steps) for j in range(len(views))]
    got = model._views_from_frames("bot", frames)
    for j, view in enumerate(views):
        assert got[view].shape == (1, steps, *sizes[j], 3)
        assert [int(got[view][0, t, 0, 0, 0]) for t in range(steps)] == [10 * t + j for t in range(steps)]


def test_a_camera_whose_frames_differ_in_size_is_named():
    model, _, _ = _observation_model()
    frames = [torch.zeros((30, 40, 3), dtype=torch.uint8), torch.zeros((60, 80, 3), dtype=torch.uint8)]
    with pytest.raises(ValueError, match="camera 'cam' sent frames of different sizes"):
        model._views_from_frames("bot", frames)


@pytest.mark.parametrize("wrist, accepted", [((360, 640), True), ((240, 320), False)],
                         ids=["same aspect, twice the size", "another aspect"])
def test_cameras_must_share_an_aspect_ratio_as_the_reference_stacks_them(wrist, accepted):
    """NVIDIA's processor stacks every view's transformed frames, so a camera at another aspect
    ratio fails there (``stack expects each tensor to be equal size``); here it is a 400."""
    two_cameras = json.loads(json.dumps(_DECODE_KWARGS))
    two_cameras["modality_configs"]["bot"]["video"]["modality_keys"] = ["cam", "wrist"]
    pre = Gr00tN17Preprocessor(two_cameras, _DECODE_STATS, {"bot": 0}, 8, 6, _BytesProcessor())
    video = {"cam": np.zeros((1, 2, 180, 320, 3), np.uint8), "wrist": np.zeros((1, 2, *wrist, 3), np.uint8)}
    state = {"arm": np.zeros((1, 1, 2), np.float32), "grip": np.zeros((1, 1, 1), np.float32)}
    call = lambda: pre(embodiment_tag="bot", video=video, language=[""], state=state)  # noqa: E731
    if accepted:
        call()
    else:
        with pytest.raises(ValueError, match="one aspect ratio"):
            call()


# each JSON type a client field can arrive as
_JSON_TYPES = {"an object": {"a": 1}, "a list": [1.0], "a number": 3, "a string": "x", "null": None}


def _observation_model():
    model = _model()
    model.processor = Gr00tN17Preprocessor(_DECODE_KWARGS, _DECODE_STATS, {"bot": 0}, 8, 6, _BytesProcessor())
    frames = [torch.full((180, 320, 3), 40 * i, dtype=torch.uint8) for i in range(2)]
    state = {"arm": np.zeros((1, 1, 2), np.float32), "grip": np.zeros((1, 1, 1), np.float32)}
    return model, frames, state


@pytest.mark.parametrize("field, value", [
    *(("embodiment_tag", v) for k, v in _JSON_TYPES.items() if k != "a string"),
    ("embodiment_tag", "not_a_robot"),
    *(("state", v) for k, v in _JSON_TYPES.items() if k != "an object"),
    *(("a state group", v) for v in _JSON_TYPES.values()),
    # null is the key left out (upstream's `normalize_generation_kwargs` drops it): the full path
    *(("walk", v) for k, v in _JSON_TYPES.items() if k != "null"),
], ids=lambda x: x if isinstance(x, str) and " " in x or x in ("embodiment_tag", "state", "walk") else repr(x))
def test_a_client_field_of_the_wrong_json_type_is_a_400_not_a_500(field, value):
    """The data worker answers ``ValueError``/``TypeError`` with a 400 and anything else with a
    500, so each field, sent as every JSON type but its own, must raise ``ValueError``."""
    model, frames, state = _observation_model()
    kwargs = {"embodiment_tag": "bot", "state": state}
    if field == "walk":
        with pytest.raises(ValueError):
            model._walk_schedule({"walk": value})
        return
    if field == "a state group":
        kwargs["state"] = {**state, "arm": value}
    else:
        kwargs[field] = value
    with pytest.raises(ValueError):
        model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": frames}, **kwargs)


@pytest.mark.parametrize("value", [[[[0.0], [0.0, 1.0]]], [[["x", "y"]]]], ids=["ragged", "strings"])
def test_a_state_group_that_is_not_a_numeric_array_is_named(value):
    """numpy's own message ("setting an array element with a sequence") names no field."""
    model, frames, state = _observation_model()
    with pytest.raises(ValueError, match="'arm': 'not a numeric array'"):
        model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": frames},
                             embodiment_tag="bot", state={**state, "arm": value})


@pytest.mark.parametrize("field", ["embodiment_tag", "state"])
def test_a_missing_client_field_is_a_400_naming_it(field):
    model, frames, state = _observation_model()
    kwargs = {"embodiment_tag": "bot", "state": state}
    del kwargs[field]
    with pytest.raises(ValueError, match=f"needs (an )?`{field}`"):
        model.process_prompt("", ["text", "image"], ["action"], tensors={"image_inputs": frames}, **kwargs)


def test_postprocess_names_every_joint_group_the_way_the_reference_does():
    """One float32 entry per joint group, keyed as NVIDIA's policy keys it
    (``gr00t_policy.py:425-432``). Groups have different widths (9, 1, 7 on DROID), so a
    flat payload in the wrong order moves the wrong joint with plausible numbers."""
    raw = _decoding_model().postprocess(_CHUNK, "action", {"embodiment_tag": "bot", "state": _RAW_STATE})
    assert isinstance(raw, bytes)
    got = msgpack.unpackb(raw)
    # the keys and their order are the embodiment's `modality_keys`
    assert list(got) == ["arm", "grip"]
    rebuilt = {
        name: np.frombuffer(g["data"], g["dtype"]).reshape(g["shape"]) for name, g in got.items()
    }
    assert [g["dtype"] for g in got.values()] == ["float32", "float32"]
    assert [g["shape"] for g in got.values()] == [[1, 2, 2], [1, 2, 1]]
    # the decoded numbers, in `modality_keys` order
    flat = np.concatenate([rebuilt["arm"], rebuilt["grip"]], axis=-1)
    assert flat.reshape(2, 3).tolist() == _DECODED
    # `grip` is the constant group
    assert rebuilt["grip"].reshape(-1).tolist() == [5.0, 5.0]


def test_postprocess_adds_the_batch_axis_the_engine_hands_back_without():
    """The worker hands back one request's ``(T, D)``, while ``unapply`` takes
    ``(B, T, D)``; without the added axis every decode would raise."""
    model = _decoding_model()
    kwargs = {"embodiment_tag": "bot", "state": _RAW_STATE}
    assert model.postprocess(_CHUNK[0], "action", kwargs) == model.postprocess(_CHUNK, "action", kwargs)


def test_postprocess_finds_the_tag_and_the_state_in_the_kwargs_the_data_worker_kept():
    """The decode needs the tag and the raw state at the end of the request; the conductor's
    kwargs are stripped of ``PREPROCESS_ONLY_KWARGS`` but the data worker's are not
    (``PreprocessWorkerThread._process_input``). Moving the state shifts the relative group,
    so a dropped state cannot pass as an absolute chunk."""
    assert {"embodiment_tag", "state"} <= set(Gr00tN17Model.PREPROCESS_ONLY_KWARGS)
    moved = {"arm": _RAW_STATE["arm"] + 100.0}
    raw = _decoding_model().postprocess(_CHUNK, "action", {"embodiment_tag": "bot", "state": moved})
    got = msgpack.unpackb(raw)
    arm = np.frombuffer(got["arm"]["data"], "float32").reshape(got["arm"]["shape"])
    grip = np.frombuffer(got["grip"]["data"], "float32").reshape(got["grip"]["shape"])
    assert arm.reshape(2, 2).tolist() == [[111.0, 119.0], [109.0, 121.0]]
    # `grip` is absolute, so moving the state must leave it alone
    assert grip.reshape(-1).tolist() == [5.0, 5.0]


@pytest.mark.parametrize("kwargs", [None, {}, {"state": _RAW_STATE}, {"embodiment_tag": None}])
def test_postprocess_without_the_requests_embodiment_tag_is_refused(kwargs):
    """No default tag is safe: eight embodiments of different widths and horizons, and a
    DROID chunk decoded on a G1's bounds looks right. A missing key and an explicit null
    arrive the same way."""
    with pytest.raises(ValueError, match="embodiment_tag"):
        _decoding_model().postprocess(_CHUNK, "action", kwargs)


@pytest.mark.parametrize("modality", ["text", ""])
def test_postprocess_refuses_a_modality_this_model_does_not_emit(modality):
    """A request asking for another modality is refused before the tag is read. One
    ``modality != "action"`` check serves every value, so "text" and the falsy "" are
    enough."""
    with pytest.raises(ValueError, match="emits actions"):
        _decoding_model().postprocess(_CHUNK, modality, {"embodiment_tag": "bot", "state": _RAW_STATE})


def test_every_output_a_walk_sends_the_client_is_one_postprocess_encodes():
    """The data worker hands every client output to ``postprocess``; an edge whose modality it
    refuses runs the whole walk and then answers 400. The reuse walks did that once."""
    model = _decoding_model()
    walks = model.get_graph_walk_graphs()
    kwargs = {"embodiment_tag": "bot", "state": _RAW_STATE}
    for walk, section in walks.items():
        edges = _edges(section) + list(getattr(section, "outputs", None) or [])
        for edge in (e for e in edges if e.next_node == EMIT_TO_CLIENT):
            assert isinstance(model.postprocess(_CHUNK, edge.output_modality, kwargs), bytes), walk


@pytest.mark.parametrize("shape", [(5, 4), (1, 1, 3)], ids=["backbone rows", "one state token"])
def test_a_reuse_walks_tensor_comes_back_exactly_with_its_dtype_and_shape(shape):
    """bf16 features widen to float32 exactly, and the entry carries what a client needs to
    rebuild them, laid out like an action group."""
    features = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    entry = msgpack.unpackb(_decoding_model().postprocess(features, "tensor", {}))
    rebuilt = np.frombuffer(entry["data"], entry["dtype"]).reshape(entry["shape"])
    assert entry["dtype"] == "float32" and tuple(entry["shape"]) == shape
    assert np.array_equal(rebuilt, features.float().numpy())


# ----------------------------------------------------------------------------
# Against the checkpoint
# ----------------------------------------------------------------------------


@needs_checkpoint
def test_the_specs_geometries_are_the_checkpoints():
    """What the geometries are read as: 16 x 64 ViT, 16 queries over 8 KV heads x 128
    LLM, 32 x 64 bridge, 32 x 48 for both DT specs. Config only."""
    model = _model(Gr00tN17Config.from_hf(GR00T, BACKBONE))
    specs = _specs(model)

    assert {
        key: (spec.config.num_qo_heads, spec.config.num_kv_heads, spec.config.head_dim)
        for key, spec in specs.items()
    } == _REAL
    assert model.config.dit.num_layers == 32  # the 1.12B the walks never build
    assert model.get_graph_walk_graphs()[_ACTION_GEN].max_iters == 4


@needs_checkpoint
def test_the_five_nodes_build_at_the_checkpoints_widths_on_meta():
    """At the checkpoint's widths on ``meta`` (no bytes for the 1.12B DT), a width read from
    the wrong field raises where the tiny config happens to line up. Loader stubbed."""
    model = _model(Gr00tN17Config.from_hf(GR00T, BACKBONE))
    model._repo_dir = GR00T
    with torch.device("meta"):
        built = {name: _unloaded(model, name) for name in _SUBMODULES}
    assert {name: type(sub) for name, sub in built.items()} == _SUBMODULES

    dt = built[DENOISE_STEP]
    assert len(dt.model.transformer_blocks) == 32
    assert sum(p.numel() for p in dt.parameters()) > 1_000_000_000
    assert all(p.device.type == "meta" for p in dt.parameters())


# The module each node's checkpoint tensors are named against: the wrapped component, or the
# submodule itself (`_create_submodule`).
_LOAD_TARGET = {
    VISION_ENCODER: lambda node: node.model,
    VL_BRIDGE: lambda node: node.model,
    STATE_ENCODER: lambda node: node,
    DENOISE_STEP: lambda node: node,
}


def _meta_node(node: str):
    model = _model(Gr00tN17Config.from_hf(GR00T, BACKBONE))
    model._repo_dir = GR00T
    with torch.device("meta"):
        return model, _unloaded(model, node)


@needs_checkpoint
@pytest.mark.parametrize("node", list(_LOAD_TARGET))
def test_the_checkpoints_tensors_name_and_shape_each_nodes_parameters(node):
    """Every tensor the loader's remapper routes to ``node`` is one of its parameters at the
    checkpoint's shape, and every parameter gets one; shapes come off the safetensors
    headers, so no tensor is read."""
    _, built = _meta_node(node)
    weight_map = json.loads((GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    remap = _remapper(node)
    checkpoint = {}
    for key, shard in weight_map.items():
        if name := remap(key):
            with safe_open(str(GR00T / shard), framework="pt") as shard_file:
                checkpoint[name] = shard_file.get_slice(key).get_shape()

    assert checkpoint == {name: list(p.shape) for name, p in _LOAD_TARGET[node](built).named_parameters()}


@needs_checkpoint
def test_the_denoise_steps_checkpoint_pieces_and_widths():
    """467 tensors over four pieces: the stack's 456 is 32 blocks x 14, the timestep
    encoder's 4 and the output projections' 4, so no ``norm_out`` and no 33rd block. The
    blocks alternate cross (keys 2048 wide) and self, cross first, and the position table
    holds 1024 rows for the 40 action rows the DT reads, so a slice off the front never raises."""
    model, node = _meta_node(DENOISE_STEP)
    config = model.config
    remap = _remapper(DENOISE_STEP)
    weight_map = json.loads((GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for key in weight_map if (name := remap(key))]

    assert Counter(name.partition(".")[0] for name in names) == {
        "action_encoder": 6, "model": 456, "action_decoder": 4, "position_embedding": 1,
    }
    assert len(names) == 467
    assert (config.backbone_embedding_dim, config.dit.inner_dim) == (2048, 1536)
    assert config.dit.num_layers == 32
    assert [block.attn1.to_k.in_features for block in node.model.transformer_blocks] == [
        config.backbone_embedding_dim if index % 2 == 0 else config.dit.inner_dim
        for index in range(config.dit.num_layers)
    ]
    assert node.position_embedding.weight.shape == (config.max_seq_len, config.input_embedding_dim)
    assert node.position_embedding.weight.shape == (1024, 1536)
    assert config.action_horizon + 1 == 41 < config.max_seq_len


@needs_checkpoint
def test_the_model_constructs_from_its_two_checkpoints():
    """``__init__`` is otherwise routed around, and its processor call was once wrong.
    Two repos: the action head is ``nvidia/GR00T-N1.7-3B``, the Qwen3-VL dims are
    ``nvidia/Cosmos-Reason2-2B``. ``max_action_horizon`` must be the config's: the
    processor refuses a too-long embodiment against it (``Gr00tN17Preprocessor.__init__``)."""
    model = Gr00tN17Model(model_path_hf=str(GR00T), backbone_path_hf=BACKBONE)

    assert model.config == Gr00tN17Config.from_hf(GR00T, BACKBONE)
    assert isinstance(model.processor, Gr00tN17Preprocessor)
    assert model.processor.max_state_dim == model.config.max_state_dim
    assert model.processor.max_action_horizon == model.config.action_horizon
    # 3 views x 2 timesteps: the widest embodiment in this checkpoint, off the processor
    assert _specs(model)[VISION_ATTN].config.max_segments_per_request == 6
    # a frozenset, as `Model.request_kwargs` declares; no `video`: frames travel the media path
    assert model.request_kwargs() == frozenset({"embodiment_tag", "state", "walk"})
    # `walk` stays on the control wire: the conductor chooses the schedule with it
    assert set(Gr00tN17Model.PREPROCESS_ONLY_KWARGS) < model.request_kwargs()
    assert "walk" not in Gr00tN17Model.PREPROCESS_ONLY_KWARGS
    # one frame per view per timestep, refused before any file is decoded
    assert model.max_input_files() == {"image": 3 * 2}
