"""GR00T N1.7's model class: the four walks, the five resource specs, the walk
schedule and the five submodules -- all on a config, with no weights anywhere.

This file is structural, so almost every assertion is an exact set or an exact
identity rather than a bar. There is nothing to round.

The walks come first, because they are [[D02]]'s claim and the claim is made in a
shape rather than in a value. Three of those shapes are load-bearing and none of
them is checkable by reading the node list:

* ``prefill`` is a ``Parallel``. The state encoder reads only the request's own
  state, so it must not wait on a 16-layer backbone. A ``Sequential`` over the
  same two branches declares a dependency that does not exist; with no edge
  between them the two sections derive the same I/O signature
  (``get_inputs_outputs`` is byte-for-byte equal), so the type **is** the
  declaration, and the independence is pinned beside it as the property that
  makes the type honest.
* A **persist** edge targets ``EMPTY_DESTINATION``. ``denoise_step`` is not in
  ``prefill``, so an edge pointing at it names a node the walk's registry has
  never heard of. The first version of this file's subject did exactly that, and
  nothing raised: the tensor is parked for the request either way, and the
  destination is only read when the edge is routed.
* The ``Loop``'s terminal output is matched into the section's own outputs **by
  name** -- pi05 documents the trap (``pi05_model.py:447-455``) and
  ``Loop.__post_init__`` enforces it by *dropping* a terminal output the section
  does not produce. A rename leaves the walk emitting nothing at all, which the
  negative control at the end of that test shows on a Loop built here.

Then what each node takes and emits, because a node's declaration is one half of
a contract whose other half is in ``submodules.py`` and nothing checks the two
against each other. Two of those joins are a one-sided rename away from a graph
that still looks connected: ``deepstack_embeds`` rides beside ``vision_embeds``
between the same pair of nodes, and ``image_rows`` is something the LLM *emits*
after deriving it, so declaring it an input of the LLM turns the walk's own edge
into a loop-back on a node that already ran.

Then the join across the walk boundary, which is the one piece of the design that
is written nowhere: ``action_gen``'s node needs five inputs -- three parked by
``prefill``'s nodes, one parked by the request, one carried by the loop -- all
matched by name across a walk boundary. Nothing in either declaration refers to
the other, and the carried one has to be the *scaffold's* name
(``DenoiseLoopSubmodule.LATENTS``) rather than a name this file chose, because the
scaffold is what reads it back.

Then the resources (PR-1 §3.4). Every head geometry is read off the config twice,
at two different widths, so a hardcoded head count cannot satisfy both. The DT
holds **two** specs of different kinds at the same 32 x 48 geometry, because a
resource is one *kind* of attention; and ``causal`` is pinned as **absent** from
the ragged config, since causality is a property of the step
(``submodules.py:227``).

Then the request's two forward-pass hops, where the bucket key's span counts are
derived from the persisted tensors' own ``dims``; then ``get_submodule``.

Then ``postprocess``, the one thing in this file that carries a value rather than
a shape. The decode itself is the processor's and is tested in
``test_gr00t_n17_preprocessing.py``; what belongs here is the three things the
*model* adds on top of it -- the batch axis the engine's ``(T, D)`` is missing,
the two request kwargs it has to find in a dict the conductor stripped, and the
concatenation order that is the client's only handle on which float is which
joint. It gets a processor built on a four-line synthetic ``statistics``, so this
file still reads no checkpoint.

The walks and the specs are checked on a config alone, through
``object.__new__`` -- no checkpoint, no processor, no network. ``get_submodule``
loads its own weights, so its tests serve the tiny config off a synthetic
checkpoint in ``tmp_path`` under the real tensor names. At the checkpoint's own
widths the five nodes are built under ``torch.device("meta")`` with the loader
stubbed, because ``denoise_step`` holds the real DT: 32 blocks of 1536 x 32 x 48,
1.12B parameters, 4.2 GiB per fp32 copy.

Four of the claims here were ``xfail`` for one round, against a version whose
carried edge was ``actions``, whose ``get_partition_forward_pass_args`` routed
nothing but that edge, whose ``embodiment_id`` was freed at the end of
``prefill``, and whose ``__init__`` could not run. Each docstring says what the
symptom was, because the symptom is the argument for the assertion.
"""

from __future__ import annotations

import dataclasses
import glob
import inspect
import re
from pathlib import Path

import msgpack
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mstar.conductor.request_info import CurrentForwardConductorMetadata, CurrentForwardPassInfo
from mstar.engine.resources import (
    AttentionStep,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
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
    Gr00tN17StateEncoderSubmodule,
    Gr00tN17VisionEncoderSubmodule,
    Gr00tN17VLBridgeSubmodule,
)
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
from mstar.model.submodule_base import NodeInputs

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

# The walk names, spelled out rather than read off the class: these are the keys
# the YAML configs and the registry name, so they are part of the contract and
# not an implementation detail.
_PREFILL, _BACKBONE_WALK, _STATE, _ACTION_GEN = "prefill", "backbone", "state", "action_gen"

# What `prefill`'s nodes park for the request, and what `action_gen` reads back.
_PERSISTED = {"vl_text", "vl_image", "state_features"}
# The loop-carried edge. Spelled out rather than imported, because the claim that
# it *is* `DenoiseLoopSubmodule.LATENTS` is a test of its own below -- importing
# the constant here would make that test compare the scaffold with itself.
_CARRIED = "latents"

# Each node's contract, read off `submodules.py`. Spelled out rather than derived:
# these are the names the tests below exist to catch changing, and a one-sided
# rename on either side of the node boundary is exactly the defect they catch.
#
# The cite beside each is the `prepare_inputs` that reads the name, or the
# `forward` that returns it.
_NODE_INPUTS = {
    # `pixel_values` are the packed frames, `image_grid_thw` the per-frame grid
    # the segment maths needs (`submodules.py:86,96`)
    VISION_ENCODER: {"pixel_values", "image_grid_thw"},
    # the ViT's two outputs, the prompt, and the grid again -- the LLM places the
    # image rows with it (`submodules.py:180-183,207`). NOT `position_ids` or
    # `image_rows`: it derives both in `prepare_inputs` (`submodules.py:196-209`),
    # and `image_rows` is something it *emits*.
    LLM: {"vision_embeds", "deepstack_embeds", "input_ids", "image_grid_thw"},
    VL_BRIDGE: {"backbone_features", "image_rows"},  # `submodules.py:361-362`
    STATE_ENCODER: {"state", "embodiment_id"},  # `submodules.py:290-291`
    # the carried sample (`denoise_loop.py:399`) and the frozen conditioning
    # (`submodules.py:536,550,563`)
    DENOISE_STEP: {_CARRIED, "vl_text", "vl_image", "state_features", "embodiment_id"},
}
_NODE_OUTPUTS = {
    VISION_ENCODER: {"vision_embeds", "deepstack_embeds"},  # `submodules.py:154`
    LLM: {"backbone_features", "image_rows"},  # `submodules.py:266`
    VL_BRIDGE: {"vl_text", "vl_image"},  # `submodules.py:419-420`
    STATE_ENCODER: {"state_features"},  # `submodules.py:331`
    DENOISE_STEP: {_CARRIED},  # `denoise_loop.py:426`
}
# What the LLM derives rather than being handed, so neither may be one of ITS
# inputs: `image_rows` is an LLM output (the bridge takes it) and `position_ids`
# never leaves the node that builds it.
_DERIVED_BY_THE_LLM = {"position_ids", "image_rows"}

# What a request carries into `prefill`, and where each one goes. Pairs rather
# than a set of names, because `image_grid_thw` is needed by two nodes: the ViT
# sizes its per-frame segments from it and the LLM places the image rows with it.
_ROUTES = (
    ("pixel_values", VISION_ENCODER), ("image_grid_thw", VISION_ENCODER),
    ("image_grid_thw", LLM), ("input_ids", LLM),
    ("state", STATE_ENCODER), ("embodiment_id", STATE_ENCODER),
)
# the distinct signal names, in first-seen order
_ROUTED = tuple(dict.fromkeys(name for name, _ in _ROUTES))

# The checkpoint's geometries (§3.4), spelled out: these are the numbers the
# config tests below exist to catch changing. Checked against the real config in
# the last section.
_REAL = {
    VISION_ATTN: (16, 16, 64),
    LLM_ATTN: (16, 8, 128),
    BRIDGE_ATTN: (32, 32, 64),
    DT_ATTN: (32, 32, 48),
    DT_XATTN: (32, 32, 48),
}


@pytest.fixture(autouse=True)
def _pinned_threads():
    """Nothing here runs a forward, but the import pulls torch in: keep the thread
    count where the other gr00t tests keep it so a stray kernel is deterministic."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ----------------------------------------------------------------------------
# A config with no two widths alike, and a model built around it
# ----------------------------------------------------------------------------

# Every head count and head dim here is a different number, and none of them is
# one of the checkpoint's (16, 8, 32, 48, 64, 128), so a spec that hardcoded a
# geometry instead of reading the config cannot land on any of them by accident.
_VISION_HEADS, _VISION_WIDTH = 5, 20
_TEXT_QO, _TEXT_KV, _TEXT_HEAD_DIM = 6, 3, 4
_BRIDGE_HEADS, _BRIDGE_HEAD_DIM = 7, 9
_DT_HEADS, _DT_HEAD_DIM = 3, 11

# Per-embodiment video keys, as the processor reads them off the checkpoint. Three
# embodiments with 1, 3 and 2 views: the widest is neither the first nor the last,
# so a `max` replaced by an index would have to be lucky twice.
_VIDEO_KEYS = {
    "gr1": ["video.ego_view"],
    "oxe": ["video.image_0", "video.image_1", "video.image_2"],
    "franka": ["video.front", "video.wrist"],
}
_MAX_VIEWS = 3


class _Processor:
    """Only the field ``max_views`` reads. The real processor needs the checkpoint's
    three json files and a Qwen3-VL ``AutoProcessor``, and none of that is a
    property of the walks or the resources."""

    def __init__(self, video_keys: dict[str, list[str]]):
        self.video_keys = video_keys


def _config(scale: int = 1) -> Gr00tN17Config:
    """The tiny config at ``scale`` x every head count.

    ``scale`` is what makes the geometry tests tests of the *read*: the same
    assertions run at two different sets of numbers, and one constant cannot be
    both.
    """
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
    """The model without its ``__init__``: the walks, the resources and the walk
    schedule are functions of the config alone, and ``__init__``'s job is to read
    that config off two checkpoints and load a processor.

    ``object.__new__`` rather than a ``skip_weight_loading`` flag (ltx2_5's
    idiom), because this class has no such flag -- adding one to serve a test
    would be a new mechanism in the model for no behaviour.
    """
    model = object.__new__(Gr00tN17Model)
    model.config = config if config is not None else _config()
    model.processor = _Processor(_VIDEO_KEYS if video_keys is None else video_keys)
    model._submodules = {}
    return model


def _info(dims: list[int], uuid: int = 0) -> TensorPointerInfo:
    """A descriptor and nothing behind it: ``get_partition_forward_pass_args``
    reads ``dims`` and nothing else."""
    return TensorPointerInfo(
        dims=dims, dtype=torch.bfloat16, nbytes=2, address=0,
        stride=[1] * len(dims), uuid=uuid, source_session_id="test:0",
        source_entity="test",
    )


def _metadata(model: Gr00tN17Model, walk: str | None = None) -> CurrentForwardConductorMetadata:
    """The request's metadata as the *model* builds it, not as a test imagines it.

    The transition hook reads the walk schedule the entry point parked in
    ``kwargs``, so a hand-rolled ``kwargs={}`` would test a state no request can
    be in. Taking it from ``get_initial_forward_pass_args`` means a change to the
    schedule shows up here as a behaviour change rather than a stale literal.
    """
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


def test_the_walks_are_d02s_four_over_d02s_five_nodes():
    """If this fails, a walk or a node was added, dropped or renamed. The names
    are a contract: the YAML partitions and node_groups key off them, and a walk
    nothing names is a walk the conductor refuses at startup."""
    model = _model()

    assert set(model.get_graph_walk_graphs()) == {_PREFILL, _BACKBONE_WALK, _STATE, _ACTION_GEN}
    assert model.nodes == [LLM, DENOISE_STEP, STATE_ENCODER, VISION_ENCODER, VL_BRIDGE]
    assert (VISION_ENCODER, LLM, STATE_ENCODER, VL_BRIDGE, DENOISE_STEP) == (
        "vision_encoder", "LLM", "state_encoder", "vl_bridge", "denoise_step",
    )


def test_prefill_is_a_parallel_whose_two_branches_never_wait_on_each_other():
    """The state encoder reads the request's own proprioceptive state and nothing
    else, so it must not sit behind a 16-layer backbone and a ViT.

    A ``Sequential`` over these same two branches would be served *identically* --
    with no edge between them, ``Sequential.get_inputs_outputs`` and
    ``Parallel.get_inputs_outputs`` return equal signatures, and
    ``top_level_entities`` flattens both the same way -- so the type is the only
    place the independence is written down, and the type alone is what a reader
    (and a reviewer) goes by. Hence both halves here: the declaration, and the
    property that makes the declaration true.
    """
    prefill = _model().get_graph_walk_graphs()[_PREFILL]

    assert isinstance(prefill, Parallel)
    conditioning, state = prefill.sections
    assert isinstance(conditioning, Sequential)
    assert list(_nodes(conditioning)) == [VISION_ENCODER, LLM, VL_BRIDGE]
    assert isinstance(state, GraphNode) and state.name == STATE_ENCODER

    # Independence: neither branch consumes anything the other produces. This is
    # exactly the condition under which a `Parallel` is correct, and the thing a
    # `Sequential` would be asserting is false.
    left, right = set(_nodes(conditioning)), set(_nodes(state))
    assert left.isdisjoint(right)
    for branch, other_nodes in ((conditioning, right), (state, left)):
        produced = {edge.name for edge in _edges(branch)}
        other = conditioning if branch is state else state
        assert not (produced & {name for name, _ in other.get_inputs_outputs().ext_inputs})
        assert not ({edge.next_node for edge in _edges(branch)} & other_nodes)
    assert prefill.get_inputs_outputs().loop_back == set()

    # and every input the walk needs arrives from the request, at the node that
    # declares it -- nothing is produced inside the Parallel. `image_grid_thw`
    # appears twice because two nodes need it, and the signature is keyed by
    # (name, destination): a name routed to one node is not routed to the other.
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
    """``denoise_step`` is in ``action_gen``, not here, so an edge pointing at it
    from ``vl_bridge`` names a node this walk's registry has never heard of. That
    was the first version of this file's subject, and nothing raised: a persisted
    tensor is parked for the request whatever its ``next_node`` says, and the
    destination is read only when the edge is *routed* -- which this one never is.
    The walk would run, the tensors would be parked, and the only thing wrong
    would be a lie in the declaration a reviewer reads.

    The rule is this walk's, not the framework's: ``persist`` and a real
    destination are compatible in general, and the runtime says so in as many
    words (``graph/runtime/python.py:1057-1060``, "persist edges may ALSO route
    to a worker"). What makes ``EMPTY_DESTINATION`` the only honest value here is
    that there is no node in ``prefill`` for these three to go to.
    ``EMPTY_DESTINATION`` is the idiom flux2_klein's ``TEXT_EMBEDS``
    (``flux2_klein_model.py:193``) and ltx2_5's ``text_video`` use for it.
    """
    prefill = _model().get_graph_walk_graphs()[_PREFILL]
    persisted = [edge for edge in _edges(prefill) if edge.persist]

    assert {edge.name for edge in persisted} == _PERSISTED
    assert {edge.next_node for edge in persisted} == {EMPTY_DESTINATION}
    assert DENOISE_STEP not in _nodes(prefill)  # there is nothing here to point at

    # the general form: no edge in this walk names a node outside it
    inside = set(_nodes(prefill)) | {EMPTY_DESTINATION, EMIT_TO_CLIENT}
    assert {edge.next_node for edge in _edges(prefill)} <= inside


def test_a_persisted_request_input_keeps_its_destination():
    """The other half of the rule above, and the one place in this model where a
    persisted edge **does** name a node. ``embodiment_id`` is a request input: it
    has to reach ``state_encoder`` *and* survive ``prefill``, because the denoise
    step reads it on every iteration (``submodules.py:550``). So its edge carries
    both -- ``next_node=state_encoder`` for the routing and ``persist=True`` so the
    model leaves it out of ``unpersist_tensors``.

    ``persist`` on an *input* edge is the model's own marker and nothing else's:
    the conductor parks every initial signal regardless
    (``conductor.py:1000``, ``persist_signals=body.initial_signals``) and releases
    what ``unpersist_tensors`` names, while the runtime reads ``persist`` only on
    node *outputs* (``graph/runtime/python.py:997``). So the flag does not
    interfere with the routing, which is what makes this edge able to be both
    things at once -- waypoint persists its request inputs the same way
    (``waypoint_model.py:607-617``).

    The release list it is the complement of is per **signal**, not per edge: one
    entry for every initial signal whose uuid this walk did not park. The obvious
    formula -- one entry per non-persisted *edge* -- is wrong twice over, and the
    two mistakes pull in opposite directions. ``image_grid_thw`` is routed to two
    nodes, so it would be released twice; and ``action_mask``, which the processor
    always produces (``preprocessing.py:203``) and no node declares, has no edge
    at all, so it would never be released -- while ``data_worker.py:583`` persists
    *every* initial signal, which makes the model the only thing that can say it
    is finished with one. An unreleased signal is held for the request's life.
    """
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
    """``Loop.__post_init__`` keeps only the terminal outputs the section actually
    produces (``base.py:641``), matching **by name**. So a terminal output named
    anything other than the loop-back edge is not a mis-emission -- it is silently
    *deleted*, and ``action_gen`` emits nothing at all. pi05 carries the warning in
    a comment (``pi05_model.py:447-455``) for the same reason.

    The negative control at the bottom is that mechanism on a Loop built here, so
    the claim is not a reading of the source.
    """
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
    # the edge is a graph-internal key and the modality is what the client reads,
    # so the two say different things on purpose: the walk carries `latents` (the
    # scaffold's name for the sample) and emits it as an `action`
    assert terminal.name != terminal.output_modality

    # the trap itself: one letter off, and the walk has no outputs
    renamed = Loop(
        section=GraphNode(
            name=DENOISE_STEP, input_names=[_CARRIED],
            outputs=[GraphEdge(next_node=DENOISE_STEP, name=_CARRIED)],
        ),
        max_iters=4,
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="action", output_modality="action")],
    )
    assert renamed.outputs == []


# ----------------------------------------------------------------------------
# What each node takes and emits
# ----------------------------------------------------------------------------


def test_each_node_declares_the_contract_its_submodule_reads():
    """A node's ``input_names`` and its output edge names are one half of a
    contract whose other half is in ``submodules.py``, and nothing checks the two
    against each other: ``prepare_inputs`` indexes ``inputs[name]`` and raises a
    ``KeyError`` naming a dict, and ``forward`` returns a dict whose keys the
    worker matches against the *graph's* edge names
    (``graph/runtime/python.py:401-405``) -- a key that matches nothing is
    dropped without a word.

    So each row of ``_NODE_INPUTS`` / ``_NODE_OUTPUTS`` is cited to the line that
    reads or returns it. Inputs are ``==``, because a node always needs all of
    them; outputs are ``<=``, because a reuse walk stops early and emits less than
    ``prefill``'s copy of the same node does. The union over the walks closes that
    back to ``==``, so an output no walk ever uses is caught too.
    """
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


def test_the_llm_is_handed_neither_of_the_two_things_it_derives():
    """``position_ids`` and ``image_rows`` are the LLM's own work: it builds the
    M-RoPE positions and finds the image rows from ``input_ids`` and
    ``image_grid_thw`` in ``prepare_inputs`` (``submodules.py:196-209``), and then
    **emits** ``image_rows`` so the bridge splits on the same tensor rather than
    deriving it a second time (``submodules.py:264-266``).

    Declaring either as a graph input is the mistake this catches, and it is a
    quiet one in two different ways. ``position_ids`` has no producer anywhere, so
    the node would simply never become ready. ``image_rows`` does have a producer
    -- the LLM itself -- so declaring it as an input of the LLM makes the walk's
    own output edge a **loop-back** on the first node that already ran
    (``Sequential.get_inputs_outputs``, ``graph/base.py:420-422``) -- a different
    graph, built from the same text.
    """
    walks = _model().get_graph_walk_graphs()
    every_input = {name for walk in walks.values() for node in _nodes(walk).values()
                   for name in node.input_names}

    # neither reaches the LLM, in any walk
    for walk_name, walk in walks.items():
        llm = _nodes(walk).get(LLM)
        if llm is not None:
            assert llm.input_names.isdisjoint(_DERIVED_BY_THE_LLM), walk_name
    # and `position_ids` is nobody's input: it never leaves the node that builds it
    assert "position_ids" not in every_input
    assert "image_rows" in _NODE_OUTPUTS[LLM]
    # the bridge is the one node that takes it, and it takes it from the LLM
    assert {name for name, inputs in _NODE_INPUTS.items() if "image_rows" in inputs} == {VL_BRIDGE}
    # and no walk turned an edge into a loop-back by accident
    for name, walk in walks.items():
        assert walk.get_inputs_outputs().loop_back == (
            {(_CARRIED, DENOISE_STEP)} if name == _ACTION_GEN else set()
        ), name


def test_every_internal_edge_lands_on_an_input_its_destination_declares():
    """The producer/consumer join *inside* a walk, derived rather than tabulated:
    every edge that names a real node has to name a signal that node declares.

    This is the one-sided rename, which is the same defect as the walk-boundary
    join but cheaper to make -- the two ends are twenty lines apart in one
    expression, so they look like they must agree. ``deepstack_embeds`` is the
    dangerous one: it rides beside ``vision_embeds`` from the same node to the
    same node, so renaming it on one side leaves a walk that still has an edge
    between those two nodes and still looks connected.
    """
    for walk_name, walk in _model().get_graph_walk_graphs().items():
        nodes = _nodes(walk)
        landed = set()
        for source in nodes.values():
            for edge in source.outputs:
                if edge.next_node in nodes:
                    assert edge.name in nodes[edge.next_node].input_names, (walk_name, edge.name)
                    landed.add((edge.name, edge.next_node))
        # and the walk's own signature agrees: an input is either landed on from
        # inside the walk or external, never neither
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
    """The join the whole design rests on, and it is written nowhere. ``prefill``
    parks three tensors and the request parks a fourth; ``action_gen``, a separate
    walk built by a separate expression, asks for them by name; the graph layer
    matches the two strings and nothing else. Renaming ``vl_text`` on either side
    alone leaves two sets that no longer meet, and the only symptom at runtime is
    a ``denoise_step`` whose inputs never complete -- the request stops, with no
    node to blame.

    Both suppliers are *derived* here rather than listed: the node-persisted set
    off ``prefill``'s own edges, the request-persisted set off
    ``get_initial_forward_pass_args``. So the test spans the two hooks that have
    to agree, and ``_PERSISTED`` is the only name written down twice.

    Spelled as a partition of ``input_names`` so a *new* input is caught too: an
    input supplied by neither hook and not carried by the loop has no supplier at
    all.
    """
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
    """The second half of the name join, and the one the graph cannot catch.

    ``denoise_step`` is a ``DenoiseLoopSubmodule`` ([[D07]]), so the carried sample
    is the scaffold's, not this file's: ``_carried_inputs`` reads
    ``inputs[name]`` for every name in ``loop_back_names`` (``denoise_loop.py:399``)
    and ``_run`` returns ``{self.LATENTS: out}`` (``denoise_loop.py:426``), while
    the worker routes a forward's outputs by the *graph's* edge names
    (``graph/runtime/python.py:401-405``). So the walk's edge name and
    ``LATENTS`` are one string that happens to be written in two places, with
    nothing between them to make them agree.

    They disagreed once, and the symptom was not subtle: with the walk naming it
    ``actions``, iteration 0 is seeded and runs and iteration 1 raises
    ``RuntimeError: loop-back edge(s) ['latents'] missing``. ltx2_5, on the same
    scaffold, names its edge ``latents`` for exactly this reason, and the
    scaffold's own docstring says a subclass that renames it must rename
    ``denoise``'s parameter to match (``denoise_loop.py:88-92``) -- which is the
    last assertion here, since the scaffold passes the carried tensors by keyword.
    """
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
    """``persist_signals`` is handed to the model and the model builds the edges:
    the conductor does not route a parked tensor into the next walk by itself. It
    parks every initial signal and every persisted node output, passes the dict to
    this hook (``conductor.py:1487``) and nothing else reads it. So every input
    ``denoise_step`` declares, apart from the one the scaffold seeds, has to appear
    here as a ``GraphEdge`` carrying that name's parked descriptors -- which is
    flux2_klein's shape (``flux2_klein_model.py:406-412``) and ltx2_5's.

    It once emitted only the carried edge, and the loop simply never became ready:
    the four conditioning inputs had no supplier, so the node's ``ReadySignals``
    never completed and the request stopped with nothing to blame.

    ``delivered`` is compared against the node's own ``input_names`` rather than a
    list, so an input added to the walk without a line here fails.
    """
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
    # each one carries the parked descriptors; the carried edge arrives empty,
    # because the scaffold draws iteration 0's sample itself
    for edge in args.inputs:
        assert edge.tensor_info == ([] if edge.name == _CARRIED else persist[edge.name])
    # the conditioning is read for the last time by this walk, so it is released
    # here; the Loop re-injects it across its own iterations (`_persist_for_loop`)
    assert args.unpersist_tensors == sum(
        (persist[name] for name in ("vl_text", "vl_image", "state_features", "embodiment_id")),
        start=[],
    )


# ----------------------------------------------------------------------------
# D02's reuse walks
# ----------------------------------------------------------------------------


def test_the_reuse_walks_expose_the_nodes_prefill_already_contains():
    """[[D02]]'s reason for existing: another model takes GR00T's backbone or its
    state encoder on its own, so each reuse walk has to be ``prefill``'s own nodes
    stopping early -- same node names, same inputs, same internal edge -- and not
    a second, parallel description that can say something else.

    Structural only. That the two walks produce the same *numbers* as the
    corresponding node inside ``prefill`` is C08 CHK-4's (``torch.equal``, per
    D02's judging table); there are no weights in this file to compare.

    ``vl_bridge`` staying out of ``backbone`` is the decision's other half: its
    weights are GR00T's ``action_head.*``, so the reusable walk ends at the raw
    Qwen3-VL features.
    """
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

    # where `prefill` forwards or parks, a reuse walk emits to the client, as a
    # `tensor`: "embedding" is not in `entrypoint.py:43` `SUPPORTED_MODALITIES`,
    # so the server rejects the response rather than the request -- the walk runs
    # to completion and the client gets an error for work that succeeded.
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
    """``walk`` on the request is the only way into either reuse walk, and the
    walk sequence is decided *here*, once, and parked in ``metadata.kwargs`` --
    flux2_klein's idiom (``flux2_klein_model.py:358-361``), where the schedule is
    built from the request's own inputs and nothing recomputes it later.

    Two things break if the schedule is wrong, and neither is visible at the
    request's own node. The first entry is what the conductor picks the worker
    graphs from (``conductor.py:1030``), so a schedule still beginning with
    ``prefill`` runs the full path and the client gets an *action* for a request
    that asked for backbone features. And a schedule of more than one entry sends
    the request on to a second walk: ``action_gen``'s node needs three tensors a
    reuse walk never parks, so the loop's inputs never complete and the request
    stops with no node to blame.

    ``walk`` is also declared on :meth:`request_kwargs`, because that list is
    every ``model_kwargs`` key the model reads (``model/base.py:757-762``) and a
    key missing from it is reported back to the client as ignored -- while in
    fact deciding the whole schedule. The existing assertion on the full list
    is behind the checkpoint skip, and this one is not.
    """
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
    """The defect that made the schedule necessary, and it is a hang rather than
    an error. A reuse walk emits to the client and ends, so its first transition
    has to say ``request_done``. Before the schedule existed the hook re-emitted
    the walk it was already in with no inputs, and the conductor reads *that
    pair* -- not done, no inputs -- as "partition self-triggers via StreamBuffer"
    (``conductor.py:1528-1531``): it sends nothing. GR00T has no stream, so
    nothing ever triggers the partition again and the request sits there until it
    is killed, holding its worker graphs.

    So ``request_done`` is the assertion that matters: re-emitting the walk would
    leave it false with an empty ``inputs``, which is exactly the state the
    conductor turns into silence.

    ``persist_signals`` is empty on purpose -- a reuse walk parks nothing, which
    is the other half of ``embodiment_id`` not being persisted on this path.
    """
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
    """A typo'd ``walk`` has to raise at admission, because every other outcome is
    worse. Falling back to the full path serves an action to a caller who asked
    for features; passing the string through makes it ``metadata.graph_walk``,
    which the conductor looks up to pick worker graphs.

    ``prefill`` and ``action_gen`` are refused by the same clause, and that is the
    interesting half: they are real walk names, so a caller is likely to try one,
    and the quiet failure is specific enough to be worth saying out loud. A
    schedule of ``[prefill]`` would run, park its three conditioning tensors
    (their edges all persist to ``EMPTY_DESTINATION``), be done, and emit nothing
    to the client at all -- a request that succeeds and returns no output.

    So the message is asserted to name all four: the two a caller may ask for,
    because ``walk`` is one opaque string in ``request_kwargs`` and this is the
    only place to learn them, and the two it may not, because being absent from a
    list is not an explanation.
    """
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
    """The routes are ``prefill``'s, so on a reuse walk some of them name a node
    the walk does not contain: the ``state`` walk is the state encoder alone, so
    ``pixel_values`` has nowhere to go.

    A reuse request carries those inputs *always*, which is what makes the drop
    the right reaction rather than a refusal. ``process_prompt`` returns
    everything the processor produced and ``preprocessing.py:201-203`` adds
    ``state``, ``embodiment_id`` and ``action_mask`` unconditionally; the data
    worker merges the lot (``data_worker.py:554``) and the conductor hands the
    whole dict to this hook (``conductor.py:1018-1023``). So a ``state`` request
    always arrives with frames and a ``backbone`` request always with a robot
    state -- raising on one would have refused *every* real reuse request, and
    made ``walk`` unreachable through the server.

    Two things are pinned, and the second is the one with teeth. The input must
    not be **routed**: an edge to a node outside the walk is delivered to the
    partition, matched against nodes that are not there and never consumed, so
    the walk would still emit its features and the only trace would be a tensor
    shipped for nothing. And it must still be **released**: ``data_worker.py:583``
    persists every initial signal and only the model can say it is done with one,
    so a dropped input that is also dropped from ``unpersist_tensors`` is held for
    the request's life -- and a reuse walk has no later hop to release it on.

    The positive control at the end is what keeps this from being a test that
    asserts nothing happened: the same input on the full path *is* routed to that
    same node, so the drop is this walk's doing and not a name nothing recognises.
    """
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
    # released: everything the request arrived with, because a reuse walk parks
    # nothing -- the dropped input included
    assert all(not edge.persist for edge in args.inputs)
    assert args.unpersist_tensors == [info for infos in signals.values() for info in infos]
    assert signals[name][0] in args.unpersist_tensors

    # the positive control
    full = model.get_initial_forward_pass_args("default", ["text", "image"], ["action"], signals)
    assert (name, node) in {(edge.name, edge.next_node) for edge in full.inputs}


def test_the_embodiment_id_is_parked_only_where_a_loop_reads_it_back():
    """``embodiment_id`` is the one request input that outlives its own forward
    pass, and only on the full path: the denoise step reads it on every one of
    the four iterations (``submodules.py:550``), so ``prefill`` must not release
    it. On a reuse walk there is no loop -- and, more to the point, no second
    forward pass and nothing that unpersists anything (the transition returns
    ``unpersist_tensors=[]``), so a parked copy is never released and the
    request leaks it for as long as it lives.

    Both directions, because the flag is one expression that reads the schedule
    (``gr00t_n17_model.py:438``) and either half can be wrong on its own: parking
    nowhere breaks the loop's third iteration, parking everywhere leaks on every
    reuse request. ``test_a_persisted_request_input_keeps_its_destination`` pins
    what the flag *means* on the full path; this pins which path gets it.

    The release list is the flag's own complement -- the parked uuids come off
    these edges and everything else the request arrived with is released -- so
    the leak is asserted as the tensor's absence from that list rather than
    inferred from the flag.
    """
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
    # and there is no later hop to release it on: the reuse walk's one transition
    # is the one that ends the request
    done = model.get_partition_forward_pass_args("default", reuse.full_metadata, {})
    assert done.request_done and done.unpersist_tensors == []


def test_the_reuse_walk_node_sets_match_the_graphs():
    """``REUSE_WALK_NODES`` is a hand-written copy of graph structure, kept
    because ``get_graph_walk_graphs()`` builds five nodes' worth of objects and
    ``get_initial_forward_pass_args`` runs once per request. A copy drifts, and
    this one drifts quietly in both directions: a node listed but no longer in
    the walk lets an input through to a node that cannot consume it, and a node
    in the walk but not listed rejects an input the walk needs -- a valid request
    refused at admission.

    The pairing with ``_walk_schedule`` is the other half, and it is the one that
    cannot be read off either declaration: a walk the schedule accepts but this
    constant does not name gets ``reachable=None``
    (``gr00t_n17_model.py:421``), which means "route everything" -- the full
    path's rule applied to a walk with two of its nodes missing. So selectable
    and listed are asserted to be the same two walks.

    Named for the comment that points here (``gr00t_n17_model.py:81-84``).
    """
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
# The five resource specs (PR-1 §3.4)
# ----------------------------------------------------------------------------


def _specs(model: Gr00tN17Model) -> dict[str, RaggedAttentionConfig]:
    return {spec.resource_key: spec for spec in model.get_node_resources()}


def test_the_five_specs_land_on_the_right_nodes():
    """§3.4's five rows, no more and no fewer. A resource the engine does not
    build is a node whose attention falls back to SDPA in silence (the DT's
    ``attends()`` leaves out an unbound span on purpose), and a spec naming the
    wrong node binds a wrapper the node never calls."""
    specs = _specs(_model())

    assert set(specs) == {VISION_ATTN, LLM_ATTN, BRIDGE_ATTN, DT_ATTN, DT_XATTN}
    assert {key: spec.nodes for key, spec in specs.items()} == {
        VISION_ATTN: {VISION_ENCODER},
        LLM_ATTN: {LLM},
        BRIDGE_ATTN: {VL_BRIDGE},
        DT_ATTN: {DENOISE_STEP},
        DT_XATTN: {DENOISE_STEP},
    }
    # no KV cache anywhere: every attention is one-shot, so nothing depends on a
    # cache's geometry
    assert all(spec.depends_on() == set() for spec in specs.values())


def test_the_dt_holds_two_specs_because_a_resource_is_one_kind_of_attention():
    """The DT's cross blocks and self blocks are the same 32 x 48 geometry, so the
    temptation is one resource for both. They cannot share: a cross plan is
    ``(q_label, kv_label)`` pairs and a self plan is one span per request, and the
    manager classes differ (``RaggedCrossAttnManager`` against
    ``RaggedAttnManager``). Collapsing them builds one wrapper and leaves the
    other kind of step unplanned.

    Pinned as exact types rather than ``isinstance``, because
    ``RaggedCrossAttentionSpec`` and ``RaggedAttentionSpec`` share a base and an
    ``isinstance`` check would pass for either against it.
    """
    specs = _specs(_model())
    dt = [spec for spec in specs.values() if spec.nodes == {DENOISE_STEP}]

    assert len(dt) == 2
    assert type(specs[DT_ATTN]) is RaggedAttentionSpec
    assert type(specs[DT_XATTN]) is RaggedCrossAttentionSpec
    assert specs[DT_ATTN].resource_class is not specs[DT_XATTN].resource_class
    assert {key for key, spec in specs.items() if type(spec) is RaggedCrossAttentionSpec} == {DT_XATTN}
    # the same geometry, which is what makes the two kinds the only difference
    for field in ("num_qo_heads", "num_kv_heads", "head_dim"):
        assert getattr(specs[DT_ATTN].config, field) == getattr(specs[DT_XATTN].config, field)


@pytest.mark.parametrize("scale", [1, 2], ids=["the tiny config", "the same config doubled"])
def test_every_head_geometry_is_read_off_the_config(scale):
    """Run twice at two different sets of head counts: a hardcoded number can
    satisfy one but not both. That is the whole point of the parametrisation --
    the real checkpoint's 32 x 48 DT and 16 queries over 8 KV heads x 128 LLM are
    easy numbers to inline, and inlining either would load every weight and run.

    The head *dims* are derived three different ways on purpose, and each is the
    only correct one for its stack: the ViT's from ``hidden_size // num_heads``
    (its config carries no head dim), the LLM's from ``head_dim`` (which is not
    ``hidden_size // num_attention_heads`` in a Qwen3), and the two action-head
    stacks' from ``attention_head_dim``.
    """
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
    # the LLM is the one stack whose head dim is not hidden_size // heads, so a
    # read of the wrong field would be invisible at a square geometry
    assert config.text.head_dim * config.text.num_attention_heads != config.text.hidden_size
    # and the five are genuinely five different geometries at this config
    assert len({geometry(key) for key in specs}) == 4  # the DT's two are equal by design


def test_the_vision_bound_is_the_only_one_that_is_not_one_per_request():
    """A ``max_segments_per_request`` sizes a CUDA-graph bucket, and the count it
    bounds is **per span label**, not per step: a wrapper is built and sized for
    each label a step plans (``flashinfer.py:81-100``, keyed by
    ``CGSlotKey(bucket, slot, label)``), and the cross resource plans one wrapper
    per *pair*, labelled ``cross_label(q_label, kv_label)``
    (``flashinfer.py:254-257``).

    So only the ViT's bound is not 1. It declares one segment per *frame*
    (``submodules.py:112``), all under one label, so a three-view embodiment
    really does present three segments of that label -- see ``max_views``.

    The DT's cross resource is the one to get wrong, and the wrong answer is
    ``len(DT_XATTN_PAIRS)``: the step declares three segments per request
    (``sa + text + image``, ``submodules.py:650``) across two pairs, which reads
    like a bound above 1. But each pair sees exactly one ``sa`` segment and one
    key segment per request, under its own label, so the per-label count is 1.
    Over-sizing here is merely wasteful; the reason to pin it is that the number
    looks like it should be 2 or 3.
    """
    specs = _specs(_model())

    assert {key: spec.config.max_segments_per_request for key, spec in specs.items()} == {
        VISION_ATTN: _MAX_VIEWS, LLM_ATTN: 1, BRIDGE_ATTN: 1, DT_ATTN: 1, DT_XATTN: 1,
    }
    # the count that makes 1 look wrong; the three-segments-per-request half is
    # `test_gr00t_n17_denoise_step.py`'s, which has the submodule to ask
    assert len(DT_XATTN_PAIRS) == 2
    # the bucket sizes are a deployment's trade, so nothing here pins a token
    # ceiling: §3.5 leaves `max_tokens_per_request` for the capture cycle
    assert all(spec.config.max_tokens_per_request is None for spec in specs.values())


def test_every_spec_pins_its_activation_dtype():
    """A cacheless ragged resource with ``dtype=None`` takes "the engine's KV
    dtype (which is actually a misnamed autocast dtype)"
    (``ragged/config.py:68-73``). GR00T has no KV cache anywhere, so all five
    would inherit that default -- and the T3 parity bar is measured in bf16, so
    the dtype the bar is stated in would track something unrelated to this model.
    z_image, flux2_klein and ltx2_5 all pin theirs for the same reason
    (``z_image_model.py:128-129``).
    """
    specs = _specs(_model())

    assert {spec.config.dtype for spec in specs.values()} == {torch.bfloat16}
    assert len(specs) == 5


def test_causality_is_declared_on_the_step_and_not_on_the_resource():
    """GR00T is the first model on ``main`` to want a *causal* ragged
    self-attention (§3.4 row 2; every other ragged user is non-causal), so the
    obvious mistake is to reach for a ``causal`` knob on the resource. There is
    none: a ragged resource is a geometry and a kernel, and whether a given
    forward masks is the step's, which is why the same resource serves the DT's
    non-causal self-attention and the LLM's causal one.
    """
    assert "causal" not in {field.name for field in dataclasses.fields(RaggedAttentionConfig)}
    assert "causal" in {field.name for field in dataclasses.fields(AttentionStep)}
    for spec in _model().get_node_resources():
        assert not hasattr(spec.config, "causal")

    # where it is declared: the LLM node's step. `declare_step` reads only its
    # inputs, so this costs no module.
    node = object.__new__(Gr00tN17LLMSubmodule)
    step = node.declare_step(_PREFILL, ["r0"], [NodeInputs(input_seq_len=7)])
    assert set(step.steps) == {LLM_ATTN}
    assert isinstance(step.steps[LLM_ATTN], AttentionStep)
    assert step.steps[LLM_ATTN].causal is True
    assert [segment.span for segment in step.segments] == [7]


# ----------------------------------------------------------------------------
# `max_views`, which is a property of the checkpoint
# ----------------------------------------------------------------------------


_VIEW_CASES = {
    "three embodiments, the widest in the middle": (_VIDEO_KEYS, 3),
    "one embodiment, one view": ({"gr1": ["video.ego_view"]}, 1),
    "five views": ({"rig": list("abcde"), "gr1": ["x"]}, 5),
}


@pytest.mark.parametrize("video_keys, expected", list(_VIEW_CASES.values()), ids=list(_VIEW_CASES))
def test_max_views_is_the_widest_embodiments_video_keys(video_keys, expected):
    """Not a constant, and not ``len`` of one embodiment's keys. It sizes the ViT's
    segment bound, so too small plans fewer spans than the forward attends over --
    and a real ragged kernel reads the *declared* length, which makes an
    under-count a wrong answer rather than a crash.

    The processor already reads the per-tag video keys off the checkpoint
    (``preprocessing.py:50``), so there is nothing here to configure: a second
    copy of the number is a number that can disagree with the checkpoint.
    """
    model = _model(video_keys=video_keys)

    assert model.max_views == expected
    assert _specs(model)[VISION_ATTN].config.max_segments_per_request == expected


# ----------------------------------------------------------------------------
# The request's two hops
# ----------------------------------------------------------------------------


def test_a_request_starts_in_prefill_with_each_input_at_the_node_that_takes_it():
    """The first hop. Every routed name has to be a name the destination node
    declares, or it arrives at a node that will never consume it and the walk
    stalls; and ``is_prefill`` has to be set, since it is what tells the engine
    this pass is the request's conditioning rather than a loop iteration.

    Only the signals the request actually carries are routed, so a partition that
    is handed fewer inputs gets fewer edges rather than empty ones.

    The routes have to *cover* the walk as well as be valid: every external input
    ``prefill`` declares needs a route, or the node that wants it never becomes
    ready. That is the half a per-edge check cannot see, and it is asserted here
    against the walk's own signature rather than against ``_ROUTES``.
    """
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
    # coverage: nothing the walk needs from outside is left unrouted
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
    """``denoise_step`` needs the two span lengths *before* their tensors arrive --
    the key picks the captured graph, which is chosen while the forward is being
    planned -- so they ride on ``step_metadata`` under the names the node reads
    back (``submodules.py:479-489``).

    Derived from the persisted tensors' own ``dims`` rather than from a count kept
    beside them: a count derived from the thing it describes cannot disagree with it,
    and disagreeing is most of what the node's own check guards against. The node raises
    when the two disagree (``submodules.py:544``), but only inside a batch, and a
    uniform miscount is silent there -- so the derivation is the real defence.

    Three span pairs, so a constant cannot be right three times.
    """
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
    """``prefill`` runs once and ``action_gen`` once -- the four denoising steps are
    the ``Loop``'s iterations, not four conductor hops, so the second call is the
    end of the request and not the second step.

    ``is_prefill`` has to go false on the way out of ``prefill``: it rides on the
    same mutable metadata object the conductor keeps for the request, so leaving
    it set would tell the engine every loop iteration is a conditioning pass.
    """
    model = _model()
    names = sorted(_PERSISTED | {"embodiment_id"})
    persist = {name: [_info([3, 63], uuid=i)] for i, name in enumerate(names)}
    metadata = _metadata(model)

    first = model.get_partition_forward_pass_args("default", metadata, persist)
    assert first.full_metadata.graph_walk == _ACTION_GEN
    assert first.full_metadata.is_prefill is False
    assert not first.request_done
    # the carried edge is routed last and arrives empty: the scaffold draws
    # iteration 0's sample itself
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


# Each node's checkpoint root. The three nodes that wrap a component hold it as
# `.model`, and the loader names their tensors relative to it; the vision tower is
# the reference's `visual.` itself, beside GR00T's own DeepStack mergers.
_ROOTS = {
    VISION_ENCODER: "backbone.model.model.visual.",
    LLM: "backbone.model.model.language_model.",
    VL_BRIDGE: "action_head.", STATE_ENCODER: "action_head.", DENOISE_STEP: "action_head.",
}


def _checkpoint_parts(config: Gr00tN17Config, node: str, name: str, shape) -> list[tuple[str, tuple]]:
    """The checkpoint tensors one served parameter is made of, in row order: one
    tensor, except the LLM's fused q/k/v and gate/up, which the checkpoint keeps apart."""
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
    """``_model()`` over a synthetic checkpoint at ``path``: every tensor the five
    nodes read, under the checkpoint's own names, each a different random draw, so
    a parameter left at its init or fed its neighbour's tensor cannot match."""
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
    """One submodule per node, cached. The identity matters rather than just the
    equality: the model loads its weights into the object it hands out, the engine
    binds resources onto it, and a compiled region guards on the identity of the
    callables it holds -- so a second build would be a second load, a module with
    no resources and a retrace.
    """
    model, _ = _served(tmp_path)
    built = {name: model.get_submodule(name) for name in _SUBMODULES}

    for name, cls in _SUBMODULES.items():
        assert type(built[name]) is cls
        assert model.get_submodule(name) is built[name]

    assert set(model._submodules) == set(_SUBMODULES) == set(model.nodes)
    assert all(model._submodules[name] is built[name] for name in _SUBMODULES)
    assert len({id(sub) for sub in built.values()}) == 5


def test_every_submodule_comes_back_in_eval_mode(tmp_path):
    """Our modules hold ``Dropout(p=0.0)`` rather than relying on the flag, so
    nothing changes numerically today -- which is exactly why this has to be
    pinned rather than observed. Every parity number C05-C08 measured is an
    eval-mode claim, and a dropout or a norm added later would silently make the
    served model a different function from the measured one. ``qwen3_5``,
    ``whisper`` and ``orpheus`` all set it at load for the same reason.

    ``nn.Module`` starts in training mode, so this is not a tautology: the flag
    has to be set by somebody.
    """
    model, _ = _served(tmp_path)
    for name in _SUBMODULES:
        submodule = model.get_submodule(name)
        assert submodule.training is False
        assert all(not child.training for child in submodule.modules())


def test_an_unknown_node_is_none_and_is_not_cached(tmp_path):
    """``None`` is the engine's "this worker does not host that node", so a typo in
    a node_group has to come back as a miss rather than as something plausible.
    And it must not be cached: ``get_submodule``'s cache is keyed by name, and a
    cached ``None`` would be indistinguishable from a built submodule on the
    second call were the miss ever to become a hit.
    """
    model, _ = _served(tmp_path)
    assert model.get_submodule("vlm") is None
    assert model.get_submodule(VISION_ENCODER.upper()) is None
    assert model.get_submodule("") is None
    assert model._submodules == {}


def test_every_served_parameter_is_the_checkpoint_tensor_it_maps_from(tmp_path):
    """The served path loads its own weights. Until C10 nothing but the tests called
    the loader, so a live server ran all five nodes on their random init -- and a DT
    on random weights still returns plausible-looking actions, no error anywhere.

    Values, not shapes: random init has every shape right. The LLM's q/k/v and
    gate/up are checked as the checkpoint's separate tensors stacked in row order.
    """
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
    """Strict, per node: the parameter that would have served its random init
    raises instead, and the message names the node, so a truncated or mismatched
    checkpoint fails at startup rather than serving plausible actions."""
    model, _ = _served(tmp_path, drop=drop)
    with pytest.raises(ValueError, match=f"^{node}: 1 parameter"):
        model.get_submodule(node)
    assert node not in model._submodules


@pytest.mark.parametrize("part", ["layers.1.self_attn.k_proj.weight", "layers.1.mlp.up_proj.weight"])
def test_one_part_missing_from_a_fused_llm_parameter_fails_the_llm_by_name(tmp_path, part):
    """A fused parameter counts as loaded once any of its parts lands, so a
    checkpoint short of one of q/k/v or gate/up served those rows at random init
    with no error. The part is named, since the fused parameter's name is not one
    the checkpoint has."""
    model, _ = _served(tmp_path, drop=_ROOTS[LLM] + part)
    with pytest.raises(ValueError, match=rf"^LLM: 1 parameter.*{re.escape(part)}"):
        model.get_submodule(LLM)
    assert LLM not in model._submodules


def test_the_served_vision_node_holds_the_vision_config_and_prepares_a_grid(tmp_path):
    """The node reads ``config.vision`` (``spatial_merge_size``,
    ``num_grid_per_side``). Handed the whole ``Gr00tN17Config`` it built and loaded
    fine, and the first camera frame raised ``AttributeError`` in
    ``prepare_inputs``: a 500 on every live request."""
    model, _ = _served(tmp_path)
    node = model.get_submodule(VISION_ENCODER)
    assert node.config is model.config.vision

    vision = model.config.vision
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    fwd = CurrentForwardPassInfo(request_id="r0", graph_walk=_PREFILL, fwd_index=0, random_seed=0, max_tokens=0)
    grid = {"image_grid_thw": [torch.tensor([[1, 4, 6]])], "pixel_values": [torch.zeros(24, patch)]}
    inputs = node.prepare_inputs(_PREFILL, fwd, grid)
    assert inputs.input_seq_len == 24


# ----------------------------------------------------------------------------
# postprocess
# ----------------------------------------------------------------------------

# One embodiment, two groups, and only what the decode reads. `arm` is relative
# with [-1, 1] bounds, so its decoded value is the chunk's own number plus the
# request's state; `grip` is absolute with q01 == q99 == 5, so it is 5 whatever
# arrives. That is the pair that makes the concatenation *order* visible in the
# bytes: one column of the output is constant and the others are not.
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
    """``_model()`` with a processor that can actually decode. The statistics are
    synthetic, so this still reads nothing off disk; the point is that
    ``postprocess`` is tested against the real ``unapply`` rather than a stub that
    would agree with whatever order the model concatenated in."""
    model = _model()
    model.processor = Gr00tN17Preprocessor(_DECODE_KWARGS, _DECODE_STATS, {"bot": 0}, 8, 6)
    return model


class _BytesProcessor:
    """A backbone processor whose ``input_ids`` are the instruction's own bytes, so
    two requests tokenize alike only if the model handed it the same text."""

    tokenizer = type("Tokenizer", (), {"padding_side": "right"})()

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        return "<user>" + conversation[0]["content"][-1]["text"]

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.tensor([list(t.encode()) for t in text]),
                "pixel_values": torch.stack(images).float()}


def test_no_instruction_is_served_as_the_empty_instruction():
    """FastAPI turns an empty ``text`` form field into ``None``, and 8 of the 20
    recorded DROID requests carry an empty instruction. Refusing ``None`` answered
    real robot data with a 400; it is the empty instruction, tokenized as such.
    A real instruction is the control: the stub does see the text."""
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


def test_postprocess_names_every_joint_group_the_way_the_reference_does():
    """[[D09]]: the payload is one entry per joint group, keyed as NVIDIA keys it.

    This is the reference's own output shape, not a wire convention of ours:
    ``Gr00tPolicy`` returns ``{joint_group: array}`` cast to float32
    (``gr00t_policy.py:425-432``, via ``decode_action``, whose keys are bare --
    the ``action.`` prefix comes only from its sim-compatibility wrapper). So a
    client written against GR00T indexes this response by the same names.

    The groups have **different widths** (9, 1, 7 on DROID), which is why
    flattening them was wrong: nothing in a flat payload says which float drives
    which actuator, so the caller would have to rebuild the layout from the
    embodiment tag, and a wrong order returns the right number of plausible
    floats and moves the wrong joint.

    float32, not the decode's float64 -- the same cast the reference makes
    (``gr00t_policy.py:430``). Each group carries its own dtype and shape, so a
    client reconstructs it without knowing the embodiment at all.
    """
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
    # the same numbers the flat form carried, now attributable to a joint group
    flat = np.concatenate([rebuilt["arm"], rebuilt["grip"]], axis=-1)
    assert flat.reshape(2, 3).tolist() == _DECODED
    # `grip` is the constant group, and it is named rather than positional now
    assert rebuilt["grip"].reshape(-1).tolist() == [5.0, 5.0]


def test_postprocess_adds_the_batch_axis_the_engine_hands_back_without():
    """The worker hands one request's output back as ``(T, D)`` -- the batch axis
    is gone by the time a single request's tensor is read out -- while ``unapply``
    takes ``(B, T, D)`` and refuses anything else, because the batch axis is what
    a relative group's per-request reference frame is matched against. If this
    model did not add the axis, every single-request decode would raise.
    """
    model = _decoding_model()
    kwargs = {"embodiment_tag": "bot", "state": _RAW_STATE}
    assert model.postprocess(_CHUNK[0], "action", kwargs) == model.postprocess(_CHUNK, "action", kwargs)


def test_postprocess_finds_the_tag_and_the_state_in_the_kwargs_the_data_worker_kept():
    """Why ``embodiment_tag`` and ``state`` are in ``PREPROCESS_ONLY_KWARGS``.
    GR00T decodes server-side ([[D06]]), so the inverse needs the checkpoint's
    statistics *and* the request's own raw state at the very end of the request --
    long after ``process_prompt`` turned them into tensors. The conductor's copy of
    the kwargs is stripped of the preprocess-only ones; the data worker's is not
    (``data_worker.py:586``), and that is the copy handed here.

    The state is checked by moving it: the decode is relative, so a shifted
    reference state has to shift every action by the same amount. A ``postprocess``
    that dropped the state on the floor would pass the order test above and return
    the relative chunk as if it were absolute.
    """
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
    """The tag selects the statistics and the joint groups, so there is no default
    that is safe: this checkpoint has eight embodiments of different widths and
    different horizons, and decoding a DROID chunk against a G1's bounds would
    return a full-looking chunk of wrong numbers. ``None`` is in the list because
    ``request_kwargs`` is a dict the client fills, so a missing key and an explicit
    null arrive the same way.
    """
    with pytest.raises(ValueError, match="embodiment_tag"):
        _decoding_model().postprocess(_CHUNK, "action", kwargs)


@pytest.mark.parametrize("modality", ["text", "image", "video", ""])
def test_postprocess_refuses_a_modality_this_model_does_not_emit(modality):
    """``output_modalities`` is the request's, and a GR00T request that asked for
    text would reach here with a 132-wide action chunk: decoding it is meaningless
    and *detokenizing* it would be worse. The check is first, before the tag is
    even read, because the wrong modality is a wrong request and not a missing
    argument.
    """
    with pytest.raises(ValueError, match="emits actions"):
        _decoding_model().postprocess(_CHUNK, modality, {"embodiment_tag": "bot", "state": _RAW_STATE})


# ----------------------------------------------------------------------------
# Against the checkpoint
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_specs_geometries_are_the_checkpoints():
    """Where §3.4's numbers come from. The config tests above say the geometries
    are *read*; this one says what they are read as -- 16 x 64 for the ViT,
    16 queries over 8 KV heads x 128 for the LLM, 32 x 64 for the bridge and
    32 x 48 for both of the DT's.

    Config only: two json files, no tensor and no module.
    """
    model = _model(Gr00tN17Config.from_hf(_GR00T, _BACKBONE))
    specs = _specs(model)

    assert {
        key: (spec.config.num_qo_heads, spec.config.num_kv_heads, spec.config.head_dim)
        for key, spec in specs.items()
    } == _REAL
    assert model.config.dit.num_layers == 32  # the 1.12B the walks never build
    assert model.get_graph_walk_graphs()[_ACTION_GEN].max_iters == 4


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_five_nodes_build_at_the_checkpoints_widths_on_meta():
    """The real model, allocated nowhere: ``meta`` means the 1.12B-parameter DT
    costs its metadata and no bytes. What this catches that the tiny config cannot
    is a dimension that only disagrees at the checkpoint's own numbers -- a stack
    whose width is read from the wrong field happens to line up at a square tiny
    config and raises here.

    Built with the loader stubbed: loading reads the whole checkpoint into tensors
    ``meta`` discards. What loading does is checked on the tiny checkpoint above.
    """
    model = _model(Gr00tN17Config.from_hf(_GR00T, _BACKBONE))
    model._repo_dir = _GR00T
    with torch.device("meta"):
        built = {name: _unloaded(model, name) for name in _SUBMODULES}
    assert {name: type(sub) for name, sub in built.items()} == _SUBMODULES

    dt = built[DENOISE_STEP]
    assert len(dt.model.transformer_blocks) == 32
    assert sum(p.numel() for p in dt.parameters()) > 1_000_000_000
    assert all(p.device.type == "meta" for p in dt.parameters())


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_model_constructs_from_its_two_checkpoints():
    """``__init__`` is the only thing in this file the tests above route around:
    they build the model with ``object.__new__`` because the walks and the specs
    are functions of the config alone. That makes ``__init__`` the one piece with
    no coverage, so it gets its own test -- and it needed one, because the call to
    the processor's factory was wrong in a way no config test could see.

    Two repos, because GR00T's weights and its backbone's config live apart: the
    action head is ``nvidia/GR00T-N1.7-3B`` and the Qwen3-VL dims come from
    ``nvidia/Cosmos-Reason2-2B`` (§3.7 row 1). No weights are read -- the config
    is two json files and the processor is a tokenizer and an image processor.

    The two numbers the factory takes beside the paths are asserted to be the
    *config's*, not defaults: ``max_action_horizon`` is what the processor refuses
    a too-long embodiment against (``preprocessing.py:69-71``), so a wrong value
    there either rejects a valid robot or accepts one the model cannot serve.
    """
    model = Gr00tN17Model(model_path_hf=str(_GR00T), backbone_path_hf=_BACKBONE)

    assert model.config == Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    assert isinstance(model.processor, Gr00tN17Preprocessor)
    assert model.processor.max_state_dim == model.config.max_state_dim
    assert model.processor.max_action_horizon == model.config.action_horizon
    # 3: the widest embodiment in this checkpoint, off the processor's video keys
    assert model.max_views == 3
    assert _specs(model)[VISION_ATTN].config.max_segments_per_request == 3
    # a frozenset, which is what `base.py:756` declares; GR00T is the only model
    # on `main` that overrides it at all, so there is no peer to copy the shape
    # from and it was a list until this assertion
    # no `video`: camera frames travel the media path, not a control-wire kwarg
    # ([[D10]]), so the only bulk data left in the kwargs is the state vector
    assert model.request_kwargs() == frozenset({"embodiment_tag", "state", "walk"})
    # the two that `process_prompt` turns into tensors are kept off the control
    # wire; `walk` is not, because the conductor chooses the schedule with it
    assert set(Gr00tN17Model.PREPROCESS_ONLY_KWARGS) < model.request_kwargs()
    assert "walk" not in Gr00tN17Model.PREPROCESS_ONLY_KWARGS
    # one frame per view per timestep, refused before any file is decoded
    assert model.max_input_files() == {"image": 3 * 2}
