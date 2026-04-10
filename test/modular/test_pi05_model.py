"""Tests for the Pi0.5 model class.

These tests focus on the structural pieces of the Pi0.5 implementation that
do not require real weights: graph walks, node-to-engine mapping, forward pass
transitions, and worker graph division. Real-weight integration is exercised
separately via end-to-end smoke tests.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import torch

from mminf.conductor.request_info import CurrentForwardConductorMetadata
from mminf.engine.base import EngineType
from mminf.graph.base import GraphNode, Loop, Sequential
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.pi05.components.flow_matching import (
    discretize_state,
    euler_step,
    sincos_timestep_embedding,
)
from mminf.model.pi05.config import Pi05Config
from mminf.model.pi05.pi05_model import Pi05Model

CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "pi05.yaml"
)


def _make_model() -> Pi05Model:
    """Construct a Pi0.5 model without downloading weights or tokenizer."""
    model = object.__new__(Pi05Model)
    model.model_path_hf = "test/pi05"
    model.cache_dir = None
    model.skip_weight_loading = True
    model.config = Pi05Config()
    model.tokenizer = None
    model._repo_dir = None
    model._submodule_cache = {}
    model.embed_tokens = None
    model.paligemma = None
    model.action_expert = None
    model.action_in_proj = None
    model.action_out_proj = None
    model.adaln_mlp = None
    model.siglip = None
    return model


# ----------------------------------------------------------------------
# Graph structure
# ----------------------------------------------------------------------


def test_pi05_graph_walks_have_expected_keys():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks.keys()) == {Pi05Model.PREFILL_WALK, Pi05Model.ACTION_GEN_WALK}


def test_pi05_prefill_is_sequential_vit_then_llm():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    prefill = walks[Pi05Model.PREFILL_WALK]
    assert isinstance(prefill, Sequential)
    assert len(prefill.sections) == 2
    first, second = prefill.sections
    assert isinstance(first, GraphNode) and first.name == "vit_encoder"
    assert isinstance(second, GraphNode) and second.name == "LLM"
    # vit_encoder must emit img_emb to LLM.
    assert any(
        edge.next_node == "LLM" and edge.name == "img_emb"
        for edge in first.outputs
    )
    # LLM consumes img_emb + text_inputs + state_inputs.
    assert set(second.input_ids) == {"img_emb", "text_inputs", "state_inputs"}


def test_pi05_action_gen_is_loop_with_action_output_emission():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    action_gen = walks[Pi05Model.ACTION_GEN_WALK]
    assert isinstance(action_gen, Loop)
    assert action_gen.n_iters == model.config.num_flow_steps == 10
    # The terminal output emits to the client with the action modality.
    assert len(action_gen.outputs) == 1
    terminal = action_gen.outputs[0]
    assert terminal.next_node == EMIT_TO_CLIENT
    assert terminal.name == "action_output"
    assert terminal.output_modality == "action"
    # Loop body is a single LLM node with two loop-back edges.
    body = action_gen.section
    assert isinstance(body, GraphNode) and body.name == "LLM"
    assert {e.name for e in body.outputs} == {"noisy_actions", "timestep_index"}
    assert all(e.next_node == "LLM" for e in body.outputs)


def test_pi05_node_engine_types():
    model = _make_model()
    types = model.get_node_engine_types()
    assert types == {
        "vit_encoder": EngineType.ENC_DEC,
        "LLM": EngineType.AR,
    }


def test_pi05_kv_cache_config_matches_pi05_config():
    model = _make_model()
    kv = model.get_kv_cache_config()
    assert kv.num_layers == model.config.num_layers
    assert kv.num_kv_heads == model.config.num_kv_heads
    assert kv.head_dim == model.config.head_dim
    assert kv.num_qo_heads == model.config.num_qo_heads


# ----------------------------------------------------------------------
# Worker graph division using the YAML config
# ----------------------------------------------------------------------


def test_pi05_worker_graphs_from_yaml():
    model = _make_model()
    worker_graphs = model.get_worker_graphs(CONFIG_PATH)
    # 2 graph walks * 2 worker graphs apiece... but the prefill walk has both
    # nodes on rank 0 and they belong to different node_groups, so they remain
    # 2 separate worker graphs. action_gen has only the LLM node.
    walks_seen = {tuple(sorted(wg.graph_walks)) for wg in worker_graphs}
    assert (Pi05Model.PREFILL_WALK,) in walks_seen
    assert (Pi05Model.ACTION_GEN_WALK,) in walks_seen


# ----------------------------------------------------------------------
# Forward pass transitions
# ----------------------------------------------------------------------


def test_pi05_initial_forward_pass_args_starts_in_prefill():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["image", "text"],
        output_modalities=["action"],
        input_signals={
            "image_inputs": [],
            "text_inputs": [],
            "state_inputs": [],
        },
    )
    assert args.full_metadata.graph_walk == Pi05Model.PREFILL_WALK
    assert args.full_metadata.is_prefill is True
    edge_targets = {(e.next_node, e.name) for e in args.inputs}
    assert ("vit_encoder", "image_inputs") in edge_targets
    assert ("LLM", "text_inputs") in edge_targets
    assert ("LLM", "state_inputs") in edge_targets


def test_pi05_prefill_transitions_to_action_gen():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["image", "text"],
        output_modalities=["action"],
        graph_walk=Pi05Model.PREFILL_WALK,
        is_prefill=True,
    )
    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
        new_tokens={},
    )
    assert result.full_metadata.graph_walk == Pi05Model.ACTION_GEN_WALK
    assert result.full_metadata.is_prefill is False
    assert result.request_done is False
    edge_names = {e.name for e in result.inputs}
    assert edge_names == {"noisy_actions", "timestep_index"}


def test_pi05_action_gen_marks_request_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["image", "text"],
        output_modalities=["action"],
        graph_walk=Pi05Model.ACTION_GEN_WALK,
        is_prefill=False,
    )
    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
        new_tokens={},
    )
    assert result.request_done is True


# ----------------------------------------------------------------------
# Postprocess
# ----------------------------------------------------------------------


def test_pi05_postprocess_action_returns_float32_bytes():
    model = _make_model()
    actions = torch.zeros(model.config.action_horizon, model.config.action_dim)
    result = model.postprocess(actions, modality="action")
    expected = (
        model.config.action_horizon * model.config.action_dim * 4
    )  # 4 bytes per float32
    assert isinstance(result, bytes)
    assert len(result) == expected


# ----------------------------------------------------------------------
# Flow matching helpers
# ----------------------------------------------------------------------


def test_sincos_timestep_embedding_shape_and_range():
    t = torch.tensor(0.5)
    emb = sincos_timestep_embedding(t, dim=16)
    assert emb.shape == (1, 16)
    assert torch.all(emb.abs() <= 1.0 + 1e-6)


def test_euler_step_shapes():
    x = torch.zeros(50, 32)
    v = torch.ones(50, 32)
    out = euler_step(x, v, dt=-0.1)
    assert out.shape == (50, 32)
    assert torch.allclose(out, torch.full_like(out, -0.1))


def test_discretize_state_round_trip_within_bin():
    state = torch.linspace(-1.0, 1.0, steps=8)
    indices = discretize_state(state, num_bins=256)
    assert indices.dtype == torch.long
    assert indices.min().item() >= 0
    assert indices.max().item() <= 255
    # Endpoints should map to the extreme bins.
    assert indices[0].item() == 0
    assert indices[-1].item() == 255
