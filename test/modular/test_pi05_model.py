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
    # LLM consumes img_emb + text_inputs (state is encoded in text_inputs
    # as a decimal-string suffix per Pi0.5's prompt format).
    assert set(second.input_ids) == {"img_emb", "text_inputs"}


def test_pi05_action_gen_is_loop_with_action_output_emission():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    action_gen = walks[Pi05Model.ACTION_GEN_WALK]
    assert isinstance(action_gen, Loop)
    assert action_gen.max_iters== model.config.num_flow_steps == 10
    # The terminal output emits to the client with the action modality.
    # Its ``name`` must match one of the section's loop-back edge names so
    # that Loop._replace_outputs_for_final_iter can swap it in on the final
    # iteration (see the comment in Pi05Model.get_graph_walk_graphs).
    assert len(action_gen.outputs) == 1
    terminal = action_gen.outputs[0]
    assert terminal.next_node == EMIT_TO_CLIENT
    assert terminal.name == "noisy_actions"
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
        "vit_encoder": EngineType.STATELESS,
        "LLM": EngineType.KV_CACHE,
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
        },
    )
    assert args.full_metadata.graph_walk == Pi05Model.PREFILL_WALK
    assert args.full_metadata.is_prefill is True
    edge_targets = {(e.next_node, e.name) for e in args.inputs}
    assert ("vit_encoder", "image_inputs") in edge_targets
    assert ("LLM", "text_inputs") in edge_targets
    # Pi0.5 has no separate state_inputs edge — state is part of text_inputs.
    assert not any(e.name == "state_inputs" for e in args.inputs)


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


# ----------------------------------------------------------------------
# Prompt formatting (matches lerobot's Pi05PrepareStateTokenizerProcessorStep)
# ----------------------------------------------------------------------


class _StubTokenizer:
    """Captures the prompt string passed to encode_prompt for assertion."""

    def __init__(self):
        self.last_prompt: str | None = None

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        self.last_prompt = prompt
        # Return a deterministic tensor of length 1; the test only inspects
        # last_prompt.
        return torch.tensor([0], dtype=torch.long)


def test_pi05_process_prompt_formats_state_into_text():
    """Pi05Model.process_prompt should produce the openpi-style template
    ``"Task: <text>, State: <bin0> <bin1> ... <bin31>;\\nAction: "`` and
    pass it as a SINGLE call to the tokenizer's encode_prompt. This matches
    ``lerobot/policies/pi05/processor_pi05.py::Pi05PrepareStateTokenizerProcessorStep``.
    """
    model = _make_model()
    stub = _StubTokenizer()
    model.tokenizer = stub

    state = torch.linspace(-1.0, 1.0, steps=4)  # 4 dims for the test
    result = model.process_prompt(
        prompt="pick up the\nblock",
        input_modalities=["image", "text"],
        output_modalities=["action"],
        robot_state=state,
    )

    # Result is a single text_inputs entry — no separate state_inputs.
    assert set(result.keys()) == {"text_inputs"}
    assert stub.last_prompt is not None
    # Underscores in the task aren't expected here, but newlines are stripped.
    assert "pick up the block" in stub.last_prompt
    # State bins are integers in [0, 255]; check the format prefix/suffix.
    assert stub.last_prompt.startswith("Task: pick up the block, State: ")
    assert stub.last_prompt.endswith(";\nAction: ")
    # 4 state values -> 4 bin numbers, the first should be 0 and last 255.
    assert " 0 " in stub.last_prompt or stub.last_prompt.split("State: ", 1)[1].startswith("0 ")
    assert "255" in stub.last_prompt


def test_pi05_process_prompt_without_state_uses_plain_text():
    model = _make_model()
    stub = _StubTokenizer()
    model.tokenizer = stub
    result = model.process_prompt(
        prompt="hello world",
        input_modalities=["text"],
        output_modalities=["action"],
    )
    assert "text_inputs" in result
    assert stub.last_prompt == "hello world"


# ----------------------------------------------------------------------
# Lerobot -> mminf weight remap (pure-function unit tests)
# ----------------------------------------------------------------------


def test_remap_lerobot_state_dict_buckets_keys_correctly():
    """Verify that ``remap_lerobot_state_dict`` routes each lerobot key to
    the right mminf submodule with the right inner key. Uses tiny tensors
    so we can run on CPU without weights."""
    from mminf.model.pi05.weight_loader import remap_lerobot_state_dict

    def t(*shape):
        return torch.zeros(*shape)

    pali = "paligemma_with_expert.paligemma.model"
    ge = "paligemma_with_expert.gemma_expert.model"
    sd = {
        # Top-level
        "action_in_proj.weight": t(1024, 32),
        "action_in_proj.bias": t(1024),
        "action_out_proj.weight": t(32, 1024),
        "action_out_proj.bias": t(32),
        "time_mlp_in.weight": t(1024, 1024),
        "time_mlp_in.bias": t(1024),
        "time_mlp_out.weight": t(1024, 1024),
        "time_mlp_out.bias": t(1024),
        # PaliGemma side
        "paligemma_with_expert.paligemma.lm_head.weight": t(257152, 2048),
        f"{pali}.language_model.norm.weight": t(2048),
        f"{pali}.language_model.layers.0.input_layernorm.weight": t(2048),
        f"{pali}.language_model.layers.0.self_attn.q_proj.weight": t(2048, 2048),
        f"{pali}.language_model.layers.0.mlp.gate_proj.weight": t(16384, 2048),
        # Vision tower
        f"{pali}.vision_tower.vision_model.embeddings.patch_embedding.weight": t(1152, 3, 14, 14),
        f"{pali}.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": t(1152),
        # Multi-modal projector -> connector
        f"{pali}.multi_modal_projector.linear.weight": t(2048, 1152),
        f"{pali}.multi_modal_projector.linear.bias": t(2048),
        # Action expert side
        f"{ge}.layers.0.self_attn.q_proj.weight": t(2048, 1024),
        f"{ge}.layers.0.input_layernorm.dense.weight": t(3072, 1024),
        f"{ge}.layers.0.input_layernorm.dense.bias": t(3072),
        f"{ge}.norm.dense.weight": t(3072, 1024),
        f"{ge}.norm.dense.bias": t(3072),
        "paligemma_with_expert.gemma_expert.lm_head.weight": t(257152, 1024),  # dropped
    }
    buckets = remap_lerobot_state_dict(sd)

    # action_in_proj
    assert set(buckets["action_in_proj"].keys()) == {"weight", "bias"}
    assert buckets["action_in_proj"]["weight"].shape == (1024, 32)

    # action_out_proj
    assert set(buckets["action_out_proj"].keys()) == {"weight", "bias"}

    # time_mlp -> linear_in / linear_out
    assert set(buckets["time_mlp"].keys()) == {
        "linear_in.weight", "linear_in.bias",
        "linear_out.weight", "linear_out.bias",
    }

    # embed_tokens — pulled from PaliGemma's lm_head
    assert "weight" in buckets["embed_tokens"]
    assert buckets["embed_tokens"]["weight"].shape == (257152, 2048)

    # paligemma transformer — language_model.* prefix stripped
    pali = buckets["paligemma"]
    assert "norm.weight" in pali
    assert "layers.0.input_layernorm.weight" in pali
    assert "layers.0.self_attn.q_proj.weight" in pali
    assert "layers.0.mlp.gate_proj.weight" in pali

    # siglip — vision_tower replaced with vision_model (Pi05SiglipEncoder owns
    # self.vision_model = SiglipVisionModel which itself has an inner
    # .vision_model attribute, so the resulting key has the double prefix).
    # multi_modal_projector.linear.* -> connector.*
    siglip = buckets["siglip"]
    assert "vision_model.vision_model.embeddings.patch_embedding.weight" in siglip
    assert "vision_model.vision_model.encoder.layers.0.layer_norm1.weight" in siglip
    assert "connector.weight" in siglip
    assert "connector.bias" in siglip
    assert not any("multi_modal_projector" in k for k in siglip.keys())

    # action_expert — gemma_expert.model.* prefix stripped, dense.* preserved
    ae = buckets["action_expert"]
    assert "layers.0.self_attn.q_proj.weight" in ae
    assert "layers.0.input_layernorm.dense.weight" in ae
    assert "layers.0.input_layernorm.dense.bias" in ae
    assert "norm.dense.weight" in ae
    assert "norm.dense.bias" in ae

    # gemma_expert.lm_head is intentionally dropped
    for bucket in buckets.values():
        assert not any("lm_head" in k for k in bucket.keys())


def test_remap_lerobot_state_dict_returns_known_top_level_buckets():
    """Sanity check on the bucket schema."""
    from mminf.model.pi05.weight_loader import remap_lerobot_state_dict

    buckets = remap_lerobot_state_dict({})
    assert set(buckets.keys()) == {
        "siglip",
        "embed_tokens",
        "paligemma",
        "action_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp",
    }
    for v in buckets.values():
        assert v == {}
