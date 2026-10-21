"""Tests for the Pi0.5 model class.

These tests focus on the structural pieces of the Pi0.5 implementation that
do not require real weights: graph walks, node-to-engine mapping, forward pass
transitions, and worker graph division. Real-weight integration is exercised
separately via end-to-end smoke tests.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.kv.config import KVSpec
from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.pi05.components.flow_matching import (
    discretize_state,
    euler_step,
    sincos_timestep_embedding,
)
from mstar.model.pi05.config import LLM_KV, Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import Pi05ViTEncoderSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

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
    model.norm_stats = None
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


@pytest.fixture
def _pi05_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("batched", "batch_size"), [(False, 1), (True, 1), (True, 2)],
)
def test_pi05_vision_keeps_fp32_under_engine_autocast(_pi05_cpu_threads, batched, batch_size):
    from mstar.engine.accelerator_graph_runner import autocast_scope
    from mstar.engine.engine import Engine

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 8)

        def forward(self, pixels):
            return self.projection(pixels.mean(dim=(-1, -2))).unsqueeze(1)

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(19)
        submodule = Pi05ViTEncoderSubmodule(
            Encoder(), Pi05Config(vit_image_size=4, num_cameras=2),
        ).to("cpu").eval()
        images = torch.linspace(-1, 1, batch_size * 96).reshape(batch_size, 2, 3, 4, 4)
        engine_inputs = ModelInputsFromEngine(
            request_ids=list(range(batch_size)), per_request_info={},
        )
        engine = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)

        with autocast_scope(None, "cpu"):
            reference = submodule.encoder(images.flatten(0, 1)).reshape(batch_size, 2, 8)
        with autocast_scope(torch.bfloat16, "cpu"):
            with autocast_scope(engine._autocast_dtype_for(submodule), "cpu"):
                if batched:
                    outputs = submodule.forward_batched("prefill", engine_inputs, images)
                    actual = torch.stack([outputs[rid]["img_emb"][0] for rid in engine_inputs.request_ids])
                else:
                    actual = submodule.forward("prefill", engine_inputs, images)["img_emb"][0].unsqueeze(0)

        assert actual.dtype == torch.float32
        assert torch.equal(actual, reference)


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
    assert set(second.input_names) == {"img_emb", "text_inputs"}


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


def test_pi05_kv_cache_config_matches_pi05_config():
    model = _make_model()
    kv_specs = [
        spec for spec in model.get_node_resources() if isinstance(spec, KVSpec)
    ]
    assert len(kv_specs) == 1
    assert kv_specs[0].nodes == {"LLM"}
    kv = kv_specs[0].config
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
    dim = 16
    # The caller owns the frequency ladder and the output buffer (both are
    # preallocated once per submodule so the action-gen step is capturable).
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    emb = sincos_timestep_embedding(
        t, dim=dim, fraction=fraction, output_buffer=torch.empty(1, dim)
    )
    assert emb.shape == (1, dim)
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
    ).new_input_tensors

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
    ).new_input_tensors
    assert "text_inputs" in result
    assert stub.last_prompt == "hello world"


def test_discretize_state_matches_openpi_digitize():
    """openpi's PaligemmaTokenizer: np.digitize(state, linspace(-1, 1, 257)[:-1]) - 1,
    so -1 below the range, 255 at or above 1, and each bin edge starts its bin."""
    import numpy as np

    edges = np.linspace(-1, 1, 257)[:-1]
    values = np.concatenate([
        np.random.default_rng(0).uniform(-1.5, 1.5, 4096), edges, edges - 1e-12,
        [-1.0, 1.0, -1.0000001, 0.99999, 2.0],
    ])
    got = discretize_state(torch.from_numpy(values), num_bins=256)
    assert got.dtype == torch.long
    np.testing.assert_array_equal(got.numpy(), np.digitize(values, bins=edges) - 1)


def test_pi05_process_prompt_tokenizes_openpi_text():
    """The task keeps its case and the space after "Action:" stays, as openpi
    tokenizes the string it builds."""
    from types import SimpleNamespace

    from mstar.model.pi05.components.tokenization import Pi05Tokenizer

    seen = []

    def hf_tokenizer(text, add_special_tokens):
        seen.append(text)
        return SimpleNamespace(input_ids=[2, 7, 9])

    model = _make_model()
    model.tokenizer = Pi05Tokenizer(hf_tokenizer, model.config)
    result = model.process_prompt(
        "Put_the Cup\naway", ["image", "text"], ["action"], robot_state=[0.5, -1.2, 1.0],
    )
    assert seen == ["Task: Put the Cup away, State: 192 -1 255;\nAction: "]
    assert result.new_input_tensors["text_inputs"][0].tolist() == [2, 7, 9]
    # the KV layout's text span counts the tokens of that text
    assert result.metadata["kv_layout"][LLM_KV]["main"].spans[1].length == 3


# ----------------------------------------------------------------------
# Deployment norm stats (openpi's quantile normalization)
# ----------------------------------------------------------------------

# A 3-dim robot in openpi's norm_stats.json layout; mean/std are there but unused.
_TINY_STATS = {
    "norm_stats": {
        "state": {"mean": [0.0, 0.0, 0.5], "std": [1.0, 1.0, 0.5],
                  "q01": [-0.8, -2.7, 0.0], "q99": [0.9, -0.45, 0.99]},
        "actions": {"mean": [0.0, 0.0], "std": [0.3, 0.3], "q01": [-0.45, 0.0], "q99": [0.44, 1.0]},
    }
}


def _openpi_state_bins(raw, q01, q99):
    """openpi: transforms.Normalize(use_quantiles=True), then PaligemmaTokenizer's np.digitize."""
    import numpy as np

    raw = np.asarray(raw, dtype=np.float64)
    q01, q99 = np.asarray(q01)[: raw.shape[-1]], np.asarray(q99)[: raw.shape[-1]]
    x = (raw - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return (np.digitize(x, bins=np.linspace(-1, 1, 257)[:-1]) - 1).tolist()


def _model_with_stats(norm_stats):
    from mstar.model.pi05.components.normalization import load_norm_stats

    model = _make_model()
    model.norm_stats = load_norm_stats(norm_stats)
    model.tokenizer = _StubTokenizer()
    return model


def test_norm_stats_file_and_inline_mapping_agree(tmp_path):
    import json

    from mstar.model.pi05.components.normalization import load_norm_stats

    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(_TINY_STATS))
    from_file = load_norm_stats(str(path))
    inline = load_norm_stats({k: {"q01": v["q01"], "q99": v["q99"]} for k, v in _TINY_STATS["norm_stats"].items()})
    for a, b in ((from_file.state, inline.state), (from_file.actions, inline.actions)):
        assert torch.equal(a.q01, b.q01) and torch.equal(a.q99, b.q99)
    assert from_file.state.q01.dtype == torch.float64
    assert load_norm_stats(None) is None
    assert load_norm_stats({"state": _TINY_STATS["norm_stats"]["state"]}).actions is None


@pytest.mark.parametrize("bad", [
    {}, {"states": {"q01": [0.0], "q99": [1.0]}}, {"state": {"mean": [0.0], "std": [1.0]}},
    {"state": {"q01": [0.0, 1.0], "q99": [1.0]}}, {"actions": {"q01": [], "q99": []}},
])
def test_norm_stats_rejects_specs_it_cannot_apply(bad):
    from mstar.model.pi05.components.normalization import load_norm_stats

    with pytest.raises(ValueError):
        load_norm_stats(bad)


def test_pi05_process_prompt_normalizes_raw_state_like_openpi():
    import numpy as np

    model = _model_with_stats(_TINY_STATS)
    stats = _TINY_STATS["norm_stats"]["state"]
    rng = np.random.default_rng(0)
    for raw in ([0.1, -1.0, 0.5], [0.95, -2.9, 0.0], *rng.uniform(-3, 3, size=(32, 3)).tolist(), [0.3, -1.2]):
        model.process_prompt("pick up the cup", ["image", "text"], ["action"], robot_state=raw)
        bins = model.tokenizer.last_prompt.split("State: ", 1)[1].split(";", 1)[0].split()
        assert [int(b) for b in bins] == _openpi_state_bins(raw, stats["q01"], stats["q99"])


def test_pi05_process_prompt_rejects_state_wider_than_norm_stats():
    model = _model_with_stats(_TINY_STATS)
    with pytest.raises(ValueError, match="robot_state has 4 values"):
        model.process_prompt("pick up the cup", ["image", "text"], ["action"], robot_state=[0.0] * 4)


def test_pi05_postprocess_unnormalizes_the_dims_the_action_stats_cover():
    import numpy as np

    model = _model_with_stats(_TINY_STATS)
    actions = torch.rand(model.config.action_horizon, model.config.action_dim) * 2 - 1
    out = np.frombuffer(model.postprocess(actions, modality="action"), dtype=np.float32).reshape(actions.shape)
    q01 = np.asarray(_TINY_STATS["norm_stats"]["actions"]["q01"])
    q99 = np.asarray(_TINY_STATS["norm_stats"]["actions"]["q99"])
    a = actions.numpy().astype(np.float64)
    np.testing.assert_allclose(out[:, :2], (a[:, :2] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(out[:, 2:], actions.numpy()[:, 2:])


def test_pi05_without_norm_stats_bins_state_and_returns_actions_as_given():
    model = _make_model()
    model.tokenizer = _StubTokenizer()
    model.process_prompt("pick up the cup", ["image", "text"], ["action"], robot_state=[0.5, -0.25])
    assert model.tokenizer.last_prompt == "Task: pick up the cup, State: 192 96;\nAction: "
    actions = torch.rand(model.config.action_horizon, model.config.action_dim)
    assert model.postprocess(actions, modality="action") == actions.numpy().tobytes()


def test_pi05_norm_stats_come_from_model_kwargs(tmp_path):
    """The serving config's model_kwargs reach Pi05Config and are parsed at startup."""
    import json

    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(_TINY_STATS))
    model = Pi05Model("unused", skip_weight_loading=True, norm_stats=str(path))
    assert model.norm_stats.state.q01.tolist() == [-0.8, -2.7, 0.0]
    with pytest.raises(FileNotFoundError):
        Pi05Model("unused", skip_weight_loading=True, norm_stats=str(tmp_path / "missing.json"))


# ----------------------------------------------------------------------
# Lerobot -> mstar weight remap (pure-function unit tests)
# ----------------------------------------------------------------------


def test_lerobot_remap_routes_keys_to_llm_wrapper_paths():
    """Verify that ``Pi05Model._lerobot_remap`` translates each LLM-side
    lerobot key to the matching parameter path on the LLM weight container.
    """
    from mstar.model.pi05.pi05_model import Pi05Model

    model = Pi05Model.__new__(Pi05Model)  # bypass __init__ — pure-method test
    remap = model._lerobot_remap

    pali = "paligemma_with_expert.paligemma.model"
    ge = "paligemma_with_expert.gemma_expert.model"

    # PaliGemma lm_head -> tied embed_tokens
    assert remap("paligemma_with_expert.paligemma.lm_head.weight") == "embed_tokens.weight"

    # PaliGemma transformer: language_model.* prefix stripped, routed to paligemma.*
    assert remap(f"{pali}.language_model.norm.weight") == "paligemma.norm.weight"
    assert remap(f"{pali}.language_model.layers.0.self_attn.q_proj.weight") == \
        "paligemma.layers.0.self_attn.q_proj.weight"
    assert remap(f"{pali}.language_model.layers.0.mlp.gate_proj.weight") == \
        "paligemma.layers.0.mlp.gate_proj.weight"

    # Vision tower / connector live on the vit_encoder submodule; the LLM
    # remap drops them so _extract_siglip_state_dict can pick them up.
    assert remap(f"{pali}.vision_tower.vision_model.embeddings.patch_embedding.weight") is None
    assert remap(f"{pali}.multi_modal_projector.linear.weight") is None

    # Gemma expert transformer routed to action_expert.*
    assert remap(f"{ge}.layers.0.self_attn.q_proj.weight") == \
        "action_expert.layers.0.self_attn.q_proj.weight"
    assert remap(f"{ge}.layers.0.input_layernorm.dense.weight") == \
        "action_expert.layers.0.input_layernorm.dense.weight"
    assert remap(f"{ge}.norm.dense.bias") == "action_expert.norm.dense.bias"

    # gemma_expert.lm_head is intentionally dropped
    assert remap("paligemma_with_expert.gemma_expert.lm_head.weight") is None

    # Top-level wrappers — names pass through unchanged
    assert remap("action_in_proj.weight") == "action_in_proj.weight"
    assert remap("action_in_proj.bias") == "action_in_proj.bias"
    assert remap("action_out_proj.weight") == "action_out_proj.weight"

    # time_mlp_in/out -> time_mlp.linear_in/out
    assert remap("time_mlp_in.weight") == "time_mlp.linear_in.weight"
    assert remap("time_mlp_in.bias") == "time_mlp.linear_in.bias"
    assert remap("time_mlp_out.weight") == "time_mlp.linear_out.weight"
    assert remap("time_mlp_out.bias") == "time_mlp.linear_out.bias"

    # Unknown keys are dropped
    assert remap("totally.unknown.key") is None


@pytest.mark.parametrize("policy_saved", [False, True])
def test_lerobot_checkpoint_keys_with_and_without_policy_prefix(tmp_path, policy_saved):
    """lerobot/pi05_droid is a saved PI05Policy: every key sits under ``model.`` and
    PaliGemma's input embedding is stored next to the lm_head it is tied to.
    It must load the same way as lerobot/pi05_base, which has neither."""
    from safetensors.torch import save_file

    lm_head = torch.randn(8, 4)
    base = {
        "paligemma_with_expert.paligemma.lm_head.weight": lm_head,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight": torch.randn(4),
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight": torch.randn(4),
        "action_in_proj.weight": torch.randn(4, 2),
        "time_mlp_in.bias": torch.randn(4),
    }
    saved = dict(base)
    if policy_saved:
        saved = {"model." + k: v for k, v in base.items()}
        saved["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = lm_head.clone()
    save_file(saved, str(tmp_path / "model.safetensors"))

    model = Pi05Model.__new__(Pi05Model)  # bypass __init__: no config or tokenizer download
    model._repo_dir = tmp_path
    model._lerobot_flat = None
    flat = model._ensure_lerobot_flat()

    assert flat.keys() == base.keys()
    assert all(torch.equal(flat[k], v) for k, v in base.items())
    assert model._lerobot_remap("paligemma_with_expert.paligemma.lm_head.weight") == "embed_tokens.weight"
    assert model._lerobot_remap("time_mlp_in.bias") == "time_mlp.linear_in.bias"
