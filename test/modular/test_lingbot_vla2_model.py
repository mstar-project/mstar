"""Deterministic CPU tests for LingBotVLA2Model, submodules, normalization, contracts, and E2E StepRunner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from mstar.api_server.openpi import OpenpiPolicy
from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.engine.state import Layout, Span
from mstar.engine.state.adapters import StateSetup, node_handles, node_resources, state_steps
from mstar.integrations.dynamo.bridges import get_tensor_spec
from mstar.model.components.lingbot_vla2_action_expert import LingBotVLAActionExpert
from mstar.model.components.lingbot_vla2_prefix import LingBotVLA2PrefixQueries
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.qwen_vision import QwenVisionModel
from mstar.model.lingbot_vla2 import (
    ACTION_WALK,
    LLM_ATTN,
    LLM_KV,
    LLM_NODES,
    PREFILL_WALK,
    SUFFIX_ATTN,
    VISION_ATTN,
    VISION_TOKENS,
    LingBotVLA2Config,
    LingBotVLA2Model,
)
from mstar.model.lingbot_vla2.normalization import (
    LingBotVLA2NormStats,
    QuantileStats,
    load_norm_stats,
    pad_vector,
    slice_vector,
)
from mstar.model.lingbot_vla2.submodules import patchify_lingbot_images
from mstar.model.multimodal import TEXT, PromptPart
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY

_VISION_PREFIX = "model.qwenvl_with_expert.qwenvl.model.visual."
_TEXT_PREFIX = "model.qwenvl_with_expert.qwenvl.model.language_model."
_EXPERT_PREFIX = "model.qwenvl_with_expert.qwen_expert.model."


def _fill_seeded(module: torch.nn.Module, gen: torch.Generator, dtype: torch.dtype = torch.float32) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim >= 2:
                val = torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.08
            elif "norm" in name:
                val = 1.0 + torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.03
            else:
                val = torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.02
            param.copy_(val.to(dtype))


def _write_offline_tokenizer(directory: Path, special_tokens: dict[str, int]) -> None:
    vocab = {
        "<unk>": 0,
        "pick": 1,
        "up": 2,
        "the": 3,
        "red": 4,
        "block": 5,
        "cube": 6,
        "left": 7,
        "right": 8,
        **special_tokens,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"})
    )


def _create_synthetic_sharded_checkpoint(
    root: Path,
    *,
    seed: int = 123,
    use_current_patch_loss: bool = True,
    nested_hf_ckpt: bool = False,
) -> tuple[Path, LingBotVLA2Config, dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    cfg = LingBotVLA2Config.for_dummy(
        num_layers=2,
        text_hidden=24,
        expert_hidden=24,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        mrope_section=(2, 3, 3),
        vocab_size=64,
        max_state_dim=8,
        max_action_dim=8,
        state_dim=6,
        action_dim=6,
        chunk_size=4,
        num_flow_steps=2,
        num_cameras=3,
        image_size=16,
        fp32_attention=True,
        with_prefix_queries=use_current_patch_loss,
    )
    special = {
        "<|pad|>": cfg.pad_token_id,
        "<|vision_start|>": cfg.vision_start_token_id,
        "<|vision_end|>": cfg.vision_end_token_id,
        "<|image_pad|>": cfg.image_token_id,
    }
    ckpt_dir = (root / "checkpoints" / "global_step_50000" / "hf_ckpt") if nested_hf_ckpt else root
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_offline_tokenizer(ckpt_dir, special)

    # Write norm_stats.json (RoboTwin bounds_99_woclip schema).
    norm_payload = {
        "norm_stats": {
            "observation.state": {
                "q01": [-1.0] * cfg.state_dim,
                "q99": [1.0] * cfg.state_dim,
            },
            "action": {
                "q01": [-2.0] * cfg.action_dim,
                "q99": [2.0] * cfg.action_dim,
            },
        }
    }
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(norm_payload))

    # Build reference modules and serialize weights into two shards using real HF keys.
    vision = QwenVisionModel(cfg.vision)
    text = Qwen3DenseLM(cfg.text, mrope_section=list(cfg.mrope_section))
    queries = LingBotVLA2PrefixQueries(cfg.prefix_queries)
    expert = LingBotVLAActionExpert(cfg.action_expert)
    for mod in (vision, text, queries, expert):
        _fill_seeded(mod, gen, torch.float32)

    shard1_tensors: dict[str, torch.Tensor] = {}
    for name, param in vision.named_parameters():
        shard1_tensors[_VISION_PREFIX + name] = param.detach().clone().contiguous()

    for name, param in text.named_parameters():
        if name == "lm_head.weight" and cfg.text.tie_word_embeddings:
            continue
        val = param.detach().clone()
        if ".qkv_proj." in name:
            q_dim = cfg.text.num_attention_heads * cfg.text.head_dim
            kv_dim = cfg.text.num_key_value_heads * cfg.text.head_dim
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            for part, chunk in (("q", q), ("k", k), ("v", v)):
                shard1_tensors[_TEXT_PREFIX + name.replace(".qkv_proj.", f".{part}_proj.")] = chunk.contiguous()
        elif ".gate_up_proj." in name:
            inter = cfg.text.intermediate_size
            gate, up = val.split([inter, inter], dim=0)
            for part, chunk in (("gate", gate), ("up", up)):
                shard1_tensors[_TEXT_PREFIX + name.replace(".gate_up_proj.", f".{part}_proj.")] = chunk.contiguous()
        else:
            shard1_tensors[_TEXT_PREFIX + name] = val.contiguous()

    for name, param in queries.named_parameters():
        shard1_tensors["model." + name] = param.detach().clone().contiguous()

    shard2_tensors: dict[str, torch.Tensor] = {}
    for name, param in expert.named_parameters():
        val = param.detach().clone()
        if name.startswith("suffix_inputs."):
            shard2_tensors["model." + name.removeprefix("suffix_inputs.")] = val.contiguous()
        elif ".self_attn.self_attn.qkv_proj." in name or ".self_attn.qkv_proj." in name:
            clean = name.replace(".self_attn.self_attn.qkv_proj.", ".self_attn.qkv_proj.")
            q_dim = cfg.action_expert.num_attention_heads * cfg.action_expert.head_dim
            kv_dim = cfg.action_expert.num_key_value_heads * cfg.action_expert.head_dim
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            for part, chunk in (("q", q), ("k", k), ("v", v)):
                shard2_tensors[_EXPERT_PREFIX + clean.replace(".qkv_proj.", f".{part}_proj.")] = chunk.contiguous()
        elif ".self_attn.self_attn." in name:
            shard2_tensors[_EXPERT_PREFIX + name.replace(".self_attn.self_attn.", ".self_attn.")] = val.contiguous()
        elif name.endswith(".mlp.gate.e_score_correction_bias"):
            shard2_tensors[
                _EXPERT_PREFIX + name.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
            ] = val.contiguous()
        elif name.endswith(".mlp.experts.gate_up_proj"):
            moe_inter = cfg.action_expert.routed_intermediate_size
            gate, up = val.split([moe_inter, moe_inter], dim=1)
            for part, chunk in (("gate_proj", gate), ("up_proj", up)):
                shard2_tensors[
                    _EXPERT_PREFIX + name.replace(".experts.gate_up_proj", f".experts.{part}")
                ] = chunk.contiguous()
        elif name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
            shared_inter = cfg.action_expert.shared_intermediate_size
            gate, up = val.split([shared_inter, shared_inter], dim=0)
            for part, chunk in (("gate_proj.weight", gate), ("up_proj.weight", up)):
                shard2_tensors[
                    _EXPERT_PREFIX + name.replace(".shared_expert.gate_up_proj.weight", f".shared_expert.{part}")
                ] = chunk.contiguous()
        else:
            shard2_tensors[_EXPERT_PREFIX + name] = val.contiguous()

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    save_file(shard1_tensors, str(ckpt_dir / shard1_name))
    save_file(shard2_tensors, str(ckpt_dir / shard2_name))

    weight_map = {k: shard1_name for k in shard1_tensors} | {k: shard2_name for k in shard2_tensors}
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    vla_cfg_json = {
        "vlm_family": "qwen3_vl",
        "expert_hidden_size": cfg.action_expert.hidden_size,
        "action_num_attention_heads": cfg.action_expert.num_attention_heads,
        "action_num_key_value_heads": cfg.action_expert.num_key_value_heads,
        "action_head_dim": cfg.action_expert.head_dim,
        "token_num_experts": cfg.action_expert.num_experts,
        "token_top_k": cfg.action_expert.top_k,
        "token_moe_intermediate_size": cfg.action_expert.routed_intermediate_size,
        "token_shared_intermediate_size": cfg.action_expert.shared_intermediate_size,
        "routed_scaling_factor": cfg.action_expert.routed_scaling_factor,
        "max_state_dim": cfg.max_state_dim,
        "max_action_dim": cfg.max_action_dim,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "num_cameras": cfg.num_cameras,
        "img_size": cfg.image_size,
        "chunk_size": cfg.chunk_size,
        "num_steps": cfg.num_flow_steps,
        "pad_token_id": cfg.pad_token_id,
        "num_task_tokens": cfg.prefix_queries.num_task_tokens,
        "use_current_patch_loss": cfg.prefix_queries.use_current_patch_loss,
        "use_current_shared_task_proj": cfg.prefix_queries.use_current_shared_task_proj,
        "use_future_depth": cfg.prefix_queries.use_future_depth,
        "use_future_video": cfg.prefix_queries.use_future_video,
        "share_future_depth_query": cfg.prefix_queries.share_future_depth_query,
        "use_shared_future_task_proj": cfg.prefix_queries.use_shared_future_task_proj,
        "qwen3_vl_config": {
            "model_type": "qwen3_vl",
            "tie_word_embeddings": cfg.text.tie_word_embeddings,
            "image_token_id": cfg.image_token_id,
            "vision_start_token_id": cfg.vision_start_token_id,
            "vision_end_token_id": cfg.vision_end_token_id,
            "vision_config": asdict(cfg.vision),
            "text_config": {
                **asdict(cfg.text),
                "rope_parameters": {
                    "rope_theta": cfg.text.rope_theta,
                    "mrope_interleaved": True,
                    "mrope_section": list(cfg.mrope_section),
                },
            },
        },
    }
    (ckpt_dir / "config.json").write_text(json.dumps(vla_cfg_json))
    return ckpt_dir, cfg, shard1_tensors | shard2_tensors


def _build_full_engine_runner(model: LingBotVLA2Model, *, dtype: torch.dtype = torch.float32):
    declarations = model.declare_state()
    setup = StateSetup()
    specs = {s.resource_key: s for s in node_resources(model, setup=setup)}
    specs[LLM_KV].config.max_num_pages = 32
    specs[LLM_KV].config.page_size = 16
    specs[LLM_KV].config.dtype = dtype

    transfer_info = TransferEngineInfo(
        my_entity_id="test_worker",
        transfer_engine=LocalTransferEngine("localhost"),
        my_session_id="test_session",
    )
    resources: dict = {}
    for key, spec in specs.items():
        if not spec.depends_on():
            info = EngineResourceInfo(
                device=torch.device("cpu"),
                transfer_engine_info=transfer_info,
                kv_dtype=dtype,
                compute_dtype=dtype,
                nodes=spec.nodes,
                needs_remote_transfer=False,
            )
            resources[key] = build_resource(spec, info)
    for key, spec in specs.items():
        if spec.depends_on():
            deps = {dep: specs[dep] for dep in spec.depends_on()}
            info = EngineResourceInfo(
                device=torch.device("cpu"),
                transfer_engine_info=transfer_info,
                kv_dtype=dtype,
                compute_dtype=dtype,
                nodes=spec.nodes,
                dependencies=deps,
                needs_remote_transfer=False,
            )
            res = build_resource(spec, info)
            res.bind_dependencies(resources)
            resources[key] = res

    built_handles: dict = {}
    submodules = {}
    for node in ("vit_encoder", "LLM", "action_expert"):
        sub = model.get_submodule(node, device="cpu")
        assert sub is not None
        sub.bind_node_resources(resources)
        handles = node_handles(model, node, resources, setup=setup, built=built_handles)
        sub.bind_node_state(handles)
        submodules[node] = sub

    runner = StepRunner(
        resources,
        node_resources={
            node: [k for k, s in specs.items() if node in s.nodes]
            for node in ("vit_encoder", "LLM", "action_expert")
        },
    )
    return runner, built_handles, submodules, declarations


def _execute_node_step(
    runner: StepRunner,
    handles: dict,
    submodule,
    walk: str,
    request_ids: tuple[str, ...],
    per_req_inputs: list[dict],
    seed: int,
):
    node_inputs = []
    for req_idx, (rid, req_in) in enumerate(zip(request_ids, per_req_inputs, strict=True)):
        fwd_info = CurrentForwardPassInfo(
            request_id=rid,
            graph_walk=walk,
            fwd_index=0,
            random_seed=seed + req_idx,
            max_tokens=1024,
        )
        node_inputs.append(submodule.prepare_inputs(walk, fwd_info, req_in))

    declared = submodule.declare_step(walk, list(request_ids), node_inputs)
    ctx = StepContext(request_ids=request_ids, graph_walk=walk, slot=0, capture=False)
    declared.set_ctx(ctx)
    expanded = state_steps(declared, handles, graph_walk=walk)
    expanded.set_ctx(ctx)
    outcome = runner.admit(expanded)
    assert outcome.ok, outcome
    plan = runner.plan(expanded)
    planned_ctx = StepContext(
        request_ids=request_ids,
        graph_walk=walk,
        slot=0,
        capture=False,
        plan_results=plan,
    )
    expanded.set_ctx(planned_ctx)
    preprocessed = submodule.preprocess(walk, None, node_inputs)
    expanded.begin_calls()
    try:
        with torch.no_grad():
            if len(request_ids) == 1:
                result = [submodule.forward(walk, None, **preprocessed)]
            else:
                result = submodule.forward_batched(walk, None, **preprocessed)
        expanded.check_calls()
        runner.commit(expanded)
        expanded.commit_calls()
        return result
    finally:
        expanded.end_calls()


def test_defaults_config_and_checkpoint_subdir_resolution(tmp_path):
    cfg = LingBotVLA2Config.for_defaults()
    assert cfg.text.num_hidden_layers == 36
    assert cfg.action_expert.num_hidden_layers == 36
    assert cfg.mrope_section == (24, 20, 20)
    assert cfg.max_state_dim == 55
    assert cfg.max_action_dim == 55
    assert cfg.state_dim == 14
    assert cfg.action_dim == 14
    assert cfg.chunk_size == 50
    assert cfg.num_flow_steps == 10
    assert cfg.tokens_per_image == (256 // 16 // 2) ** 2

    nested_dir, _, _ = _create_synthetic_sharded_checkpoint(tmp_path, nested_hf_ckpt=True)
    # Also test rope_scaling fallback when rope_parameters is absent.
    raw = json.loads((nested_dir / "config.json").read_text())
    txt = raw["qwen3_vl_config"]["text_config"]
    txt["rope_scaling"] = txt.pop("rope_parameters")
    (nested_dir / "config.json").write_text(json.dumps(raw))

    model = LingBotVLA2Model(str(tmp_path))
    assert model._ensure_checkpoint_dir() == nested_dir
    assert model.config.mrope_section == (2, 3, 3)


def test_normalization_quantile_roundtrip_and_padding(tmp_path):
    q = QuantileStats(q01=(0.0, -4.0, 2.0), q99=(10.0, 4.0, 2.0))
    stats = LingBotVLA2NormStats(state=q, actions=q)

    raw = torch.tensor([5.0, -2.0, 7.0], dtype=torch.float32)
    normed = stats.normalize_state(raw, max_state_dim=6)
    assert normed.shape == (6,)
    # 5.0 in [0, 10] -> 0.0; -2.0 in [-4, 4] -> -0.5; constant dim stays centered safely without NaN.
    torch.testing.assert_close(normed[:3], torch.tensor([0.0, -0.5, 0.0]))
    torch.testing.assert_close(normed[3:], torch.zeros(3))

    unnormed = stats.unnormalize_actions(normed.unsqueeze(0), active_dim=3)
    assert unnormed.shape == (1, 3)
    torch.testing.assert_close(unnormed[0, :2], raw[:2], rtol=1e-5, atol=1e-5)

    assert slice_vector(pad_vector(raw, 8), 3).shape == (3,)

    ds_file = tmp_path / "dataset_statistics.json"
    ds_file.write_text(
        json.dumps(
            {
                "observation.state": {"q01": [0.0, 0.0], "q99": [2.0, 4.0]},
                "action": {"q01": [-1.0, -2.0], "q99": [1.0, 2.0]},
            }
        )
    )
    loaded = load_norm_stats(str(ds_file))
    assert loaded is not None and loaded.state is not None and loaded.actions is not None
    assert loaded.state.q99 == (2.0, 4.0)


def test_registry_config_yaml_and_validation():
    assert MODEL_REGISTRY["lingbot_vla2"] == ("mstar.model.lingbot_vla2.lingbot_vla2_model", "LingBotVLA2Model")
    assert HF_MODELS["lingbot_vla2"] == {"model_path_hf": "robbyant/lingbot-vla-v2-6b-robotwin"}

    yaml_path = Path("configs/lingbot_vla2.yaml")
    assert yaml_path.exists()
    parsed = yaml.safe_load(yaml_path.read_text())
    assert parsed["model"] == "lingbot_vla2"

    model = LingBotVLA2Model("unused", skip_weight_loading=True)
    model.validate_config_yaml(parsed, str(yaml_path))

    bad_cfg = {
        "node_groups": [
            {"node_names": ["vit_encoder"], "ranks": [0]},
            {"node_names": ["LLM"], "ranks": [0]},
            {"node_names": ["action_expert"], "ranks": [1]},
        ]
    }
    with pytest.raises(ValueError, match="node_groups"):
        model.validate_config_yaml(bad_cfg, "bad.yaml")

    assert model.get_submodule("vit_encoder", device="cpu") is not None
    assert model.get_submodule("LLM", device="cpu") is not None
    assert model.get_submodule("action_expert", device="cpu") is not None
    assert model.get_submodule("nonexistent_node", device="cpu") is None


def test_prompt_processing_image_io_and_state_validation():
    cfg = LingBotVLA2Config.for_dummy(
        num_layers=2,
        text_hidden=24,
        expert_hidden=24,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        mrope_section=(2, 3, 3),
        vocab_size=64,
        max_state_dim=8,
        max_action_dim=8,
        state_dim=6,
        action_dim=6,
        chunk_size=4,
        num_flow_steps=2,
        num_cameras=2,
        image_size=16,
        with_prefix_queries=True,
    )
    model = LingBotVLA2Model("unused", skip_weight_loading=True, config=cfg)
    assert model.robot_io() == ("robot_state", cfg.chunk_size, cfg.action_dim)
    assert model.request_kwargs() == frozenset({"robot_state"})
    assert model.max_input_files() == {"image": 2}
    assert set(model.kv_labels()[LLM_KV]) == {"main"}

    gen = torch.Generator().manual_seed(55)
    img_u8 = torch.randint(0, 256, (3, 20, 24), dtype=torch.uint8, generator=gen)
    loaded = model.load_decoded_image(img_u8, "cpu").data
    assert loaded.dtype == torch.uint8

    patches, grid = patchify_lingbot_images(
        [loaded, loaded],
        image_size=cfg.image_size,
        patch_size=cfg.vision.patch_size,
        temporal_patch_size=cfg.vision.temporal_patch_size,
        spatial_merge_size=cfg.vision.spatial_merge_size,
    )
    assert grid == [(1, 4, 4), (1, 4, 4)]
    assert patches.shape == (32, 3 * 2 * 4 * 4)
    assert float(patches.min()) >= -1.0 - 1e-5 and float(patches.max()) <= 1.0 + 1e-5

    out = model.process_prompt(
        "pick block",
        ["image", "text"],
        ["action"],
        tensors={"image_inputs": [loaded, loaded]},
        robot_state=[0.1, -0.2, 0.3],
    )
    assert set(out.new_input_tensors) == {"text_inputs", "robot_state"}
    text_ids = out.new_input_tensors["text_inputs"][0]
    assert int((text_ids == cfg.image_token_id).sum()) == 2 * cfg.tokens_per_image
    layout: Layout = out.metadata["state_layout"][LLM_KV]["main"]
    assert layout.scratch == 1 + cfg.chunk_size
    assert layout.spans[0] == Span(
        "rows",
        PREFILL_WALK,
        int(text_ids.numel()) + cfg.prefix_queries.num_query_tokens,
        int(text_ids.numel()) - 2 * cfg.tokens_per_image + 2 * 2 + cfg.prefix_queries.num_query_tokens,
        "text_inputs",
    )

    # Interleaved prompt_parts works too.
    interleaved = model.process_prompt(
        None,
        ["image", "text"],
        ["action"],
        tensors={"image_inputs": [loaded]},
        prompt_parts=[PromptPart("image", index=0), PromptPart(TEXT, text="pick")],
        robot_state=[0.0, 0.0],
    )
    assert int((interleaved.new_input_tensors["text_inputs"][0] == cfg.image_token_id).sum()) == cfg.tokens_per_image

    # Error checks.
    with pytest.raises(ValueError, match="at least one camera image"):
        model.process_prompt("pick", ["text"], ["action"], tensors={}, robot_state=[0.0])
    with pytest.raises(ValueError, match="num_cameras"):
        model.process_prompt(
            "pick",
            ["image"],
            ["action"],
            tensors={"image_inputs": [loaded, loaded, loaded]},
            robot_state=[0.0],
        )
    with pytest.raises(ValueError, match="requires robot_state"):
        model.process_prompt("pick", ["image"], ["action"], tensors={"image_inputs": [loaded]})
    with pytest.raises(ValueError, match="NaN or inf"):
        model.process_prompt(
            "pick",
            ["image"],
            ["action"],
            tensors={"image_inputs": [loaded]},
            robot_state=[float("nan")],
        )
    with pytest.raises(ValueError, match="robot_state"):
        model.process_prompt("pick", ["image"], ["action"], tensors={"image_inputs": [loaded]}, robot_state=["bad"])
    with pytest.raises(ValueError, match="robot_state"):
        model.process_prompt(
            "pick",
            ["image"],
            ["action"],
            tensors={"image_inputs": [loaded]},
            robot_state=[True, False],
        )


@pytest.mark.parametrize("use_queries", [False, True], ids=["no_queries", "with_align_queries"])
def test_full_synthetic_sharded_checkpoint_e2e_step_runner(tmp_path, use_queries):
    ckpt_dir, cfg, _ = _create_synthetic_sharded_checkpoint(
        tmp_path, seed=777, use_current_patch_loss=use_queries
    )
    model = LingBotVLA2Model(str(ckpt_dir))
    assert model.norm_stats is not None
    assert model.tokenizer is not None

    runner, handles, submodules, _ = _build_full_engine_runner(model, dtype=torch.float32)

    gen = torch.Generator().manual_seed(9001)
    img1 = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8, generator=gen)
    img2 = torch.randint(0, 256, (3, 24, 20), dtype=torch.uint8, generator=gen)
    state_a = [0.25, -0.5, 0.1, 0.0, 0.3, -0.1]
    state_b = [-0.2, 0.4, -0.3, 0.2, -0.1, 0.5]

    req_a = model.process_prompt(
        "pick up the red block",
        ["image", "text"],
        ["action"],
        tensors={"image_inputs": [img1, img2]},
        robot_state=state_a,
    )
    req_b = model.process_prompt(
        "pick left cube",
        ["image", "text"],
        ["action"],
        tensors={"image_inputs": [img1]},
        robot_state=state_b,
    )
    steps = model.walk_plan(
        "main",
        ["image", "text"],
        ["action"],
        {"image_inputs": [object()], "robot_state": [object()]},
        {},
    )
    assert [s.walk for s in steps] == [PREFILL_WALK, ACTION_WALK]

    def _run_requests(requests: list[tuple[str, object, list[torch.Tensor]]], seed: int):
        rids = tuple(rid for rid, _, _ in requests)
        for rid in rids:
            runner.ingest_request(rid)
        try:
            vit_out = _execute_node_step(
                runner,
                handles,
                submodules["vit_encoder"],
                PREFILL_WALK,
                rids,
                [{"image_inputs": imgs} for _, _, imgs in requests],
                seed=seed,
            )
            llm_out = _execute_node_step(
                runner,
                handles,
                submodules["LLM"],
                PREFILL_WALK,
                rids,
                [
                    {
                        "text_inputs": req.new_input_tensors["text_inputs"],
                        "img_emb": [vo["img_emb"]],
                        "deepstack_features": [vo["deepstack_features"]],
                    }
                    for (_, req, _), vo in zip(requests, vit_out, strict=True)
                ],
                seed=seed,
            )
            act_out = _execute_node_step(
                runner,
                handles,
                submodules["action_expert"],
                ACTION_WALK,
                rids,
                [
                    {
                        "prefix_done": [lo["prefix_done"]],
                        "robot_state": req.new_input_tensors["robot_state"],
                    }
                    for (_, req, _), lo in zip(requests, llm_out, strict=True)
                ],
                seed=seed,
            )
            return act_out
        finally:
            for rid in rids:
                runner.remove_request(rid)

    # bs=1 determinism & valid action bytes output
    run1 = _run_requests([("r1", req_a, [img1, img2])], seed=42)
    run2 = _run_requests([("r1_again", req_a, [img1, img2])], seed=42)
    torch.testing.assert_close(run1[0]["actions"], run2[0]["actions"], rtol=0, atol=0)
    assert torch.isfinite(run1[0]["actions"]).all()

    raw_bytes = model.postprocess(run1[0]["actions"], "action")
    decoded = np.frombuffer(raw_bytes, dtype=np.float32).reshape(cfg.chunk_size, cfg.action_dim)
    assert np.isfinite(decoded).all()

    # Different noise seed changes actions, confirming genuine denoising path.
    run_diff_seed = _run_requests([("r1_seed", req_a, [img1, img2])], seed=43)
    assert not torch.allclose(run1[0]["actions"], run_diff_seed[0]["actions"])

    # Batched bs=2 execution matches single-request execution closely.
    batched = _run_requests([("b1", req_a, [img1, img2]), ("b2", req_b, [img1])], seed=42)
    assert len(batched) == 2
    torch.testing.assert_close(batched[0]["actions"], run1[0]["actions"], rtol=1e-5, atol=1e-5)

    # Different state produces different actions.
    req_a2 = model.process_prompt(
        "pick up the red block",
        ["image", "text"],
        ["action"],
        tensors={"image_inputs": [img1, img2]},
        robot_state=[0.9, 0.9, -0.8, 0.5, -0.5, 0.7],
    )
    run_diff_state = _run_requests([("r_state", req_a2, [img1, img2])], seed=42)
    assert not torch.allclose(run1[0]["actions"], run_diff_state[0]["actions"])


def test_openpi_and_dynamo_integration_specs(tmp_path):
    cfg = LingBotVLA2Config.for_defaults()
    model = LingBotVLA2Model("unused", skip_weight_loading=True, config=cfg)

    policy = OpenpiPolicy.from_config(
        {
            "images": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "state": ["observation/state"],
            "action_dims": cfg.action_dim,
        },
        model,
    )
    assert policy.state_kwarg == "robot_state"
    assert policy.action_horizon == cfg.chunk_size
    assert policy.action_dim == cfg.action_dim

    spec = get_tensor_spec("lingbot_vla2")
    assert spec is not None
    model_cfg = spec.model_config("lingbot_vla2")
    assert model_cfg["outputs"][0]["shape"] == [-1, 14]

    rng = np.random.default_rng(11)
    images = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)
    tensors = {
        "images": {
            "metadata": {"name": "images", "data_type": "Uint8", "shape": [2, 16, 16, 3]},
            "data": {"data_type": "Uint8", "values": images.tobytes()},
        },
        "state": {
            "metadata": {"name": "state", "data_type": "Float32", "shape": [14]},
            "data": {"data_type": "Float32", "values": [0.1] * 14},
        },
        "prompt": {
            "metadata": {"name": "prompt", "data_type": "Bytes", "shape": [1]},
            "data": {"data_type": "Bytes", "values": [b"pick up block"]},
        },
    }
    submit = spec.submit_args(tensors, tmp_path / "uploads")
    assert submit.input_modalities == ["image", "text"]
    assert submit.output_modalities == ["action"]
    assert len(submit.file_paths["image"]) == 2
    assert len(submit.model_kwargs["robot_state"]) == 14

    with pytest.raises(ValueError, match="state"):
        spec.submit_args({"images": tensors["images"], "prompt": tensors["prompt"]}, tmp_path / "uploads2")

    out_bytes = np.zeros((4, 14), dtype="<f4").tobytes()
    chunk = type("Chunk", (), {"modality": "action", "data": out_bytes})()
    decoded = spec.output_tensors(chunk)
    assert decoded is not None and decoded[0]["metadata"]["shape"] == [4, 14]


def test_declaration_digests_deterministic_across_hash_seeds():
    def _digest_for_seed(hash_seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        code = (
            "import json; "
            "from mstar.engine.state.analysis import declaration_digests; "
            "from mstar.model.lingbot_vla2 import LingBotVLA2Model; "
            "m = LingBotVLA2Model('unused', skip_weight_loading=True); "
            "print(json.dumps(declaration_digests(m.declare_state()), sort_keys=True))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    assert _digest_for_seed("1") == _digest_for_seed("999")
    # Also verify exact keys declared in contract directly in-process.
    model = LingBotVLA2Model("unused", skip_weight_loading=True)
    decls = {d.name: d for d in model.declare_state()}
    assert set(decls) == {LLM_KV, VISION_TOKENS}
    assert decls[LLM_KV].nodes == LLM_NODES
    assert set(decls[LLM_KV].reads) == {LLM_ATTN, SUFFIX_ATTN}
    assert set(decls[VISION_TOKENS].reads) == {VISION_ATTN}
