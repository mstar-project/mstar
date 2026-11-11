"""Qwen3 drafting through the state contract builds what it built by hand: the specs, the verify node, and the prompt
and verify steps the engine runs, field for field.

The hand-written specs, verify node and steps below are the reference, copied from the model and the scaffold's
resource path as they stood before the port, so the adapter can't pass by agreeing with itself. A step is compared
as the engine runs it: the node's declaration expanded, then the drafter's share added (`drafter_step`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml

pytest.importorskip("triton")

from qwen3_cpu import bind_share, build_submodule, draft_block, make_model, write_checkpoints  # noqa: E402

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep  # noqa: E402
from mstar.engine.resources.draft.drafter import ModelGeometry, VerifyNode, drafter_resources  # noqa: E402
from mstar.engine.resources.draft.node import drafter_step  # noqa: E402
from mstar.engine.resources.kv.config import KVSpec, KVStep, LayerSetSlot, PagedKVConfig  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerStep  # noqa: E402
from mstar.engine.resources.step import Segment, SubmoduleStep  # noqa: E402
from mstar.engine.state import Draft  # noqa: E402
from mstar.engine.state.adapters import StateSetup, draft_config, node_handles, node_resources  # noqa: E402
from mstar.engine.state.analysis import declaration_digests  # noqa: E402
from mstar.model.base import enabled_resources, serving_resources  # noqa: E402
from mstar.model.hf_config import hf_weight_dtype  # noqa: E402
from mstar.model.keys import KV_CACHE, PREFILL_TEXT, ROPE, SAMPLER  # noqa: E402
from mstar.model.multimodal import TEXT  # noqa: E402
from mstar.model.qwen3.config import ATTN, LLM_NODE  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, engine_step  # noqa: E402

K = 3
DSPARK_CONFIG = Path(__file__).parents[2] / "configs" / "qwen3_8b_dspark.yaml"


@pytest.fixture
def model(tmp_path):
    target, dspark = write_checkpoints(tmp_path)
    return make_model(target, draft_block(dspark, K))


def _reference_specs(model) -> list:
    """What the hand-written ``get_node_resources`` built, the drafter joined by `drafter_resources`."""
    cfg, nodes = model.config, {LLM_NODE}
    specs = [
        KVSpec(
            resource_key=KV_CACHE, nodes=nodes,
            config=PagedKVConfig(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads, num_qo_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim, max_seq_len=cfg.max_position_embeddings,
            ),
        ),
        AttentionSpec(
            resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE, splits=model.mixed_splits()),
        ),
        PositionSpec(
            resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE, rope_theta=cfg.rope_theta),
        ),
        model._sampler_spec(cfg.vocab_size),
    ]
    return drafter_resources(specs, model.drafter_decl, nodes, model.draft)


def test_specs_equal_the_hand_written_ones(model):
    specs = enabled_resources(model, {}, StateSetup())
    reference = _reference_specs(model)
    assert [spec.resource_key for spec in specs] == [spec.resource_key for spec in reference]
    for spec, expected in zip(specs, reference, strict=True):
        assert spec == expected, f"{expected.resource_key} differs from the hand-written spec"
        # a pool's joined layer sets sit out of its equality
        for field in ("layer_sets", "joined", "extra_storages"):
            got = getattr(getattr(spec, "config", None), field, None)
            want = getattr(getattr(expected, "config", None), field, None)
            assert got == want, f"{expected.resource_key}.{field} differs from the hand-written spec"


def test_the_verify_node_equals_the_hand_written_one(model):
    node = model.llm_node_class()
    cfg, decl = model.config, model.drafter_decl
    reference = VerifyNode(
        LLM_NODE, node.DECODE_WALK, model.draft, decl.needs,
        token_stop=node.node_token_stop(cfg.stop_token_ids, model.encoders, model.TEXT_PROMPTS),
        device_loopback=node.node_device_loopback(),
        input_kinds=frozenset({TEXT}),
        target=ModelGeometry(
            num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size, dtype=hf_weight_dtype(model.hf_config),
        ),
        checkpoint=str(model._drafter_dir),
        prefill_steps=decl.prefill_steps(), verify_steps=decl.verify_steps(),
    )
    assert model.verify_nodes() == [reference]


class _Stub:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def run(self, q, k=None, v=None, label=None):
        return q


def _node(model):
    sub = build_submodule(model)
    resources = {spec.resource_key: _Stub() for spec in node_resources(model)}
    sub.bind_node_state(node_handles(model, LLM_NODE, resources))
    share = bind_share(sub, K).drafter_share
    return sub, share._replace(
        prompt_steps=model.drafter_decl.prefill_steps(), verify_steps=model.drafter_decl.verify_steps(),
    )


def _assert_steps_equal(step: SubmoduleStep, reference: SubmoduleStep) -> None:
    assert step.state, "a ported walk declares its state"
    # the handles the step opens, checks and commits: the KV array's and the sampler's
    assert {handle.name for handle in step.state_handles} >= {KV_CACHE, SAMPLER}
    assert step.steps.keys() == reference.steps.keys()
    assert step.segments == reference.segments
    for key, expected in reference.steps.items():
        assert step.steps[key] == expected, f"{key} differs from the hand-written step"


def test_a_prompt_step_equals_the_hand_written_one(model):
    sub, share = _node(model)
    rids, lengths = [1, 2], [5, 3]
    inputs = [ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n) for n in lengths]
    step = drafter_step(engine_step(sub, graph_walk=PREFILL_TEXT, request_ids=rids, inputs=inputs), PREFILL_TEXT, share)
    segments = [Segment(rid, "main", n) for rid, n in zip(rids, lengths, strict=True)]
    # a verifying node applies no repetition penalty, so tracks no prompt
    reference = SubmoduleStep(segments=segments, steps={
        KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True), SAMPLER: SamplerStep(apply_penalty=False),
        ROPE: PositionStep(), **share.prompt_steps,
    })
    _assert_steps_equal(step, reference)


def test_a_verify_step_equals_the_hand_written_one(model):
    sub, share = _node(model)
    rids = [1, 2]
    inputs = [ARNodeInputs(input_seq_len=K + 1), ARNodeInputs(input_seq_len=K + 1)]
    step = drafter_step(engine_step(sub, graph_walk="decode", request_ids=rids, inputs=inputs), "decode", share)
    reference = SubmoduleStep(segments=[Segment(rid, "main", K + 1) for rid in rids], steps={
        KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True, verify=True),
        SAMPLER: SamplerStep(segments=[Segment(rid, "main", 1) for rid in rids], apply_penalty=False),
        ROPE: PositionStep(), **share.verify_steps,
    })
    _assert_steps_equal(step, reference)


def test_the_declared_draft_names_the_models_drafter_and_block(model):
    """The adapter checks the Draft's settings as the draft block they came from and builds what the model built."""
    (draft,) = [declared for declared in model.declare_state() if isinstance(declared, Draft)]
    assert draft_config(draft) == model.draft and model.drafter(draft) is model.drafter_decl
    assert draft.checkpoint == str(model._drafter_dir) and draft.num_layers == model.config.num_hidden_layers


def test_a_drafting_deployment_builds_its_specs_handles_and_digests(model):
    """What every process runs at startup on the declarations: the serving specs, a node's handles, and the digests
    the conductor, the API server and the workers compare."""
    deployment = yaml.safe_load(DSPARK_CONFIG.read_text())
    specs = serving_resources(model, deployment)
    kv = next(spec for spec in specs if spec.resource_key == KV_CACHE)
    assert kv.config.max_num_pages == deployment["resources"][KV_CACHE]["max_num_pages"]
    joined = model.drafter_decl.needs.kv
    assert kv.config.layer_sets[joined.name] == LayerSetSlot(model.config.num_hidden_layers, joined.num_layers)
    handles = node_handles(model, LLM_NODE, {spec.resource_key: _Stub() for spec in specs})
    assert set(handles) == {KV_CACHE, SAMPLER}, "a Draft has no handle"
    digests = declaration_digests(model.declare_state(), model.kernel_settings())
    assert digests.keys() == {KV_CACHE, SAMPLER, "draft", "kernel_settings()"}
    assert digests == declaration_digests(model.declare_state(), model.kernel_settings())
