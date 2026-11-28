"""Qwen3 drafting through the state contract builds what it built by hand: the specs, the verify node, and the prompt
and verify steps the engine runs, field for field.

The hand-written specs, verify node and steps below are the reference, copied from the model and the scaffold's
resource path as they stood before the port, so the adapter can't pass by agreeing with itself. A step is compared
as the engine runs it (`engine_step`): the node's declaration expanded, then the drafter's share added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml

pytest.importorskip("triton")

from qwen3_cpu import build_submodule, draft_block, make_model, write_checkpoints  # noqa: E402
from test_drafter_share import DRAFT_MODEL, EAGLE3  # noqa: E402

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep  # noqa: E402
from mstar.engine.resources.draft.drafter import ModelGeometry, VerifyNode, drafter_resources  # noqa: E402
from mstar.engine.resources.draft.node import DrafterShare  # noqa: E402
from mstar.engine.resources.kv.config import KVSpec, KVStep, LayerSetSlot, PagedKVConfig  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerStep  # noqa: E402
from mstar.engine.resources.step import Segment, SubmoduleStep  # noqa: E402
from mstar.engine.state import Draft  # noqa: E402
from mstar.engine.state.adapters import StateSetup, draft_config, node_handles, node_resources  # noqa: E402
from mstar.engine.state.analysis import declaration_digests  # noqa: E402
from mstar.model.base import enabled_resources, serving_resources  # noqa: E402
from mstar.model.hf_config import hf_weight_dtype  # noqa: E402
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, ROPE, SAMPLER  # noqa: E402
from mstar.model.multimodal import TEXT  # noqa: E402
from mstar.model.qwen3.config import ATTN, LLM_NODE  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, engine_step  # noqa: E402

K = 3
DSPARK_CONFIG = Path(__file__).parents[2] / "configs" / "qwen3_8b_dspark.yaml"


@pytest.fixture(params=["dspark", "eagle3", "draft_model", "draft_model_own_kv"])
def model(tmp_path, request):
    """A drafting Qwen3 of each method, its drafter's checkpoint config-only."""
    target, dspark = write_checkpoints(tmp_path)
    if request.param == "dspark":
        return make_model(target, draft_block(dspark, K))
    method = request.param.removesuffix("_own_kv")
    config = {"draft_model": DRAFT_MODEL, "eagle3": EAGLE3}[method]
    if request.param.endswith("_own_kv"):
        # a drafter of another KV geometry than the target's, whose layers get a storage of their own
        config = {**config, "num_key_value_heads": 1, "head_dim": 8}
    draft = tmp_path / request.param
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    return make_model(target, {"method": method, "model": str(draft), "num_draft_tokens": K})


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
        mixed_walk=MIXED if model.mixes_verify else None,
    )
    assert model.verify_nodes() == [reference]


class _Stub:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def run(self, q, k=None, v=None, label=None):
        return q


def _node(model):
    """The node with its handles over stand-in resources and the share the engine binds, its drafter's steps too."""
    sub = build_submodule(model)
    resources = {spec.resource_key: _Stub() for spec in node_resources(model)}
    sub.bind_node_state(node_handles(model, LLM_NODE, resources))
    given = dict(
        verify_walk=sub.DECODE_WALK, sampler_key=SAMPLER, num_draft_tokens=K, prompt_walks=sub.token_stop.prefill_walks,
        prompt_steps=model.drafter_decl.prefill_steps(), verify_steps=model.drafter_decl.verify_steps(),
    )
    share = DrafterShare(**{
        name: given.get(name, frozenset() if name.endswith("writes") else {})
        for name in DrafterShare._fields if name not in DrafterShare._field_defaults
    })._replace(mixed_walk=MIXED if model.mixes_verify else None)
    sub.bind_draft(share)
    return sub, share


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
    step = engine_step(sub, graph_walk=PREFILL_TEXT, request_ids=rids, inputs=inputs)
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
    step = engine_step(sub, graph_walk="decode", request_ids=rids, inputs=inputs)
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


def test_a_mixed_step_equals_the_hand_written_one(model):
    """A verify row and a prompt row in one step: the target's attention on its verify kernel, the sampler as a
    prompt step's, and both of the drafter's shares."""
    if not model.mixes_verify:
        pytest.skip("the drafter runs no mixed steps")
    sub, share = _node(model)
    rids = [1, 2]
    inputs = [
        ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=K + 1, graph_walk="decode"),
        ARNodeInputs(input_ids=torch.arange(5), input_seq_len=5, graph_walk=PREFILL_TEXT),
    ]
    step = engine_step(sub, graph_walk=MIXED, request_ids=rids, inputs=inputs)
    segments = [Segment(1, "main", K + 1), Segment(2, "main", 5)]
    reference = SubmoduleStep(segments=segments, steps={
        KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True, verify=True), SAMPLER: SamplerStep(apply_penalty=False),
        ROPE: PositionStep(), **share.prompt_steps, **share.verify_steps,
    })
    _assert_steps_equal(step, reference)


def _walk_inputs(model, walk: str) -> list:
    if walk == "decode":
        return [ARNodeInputs(input_seq_len=K + 1), ARNodeInputs(input_seq_len=K + 1)]
    if walk == MIXED:
        return [ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=K + 1, graph_walk="decode"),
                ARNodeInputs(input_ids=torch.arange(5), input_seq_len=5, graph_walk=PREFILL_TEXT)]
    return [ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n) for n in (5, 3)]


@pytest.mark.parametrize("walk", [PREFILL_TEXT, "decode", MIXED])
def test_the_share_keeps_the_handles_and_labels_and_adds_only_its_own_steps(model, walk, monkeypatch):
    """The drafter's share is applied after the state is expanded: the step keeps the handles and the combined
    labels the same step has with drafting off, and gains only the drafter's own steps."""
    if walk == MIXED and not model.mixes_verify:
        pytest.skip("the drafter runs no mixed steps")
    sub, share = _node(model)
    kwargs = dict(graph_walk=walk, request_ids=[1, 2], inputs=_walk_inputs(model, walk))
    drafted = engine_step(sub, **kwargs)
    monkeypatch.setattr(sub, "drafter_share", None)
    plain = engine_step(sub, **kwargs)
    own = {PREFILL_TEXT: share.prompt_steps, "decode": share.verify_steps,
           MIXED: {**share.prompt_steps, **share.verify_steps}}[walk]
    assert drafted.state_handles and drafted.state_handles == plain.state_handles
    assert drafted.combined_labels == plain.combined_labels and drafted.state == plain.state
    assert drafted.steps.keys() == plain.steps.keys() | own.keys() and not plain.steps.keys() & own.keys()
