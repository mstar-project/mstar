"""Qwen3 on the state contract builds the specs and steps it wrote by hand, field for field.

The engine sees the same objects whether a model declares its state or writes its specs, so the port changes nothing
it runs. The hand-written specs and steps below are the reference, copied from the model as it stood before the port,
so the adapter can't pass by agreeing with itself.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

# the resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep  # noqa: E402
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import Segment, SubmoduleStep  # noqa: E402
from mstar.engine.state.adapters import StateSetup, node_handles, node_resources  # noqa: E402
from mstar.model.base import enabled_resources, serving_resources  # noqa: E402
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, ROPE, SAMPLER  # noqa: E402
from mstar.model.multimodal_decoder import MIXED_SPLIT  # noqa: E402
from mstar.model.qwen3.config import ATTN, LLM_NODE  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, engine_step  # noqa: E402

CONFIG = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=False,
    eos_token_id=96, torch_dtype="float32",
)
PUBLISHED = Path(__file__).parents[4] / "configs" / "qwen3_8b.yaml"


def _model(tmp_path, generation: dict | None = None):
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    (tmp_path / "generation_config.json").write_text(json.dumps(generation or {"do_sample": False}))
    return Qwen3Model(model_path_hf=str(tmp_path))


def _reference(model) -> list:
    """The specs Qwen3's hand-written ``get_node_resources`` returned."""
    cfg, nodes = model.config, {LLM_NODE}
    return [
        KVSpec(
            resource_key=KV_CACHE, nodes=nodes,
            config=PagedKVConfig(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads, num_qo_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                max_seq_len=cfg.max_position_embeddings,
            ),
        ),
        AttentionSpec(
            resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE, splits=(MIXED_SPLIT,)),
        ),
        PositionSpec(
            resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE, rope_theta=cfg.rope_theta),
        ),
        model._sampler_spec(cfg.vocab_size),
    ]


def _assert_equal_per_key(specs, reference) -> None:
    by_key = {spec.resource_key: spec for spec in specs}
    assert len(by_key) == len(specs), "two specs under one key"
    expected = {spec.resource_key: spec for spec in reference}
    assert by_key.keys() == expected.keys()
    for key, spec in expected.items():
        assert by_key[key] == spec, f"{key} differs from the hand-written spec"


@pytest.mark.parametrize("generation", [{"do_sample": False}, {"min_p": 0.05}], ids=["greedy", "min_p"])
def test_specs_equal_the_hand_written_ones(tmp_path, generation):
    model = _model(tmp_path, generation)
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference(model))


def test_specs_equal_the_hand_written_ones_on_the_published_config(tmp_path):
    deployment = yaml.safe_load(PUBLISHED.read_text())
    model = _model(tmp_path)
    expected = _reference(model)
    apply_yaml_overrides(expected, deployment)
    specs = serving_resources(model, deployment)
    _assert_equal_per_key(specs, expected)
    kv = next(spec for spec in specs if spec.resource_key == KV_CACHE)
    assert kv.config.max_num_pages == deployment["resources"][KV_CACHE]["max_num_pages"], "the YAML should reach it"


class _StubResource:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def run(self, q, k=None, v=None, label=None):
        return q


def _node(tmp_path):
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule

    model = _model(tmp_path)
    node = Qwen3LMSubmodule(None, model.config)
    resources = {spec.resource_key: _StubResource() for spec in node_resources(model)}
    node.bind_node_state(node_handles(model, LLM_NODE, resources))
    return node


def _reference_step(rids, inputs, prefill_tokens) -> SubmoduleStep:
    """The step the resource-path scaffold declared for Qwen3: its ``layer_steps`` and ``node_step``."""
    return SubmoduleStep(
        segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(rids, inputs, strict=True)],
        steps={
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            ROPE: PositionStep(),
        },
    )


@pytest.mark.parametrize("walk", [PREFILL_TEXT, "decode", MIXED])
def test_steps_equal_the_hand_written_ones(tmp_path, walk):
    rows = {
        PREFILL_TEXT: [(PREFILL_TEXT, 5), (PREFILL_TEXT, 3)],
        "decode": [("decode", 1), ("decode", 1)],
        MIXED: [("decode", 1), (PREFILL_TEXT, 5)],
    }[walk]
    rids = [1, 2]
    inputs = [
        ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n, graph_walk=None if walk != MIXED else own)
        for own, n in rows
    ]
    tokens = {rid: inp.input_ids for rid, inp, (own, _) in zip(rids, inputs, rows, strict=True) if own != "decode"}
    step = engine_step(_node(tmp_path), graph_walk=walk, request_ids=rids, inputs=inputs)
    reference = _reference_step(rids, inputs, tokens)
    assert step.state, "a ported walk declares its state"
    assert step.steps.keys() == reference.steps.keys()
    assert step.segments == reference.segments and step.cg_key_info == reference.cg_key_info
    for key, expected in reference.steps.items():
        actual = step.steps[key]
        if isinstance(expected, SamplerStep):
            # dataclass equality can't compare a dict of tensors
            got, want = actual.prefill_tracked_tokens, expected.prefill_tracked_tokens
            assert got.keys() == want.keys() and all(torch.equal(got[rid], want[rid]) for rid in want)
            actual, expected = replace(actual, prefill_tracked_tokens={}), replace(expected, prefill_tracked_tokens={})
        assert actual == expected, f"{key} differs from the hand-written step"


def test_the_sampler_is_the_scaffolds(tmp_path):
    """The declared sampler builds the scaffold's `_sampler_spec`, min-p on where the checkpoint defaults it."""
    model = _model(tmp_path, {"min_p": 0.05})
    (spec,) = [s for s in node_resources(model) if isinstance(s, SamplerSpec)]
    assert spec == model._sampler_spec(CONFIG["vocab_size"]) and spec.enable_min_p
