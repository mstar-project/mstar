"""A drafter declared through the state contract (`Draft`) builds what the draft package builds for one by hand.

A `Draft` names the node's KV array, its sampler and the walks it verifies on, the drafter by its draft-block key
and checkpoint, the draft settings and the target's geometry, all plain values. The adapter checks the settings as a
draft block (`DraftConfig`), takes the drafter's declaration from the model (`Model.drafter`), joins it to the specs
the arrays build (`drafter_resources`) and declares the verify node (`VerifyNode`); the engine marks the verify walk's
attentions and its one-row sampler (`drafter_step`), which a node on the contract can't declare.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

pytest.importorskip("triton")

from mstar.engine.resources.attn.config import AttentionStep  # noqa: E402
from mstar.engine.resources.draft.config import DraftConfig  # noqa: E402
from mstar.engine.resources.draft.drafter import (  # noqa: E402
    ChunkBoundary,
    DrafterDecl,
    DrafterNeeds,
    ModelGeometry,
    VerifyNode,
    drafter_resources,
)
from mstar.engine.resources.draft.node import DrafterShare, drafter_step  # noqa: E402
from mstar.engine.resources.kv.config import KVSpec, KVStep  # noqa: E402
from mstar.engine.resources.position.config import PositionStep  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep  # noqa: E402
from mstar.engine.resources.step import Segment, SubmoduleStep  # noqa: E402
from mstar.engine.state import Array, Attend, Causal, Draft, KVEntry, Positions, Read, Rope, Sampler  # noqa: E402
from mstar.engine.state.adapters import build_specs, draft_config, node_resources, verify_node  # noqa: E402
from mstar.engine.state.analysis import declaration_digests, state_declaration_problems  # noqa: E402
from mstar.model.base import Model  # noqa: E402

NODE, K = "LLM", 3
WALKS = ("prefill", "decode")


class _Decl(DrafterDecl):
    """A drafter with no layers in the pool and one step of its own on each walk."""

    needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def node_resources(self, nodes):
        return []

    def prefill_steps(self):
        return {"own": KVStep()}

    def verify_steps(self):
        return {"own": PositionStep()}


def _config(k: int = K) -> DraftConfig:
    return DraftConfig.parse({"method": "dspark", "model": "toy", "num_draft_tokens": k})


def _arrays() -> list:
    return [
        Array(
            "kv_cache", nodes={NODE}, layers=frozenset(range(4)), entry=KVEntry(heads=2, head_dim=16), max_len=256,
            positions=Positions(rope=Rope(theta=10_000.0)), reads={"attn": Read(mask=Causal(), op=Attend(q_heads=4))},
        ),
        Sampler("sampler", nodes={NODE}, vocab_size=97, penalty=True),
    ]


def _draft(**overrides) -> Draft:
    fields = dict(
        nodes={NODE}, array="kv_cache", sampler="sampler", walk="decode", method="dspark", model="toy",
        checkpoint="/drafter", num_draft_tokens=K, num_layers=6, hidden_size=32, dtype="bfloat16",
    )
    return Draft("draft", **{**fields, **overrides})


def _model(declarations) -> SimpleNamespace:
    return SimpleNamespace(
        declare_state=lambda: declarations, kernel_settings=lambda: {}, get_combined_graph_walks=lambda: {},
        get_node_resources=lambda: [], draft_node_fields=lambda draft: {"token_stop": "stop"},
        drafter=lambda draft: _Decl(),
        get_graph_walk_graphs=lambda: {walk: SimpleNamespace(get_nodes=lambda: [NODE]) for walk in WALKS},
        kv_labels=lambda: {}, get_chunking_policies=lambda: {}, rebuild_policies=lambda: {},
    )


def test_a_drafts_specs_are_the_draft_packages_over_the_declared_ones():
    specs = node_resources(_model([*_arrays(), _draft()]))
    expected = drafter_resources(build_specs(_arrays()), _Decl(), {NODE}, _config())
    assert specs == expected
    kv = next(spec for spec in specs if isinstance(spec, KVSpec))
    sampler = next(spec for spec in specs if isinstance(spec, SamplerSpec))
    assert kv.draft_slack == _config().kv_slack and sampler.verifies and not sampler.enable_repetition_penalty


def test_without_a_draft_the_specs_are_the_arrays_alone():
    specs = node_resources(_model(_arrays()))
    assert specs == build_specs(_arrays())
    assert not next(spec for spec in specs if isinstance(spec, SamplerSpec)).verifies


def test_the_verify_node_reads_the_targets_geometry_off_the_declarations():
    declarations = [*_arrays(), _draft(mixed_walk="mixed")]
    node = verify_node(declarations[-1], declarations, _Decl(), token_stop="stop")
    # the target's depth is declared: on a hybrid model the KV array holds only some of its layers
    assert node == VerifyNode(
        NODE, "decode", _config(), _Decl.needs, token_stop="stop",
        target=ModelGeometry(
            num_layers=6, num_kv_heads=2, head_dim=16, hidden_size=32, vocab_size=97, dtype="bfloat16",
        ),
        checkpoint="/drafter", prefill_steps={"own": KVStep()}, verify_steps={"own": PositionStep()},
        mixed_walk="mixed",
    )


def test_a_model_declares_its_verify_nodes_by_its_drafts():
    model = _model([*_arrays(), _draft()])
    (node,) = Model.declared_verify_nodes(model)
    assert node.node == NODE and node.token_stop == "stop" and node.config == _config()
    assert Model.declared_verify_nodes(_model(_arrays())) == []


def test_a_model_that_drafts_nothing_never_reads_its_declarations_for_it():
    """The default `verify_nodes` is main's: a model that doesn't draft keeps its startup path."""
    def unread():
        raise AssertionError("verify_nodes read the declarations")

    assert Model.verify_nodes(SimpleNamespace(declare_state=unread)) == []


def test_a_draft_is_a_value_the_digests_read():
    """Every field is a plain value, so two processes that build it from one block digest it alike."""
    digest = declaration_digests([_draft()])["draft"]
    assert digest == declaration_digests([_draft()])["draft"]
    assert digest != declaration_digests([_draft(num_draft_tokens=K + 1)])["draft"]
    assert hash(_draft()) == hash(_draft()) and _draft() == _draft()
    json.dumps(digest)


def test_its_settings_are_checked_as_a_draft_block():
    assert draft_config(_draft()) == _config()
    with pytest.raises(ValueError, match="method"):
        draft_config(_draft(method="medusa"))


def test_a_draft_names_what_its_node_declares():
    assert state_declaration_problems(_model([*_arrays(), _draft()])) == []
    problems = state_declaration_problems(_model([*_arrays(), _draft(array="kv", sampler="other", walk="verify")]))
    assert [p.split(":")[0] for p in problems] == ["Draft 'draft'.array", "Draft 'draft'.sampler", "Draft 'draft'.walk"]


@pytest.mark.parametrize("fields, message", [
    (dict(nodes={NODE, "other"}), "must name one node"),
    (dict(mixed_walk="decode"), "names its verify walk"),
    (dict(num_draft_tokens=0), "num_draft_tokens must be a positive int"),
    (dict(num_layers=True), "num_layers must be a positive int"),
    (dict(mode="fast"), "mode must be"),
])
def test_a_draft_refuses_a_shape_it_cant_serve(fields, message):
    with pytest.raises(ValueError, match=message):
        _draft(**fields)


def _share(**given) -> DrafterShare:
    """A share with ``given``, and every other field the engine binds empty."""
    return DrafterShare(**{
        name: given.get(name, frozenset() if name.endswith("writes") else {})
        for name in DrafterShare._fields if name not in DrafterShare._field_defaults
    })._replace(**{name: value for name, value in given.items() if name in DrafterShare._field_defaults})


SHARE = _share(
    verify_walk="decode", sampler_key="sampler", num_draft_tokens=K, prompt_walks=frozenset({"prefill"}),
    prompt_steps={}, verify_steps={"own": PositionStep()}, mixed_walk="mixed",
)


def _step(sampler=SamplerStep(apply_penalty=False), attention=AttentionStep(causal=True)) -> SubmoduleStep:
    return SubmoduleStep(
        segments=[Segment(1, "main", K + 1), Segment(2, "main", K + 1)],
        steps={"kv_cache": KVStep(), "attn": attention, "sampler": sampler, "rope": PositionStep()},
    )


ROWS = [Segment(1, "main", 1), Segment(2, "main", 1)]


def test_a_verify_step_runs_its_attention_on_the_verify_kernel_and_samples_a_row_a_request():
    declared = _step()
    step = drafter_step(declared, "decode", SHARE)
    assert step.steps["attn"] == replace(declared.steps["attn"], verify=True)
    assert step.steps["sampler"] == replace(declared.steps["sampler"], segments=ROWS)
    assert step.steps["kv_cache"] == declared.steps["kv_cache"] and step.steps["own"].segments == declared.segments


def test_a_mixed_step_verifies_its_attention_and_keeps_its_samplers_rows():
    declared = _step()
    step = drafter_step(declared, "mixed", SHARE)
    assert step.steps["attn"].verify and step.steps["sampler"] == declared.steps["sampler"]


def test_a_mixed_step_runs_its_drafters_attentions_on_the_verify_kernel_its_prompt_ones_included():
    """A drafter attention declared for prompt steps alone also runs on a mixed step, beside its verify rows:
    left off the verify kernel, it would plan their attention on the host, over their reserved lengths."""
    share = SHARE._replace(prompt_steps={"draft_attn": AttentionStep(causal=True)})
    assert drafter_step(_step(), "mixed", share).steps["draft_attn"].verify
    assert not drafter_step(_step(), "prefill", share).steps["draft_attn"].verify


def test_a_prompt_step_and_a_node_without_a_drafter_are_left_alone():
    plain = _step()
    assert drafter_step(plain, "prefill", SHARE).steps["attn"] == plain.steps["attn"]
    assert drafter_step(plain, "decode", None) is plain


def test_what_a_node_on_resources_declares_itself_stands():
    declared = _step(SamplerStep(apply_penalty=False, segments=ROWS), AttentionStep(causal=True, verify=True))
    step = drafter_step(declared, "decode", SHARE)
    assert step.steps["attn"] == declared.steps["attn"] and step.steps["sampler"] == declared.steps["sampler"]
